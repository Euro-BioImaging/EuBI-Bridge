"""``metadata.metadata_reader='micro'``: metadata (and with it the pixels)
from micro-reader, no Bio-Formats -- and so no JVM -- for the files it reads
(eubi_bridge/core/micro_source.py: micro_omemeta, micro_extras).

The OME object built from micro-reader, the scene managers the writer reads
from (scales, channels, position -> translation, wavelengths -> the
``eubi_bridge`` attributes), the per-file fallback to Bio-Formats, and the
JVM staying off until something needs it.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("micro_reader")
tifffile = pytest.importorskip("tifffile")

from eubi_bridge.core import micro_source  # noqa: E402

RNG = np.random.default_rng(11)


@pytest.fixture(autouse=True)
def _fresh_files():
    yield
    micro_source.close_all()


def _ome_tiff(tmp_path, name="img.ome.tif"):
    """Two scenes with pixel sizes, channel names / colours / wavelengths and
    stage positions (the image's centre, as Bio-Formats writes them)."""
    a = RNG.integers(0, 4000, (2, 3, 20, 24)).astype("u2")          # C Z Y X
    b = RNG.integers(0, 4000, (2, 16, 18)).astype("u2")             # C Y X
    path = str(tmp_path / name)
    channel = {"Name": ["DAPI", "GFP"], "Color": [16777215 * 256 + 255 - 2**32, 65535 * 256 + 255],
               "EmissionWavelength": [461.0, 510.0], "ExcitationWavelength": [405.0, 488.0]}
    with tifffile.TiffWriter(path, ome=True) as tw:
        tw.write(a, photometric="minisblack", metadata={
            "axes": "CZYX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5, "PhysicalSizeZ": 2.0,
            "Channel": channel,
            "Plane": {"PositionX": [100.0] * 6, "PositionXUnit": ["µm"] * 6,
                      "PositionY": [50.0] * 6, "PositionYUnit": ["µm"] * 6}})
        tw.write(b, photometric="minisblack", metadata={
            "axes": "CYX", "PhysicalSizeX": 0.25, "PhysicalSizeY": 0.25})
    return path, a, b


def test_ome_object_from_micro_reader(tmp_path):
    from eubi_bridge.core.micro_source import MicroReader, micro_omemeta
    path, a, b = _ome_tiff(tmp_path)
    ome = micro_omemeta(MicroReader.open(path))
    assert len(ome.images) == 2
    p = ome.images[0].pixels
    assert (p.size_t, p.size_c, p.size_z, p.size_y, p.size_x) == (1, 2, 3, 20, 24)
    assert p.type.value == "uint16"
    assert (p.physical_size_x, p.physical_size_z) == (0.5, 2.0)
    assert p.physical_size_x_unit.name == "MICROMETER"
    assert [c.name for c in p.channels] == ["DAPI", "GFP"]
    assert p.channels[1].emission_wavelength == 510.0
    assert p.channels[1].emission_wavelength_unit.name == "NANOMETER"
    q = ome.images[1].pixels
    assert (q.size_z, q.physical_size_x, q.physical_size_z) == (1, 0.25, None)   # D2: unset


def _load(path, metadata_reader="micro"):
    import asyncio

    from eubi_bridge.core.data_manager import ArrayManager

    async def load():
        manager = ArrayManager(path, metadata_reader=metadata_reader)
        return await manager.load_scenes(scene_indices="all", mosaic_tile_index=0)
    return list(asyncio.run(load()).values())


def test_scenes_load_from_micro_reader_alone(tmp_path, monkeypatch):
    """The layer the writer reads from, with every Bio-Formats metadata path
    made to fail: metadata_reader='micro' needs none of them, and reads the
    pixels with micro-reader too."""
    from eubi_bridge.core import data_manager
    from eubi_bridge.core.micro_source import MicroSource

    def no_bio_formats(*args, **kwargs):
        raise AssertionError("Bio-Formats metadata was read")

    for name in ("read_metadata_via_bfio", "read_metadata_via_bioio_bioformats",
                 "read_metadata_via_extension"):
        monkeypatch.setattr(data_manager, name, no_bio_formats)
    path, a, b = _ome_tiff(tmp_path)
    first, second = _load(path)
    assert isinstance(first.array._source, MicroSource)
    np.testing.assert_array_equal(np.asarray(first.array), a[None])          # C Z Y X
    assert first.scaledict["x"] == 0.5 and first.scaledict["z"] == 2.0
    assert [c["label"] for c in first.channels] == ["DAPI", "GFP"]
    # D3: the first pixel's centre -- the image's centre (100, 50) minus
    # 11.5 / 9.5 pixels of 0.5 um -- becomes the translation
    assert first.origindict == pytest.approx({"x": 100 - 11.5 * 0.5, "y": 50 - 9.5 * 0.5})
    # D4: wavelengths for the eubi_bridge attributes
    waves = first.acquisition_extras["channel_wavelengths"]
    assert waves[1]["emission_wavelength"] == {"value": 510.0, "unit": "nanometer"}
    assert second.scaledict["x"] == 0.25 and second.origindict == {}


def test_a_file_micro_reader_does_not_take_falls_back_to_bio_formats(tmp_path, monkeypatch):
    """Per file: Bio-Formats metadata, and the standard pixel reader with it."""
    from ome_types import from_xml

    from eubi_bridge.core import data_manager
    from eubi_bridge.core.micro_source import MicroReader
    path, a, _ = _ome_tiff(tmp_path)
    calls = []

    async def bfio_stub(p, **kwargs):
        calls.append(p)
        return from_xml(tifffile.TiffFile(path).ome_metadata)

    monkeypatch.setattr(data_manager, "read_metadata_via_bfio", bfio_stub)
    monkeypatch.setattr(MicroReader, "open", classmethod(lambda cls, *a, **k: None))
    first, _ = _load(path)
    assert calls == [path]
    assert not hasattr(first.array, "_source") or \
        not isinstance(first.array._source, micro_source.MicroSource)
    assert first.origindict == {}                # positions only from micro-reader


def test_wavelengths_are_written_without_a_split():
    from eubi_bridge.conversion.conversion_worker import build_acquisition_metadata
    extras = {"channel_wavelengths": [{"emission_wavelength": {"value": 510.0,
                                                               "unit": "nanometer"}}]}
    manager = SimpleNamespace(acquisition_extras=extras, origindict={}, series_path="img",
                              mosaic_tile_index=None, series=0)
    job = SimpleNamespace(conversion=SimpleNamespace(export_acquisition_metadata=None),
                          input_path="img.tif")
    assert build_acquisition_metadata(manager, job) == extras
    job.conversion.export_acquisition_metadata = False
    assert build_acquisition_metadata(manager, job) is None


def _run_python(code: str, timeout=600) -> str:
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True,
                            text=True, timeout=timeout)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    return result.stdout


def test_conversion_with_micro_metadata_starts_no_jvm(tmp_path):
    """End to end in a fresh process: no JVM, and the output carries the
    scales, the translation and the wavelengths."""
    path, a, _ = _ome_tiff(tmp_path)
    out = tmp_path / "out"
    stdout = _run_python(f"""
        import jpype
        from eubi_bridge.ebridge import EuBIBridge
        EuBIBridge().to_zarr({path!r}, {str(out)!r}, metadata_reader="micro",
                             use_threading=True, verbose=False, scene_index=0)
        print("JVM", jpype.isJVMStarted())
    """)
    assert "JVM False" in stdout
    (store,) = out.glob("*.zarr")
    attrs = json.loads((store / ".zattrs").read_text()) if (store / ".zattrs").exists() \
        else json.loads((store / "zarr.json").read_text())["attributes"]
    attrs = attrs.get("ome", attrs)
    transforms = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    kinds = {t["type"]: t for t in transforms}
    assert kinds["scale"]["scale"][-1] == 0.5
    assert kinds["translation"]["translation"][-1] == pytest.approx(100 - 11.5 * 0.5)
    eubi = json.loads((store / ".zattrs").read_text()).get("eubi_bridge", {}) \
        if (store / ".zattrs").exists() else attrs.get("eubi_bridge", {})
    assert "channel_wavelengths" in json.dumps(eubi)


def test_lazy_jvm_starts_on_first_java_use():
    """Lazy mode does not start the JVM, but the first Java use (as bfio /
    bioio-bioformats make it, through scyjava) starts it with the bundled
    JARs: a file that falls back to Bio-Formats still works."""
    stdout = _run_python("""
        import scyjava
        from eubi_bridge.utils.jvm_manager import set_jvm_lazy, soft_start_jvm
        set_jvm_lazy(True)
        soft_start_jvm()
        print("before", scyjava.jvm_started())
        System = scyjava.jimport("java.lang.System")
        print("after", scyjava.jvm_started(), bool(str(System.getProperty("java.version"))))
        Reader = scyjava.jimport("loci.formats.ImageReader")    # the bundled Bio-Formats
        print("bioformats", Reader is not None)
    """)
    assert "before False" in stdout
    assert "after True True" in stdout
    assert "bioformats True" in stdout


def _jvm_after(code: str) -> str:
    """Run *code* in a fresh process; print whether it started a JVM."""
    return _run_python(textwrap.dedent(code)
                       + "\nimport jpype\nprint('JVM', jpype.isJVMStarted())\n")


def test_a_batch_from_the_gui_starts_no_jvm(tmp_path):
    """The GUI's conversion subprocess, given a batch: the readers ride in the
    table, not in its kwargs.  It used to start the JVM whenever the kwargs
    did not say metadata_reader='micro' -- for every batch."""
    path, _, _ = _ome_tiff(tmp_path)
    out = tmp_path / "out"
    stdout = _jvm_after(f"""
        if __name__ == "__main__":
            import queue, pandas as pd
            from eubi_bridge.qt_gui.workers._conv_subprocess import _conversion_subprocess
            table = pd.DataFrame({{"input_path": [{path!r}], "output_path": [{str(out)!r}]}})
            logs, result = queue.Queue(), queue.Queue()
            _conversion_subprocess({{"input_path": table, "output_path": None,
                                     "to_zarr_kwargs": {{"use_threading": True,
                                                         "scene_index": 0}}}}, logs, result)
            import sys
            sys.__stdout__.write(str(result.get()) + "\\n")
            import jpype
            sys.__stdout__.write(f"JVM {{jpype.isJVMStarted()}}\\n")
            sys.exit(0)
    """)
    assert "('ok', None)" in stdout, stdout
    assert "JVM False" in stdout


def test_table_rows_decide_whether_java_is_needed(tmp_path):
    """The config says bfio, every row of the table says micro: no JVM."""
    path, _, _ = _ome_tiff(tmp_path)
    config = tmp_path / "config"
    stdout = _jvm_after(f"""
        import pandas as pd
        from eubi_bridge.ebridge import EuBIBridge
        EuBIBridge(configpath={str(config)!r}).configure.metadata(metadata_reader="bfio")
        table = pd.DataFrame({{"input_path": [{path!r}],
                               "output_path": [{str(tmp_path / 'out')!r}],
                               "metadata_reader": ["micro"]}})
        EuBIBridge(configpath={str(config)!r}).to_zarr(table, None, use_threading=True,
                                                       verbose=False, scene_index=0)
    """)
    assert "JVM False" in stdout
