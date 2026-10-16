"""Roadmap step 2: the older readers (Bio-Formats, bioio, CZI, IMS, H5) on the new path,
through ``LegacyFile`` / ``LegacyImage`` (eubi_bridge/conversion/legacy_image.py).

Every case is converted through the real CLI and must write what eubi-bridge 0.1.x wrote
(golden files, tests/_golden.py, recorded before the old path was retired in step 6).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import requires_full
from tests import _golden
from tests import test_pyramid_path as tp
from tests.test_pyramid_path import _eubi, _ome_tiff

tifffile = pytest.importorskip("tifffile")
zarr = pytest.importorskip("zarr")
pytestmark = requires_full                   # the older readers: not in eubi-bridge-lite

RNG = np.random.default_rng(23)


@pytest.fixture(autouse=True)
def _seeded(monkeypatch):
    """The same inputs whichever tests run before (golden outputs hash pixels)."""
    global RNG
    RNG = np.random.default_rng(23)
    monkeypatch.setattr(tp, "RNG", np.random.default_rng(7))      # _ome_tiff's
STANDARD = ["--pixel_reader", "standard", "--metadata_reader", "bfio"]


def _convert(src: Path, out: Path, new: bool, args: list) -> str:
    env = dict(os.environ, EUBI_NEW_PATH="1" if new else "0", COLUMNS="250")
    result = subprocess.run([_eubi(), "to_zarr", str(src), str(out), *args],
                            capture_output=True, text=True, env=env)
    log = " ".join((result.stdout + result.stderr).split())
    assert result.returncode == 0, log[-3000:]
    return log


def _both(src: Path, tmp_path: Path, args: list, name: str) -> dict:
    """`src` converted, checked against 0.1.x's output (golden `name`)."""
    log = _convert(src, tmp_path / "new", True, args)
    assert re.search(r"new path, \d+ output\(s\), older readers", log), log[-2000:]

    def old():
        _convert(src, tmp_path / "old", False, args)
        return tmp_path / "old"
    return _golden.check(name, tmp_path / "new", tmp_path, old=old)


def _two_series_btf(path: Path) -> Path:
    """Two series that differ; .btf has no native reader, so Bio-Formats reads it."""
    with tifffile.TiffWriter(str(path), ome=True) as tw:
        tw.write(np.full((2, 32, 40), 10, "u1"), photometric="minisblack",
                 metadata={"axes": "CYX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5})
        tw.write(np.full((2, 32, 40), 200, "u1"), photometric="minisblack",
                 metadata={"axes": "CYX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5})
    return path


def test_bioformats_file_on_both_paths(tmp_path):
    out = _both(_two_series_btf(tmp_path / "two.ome.btf"), tmp_path,
                ["--scene_index", "all", *STANDARD], "legacy_image-bioformats_two_series")
    assert sorted(out) == ["two_scene0.zarr", "two_scene1.zarr"]
    values = [np.unique(np.asarray(zarr.open_group(str(tmp_path / "new" / name))["0"])).tolist()
              for name in sorted(out)]
    assert values == [[10], [200]]                          # each scene its own pixels


@pytest.mark.parametrize("args", [["--scene_index", "all"],
                                  ["--scene_index", "1", "--channel_labels", "0,A",
                                   "--dtype", "uint8", "--channel_intensity_limits", "auto"]],
                         ids=["defaults", "settings"])
def test_ome_tiff_with_the_standard_reader_on_both_paths(tmp_path, args):
    """bioio's OME-TIFF reader (dask) with bfio's metadata: names, colours, sizes."""
    _both(_ome_tiff(tmp_path / "img.ome.tif"), tmp_path, [*args, *STANDARD],
          f"legacy_image-ome_tiff-{'defaults' if len(args) == 2 else 'settings'}")


def test_imagej_tiff_with_the_standard_reader_on_both_paths(tmp_path):
    src = tmp_path / "stack.tif"
    tifffile.imwrite(str(src), RNG.integers(0, 4000, (5, 2, 30, 34)).astype("u2"),
                     imagej=True, resolution=(1 / 0.2, 1 / 0.2),
                     metadata={"axes": "ZCYX", "spacing": 1.5, "unit": "um"})
    _both(src, tmp_path, STANDARD, "legacy_image-imagej_tiff")


def test_legacy_file_reads_like_micro_reader(tmp_path):
    """The interface the new path reads: indices, image(index), the reader's own array.
    bioio's OME-TIFF readers (no Java in this process)."""
    from eubi_bridge.conversion.legacy_image import LegacyFile
    from eubi_bridge.core.config_models import ConversionJob
    src = tmp_path / "one.ome.tif"
    data = RNG.integers(0, 4000, (2, 3, 32, 40)).astype("u2")             # C Z Y X
    tifffile.imwrite(str(src), data, ome=True, photometric="minisblack",
                     metadata={"axes": "CZYX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5,
                               "PhysicalSizeZ": 2.0, "Channel": {"Name": ["DAPI", "GFP"]}})
    job = ConversionJob.from_kwargs(str(src), str(tmp_path / "o"),
                                    {"pixel_reader": "standard", "metadata_reader": "bioio"})
    f = LegacyFile(job)
    try:
        assert f.indices == [{"scene": 0}]
        image = f.image({"scene": 0})
        assert image.axes == "tczyx" and image.shape == (1, 2, 3, 32, 40)
        assert hasattr(image.array, "dask")                       # the reader's own array
        assert image.metadata.pixel_sizes["z"].value == 2.0
        assert [ch.name for ch in image.metadata.channels] == ["DAPI", "GFP"]
        np.testing.assert_array_equal(image[0, :, :, :, :], data)
    finally:
        f.close()


def test_resource_backed_arrays_are_read_one_region_at_a_time():
    """bioio-nd2's array (resource-backed) is not thread-safe: LegacyImage reads it one
    region at a time, as eubi-bridge's own writer does (ozp's dask writer has no lock);
    other arrays are used as they are."""
    import threading
    import time

    import dask.array as da
    rbda = pytest.importorskip("resource_backed_dask_array")

    from eubi_bridge.conversion.legacy_image import _thread_safe

    class _Ctx:
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    active, peak, guard = [0], [0], threading.Lock()

    def slow(block):
        with guard:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.02)
        with guard:
            active[0] -= 1
        return block

    data = np.arange(8 * 64 * 64, dtype="u2").reshape(8, 64, 64)
    plain = da.from_array(data, chunks=(1, 32, 32))
    assert _thread_safe(plain) is plain
    backed = rbda.resource_backed_dask_array(plain.map_blocks(slow, dtype=data.dtype), _Ctx())
    safe = _thread_safe(backed)
    np.testing.assert_array_equal(safe.compute(scheduler="threads", num_workers=8), data)
    assert peak[0] == 1
    peak[0] = 0
    backed.compute(scheduler="threads", num_workers=8)          # unprotected: in parallel
    assert peak[0] > 1


def test_a_czi_phase_is_the_one_asked_for(tmp_path):
    """phase_index reaches the CZI reader (0.1.x never passed it on in a unary
    conversion, so phase 0 was always written)."""
    from tests._czi_writer import raw_payload, write_czi
    tiles = [dict(scene=0, plane={"C": 0, "H": h}, x=0, y=0, height=24, width=32, dtype="u1",
                  compression=0, payload=raw_payload(np.full((24, 32), 10 + 100 * h, "u1")))
             for h in range(2)]
    src = tmp_path / "phases.czi"
    write_czi(str(src), tiles)
    values = {}
    for phase in (0, 1):
        out = tmp_path / f"phase{phase}"
        log = _convert(src, out, True, ["--phase_index", str(phase)])
        assert "older readers" in log, log[-1500:]
        (store,) = out.glob("*.zarr")
        values[phase] = np.unique(np.asarray(zarr.open_group(str(store), mode="r")["0"])).tolist()
    assert values == {0: [10], 1: [110]}


@pytest.mark.parametrize("name", ["img.ome.tif", "IMG.OME.TIF", "Img.Ome.Tiff"])
def test_suffixes_are_matched_regardless_of_case(tmp_path, name):
    """An OME-TIFF named in capitals got Bio-Formats until 0.1.6 (the routing compared
    suffixes case-sensitively); it gets bioio's OME-TIFF reader whatever the case."""
    import asyncio

    from eubi_bridge.core.data_manager import _is_bioformats_backed
    from eubi_bridge.core.readers import read_single_image
    src = tmp_path / name
    tifffile.imwrite(str(src), RNG.integers(0, 255, (2, 16, 20)).astype("u1"), ome=True,
                     photometric="minisblack", metadata={"axes": "CYX"})
    reader = asyncio.run(read_single_image(str(src), pixel_reader="standard"))
    assert type(reader.img).__module__.startswith("bioio_ome_tiff"), type(reader.img).__module__
    assert not _is_bioformats_backed(reader)
