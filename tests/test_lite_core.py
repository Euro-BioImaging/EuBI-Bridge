"""eubi-bridge-lite: the core runs without the standard-only stack (micro-reader
docs/PLAN.md, "Step 4 plan: eubi-bridge-lite", phase A).

Lite installs neither the JVM / Bio-Formats stack nor dask.  Each test runs
in a fresh process whose imports of those packages fail (tests/_lite/
sitecustomize.py; worker processes inherit it):

- every module of eubi_bridge imports;
- conversions through micro-reader run end to end -- unary on threads and
  on worker processes, aggregative, a mosaic split into tiles -- and write
  the same pixels;
- in those runs, every attempt to import the standard stack is a deliberate
  probe (eubi_bridge/utils/optional_deps.py).  An attempt elsewhere is code
  that swallows the ImportError: a feature lost silently in lite.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("micro_reader")
tifffile = pytest.importorskip("tifffile")

from .test_czi_real_mosaic import zeiss_mosaic  # noqa: E402,F401 - the real Zeiss mosaic

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

#: what eubi-bridge-lite does not install
STANDARD_ONLY = (
    "dask", "distributed", "dask_jobqueue", "xarray",
    "bioio", "bioio_base", "bioio_bioformats", "bioio_czi", "bioio_imageio", "bioio_lif",
    "bioio_nd2", "bioio_ome_tiff", "bioio_tifffile",
    "bfio", "scyjava", "jpype", "bioformats_jar", "install_jdk",
    "aicspylibczi", "pylibCZIrw", "nd2", "readlif", "imageio", "imageio_ffmpeg", "matplotlib",
)
#: modules that are not part of eubi-bridge's code paths: the Streamlit views
#: (streamlit is not a dependency at all)
NOT_CORE = ("eubi_bridge.views",)
#: where a lite run may try to import the standard stack: eubi-bridge's probes,
#: and libraries' own optional imports (which they handle themselves)
PROBES = ("utils/optional_deps.py:", "third-party:")


def _run_lite(code: str, attempts: Path = None, timeout: int = 900) -> subprocess.CompletedProcess:
    """*code* in a fresh Python whose imports of STANDARD_ONLY fail; the
    attempts are logged to *attempts*, if given."""
    env = dict(os.environ)
    env["EUBI_LITE_BLOCK"] = ",".join(STANDARD_ONLY)
    if attempts is not None:
        env["EUBI_LITE_ATTEMPTS"] = str(attempts)
    env["PYTHONPATH"] = os.pathsep.join([str(HERE / "_lite"), str(ROOT)]
                                        + [p for p in [env.get("PYTHONPATH")] if p])
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=str(ROOT), env=env,
                          capture_output=True, text=True, timeout=timeout)


def _check(result: subprocess.CompletedProcess) -> str:
    assert result.returncode == 0, (result.stdout[-3000:] + "\n" + result.stderr[-6000:])
    return result.stdout


def _no_hidden_attempts(attempts: Path) -> None:
    """Every logged attempt to import the standard stack came from a probe."""
    lines = attempts.read_text(encoding="utf-8").splitlines() if attempts.exists() else []
    hidden = sorted({line for line in lines if not line.split("\t")[1].startswith(PROBES)})
    assert not hidden, ("standard-only imports tried (and swallowed) in a lite run:\n"
                        + "\n".join(hidden))


def _convert(tmp_path: Path, source, out: Path, **options) -> None:
    """``EuBIBridge().to_zarr`` in a lite process; no hidden attempts."""
    attempts = tmp_path / "attempts.txt"
    args = ", ".join(f"{k}={v!r}" for k, v in options.items())
    _check(_run_lite(f"""
        if __name__ == "__main__":
            from eubi_bridge.ebridge import EuBIBridge
            EuBIBridge().to_zarr({str(source)!r}, {str(out)!r}, metadata_reader="micro",
                                 verbose=False, {args})
    """, attempts=attempts))
    _no_hidden_attempts(attempts)


def _read_zarr(store: Path) -> np.ndarray:
    import zarr
    return np.asarray(zarr.open_group(str(store), mode="r")["0"])


def _transforms(store: Path) -> dict:
    import zarr
    attrs = dict(zarr.open_group(str(store), mode="r").attrs)
    attrs = attrs.get("ome", attrs)
    return {t["type"]: t[t["type"]]
            for t in attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"]}


def test_the_blocker_blocks(tmp_path):
    """The harness itself: a blocked package cannot be imported, and the
    attempt is logged with the eubi_bridge function that made it."""
    attempts = tmp_path / "attempts.txt"
    out = _check(_run_lite("""
        from eubi_bridge.utils.optional_deps import is_installed
        print("dask installed:", is_installed("dask"))
    """, attempts=attempts))
    assert "dask installed: False" in out
    assert attempts.read_text().startswith("dask\tutils/optional_deps.py:is_installed")


def test_every_module_imports_without_the_standard_stack():
    out = _check(_run_lite(f"""
        import importlib, json, pkgutil, traceback
        import eubi_bridge
        failed = {{}}
        for info in pkgutil.walk_packages(eubi_bridge.__path__, "eubi_bridge."):
            if info.name.startswith({NOT_CORE!r}):
                continue
            try:
                importlib.import_module(info.name)
            except BaseException as exc:
                frames = [f for f in traceback.extract_tb(exc.__traceback__)
                          if "eubi_bridge" in f.filename]
                where = (f"{{frames[-1].filename.split('eubi_bridge')[-1]}}:{{frames[-1].lineno}}"
                         if frames else "?")
                failed[info.name] = f"{{type(exc).__name__}}: {{exc}} @ {{where}}"
        print(json.dumps(failed))
    """))
    failed = json.loads(out.strip().splitlines()[-1])
    assert not failed, "\n".join(f"{m}: {why}" for m, why in failed.items())


@pytest.mark.parametrize("workers", ["threads", "processes"])
def test_a_conversion_runs_without_the_standard_stack(tmp_path, workers):
    data = np.random.default_rng(5).integers(0, 4000, (2, 3, 6, 40, 50)).astype("u2")
    tifffile.imwrite(tmp_path / "img.ome.tif", data, ome=True, metadata={
        "axes": "TCZYX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5, "PhysicalSizeZ": 2.0})
    out = tmp_path / "out"
    options = {"use_threading": True} if workers == "threads" else {}
    _convert(tmp_path, tmp_path / "img.ome.tif", out, **options)
    (store,) = out.glob("*.zarr")
    np.testing.assert_array_equal(_read_zarr(store), data)


def test_an_aggregative_conversion_runs_without_the_standard_stack(tmp_path):
    planes = np.random.default_rng(6).integers(0, 4000, (4, 40, 50)).astype("u2")
    src = tmp_path / "in"
    src.mkdir()
    for z, plane in enumerate(planes):
        tifffile.imwrite(src / f"stack_z{z:03d}.tif", plane)
    out = tmp_path / "out"
    _convert(tmp_path, src, out, includes="stack", z_tag="_z", concatenation_axes="z")
    (store,) = out.glob("*.zarr")
    np.testing.assert_array_equal(np.squeeze(_read_zarr(store)), planes)


def test_a_mosaic_splits_into_placed_tiles_without_the_standard_stack(zeiss_mosaic, tmp_path):
    """A real Zeiss 2 x 2 mosaic, one output per tile: each tile keeps its
    stage position (micro-reader's; no aicspylibczi), so the tiles sit as far
    apart as aicspylibczi's bounding boxes say -- to within a pixel: the
    stage positions are measured (tile 1 is 23.04 um right of tile 0), the
    subblock boxes put them on the pixel grid (230 px of 0.1 um)."""
    from eubi_bridge.core.data_manager import czi_mosaic_tile_origins
    out = tmp_path / "out"
    _convert(tmp_path, zeiss_mosaic, out, as_mosaic=False, scene_index=0,
             mosaic_tile_index="all")
    stores = sorted(out.glob("*.zarr"))
    assert len(stores) == 4
    boxes = czi_mosaic_tile_origins(zeiss_mosaic)            # pixels, from aicspylibczi
    placed = {int(s.stem.rsplit("_tile", 1)[1]): _transforms(s) for s in stores}
    for m, transforms in placed.items():
        for axis, k in (("y", -2), ("x", -1)):
            step = transforms["scale"][k]
            moved = transforms["translation"][k] - placed[0]["translation"][k]
            assert moved == pytest.approx((boxes[m][axis] - boxes[0][axis]) * step, abs=step)


# -- phase B: micro-reader is the default; lite refuses what it lacks, clearly ----------

def _convert_fresh(tmp_path, source, out, expect_ok=True, **options):
    """``EuBIBridge(configpath=<fresh>).to_zarr`` in a lite process: the
    installation defaults, not this machine's config file."""
    attempts = tmp_path / "attempts.txt"
    args = "".join(f", {k}={v!r}" for k, v in options.items())
    result = _run_lite(f"""
        if __name__ == "__main__":
            from eubi_bridge.ebridge import EuBIBridge
            EuBIBridge(configpath={str(tmp_path / 'config')!r}).to_zarr(
                {str(source)!r}, {str(out)!r}, verbose=False{args})
    """, attempts=attempts)
    if expect_ok:
        _check(result)
        _no_hidden_attempts(attempts)
    return result


def test_micro_reader_is_the_installation_default():
    from eubi_bridge.core.config_models import MetadataConfig, ReaderConfig
    from eubi_bridge.ebridge import ConfigManager
    assert ConfigManager._ROOT_DEFAULTS["readers"]["pixel_reader"] == "micro"
    assert ConfigManager._ROOT_DEFAULTS["metadata"]["metadata_reader"] == "micro"
    assert ReaderConfig().pixel_reader == "micro"
    assert MetadataConfig().metadata_reader == "micro"


def test_a_conversion_with_the_defaults_needs_no_standard_stack(tmp_path):
    data = np.random.default_rng(7).integers(0, 4000, (2, 5, 30, 40)).astype("u2")
    tifffile.imwrite(tmp_path / "img.ome.tif", data, ome=True, metadata={"axes": "CZYX"})
    out = tmp_path / "out"
    _convert_fresh(tmp_path, tmp_path / "img.ome.tif", out)
    (store,) = out.glob("*.zarr")
    np.testing.assert_array_equal(np.squeeze(_read_zarr(store)), data)


@pytest.mark.parametrize("options, says", [
    ({"metadata_reader": "bfio"}, "metadata_reader='bfio'"),
    ({"on_slurm": True}, "on_slurm=True"),
])
def test_lite_refuses_the_full_stack_clearly(tmp_path, options, says):
    tifffile.imwrite(tmp_path / "img.tif", np.zeros((8, 8), "u2"))
    result = _convert_fresh(tmp_path, tmp_path / "img.tif", tmp_path / "out",
                            expect_ok=False, **options)
    assert result.returncode != 0
    assert "NeedsFullEubiBridge" in result.stderr and says in result.stderr
    assert "pip install eubi-bridge" in result.stderr


def test_a_file_micro_reader_does_not_read_is_refused_by_name(tmp_path):
    """In lite there is no Bio-Formats to fall back to: the error names the file."""
    source = tmp_path / "img.dv"                   # DeltaVision: Bio-Formats only
    source.write_bytes(b"not read by micro-reader" * 8)
    result = _convert_fresh(tmp_path, source, tmp_path / "out", expect_ok=False)
    assert result.returncode != 0
    assert "micro-reader does not read this file" in result.stderr
    assert "img.dv" in result.stderr and "pip install eubi-bridge" in result.stderr


def test_imaris_goes_through_micro_reader_and_keeps_its_pyramid(tmp_path):
    """An .ims through micro-reader in lite (eubi's own Imaris reader needs
    dask); keep_existing_resolutions writes the file's stored levels."""
    import h5py
    from .test_ims_reader import _write_synthetic_ims
    source = tmp_path / "img.ims"
    _write_synthetic_ims(source, n_resolution_levels=2, n_channels=2, base_shape=(4, 16, 16))
    with h5py.File(source, "r") as f:
        stored = [np.stack([f[f"/DataSet/ResolutionLevel {r}/TimePoint 0/Channel {c}/Data"][()]
                            for c in range(2)]) for r in range(2)]
    out = tmp_path / "out"
    _convert_fresh(tmp_path, source, out, keep_existing_resolutions=True)
    (store,) = out.glob("*.zarr")
    import zarr
    group = zarr.open_group(str(store), mode="r")
    for level, expected in enumerate(stored):
        np.testing.assert_array_equal(np.squeeze(np.asarray(group[str(level)])), expected)


def test_the_full_installation_is_not_taken_for_lite():
    """Where the Bio-Formats path's modules import, `has_bioformats` says so --
    otherwise the full eubi-bridge would refuse its own Bio-Formats readers."""
    import importlib
    from eubi_bridge.utils.capabilities import BIOFORMATS_MODULES, has_bioformats
    try:
        for module in BIOFORMATS_MODULES:
            importlib.import_module(module)
    except ImportError:
        pytest.skip("not the full installation")
    assert has_bioformats()
    for module in ("bfio", "bioio_base", "bioio_bioformats", "scyjava"):
        assert module in BIOFORMATS_MODULES          # what the readers really import


# -- phase C: the GUI in lite ------------------------------------------------------------

#: the GUI's test files: the Inspect page (viewer), the Convert page, batch mode
GUI_TESTS = ("test_inspect_page.py", "test_convert_page_actions.py", "test_gui_widgets.py",
             "test_lite_gui.py", "test_batch_cell_editing.py",
             "test_batch_dialog_dependencies.py", "test_batch_without_csv.py")


def test_the_gui_tests_pass_without_the_standard_stack():
    """The GUI -- viewer, Convert page, batch mode -- runs in eubi-bridge-lite."""
    result = _run_lite(f"""
        import os, sys
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import pytest
        sys.exit(pytest.main([*{[str(HERE / f) for f in GUI_TESTS]!r},
                              "-q", "-p", "no:cacheprovider", "-o", "addopts="]))
    """)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
    assert " passed" in result.stdout and " error" not in result.stdout


def test_batch_columns_offer_only_what_lite_can_do():
    out = _check(_run_lite("""
        from eubi_bridge.qt_gui.core import batch
        specs = {p.key: p for p in batch._PARAM_SPECS}
        print(specs["metadata_reader"].choices, specs["pixel_reader"].choices,
              "force_bioformats" in specs)
    """))
    assert out.strip().splitlines()[-1] == "('micro',) ('micro',) False"
