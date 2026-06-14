"""
Real-microscopy conversion audit — exercises ALL to_zarr parameter categories
(conversion, downscale, cluster, reader, concatenation) against real CZI / LIF /
ND2 / LSM files, not just synthetic TIFFs.

Files are referenced *in place* (never copied — some are GB-scale) from
``EUBI_TEST_DATA`` (default: ``C:/Users/oezdemir/Desktop/ome/input``). Every test
skips automatically when its source file is absent, so the suite is safe to run
anywhere. Marked ``realdata`` so it is opt-in:

    pytest -m realdata tests/test_realdata_conversions.py

Each conversion runs through the real ``eubi`` CLI (matching how users invoke it)
and asserts the *observable effect* of the parameter on the output OME-Zarr.

Reader-dimension coverage tests BOTH modes:
  * **extraction**  — one OME-Zarr per index (``scene_index all``,
    ``mosaic_tile_index all``, ``view_index all``, ``illumination_index all``).
  * **combination** — multiple indices grouped into a single OME-Zarr
    (``as_mosaic``, ``concat_views``, ``concat_illuminations``).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import zarr

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.validation_utils import (
    get_actual_zarr_path,
    get_resolution_count,
    validate_zarr_format,
)

pytestmark = pytest.mark.realdata

# ---------------------------------------------------------------------------
# File manifest — relative to the real-data root (skip-if-absent).
# ---------------------------------------------------------------------------

DATA_ROOT = Path(os.environ.get("EUBI_TEST_DATA", r"C:/Users/oezdemir/Desktop/ome/input"))

FILES = {
    "czi_workhorse": "czi/7015307/T=3_Z=5_CH=2.czi",          # tczyx (3,2,5,256,256)
    "czi_scenes":    "czi/7015307/S=2_T=3_Z=5_CH=1.czi",      # 2 scenes, 1 tile
    "czi_mosaic":    "czi/7015307/S=2_2x2_CH=1.czi",          # 2 scenes x 2x2 tiles
    "czi_rgb":       "czi/synthetic_rgb.czi",                 # RGB samples
    "czi_views":     "czi/MouseBrain_41Slices_1Tile_3Channel_2Illuminations_2Angles.czi",
    "lsm":           "small_dataset/FtsZ2-1_GFP_KO2-1_no10G.lsm",  # tczyx (1,3,12,512,512)
    "lif":           "medium_dataset/19_07_19_Lennard.lif",
    "nd2":           "medium_dataset/190821_L1_IRS-Akt_CS2003.nd2",
}

# Known full (un-squeezed) base shape for the workhorse, in tczyx order.
WORKHORSE_TCZYX = {"t": 3, "c": 2, "z": 5, "y": 256, "x": 256}


def _path(key: str) -> Path:
    p = DATA_ROOT / FILES[key]
    if not p.exists():
        pytest.skip(f"real-data file not available: {p}")
    return p


# ---------------------------------------------------------------------------
# CLI + output helpers
# ---------------------------------------------------------------------------

def _eubi_exe() -> str:
    exe = shutil.which("eubi")
    if exe is None:
        scripts = Path(sys.executable).parent / "Scripts"
        for cand in (scripts / "eubi.exe", scripts / "eubi",
                     Path(sys.executable).parent / "eubi"):
            if cand.exists():
                exe = str(cand)
                break
    if exe is None:
        pytest.skip("eubi executable not found on PATH")
    return exe


def _convert(src: Path, out: Path, *opts: str) -> subprocess.CompletedProcess:
    """Run `eubi to_zarr SRC OUT [opts]`; assert success."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    res = subprocess.run(
        [_eubi_exe(), "to_zarr", str(src), str(out), *opts],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env, timeout=3600,
    )
    assert res.returncode == 0, (
        f"to_zarr {src.name} {opts} failed:\n{res.stderr[-2000:]}\n{res.stdout[-1000:]}"
    )
    return res


def _output_zarrs(out: Path) -> list[Path]:
    """Return all leaf OME-Zarr groups under *out*.

    A leaf is a ``*.zarr`` directory that does not itself contain a nested
    ``*.zarr`` directory (so wrapper dirs like ``output.zarr/inner.zarr`` count
    only ``inner.zarr``, and resolution-level arrays inside a group — ``0``,
    ``1``, … — are never miscounted as separate outputs).
    """
    zdirs = [p for p in out.rglob("*.zarr") if p.is_dir()]
    leaves = [
        p for p in zdirs
        if not any(c.is_dir() and c.suffix == ".zarr" for c in p.iterdir())
    ]
    return sorted(set(leaves))


def _base_shape_axes(zarr_group: Path) -> tuple[tuple[int, ...], list[str]]:
    actual = get_actual_zarr_path(zarr_group)
    gr = zarr.open_group(actual, mode="r")
    shape = gr["0"].shape
    attrs_file = actual / ".zattrs"
    if attrs_file.exists():
        attrs = json.loads(attrs_file.read_text())
    else:
        attrs = json.loads((actual / "zarr.json").read_text()).get("attributes", {})
    axes = [a["name"] for a in attrs["multiscales"][0]["axes"]]
    return shape, axes


def _axis(zarr_group: Path, axis: str) -> int:
    shape, axes = _base_shape_axes(zarr_group)
    return shape[axes.index(axis)]


def _omero_labels(zarr_group: Path) -> list[str]:
    """Return the OMERO channel labels of an OME-Zarr group."""
    actual = get_actual_zarr_path(zarr_group)
    attrs_file = actual / ".zattrs"
    if attrs_file.exists():
        attrs = json.loads(attrs_file.read_text())
    else:
        attrs = json.loads((actual / "zarr.json").read_text()).get("attributes", {})
    return [c.get("label") for c in attrs.get("omero", {}).get("channels", [])]


# ===========================================================================
# CONVERSION category
# ===========================================================================

@pytest.mark.parametrize("fmt", [2, 3])
def test_conversion_zarr_format(tmp_path, fmt):
    out = tmp_path / "o"
    _convert(_path("czi_workhorse"), out, "--zarr_format", str(fmt))
    z = _output_zarrs(out)
    assert z, "no zarr produced"
    assert validate_zarr_format(z[0]) == fmt


@pytest.mark.parametrize("axis,opt,expected", [
    ("t", ("--time_range", "0,2"), 2),
    ("c", ("--channel_range", "0,1"), 1),
    ("z", ("--z_range", "0,3"), 3),
    ("y", ("--y_range", "0,128"), 128),
    ("x", ("--x_range", "0,128"), 128),
])
def test_conversion_crop_ranges(tmp_path, axis, opt, expected):
    """Each *_range crops the corresponding axis (squeeze off to keep the axis)."""
    out = tmp_path / "o"
    _convert(_path("czi_workhorse"), out, "--squeeze", "False", *opt)
    z = _output_zarrs(out)
    assert z
    assert _axis(z[0], axis) == expected


@pytest.mark.parametrize("dtype", ["uint16", "float32"])
def test_conversion_dtype_cast(tmp_path, dtype):
    out = tmp_path / "o"
    _convert(_path("czi_workhorse"), out, "--dtype", dtype)
    z = _output_zarrs(out)
    assert z
    actual = get_actual_zarr_path(z[0])
    assert str(zarr.open_group(actual, mode="r")["0"].dtype) == dtype


@pytest.mark.parametrize("comp", ["zstd", "gzip", "blosc", "none"])
def test_conversion_compressor(tmp_path, comp):
    out = tmp_path / "o"
    _convert(_path("czi_workhorse"), out, "--compressor", comp)
    assert _output_zarrs(out), f"no zarr for compressor={comp}"


def test_conversion_save_omexml_false(tmp_path):
    out = tmp_path / "o"
    _convert(_path("czi_workhorse"), out, "--save_omexml", "False")
    assert not list(out.rglob("*.ome.xml")), "OME-XML sidecar written despite save_omexml False"


def test_conversion_sharding_v3(tmp_path):
    out = tmp_path / "o"
    _convert(_path("czi_workhorse"), out,
             "--zarr_format", "3", "--z_shard_coef", "2",
             "--y_shard_coef", "2", "--x_shard_coef", "2")
    z = _output_zarrs(out)
    assert z and validate_zarr_format(z[0]) == 3


# ===========================================================================
# DOWNSCALE category
# ===========================================================================

@pytest.mark.parametrize("n_layers,expected", [(1, 1), (3, 3)])
def test_downscale_n_layers(tmp_path, n_layers, expected):
    out = tmp_path / "o"
    _convert(_path("czi_workhorse"), out, "--n_layers", str(n_layers))
    z = _output_zarrs(out)
    assert z
    assert get_resolution_count(z[0]) == expected


@pytest.mark.parametrize("method", ["simple", "mean", "median", "min", "max", "mode"])
def test_downscale_methods(tmp_path, method):
    out = tmp_path / "o"
    _convert(_path("czi_workhorse"), out, "--n_layers", "2", "--downscale_method", method)
    assert _output_zarrs(out), f"no zarr for downscale_method={method}"


# ===========================================================================
# CLUSTER category — assert it runs and produces valid, unaffected output
# ===========================================================================

@pytest.mark.parametrize("opt", [
    ("--max_workers", "2"),
    ("--queue_size", "8"),
    ("--region_size_mb", "128"),
    ("--max_concurrency", "2"),
    ("--max_concurrent_scenes", "2"),
    ("--use_threading", "True"),
    ("--tensorstore_data_copy_concurrency", "2"),
    ("--bf_tile_size_mb", "256"),
    ("--force_bioformats", "True"),
])
def test_cluster_params_run(tmp_path, opt):
    out = tmp_path / "o"
    _convert(_path("czi_workhorse"), out, *opt)
    z = _output_zarrs(out)
    assert z, f"no zarr for {opt}"
    assert _axis(z[0], "x") == WORKHORSE_TCZYX["x"]


# ===========================================================================
# READER category — EXTRACTION (one OME-Zarr per index)
# ===========================================================================

def test_reader_scene_extraction(tmp_path):
    """scene_index all -> one OME-Zarr per scene."""
    out = tmp_path / "o"
    _convert(_path("czi_scenes"), out, "--scene_index", "all")
    z = _output_zarrs(out)
    assert len(z) >= 2, f"scene extraction expected >=2 outputs, got {len(z)}"


def test_reader_tile_extraction(tmp_path):
    """mosaic_tile_index all (as_mosaic False) -> one OME-Zarr per tile (scene 0)."""
    out = tmp_path / "o"
    _convert(_path("czi_mosaic"), out,
             "--scene_index", "0", "--as_mosaic", "False",
             "--mosaic_tile_index", "all")
    z = _output_zarrs(out)
    assert len(z) >= 2, f"tile extraction expected >=2 outputs, got {len(z)}"


def test_reader_view_extraction(tmp_path):
    """view_index all (concat_views False) -> one OME-Zarr per view."""
    out = tmp_path / "o"
    _convert(_path("czi_views"), out,
             "--view_index", "all", "--illumination_index", "0",
             "--concat_views", "False")
    z = _output_zarrs(out)
    assert len(z) >= 2, f"view extraction expected >=2 outputs, got {len(z)}"


def test_reader_illumination_extraction(tmp_path):
    """illumination_index all (concat_illuminations False) -> one OME-Zarr per illumination."""
    out = tmp_path / "o"
    _convert(_path("czi_views"), out,
             "--illumination_index", "all", "--view_index", "0",
             "--concat_illuminations", "False")
    z = _output_zarrs(out)
    assert len(z) >= 2, f"illumination extraction expected >=2 outputs, got {len(z)}"


# ===========================================================================
# READER category — COMBINATION (multiple indices grouped into one OME-Zarr)
# ===========================================================================

def test_reader_tile_combination(tmp_path):
    """as_mosaic True -> tiles stitched into a single OME-Zarr (scene 0)."""
    out = tmp_path / "o"
    _convert(_path("czi_mosaic"), out, "--scene_index", "0", "--as_mosaic", "True")
    z = _output_zarrs(out)
    assert len(z) == 1, f"mosaic combination expected 1 output, got {len(z)}"


# MouseBrain has C=3 per view/illumination; concatenating 2 along the channel
# axis must yield 6 channels (not just a single grouped output).
_VIEWS_BASE_C = 3
_VIEWS_N = 2


def test_reader_view_combination(tmp_path):
    """concat_views True + view_index all -> ONE OME-Zarr with views concatenated
    along the channel axis (2 views x 3ch = 6 channels)."""
    out = tmp_path / "o"
    _convert(_path("czi_views"), out,
             "--view_index", "all", "--illumination_index", "0",
             "--concat_views", "True")
    z = _output_zarrs(out)
    assert len(z) == 1, f"view combination expected 1 output, got {len(z)}"
    assert _axis(z[0], "c") == _VIEWS_N * _VIEWS_BASE_C, (
        f"concat_views: expected {_VIEWS_N * _VIEWS_BASE_C} channels, "
        f"got {_axis(z[0], 'c')}"
    )
    # Channel labels must carry per-view provenance: View0_*, View1_*.
    labels = _omero_labels(z[0])
    assert sum(l.startswith("View0_") for l in labels) == _VIEWS_BASE_C, labels
    assert sum(l.startswith("View1_") for l in labels) == _VIEWS_BASE_C, labels


def test_reader_illumination_combination(tmp_path):
    """concat_illuminations True + illumination_index all -> ONE OME-Zarr with
    illuminations concatenated along the channel axis (2 illum x 3ch = 6)."""
    out = tmp_path / "o"
    _convert(_path("czi_views"), out,
             "--illumination_index", "all", "--view_index", "0",
             "--concat_illuminations", "True")
    z = _output_zarrs(out)
    assert len(z) == 1, f"illumination combination expected 1 output, got {len(z)}"
    assert _axis(z[0], "c") == _VIEWS_N * _VIEWS_BASE_C, (
        f"concat_illuminations: expected {_VIEWS_N * _VIEWS_BASE_C} channels, "
        f"got {_axis(z[0], 'c')}"
    )
    # Channel labels must carry per-illumination provenance: Illu0_*, Illu1_*.
    labels = _omero_labels(z[0])
    assert sum(l.startswith("Illu0_") for l in labels) == _VIEWS_BASE_C, labels
    assert sum(l.startswith("Illu1_") for l in labels) == _VIEWS_BASE_C, labels


def test_reader_view_illumination_combination(tmp_path):
    """concat_views + concat_illuminations together -> ONE OME-Zarr with the full
    view x illumination x channel cartesian product (2 x 2 x 3 = 12), each channel
    labeled View{v}_Illu{i}_<name>."""
    out = tmp_path / "o"
    _convert(_path("czi_views"), out,
             "--view_index", "all", "--illumination_index", "all",
             "--concat_views", "True", "--concat_illuminations", "True")
    z = _output_zarrs(out)
    assert len(z) == 1, f"both-combined expected 1 output, got {len(z)}"
    assert _axis(z[0], "c") == _VIEWS_N * _VIEWS_N * _VIEWS_BASE_C, _axis(z[0], "c")
    labels = _omero_labels(z[0])
    for v in range(_VIEWS_N):
        for i in range(_VIEWS_N):
            n = sum(l.startswith(f"View{v}_Illu{i}_") for l in labels)
            assert n == _VIEWS_BASE_C, f"View{v}_Illu{i}_ count={n}: {labels}"


def test_reader_rgb(tmp_path):
    """RGB CZI must convert and yield a 3-sample/channel output."""
    out = tmp_path / "o"
    _convert(_path("czi_rgb"), out)
    z = _output_zarrs(out)
    assert z
    assert _axis(z[0], "c") == 3


# ===========================================================================
# FORMAT coverage — each reader produces a valid OME-Zarr with defaults
# ===========================================================================

@pytest.mark.parametrize("key", ["lsm", "lif", "nd2"])
def test_format_coverage(tmp_path, key):
    out = tmp_path / "o"
    _convert(_path(key), out)
    z = _output_zarrs(out)
    assert z, f"{key}: no zarr produced"
    shape, axes = _base_shape_axes(z[0])
    assert "y" in axes and "x" in axes and len(shape) == len(axes)
