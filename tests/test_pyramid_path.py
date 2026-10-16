"""The 0.2.0 conversion path (eubi_bridge/conversion/pyramid_path.py, switched on with
EUBI_NEW_PATH=1): micro-reader -> ome_zarr_pyramid, no ArrayManager.

Every case is converted by both paths through the real CLI and the outputs must be the
same: group metadata (minus the version stamp), and per level shape, dtype, chunks and
pixels. Jobs the new path does not take yet must fall back to the old path.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("micro_reader")
pytest.importorskip("ome_zarr_pyramid")
tifffile = pytest.importorskip("tifffile")
zarr = pytest.importorskip("zarr")

RNG = np.random.default_rng(7)


def _ome_tiff(path: Path) -> Path:
    """Two scenes with pixel sizes, named and coloured channels, wavelengths and a
    stage position on the first."""
    a = RNG.integers(0, 4000, (2, 3, 40, 48)).astype("u2")          # C Z Y X
    b = RNG.integers(0, 4000, (2, 32, 36)).astype("u2")             # C Y X
    channel = {"Name": ["DAPI", "GFP"], "Color": [-16776961, 16711935],
               "EmissionWavelength": [461.0, 510.0], "ExcitationWavelength": [405.0, 488.0]}
    with tifffile.TiffWriter(str(path), ome=True) as tw:
        tw.write(a, photometric="minisblack", metadata={
            "axes": "CZYX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5, "PhysicalSizeZ": 2.0,
            "Channel": channel,
            "Plane": {"PositionX": [100.0] * 6, "PositionXUnit": ["µm"] * 6,
                      "PositionY": [50.0] * 6, "PositionYUnit": ["µm"] * 6}})
        tw.write(b, photometric="minisblack", metadata={
            "axes": "CYX", "PhysicalSizeX": 0.25, "PhysicalSizeY": 0.25})
    return path


def _eubi() -> str:
    found = shutil.which("eubi")
    if found:
        return found
    for name in ("eubi.exe", "eubi"):
        for folder in (Path(sys.executable).parent, Path(sys.executable).parent / "Scripts"):
            if (folder / name).exists():
                return str(folder / name)
    pytest.skip("no eubi executable next to this Python")


def _convert(src: Path, out: Path, new: bool, args: list) -> str:
    env = dict(os.environ, EUBI_NEW_PATH="1" if new else "0", COLUMNS="250")
    result = subprocess.run([_eubi(), "to_zarr", str(src), str(out), *args],
                            capture_output=True, text=True, env=env)
    log = " ".join((result.stdout + result.stderr).split())
    assert result.returncode == 0, log[-3000:]
    return log


def _strip_version(attrs: dict) -> dict:
    attrs = json.loads(json.dumps(attrs))
    for holder in (attrs, attrs.get("ome") or {}):
        block = holder.get("eubi_bridge")
        if isinstance(block, dict):
            block.pop("version", None)
    return attrs


def _describe(root: Path) -> dict:
    """{store name: (attrs, [(shape, dtype, chunks, pixels) per level])}."""
    out = {}
    for store in sorted(root.glob("*.zarr")):
        group = zarr.open_group(str(store), mode="r")
        levels = [(a.shape, str(a.dtype), a.chunks, np.asarray(a[...]))
                  for _, a in sorted(group.arrays(), key=lambda kv: int(kv[0]))]
        out[store.name] = (_strip_version(group.attrs.asdict()), levels)
    return out


def _same(old: dict, new: dict) -> None:
    assert sorted(old) == sorted(new)
    for name in old:
        (attrs_o, levels_o), (attrs_n, levels_n) = old[name], new[name]
        assert attrs_n == attrs_o, name
        assert len(levels_n) == len(levels_o), name
        for (so, do, co, po), (sn, dn, cn, pn) in zip(levels_o, levels_n):
            assert (sn, dn, cn) == (so, do, co), name
            np.testing.assert_array_equal(pn, po)


ALL = ["--scene_index", "all", "--mosaic_tile_index", "all"]
CASES = {
    "defaults": ALL,
    "settings": ALL + ["--channel_labels", "0,Nuclei;1,auto", "--channel_colors", "1,red",
                       "--z_scale", "3", "--y_range", "0,24", "--dtype", "uint8",
                       "--channel_intensity_limits", "from_array"],
    "v05_no_squeeze": ["--scene_index", "1", "--ome_zarr_version", "0.5", "--squeeze",
                       "False"],
    "auto_windows": ["--scene_index", "0", "--channel_intensity_limits", "auto"],
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_new_path_writes_what_the_old_one_does(tmp_path, case):
    src = _ome_tiff(tmp_path / "img.ome.tif")
    _convert(src, tmp_path / "old", False, CASES[case])
    log = _convert(src, tmp_path / "new", True, CASES[case])
    assert re.search(r"EUBI_NEW_PATH: \S+ new path", log), log[-2000:]
    old, new = _describe(tmp_path / "old"), _describe(tmp_path / "new")
    assert old
    _same(old, new)


def test_new_path_writes_the_ome_xml_in_the_same_write(tmp_path):
    src = _ome_tiff(tmp_path / "img.ome.tif")
    _convert(src, tmp_path / "new", True, ["--scene_index", "0"])
    store = tmp_path / "new" / "img_scene0.zarr"
    xml = (store / "OME" / "METADATA.ome.xml").read_text(encoding="utf-8")
    assert 'Name="DAPI"' in xml and 'SizeX="48"' in xml
    assert zarr.open_group(str(store / "OME"), mode="r").attrs["series"] == ["0"]


def test_auto_windows_come_from_the_coarsest_level(tmp_path):
    """channel_intensity_limits='auto': start / end = the 0.1 and 99.9 percentiles of the
    smallest written level, min / max the dtype's range (decided 2026-10-15; 0.1.5 wrote
    the dtype's range, as from_dtype)."""
    src = _ome_tiff(tmp_path / "img.ome.tif")
    _convert(src, tmp_path / "new", True, ["--channel_intensity_limits", "auto"])
    group = zarr.open_group(str(tmp_path / "new" / "img_scene0.zarr"), mode="r")
    coarsest = np.asarray(group[str(len(list(group.array_keys())) - 1)])
    channels = group.attrs["omero"]["channels"]
    axes = [a["name"] for a in group.attrs["multiscales"][0]["axes"]]
    for k, ch in enumerate(channels):
        lo, hi = np.percentile(np.take(coarsest, k, axis=axes.index("c")), (0.1, 99.9))
        assert ch["window"] == {"min": 0, "max": 65535, "start": int(np.floor(lo)),
                                "end": int(np.ceil(hi))}
        assert 0 < ch["window"]["start"] < ch["window"]["end"] < 4000


def test_jobs_the_new_path_does_not_take_yet_use_the_old_one(tmp_path):
    """Since step 2 only writer='eubi' (to be retired) keeps the old path."""
    src = _ome_tiff(tmp_path / "img.ome.tif")
    log = _convert(src, tmp_path / "out", True, ["--writer", "eubi"])
    assert "old path (writer='eubi')" in log
    assert (tmp_path / "out" / "img_scene0.zarr").is_dir()


def test_an_imagej_hyperstack_on_both_paths(tmp_path):
    """Step 1c, group 1 (TIFF family): an ImageJ hyperstack with z spacing and calibration."""
    src = tmp_path / "stack.tif"
    data = RNG.integers(0, 4000, (3, 2, 30, 34)).astype("u2")                 # Z C Y X
    tifffile.imwrite(str(src), data, imagej=True, resolution=(1 / 0.2, 1 / 0.2),
                     metadata={"axes": "ZCYX", "spacing": 1.5, "unit": "um"})
    _convert(src, tmp_path / "old", False, [])
    log = _convert(src, tmp_path / "new", True, [])
    assert re.search(r"EUBI_NEW_PATH: \S+ new path", log), log[-2000:]
    _same(_describe(tmp_path / "old"), _describe(tmp_path / "new"))


def _mosaic_czi(path: Path, tiles_per_scene=(4, 4)) -> Path:
    """A CZI with one 2-D mosaic per scene (32 x 32 tiles, two per row), uint16, 0.5 µm
    pixels and each tile's stage position (as a microscope writes them)."""
    from tests._czi_writer import raw_payload, write_czi
    scaling = ('<Scaling><Items><Distance Id="X"><Value>5E-07</Value></Distance>'
               '<Distance Id="Y"><Value>5E-07</Value></Distance></Items></Scaling>')
    tiles = []
    for scene, n in enumerate(tiles_per_scene):
        for m in range(n):
            x, y = 200 * scene + 32 * (m % 2), 32 * (m // 2)
            tile = RNG.integers(0, 4000, (32, 32)).astype("u2")
            tiles.append(dict(scene=scene, mosaic=m, plane={"C": 0}, x=x, y=y, height=32,
                              width=32, dtype="u2", compression=0, payload=raw_payload(tile),
                              tags={"StageXPosition": 1000 + 0.5 * x,
                                    "StageYPosition": 500 + 0.5 * y}))
    write_czi(str(path), tiles, metadata=scaling)
    return path


@pytest.mark.parametrize("args", [ALL, ["--scene_index", "all", "--as_mosaic"]],
                         ids=["tiles", "stitched"])
def test_czi_mosaics_on_both_paths(tmp_path, args):
    """Step 1c, group 3: tiles as separate outputs (named, positioned) or stitched."""
    src = _mosaic_czi(tmp_path / "mosaic.czi")
    _convert(src, tmp_path / "old", False, args)
    log = _convert(src, tmp_path / "new", True, args)
    assert re.search(r"EUBI_NEW_PATH: \S+ new path", log), log[-2000:]
    old = _describe(tmp_path / "old")
    assert len(old) == (8 if args is ALL else 2)
    _same(old, _describe(tmp_path / "new"))


def test_each_scene_has_its_own_tiles(tmp_path):
    """Roadmap 'Bug: scenes with different tile counts': 0.1.5 used scene 0's tile count
    for every scene. The new path writes each scene's own tiles, at their own places."""
    src = _mosaic_czi(tmp_path / "mosaic.czi", tiles_per_scene=(2, 3))
    _convert(src, tmp_path / "new", True, ALL)
    names = sorted(p.name for p in (tmp_path / "new").glob("*.zarr"))
    assert names == ["mosaic_scene0_tile0.zarr", "mosaic_scene0_tile1.zarr",
                     "mosaic_scene1_tile0.zarr", "mosaic_scene1_tile1.zarr",
                     "mosaic_scene1_tile2.zarr"]
    places = set()
    for name in names:
        attrs = zarr.open_group(str(tmp_path / "new" / name), mode="r").attrs
        transforms = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"]
        places.add(tuple(next((t["translation"] for t in transforms
                               if t["type"] == "translation"), [])))
        assert attrs["eubi_bridge"]["acquisition"]["tile_index"] == int(name[-6])
    assert len(places) == 5                                     # five different tiles


def _rgb_czi(path: Path, channels: int) -> Path:
    """A CZI of `channels` RGB channels (Bgr24), one 40 x 48 plane each."""
    from tests._czi_writer import raw_payload, write_czi
    tiles = [dict(scene=0, plane={"C": c}, x=0, y=0, height=40, width=48, dtype="u1",
                  samples=3, compression=0,
                  payload=raw_payload(RNG.integers(0, 255, (40, 48, 3)).astype("u1")))
             for c in range(channels)]
    write_czi(str(path), tiles)
    return path


@pytest.mark.parametrize("channels", [1, 2])
def test_rgb_channels_on_both_paths(tmp_path, channels):
    """Step 1c, group 4: samples become channels, named by eubi-bridge's rule (one
    channel: plain white channels; several: "<name> R/G/B" tinted)."""
    src = _rgb_czi(tmp_path / "rgb.czi", channels)
    _convert(src, tmp_path / "old", False, ALL)
    _convert(src, tmp_path / "new", True, ALL)
    old = _describe(tmp_path / "old")
    _same(old, _describe(tmp_path / "new"))
    labels = [ch["label"] for ch in next(iter(old.values()))[0]["omero"]["channels"]]
    assert len(labels) == 3 * channels


def _views_czi(path: Path, view_offset_um: float = 0.0) -> Path:
    """A CZI with 2 views x 2 channels (32 x 40, uint16, 0.5 µm pixels); view 1 sits
    `view_offset_um` further in x."""
    from tests._czi_writer import raw_payload, write_czi
    scaling = ('<Scaling><Items><Distance Id="X"><Value>5E-07</Value></Distance>'
               '<Distance Id="Y"><Value>5E-07</Value></Distance></Items></Scaling>')
    tiles = [dict(scene=0, plane={"C": c, "V": v}, x=0, y=0, height=32, width=40,
                  dtype="u2", compression=0,
                  payload=raw_payload(RNG.integers(0, 4000, (32, 40)).astype("u2")),
                  tags={"StageXPosition": 1000 + view_offset_um * v, "StageYPosition": 500})
             for v in range(2) for c in range(2)]
    write_czi(str(path), tiles, metadata=scaling)
    return path


@pytest.mark.parametrize("args", [["--view_index", "all"],
                                  ["--view_index", "all", "--concat_views"]],
                         ids=["separate", "joined"])
def test_czi_views_on_both_paths(tmp_path, args):
    """Step 1c, group 5: one output per view (``_view{v}``), or the views joined along c
    (``_view_concat``, channels ``View{v}_<name>``)."""
    src = _views_czi(tmp_path / "views.czi")
    _convert(src, tmp_path / "old", False, args)
    log = _convert(src, tmp_path / "new", True, args)
    assert re.search(r"EUBI_NEW_PATH: \S+ new path", log), log[-2000:]
    old = _describe(tmp_path / "old")
    assert sorted(old) == (["views_view0.zarr", "views_view1.zarr"] if len(args) == 2
                           else ["views_view_concat.zarr"])
    _same(old, _describe(tmp_path / "new"))


def test_views_that_differ_are_not_joined(tmp_path):
    """Decided 2026-10-15: views at different positions (or with different pixel sizes)
    cannot share one OME-Zarr image; joining them is a clear error naming the way out.
    0.1.5 joined them and kept the last view's position for all channels."""
    src = _views_czi(tmp_path / "views.czi", view_offset_um=300.0)
    env = dict(os.environ, EUBI_NEW_PATH="1", COLUMNS="250")
    result = subprocess.run([_eubi(), "to_zarr", str(src), str(tmp_path / "joined"),
                             "--view_index", "all", "--concat_views"],
                            capture_output=True, text=True, env=env)
    log = " ".join((result.stdout + result.stderr).split())
    assert result.returncode != 0
    assert "cannot be joined" in log and "Convert without --concat_views" in log
    assert not list((tmp_path / "joined").glob("*.zarr"))
    _convert(src, tmp_path / "separate", True, ["--view_index", "all"])
    places = []
    for name in ("views_view0.zarr", "views_view1.zarr"):
        attrs = zarr.open_group(str(tmp_path / "separate" / name), mode="r").attrs
        transforms = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"]
        places.append(next(t["translation"] for t in transforms if t["type"] == "translation"))
    assert places[1][-1] - places[0][-1] == pytest.approx(300.0)


def test_the_new_path_is_the_default_and_eubi_new_path_0_goes_back(tmp_path):
    """Step 1d: micro-reader files take the new path without any switch; EUBI_NEW_PATH=0
    is the private way back to the old path (until step 6 deletes it)."""
    src = _ome_tiff(tmp_path / "img.ome.tif")
    env = {k: v for k, v in os.environ.items() if k != "EUBI_NEW_PATH"}
    env["COLUMNS"] = "250"
    for name, extra in (("default", {}), ("old", {"EUBI_NEW_PATH": "0"})):
        result = subprocess.run([_eubi(), "to_zarr", str(src), str(tmp_path / name),
                                 "--scene_index", "0"], capture_output=True, text=True,
                                env=dict(env, **extra))
        log = " ".join((result.stdout + result.stderr).split())
        assert result.returncode == 0, log[-2000:]
        assert ("new path" in log) == (name == "default"), log[-2000:]
    _same(_describe(tmp_path / "old"), _describe(tmp_path / "default"))


def test_micro_check_pixels_is_retired_with_a_warning(tmp_path):
    """Decision 'D5: micro_check_pixels': saved configs still load; it warns once that
    it does nothing and how to compare instead."""
    src = _ome_tiff(tmp_path / "img.ome.tif")
    log = _convert(src, tmp_path / "new", True, ["--scene_index", "0",
                                                  "--micro_check_pixels", "True"])
    assert "micro_check_pixels no longer has an effect" in log
    assert "pixel_reader='standard'" in log
    assert (tmp_path / "new" / "img_scene0.zarr").is_dir()


def _ome_zarr_input(path: Path, translation=None) -> Path:
    """An OME-Zarr 0.4 image (c z y x, named + coloured channels) as a conversion input."""
    from ome_zarr_pyramid import IO, Pyramid
    data = RNG.integers(0, 4000, (2, 6, 48, 56)).astype("u2")
    pyr = Pyramid().from_arrays([data], axis_order="czyx", scales=[[1, 2.0, 0.5, 0.5]],
                                unit_list=[None, "micrometer", "micrometer", "micrometer"],
                                version="0.4")
    pyr = pyr.set_channels({0: {"label": "DAPI", "color": "0000FF"}, 1: {"label": "GFP"}})
    if translation:
        pyr.meta.update_translation("0", translation)
    IO().write_pyramid(pyr, str(path), overwrite=True)
    return path


def test_an_ome_zarr_input_on_both_paths(tmp_path):
    """Roadmap step 2, part 3: read by ozp itself (NgffFile), the same output."""
    src = _ome_zarr_input(tmp_path / "in.ome.zarr")
    _convert(src, tmp_path / "old", False, [])
    log = _convert(src, tmp_path / "new", True, [])
    assert "new path, 1 output(s), ome_zarr_pyramid" in log, log[-2000:]
    _same(_describe(tmp_path / "old"), _describe(tmp_path / "new"))


def test_an_ome_zarr_inputs_translation_is_kept(tmp_path):
    """Decided 2026-10-15: the input's translation is kept (0.1.5 dropped it)."""
    src = _ome_zarr_input(tmp_path / "in.ome.zarr", translation=[0.0, 0.0, 12.5, -3.0])
    _convert(src, tmp_path / "new", True, [])
    attrs = zarr.open_group(str(tmp_path / "new" / "in.zarr"), mode="r").attrs
    transforms = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    assert {"type": "translation", "translation": [0.0, 0.0, 12.5, -3.0]} in transforms


def test_kept_levels_on_both_paths(tmp_path):
    """Roadmap step 2, part 4: keep_existing_resolutions writes an input's stored levels
    as they are (an OME-Zarr input with 3 levels here), on both paths alike."""
    from ome_zarr_pyramid import IO, Pyramid
    data = RNG.integers(0, 4000, (2, 4, 96, 80)).astype("u2")
    pyr = Pyramid().from_arrays([data], axis_order="czyx", scales=[[1, 2.0, 0.5, 0.5]],
                                unit_list=[None, "micrometer", "micrometer", "micrometer"],
                                version="0.4").downscale(n_layers=3, defer=False)
    src = tmp_path / "levels.ome.zarr"
    IO().write_pyramid(pyr, str(src), overwrite=True)
    args = ["--keep_existing_resolutions"]
    _convert(src, tmp_path / "old", False, args)
    log = _convert(src, tmp_path / "new", True, args)
    assert "new path, 1 output(s)" in log, log[-2000:]
    old = _describe(tmp_path / "old")
    assert len(next(iter(old.values()))[1]) == 3
    _same(old, _describe(tmp_path / "new"))


# -- the pieces ------------------------------------------------------------------------

def test_positions_follow_eubis_index_options(caplog):
    from eubi_bridge.conversion.pyramid_path import _positions
    assert _positions("all", 3, "scene") == [0, 1, 2]
    assert _positions("0,2", 3, "scene") == [0, 2]
    assert _positions(1, 3, "scene") == [1]
    assert _positions([2, 5], 3, "scene") == [2]                 # 5 skipped, with a warning
    with pytest.raises(ValueError, match="valid indices are 0..2"):
        _positions(7, 3, "scene")


def test_unsupported_names_the_reason():
    from eubi_bridge.conversion.pyramid_path import _micro_reason, unsupported
    from eubi_bridge.core.config_models import ConversionJob
    job = ConversionJob.from_kwargs(input_path="a.ome.tif", output_path="o", kwargs={})
    assert unsupported(job) == "" and _micro_reason(job) == ""
    vsi = job.model_copy(update={"input_path": "a.vsi"})
    assert unsupported(vsi) == ""                         # the older readers take it
    assert "format" in _micro_reason(vsi)
    job.conversion.writer = "eubi"
    assert unsupported(job) == "writer='eubi'"
