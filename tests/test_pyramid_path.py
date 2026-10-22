"""The conversion path of eubi-bridge 0.2.0 (eubi_bridge/conversion/pyramid_path.py):
micro-reader (else the older readers) -> ome_zarr_pyramid, no ArrayManager.

Every case is converted through the real CLI and must write what eubi-bridge 0.1.x
wrote: group metadata (minus the version stamp), and per level shape, dtype, chunks and
pixels. Until 0.2.0 each case ran on both paths; the old path is retired (legacy_0_1/),
so its outputs are golden files (tests/_golden.py, tests/golden_0_1/).
"""
from __future__ import annotations

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

from tests import _golden  # noqa: E402

RNG = np.random.default_rng(7)


@pytest.fixture(autouse=True)
def _seeded():
    """Every test draws the same inputs, whichever tests run before it (the golden
    outputs hash their pixels)."""
    global RNG
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


def _convert(src: Path, out: Path, args: list) -> str:
    env = dict(os.environ, COLUMNS="250")
    result = subprocess.run([_eubi(), "to_zarr", str(src), str(out), *args],
                            capture_output=True, text=True, env=env)
    log = " ".join((result.stdout + result.stderr).split())
    assert result.returncode == 0, log[-3000:]
    return log


ALL = ["--series_index", "all", "--mosaic_tile_index", "all"]
CASES = {
    "defaults": ALL,
    "settings": ALL + ["--channel_labels", "0,Nuclei;1,auto", "--channel_colors", "1,red",
                       "--z_scale", "3", "--y_range", "0,24", "--dtype", "uint8",
                       "--channel_intensity_limits", "from_array"],
    "v05_no_squeeze": ["--series_index", "1", "--ome_zarr_version", "0.5", "--squeeze",
                       "False"],
    "auto_windows": ["--series_index", "0", "--channel_intensity_limits", "auto"],
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_new_path_writes_what_the_old_one_does(tmp_path, case):
    src = _ome_tiff(tmp_path / "img.ome.tif")
    log = _convert(src, tmp_path / "new", CASES[case])
    assert re.search(r"\d+ output\(s\), read by", log), log[-2000:]
    _golden.check(f"pyramid_path-ome_tiff-{case}", tmp_path / "new", tmp_path)


def test_new_path_writes_the_ome_xml_in_the_same_write(tmp_path):
    src = _ome_tiff(tmp_path / "img.ome.tif")
    _convert(src, tmp_path / "new", ["--series_index", "0"])
    store = tmp_path / "new" / "img_scene0.zarr"
    xml = (store / "OME" / "METADATA.ome.xml").read_text(encoding="utf-8")
    assert 'Name="DAPI"' in xml and 'SizeX="48"' in xml
    # a plain file: an image root that is no bioformats2raw layout gets no OME group
    # (ozp 0.3.1; 0.3.0 wrote one with series ["0"], which validators reject)
    assert not (store / "OME" / "zarr.json").exists()


def test_auto_windows_come_from_the_coarsest_level(tmp_path):
    """channel_intensity_limits='auto': start / end = the 0.1 and 99.9 percentiles of the
    smallest written level, min / max the dtype's range (decided 2026-10-15; 0.1.5 wrote
    the dtype's range, as from_dtype)."""
    src = _ome_tiff(tmp_path / "img.ome.tif")
    _convert(src, tmp_path / "new", ["--channel_intensity_limits", "auto"])
    group = zarr.open_group(str(tmp_path / "new" / "img_scene0.zarr"), mode="r")
    coarsest = np.asarray(group[str(len(list(group.array_keys())) - 1)])
    channels = group.attrs["omero"]["channels"]
    axes = [a["name"] for a in group.attrs["multiscales"][0]["axes"]]
    for k, ch in enumerate(channels):
        lo, hi = np.percentile(np.take(coarsest, k, axis=axes.index("c")), (0.1, 99.9))
        assert ch["window"] == {"min": 0, "max": 65535, "start": int(np.floor(lo)),
                                "end": int(np.ceil(hi))}
        assert 0 < ch["window"]["start"] < ch["window"]["end"] < 4000


def test_writer_eubi_is_retired_and_writes_with_ozp(tmp_path):
    """writer='eubi' (eubi-bridge's own writer, retired in 0.2.0, roadmap step 6): still
    accepted, warned about once, written by ome_zarr_pyramid on the new path."""
    src = _ome_tiff(tmp_path / "img.ome.tif")
    log = _convert(src, tmp_path / "out", ["--writer", "eubi"])
    assert "writer='eubi' is no longer available" in log
    assert re.search(r"\d+ output\(s\), read by", log), log[-2000:]
    assert (tmp_path / "out" / "img_scene0.zarr").is_dir()


def test_an_imagej_hyperstack_on_both_paths(tmp_path):
    """Step 1c, group 1 (TIFF family): an ImageJ hyperstack with z spacing and calibration."""
    src = tmp_path / "stack.tif"
    data = RNG.integers(0, 4000, (3, 2, 30, 34)).astype("u2")                 # Z C Y X
    tifffile.imwrite(str(src), data, imagej=True, resolution=(1 / 0.2, 1 / 0.2),
                     metadata={"axes": "ZCYX", "spacing": 1.5, "unit": "um"})
    log = _convert(src, tmp_path / "new", [])
    assert re.search(r"\d+ output\(s\), read by", log), log[-2000:]
    _golden.check("pyramid_path-imagej_hyperstack", tmp_path / "new", tmp_path)


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


@pytest.mark.parametrize("args", [ALL, ["--series_index", "all", "--as_mosaic"]],
                         ids=["tiles", "stitched"])
def test_czi_mosaics_on_both_paths(tmp_path, args):
    """Step 1c, group 3: tiles as separate outputs (named, positioned) or stitched."""
    src = _mosaic_czi(tmp_path / "mosaic.czi")
    log = _convert(src, tmp_path / "new", args)
    assert re.search(r"\d+ output\(s\), read by", log), log[-2000:]
    new = _golden.check(f"pyramid_path-czi_mosaic-{'tiles' if args is ALL else 'stitched'}",
                        tmp_path / "new", tmp_path)
    assert len(new) == (8 if args is ALL else 2)


def test_each_scene_has_its_own_tiles(tmp_path):
    """Roadmap 'Bug: scenes with different tile counts': 0.1.5 used scene 0's tile count
    for every scene. The new path writes each scene's own tiles, at their own places."""
    src = _mosaic_czi(tmp_path / "mosaic.czi", tiles_per_scene=(2, 3))
    _convert(src, tmp_path / "new", ALL)
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


def _check_rgb_tinted(name: str, new_root: Path, tmp: Path) -> dict:
    """The output against the golden 0.1.x output *name*, whose one-channel RGB samples
    were white: from 0.3.0 they are tinted red / green / blue (decided 2026-10-20),
    everything else as recorded."""
    expected = _golden.golden(name)
    for store in expected.values():
        attrs = store["attrs"]
        for i, channel in enumerate(attrs.get("ome", attrs)["omero"]["channels"]):
            assert channel["color"] == "FFFFFF"                 # as 0.1.x wrote them
            channel["color"] = ("FF0000", "00FF00", "0000FF")[i % 3]
    actual = _golden.describe(new_root, tmp)
    _golden.assert_same(expected, actual, name)
    return actual


@pytest.mark.parametrize("channels", [1, 2])
def test_rgb_channels_on_both_paths(tmp_path, channels):
    """Step 1c, group 4: samples become channels, named by eubi-bridge's rule: "<name>
    R/G/B", tinted; one unnamed channel: Channel 0/1/2, tinted from 0.3.0 (0.1.x: white)."""
    src = _rgb_czi(tmp_path / "rgb.czi", channels)
    _convert(src, tmp_path / "new", ALL)
    name = f"pyramid_path-rgb_czi-{channels}"
    new = (_check_rgb_tinted(name, tmp_path / "new", tmp_path) if channels == 1
           else _golden.check(name, tmp_path / "new", tmp_path))
    labels = [ch["label"] for ch in next(iter(new.values()))["attrs"]["omero"]["channels"]]
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
    log = _convert(src, tmp_path / "new", args)
    assert re.search(r"\d+ output\(s\), read by", log), log[-2000:]
    new = _golden.check(f"pyramid_path-czi_views-{'separate' if len(args) == 2 else 'joined'}",
                        tmp_path / "new", tmp_path)
    assert sorted(new) == (["views_view0.zarr", "views_view1.zarr"] if len(args) == 2
                           else ["views_view_concat.zarr"])


def test_rgb_views_joined_as_0_1_joined_them(tmp_path):
    """Roadmap step 6 (0.1.x took them on its own path): each view's RGB samples become
    channels, joined as View{v}_Ch{i}; tinted red / green / blue from 0.3.0."""
    from tests._czi_writer import raw_payload, write_czi
    tiles = [dict(scene=0, plane={"C": 0, "V": v}, x=0, y=0, height=24, width=32,
                  dtype="u1", samples=3, compression=0,
                  payload=raw_payload(RNG.integers(0, 255, (24, 32, 3)).astype("u1")))
             for v in range(2)]
    src = tmp_path / "rgb_views.czi"
    write_czi(str(src), tiles)
    args = ["--view_index", "all", "--concat_views"]
    log = _convert(src, tmp_path / "new", args)
    assert re.search(r"\d+ output\(s\), read by", log), log[-2000:]
    new = _check_rgb_tinted("pyramid_path-rgb_views_joined", tmp_path / "new", tmp_path)
    attrs = next(iter(new.values()))["attrs"]
    assert [c["label"] for c in attrs["omero"]["channels"]] == [
        f"View{v}_Ch{i}" for v in range(2) for i in range(3)]


def test_views_that_differ_are_not_joined(tmp_path):
    """Decided 2026-10-15: views at different positions (or with different pixel sizes)
    cannot share one OME-Zarr image; joining them is a clear error naming the way out.
    0.1.5 joined them and kept the last view's position for all channels."""
    src = _views_czi(tmp_path / "views.czi", view_offset_um=300.0)
    env = dict(os.environ, COLUMNS="250")
    result = subprocess.run([_eubi(), "to_zarr", str(src), str(tmp_path / "joined"),
                             "--view_index", "all", "--concat_views"],
                            capture_output=True, text=True, env=env)
    log = " ".join((result.stdout + result.stderr).split())
    assert result.returncode != 0
    assert "cannot be joined" in log and "Convert without --concat_views" in log
    assert not list((tmp_path / "joined").glob("*.zarr"))
    _convert(src, tmp_path / "separate", ["--view_index", "all"])
    places = []
    for name in ("views_view0.zarr", "views_view1.zarr"):
        attrs = zarr.open_group(str(tmp_path / "separate" / name), mode="r").attrs
        transforms = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"]
        places.append(next(t["translation"] for t in transforms if t["type"] == "translation"))
    assert places[1][-1] - places[0][-1] == pytest.approx(300.0)


def test_micro_check_pixels_is_retired_with_a_warning(tmp_path):
    """Decision 'D5: micro_check_pixels': saved configs still load; it warns once that
    it does nothing and how to compare instead."""
    src = _ome_tiff(tmp_path / "img.ome.tif")
    log = _convert(src, tmp_path / "new", ["--series_index", "0",
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
    log = _convert(src, tmp_path / "new", [])
    assert "1 output(s), read by ome_zarr_pyramid" in log, log[-2000:]
    _golden.check("pyramid_path-ome_zarr_input", tmp_path / "new", tmp_path)


def test_an_ome_zarr_inputs_translation_is_kept(tmp_path):
    """Decided 2026-10-15: the input's translation is kept (0.1.5 dropped it)."""
    src = _ome_zarr_input(tmp_path / "in.ome.zarr", translation=[0.0, 0.0, 12.5, -3.0])
    _convert(src, tmp_path / "new", [])
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
    log = _convert(src, tmp_path / "new", args)
    assert "1 output(s), read by" in log, log[-2000:]
    new = _golden.check("pyramid_path-kept_levels", tmp_path / "new", tmp_path)
    assert len(next(iter(new.values()))["levels"]) == 3


@pytest.mark.parametrize("keep", [False, True], ids=["computed", "kept"])
def test_an_ome_zarr_input_without_a_name_and_with_named_levels(tmp_path, keep):
    """NGFF leaves the multiscales name optional and names levels freely ('s0', 's1'):
    such an input converts (it raised KeyError: 'name'), and kept levels keep their names
    and pixels, as 0.1.x's own writer kept them."""
    data = RNG.integers(0, 4000, (2, 64, 80)).astype("u2")
    src = tmp_path / "in.ome.zarr"
    group = zarr.open_group(str(src), mode="w", zarr_format=2)
    group.create_array("s0", data=data, chunks=(1, 32, 32))
    group.create_array("s1", data=data[:, ::2, ::2], chunks=(1, 32, 32))
    axes = [{"name": "c", "type": "channel"}] + [
        {"name": ax, "type": "space", "unit": "micrometer"} for ax in "yx"]
    group.attrs["multiscales"] = [{"version": "0.4", "axes": axes, "datasets": [
        {"path": p, "coordinateTransformations": [{"type": "scale", "scale": [1, s, s]}]}
        for p, s in (("s0", 0.5), ("s1", 1.0))]}]
    _convert(src, tmp_path / "out", ["--keep_existing_resolutions"] if keep else [])
    (store,) = (tmp_path / "out").glob("*.zarr")
    out = zarr.open_group(str(store), mode="r")
    base = "s0" if keep else "0"
    np.testing.assert_array_equal(np.squeeze(out[base][...]), data)
    if keep:
        assert sorted(out.array_keys()) == ["s0", "s1"]
        np.testing.assert_array_equal(np.squeeze(out["s1"][...]), data[:, ::2, ::2])


# -- the pieces ------------------------------------------------------------------------

def test_positions_follow_eubis_index_options(caplog):
    from eubi_bridge.conversion.pyramid_path import _positions
    assert _positions("all", 3, "series") == [0, 1, 2]
    assert _positions("0,2", 3, "series") == [0, 2]
    assert _positions(1, 3, "series") == [1]
    assert _positions([2, 5], 3, "series") == [2]                 # 5 skipped, with a warning
    with pytest.raises(ValueError, match="valid indices are 0..2"):
        _positions(7, 3, "series")


def test_micro_reader_names_why_it_does_not_take_a_file():
    from eubi_bridge.conversion.pyramid_path import _micro_reason
    from eubi_bridge.core.config_models import ConversionJob
    job = ConversionJob.from_kwargs(input_path="a.ome.tif", output_path="o", kwargs={})
    assert _micro_reason(job) == ""
    vsi = job.model_copy(update={"input_path": "a.vsi"})
    assert "format" in _micro_reason(vsi)                 # the older readers take it


def test_smart_downscaling_makes_the_levels_isotropic(tmp_path):
    """apply_smart_downscaling: z 2.0 um, y / x 0.5 um -> y and x shrink by 4 per level,
    z stays, so the next level's voxels are isotropic (2.0 um)."""
    src = tmp_path / "aniso.ome.tif"
    data = RNG.integers(0, 4000, (8, 128, 160)).astype("u2")
    tifffile.imwrite(str(src), data, ome=True, photometric="minisblack", metadata={
        "axes": "ZYX", "PhysicalSizeZ": 2.0, "PhysicalSizeY": 0.5, "PhysicalSizeX": 0.5})
    _convert(src, tmp_path / "out", ["--apply_smart_downscaling", "True", "--n_layers", "2"])
    (store,) = (tmp_path / "out").glob("*.zarr")
    group = zarr.open_group(str(store), mode="r")
    attrs = group.attrs.asdict()
    datasets = attrs.get("ome", attrs)["multiscales"][0]["datasets"]
    scales = [d["coordinateTransformations"][0]["scale"][-3:] for d in datasets]
    assert scales == [[2.0, 0.5, 0.5], [2.0, 2.0, 2.0]]
    assert group[datasets[1]["path"]].shape[-3:] == (8, 32, 40)
