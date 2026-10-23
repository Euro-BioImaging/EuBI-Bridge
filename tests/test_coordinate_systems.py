"""Each output keeps the coordinate system its position is in (spatial plan 9.18, 3.26).

eubi takes every image with ``coll.pyramid(key)``, which stamps it with its system (the
custom metadata ``eubi:coordinate_system`` and its definition ``eubi:coordinate_systems``),
so outputs converted from one file can be placed together by another tool. The system's
axes are the output's own (after squeeze or crop, plan 3.12). An OME-Zarr input keeps the
system it names, and an NGFF 0.6 image placed by a scene is read with its position
(``apply_scene=True``, 5.5). An aggregated output takes the first file's system.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

micro_reader = pytest.importorskip("micro_reader")
pytest.importorskip("ome_zarr_pyramid")
tifffile = pytest.importorskip("tifffile")
zarr = pytest.importorskip("zarr")

from ome_zarr_pyramid.core.spatial import SYSTEM_KEY, SYSTEMS_KEY  # noqa: E402

RNG = np.random.default_rng(11)


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


def _attrs(store: Path) -> dict:
    attrs = dict(zarr.open_group(str(store), mode="r").attrs)
    return attrs


def _ngff(attrs: dict) -> dict:
    return attrs.get("ome") if isinstance(attrs.get("ome"), dict) else attrs


def _translation(attrs: dict) -> list:
    ms = _ngff(attrs)["multiscales"][0]
    return next(t["translation"] for t in ms["datasets"][0]["coordinateTransformations"]
                if t["type"] == "translation")


def _positioned_tiff(path: Path, x=100.0, y=50.0) -> Path:
    """One C Z Y X image with pixel sizes and a stage position."""
    data = RNG.integers(0, 4000, (2, 3, 40, 48)).astype("u2")
    with tifffile.TiffWriter(str(path), ome=True) as tw:
        tw.write(data, photometric="minisblack", metadata={
            "axes": "CZYX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5, "PhysicalSizeZ": 2.0,
            "Plane": {"PositionX": [x] * 6, "PositionXUnit": ["µm"] * 6,
                      "PositionY": [y] * 6, "PositionYUnit": ["µm"] * 6}})
    return path


def _needs_systems(path: Path):
    """(the system micro-reader places `path`'s image in, its (y, x) position in um);
    skips with a micro-reader that declares no systems (before the release with
    coordinate systems, plan 9.7). An OME-TIFF's stage position is the field's centre
    for micro-reader, so the position is its own, not the file's PositionX / Y."""
    with micro_reader.open(str(path)) as f:
        if not hasattr(f, "coordinate_systems"):
            pytest.skip("this micro-reader declares no coordinate systems")
        meta = f.images[0].metadata
        sid = getattr(meta, "coordinate_system", None)
        yx = [float(meta.position[a].value) for a in "yx"]
    if sid is None:
        pytest.skip("micro-reader places this image in no system")
    return str(sid), yx


def test_an_output_keeps_its_coordinate_system(tmp_path):
    src = _positioned_tiff(tmp_path / "img.ome.tif")
    sid, yx = _needs_systems(src)
    _convert(src, tmp_path / "out", ["--squeeze", "True"])
    attrs = _attrs(tmp_path / "out" / "img.zarr")
    assert attrs[SYSTEM_KEY] == sid
    (definition,) = attrs[SYSTEMS_KEY]
    assert definition["name"] == sid
    # the system's axes are the output's own (here c z y x), not the file's t c z y x
    axes = [a["name"] for a in _ngff(attrs)["multiscales"][0]["axes"]]
    assert [a["name"] for a in definition["axes"]] == axes
    assert _translation(attrs)[-2:] == yx


def test_an_unplaced_image_names_no_system(tmp_path):
    src = tmp_path / "plain.ome.tif"
    tifffile.imwrite(str(src), RNG.integers(0, 4000, (2, 32, 36)).astype("u2"), ome=True,
                     photometric="minisblack",
                     metadata={"axes": "CYX", "PhysicalSizeX": 0.25, "PhysicalSizeY": 0.25})
    _convert(src, tmp_path / "out", [])
    attrs = _attrs(tmp_path / "out" / "plain.zarr")
    assert SYSTEM_KEY not in attrs and SYSTEMS_KEY not in attrs


def _ome_zarr_with_system(path: Path, sid="lab-stage", translation=(0.0, 0.0, 12.5, -3.0)):
    from ome_zarr_pyramid import IO, Pyramid
    from ome_zarr_pyramid.core.spatial import minimal_system, stamp_system
    data = RNG.integers(0, 4000, (2, 6, 48, 56)).astype("u2")
    pyr = Pyramid().from_arrays([data], axis_order="czyx", scales=[[1, 2.0, 0.5, 0.5]],
                                unit_list=[None, "micrometer", "micrometer", "micrometer"],
                                version="0.5")
    pyr.meta.update_translation("0", list(translation))
    stamp_system(pyr, minimal_system(sid, pyr))
    IO().write_pyramid(pyr, str(path), overwrite=True)
    return path


def test_an_ome_zarr_inputs_system_is_kept(tmp_path):
    src = _ome_zarr_with_system(tmp_path / "in.ome.zarr")
    _convert(src, tmp_path / "out", ["--ome_zarr_version", "0.5"])
    attrs = _attrs(tmp_path / "out" / "in.zarr")
    assert attrs[SYSTEM_KEY] == "lab-stage"
    assert [d["name"] for d in attrs[SYSTEMS_KEY]] == ["lab-stage"]
    assert _translation(attrs) == [0.0, 0.0, 12.5, -3.0]


def test_an_image_placed_by_an_ngff_06_scene_keeps_its_position(tmp_path):
    """An image read alone out of a 0.6 scene-form store holds only its alignment in its
    levels; eubi reads it with apply_scene=True, so the output is where the scene put it."""
    from ome_zarr_pyramid import Pyramid, PyramidCollection
    from ome_zarr_pyramid.core.spatial import CoordinateSystem
    pyrs = {}
    for i, (y, x) in enumerate([(0.0, 0.0), (40.0, 70.0)]):
        data = RNG.integers(0, 4000, (2, 32, 36)).astype("u2")
        pyr = Pyramid().from_arrays([data], axis_order="cyx", scales=[[1, 0.5, 0.5]],
                                    unit_list=[None, "micrometer", "micrometer"],
                                    version="0.6")
        pyr.meta.update_translation("0", [0.0, y, x])
        pyrs[(("series", i),)] = pyr
    coll = PyramidCollection(pyrs, coordinate_systems=[CoordinateSystem("stage")],
                             system_of={k: "stage" for k in pyrs})
    store = tmp_path / "set.ome.zarr"
    coll.write_series(str(store), version="0.6")
    assert "scene" in _ngff(_attrs(store))                  # the scene form was written
    _convert(store / "1", tmp_path / "out", [])
    (out,) = (tmp_path / "out").glob("*.zarr")
    attrs = _attrs(out)
    assert _translation(attrs)[-2:] == [40.0, 70.0]
    assert attrs[SYSTEM_KEY] == "stage"


def test_an_aggregated_output_takes_the_first_files_system(tmp_path):
    folder = tmp_path / "set"
    folder.mkdir()
    for t, x in enumerate((100.0, 900.0)):            # the second file's position differs
        _positioned_tiff(folder / f"img_T{t + 1}.ome.tif", x=x)
    sid, yx = _needs_systems(folder / "img_T1.ome.tif")
    _convert(folder, tmp_path / "out", ["--time_tag", "T", "--concatenation_axes", "t"])
    (out,) = (tmp_path / "out").glob("*.zarr")
    attrs = _attrs(out)
    assert attrs[SYSTEM_KEY] == sid
    assert _translation(attrs)[-2:] == yx              # the first file's position


def test_the_system_gets_the_outputs_axes_without_changing_its_name():
    from ome_zarr_pyramid import Pyramid
    from ome_zarr_pyramid.core.spatial import (Axis, CoordinateSystem, pyramid_system,
                                               stamp_system)

    from eubi_bridge.conversion.pyramid_path import _system_for_output
    pyr = Pyramid().from_arrays([np.zeros((2, 8, 8), "u2")], axis_order="cyx",
                                unit_list=[None, "micrometer", "micrometer"])
    stamp_system(pyr, CoordinateSystem("stage", "Stage of the scope",
                                       tuple(Axis(a) for a in "tczyx")))
    out = _system_for_output(pyr)
    sid, cs = pyramid_system(out)
    assert (sid, cs.name, [a.name for a in cs.axes]) == ("stage", "Stage of the scope",
                                                         ["c", "y", "x"])
    assert _system_for_output(out) is out                   # already fits: unchanged
