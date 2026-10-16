"""Aggregative conversion on the 0.2.0 path (eubi_bridge/conversion/aggregative_path.py,
roadmap step 3): many files joined into one OME-Zarr image without ArrayManager.

It must write the image eubi-bridge 0.1.x wrote (pixels, scales, chunks, channels),
except the one decided change: unnamed channels are numbered across the output from 0, as
on unary conversions ("Channel 0", "Channel 1"; 0.1.5 named them all "Channel 1"). The
0.1.x outputs are golden files (tests/_golden.py), recorded before the old path was
retired (roadmap step 6). Files that do not agree, stray files, multi-file OME-TIFFs and
files of several scenes give clear errors.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("micro_reader")
pytest.importorskip("ome_zarr_pyramid")
tifffile = pytest.importorskip("tifffile")
zarr = pytest.importorskip("zarr")

from tests import _golden  # noqa: E402

RNG = np.random.default_rng(11)


@pytest.fixture(autouse=True)
def _seeded():
    """The same inputs whichever tests run before (golden outputs hash pixels)."""
    global RNG
    RNG = np.random.default_rng(11)


def _old(src, out, args):
    """`src` converted on eubi-bridge 0.1.x's path (golden recording only)."""
    _convert(src, out, False, args)
    return out


def _labels(described: dict) -> list:
    (store,) = described.values()
    attrs = store["attrs"]
    return [(c["label"], c["color"]) for c in attrs.get("ome", attrs)["omero"]["channels"]]


def _eubi() -> str:
    found = shutil.which("eubi")
    if found:
        return found
    for name in ("eubi.exe", "eubi"):
        for folder in (Path(sys.executable).parent, Path(sys.executable).parent / "Scripts"):
            if (folder / name).exists():
                return str(folder / name)
    pytest.skip("no eubi executable next to this Python")


def _run(src: Path, out: Path, new: bool, args: list) -> subprocess.CompletedProcess:
    env = dict(os.environ, EUBI_NEW_PATH="1" if new else "0", COLUMNS="250")
    return subprocess.run([_eubi(), "to_zarr", str(src), str(out), *args],
                          capture_output=True, text=True, env=env)


def _convert(src: Path, out: Path, new: bool, args: list) -> str:
    result = _run(src, out, new, args)
    log = " ".join((result.stdout + result.stderr).split())
    assert result.returncode == 0, log[-3000:]
    return log


def _channel_time_set(folder: Path, names=None, dtype="u2", shape=(3, 40, 48)) -> Path:
    """2 channels x 3 time points, one Z Y X stack per file: Channel{c}_T{t}.ome.tif."""
    folder.mkdir(parents=True, exist_ok=True)
    for c in range(2):
        for t in range(3):
            data = RNG.integers(0, 4000, shape).astype(dtype)
            meta = {"axes": "ZYX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5,
                    "PhysicalSizeZ": 2.0}
            if names:
                meta["Channel"] = {"Name": [names[c]]}
            tifffile.imwrite(str(folder / f"Channel{c + 1}_T{t + 1}.ome.tif"), data,
                             ome=True, photometric="minisblack", metadata=meta)
    return folder


CT = ["--channel_tag", "Channel", "--time_tag", "T", "--concatenation_axes", "ct"]


def _describe(root: Path) -> dict:
    out = {}
    for store in sorted(root.glob("*.zarr")):
        group = zarr.open_group(str(store), mode="r")
        attrs = dict(group.attrs)
        levels = []
        for path in sorted(k for k in group.array_keys()):
            arr = group[path]
            levels.append((arr.shape, str(arr.dtype), arr.chunks, np.asarray(arr)))
        out[store.name] = (attrs, levels)
    return out


def _channels(store: Path) -> list:
    attrs = dict(zarr.open_group(str(store), mode="r").attrs)
    return (attrs.get("ome", attrs).get("omero") or {}).get("channels")


def test_named_channels_as_0_1_wrote_them(tmp_path):
    src = _channel_time_set(tmp_path / "in", names=["DAPI", "GFP"])
    log = _convert(src, tmp_path / "new", True, CT)
    assert "new path, 6 file(s) joined" in log
    old = _golden.golden("aggregative-named_channels", tmp_path,
                         old=lambda: _old(src, tmp_path / "old", CT))
    new = _golden.describe(tmp_path / "new", tmp_path)
    assert list(old) == list(new) and len(new) == 1
    (name,) = new
    for key in ("multiscales", "omero"):
        assert new[name]["attrs"][key] == old[name]["attrs"][key], key
    assert new[name]["levels"] == old[name]["levels"]
    assert new[name]["levels"][0][0] == [3, 2, 3, 40, 48]
    assert [label for label, _ in _labels(new)] == ["DAPI", "GFP"]


def test_unnamed_channels_are_numbered_across_the_output(tmp_path):
    src = _channel_time_set(tmp_path / "in")
    _convert(src, tmp_path / "new", True, CT)
    old = _golden.golden("aggregative-unnamed_channels", tmp_path,
                         old=lambda: _old(src, tmp_path / "old", CT))
    new = _golden.describe(tmp_path / "new", tmp_path)
    assert list(old) == list(new)
    assert [label for label, _ in _labels(old)] == ["Channel 1", "Channel 1"]          # 0.1.5
    assert [label for label, _ in _labels(new)] == ["Channel 0", "Channel 1"]
    assert [c for _, c in _labels(new)] == [c for _, c in _labels(old)] == ["FF0000", "00FF00"]
    assert [s["levels"] for s in new.values()] == [s["levels"] for s in old.values()]


def test_rgb_files_are_joined_as_0_1_joined_them(tmp_path):
    """Roadmap step 6 (0.1.x took them on its own path): samples become channels,
    coloured red / green / blue, numbered from 0 (the decided naming)."""
    src = tmp_path / "rgb"
    src.mkdir()
    for t in range(2):
        tifffile.imwrite(str(src / f"img_T{t}.tif"),
                         RNG.integers(0, 255, (24, 32, 3)).astype("u1"), photometric="rgb")
    args = ["--time_tag", "T", "--concatenation_axes", "t"]
    log = _convert(src, tmp_path / "new", True, args)
    assert "new path, 2 file(s) joined" in log
    old = _golden.golden("aggregative-rgb", tmp_path, old=lambda: _old(src, tmp_path / "old", args))
    new = _golden.describe(tmp_path / "new", tmp_path)
    assert _labels(old) == [("Channel 1", "FF0000"), ("Channel 2", "00FF00"),
                            ("Channel 3", "0000FF")]                                  # 0.1.5
    assert _labels(new) == [("Channel 0", "FF0000"), ("Channel 1", "00FF00"),
                            ("Channel 2", "0000FF")]
    assert [s["levels"] for s in new.values()] == [s["levels"] for s in old.values()]


@pytest.mark.parametrize("what", ["dtype", "shape"])
def test_files_that_do_not_agree_give_a_clear_error(tmp_path, what):
    src = _channel_time_set(tmp_path / "in")
    odd = src / "Channel2_T2.ome.tif"
    odd.unlink()
    data = (RNG.integers(0, 200, (3, 40, 48)).astype("u1") if what == "dtype"
            else RNG.integers(0, 4000, (3, 40, 50)).astype("u2"))
    tifffile.imwrite(str(odd), data, ome=True, photometric="minisblack",
                     metadata={"axes": "ZYX"})
    result = _run(src, tmp_path / "new", True, CT)
    log = " ".join((result.stdout + result.stderr).split())
    assert "Channel2_T2.ome.tif" in log
    assert ("pixel type uint8" in log) if what == "dtype" else ("has 50 along x" in log), \
        log[-3000:]
    assert not list((tmp_path / "new").glob("*.zarr/0"))


def test_screen_group_leaves_out_a_companion_and_refuses_mixed_formats(caplog):
    from eubi_bridge.conversion.aggregative_path import screen_group
    files = ["/d/a_T1.ome.tif", "/d/a_T2.ome.tif", "/d/a.companion.ome"]
    assert screen_group(files) == files[:2]
    assert screen_group(["/d/a_T1.tif", "/d/a_T2.TIFF", "/d/a_T3.ome.tif"]) == \
        ["/d/a_T1.tif", "/d/a_T2.TIFF", "/d/a_T3.ome.tif"]           # one family: TIFF
    with pytest.raises(ValueError, match=r"2 formats.*Not TIFF: a_T3_thumb\.png.*--excludes"):
        screen_group(["/d/a_T1.tif", "/d/a_T2.tif", "/d/a_T3_thumb.png"])


def test_a_stray_file_among_the_inputs_is_refused_before_anything_is_read(tmp_path):
    src = _channel_time_set(tmp_path / "in")
    (src / "Channel1_T1_thumb.png").write_bytes(b"not an image")
    result = _run(src, tmp_path / "new", True, CT)
    log = " ".join((result.stdout + result.stderr).split())
    assert "Channel1_T1_thumb.png" in log and "--excludes" in log, log[-3000:]


def _multifile_ome(folder: Path, layout: str) -> Path:
    """3 files, one time point each; 'full': every file's OME-XML lists all three;
    'stub': the first holds it, the others are BinaryOnly stubs pointing to it."""
    folder.mkdir(parents=True, exist_ok=True)
    names = [f"img_T{t}.ome.tif" for t in range(3)]
    uuids = [f"urn:uuid:{uuid.uuid4()}" for _ in names]
    ns = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
    tiffdata = "".join(
        f'<TiffData FirstT="{t}" FirstC="0" FirstZ="0" IFD="0" PlaneCount="1">'
        f'<UUID FileName="{n}">{u}</UUID></TiffData>' for t, (n, u) in enumerate(zip(names, uuids)))
    for t, (name, own) in enumerate(zip(names, uuids)):
        if layout == "full" or t == 0:
            xml = (f'<OME xmlns="{ns}" UUID="{own}"><Image ID="Image:0"><Pixels ID="Pixels:0" '
                   f'DimensionOrder="XYZCT" Type="uint16" SizeX="32" SizeY="24" SizeZ="1" '
                   f'SizeC="1" SizeT="3"><Channel ID="Channel:0:0" SamplesPerPixel="1"/>'
                   f'{tiffdata}</Pixels></Image></OME>')
        else:
            xml = (f'<OME xmlns="{ns}" UUID="{own}"><BinaryOnly MetadataFile="{names[0]}" '
                   f'UUID="{uuids[0]}"/></OME>')
        tifffile.imwrite(str(folder / name), np.full((24, 32), 100 * (t + 1), "u2"),
                         description=xml, metadata=None, photometric="minisblack")
    return folder


@pytest.mark.parametrize("layout", ["full", "stub"])
def test_a_multi_file_ome_tiff_is_not_joined_as_the_whole_set_from_each_file(tmp_path, layout):
    src = _multifile_ome(tmp_path / "in", layout)
    result = _run(src, tmp_path / "new", True,
                  ["--time_tag", "T", "--concatenation_axes", "t"])
    log = " ".join((result.stdout + result.stderr).split())
    assert "multi-file OME-TIFF" in log and "img_T0.ome.tif" in log, log[-3000:]
    assert not list((tmp_path / "new").glob("*.zarr/0"))


def test_many_files_never_stay_open_together(tmp_path, monkeypatch):
    """600 files joined along t through micro-reader's bounded file cache (shrunk to 8
    here): it never holds more than that plus the files being read at that moment (one
    per opening thread; a file in use is not closed under its reader), and every plane
    lands in its place."""
    from eubi_bridge.conversion import aggregative_path
    from eubi_bridge.core import micro_source
    monkeypatch.setattr(micro_source, "MAX_OPEN_FILES", 8)
    src = tmp_path / "in"
    src.mkdir()
    n = 600
    for t in range(n):
        tifffile.imwrite(str(src / f"img_T{t:04d}.tif"), np.full((16, 20), t, "u2"))
    peak = [0]
    original = micro_source._evict

    def evict():
        original()
        peak[0] = max(peak[0], len(micro_source._files))

    monkeypatch.setattr(micro_source, "_evict", evict)
    paths = [str(p) for p in sorted(src.glob("*.tif"))]
    out = tmp_path / "out" / "img_T_tset"
    done = asyncio.run(aggregative_path.convert_group(paths, str(out), {
        "time_tag": "T", "concatenation_axes": "t", "metadata_reader": "micro",
        "overwrite": True, "squeeze": False}))
    assert done
    assert 0 < peak[0] <= 8 + aggregative_path.OPEN_THREADS
    assert len(micro_source._files) <= 8                  # idle again: back to the cap
    arr = np.asarray(zarr.open_group(f"{out}.zarr", mode="r")["0"])
    assert arr.shape == (n, 1, 1, 16, 20)
    np.testing.assert_array_equal(arr[:, 0, 0, 0, 0], np.arange(n))


def test_a_conversion_table_runs_both_kinds_in_one_pool(tmp_path, caplog):
    """One-to-one rows and an aggregative group of one table run in one pool: a broken
    one-to-one file does not stop the group (0.1.5 ran the group only after every
    one-to-one job had succeeded), and the failure is reported at the end."""
    import logging

    from eubi_bridge.ebridge import EuBIBridge
    src = tmp_path / "in"
    src.mkdir()
    tifffile.imwrite(str(src / "single.tif"), np.full((16, 20), 7, "u2"))
    (src / "broken.tif").write_bytes(b"not a tiff")
    for t in range(3):
        tifffile.imwrite(str(src / f"series_T{t}.tif"), np.full((16, 20), t, "u2"))
    out = tmp_path / "out"
    rows = ["input_path,output_path,aggregative_group,concatenation_axes,time_tag",
            f"{src / 'single.tif'},{out},,,", f"{src / 'broken.tif'},{out},,,"]
    rows += [f"{src / f'series_T{t}.tif'},{out},G,t,T" for t in range(3)]
    table = tmp_path / "table.csv"
    table.write_text("\n".join(rows) + "\n")
    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError, match="1/3 tasks failed"):
        EuBIBridge().to_zarr(str(table), use_threading=True, max_processes=3, squeeze=False,
                             verbose=False)
    assert "2 one-to-one job(s) and 1 aggregative output(s) in one pool" in caplog.text
    assert (out / "single.zarr").is_dir()
    (joined,) = [p for p in out.glob("*.zarr") if p.name != "single.zarr"]
    arr = np.asarray(zarr.open_group(str(joined), mode="r")["0"])
    assert arr.shape == (3, 1, 1, 16, 20)
    np.testing.assert_array_equal(arr[:, 0, 0, 0, 0], [0, 1, 2])


def test_scene_index_all_joins_files_of_one_scene_on_the_new_path(tmp_path):
    """The GUI's batch default is scene_index='all': with one scene per file the new path
    joins them (it used to hand every GUI aggregative group to the old path)."""
    src = _channel_time_set(tmp_path / "in")
    args = CT + ["--scene_index", "all"]
    log = _convert(src, tmp_path / "new", True, args)
    assert "new path, 6 file(s) joined" in log
    old = _golden.golden("aggregative-scene_index_all", tmp_path,
                         old=lambda: _old(src, tmp_path / "old", args))
    new = _golden.describe(tmp_path / "new", tmp_path)
    assert [s["levels"] for s in new.values()] == [s["levels"] for s in old.values()]


@pytest.mark.parametrize("scenes", ["all", "0,1"])
def test_files_of_several_scenes_give_a_clear_error(tmp_path, scenes):
    """Roadmap step 6: an aggregative output joins ONE scene of each file. 0.1.5 folded
    the scenes of scene_index='all' into the joined axis (positions became time points)
    and crashed on a list."""
    src = tmp_path / "in"
    src.mkdir()
    for t in range(2):
        with tifffile.TiffWriter(str(src / f"img_T{t}.ome.tif"), ome=True) as tw:
            for scene in range(2):
                tw.write(np.full((16, 20), 10 * t + scene, "u2"), photometric="minisblack",
                         metadata={"axes": "YX"})
    result = _run(src, tmp_path / "new", True,
                  ["--time_tag", "T", "--concatenation_axes", "t", "--scene_index", scenes])
    log = " ".join((result.stdout + result.stderr).split())
    assert result.returncode != 0
    assert "joins one scene of each file" in log and "--scene_index" in log, log[-2000:]
    assert not list((tmp_path / "new").glob("*.zarr/0"))
