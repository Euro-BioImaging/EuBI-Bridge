"""A join's memory stays bounded (Workplan E25, the BARD report of 2026-11-05).

A join of thousands of one-plane files kept 128 files open, so every region reopened the
files of the last; each closed file left tifffile's objects in reference cycles, which
Python frees only in a rare full collection, and glibc kept the memory they took: the
worker grew past its plan over hours. These tests pin the three parts of the fix:

- a join keeps more files open (``join_open_files``), within the process's file handles;
- the garbage closed files leave is collected as files close (``_collect_if_due``);
- conversion workers use fewer glibc malloc arenas (``limit_malloc_arenas``).

The memory over time of a whole join is ``test_join_memory_flat.py``.
"""
from __future__ import annotations

import gc
import os
import sys
import uuid

import numpy as np
import pytest
import tifffile

from eubi_bridge.conversion import worker_init
from eubi_bridge.core import micro_source as ms

NS = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
CHANNELS = ("405", "488", "561", "638")


def make_stub_set(folder, nz: int, yx: int = 32, channels=CHANNELS) -> list:
    """A Viventis-like set: BinaryOnly stub TIFFs t0001_<ch>_plane<z>.tif and one
    ome-tiff.companion.ome describing them all. Returns the TIFF paths."""
    os.makedirs(folder, exist_ok=True)
    companion = f"urn:uuid:{uuid.uuid4()}"
    rng = np.random.default_rng(1)
    paths, tiff_data = [], []
    for z in range(nz):
        for c, ch in enumerate(channels):
            name = f"t0001_{ch}_plane{z + 1:04d}.tif"
            own = f"urn:uuid:{uuid.uuid4()}"
            stub = (f'<?xml version="1.0" encoding="UTF-8"?><OME xmlns="{NS}" UUID="{own}">'
                    f'<BinaryOnly MetadataFile="ome-tiff.companion.ome" UUID="{companion}"/></OME>')
            plane = rng.integers(0, 4000, (yx, yx), dtype=np.uint16)
            path = os.path.join(folder, name)
            tifffile.imwrite(path, plane, description=stub, metadata=None)
            paths.append(path)
            tiff_data.append(f'<TiffData FirstC="{c}" FirstZ="{z}" FirstT="0" IFD="0" PlaneCount="1">'
                             f'<UUID FileName="{name}">{own}</UUID></TiffData>')
    chans = "".join(f'<Channel ID="Channel:0:{c}" Name="{ch}" SamplesPerPixel="1"/>'
                    for c, ch in enumerate(channels))
    xml = (f'<?xml version="1.0" encoding="UTF-8"?><OME xmlns="{NS}" UUID="{companion}">'
           f'<Image ID="Image:0" Name="well 1"><Pixels ID="Pixels:0" DimensionOrder="XYZCT" '
           f'Type="uint16" SizeX="{yx}" SizeY="{yx}" SizeZ="{nz}" SizeC="{len(channels)}" '
           f'SizeT="1" PhysicalSizeX="0.5" PhysicalSizeY="0.5" PhysicalSizeZ="2.0">'
           f'{chans}{"".join(tiff_data)}</Pixels></Image></OME>')
    with open(os.path.join(folder, "ome-tiff.companion.ome"), "w", encoding="utf-8") as fh:
        fh.write(xml)
    return paths


@pytest.fixture
def clean_cache(monkeypatch):
    """An empty open-file cache, put back as it was afterwards."""
    ms.close_all()
    monkeypatch.setattr(ms, "_join_limits", [])
    monkeypatch.setattr(ms, "_closed", 0)
    monkeypatch.setattr(ms, "_last_collect", 0.0)
    yield
    ms.close_all()


# -- how many files a join keeps open ------------------------------------------------

def test_a_small_join_keeps_the_default(monkeypatch):
    monkeypatch.setattr(ms, "_handle_room", lambda wanted: 10**6)
    assert ms.join_open_files(100) == ms.MAX_OPEN_FILES
    assert ms.join_open_files(ms.MAX_OPEN_FILES) == ms.MAX_OPEN_FILES


def test_a_large_join_keeps_up_to_join_open_files(monkeypatch):
    monkeypatch.setattr(ms, "_handle_room", lambda wanted: 10**6)
    assert ms.join_open_files(500) == 500
    assert ms.join_open_files(10288) == ms.JOIN_OPEN_FILES


def test_the_file_handles_bound_the_open_files(monkeypatch):
    """Each open file holds a handle per decode thread: the limit follows the handles
    the process may hold, never below the default."""
    import micro_reader
    monkeypatch.setattr(micro_reader, "decode_threads", lambda: 4)        # 6 per file
    monkeypatch.setattr(ms, "_handle_room", lambda wanted: 4096)
    assert ms.join_open_files(10288) == (4096 - ms.SPARE_HANDLES) // 6
    monkeypatch.setattr(ms, "_handle_room", lambda wanted: 1024)          # a default soft limit
    assert ms.join_open_files(10288) == ms.MAX_OPEN_FILES


def test_the_handle_limit_is_raised_where_allowed():
    """POSIX: the soft limit goes up towards the hard one, never above it."""
    resource = pytest.importorskip("resource")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        room = ms._handle_room(soft + 16)
        if hard == resource.RLIM_INFINITY or hard >= soft + 16:
            assert room is None or room >= soft + 16 or room == 10240
        else:
            assert room == soft
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def test_the_memory_plan_bounds_the_open_files(monkeypatch):
    """Any format: no more files open than an eighth of the process's plan holds at what
    one of them holds; never fewer than the default."""
    monkeypatch.setattr(ms, "_handle_room", lambda wanted: 10**6)
    plan = 2500
    assert ms.join_open_files(10288, cost_mb=0.25, memory_mb=plan) == ms.JOIN_OPEN_FILES
    assert ms.join_open_files(10288, cost_mb=0.9, memory_mb=plan) == int(plan / 8 / 0.9)
    assert ms.join_open_files(10288, cost_mb=28, memory_mb=plan) == ms.MAX_OPEN_FILES
    assert ms.join_open_files(10288) == ms.JOIN_OPEN_FILES        # not measured: the count


def _whole_set_xml_files(folder, n_files: int, planes: int) -> list:
    """OME-TIFFs each carrying the OME-XML of a *planes*-plane set (one plane per file,
    the first *n_files* written): opened alone, each holds a parse of all of it."""
    os.makedirs(folder, exist_ok=True)
    names = [f"p{z:05d}.tif" for z in range(planes)]
    xml = (f'<?xml version="1.0" encoding="UTF-8"?><OME xmlns="{NS}"><Image ID="Image:0">'
           f'<Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16" SizeX="8" SizeY="8" '
           f'SizeZ="{planes}" SizeC="1" SizeT="1">'
           + "".join(f'<TiffData FirstZ="{z}" IFD="0" PlaneCount="1"><UUID FileName="{name}">'
                     f'urn:uuid:{uuid.uuid4()}</UUID></TiffData>' for z, name in enumerate(names))
           + "</Pixels></Image></OME>")
    paths = []
    for name in names[:n_files]:
        path = os.path.join(folder, name)
        tifffile.imwrite(path, np.zeros((8, 8), np.uint16), description=xml, metadata=None)
        paths.append(path)
    return paths


def test_what_one_open_file_holds_is_measured(tmp_path, clean_cache):
    """The join measures its own files: stub OME-TIFFs hold little (their companion's
    parse is shared), files carrying a whole set's OME-XML hold a parse each."""
    stubs = make_stub_set(str(tmp_path / "stubs"), nz=200)
    stub_cost = ms.open_file_cost_mb(stubs)
    assert stub_cost is not None and stub_cost < 1.0, stub_cost
    heavy = _whole_set_xml_files(str(tmp_path / "whole"), n_files=3, planes=3000)
    heavy_cost = ms.open_file_cost_mb(heavy)
    assert heavy_cost is not None and heavy_cost > 4 * stub_cost and heavy_cost > 1.0, (
        stub_cost, heavy_cost)
    assert ms.open_file_cost_mb(stubs[:1]) is None                # one file: nothing to tell


def test_the_limit_holds_while_the_join_runs(clean_cache, monkeypatch):
    monkeypatch.setattr(ms, "_handle_room", lambda wanted: 10**6)
    assert ms.open_files_limit() == ms.MAX_OPEN_FILES
    with ms.open_files_for_join(600) as limit:
        assert limit == 600 and ms.open_files_limit() == 600
        with ms.open_files_for_join(10288):              # two joins in one process
            assert ms.open_files_limit() == ms.JOIN_OPEN_FILES
        assert ms.open_files_limit() == 600
    assert ms.open_files_limit() == ms.MAX_OPEN_FILES


# -- closed files' garbage -------------------------------------------------------------

def _live(kind: str) -> int:
    return sum(1 for o in gc.get_objects() if type(o).__name__ == kind)


def _read_all(paths) -> None:
    for path in paths:
        arr, why = ms.micro_array(path, 0, None, multifile=False)
        assert arr is not None, why
        np.asarray(arr[:, :, :, :8, :8])


def test_closed_files_are_collected_as_they_close(tmp_path, clean_cache, monkeypatch):
    """With Python's automatic collection off (as it is in effect between rare full
    collections in a long join), closed files' tifffile objects stay alive unless the
    cache collects them: with the collection, no more than the open files and one batch
    of closes; without it, one per file read."""
    paths = make_stub_set(str(tmp_path / "set"), nz=30)            # 120 files
    monkeypatch.setattr(ms, "MAX_OPEN_FILES", 8)
    monkeypatch.setattr(ms, "COLLECT_EVERY_S", 0.0)
    monkeypatch.setattr(ms, "COLLECT_AFTER_CLOSES", 16)
    gc.collect()
    before = _live("TiffFile")
    gc.disable()
    try:
        _read_all(paths)
        with_collection = _live("TiffFile") - before
        ms.close_all()
        gc.collect()
        monkeypatch.setattr(ms, "COLLECT_AFTER_CLOSES", 10**9)    # as before the fix
        _read_all(paths)
        without = _live("TiffFile") - before
    finally:
        gc.enable()
    assert with_collection <= 8 + 16 + 2, with_collection
    assert without >= len(paths) - 8, without                       # the test has teeth


# -- malloc arenas ------------------------------------------------------------------

def test_the_users_malloc_arena_max_stands(monkeypatch):
    monkeypatch.setenv("MALLOC_ARENA_MAX", "4")
    assert worker_init.limit_malloc_arenas() is False


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="glibc only")
def test_the_arenas_are_limited_on_glibc(monkeypatch):
    monkeypatch.delenv("MALLOC_ARENA_MAX", raising=False)
    import ctypes
    if not hasattr(ctypes.CDLL(None), "gnu_get_libc_version"):
        pytest.skip("not glibc")
    assert worker_init.limit_malloc_arenas() is True


@pytest.mark.skipif(sys.platform.startswith("linux"), reason="the other systems")
def test_the_arenas_are_left_alone_elsewhere(monkeypatch):
    monkeypatch.delenv("MALLOC_ARENA_MAX", raising=False)
    assert worker_init.limit_malloc_arenas() is False
