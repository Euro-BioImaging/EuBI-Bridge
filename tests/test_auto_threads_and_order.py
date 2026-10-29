"""E1: decode_threads "auto", an explicit decode_memory capped at half the process,
and the biggest inputs started first."""
import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from eubi_bridge.conversion import dispatcher
from eubi_bridge.conversion.conversion_worker import _decode_sharers
from eubi_bridge.core import memory
from eubi_bridge.core.config_models import ClusterConfig


# -- the setting --------------------------------------------------------------------

def test_decode_threads_is_auto_by_default_and_takes_numbers():
    assert ClusterConfig().decode_threads == "auto"
    assert ClusterConfig(decode_threads=0).decode_threads == "auto"      # the GUI's 0
    assert ClusterConfig(decode_threads="AUTO").decode_threads == "auto"
    assert ClusterConfig(decode_threads="6").decode_threads == 6
    for bad in (-1, "many"):
        with pytest.raises(ValueError, match="decode_threads"):
            ClusterConfig(decode_threads=bad)


# -- how many threads -----------------------------------------------------------------

@pytest.fixture
def cores(monkeypatch):
    def set_cores(n):
        monkeypatch.setattr(memory, "available_cores", lambda: n)
    return set_cores


def test_auto_shares_the_cores_between_the_conversions(cores):
    cores(16)
    assert memory.decode_threads("auto", running=4) == 4
    assert memory.decode_threads("auto", running=1) == 16
    assert memory.decode_threads("auto", running=32) == 1            # never 0
    cores(64)
    assert memory.decode_threads("auto", running=1) == memory.MAX_DECODE_THREADS
    assert memory.decode_threads(6, running=4) == 6                  # a number as given


def test_auto_gives_fewer_threads_when_the_reader_share_is_small(cores):
    cores(16)
    # a 400 MB process: the reader's quarter (100 MB) less the caches (64 MB) holds
    # two 16 MB blocks, so 2 threads rather than memory taken from the writer
    assert memory.decode_threads("auto", 1, process=400, unit_mb=16) == 2
    # a block over 16 MB streams by rows: a thread still needs 16 MB
    assert memory.decode_threads("auto", 1, process=400, unit_mb=200) == 2
    assert memory.decode_threads("auto", 1, process=8000, unit_mb=4) == 16


def test_max_threads_is_auto_too(cores):
    from eubi_bridge.conversion.conversion_worker import _threads_per_scene
    assert ClusterConfig().max_threads == "auto"
    assert ClusterConfig(max_threads=0).max_threads == "auto"
    with pytest.raises(ValueError, match="max_threads"):
        ClusterConfig(max_threads="many")
    cores(16)
    assert memory.write_threads("auto", running=4) == 4
    assert memory.write_threads("auto", running=32) == 1
    assert memory.write_threads(3, running=4) == 3
    # 4 processes on 16 cores: 4 threads each, shared by 2 scenes
    assert _threads_per_scene(ClusterConfig(max_processes=4, max_concurrent_scenes=2)) == 2
    assert _threads_per_scene(ClusterConfig(max_threads=6)) == 6


def test_the_memory_line_is_logged_once_per_plan(caplog):
    from types import SimpleNamespace
    from eubi_bridge.conversion import conversion_worker
    conversion_worker._memory_logged.clear()
    job = SimpleNamespace(cluster=ClusterConfig(process_memory_mb=2000),
                          readers=SimpleNamespace(pixel_reader="bioformats"),
                          metadata=SimpleNamespace(metadata_reader="bioformats"))
    with caplog.at_level(logging.DEBUG, logger=conversion_worker.logger.name):
        for _field in range(5):                       # five fields of one plate
            conversion_worker._memory_plan_for(None, job)
    lines = [r for r in caplog.records if r.getMessage().startswith("Memory:")]
    assert [r.levelno for r in lines] == [logging.INFO] + [logging.DEBUG] * 4


def test_available_cores_follows_slurm(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "3")
    assert memory.available_cores() == 3
    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    assert memory.available_cores() >= 1


def test_conversions_in_one_process_share_one_decode_pool():
    assert _decode_sharers(ClusterConfig(use_threading=True, max_processes=4)) == 1
    assert _decode_sharers(ClusterConfig(max_processes=4)) == 4
    assert _decode_sharers(ClusterConfig(max_processes=4, concurrent_jobs=2)) == 2


# -- the explicit reader memory ---------------------------------------------------------

def test_an_explicit_decode_memory_is_capped_at_half_the_process(caplog):
    memory._warned.clear()
    with caplog.at_level(logging.WARNING, logger="eubi_bridge.core.memory"):
        decode, cache = memory.reader_share(995, "1GB", 4, unit_mb=8)
        memory.reader_share(995, "1GB", 4, unit_mb=8)
    assert decode + cache == pytest.approx(995 / 2)          # was 1024 of the 995 MB
    assert sum("more than half" in r.getMessage() for r in caplog.records) == 1
    small, small_cache = memory.reader_share(995, "300MB", 4, unit_mb=8)
    assert small + small_cache == pytest.approx(300)         # within half: as given


def test_the_cap_never_leaves_too_little_for_one_block():
    decode, _cache = memory.reader_share(400, "1GB", 4, unit_mb=300)
    assert decode >= 300                                     # a 300 MB block still decodes


# -- the biggest first ---------------------------------------------------------------------

def _file(path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)
    return str(path)


def test_inputs_are_ordered_by_what_they_hold(tmp_path):
    small = _file(tmp_path / "a.czi", 10)
    big = _file(tmp_path / "b.czi", 5000)
    plate = tmp_path / "plate"
    index = _file(plate / "Index.idx.xml", 100)               # an index: its folder counts
    for k in range(4):
        _file(plate / f"r01c01f0{k}.tiff", 2000)
    folder = tmp_path / "set"
    _file(folder / "x.tif", 3000)
    same = _file(tmp_path / "c.czi", 10)
    paths = [small, big, index, str(folder), "s3://bucket/x.zarr", same]
    order = dispatcher._largest_first(paths)
    assert [paths[i] for i in order] == [index, big, str(folder), small, same,
                                         "s3://bucket/x.zarr"]
    # a group of files for one output counts all of them
    assert dispatcher._input_bytes([small, big]) == 5010


def test_jobs_start_biggest_first_and_report_in_their_own_order():
    started = []

    def work(job):
        started.append(job)
        if job == "b":
            raise RuntimeError("b broke")
        return job.upper()

    jobs = ["a", "b", "c"]
    with pytest.raises(RuntimeError, match=r"Task 1: b broke"):     # b is still task 1
        asyncio.run(dispatcher._dispatch(jobs, work, ThreadPoolExecutor, {"max_workers": 1},
                                         use_threading=True, tsc=1, order=[2, 0, 1]))
    assert started == ["c", "a", "b"]

    results = asyncio.run(dispatcher._dispatch(["a", "c"], lambda j: j.upper(), ThreadPoolExecutor,
                                               {"max_workers": 1}, use_threading=True,
                                               tsc=1, order=[1, 0]))
    assert results == ["A", "C"]


# -- the GUI ----------------------------------------------------------------------------

def test_the_gui_shows_auto_as_zero_and_saves_it_as_auto():
    from eubi_bridge.qt_gui.server.config_manager import _config_to_react, _react_to_config
    from eubi_bridge.qt_gui.workers.conversion_worker import _build_kwargs
    gui = {"cluster": {"decodeThreads": 0}, "reader": {"useMicroReader": True}}
    backend = _react_to_config(gui)
    assert backend["cluster"]["decode_threads"] == "auto"
    assert _config_to_react(backend)["cluster"]["decodeThreads"] == 0
    assert _build_kwargs(gui)["decode_threads"] == "auto"
    assert _config_to_react({"cluster": {"decode_threads": 5}})["cluster"]["decodeThreads"] == 5
