"""Memory that finished conversions held goes to the ones still busy (Workplan E27).

A run's budget used to be split once, over max_processes, before the first conversion
started; when conversions ended, what they held stayed unused and the long ones kept
their small share to the end (Bugra's cluster run of the BARD wells, 2026-11-06: three
small wells done, two large ones still on level 0 with 2 GB each of 8). Now the run
records its conversions and how many have ended (memory_share.plan_tasks); once fewer
are left than run at once, each busy one may grow its claim (memory_share.regrow), and
the writer takes it before each pass and during one (dyna-zarr's io.memory_share).
"""
from __future__ import annotations

import contextlib
import json
import subprocess
import sys

import pytest

from eubi_bridge.core import memory_share


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv(memory_share.LEDGER_ENV, str(tmp_path / "ledger"))
    monkeypatch.delenv(memory_share.ENV, raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(memory_share, "POLL_S", 0.05)
    memory_share._said_waiting.clear()
    return memory_share.ledger_dir()


def _budget(gb):
    return lambda setting: gb * 1024.0


def _mb(claim):
    return json.loads(claim.path.read_text())["mb"]


def _run_entry():
    return json.loads(memory_share._own_run().read_text())


def test_the_busy_ones_grow_once_the_others_have_ended(ledger):
    with memory_share.run():
        memory_share.run_total_mb("4GB", _budget(4))
        memory_share.plan_tasks(4, 4, 4000.0)
        with contextlib.ExitStack() as keep:
            a, b = (keep.enter_context(memory_share.task_claim(1000.0)) for _ in range(2))
            with memory_share.task_claim(1000.0), memory_share.task_claim(1000.0):
                assert memory_share.regrow(a.path) == (1000.0, 4)    # all four busy
            assert _run_entry()["ended"] == 2
            # two left: each may have half of the run's 4000 MB
            assert memory_share.regrow(a.path) == (2000.0, 2)
            assert memory_share.regrow(b.path) == (2000.0, 2)
            assert _mb(a) + _mb(b) <= 4000.0


def test_nothing_grows_while_conversions_wait_their_turn(ledger):
    # six conversions, four at once: one ended, but the next one takes its place
    with memory_share.run():
        memory_share.run_total_mb("4GB", _budget(4))
        memory_share.plan_tasks(6, 4, 4000.0)
        with memory_share.task_claim(1000.0) as a:
            with memory_share.task_claim(1000.0):
                pass
            assert memory_share.regrow(a.path) == (1000.0, 4)


def test_never_past_what_the_others_hold_and_never_less(ledger):
    with memory_share.run():
        memory_share.run_total_mb("3GB", _budget(3))
        memory_share.plan_tasks(3, 3, 3000.0)
        with memory_share.task_claim(1000.0) as a, memory_share.task_claim(1000.0) as b:
            with memory_share.task_claim(1000.0):
                pass
            data = json.loads(b.path.read_text())
            b.path.write_text(json.dumps({**data, "mb": 2500.0}))   # b grew first
            # a's fair part is 1500, but only 500 is free: it keeps its 1000
            assert memory_share.regrow(a.path) == (1000.0, 2)
            assert _mb(a) == 1000.0


def test_beside_another_run_the_part_is_shared(ledger):
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        ledger.mkdir(parents=True, exist_ok=True)
        (ledger / "run-other.json").write_text(json.dumps(
            {"pid": other.pid, "started_at": memory_share._start_time(other.pid),
             "budget_mb": 4000.0}))
        (ledger / "claim-other.json").write_text(json.dumps(
            {"pid": other.pid, "started_at": memory_share._start_time(other.pid),
             "mb": 1000.0}))
        with memory_share.run():
            memory_share.run_total_mb("4GB", _budget(4))
            memory_share.plan_tasks(2, 2, 4000.0)
            with memory_share.task_claim(4000.0) as a:     # half: 2000, two runs
                with memory_share.task_claim(4000.0):
                    pass
                # one conversion left, two runs: 4000 / 1 / 2 = 2000 at most
                mb, busy = memory_share.regrow(a.path)
                assert busy == 1 and mb <= 2000.0
                assert mb + 1000.0 <= 4000.0
    finally:
        other.kill()
        other.wait()


def test_without_recorded_conversions_nothing_grows(ledger):
    with memory_share.run():
        memory_share.run_total_mb("4GB", _budget(4))
        with memory_share.task_claim(1000.0) as a:
            assert memory_share.regrow(a.path) is None
    assert memory_share.regrow(None) is None


def test_a_failed_conversion_counts_as_ended(ledger):
    with memory_share.run():
        memory_share.run_total_mb("4GB", _budget(4))
        memory_share.plan_tasks(2, 2, 2000.0)
        with pytest.raises(RuntimeError):
            with memory_share.task_claim(1000.0):
                raise RuntimeError("damaged file")
        assert _run_entry()["ended"] == 1


def test_a_long_conversion_takes_what_the_short_ones_left(tmp_path, monkeypatch):
    """A real run, conversions as threads: one large image and two tiny ones, two at
    once. The large one reads slowly (a delay per plane), so it is still on its first
    pass when both tiny ones have ended: its share grows while it writes (its log says
    so), and its output is complete."""
    np = pytest.importorskip("numpy")
    tifffile = pytest.importorskip("tifffile")
    import importlib
    import time

    import zarr
    from dyna_zarr import io as dyna_io
    if not hasattr(dyna_io, "memory_share"):
        pytest.skip("needs dyna-zarr with io.memory_share (0.1.2)")
    from dyna_zarr.dynamic_array import DynamicArray
    from eubi_bridge.ebridge import EuBIBridge
    monkeypatch.setattr(importlib.import_module("dyna_zarr.io"), "_SHARE_POLL_S", 0.05)
    monkeypatch.setenv(memory_share.LEDGER_ENV, str(tmp_path / "ledger"))
    monkeypatch.delenv(memory_share.ENV, raising=False)
    read = DynamicArray._read_direct

    def slow(self, key):                  # 0.1 s per plane read of the large image
        if tuple(self.shape[-3:]) == (48, 1024, 1024) and isinstance(key, tuple):
            z = key[len(key) - 3]
            planes = len(range(*z.indices(48))) if isinstance(z, slice) else 1
            time.sleep(0.1 * planes)
        return read(self, key)
    monkeypatch.setattr(DynamicArray, "_read_direct", slow)
    src = tmp_path / "in"
    src.mkdir()
    big = np.random.default_rng(1).integers(0, 4000, (48, 1024, 1024), dtype="u2")
    tifffile.imwrite(src / "a_big.tif", big, metadata={"axes": "ZYX"})
    for name in ("b_small", "c_small"):
        tifffile.imwrite(src / f"{name}.tif", np.ones((2, 64, 64), "u2"),
                         metadata={"axes": "ZYX"})
    log = tmp_path / "run.log"
    EuBIBridge().to_zarr(str(src), str(tmp_path / "out"), use_threading=True,
                         max_processes=2, memory_budget="2GB", max_threads=2,
                         region_size_mb=4, ome_zarr_version="0.5", log_file=str(log),
                         verbose=False)
    text = log.read_text(encoding="utf-8", errors="replace")
    assert "a_big.zarr: conversions of this run have ended; this one now has 2048 MB"         in text, text[-4000:]
    # the line comes from the share dyna polls DURING the pass (one pass here, its two
    # threads given: dyna's own "memory share grew" needs a later pass or more threads,
    # dyna-zarr's tests/test_memory_share.py)
    np.testing.assert_array_equal(
        zarr.open_array(str(tmp_path / "out" / "a_big.zarr" / "0"))[...].squeeze(), big)
