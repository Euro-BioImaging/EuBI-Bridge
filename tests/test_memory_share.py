"""One memory budget shared by the eubi runs on one machine / in one SLURM job
(core/memory_share.py; the BARD report of 2026-11-04: four batches side by side each
planned with the whole budget, and the job was killed).

The other runs here are fakes: ledger entries written for a live helper process."""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time

import pytest

from eubi_bridge.core import memory, memory_share


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv(memory_share.LEDGER_ENV, str(tmp_path / "ledger"))
    monkeypatch.delenv(memory_share.ENV, raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(memory_share, "POLL_S", 0.05)
    memory_share._said_waiting.clear()
    return memory_share.ledger_dir()


@pytest.fixture
def other_process():
    """A live process standing for another eubi run."""
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    yield p
    p.kill()
    p.wait()


def _entry(folder, name, pid, **fields):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.json"
    path.write_text(json.dumps({"pid": pid, "started_at": memory_share._start_time(pid),
                                **fields}))
    return path


def _total(budget_gb):
    return lambda setting: budget_gb * 1024.0


def test_a_run_alone_plans_with_its_own_budget_and_claims_it(ledger):
    with memory_share.run() as path:
        assert path.parent == ledger and path.exists()
        assert memory_share.run_total_mb("8GB", _total(8)) == 8192
        with memory_share.task_share(2048.0) as granted:
            assert granted == 2048.0
            (claim,) = ledger.glob("claim-*.json")
            assert json.loads(claim.read_text())["mb"] == 2048.0
        assert not list(ledger.glob("claim-*.json"))
    assert not path.exists()                                  # unregistered at the end


def test_outside_a_run_nothing_is_shared(ledger):
    with memory_share.task_share(1000.0) as granted:
        assert granted == 1000.0
    assert not ledger.exists()


def test_auto_beside_another_run_takes_the_shared_budget(ledger, other_process, caplog):
    _entry(ledger, "run-other", other_process.pid, budget_mb=120 * 1024.0)
    with memory_share.run(), caplog.at_level(logging.INFO):
        # its own 'auto' would be half of what the other run left free
        assert memory_share.run_total_mb("auto", _total(4)) == 120 * 1024
    assert any("the budget shared with 1 other eubi run(s) here" in r.getMessage()
               for r in caplog.records)


def test_the_smallest_budget_is_shared(ledger, other_process):
    _entry(ledger, "run-other", other_process.pid, budget_mb=120 * 1024.0)
    with memory_share.run():
        assert memory_share.run_total_mb("10GB", _total(10)) == 10 * 1024
        # and recorded, so the other run's next conversions share 10 GB too
        assert memory_share.pool_mb(memory_share.live_runs()) == 10 * 1024


def test_four_runs_of_120gb_share_120gb_not_480(ledger, other_process):
    """The BARD case: each conversion gets its fair part, and the claims of all runs
    stay within the one budget."""
    for k in range(3):
        _entry(ledger, f"run-other{k}", other_process.pid, budget_mb=120 * 1024.0)
    with memory_share.run():
        total = memory_share.run_total_mb("120GB", _total(120))
        assert total == 120 * 1024
        with memory_share.task_share(30 * 1024.0) as granted:  # its plan: 120 / 4 at once
            assert granted == pytest.approx(30 * 1024 / 4)     # a quarter: 4 runs


def test_a_conversion_waits_while_the_others_hold_the_budget(ledger, other_process,
                                                             caplog):
    _entry(ledger, "run-other", other_process.pid, budget_mb=8 * 1024.0)
    held = _entry(ledger, "claim-other", other_process.pid, mb=8 * 1024.0)

    # The other conversion ends once this one says it waits - not after a fixed delay: on a
    # slow runner the run's setup (the ledger lock retries once a second on Windows) took
    # longer than such a delay, the memory was free before it was asked for, and nothing
    # waited (CI, release v0.4.0, Windows / Python 3.11). The fallback only keeps a broken
    # wait from hanging the test.
    ended = threading.Event()

    class _EndTheOther(logging.Handler):
        def emit(self, record):
            if "a.tif: waiting for memory" in record.getMessage():
                held.unlink(missing_ok=True)
                ended.set()

    fallback = threading.Timer(30.0, lambda: held.unlink(missing_ok=True))
    fallback.start()
    handler = _EndTheOther(level=logging.INFO)
    share_logger = logging.getLogger(memory_share.logger.name)
    share_logger.addHandler(handler)
    try:
        with memory_share.run(), caplog.at_level(logging.INFO):
            memory_share.run_total_mb("8GB", _total(8))
            with memory_share.task_share(4096.0, label="a.tif") as granted:
                pass
    finally:
        share_logger.removeHandler(handler)
        fallback.cancel()
    assert ended.is_set()                                     # it waited for the other
    assert granted == 2048.0                                  # half: two runs
    messages = [r.getMessage() for r in caplog.records]
    assert any("a.tif: waiting for memory" in m for m in messages)
    assert any("a.tif: 2048 MB of the shared 8.0 GB (1 other eubi run(s) here)" in m
               for m in messages)


def test_it_starts_with_less_when_half_its_part_is_free(ledger, other_process):
    _entry(ledger, "run-other", other_process.pid, budget_mb=8 * 1024.0)
    _entry(ledger, "claim-other", other_process.pid, mb=6.5 * 1024)
    with memory_share.run():
        memory_share.run_total_mb("8GB", _total(8))
        with memory_share.task_share(4096.0) as granted:      # fair 2048, 1536 free
            assert granted == pytest.approx(1536.0)


def test_entries_of_ended_processes_do_not_count(ledger):
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    _entry(ledger, "run-gone", gone.pid, budget_mb=1024.0)
    _entry(ledger, "claim-gone", gone.pid, mb=1024.0)
    with memory_share.run():
        assert memory_share.run_total_mb("8GB", _total(8)) == 8192
        with memory_share.task_share(4096.0) as granted:
            assert granted == 4096.0
    assert not (ledger / "run-gone.json").exists()


def test_slurm_jobs_keep_separate_ledgers(ledger, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "4711")
    assert memory_share.ledger_dir().name == "slurm-4711"
    monkeypatch.delenv("SLURM_JOB_ID")
    assert memory_share.ledger_dir().name == "machine"


def test_process_mb_plans_with_the_shared_budget(ledger, other_process, monkeypatch):
    from eubi_bridge.core.config_models import ClusterConfig
    monkeypatch.setattr(memory, "available_mb", lambda: 500 * 1024.0)
    _entry(ledger, "run-other", other_process.pid, budget_mb=16 * 1024.0)
    with memory_share.run():
        # 'auto' alone would be 250 GB here; beside the other run it is the shared 16
        assert memory.process_mb(ClusterConfig(max_processes=4), 4) == pytest.approx(4096)


def _claims_seen(folder, stop, peaks):
    while not stop.is_set():
        total = 0.0
        for path in folder.glob("claim-*.json"):
            try:
                total += json.loads(path.read_text())["mb"]
            except (OSError, ValueError, KeyError):
                pass
        peaks.append(total)
        time.sleep(0.01)


def test_two_real_runs_side_by_side_stay_within_one_budget(ledger, tmp_path):
    """Two `eubi to_zarr` runs at once, 3 GB each: their conversions' claims together
    never pass 3 GB, and each run says it shares."""
    np = pytest.importorskip("numpy")
    tifffile = pytest.importorskip("tifffile")
    import os
    import shutil
    eubi = shutil.which("eubi")
    if eubi is None:
        pytest.skip("the eubi command is not on PATH")
    runs = []
    for k in range(2):
        src = tmp_path / f"in{k}"
        src.mkdir()
        for i in range(4):
            tifffile.imwrite(src / f"f{i}.tif", np.zeros((4, 256, 256), "u2"))
        runs.append(src)
    stop, peaks = threading.Event(), []
    watcher = threading.Thread(target=_claims_seen, args=(ledger, stop, peaks), daemon=True)
    watcher.start()
    env = {**os.environ, "MICRO_READER_SKIP_EUBI_CHECK": "1"}
    procs = [subprocess.Popen([eubi, "to_zarr", str(src), str(tmp_path / f"out{k}"),
                               "--memory_budget", "3GB", "--max_processes", "2",
                               "--log_file", "False"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                              text=True, errors="replace")
             for k, src in enumerate(runs)]
    outputs = [p.communicate(timeout=600)[0] for p in procs]
    stop.set()
    watcher.join(5)
    assert [p.returncode for p in procs] == [0, 0], "\n".join(o[-3000:] for o in outputs)
    assert max(peaks) <= 3 * 1024 + 1e-6
    for k in range(2):
        assert len(list((tmp_path / f"out{k}").glob("*.zarr"))) == 4
    assert any("other eubi run(s) here" in o for o in outputs)
