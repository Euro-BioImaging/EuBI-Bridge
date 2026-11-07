"""Each run's log on disk (Workplan E26, BARD report 2026-11-04): a crash that ends the
remote desktop took the log with it. to_zarr now writes it to a file in the config
folder's ``logs``, line by line, worker processes included."""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time

import numpy as np
import pytest

from eubi_bridge.utils import run_log

tifffile = pytest.importorskip("tifffile")


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    # a run's INFO lines reach the handlers as they do once setup_logging ran
    monkeypatch.setattr(logging.getLogger(), "level", logging.INFO)
    folder = tmp_path / "config"
    monkeypatch.setenv("EUBI_CONFIG_DIR", str(folder))
    monkeypatch.delenv(run_log.ENV, raising=False)
    return folder


def _logs(config_dir):
    return sorted((config_dir / "logs").glob("eubi-*.log"))


def test_a_run_writes_its_log_and_names_the_file(config_dir, caplog):
    log = logging.getLogger("eubi_bridge.test")
    with caplog.at_level(logging.INFO):
        with run_log.run_log() as path:
            assert os.environ[run_log.ENV] == str(path)
            log.info("inside the run")
    log.info("after the run")
    assert _logs(config_dir) == [path]
    text = path.read_text(encoding="utf-8")
    assert f"This run's log is also written to {path}" in text
    assert "inside the run" in text and "after the run" not in text
    assert run_log.ENV not in os.environ
    assert not any(isinstance(h, run_log.RunLogHandler) for h in logging.getLogger().handlers)
    assert any(f"This run's log is also written to {path}" in r.getMessage()
               for r in caplog.records)


@pytest.mark.parametrize("off", [False, "false", "off"])
def test_log_file_false_writes_none(config_dir, off):
    with run_log.run_log(off) as path:
        logging.getLogger("eubi_bridge.test").info("x")
    assert path is None and not (config_dir / "logs").exists()


def test_a_path_is_used_as_given(config_dir, tmp_path):
    target = tmp_path / "mine" / "run.log"
    with run_log.run_log(str(target)) as path:
        logging.getLogger("eubi_bridge.test").info("here")
    assert path == target and "here" in target.read_text(encoding="utf-8")
    with run_log.run_log(str(tmp_path / "mine")) as path:     # a folder: a new file in it
        pass
    assert path.parent == tmp_path / "mine" and path.name.startswith("eubi-")


def test_the_newest_runs_are_kept(config_dir):
    folder = config_dir / "logs"
    folder.mkdir(parents=True)
    for i in range(30):
        (folder / f"eubi-20260101-0000{i:02d}-1.log").write_text("old")
    with run_log.run_log() as path:
        pass
    kept = _logs(config_dir)
    assert len(kept) == run_log.KEEP and path in kept
    assert kept[0].name == "eubi-20260101-000011-1.log"      # the 11 oldest went


def test_an_error_leaving_the_run_is_written(config_dir):
    from eubi_bridge.utils.errors import InputError
    with pytest.raises(InputError):
        with run_log.run_log() as path:
            raise InputError("x.tif: no such file")
    assert "The run stopped: x.tif: no such file" in path.read_text(encoding="utf-8")


def test_setup_logging_in_a_worker_appends_to_the_run_log(config_dir, monkeypatch, tmp_path):
    from eubi_bridge.utils.logging_config import setup_logging
    target = tmp_path / "w.log"
    monkeypatch.setenv(run_log.ENV, str(target))
    root = logging.getLogger()
    saved = root.handlers[:]
    try:
        setup_logging()
        setup_logging()                                       # once, not twice
        assert sum(isinstance(h, run_log.RunLogHandler) for h in root.handlers) == 1
        logging.getLogger("eubi_bridge.worker").info("from a worker")
    finally:
        for h in root.handlers[:]:
            if isinstance(h, run_log.RunLogHandler):
                h.close()
        root.handlers[:] = saved
    assert "from a worker" in target.read_text(encoding="utf-8")


# --- processes --------------------------------------------------------------------------

def _worker_then_wait(path, ready, n):
    os.environ[run_log.ENV] = path
    from eubi_bridge.utils.logging_config import setup_logging
    setup_logging()
    log = logging.getLogger("eubi_bridge.worker")
    for i in range(n):
        log.info(f"line {i} of {os.getpid()}")
    ready.set()
    time.sleep(120)                                           # killed before this ends


def _worker_lines(path, n):
    os.environ[run_log.ENV] = path
    from eubi_bridge.utils.run_log import add_worker_handler
    add_worker_handler()
    logging.getLogger().setLevel(logging.INFO)
    log = logging.getLogger("eubi_bridge.worker")
    for i in range(n):
        log.info(f"line {i} of {os.getpid()} " + "x" * 200)


def test_a_worker_killed_mid_run_leaves_its_lines(tmp_path):
    path = str(tmp_path / "run.log")
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    p = ctx.Process(target=_worker_then_wait, args=(path, ready, 5))
    p.start()
    try:
        assert ready.wait(120)
        p.kill()                                              # as the OOM killer does
        p.join(30)
    finally:
        if p.is_alive():
            p.kill()
    text = open(path, encoding="utf-8").read()
    assert all(f"line {i} of {p.pid}" in text for i in range(5))


def test_processes_appending_at_once_lose_no_line(tmp_path):
    path = str(tmp_path / "run.log")
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_worker_lines, args=(path, 300)) for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
    lines = open(path, encoding="utf-8").read().splitlines()
    assert len(lines) == 4 * 300
    for p in procs:
        mine = [ln for ln in lines if f" of {p.pid} " in ln]
        assert len(mine) == 300 and all(ln.endswith("x" * 200) for ln in mine)


def test_to_zarr_writes_the_log_of_its_worker_processes(config_dir, tmp_path):
    from eubi_bridge.ebridge import EuBIBridge
    for name in ("a.tif", "b.tif"):
        tifffile.imwrite(tmp_path / name, np.zeros((8, 8), np.uint8))
    EuBIBridge().to_zarr(str(tmp_path), str(tmp_path / "out"), includes="*.tif",
                         max_processes=2, use_threading=False)
    (path,) = _logs(config_dir)
    text = path.read_text(encoding="utf-8")
    assert "Conversion starting." in text
    assert "SpawnProcess" in text or "ForkProcess" in text    # a worker's own line


def test_to_zarr_with_log_file_false(config_dir, tmp_path):
    from eubi_bridge.ebridge import EuBIBridge
    tifffile.imwrite(tmp_path / "a.tif", np.zeros((8, 8), np.uint8))
    EuBIBridge().to_zarr(str(tmp_path / "a.tif"), str(tmp_path / "out"), log_file=False,
                         use_threading=True)
    assert _logs(config_dir) == []
