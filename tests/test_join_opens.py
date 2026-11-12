"""A file several threads need at once is opened once (core/micro_source._using).

The threads of a join read the same files in the same order, so when a file had been
closed they all missed it at once and each opened it, format detection and all, only to
close all but one copy: on 768 one-plane TIFFs each file was opened ~5 times with 4
threads and ~8 times with 8, and 8 threads ran slower than 4 (Workplan E30, 2026-11-12).
"""
import threading
import time

import pytest

from eubi_bridge.core import micro_source as ms


class _File:
    def __init__(self, path):
        self.path, self.closed = path, False

    def close(self):
        self.closed = True


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(ms, "_files", type(ms._files)())
    monkeypatch.setattr(ms, "_opening", {})
    monkeypatch.setattr(ms, "require_micro_reader", lambda: None)
    return monkeypatch


def _together(n, fn):
    start = threading.Barrier(n)
    results, errors = [None] * n, []

    def run(i):
        start.wait()
        try:
            results[i] = fn()
        except Exception as exc:                          # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not any(t.is_alive() for t in threads), "a thread hung"
    return results, errors


def test_eight_threads_open_a_file_once(fresh):
    opened = []

    def slow_open(path, **kw):
        opened.append(path)
        time.sleep(0.2)                       # the others arrive while it opens
        return _File(path)

    fresh.setattr(ms, "open_renamed", slow_open)

    def use():
        with ms._using("a.tif", "separate", False) as entry:
            return entry.file

    files, errors = _together(8, use)
    assert not errors
    assert opened == ["a.tif"]
    assert len({id(f) for f in files}) == 1 and not files[0].closed
    assert not ms._opening


def test_a_failed_open_lets_the_others_try_and_hangs_nobody(fresh):
    calls = []

    def failing_open(path, **kw):
        calls.append(path)
        time.sleep(0.05)
        raise OSError("disk gone")

    fresh.setattr(ms, "open_renamed", failing_open)

    def use():
        with ms._using("b.tif", "separate", False):
            pass

    _, errors = _together(4, use)
    assert len(errors) == 4 and all(isinstance(e, OSError) for e in errors)
    assert 1 <= len(calls) <= 4
    assert not ms._opening and not ms._files
