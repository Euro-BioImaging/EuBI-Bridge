"""A conversion run always ends (eubi-bridge 0.1.5).

In 0.1.4 every conversion that started Java inside a worker process never ended: JPype
destroyed the JVM at exit, which waits for every non-daemon Java thread, and Bio-Formats
can leave one running - so the worker never exited, and the dispatcher's
``shutdown(wait=True)`` waited for it forever, after the conversion had finished (or
reported its error). Two fixes: the JVM is not destroyed at exit, and the dispatcher stops
a worker that has not exited a while after all work is done.
"""
import asyncio
import multiprocessing as mp
import sys
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def test_the_jvm_is_not_destroyed_at_exit(monkeypatch):
    jpype = pytest.importorskip("jpype")
    scyjava = pytest.importorskip("scyjava")
    from eubi_bridge.utils import jvm_manager
    jars = list((Path(jvm_manager.__file__).resolve().parents[1] / "bioformats").glob("*.jar"))
    if not jars:
        pytest.skip("no bundled Bio-Formats JARs (eubi-bridge-lite)")
    started = []
    monkeypatch.setattr(scyjava, "jvm_started", lambda: False)
    monkeypatch.setattr(jpype, "startJVM", lambda *a, **k: started.append(True))
    monkeypatch.setattr(jvm_manager, "find_libjvm", lambda: "libjvm")
    monkeypatch.setattr(jpype.config, "destroy_jvm", True)
    jvm_manager._start_jvm_now()
    assert started
    assert jpype.config.destroy_jvm is False


def test_a_worker_that_never_exits_does_not_hang_the_run(tmp_path, monkeypatch):
    # a worker that, like one holding a Java thread, finishes its job but never exits
    (tmp_path / "stuck_worker.py").write_text(textwrap.dedent('''
        import threading, time

        def work(job):
            threading.Thread(target=time.sleep, args=(600,)).start()   # non-daemon
            return {"status": "ok"}
    '''))
    monkeypatch.syspath_prepend(str(tmp_path))
    import stuck_worker
    from eubi_bridge.conversion import dispatcher
    monkeypatch.setattr(dispatcher, "_WORKER_EXIT_S", 2.0)
    job = SimpleNamespace(cluster=SimpleNamespace(tensorstore_data_copy_concurrency=1))
    t0 = time.monotonic()
    results = asyncio.run(dispatcher._dispatch(
        [job], stuck_worker.work, ProcessPoolExecutor,
        {"max_workers": 1, "mp_context": mp.get_context("spawn")}, False))
    assert results == [{"status": "ok"}]
    assert time.monotonic() - t0 < 60
