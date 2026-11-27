"""The classic viewer's render thread ends with its viewer, never under it.

CI (2026-11-25, Lite on Python 3.14, ubuntu): the GUI tests crashed with a
segmentation fault while a RenderWorker waited for work. The worker was a Qt child
of its viewer and stopped only in the viewer's closeEvent, so a viewer freed
without being closed (a test's page, or the viewer inside a closed main window)
took a running QThread with it. Each case runs in a fresh Python: on the old code it
crashes that process, not the test run.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from tests.conftest import qt_available

if not qt_available():
    pytest.skip("PyQt6 unavailable or no usable Qt platform plugin", allow_module_level=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SETUP = """
import gc, os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6 import sip
from PyQt6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from eubi_bridge.qt_gui.widgets.zarr_viewer import ZarrViewer
from eubi_bridge.qt_gui.workers import render_worker
viewer = ZarrViewer()
worker = viewer._render_worker
assert worker.isRunning()
"""


def _run(code: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    return subprocess.run([sys.executable, "-X", "faulthandler", "-c",
                           textwrap.dedent(SETUP) + textwrap.dedent(code)],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)


def _ok(result):
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[:4000]
    assert "ended" in result.stdout, result.stdout


def test_a_viewer_deleted_while_its_thread_waits_stops_the_thread():
    _ok(_run("""
        sip.delete(viewer)                  # C++ object gone; no closeEvent
        assert worker.wait(3000), "the render thread is still running"
        print("ended")
    """))


def test_a_viewer_freed_by_python_without_closing_stops_the_thread():
    _ok(_run("""
        del viewer
        gc.collect()
        assert worker.wait(3000), "the render thread is still running"
        print("ended")
    """))


def test_python_exits_with_a_viewer_still_open():
    # the viewer is neither closed nor freed: the exit hook stops its thread
    _ok(_run("""
        print("ended")
    """))


def test_the_application_quitting_stops_every_render_thread():
    _ok(_run("""
        from PyQt6.QtCore import QTimer
        second = ZarrViewer()
        QTimer.singleShot(0, app.quit)
        app.exec()                          # aboutToQuit -> render_worker.stop_all
        assert worker.isFinished() and second._render_worker.isFinished()
        print("ended")
    """))


def test_closing_the_viewer_still_stops_its_thread():
    _ok(_run("""
        viewer.close()
        assert worker.wait(3000)
        print("ended")
    """))
