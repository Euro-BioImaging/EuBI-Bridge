"""Windows: Qt's old C++ runtime must not crash aicspylibczi.

PyQt6 bundles msvcp140.dll 14.26.  Once Qt has loaded it, Windows hands that
copy to every later module asking for msvcp140.dll, and aicspylibczi (built
with Visual Studio 17.10+) crashed with an access violation opening any CZI:
the Windows CI runners died in the first CZI test after the GUI tests.
Importing eubi_bridge loads the system's newer runtime first.

Each case runs in a fresh process: the crash kills the process, and the DLL
loaded first is per process.  Qt's copy is loaded by its full path, as it
wins on a python.org Python (a conda Python has its own newer copy next to
python.exe, found first, so plain ``import PyQt6`` would hide the problem).
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows DLL loading")
pytest.importorskip("aicspylibczi")
pytest.importorskip("PyQt6")


def _czi(tmp_path) -> str:
    from tests._czi_writer import raw_payload, write_czi
    path = str(tmp_path / "one.czi")
    write_czi(path, [dict(scene=0, plane={"C": 0}, x=0, y=0, height=8, width=8, dtype="u2",
                          compression=0, payload=raw_payload(np.ones((8, 8), "u2")))])
    return path


def _qt_runtime() -> str:
    import os

    import PyQt6
    path = os.path.join(os.path.dirname(PyQt6.__file__), "Qt6", "bin", "msvcp140.dll")
    if not os.path.exists(path):
        pytest.skip("this PyQt6 bundles no msvcp140.dll")
    return path


def _open_after_qt(path: str, eubi_first: bool) -> subprocess.CompletedProcess:
    code = ("import eubi_bridge\n" if eubi_first else "") + (
        "import ctypes\n"
        f"ctypes.WinDLL({_qt_runtime()!r})\n"
        "from aicspylibczi import CziFile\n"
        f"with open({path!r}, 'rb') as f:\n"
        "    print('dims', CziFile(f).dims)\n")
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=120)


def test_a_czi_opens_after_qt_once_eubi_bridge_is_imported(tmp_path):
    result = _open_after_qt(_czi(tmp_path), eubi_first=True)
    assert result.returncode == 0, (result.returncode, result.stderr[-2000:])
    assert "dims" in result.stdout


def test_without_it_qt_s_runtime_crashes_aicspylibczi(tmp_path):
    """The failure the fix prevents: the test above is meaningful only while
    this one crashes."""
    result = _open_after_qt(_czi(tmp_path), eubi_first=False)
    if result.returncode == 0:
        pytest.skip("Qt's old runtime no longer crashes aicspylibczi: the workaround "
                    "in eubi_bridge/__init__.py may be unneeded")
    assert result.returncode != 0
