"""EuBI-Bridge: Distributed OME-Zarr image conversion toolkit."""


def _load_the_system_cpp_runtime():
    """Windows: load the system's C++ runtime (msvcp140.dll) before Qt can.

    PyQt6 bundles an old msvcp140.dll (14.26).  Windows reuses a DLL already
    loaded under the same name, so once Qt has loaded its copy, extensions
    built with Visual Studio 17.10 or later get the old one too, and their
    std::mutex crashes the process with an access violation (aicspylibczi
    opening any CZI did, on the Windows CI runners).  A newer runtime serves
    Qt's older code fine; only the reverse breaks.  No effect if a copy is
    already loaded.
    """
    import sys
    if sys.platform != "win32":
        return
    import ctypes
    import os
    path = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "msvcp140.dll")
    try:
        ctypes.WinDLL(path)
    except OSError:
        pass


_load_the_system_cpp_runtime()

try:
    from importlib.metadata import version as _metadata_version
    __version__ = _metadata_version("eubi_bridge")
except Exception:
    __version__ = "0.0.0+dev"
