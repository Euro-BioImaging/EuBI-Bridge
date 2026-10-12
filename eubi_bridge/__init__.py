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

def _installed_distribution():
    """(distribution name, version) of the installed package: eubi-bridge,
    or eubi-bridge-lite (the same package without Bio-Formats and dask)."""
    from importlib.metadata import PackageNotFoundError, version
    for name in ("eubi_bridge", "eubi-bridge-lite"):
        try:
            return name.replace("_", "-"), version(name)
        except PackageNotFoundError:
            continue
    return None, "0.0.0+dev"


#: "eubi-bridge" or "eubi-bridge-lite" (None when not installed, e.g. a source checkout)
__distribution__, __version__ = _installed_distribution()


def _warn_if_both_installed() -> None:
    """eubi-bridge and eubi-bridge-lite are the same package with different
    dependencies: they install the same files, so uninstalling either one
    deletes files the other needs.  The way out is to remove both."""
    from importlib.metadata import PackageNotFoundError, version
    installed = []
    for name in ("eubi_bridge", "eubi-bridge-lite"):
        try:
            version(name)
            installed.append(name)
        except PackageNotFoundError:
            pass
    if len(installed) == 2:
        import warnings
        warnings.warn(
            "eubi-bridge and eubi-bridge-lite are both installed, and removing just "
            "one would break the other. Run pip uninstall -y eubi-bridge "
            "eubi-bridge-lite, then install the one you want.", stacklevel=2)


_warn_if_both_installed()
