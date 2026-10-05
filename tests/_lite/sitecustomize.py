"""Loaded by every Python process started with this folder on PYTHONPATH
(tests/test_lite_core.py): the modules named in ``EUBI_LITE_BLOCK``
(comma-separated top-level names) cannot be imported -- as if
eubi-bridge-lite were installed without them.  Worker processes inherit
the environment, so they are blocked too.

Each attempt is appended to the file ``EUBI_LITE_ATTEMPTS`` names, as
``module<TAB>file:function`` of the innermost eubi_bridge frame that made it,
so the tests can tell deliberate probes from a feature lost silently.
"""
import os
import sys
import traceback

_blocked = frozenset(m for m in os.environ.get("EUBI_LITE_BLOCK", "").split(",") if m)
_log = os.environ.get("EUBI_LITE_ATTEMPTS")


def _site() -> str:
    """The code whose import statement this is: ``file:function`` inside
    eubi_bridge, or ``third-party:file`` for a library's own optional import
    (e.g. pint probing for dask), which that library handles itself."""
    for frame in reversed(traceback.extract_stack()[:-2]):
        name = frame.filename.replace("\\", "/")
        if name.startswith("<frozen importlib") or name.endswith("/sitecustomize.py"):
            continue
        if "/eubi_bridge/" in name:
            return f"{name.split('/eubi_bridge/')[-1]}:{frame.name}"
        return f"third-party:{name.split('/site-packages/')[-1]}"
    return "?"


class _Blocked:
    def find_spec(self, name, path=None, target=None):
        top = name.partition(".")[0]
        if top in _blocked:
            if _log:
                with open(_log, "a", encoding="utf-8") as fh:
                    fh.write(f"{name}\t{_site()}\n")
            raise ModuleNotFoundError(f"No module named {name!r} (blocked: not in eubi-bridge-lite)",
                                      name=name)
        return None


if _blocked:
    sys.meta_path.insert(0, _Blocked())
