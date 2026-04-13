"""
Subprocess entry point for conversion — no Qt imports.

Runs in a spawned child process.  All conversion logic lives here so that
cancel() can reliably kill the whole process tree without needing cooperation
from the asyncio event loop or from concurrent.futures machinery inside
eubi_bridge.conversion.converter.
"""
from __future__ import annotations

import re
import sys

_SEP      = "\x01"
# Matches CSI sequences, OSC hyperlinks (ESC ] ... BEL|ST), and other Fe escapes.
_ANSI_ESC = re.compile(
    r'\x1B(?:'
    r'\][^\x07\x1B]*(?:\x07|\x1B\\)'   # OSC: ESC ] ... BEL or ESC \
    r'|\[[0-9;:<=>?]*[ -/]*[@-~]'       # CSI: ESC [ params final
    r'|[@-Z\\-_]'                        # Fe:  ESC + single char
    r')'
)


# ── Process-tree kill ─────────────────────────────────────────────────────────

def _kill_process_tree(pid: int) -> None:
    """Kill *pid* and all its descendants (cross-platform)."""
    if sys.platform == "win32":
        import subprocess
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,
        )
    else:
        import os, signal
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass


# ── Subprocess entry point ────────────────────────────────────────────────────

def _conversion_subprocess(call_args: dict, log_queue, result_queue) -> None:
    """
    Runs inside a spawned child process.

    *call_args* keys
    ----------------
    input_path, output_path, includes, excludes,
    time_tag, channel_tag, z_tag, y_tag, x_tag, concatenation_axes,
    to_zarr_kwargs  (dict of keyword args for bridge.to_zarr)

    Log messages are pushed to *log_queue* as structured strings.
    ("ok", None) or ("err", traceback_str) is pushed to *result_queue*.
    """
    import logging as _log
    import traceback as _tb

    # ── Redirect stdout/stderr FIRST so that rich (imported below) creates its
    #    Console against our capture object, not the real file descriptor. ─────
    class _QCapture:
        def write(self, text: str):
            if text and text.strip():
                for line in text.strip().splitlines():
                    clean = _ANSI_ESC.sub("", line).strip()
                    if clean:
                        log_queue.put_nowait(clean)
        def flush(self):  pass
        def isatty(self): return False

    sys.stdout = _QCapture()
    sys.stderr = _QCapture()

    # ── Route all logging to log_queue ───────────────────────────────────────
    class _QHandler(_log.Handler):
        def emit(self, record):
            try:
                import time as _t
                ts  = _t.strftime("%H:%M:%S", _t.localtime(record.created))
                src = f"{record.filename}:{record.lineno}"
                msg = _ANSI_ESC.sub("", record.getMessage())
                log_queue.put_nowait(f"{ts}{_SEP}{record.levelname}{_SEP}{src}{_SEP}{msg}")
            except Exception:
                pass

    root = _log.getLogger()
    root.handlers.clear()
    root.addHandler(_QHandler())
    root.setLevel(_log.INFO)

    # ── Import eubi_bridge after capture is in place ──────────────────────────
    import eubi_bridge.conversion.converter as _conv_module
    from eubi_bridge.mp_logging_setup import setup_mp_logging_with_worker_init

    # ── Patch ProcessPoolExecutor so worker sub-processes log to the queue ────
    _orig_ppe = _conv_module.ProcessPoolExecutor
    _lq       = log_queue  # captured in closure

    class _LoggingPPE(_orig_ppe):
        def __init__(self, max_workers=None, **kw):
            raw = kw.get("initargs", (1,))[0] if "initargs" in kw else 1
            tsc = raw if isinstance(raw, int) else 1
            kw["initializer"] = setup_mp_logging_with_worker_init
            kw["initargs"]    = (_lq, tsc)
            super().__init__(max_workers, **kw)

    _conv_module.ProcessPoolExecutor = _LoggingPPE

    try:
        from eubi_bridge.utils.jvm_manager import soft_start_jvm
        soft_start_jvm()

        from eubi_bridge.ebridge import EuBIBridge
        bridge = EuBIBridge()
        bridge.to_zarr(
            input_path        = call_args["input_path"],
            output_path       = call_args["output_path"],
            includes          = call_args.get("includes"),
            excludes          = call_args.get("excludes"),
            time_tag          = call_args.get("time_tag"),
            channel_tag       = call_args.get("channel_tag"),
            z_tag             = call_args.get("z_tag"),
            y_tag             = call_args.get("y_tag"),
            x_tag             = call_args.get("x_tag"),
            concatenation_axes= call_args.get("concatenation_axes"),
            **call_args["to_zarr_kwargs"],
        )
        result_queue.put(("ok", None))
    except Exception:
        result_queue.put(("err", _tb.format_exc()))
    finally:
        _conv_module.ProcessPoolExecutor = _orig_ppe
