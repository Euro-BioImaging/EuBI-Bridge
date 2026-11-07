"""Each run's log on disk (Workplan E26).

The log lived only in the terminal or the GUI's Convert panel, so a crash that took
the desktop with it (a BARD remote desktop ended by the out-of-memory killer, report
of 2026-11-04) left nothing to read. Each ``to_zarr`` run now also writes its log to a
file, one line at a time straight to the disk, so the lines before a kill survive:

- where: ``<config folder>/logs/eubi-<start time>-<process id>.log`` (the config
  folder: ``EUBI_CONFIG_DIR``, else ``~/.eubi_bridge``); the newest :data:`KEEP` are
  kept. ``log_file=<path>`` writes there instead, ``log_file=False`` writes none;
- the run's first line names the file;
- worker processes write to the same file: the main process puts its path in
  :data:`ENV`, and :func:`eubi_bridge.utils.logging_config.setup_logging`, which every
  worker runs, adds the handler.
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Iterator, Optional, Union

logger = logging.getLogger("eubi_bridge.run_log")

#: the environment variable that hands the run's log file to worker processes
ENV = "EUBI_RUN_LOG"
#: how many run logs the logs folder keeps
KEEP = 20
_PREFIX = "eubi-"
#: the region locked around a write on Windows: far past any log's end, so a reader
#: of the file (an editor) is never refused
_LOCK_AT = 1 << 40


def config_folder(configpath: Optional[str] = None) -> Path:
    """The config folder, as :class:`eubi_bridge.ebridge.EuBIBridge` finds it."""
    return Path(configpath or os.environ.get("EUBI_CONFIG_DIR")
                or Path.home() / ".eubi_bridge")


def log_folder(configpath: Optional[str] = None) -> Path:
    """Where the run logs are kept."""
    return config_folder(configpath) / "logs"


class RunLogHandler(logging.Handler):
    """Appends each record to *path* as one line, written straight to the disk (no
    buffer to lose). Several processes may append to one file: on POSIX an append of
    one line is one ``write``; on Windows, where appends of two processes can overwrite
    each other, each write is done under a byte-range lock."""

    def __init__(self, path: Union[str, Path]):
        super().__init__()
        self.path = str(path)
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0)
        self._fd = os.open(self.path, flags, 0o644)
        self._write_lock = threading.Lock()
        self.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(processName)s %(filename)s:%(lineno)d "
            "%(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = self.format(record).replace("\r\n", "\n")
            data = (text.rstrip("\n") + "\n").encode("utf-8", errors="replace")
            with self._write_lock:
                if self._fd is not None:
                    _append(self._fd, data)
        except Exception:                                       # noqa: BLE001
            self.handleError(record)

    def close(self) -> None:
        with self._write_lock:
            if self._fd is not None:
                try:
                    os.close(self._fd)
                except OSError:
                    pass
                self._fd = None
        super().close()


def _append(fd: int, data: bytes) -> None:
    if sys.platform != "win32":
        os.write(fd, data)
        return
    import msvcrt
    locked = False
    try:
        os.lseek(fd, _LOCK_AT, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)      # retries for up to 10 s
        locked = True
    except OSError:
        pass                                        # write anyway: a line beats none
    try:
        os.write(fd, data)                          # O_APPEND: at the end
    finally:
        if locked:
            try:
                os.lseek(fd, _LOCK_AT, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass


def add_worker_handler(root: Optional[logging.Logger] = None) -> None:
    """In a worker process: append to the run's log file the main process named in
    :data:`ENV`, if any (called by ``setup_logging``)."""
    path = os.environ.get(ENV)
    if not path:
        return
    root = root or logging.getLogger()
    if any(isinstance(h, RunLogHandler) and h.path == path for h in root.handlers):
        return
    try:
        root.addHandler(RunLogHandler(path))
    except OSError:
        pass                                        # the log file is a help, not a need


def _is_off(setting) -> bool:
    return setting is False or (isinstance(setting, str)
                                and setting.strip().lower() in ("false", "off", "no", "0", ""))


def _is_auto(setting) -> bool:
    return setting is None or setting is True or (
        isinstance(setting, str) and setting.strip().lower() in ("true", "auto", "on", "yes"))


def _prune(folder: Path, keep: int) -> None:
    """Remove the oldest run logs so that *keep* remain."""
    logs = sorted(folder.glob(f"{_PREFIX}*.log"))
    for old in logs[:max(0, len(logs) - keep)]:
        try:
            old.unlink()
        except OSError:
            pass                                    # another run still writes it


def new_log_path(configpath: Optional[str] = None) -> Path:
    """A new run log's path in the logs folder, the oldest pruned to make room."""
    folder = log_folder(configpath)
    folder.mkdir(parents=True, exist_ok=True)
    _prune(folder, KEEP - 1)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return folder / f"{_PREFIX}{stamp}-{os.getpid()}.log"


@contextlib.contextmanager
def run_log(setting=True, configpath: Optional[str] = None) -> Iterator[Optional[Path]]:
    """Write this run's log to a file while the block runs.

    *setting*: ``True`` / ``None`` / ``"auto"`` (a new file in the logs folder), a path
    (that file, appended to; a folder: a new file in it), or ``False`` (none). A run
    inside a run that already writes a log (``ENV`` set) writes to that one. An error
    leaving the block is written to the file too (the command line prints it only to
    the terminal)."""
    if _is_off(setting) or os.environ.get(ENV):
        yield None
        return
    try:
        if _is_auto(setting):
            path = new_log_path(configpath)
        else:
            path = Path(os.path.expanduser(str(setting)))
            if path.is_dir():
                path = path / f"{_PREFIX}{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}.log"
            path.parent.mkdir(parents=True, exist_ok=True)
        handler = RunLogHandler(path)
    except OSError as exc:
        logger.warning(f"This run's log cannot be written to a file ({exc}); it is shown "
                       f"here only.")
        yield None
        return
    root = logging.getLogger()
    root.addHandler(handler)
    os.environ[ENV] = str(path)
    logger.info(f"This run's log is also written to {path}")
    try:
        yield path
    except BaseException as exc:
        try:
            from eubi_bridge.utils.errors import InputError, TasksFailedError
            said = isinstance(exc, (InputError, TasksFailedError))
            handler.handle(logger.makeRecord(
                logger.name, logging.ERROR, __file__, 0,
                f"The run stopped: {exc}" if said
                else f"The run stopped: {type(exc).__name__}: {exc}",
                None, None if said else (type(exc), exc, exc.__traceback__)))
        except Exception:                                       # noqa: BLE001
            pass
        raise
    finally:
        os.environ.pop(ENV, None)
        # setup_logging run in this process meanwhile replaces the handler by its own
        # for the same file: remove every one
        for h in [handler] + [h for h in root.handlers
                              if isinstance(h, RunLogHandler) and h.path == str(path)]:
            root.removeHandler(h)
            h.close()
