"""One memory budget shared by the eubi runs on one machine (or in one SLURM job).

Each run planned with the whole ``memory_budget`` and knew nothing of the others, so
runs started side by side multiplied it: four batches on a BARD desktop, each with
``memory_budget=120GB`` (or ``auto``) on a 128 GB job, planned 480 GB and the kernel
ended the job (report of 2026-11-04, Workplan E25). Now the runs keep a small ledger:

- **runs**: every ``to_zarr`` run registers (:func:`run`) with the budget it plans
  with (:func:`run_total_mb`). All live runs share ONE budget: the smallest of theirs.
  A run whose ``memory_budget`` is ``"auto"`` that starts beside others takes the
  budget already in use (its own "half of what is free" would count the others' use
  as gone, and shrink the budget for all);
- **claims**: each conversion claims its memory when it starts (:func:`task_share`)
  and gives it back when it ends. It gets its run's fair part (its planned memory over
  the live runs), never more than the budget less what the other conversions hold. A
  conversion that would get less than half its fair part waits, with one line, until
  others end.
- **handing on** (:func:`regrow`): a run records how many conversions it has and how
  many have ended (:func:`plan_tasks`). Once fewer are left than run at once, the ones
  still busy share what the finished ones held: each may grow its claim to the run's
  memory over the conversions left (over the live runs, as above), never past the
  budget less the other claims, and never shrinking. The writer asks again before each
  pass and every few seconds during one (dyna-zarr's ``io.memory_share``).

The ledger is a folder of small JSON files under the temp folder, one per user and per
scope (the SLURM job, else the machine), written under a file lock; a run or claim
whose process has ended no longer counts. ``EUBI_MEMORY_LEDGER`` moves it (the tests'
own). Worker processes find the run through :data:`ENV`, set by the main process.
SLURM-submitted conversions (``on_slurm``) have their own worker memory and take no part.
"""
from __future__ import annotations

import contextlib
import getpass
import json
import logging
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Iterator, List, Optional

logger = logging.getLogger("eubi_bridge.memory_share")

#: the environment variable naming this run's file in the ledger (workers inherit it)
ENV = "EUBI_MEMORY_RUN"
#: the environment variable that moves the ledger (tests)
LEDGER_ENV = "EUBI_MEMORY_LEDGER"
#: how often a waiting conversion looks again, in seconds
POLL_S = 2.0
#: a conversion starts once it can get this part of its fair share
START_FRACTION = 0.5

_counter_lock = threading.Lock()
_counter = 0
_said_waiting: set = set()


# --- where ------------------------------------------------------------------------------

def _scope() -> str:
    job = os.environ.get("SLURM_JOB_ID", "")
    return f"slurm-{job}" if job.isalnum() else "machine"


def ledger_dir() -> Path:
    """This user's ledger for this scope (the SLURM job, else the machine)."""
    moved = os.environ.get(LEDGER_ENV)
    if moved:
        return Path(moved) / _scope()
    try:
        user = getpass.getuser()
    except Exception:                                           # noqa: BLE001
        user = str(getattr(os, "getuid", lambda: "user")())
    user = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in user) or "user"
    return Path(tempfile.gettempdir()) / f"eubi-bridge-memory-{user}" / _scope()


# --- processes --------------------------------------------------------------------------

def _start_time(pid: int) -> Optional[float]:
    try:
        import psutil
        return round(psutil.Process(pid).create_time(), 2)
    except Exception:                                           # noqa: BLE001
        return None


def _alive(entry: dict) -> bool:
    """The process that wrote *entry* still runs (and is the same process: a pid can be
    reused)."""
    pid = entry.get("pid")
    if not isinstance(pid, int):
        return False
    try:
        import psutil
        if not psutil.pid_exists(pid):
            return False
    except ImportError:                                         # pragma: no cover
        if sys.platform == "win32":
            return True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    started = entry.get("started_at")
    now = _start_time(pid)
    return started is None or now is None or abs(now - started) < 1.0


# --- the ledger -------------------------------------------------------------------------

#: threads of one process take this first: on Windows a byte-range lock that another
#: handle holds (another thread's, too) is retried only once a second
_process_lock = threading.RLock()


@contextlib.contextmanager
def _locked(folder: Path):
    """The ledger's lock, across processes (and threads)."""
    with _process_lock:
        with _file_locked(folder):
            yield


@contextlib.contextmanager
def _file_locked(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(folder / ".lock"), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if sys.platform == "win32":
            import msvcrt
            for _ in range(60):                    # LK_LOCK itself tries for 10 s
                try:
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    continue
            try:
                yield
            finally:
                try:
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _read_all(folder: Path, prefix: str) -> List[dict]:
    """The live entries of one kind; those of ended processes are removed."""
    found = []
    for path in folder.glob(f"{prefix}-*.json"):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue                               # being written: the lock prevents it
        if _alive(entry):
            entry["_path"] = str(path)
            found.append(entry)
        else:
            with contextlib.suppress(OSError):
                path.unlink()
    return found


def _write(path: Path, entry: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(entry), encoding="utf-8")
    os.replace(tmp, path)


def _own_run() -> Optional[Path]:
    path = os.environ.get(ENV)
    return Path(path) if path else None


def live_runs() -> List[dict]:
    folder = ledger_dir()
    if not folder.exists():
        return []
    with _locked(folder):
        return _read_all(folder, "run")


def pool_mb(runs: List[dict]) -> Optional[float]:
    """The budget the live runs share: the smallest of theirs."""
    budgets = [r["budget_mb"] for r in runs if isinstance(r.get("budget_mb"), (int, float))]
    return min(budgets) if budgets else None


# --- runs -------------------------------------------------------------------------------

@contextlib.contextmanager
def run() -> Iterator[Optional[Path]]:
    """Register this run in the ledger while the block runs (``to_zarr``). A run inside
    a run of this process (``ENV`` set by it) is that run."""
    own = _own_run()
    if own is not None:
        yield own
        return
    pid = os.getpid()
    try:
        folder = ledger_dir()
        with _locked(folder):
            _read_all(folder, "run")               # drops ended runs
            path = folder / f"run-{pid}-{time.time_ns()}.json"
            _write(path, {"pid": pid, "started_at": _start_time(pid), "budget_mb": None,
                          "since": time.time()})
    except OSError as exc:
        logger.warning(f"The memory budget cannot be shared with other eubi runs here "
                       f"({exc}): this run plans with all of it.")
        yield None
        return
    os.environ[ENV] = str(path)
    try:
        yield path
    finally:
        os.environ.pop(ENV, None)
        with contextlib.suppress(OSError):
            path.unlink()


def run_total_mb(memory_budget, own_total) -> float:
    """The budget this run plans with: its own (*own_total*, a function giving
    :func:`eubi_bridge.core.memory.total_mb`), or, beside other live runs, the budget
    they share. Recorded in the ledger the first time, so the others share it too."""
    path = _own_run()
    if path is None:
        return own_total(memory_budget)
    from eubi_bridge.core.memory import is_auto
    folder = path.parent
    try:
        with _locked(folder):
            runs = _read_all(folder, "run")
            mine = next((r for r in runs if r["_path"] == str(path)), None)
            others = [r for r in runs if r["_path"] != str(path)]
            if mine is not None and isinstance(mine.get("budget_mb"), (int, float)):
                return pool_mb(runs)
            shared = pool_mb(others)
            if is_auto(memory_budget) and shared is not None:
                budget = shared                    # 'auto' beside others: the shared one
                logger.info(f"memory_budget='auto': {shared / 1024:.1f} GB, the budget "
                            f"shared with {len(others)} other eubi run(s) here.")
            else:
                budget = own_total(memory_budget)
            if mine is not None:
                mine = {k: v for k, v in mine.items() if k != "_path"}
                _write(path, {**mine, "budget_mb": budget})
            total = min(budget, shared) if shared is not None else budget
            if others:
                logger.info(
                    f"Memory: {len(others)} other eubi run(s) here; the runs share "
                    f"{total / 1024:.1f} GB (the smallest of their budgets), each "
                    f"conversion taking its part when it starts.")
            return total
    except OSError:
        return own_total(memory_budget)


# --- claims -----------------------------------------------------------------------------

def _claim_path(folder: Path) -> Path:
    global _counter
    with _counter_lock:
        _counter += 1
        n = _counter
    return folder / f"claim-{os.getpid()}-{threading.get_ident()}-{n}.json"


def _grant(folder: Path, wanted: float, run_path: Optional[Path] = None):
    """``(granted MB or None to wait, fair, pool, other runs)`` and the claim written."""
    runs = _read_all(folder, "run")
    pool = pool_mb(runs)
    if pool is None:                               # no budget known yet: plan as before
        return wanted, wanted, pool, max(0, len(runs) - 1), None
    if len(runs) <= 1:
        # alone: all of the plan, but claimed, so a run that starts meanwhile sees it
        granted = fair = wanted
    else:
        claims = _read_all(folder, "claim")
        held = sum(float(c.get("mb", 0)) for c in claims)
        fair = wanted / len(runs)
        granted = min(fair, pool - held)
        if granted < START_FRACTION * fair:
            return None, fair, pool, len(runs) - 1, None
    path = _claim_path(folder)
    pid = os.getpid()
    _write(path, {"pid": pid, "started_at": _start_time(pid), "mb": granted,
                  "run": str(run_path) if run_path is not None else None})
    return granted, fair, pool, len(runs) - 1, path


class Claim:
    """One conversion's claim: the MB it plans with (``mb``) and its ledger file
    (``path``, None when nothing was claimed)."""

    __slots__ = ("mb", "path")

    def __init__(self, mb: Optional[float], path: Optional[Path]):
        self.mb, self.path = mb, path


@contextlib.contextmanager
def task_share(wanted_mb: Optional[float], on_slurm: bool = False,
               label: str = "") -> Iterator[Optional[float]]:
    """One conversion's memory, claimed from the shared budget while the block runs:
    *wanted_mb* (its run's plan) when the run is alone, else its fair part, waiting
    while the other conversions hold too much. Yields the MB to plan with (None: plan
    as before)."""
    with task_claim(wanted_mb, on_slurm, label) as claim:
        yield claim.mb


@contextlib.contextmanager
def task_claim(wanted_mb: Optional[float], on_slurm: bool = False,
               label: str = "") -> Iterator[Claim]:
    """:func:`task_share`, yielding the :class:`Claim` (its ``path`` is what
    :func:`regrow` takes). When the block ends the claim is given back and the
    conversion counts as ended for its run (:func:`plan_tasks`)."""
    own = _own_run()
    if own is None or on_slurm:
        yield Claim(wanted_mb, None)
        return
    if not wanted_mb:
        try:
            yield Claim(wanted_mb, None)
        finally:
            _task_ended(own)
        return
    folder = own.parent
    claim = None
    waited = 0.0
    try:
        while True:
            try:
                with _locked(folder):
                    granted, fair, pool, others, claim = _grant(folder, float(wanted_mb),
                                                                own)
            except OSError:
                granted, claim = float(wanted_mb), None
            if granted is not None:
                break
            key = (os.getpid(), label)
            if key not in _said_waiting:
                _said_waiting.add(key)
                logger.info(f"{label or 'A conversion'}: waiting for memory; other eubi "
                            f"conversions here hold the shared {pool / 1024:.1f} GB "
                            f"({others} other run(s)).")
            time.sleep(POLL_S)
            waited += POLL_S
        if granted < float(wanted_mb):
            logger.info(f"{label or 'This conversion'}: {granted:.0f} MB of the shared "
                        f"{pool / 1024:.1f} GB ({others} other eubi run(s) here)"
                        + (f", after waiting {waited:.0f} s" if waited else "") + ".")
        yield Claim(granted, claim)
    finally:
        if claim is not None:
            with contextlib.suppress(OSError):
                claim.unlink()
        _task_ended(own)


# --- handing on what finished conversions held ------------------------------------------

def _update_run(path: Path, change) -> Optional[dict]:
    """Rewrite this run's ledger entry with ``change(entry)`` (under the lock held by
    the caller); None when the run is gone."""
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    entry = change(entry)
    _write(path, entry)
    return entry


def plan_tasks(count: int, at_once: int, spread_mb: float) -> None:
    """Record, in this run's ledger entry, *count* more conversions of which up to
    *at_once* run at once, sharing *spread_mb* (the run's memory for its conversions:
    each one's plan x *at_once*). Called by the dispatcher before the pool starts."""
    path = _own_run()
    if path is None or count < 1 or not spread_mb:
        return

    def change(entry):
        return {**entry, "tasks": int(entry.get("tasks") or 0) + int(count),
                "at_once": max(int(entry.get("at_once") or 0), int(at_once)),
                "spread_mb": float(spread_mb)}
    try:
        with _locked(path.parent):
            _update_run(path, change)
    except OSError:
        pass


def _task_ended(own: Path) -> None:
    try:
        with _locked(own.parent):
            _update_run(own, lambda e: {**e, "ended": int(e.get("ended") or 0) + 1})
    except OSError:
        pass


def regrow(claim_path) -> Optional[tuple]:
    """``(MB, conversions still busy)`` for the conversion holding *claim_path*: its
    claim, grown (and rewritten) when finished conversions of its run left memory free;
    None when its run records no conversions (:func:`plan_tasks`) or the ledger is gone.

    The run's memory (``spread_mb``) over the conversions not yet ended, at most as many
    as run at once, and over the live runs; never more than the budget less the other
    claims (all runs); never less than the claim already holds."""
    if not claim_path:
        return None
    path = Path(claim_path)
    try:
        with _locked(path.parent):
            try:
                mine = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
            run_path = mine.get("run")
            runs = _read_all(path.parent, "run")
            run = next((r for r in runs
                        if run_path and Path(r["_path"]).name == Path(run_path).name), None)
            if run is None:
                return None
            tasks, spread = int(run.get("tasks") or 0), run.get("spread_mb")
            if tasks < 1 or not isinstance(spread, (int, float)):
                return None
            left = max(1, tasks - int(run.get("ended") or 0))
            busy = max(1, min(int(run.get("at_once") or left), left))
            held = float(mine.get("mb") or 0.0)
            fair = float(spread) / busy / max(1, len(runs))
            others = sum(float(c.get("mb", 0)) for c in _read_all(path.parent, "claim")
                         if Path(c["_path"]).name != path.name)
            # alone: the run's own memory for conversions; beside others: the shared
            # budget (as when the claims were made)
            limit = float(spread) if len(runs) <= 1 else (pool_mb(runs) or float(spread))
            grown = max(held, min(fair, limit - others))
            if grown > held + 1.0:
                _write(path, {**{k: v for k, v in mine.items() if k != "_path"},
                              "mb": grown})
                return grown, busy
            return held, busy
    except OSError:
        return None
