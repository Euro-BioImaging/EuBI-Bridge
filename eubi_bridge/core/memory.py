"""How a conversion's memory is shared out.

The user sets one number, ``memory_budget``: the memory the whole run may use on this
machine (``"auto"``: half the RAM that is free when the run starts). On SLURM, each
worker gets ``slurm_worker_memory`` instead (the SLURM job's memory). From that:

1. **per process**: the budget / the conversions running at once (``concurrent_jobs``),
   resolved ONCE in the main process, before the pool starts;
2. in each worker process:

   - what the process already holds (interpreter, libraries; measured once);
   - the reader's share: micro-reader's decoding (``decode_memory``, ``"auto"``:
     what the file's decode unit needs, see :func:`reader_share`), or the JVM heap
     (``jvm_memory``) for Bio-Formats pixels;
   - the rest, shared by the scenes converted at once: each scene's write budget,
     handed to the writer as the memory it may add (``memory_budget_mb``);

3. ``region_size_mb`` ``"auto"``: the scene's write budget over (3 x its threads),
   8..256 MB (a thread holds a region read, transformed and written: ~3 regions).

micro-reader's decode threads, ``decode_threads`` ``"auto"`` (:func:`decode_threads`):
the cores over the conversions running at once, at most 16, and no more than the
reader's share can hold blocks for.  An explicit ``decode_memory`` is capped at half
the process (:func:`reader_share`), so reading never starves the writer.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)

MB = float(1 << 20)
#: micro-reader keeps blocks up to this size decoded whole; larger ones are streamed
STREAM_MB = 16.0
#: what micro-reader's decoded-block caches may hold together, with decode_memory='auto'
CACHE_MB = 64.0
#: the least micro-reader gets (its own floor; a camera frame)
MIN_READER_MB = 64.0
#: region size bounds for region_size_mb='auto'
MIN_REGION_MB, MAX_REGION_MB = 8.0, 256.0
#: the least a scene's write gets; below this the budget is too small to be useful
MIN_SCENE_MB = 64.0
#: the most decode threads ``decode_threads="auto"`` gives one process (micro-reader's
#: own default)
MAX_DECODE_THREADS = 16
#: the most threads ``max_threads="auto"`` gives one conversion
MAX_WRITE_THREADS = 16
_warned: set = set()

_UNITS = {"": 1 / MB, "b": 1 / MB, "k": 1 / 1024, "kb": 1 / 1024, "kib": 1 / 1024,
          "m": 1.0, "mb": 1.0, "mib": 1.0, "g": 1024.0, "gb": 1024.0, "gib": 1024.0,
          "t": 1024.0 ** 2, "tb": 1024.0 ** 2, "tib": 1024.0 ** 2}


def parse_mb(value: Union[int, float, str]) -> float:
    """MiB from 512 (a number: MB), "512MB", "4GB", "4 GiB", "1.5g" (binary units)."""
    if isinstance(value, (int, float)):
        return float(value)
    match = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z]*)\s*", str(value))
    if not match or match.group(2).lower() not in _UNITS:
        raise ValueError(f"memory size {value!r}: a number of MB or e.g. '8GB', '512MB'")
    unit = match.group(2).lower()
    number = float(match.group(1))
    return number if unit == "" else number * _UNITS[unit]


def is_auto(value) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() == "auto")


def available_mb() -> Optional[float]:
    try:
        import psutil
        return psutil.virtual_memory().available / MB
    except Exception:                                           # noqa: BLE001
        return None


def total_mb(memory_budget) -> float:
    """The whole run's memory on this machine: an explicit size, or (``"auto"``) half
    the RAM free now. Warns when an explicit size is more than is free."""
    free = available_mb()
    if is_auto(memory_budget):
        return 0.5 * free if free else 4096.0
    mb = parse_mb(memory_budget)
    if free and mb > free:
        logger.warning(f"memory_budget={memory_budget} is more than the {free / 1024:.1f} GB "
                       f"of RAM free now: the run may swap or be killed.")
    return mb


def process_mb(cluster, running: Optional[int] = None, held_mb: float = 0.0) -> float:
    """One conversion process's memory: ``slurm_worker_memory`` on SLURM, else the
    run's budget, less what the main process holds when its conversions run in worker
    processes (``held_mb``: the budget is the whole run's), over the conversions running
    at once."""
    if getattr(cluster, "on_slurm", False):
        return parse_mb(cluster.slurm_worker_memory)
    running = max(1, int(running or getattr(cluster, "concurrent_jobs", None)
                         or cluster.max_processes or 1))
    total = total_mb(cluster.memory_budget)
    return max(total / (4 * running), (total - max(0.0, held_mb)) / running)


def main_process_mb() -> float:
    """What this (the main, dispatching) process holds now, in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / MB
    except Exception:                                           # noqa: BLE001
        return 0.0


_baseline_lock = threading.Lock()
_baseline: Optional[float] = None


def baseline_mb() -> float:
    """What this process held when its first conversion started (interpreter,
    libraries): measured once, so concurrent scenes do not count each other."""
    global _baseline
    with _baseline_lock:
        if _baseline is None:
            try:
                import psutil
                _baseline = psutil.Process().memory_info().rss / MB
            except Exception:                                   # noqa: BLE001
                _baseline = 0.0
        return _baseline


def available_cores() -> int:
    """The cores this process may use: a SLURM task's CPUs, else its CPU affinity
    (cgroups and ``taskset`` included), else every core."""
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit() and int(slurm) > 0:
        return int(slurm)
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def decode_threads(setting, running: int = 1, process: Optional[float] = None,
                   unit_mb: Optional[float] = None) -> int:
    """micro-reader's decode threads for one conversion process.

    A number is used as given.  ``"auto"``: the cores over the conversions running at
    once (*running*; 1 when they share one process and so one decode pool), at most
    ``MAX_DECODE_THREADS`` -- every core busy, none oversubscribed.  Once the file's
    decode unit is known (*unit_mb*, with the process's memory *process*), no more
    threads than the reader's quarter of the process can hold a block for each (a
    block up to 16 MB decodes whole, a larger one streams): fewer threads rather than
    memory taken from the writer.  At least 1."""
    if not is_auto(setting):
        return max(1, int(setting))
    n = max(1, min(MAX_DECODE_THREADS, available_cores() // max(1, int(running or 1))))
    if process and unit_mb:
        per_thread = min(max(float(unit_mb), 1.0), STREAM_MB)
        fit = int((float(process) / 4 - CACHE_MB) // per_thread)
        n = max(1, min(n, fit))
    return n


def write_threads(setting, running: int = 1) -> int:
    """One conversion's threads, each reading and writing one region at a time.

    A number is used as given.  ``"auto"``: the cores over the conversions running at
    once (*running*), at most ``MAX_WRITE_THREADS``.  At least 1."""
    if not is_auto(setting):
        return max(1, int(setting))
    return max(1, min(MAX_WRITE_THREADS, available_cores() // max(1, int(running or 1))))


def _warn_once(key, text: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(text)


def reader_share(process: float, decode_memory, decode_threads: int,
                 unit_mb: Optional[float]) -> Tuple[float, float]:
    """micro-reader's ``(decode limit, cache limit)`` in MB; their sum is its share.

    ``"auto"``: blocks up to 16 MB are decoded whole (``decode_threads`` at once),
    larger ones are streamed by rows, but a block whose codec cannot stream needs all of
    it: the decode limit is ``max(threads x min(unit, 16 MB), unit, 64 MB)`` (64 MB is
    micro-reader's own floor), so no block of the file is ever refused. The caches get
    64 MB (what one file had before they were shared). When that is more than a quarter
    of the process, both shrink to a quarter, unless one decode unit alone needs more:
    then it gets what it needs, with a warning (the write gets less). An explicit size:
    at most half the process (one warning), but never less than one decode unit and
    the caches; the caches get min(64 MB, a quarter of it), the rest decodes.
    """
    if not is_auto(decode_memory):
        total = max(2 * MIN_READER_MB, parse_mb(decode_memory))
        cap = max(2 * MIN_READER_MB, float(process) / 2)
        if total > cap:
            _warn_once(("decode_memory", decode_memory, round(process)),
                       f"decode_memory={decode_memory!r} is more than half the "
                       f"{process:.0f} MB this conversion process has: micro-reader gets "
                       f"{cap:.0f} MB, so writing keeps the rest (raise memory_budget, or "
                       f"lower max_processes, to give reading more).")
            total = cap
        cache = min(CACHE_MB, total / 4)
        # never too little for one block of the file (it would be refused)
        return max(total - cache, float(unit_mb or 0)), cache
    unit = STREAM_MB if unit_mb is None else max(float(unit_mb), 0.0)
    decode = max(max(1, int(decode_threads)) * min(unit, STREAM_MB), unit, MIN_READER_MB)
    cache = CACHE_MB
    cap = process / 4
    if decode + cache > cap:
        if unit > cap:                     # one block alone needs more: never refuse it
            logger.warning(
                f"decode_memory='auto': this file decodes blocks of {unit:.0f} MB that "
                f"cannot all be streamed; micro-reader gets {decode + cache:.0f} MB, more "
                f"than a quarter of the {process:.0f} MB per process, and the write less.")
        else:
            cache = min(CACHE_MB, cap / 4)
            decode = max(cap - cache, unit, MIN_READER_MB)
    return decode, cache


def scene_mb(process: float, baseline: float, reader: float, scenes: int) -> float:
    """Each scene's write budget: what is left of the process, shared by the scenes
    converted at once."""
    left = (process - baseline - reader) / max(1, int(scenes or 1))
    if left < MIN_SCENE_MB:
        logger.warning(
            f"The memory left for writing is {left:.0f} MB per scene ({process:.0f} MB per "
            f"process, {baseline:.0f} MB already held, {reader:.0f} MB for reading): raise "
            f"memory_budget, or lower max_processes / max_concurrent_scenes. Writing with "
            f"{MIN_SCENE_MB:.0f} MB.")
        return MIN_SCENE_MB
    return left


def region_mb(region_size_mb, scene: float, threads: int) -> float:
    """``region_size_mb``: explicit, or (``"auto"``) the scene's write budget over
    3 x its threads, within 8..256 MB."""
    if not is_auto(region_size_mb):
        return float(parse_mb(region_size_mb))
    return min(MAX_REGION_MB, max(MIN_REGION_MB, scene / (3 * max(1, int(threads)))))
