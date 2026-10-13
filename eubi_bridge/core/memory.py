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
"""
from __future__ import annotations

import logging
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
    the caches get min(64 MB, a quarter of it), the rest decodes.
    """
    if not is_auto(decode_memory):
        total = max(2 * MIN_READER_MB, parse_mb(decode_memory))
        cache = min(CACHE_MB, total / 4)
        return total - cache, cache
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
