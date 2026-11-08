"""A long join's memory over time stays flat (Workplan E25, the BARD report of 2026-11-05).

The report: a join of 10288 one-plane files grew past its plan over hours and took the
machine down. Two checks over the whole run of a real join, sampled every 0.25 s:

- every system: the tifffile objects alive stay bounded and show no upward trend
  (before the fix they climbed through the write, one per file reopened: 6012 at the end
  of a 6000-file join, against at most about 1160 with it);
- Linux: the worker process's memory stays flat and within its plan (glibc kept the
  memory the garbage took; this is where the machine ran out).
"""
from __future__ import annotations

import os
import sys
import threading
import time
import weakref

import pytest

from eubi_bridge.core import micro_source as ms
from tests.test_join_memory import make_stub_set

#: the set: 4 channels x NZ planes of YX x YX, one plane per BinaryOnly stub TIFF. Sized
#: so the old behaviour shows (calibrated on Windows: 4000 files, before the fix 4128
#: tifffile objects alive at the end and climbing, with it at most 700; 2400 files were
#: too few to tell them apart safely)
NZ, YX = 1000, 128
#: the bound's parts as released (micro_source.JOIN_OPEN_FILES, COLLECT_AFTER_CLOSES),
#: written out: a change of those settings must not move the bound with it
JOIN_OPEN, AFTER_CLOSES = 1024, 256
TAGS = dict(includes="*.tif", concatenation_axes="cz", channel_tag="_405,_488,_561,_638",
            z_tag="plane")


@pytest.fixture(scope="module")
def stub_set(tmp_path_factory):
    folder = tmp_path_factory.mktemp("flat") / "set"
    make_stub_set(str(folder), nz=NZ, yx=YX)
    return str(folder)


class LiveTiffFiles:
    """Counts the tifffile.TiffFile objects alive, without gc.get_objects() (which,
    called while other threads build tuples, makes CPython's tuple resize fail)."""

    def __init__(self, monkeypatch):
        import tifffile
        self.n = 0
        original = tifffile.TiffFile.__init__
        watch = self

        def counted(tif, *args, **kwargs):
            original(tif, *args, **kwargs)
            watch.n += 1
            weakref.finalize(tif, watch._freed)

        monkeypatch.setattr(tifffile.TiffFile, "__init__", counted)

    def _freed(self):
        self.n -= 1


class Sampler:
    """Samples ``value()`` every *interval* seconds in a thread while the block runs."""

    def __init__(self, value, interval: float = 0.25):
        self.value, self.interval, self.samples = value, interval, []
        self._stop = threading.Event()

    def __enter__(self):
        def run():
            while not self._stop.is_set():
                self.samples.append(self.value())
                time.sleep(self.interval)
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def _thirds(values):
    third = max(1, len(values) // 3)
    return values[:third], values[third:-third] or values[third:], values[-third:]


def _median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def test_the_bound_follows_the_released_settings():
    """The bound below is written out; it stands for these settings."""
    assert (ms.JOIN_OPEN_FILES, ms.COLLECT_AFTER_CLOSES) == (JOIN_OPEN, AFTER_CLOSES)


def test_closed_files_do_not_pile_up_over_a_join(stub_set, tmp_path, monkeypatch):
    """Over the whole join, the tifffile objects alive stay within the open files plus
    the closes between two collections, and the last third of the run holds no more
    than the first."""
    from eubi_bridge.ebridge import EuBIBridge
    monkeypatch.setattr(ms, "COLLECT_EVERY_S", 0.25)        # collections due in a short run
    live = LiveTiffFiles(monkeypatch)
    with Sampler(lambda: live.n) as counts:
        EuBIBridge().to_zarr(stub_set, str(tmp_path / "out"), use_threading=True,
                             max_processes=1, max_threads=4, decode_threads=4,
                             memory_budget="1GB", overwrite=True, **TAGS)
    samples = [n for n in counts.samples if n > 0]
    assert len(samples) >= 12, f"too few samples ({len(samples)}): the join ran too short"
    bound = min(4 * NZ, JOIN_OPEN) + 2 * AFTER_CLOSES
    assert max(samples) <= bound, (f"{max(samples)} tifffile objects alive at once, more "
                                   f"than the open files and the closes between collections "
                                   f"({bound}): closed files pile up")
    first, _middle, last = _thirds(samples)
    assert _median(last) <= _median(first) + 2 * AFTER_CLOSES, (
        f"the tifffile objects alive climb over the join: median {_median(first)} in the "
        f"first third, {_median(last)} in the last")


#: the worker-memory test's set: planes big enough that the working set fills early and
#: a long steady write follows (4000 files of 0.5 MB, about 30 s in WSL)
RSS_NZ, RSS_YX, RSS_BUDGET_MB = 1000, 512, 512


@pytest.fixture(scope="module")
def large_stub_set(tmp_path_factory):
    folder = tmp_path_factory.mktemp("flat_rss") / "set"
    make_stub_set(str(folder), nz=RSS_NZ, yx=RSS_YX)
    return str(folder)


def _quarters(values):
    q = max(1, len(values) // 4)
    return values[2 * q:3 * q], values[3 * q:]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="glibc keeps the memory "
                    "garbage took on Linux; elsewhere the objects test above covers it")
def test_a_join_workers_memory_stays_flat(large_stub_set, tmp_path):
    """A join in a worker process, as a batch runs it, the worker's memory sampled every
    0.25 s: it stays within its plan, and does not climb at the end of the run.

    Calibrated in WSL (2026-11-05, this set, 512 MB): with the fix the worker levels off
    at 340-360 MB (peak 0.7 x the budget, the last quarter 5 MB above the third); before
    it, it climbed all the way, 320 -> 750 MB (peak 1.47 x, last quarter 70 MB above)."""
    import psutil
    from eubi_bridge.ebridge import EuBIBridge
    me = psutil.Process()

    def worker_rss():
        kids = me.children(recursive=True)
        return max((k.memory_info().rss for k in kids), default=0) / 2**20

    with Sampler(worker_rss) as rss:
        EuBIBridge().to_zarr(large_stub_set, str(tmp_path / "out"), max_processes=1,
                             max_threads=4, decode_threads=4,
                             memory_budget=f"{RSS_BUDGET_MB}MB", overwrite=True, **TAGS)
    samples = [r for r in rss.samples if r > 0]
    assert len(samples) >= 40, f"too few samples ({len(samples)}): the join ran too short"
    assert max(samples) <= 1.2 * RSS_BUDGET_MB, (
        f"the worker held {max(samples):.0f} MB at its peak, past its plan of "
        f"{RSS_BUDGET_MB} MB")
    third_q, last_q = _quarters(samples)
    rise = _median(last_q) - _median(third_q)
    assert rise <= 0.05 * _median(third_q) + 24, (
        f"the worker's memory climbs at the end of the join: median {_median(third_q):.0f} MB "
        f"in the third quarter of the run, {_median(last_q):.0f} MB in the last")
