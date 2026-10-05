"""The read lock that keeps ND2 conversions correct must keep applying.

bioio-nd2's dask array is not thread-safe: read from several threads at once
it returns wrong pixels without an error, or crashes the process (seen with 4
threads on real files).  eubi-bridge reads regions with 2 x max_concurrency
threads, and stays correct only because ``writers._read_region`` serialises
reads of *resource-backed* dask arrays through a per-array lock -- a lock
added for Bio-Formats that also covers bioio-nd2, because both return a
``ResourceBackedDaskArray``.

That protection hangs on a type check.  These tests fail, instead of data
silently getting corrupted, if any link breaks: bioio-nd2 no longer returning
that type, one of eubi-bridge's transforms between reader and writer losing
it, or the writer no longer taking the lock.
"""
from __future__ import annotations

import threading
import time

import dask.array as da
import numpy as np
import pytest

rbda = pytest.importorskip("resource_backed_dask_array")
from resource_backed_dask_array import ResourceBackedDaskArray, resource_backed_dask_array  # noqa: E402

from eubi_bridge.core.writers import _is_resource_backed, _read_region  # noqa: E402


class _Ctx:
    """The minimal resource a ResourceBackedDaskArray needs."""
    closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _resource_backed(shape=(2, 3, 1, 32, 40, 3)):
    data = np.arange(np.prod(shape), dtype=np.uint32).reshape(shape)
    return data, resource_backed_dask_array(da.from_array(data), _Ctx())


# -- 1. bioio-nd2 hands out a resource-backed array ------------------------------------

def test_bioio_nd2_returns_a_resource_backed_array(tmp_path):
    """If this fails, bioio-nd2 changed what it returns: the writer's lock no
    longer applies to ND2.  Check whether the new array is thread-safe before
    relaxing anything."""
    pytest.importorskip("bioio_nd2")
    from bioio_nd2 import Reader

    from tests._nd2_writer import write_nd2
    from eubi_bridge.core.pff_reader import BioIOReader

    path = str(tmp_path / "x.nd2")
    frames = np.random.default_rng(0).integers(0, 65535, (6, 20, 30, 2, 1), dtype="u2")
    write_nd2(path, frames, [("T", 3), ("Z", 2)])
    reader = BioIOReader(path, Reader(path))
    arr = reader.get_image_dask_data()
    assert isinstance(arr, ResourceBackedDaskArray), type(arr)
    np.testing.assert_array_equal(arr.compute(),
                                  frames.reshape(3, 2, 20, 30, 2).transpose(0, 4, 1, 2, 3))


def test_bioio_nd2_rgb_fold_keeps_the_lock(tmp_path):
    """Multi-channel RGB goes through sample_channels.fold_dask."""
    pytest.importorskip("bioio_nd2")
    from bioio_nd2 import Reader

    from tests._nd2_writer import write_nd2
    from eubi_bridge.core.pff_reader import BioIOReader

    path = str(tmp_path / "rgb.nd2")
    frames = np.random.default_rng(1).integers(0, 255, (2, 16, 24, 2, 3), dtype="u1")
    write_nd2(path, frames, [("T", 2)])
    reader = BioIOReader(path, Reader(path))
    assert reader.sample_layout == (2, 3)
    assert isinstance(reader.get_image_dask_data(), ResourceBackedDaskArray)


# -- 2. eubi-bridge's transforms between reader and writer keep it ---------------------

@pytest.mark.parametrize("transform", [
    ("bioio sample selection", lambda a: a[..., 0]),
    ("RGB fold (sample_channels)", lambda a: __import__(
        "eubi_bridge.core.sample_channels", fromlist=["fold_dask"]).fold_dask(a)),
    ("squeeze (manager.squeeze)", lambda a: da.squeeze(a[..., 0])),
    ("crop (manager.crop)", lambda a: a[..., 0][:, 1:3, :, 4:20, 5:30]),
    ("transpose (manager.transpose)", lambda a: a[..., 0].transpose(0, 2, 1, 3, 4)),
    ("dtype cast", lambda a: a.astype(np.float32)),
    ("rechunk", lambda a: a.rechunk((1, 1, 1, 16, 20, 3))),
], ids=lambda t: t[0])
def test_transforms_keep_the_array_resource_backed(transform):
    _, arr = _resource_backed()
    out = transform[1](arr)
    assert _is_resource_backed(out), f"{transform[0]} returned a plain {type(out).__name__}"


# -- 3. the writer serialises reads of resource-backed arrays --------------------------

def _max_concurrent_reads(arr_factory, threads=8):
    """Read regions of the array from *threads* threads through the writer's
    _read_region; return the most block reads that ran at the same time."""
    active, peak = [0], [0]
    lock = threading.Lock()

    def slow_block(block):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.02)
        with lock:
            active[0] -= 1
        return block

    arr = arr_factory(slow_block)
    regions = [(slice(t, t + 1), slice(c, c + 1)) for t in range(arr.shape[0])
               for c in range(arr.shape[1])]
    workers = [threading.Thread(target=lambda r=r: _read_region(arr, r)) for r in regions]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    return peak[0]


def test_writer_serialises_resource_backed_reads():
    data = np.zeros((4, 4, 1, 16, 16), np.uint16)

    def plain(fn):
        return da.from_array(data, chunks=(1, 1, 1, 16, 16)).map_blocks(fn)

    def backed(fn):
        return resource_backed_dask_array(plain(fn), _Ctx())

    # control: without the lock, reads really do overlap (so the check below
    # can tell the difference)
    assert _max_concurrent_reads(plain) > 1
    assert _max_concurrent_reads(backed) == 1
