"""Intensity limits from the data (``channel_intensity_limits='from_array'``)
without dask: ``array_utils.channel_extrema``.

It must be exact, and it must read the array the way it is stored: blocks
aligned to its chunks.  Read plane by plane, an output chunked in 101-deep
cubes decompressed every chunk 101 times -- 16 minutes instead of 20 s on a
35 GB stack, which looked like a conversion that never finished.
"""
from __future__ import annotations

import numpy as np
import pytest
import zarr

from eubi_bridge.utils.array_utils import channel_extrema

RNG = np.random.default_rng(21)


class _Recording:
    """An array that records the region of every read."""

    def __init__(self, arr):
        self._arr, self.shape, self.dtype, self.chunks = arr, arr.shape, arr.dtype, arr.chunks
        self.keys = []

    def __getitem__(self, key):
        self.keys.append(key)
        return self._arr[key]


@pytest.mark.parametrize("c_index", [1, None])
def test_extrema_are_exact(c_index):
    a = RNG.integers(-500, 3000, (2, 3, 17, 45, 61)).astype("i2")
    z = zarr.array(a, chunks=(1, 2, 5, 16, 20))
    lows, highs = channel_extrema(z, c_index, max_bytes=4096)
    if c_index is None:
        assert (lows, highs) == ([int(a.min())], [int(a.max())])
    else:
        assert lows == [int(a[:, c].min()) for c in range(3)]
        assert highs == [int(a[:, c].max()) for c in range(3)]


def test_reads_are_chunk_aligned_and_bounded():
    """Every read starts on a chunk boundary, so no chunk is decompressed
    twice, and stays within the memory limit."""
    a = RNG.integers(0, 255, (40, 64, 96)).astype("u1")
    rec = _Recording(zarr.array(a, chunks=(10, 16, 32)))
    max_bytes = 10 * 16 * 96 * 2                       # two chunk rows, whole width
    assert channel_extrema(rec, None, max_bytes=max_bytes) == ([int(a.min())], [int(a.max())])
    for key in rec.keys:
        assert all(isinstance(k, slice) for k in key), f"not a block read: {key}"
        assert [k.start % c for k, c in zip(key, rec.chunks)] == [0, 0, 0], key
        assert np.prod([k.stop - k.start for k in key]) <= max_bytes
    covered = sum(int(np.prod([k.stop - k.start for k in key])) for key in rec.keys)
    assert covered == a.size                           # each voxel read once
