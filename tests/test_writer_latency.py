"""The writers end when the last region is written, not on a polling timer.

They used to poll: writer threads took regions with a 1 s timeout and
stopped on an empty queue, and the base-layer write waited for a progress
monitor that slept 2 s before every check.  Every write -- every scene, tile
and pyramid -- paid up to 3 s, however small: a 46-image LIF spent 127 of its
187 s waiting (2026-10-01).  These tests write tiny arrays and require them
to finish well under the old floor, with the right pixels.  The pyramid
writer (same fix) needs a full OME-Zarr image around it; it is covered by
the conversion timing in test_conversion_has_no_fixed_wait.
"""
from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from eubi_bridge.core.writers import write_with_queue_async

RNG = np.random.default_rng(2)
#: the old code needed >= 2 s for any base-layer write
LIMIT_S = 0.9


def _base(tmp_path, data, name, **kw):
    out = str(tmp_path / name)
    t0 = time.perf_counter()
    asyncio.run(write_with_queue_async(
        arr=data, output_path=out, output_chunks=(1, 32, 32), zarr_format=2,
        dtype=data.dtype, overwrite=True, max_concurrency=4, **kw))
    return out, time.perf_counter() - t0


@pytest.mark.parametrize("region_mb,verbose", [(8.0, False), (0.002, True)],
                         ids=["one-region", "many-regions-logged"])
def test_base_layer_write_ends_with_the_last_region(tmp_path, region_mb, verbose):
    import zarr
    data = RNG.integers(0, 4000, (2, 96, 80)).astype("u2")
    _base(tmp_path, data, "warm.zarr")                      # imports, first tensorstore open
    out, seconds = _base(tmp_path, data, "out.zarr", region_size_mb=region_mb,
                         verbose=verbose)
    np.testing.assert_array_equal(zarr.open_array(out, mode="r")[:], data)
    assert seconds < LIMIT_S, f"a tiny write took {seconds:.2f} s"


def test_conversion_has_no_fixed_wait(tmp_path):
    """A whole conversion of a tiny image, pyramid included, in-process: the
    old writers alone added ~3 s of waiting to it."""
    import tifffile

    from eubi_bridge.ebridge import EuBIBridge
    source = tmp_path / "tiny.tif"
    tifffile.imwrite(source, RNG.integers(0, 255, (2, 64, 64)).astype("u1"),
                     imagej=True, metadata={"axes": "CYX"})
    EuBIBridge().to_zarr(str(source), str(tmp_path / "warm"), verbose=False,
                         use_threading=True, max_workers=1)
    t0 = time.perf_counter()
    EuBIBridge().to_zarr(str(source), str(tmp_path / "out"), verbose=False,
                         use_threading=True, max_workers=1)
    seconds = time.perf_counter() - t0
    assert list((tmp_path / "out").glob("*.zarr"))
    # measured: ~2.5 s now (metadata, JVM reuse), ~5.5 s with the polling writers
    assert seconds < 4.0, f"a tiny conversion took {seconds:.2f} s"
