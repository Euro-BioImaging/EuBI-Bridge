"""Narrowing dtype conversions must clip, not wrap.

The output store is created with the requested dtype, so region data has to be
cast to match it.  Before this, a narrowing cast reached TensorStore unconverted
and raised "Cannot cast ... according to the rule 'safe'"; a plain astype would
instead have wrapped (uint16 4000 -> uint8 160), silently corrupting the image.
"""
from __future__ import annotations

import sys
from pathlib import Path

import warnings

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eubi_bridge.core.writers import cast_to_dtype


class TestClipping:
    def test_narrowing_saturates_instead_of_wrapping(self):
        src = np.array([0, 160, 255, 4000, 65535], dtype=np.uint16)
        out = cast_to_dtype(src, "uint8")
        assert out.tolist() == [0, 160, 255, 255, 255]
        # A plain astype would have produced 160 for 4000.
        assert out[3] != src[3].astype(np.uint8)

    def test_widening_is_a_pure_dtype_change(self):
        src = np.array([0, 255], dtype=np.uint8)
        out = cast_to_dtype(src, "uint16")
        assert out.tolist() == [0, 255]
        assert out.dtype == np.uint16

    def test_same_dtype_returns_the_input_unchanged(self):
        src = np.zeros((2, 2), dtype=np.uint16)
        assert cast_to_dtype(src, "uint16") is src

    def test_float_to_int_rounds_then_clips(self):
        src = np.array([-5.0, 0.4, 254.7, 300.0], dtype=np.float32)
        assert cast_to_dtype(src, "uint8").tolist() == [0, 0, 255, 255]

    def test_int_to_float_is_not_clipped(self):
        src = np.array([0, 65535], dtype=np.uint16)
        out = cast_to_dtype(src, "float32")
        assert out.tolist() == [0.0, 65535.0]

    def test_signedness_change_clips_at_zero(self):
        src = np.array([-100, 0, 100], dtype=np.int16)
        assert cast_to_dtype(src, "uint8").tolist() == [0, 0, 100]


class TestNonFiniteFloats:
    """NaN and infinities have no integer counterpart.

    NaN survives ``clip`` and then casts to a platform-dependent value with a
    RuntimeWarning, so it is mapped to 0 explicitly.  Infinities clamp to the
    target's bounds like any other out-of-range value.
    """

    def test_nan_becomes_zero(self):
        src = np.array([np.nan, 5.0], dtype=np.float32)
        assert cast_to_dtype(src, "uint8").tolist() == [0, 5]

    def test_infinities_clamp_to_the_target_bounds(self):
        src = np.array([np.inf, -np.inf], dtype=np.float32)
        assert cast_to_dtype(src, "uint8").tolist() == [255, 0]
        assert cast_to_dtype(src, "int16").tolist() == [32767, -32768]

    def test_no_runtime_warning_is_raised(self):
        """The warning signalled genuinely undefined output."""
        src = np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            cast_to_dtype(src, "uint8")

    def test_finite_floats_are_unaffected(self):
        src = np.array([0.4, 254.7], dtype=np.float32)
        assert cast_to_dtype(src, "uint8").tolist() == [0, 255]

    def test_dask_matches_numpy_on_non_finite(self):
        da = pytest.importorskip("dask.array")
        src = np.array([np.nan, np.inf, -np.inf, 5.0], dtype=np.float32)
        expected = cast_to_dtype(src, "uint8")
        got = cast_to_dtype(da.from_array(src, chunks=2), "uint8").compute()
        assert got.tolist() == expected.tolist()

    def test_float_target_keeps_non_finite_values(self):
        """Only integer targets need the substitution."""
        src = np.array([np.nan, np.inf], dtype=np.float64)
        out = cast_to_dtype(src, "float32")
        assert np.isnan(out[0]) and np.isinf(out[1])


class TestLaziness:
    """The cast must not materialise a whole array in memory."""

    def test_dask_input_stays_lazy(self):
        da = pytest.importorskip("dask.array")
        arr = da.zeros((100, 100, 100), dtype=np.uint16, chunks=(10, 100, 100))
        out = cast_to_dtype(arr, "uint8")
        assert hasattr(out, "dask"), "cast computed the array eagerly"
        assert not isinstance(out, np.ndarray)

    def test_dask_chunking_is_preserved(self):
        da = pytest.importorskip("dask.array")
        arr = da.zeros((100, 100), dtype=np.uint16, chunks=(10, 100))
        assert cast_to_dtype(arr, "uint8").chunks == arr.chunks

    def test_dask_values_are_correct_once_computed(self):
        da = pytest.importorskip("dask.array")
        src = np.array([0, 4000, 65535], dtype=np.uint16)
        out = cast_to_dtype(da.from_array(src, chunks=2), "uint8")
        assert out.compute().tolist() == [0, 255, 255]

    def test_memory_is_bounded_by_the_region(self):
        """Regions are cast one at a time, so peak scales with region size."""
        import tracemalloc
        region = np.zeros((64, 256, 256), dtype=np.uint16)   # ~8 MB
        tracemalloc.start()
        cast_to_dtype(region, "uint8")
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # Generous bound: a few multiples of one region, never the whole array.
        assert peak < region.nbytes * 4
