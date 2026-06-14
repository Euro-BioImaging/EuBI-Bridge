"""
Tests for ``store_existing_pyramid_async`` (Track C writer-side): writing a
pre-built multi-layer ``Pyramid`` verbatim, without recomputing levels via
``downscale_with_tensorstore_async``.
"""

import asyncio

import dask.array as da
import numpy as np
import pytest

from eubi_bridge.core import writers
from eubi_bridge.core.writers import store_existing_pyramid_async
from eubi_bridge.ngff.multiscales import Pyramid


@pytest.fixture
def two_layer_pyramid():
    rng = np.random.default_rng(42)
    arr0 = da.from_array((rng.random((1, 2, 4, 16, 16)) * 255).astype(np.uint8))
    arr1 = da.from_array((rng.random((1, 2, 2, 8, 8)) * 255).astype(np.uint8))
    pyr = Pyramid().from_arrays(
        arrays=[arr0, arr1],
        axis_order='tczyx',
        unit_list=['second', 'micrometer', 'micrometer', 'micrometer'],
        scales=[[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0, 2.0]],
        version='0.4',
        name='Series_0',
    )
    return pyr, arr0, arr1


def test_store_existing_pyramid_writes_all_levels(tmp_test_data, two_layer_pyramid):
    pyr, arr0, arr1 = two_layer_pyramid
    output_path = tmp_test_data / "existing_pyramid.zarr"

    asyncio.run(store_existing_pyramid_async(
        pyr=pyr,
        output_path=str(output_path),
        axes='tczyx',
        units=['second', 'micrometer', 'micrometer', 'micrometer'],
        channel_meta='auto',
        zarr_format=2,
        auto_chunk=True,
        output_chunks=None,
        output_shard_coefficients=None,
        overwrite=True,
    ))

    result = Pyramid(str(output_path))
    layers = result.layers
    assert set(layers.keys()) == {'0', '1'}

    np.testing.assert_array_equal(np.asarray(layers['0']), arr0.compute())
    np.testing.assert_array_equal(np.asarray(layers['1']), arr1.compute())

    scale0 = result.meta.get_scale('0')
    scale1 = result.meta.get_scale('1')
    axis_order = result.meta.axis_order
    for ax, factor in zip(axis_order, (1.0, 1.0, 2.0, 2.0, 2.0)):
        i = axis_order.index(ax)
        assert scale1[i] == pytest.approx(scale0[i] * factor)


def test_store_existing_pyramid_skips_downscaling(tmp_test_data, two_layer_pyramid, monkeypatch):
    pyr, arr0, arr1 = two_layer_pyramid
    output_path = tmp_test_data / "existing_pyramid_no_downscale.zarr"

    async def _raise(*args, **kwargs):
        raise AssertionError("downscale_with_tensorstore_async should not be called")

    monkeypatch.setattr(writers, "downscale_with_tensorstore_async", _raise)

    asyncio.run(store_existing_pyramid_async(
        pyr=pyr,
        output_path=str(output_path),
        axes='tczyx',
        units=['second', 'micrometer', 'micrometer', 'micrometer'],
        channel_meta='auto',
        zarr_format=2,
        auto_chunk=True,
        output_chunks=None,
        output_shard_coefficients=None,
        overwrite=True,
    ))

    result = Pyramid(str(output_path))
    assert set(result.layers.keys()) == {'0', '1'}
