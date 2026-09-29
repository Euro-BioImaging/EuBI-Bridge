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


@pytest.mark.parametrize("squeeze", [True, False])
def test_intensity_limits_from_array_with_kept_resolutions(tmp_path, squeeze):
    """A kept source pyramid can have a singleton axis, and the post-write
    'limits from array' step must still save the channel windows.

    That step used to squeeze the reopened output when ``squeeze`` was on (the
    default), which swaps its on-disk pyramid for a detached in-memory one, so
    saving failed with 'No zarr group connected' -- every pyramidal IMS input
    in a batch with both options set.  With ``squeeze`` off it already worked;
    both are pinned so neither regresses.
    """
    import json

    import tifffile

    from eubi_bridge.ebridge import EuBIBridge

    # A two-level OME-Zarr that keeps its singleton t axis, as an IMS
    # pyramid does.
    source = tmp_path / "src.tif"
    data = np.zeros((1, 8, 2, 64, 64), dtype=np.uint8)
    data[:, :, 0] = 40
    data[:, :, 1] = 90
    data[0, 0, 0, 0, 0] = 10
    tifffile.imwrite(source, data, imagej=True, metadata={"axes": "TZCYX"})
    bridge = EuBIBridge()
    bridge.to_zarr(str(source), str(tmp_path / "pyr"), squeeze=False,
                   n_layers=2, verbose=False)
    pyramid = next((tmp_path / "pyr").glob("*.zarr"))

    out = tmp_path / "out"
    bridge.to_zarr(str(pyramid), str(out), keep_existing_resolutions=True,
                   channel_intensity_limits="from_array", squeeze=squeeze,
                   verbose=False)

    store = next(out.glob("*.zarr"))
    attrs = json.loads((store / ".zattrs").read_text())
    windows = [ch["window"] for ch in attrs["omero"]["channels"]]
    assert [(w["start"], w["end"]) for w in windows] == [(10, 40), (90, 90)]
    # The kept pyramid is written unsqueezed, and so is its metadata.
    assert [a["name"] for a in attrs["multiscales"][0]["axes"]] == list("tczyx")
