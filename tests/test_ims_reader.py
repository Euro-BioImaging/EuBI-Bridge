"""
Tests for the native Imaris (.ims) reader (Track A) and the
``keep_existing_resolutions`` pyramid-preservation path (Track C).

Builds tiny synthetic ``.ims``-shaped HDF5 files with h5py to avoid depending
on real Imaris fixtures.
"""

import asyncio

import h5py
import numpy as np
import pytest

from eubi_bridge.core.data_manager import IMSImageMeta, _build_ims_omemeta
from eubi_bridge.core.ims_reader import IMSReader, read_ims


def _char_attr(value) -> np.ndarray:
    """Encode a value as an Imaris-style array of 1-byte char attributes."""
    return np.array(list(str(value)), dtype='S1')


def _write_synthetic_ims(path, n_resolution_levels=2, n_timepoints=1, n_channels=2,
                          base_shape=(4, 16, 16)):
    """Write a minimal .ims-shaped HDF5 file with `n_resolution_levels` levels.

    Resolution level ``r`` has shape ``base_shape / 2**r`` (per axis, floor).
    """
    with h5py.File(path, 'w') as f:
        for r in range(n_resolution_levels):
            shape = tuple(max(1, s // (2 ** r)) for s in base_shape)
            for t in range(n_timepoints):
                for c in range(n_channels):
                    grp = f.require_group(
                        f'/DataSet/ResolutionLevel {r}/TimePoint {t}/Channel {c}'
                    )
                    data = (np.random.rand(*shape) * 255).astype(np.uint16)
                    grp.create_dataset('Data', data=data)

        img = f.create_group('DataSetInfo/Image')
        img.attrs['X'] = _char_attr(base_shape[2])
        img.attrs['Y'] = _char_attr(base_shape[1])
        img.attrs['Z'] = _char_attr(base_shape[0])
        img.attrs['ExtMin0'] = _char_attr(0)
        img.attrs['ExtMax0'] = _char_attr(base_shape[2])
        img.attrs['ExtMin1'] = _char_attr(0)
        img.attrs['ExtMax1'] = _char_attr(base_shape[1])
        img.attrs['ExtMin2'] = _char_attr(0)
        img.attrs['ExtMax2'] = _char_attr(base_shape[0])

        for c in range(n_channels):
            ch = f.create_group(f'DataSetInfo/Channel {c}')
            ch.attrs['Name'] = _char_attr(f'Channel {c}')


@pytest.fixture
def synthetic_ims_file(tmp_test_data):
    path = tmp_test_data / "synthetic.ims"
    _write_synthetic_ims(path, n_resolution_levels=2, n_timepoints=1, n_channels=2,
                          base_shape=(4, 16, 16))
    return path


def test_read_ims_basic(synthetic_ims_file):
    reader = read_ims(str(synthetic_ims_file))
    assert isinstance(reader, IMSReader)
    assert reader.n_scenes == 1
    assert reader.n_tiles == 1

    data = reader.get_image_dask_data()
    assert data.shape == (1, 2, 4, 16, 16)


def test_n_resolution_levels(synthetic_ims_file):
    reader = read_ims(str(synthetic_ims_file))
    assert reader.n_resolution_levels == 2


def test_get_resolution_level_dask_data(synthetic_ims_file):
    reader = read_ims(str(synthetic_ims_file))
    level0 = reader.get_resolution_level_dask_data(0)
    level1 = reader.get_resolution_level_dask_data(1)
    assert level0.shape == (1, 2, 4, 16, 16)
    assert level1.shape == (1, 2, 2, 8, 8)


def test_ims_image_meta_basic(synthetic_ims_file):
    meta = IMSImageMeta(str(synthetic_ims_file))
    asyncio.run(meta.read_dataset())

    assert meta.n_scenes == 1
    pixels = meta.get_pixels()
    assert pixels.size_c == 2
    assert pixels.size_t == 1
    assert pixels.size_x == 16
    assert pixels.size_y == 16
    assert pixels.size_z == 4

    scaledict = meta.get_scaledict()
    assert scaledict['x'] == pytest.approx(1.0)
    assert scaledict['y'] == pytest.approx(1.0)
    assert scaledict['z'] == pytest.approx(1.0)

    assert meta.arraydata.shape == (1, 2, 4, 16, 16)


def test_ims_image_meta_keep_existing_resolutions(synthetic_ims_file):
    meta = IMSImageMeta(str(synthetic_ims_file), keep_existing_resolutions=True)
    asyncio.run(meta.read_dataset())

    pyr = asyncio.run(meta.get_pyramid())
    layers = pyr.layers
    assert set(layers.keys()) == {'0', '1'}
    assert layers['0'].shape == (1, 2, 4, 16, 16)
    assert layers['1'].shape == (1, 2, 2, 8, 8)

    scale0 = pyr.meta.get_scale('0')
    scale1 = pyr.meta.get_scale('1')
    # Level 1 is downsampled by 2x in z, y, x relative to level 0.
    axis_order = pyr.meta.axis_order
    for ax, factor in zip(axis_order, (1.0, 1.0, 2.0, 2.0, 2.0)):
        i = axis_order.index(ax)
        assert scale1[i] == pytest.approx(scale0[i] * factor)


def test_ims_image_meta_without_keep_existing_resolutions_single_layer(synthetic_ims_file):
    meta = IMSImageMeta(str(synthetic_ims_file), keep_existing_resolutions=False)
    asyncio.run(meta.read_dataset())

    pyr = asyncio.run(meta.get_pyramid())
    assert set(pyr.layers.keys()) == {'0'}
    assert pyr.layers['0'].shape == (1, 2, 4, 16, 16)


# --- Imaris' chunk padding -------------------------------------------------
#
# Real Imaris files pad every Data dataset to whole HDF5 chunks: a 424 x 424
# image is stored as 512 x 512 with zeros beyond, and the real size is in the
# Channel group's ImageSizeZ/Y/X attributes (per resolution level).  The
# reader used to wrap the padded dataset, so conversions carried the zero
# margin.

REAL, PADDED = (5, 21, 27), (8, 32, 32)


def _write_padded_ims(path, n_resolution_levels=2, n_channels=2):
    """Like Imaris: each level half the size (rounded down), padded to
    ``PADDED / 2**r``, with the real size in ImageSize attributes.  Returns
    the real (unpadded) data per level as (C, Z, Y, X)."""
    rng = np.random.default_rng(0)
    real_levels = []
    with h5py.File(path, 'w') as f:
        for r in range(n_resolution_levels):
            real = tuple(max(1, s >> r) for s in REAL)
            padded = tuple(max(1, s >> r) for s in PADDED)
            level = rng.integers(1, 255, size=(n_channels, *real), dtype=np.uint16)
            for c in range(n_channels):
                grp = f.require_group(f'/DataSet/ResolutionLevel {r}/TimePoint 0/Channel {c}')
                data = np.zeros(padded, dtype=np.uint16)
                data[:real[0], :real[1], :real[2]] = level[c]
                grp.create_dataset('Data', data=data, chunks=padded)
                for axis, size in zip('ZYX', real):
                    grp.attrs[f'ImageSize{axis}'] = _char_attr(size)
            real_levels.append(level)
        img = f.create_group('DataSetInfo/Image')
        for axis, size, i in zip('ZYX', REAL, (2, 1, 0)):
            img.attrs[axis] = _char_attr(size)
            img.attrs[f'ExtMin{i}'] = _char_attr(0)
            img.attrs[f'ExtMax{i}'] = _char_attr(size * 0.5)       # 0.5 um pixels
        for c in range(n_channels):
            f.create_group(f'DataSetInfo/Channel {c}').attrs['Name'] = _char_attr(f'Ch{c}')
    return real_levels


def test_padding_is_cropped_at_every_level(tmp_test_data):
    path = tmp_test_data / "padded.ims"
    real_levels = _write_padded_ims(path)
    reader = read_ims(str(path))
    for r, real in enumerate(real_levels):
        data = np.asarray(reader.get_resolution_level_dask_data(r))
        assert data.shape == (1, *real.shape)
        np.testing.assert_array_equal(data[0], real)


def test_padded_array_matches_its_metadata(tmp_test_data):
    """Pixel sizes were always right (extent / real size, as Bio-Formats);
    only the array was too large for them."""
    path = tmp_test_data / "padded.ims"
    _write_padded_ims(path)
    meta = IMSImageMeta(str(path))
    asyncio.run(meta.read_dataset())
    pixels = meta.get_pixels()
    assert (pixels.size_z, pixels.size_y, pixels.size_x) == REAL
    assert meta.arraydata.shape == (1, 2, *REAL)
    assert meta.get_scaledict()['x'] == pytest.approx(0.5)


@pytest.mark.parametrize("keep", [False, True], ids=["rebuilt_pyramid", "kept_pyramid"])
def test_conversion_writes_the_real_size(tmp_path, keep):
    """End to end: the written OME-Zarr has the image, not the padding."""
    import zarr

    from eubi_bridge.ebridge import EuBIBridge

    source = tmp_path / "padded.ims"
    real_levels = _write_padded_ims(source)
    out = tmp_path / "out"
    EuBIBridge().to_zarr(str(source), str(out), squeeze=False, n_layers=2,
                         keep_existing_resolutions=keep, verbose=False)
    root = zarr.open_group(str(next(out.glob("*.zarr"))), mode='r')
    level0 = np.asarray(root['0'])
    assert level0.shape == (1, 2, *REAL)
    np.testing.assert_array_equal(level0[0], real_levels[0])
    if keep:
        np.testing.assert_array_equal(np.asarray(root['1'])[0], real_levels[1])


def test_build_ims_omemeta_malformed(tmp_test_data):
    path = tmp_test_data / "malformed.ims"
    with h5py.File(path, 'w') as f:
        f.create_group('/DataSet/ResolutionLevel 0/TimePoint 0/Channel 0')
        f['/DataSet/ResolutionLevel 0/TimePoint 0/Channel 0'].create_dataset(
            'Data', data=np.zeros((2, 2, 2), dtype=np.uint16))
        # Deliberately omit /DataSetInfo/Image

    with pytest.raises(Exception):
        _build_ims_omemeta(str(path))
