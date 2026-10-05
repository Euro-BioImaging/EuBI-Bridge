"""RGB (multi-sample) pixels become channels, C x S of them, for every reader.

Files with real channels *and* RGB samples used to lose samples: every reader
kept sample 0 only (a 2-channel RGB ND2 wrote 2 channels of red; CZI logged a
warning; the TIFF readers called S "spurious" and dropped it).  Single-channel
RGB already became 3 channels and must stay exactly as it was.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import dask.array as da
import numpy as np
import pytest

from eubi_bridge.core.sample_channels import (SamplesAsChannels, expand_channel_metadata,
                                              fold_dask)

H, W = 40, 56


def _channels_major(tczyxs):
    """(T, C, Z, Y, X, S) -> (T, C*S, Z, Y, X): c0 s0, c0 s1, ..., c1 s0, ..."""
    t, c, z, y, x, s = tczyxs.shape
    return np.moveaxis(tczyxs, -1, 2).reshape(t, c * s, z, y, x)


def _omero_channels(store: Path):
    attrs = json.loads((store / ".zattrs").read_text())
    return [(ch.get("label"), ch.get("color")) for ch in attrs["omero"]["channels"]]


# -- the shared helper --------------------------------------------------------------

def test_fold_dask_is_channel_major_and_lazy():
    data = np.arange(2 * 3 * 2 * 4 * 5 * 3).reshape(2, 3, 2, 4, 5, 3)
    folded = fold_dask(da.from_array(data, chunks=(1, 1, 1, 4, 5, 3)))
    assert isinstance(folded, da.Array)
    np.testing.assert_array_equal(folded.compute(), _channels_major(data))
    np.testing.assert_array_equal(fold_dask(da.from_array(data), reverse=True).compute(),
                                  _channels_major(data[..., ::-1]))


def test_samples_as_channels_reads_only_what_is_asked():
    data = np.random.default_rng(0).integers(0, 255, (2, 3, 2, 8, 9, 3), dtype=np.uint8)
    reads = []

    class Source:
        shape, dtype, chunks = data.shape, data.dtype, None

        def __getitem__(self, key):
            reads.append(key)
            return data[key]

    folded, ref = SamplesAsChannels(Source()), _channels_major(data)
    assert folded.shape == ref.shape
    for key in [(slice(None),), (0, slice(2, 7)), (1, 4, 1, slice(2, 5), slice(3, 9)),
                (slice(None), slice(5, 9), 0), (0, 8)]:
        np.testing.assert_array_equal(folded[key], ref[key])
    # channels 5..8 are samples of real channels 1 and 2 only
    assert reads[3][1] == slice(1, 3)


def test_metadata_already_listing_samples_is_left_alone():
    """Bio-Formats lists single-channel RGB as 3 channels: unchanged."""
    from ome_types.model import Channel, Pixels
    pixels = Pixels(dimension_order="XYZCT", type="uint8", size_x=4, size_y=4, size_z=1,
                    size_t=1, size_c=3,
                    channels=[Channel(id=f"Channel:{i}", name=n, samples_per_pixel=1)
                              for i, n in enumerate(("Brightfield", "Brightfield", "x"))])
    before = [ch.model_dump() for ch in pixels.channels]
    expand_channel_metadata(pixels, 1, 3)
    assert [ch.model_dump() for ch in pixels.channels] == before


def test_metadata_for_several_rgb_channels():
    from ome_types.model import Channel, Pixels
    pixels = Pixels(dimension_order="XYZCT", type="uint8", size_x=4, size_y=4, size_z=1,
                    size_t=1, size_c=2,
                    channels=[Channel(id="Channel:0", name="DIC", samples_per_pixel=3),
                              Channel(id="Channel:1", samples_per_pixel=3)])
    expand_channel_metadata(pixels, 2, 3)
    expand_channel_metadata(pixels, 2, 3)                      # idempotent
    assert pixels.size_c == 6
    assert [c.name for c in pixels.channels] == ["DIC R", "DIC G", "DIC B",
                                                 "Channel 1 R", "Channel 1 G", "Channel 1 B"]
    assert [c.color.as_hex().upper() for c in pixels.channels] == \
        ["#F00", "#0F0", "#00F"] * 2


# -- bioio readers (ND2, LIF, bioio OME-TIFF, PNG/JPG, Bio-Formats fallback) -----------

class _FakeBioioImage:
    """What bioio-nd2 shows for a 2-channel RGB file: C=2 and S=3; asking for
    TCZYX without S returns sample 0 (bioio's behaviour)."""

    def __init__(self, data):                                   # T C Z Y X S
        self.data = data
        order = "TCZYXS"
        self.dims = type("Dims", (), dict(zip(order, data.shape), order=order))()
        self.scenes = ("0",)

    def get_image_dask_data(self, order):
        arr = da.from_array(self.data)
        return arr if order == "TCZYXS" else arr[..., 0]


@pytest.mark.parametrize("channels", [2, 1])
def test_bioio_reader_keeps_every_sample(channels):
    from eubi_bridge.core.pff_reader import BioIOReader
    data = np.random.default_rng(1).integers(0, 255, (2, channels, 1, 6, 7, 3), dtype=np.uint8)
    reader = BioIOReader("x.nd2", _FakeBioioImage(data))
    assert reader.sample_layout == (channels, 3)
    np.testing.assert_array_equal(np.asarray(reader.get_image_dask_data()),
                                  _channels_major(data))


# -- CZI ------------------------------------------------------------------------------

def _write_two_channel_rgb_czi(path):
    pyczi = pytest.importorskip("pylibCZIrw.czi")
    bgr = np.random.default_rng(2).integers(0, 255, (2, H, W, 3), dtype=np.uint8)
    with pyczi.create_czi(str(path)) as czi:
        for c in range(2):
            czi.write(data=bgr[c], plane={"T": 0, "Z": 0, "C": c}, location=(0, 0))
    rgb = bgr[..., ::-1]                                        # CZI stores B, G, R
    return _channels_major(rgb[None, :, None])                  # (1, 6, 1, H, W)


@pytest.mark.parametrize("as_mosaic", [False, True])
def test_czi_two_rgb_channels_keep_every_sample(tmp_path, as_mosaic):
    from eubi_bridge.core.czi_reader import read_czi
    expected = _write_two_channel_rgb_czi(tmp_path / "rgb2.czi")
    reader = read_czi(str(tmp_path / "rgb2.czi"), as_mosaic=as_mosaic)
    assert reader.sample_layout == (2, 3)
    np.testing.assert_array_equal(np.asarray(reader.get_image_dask_data()), expected)


def test_czi_two_rgb_channels_end_to_end(tmp_path):
    from eubi_bridge.ebridge import EuBIBridge
    expected = _write_two_channel_rgb_czi(tmp_path / "rgb2.czi")
    EuBIBridge().to_zarr(str(tmp_path / "rgb2.czi"), str(tmp_path / "out"),
                         squeeze=False, verbose=False)
    store = next((tmp_path / "out").glob("*.zarr"))
    import zarr
    np.testing.assert_array_equal(np.asarray(zarr.open_group(str(store), mode="r")["0"]),
                                  expected)
    labels_colors = _omero_channels(store)
    assert [c for _, c in labels_colors] == ["FF0000", "00FF00", "0000FF"] * 2
    assert [label[-1] for label, _ in labels_colors] == list("RGBRGB")


# -- TIFF -----------------------------------------------------------------------------

def _write_two_channel_rgb_tiff(path):
    import tifffile
    rgb = np.random.default_rng(3).integers(0, 255, (2, H, W, 3), dtype=np.uint8)
    tifffile.imwrite(str(path), rgb, photometric="rgb", metadata={"axes": "CYXS"})
    return _channels_major(rgb[None, :, None])


def test_tiff_two_rgb_channels_end_to_end(tmp_path):
    from eubi_bridge.ebridge import EuBIBridge
    expected = _write_two_channel_rgb_tiff(tmp_path / "rgb2.tif")
    EuBIBridge().to_zarr(str(tmp_path / "rgb2.tif"), str(tmp_path / "out"),
                         squeeze=False, verbose=False)
    store = next((tmp_path / "out").glob("*.zarr"))
    import zarr
    np.testing.assert_array_equal(np.asarray(zarr.open_group(str(store), mode="r")["0"]),
                                  expected)
    assert [c for _, c in _omero_channels(store)] == ["FF0000", "00FF00", "0000FF"] * 2


def test_tiff_bioio_reader_keeps_every_sample(tmp_path):
    pytest.importorskip("bioio_tifffile")
    from bioio_tifffile.reader import Reader
    from eubi_bridge.core.tiff_reader import TIFFBioIOReader
    expected = _write_two_channel_rgb_tiff(tmp_path / "rgb2.tif")
    img = Reader(str(tmp_path / "rgb2.tif"))
    if not ({"C", "S"} <= set(img.dims.order)):
        pytest.skip(f"bioio-tifffile shows {img.dims.order}")
    reader = TIFFBioIOReader(str(tmp_path / "rgb2.tif"), img)
    np.testing.assert_array_equal(np.asarray(reader.get_image_dask_data()), expected)


# -- real ND2 (skipped when the test data are not on this machine) --------------------

REAL_ND2 = Path(os.environ.get("EUBI_TEST_DATA",
                               "C:/Users/oezdemir/Desktop/quarantine/temp/ome/input")) \
    / "medium_dataset" / "imageJ_test.nd2"


@pytest.mark.skipif(not REAL_ND2.exists(), reason="real ND2 test data not available")
def test_real_two_channel_rgb_nd2_keeps_every_sample(tmp_path):
    """imageJ_test.nd2: T13 C2 Y2048 X2880 S3; wrote 2 channels of red before."""
    nd2 = pytest.importorskip("nd2")
    import zarr
    from eubi_bridge.ebridge import EuBIBridge
    with nd2.ND2File(str(REAL_ND2)) as f:
        truth = f.asarray()                                    # T C Y X S
    EuBIBridge().to_zarr(str(REAL_ND2), str(tmp_path / "out"), verbose=False)
    store = next((tmp_path / "out").glob("*.zarr"))
    out = np.asarray(zarr.open_group(str(store), mode="r")["0"])
    expected = truth.transpose(0, 1, 4, 2, 3).reshape(truth.shape[0], -1, *truth.shape[2:4])
    assert out.shape == expected.shape
    np.testing.assert_array_equal(out, expected)
