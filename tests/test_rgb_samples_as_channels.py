"""RGB (multi-sample) pixels become channels, C x S of them, for every reader.

Files with real channels *and* RGB samples used to lose samples: every reader
kept sample 0 only (a 2-channel RGB ND2 wrote 2 channels of red; CZI logged a
warning; the TIFF readers called S "spurious" and dropped it).  Single-channel
RGB already became 3 channels; from 0.3.0 they are named and tinted as several
channels' samples are ("<name> R/G/B"; unnamed: Channel 0/1/2, tinted).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from tests.conftest import requires_full  # noqa: E402

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

@requires_full
def test_fold_dask_is_channel_major_and_lazy():
    import dask.array as da
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


# -- one RGB channel: named and tinted like several (decided 2026-10-20) --------------

RED, GREEN, BLUE, WHITE = "#F00", "#0F0", "#00F", "#FFF"


def _pixels(names, samples, size_c=None):
    """OME pixels as a reader leaves them: one entry per real channel, ``size_c`` either
    the real channels (Bio-Formats) or already C x S (micro_omemeta)."""
    from ome_types.model import Channel, Pixels
    channels = [Channel(id=f"Channel:{i}", samples_per_pixel=samples, **({"name": n} if n else {}))
                for i, n in enumerate(names)]
    return Pixels(dimension_order="XYZCT", type="uint8", size_x=4, size_y=4, size_z=1,
                  size_t=1, size_c=size_c or len(names), channels=channels)


def _named(pixels):
    return [(c.name, c.color.as_hex().upper()) for c in pixels.channels]


@pytest.mark.parametrize("size_c", [1, 3], ids=["bioformats", "micro"])
def test_one_named_rgb_channel_is_named_like_several(size_c):
    """Carlos' pollen ND2 ('Brightfield') and CZI ('Bright'): one channel of 3 samples. It
    lost its name (Channel 0/1/2) before; now "<name> R/G/B", tinted red / green / blue."""
    pixels = expand_channel_metadata(_pixels(["Brightfield"], 3, size_c), 1, 3)
    expand_channel_metadata(pixels, 1, 3)                      # idempotent
    assert pixels.size_c == 3
    assert _named(pixels) == [("Brightfield R", RED), ("Brightfield G", GREEN),
                              ("Brightfield B", BLUE)]


@pytest.mark.parametrize("names", [[None], [""], []], ids=["unnamed", "empty", "none listed"])
def test_one_unnamed_rgb_channel_keeps_its_numbers_and_is_tinted(names):
    """No name: no label, so the writer's Channel 0/1/2 as before; tinted now (it was
    white)."""
    pixels = expand_channel_metadata(_pixels(names, 3, size_c=1), 1, 3)
    assert pixels.size_c == 3
    assert _named(pixels) == [(None, RED), (None, GREEN), (None, BLUE)]


def test_rgba_and_sample_counts_without_labels():
    """RGBA: R / G / B / A, the alpha white. Two samples: numbered, no tint (white)."""
    assert _named(expand_channel_metadata(_pixels(["DIC"], 4), 1, 4)) == [
        ("DIC R", RED), ("DIC G", GREEN), ("DIC B", BLUE), ("DIC A", WHITE)]
    assert _named(expand_channel_metadata(_pixels([None], 4), 1, 4)) == [
        (None, RED), (None, GREEN), (None, BLUE), (None, WHITE)]
    assert _named(expand_channel_metadata(_pixels(["DIC"], 2), 1, 2)) == [
        ("DIC 0", WHITE), ("DIC 1", WHITE)]
    assert _named(expand_channel_metadata(_pixels([None], 2), 1, 2)) == [
        (None, WHITE), (None, WHITE)]
    assert _named(expand_channel_metadata(_pixels(["A", None], 2), 2, 2)) == [
        ("A 0", WHITE), ("A 1", WHITE), ("Channel 1 0", WHITE), ("Channel 1 1", WHITE)]


class _Image:
    """A micro-reader image with RGB samples folded into c: its ``source`` keeps the S
    axis, its metadata one channel per real channel."""

    def __init__(self, names, samples):
        from types import SimpleNamespace
        c = len(names)
        self.source = SimpleNamespace(axes="tczyxs", shape=(1, c, 1, 4, 4, samples))
        self.axes, self.shape = "tczyx", (1, c * samples, 1, 4, 4)
        self.metadata = SimpleNamespace(channels=[
            SimpleNamespace(name=n, color="#FFFFFF", emission_wavelength=None,
                            excitation_wavelength=None) for n in names])


def _short(hex6):
    """FF0000 -> #F00, as ome_types writes it."""
    from ome_types.model.simple_types import Color
    return Color(hex6).as_hex().upper()


@pytest.mark.parametrize("names, samples", [
    (["Brightfield"], 3), ([None], 3), (["DIC"], 4), ([None], 4), (["DIC"], 2), ([None], 2),
    (["DIC", None], 3), ([None, None], 3)])
def test_every_path_names_samples_alike(names, samples):
    """The older readers (expand_channel_metadata), micro-reader's images on the ozp path
    (apply_rgb_channels, also show_pixel_meta's) and joined files (aggregative _channels)
    name and tint the sample channels alike."""
    from eubi_bridge.conversion.aggregative_path import _channels
    from eubi_bridge.conversion.pyramid_path import rgb_sample_channels
    legacy = [(c.name or f"Channel {i}", c.color.as_hex().upper())
              for i, c in enumerate(expand_channel_metadata(_pixels(names, samples),
                                                            len(names), samples).channels)]
    micro = [(label or f"Channel {i}", _short(color))
             for i, (label, color) in enumerate(rgb_sample_channels(_Image(names, samples)))]
    assert micro == legacy
    total = len(names) * samples
    joined = _channels(["f"], {"f": _Image(names, samples)}, {"f": total}, total, {})
    assert [(c.name or f"Channel {i}", _short(c.color)) for i, c in enumerate(joined)] == legacy


def test_a_file_without_samples_is_not_renamed():
    from types import SimpleNamespace
    from eubi_bridge.conversion.pyramid_path import apply_rgb_channels, rgb_sample_channels
    image = SimpleNamespace(axes="tczyx", shape=(1, 3, 1, 4, 4), metadata=None)
    assert rgb_sample_channels(image) is None
    pyr = object()
    assert apply_rgb_channels(pyr, image) is pyr


def _named_rgb_czi(path: Path, name: str = "Bright") -> Path:
    """One Bgr24 channel named *name* (as Carlos' pollen CZIs: 'Bright')."""
    from tests._czi_writer import raw_payload, write_czi
    meta = ('<Information><Image><Dimensions><Channels><Channel Id="Channel:0" '
            f'Name="{name}"></Channel></Channels></Dimensions></Image></Information>')
    rgb = np.random.default_rng(4).integers(0, 255, (H, W, 3), dtype=np.uint8)
    write_czi(str(path), [dict(scene=0, plane={"C": 0}, x=0, y=0, height=H, width=W,
                               dtype="u1", samples=3, compression=0,
                               payload=raw_payload(rgb))], metadata=meta)
    return path


def _rgb_tiff(path: Path, name=None) -> Path:
    import tifffile
    rgb = np.random.default_rng(5).integers(0, 255, (H, W, 3), dtype=np.uint8)
    if name is None:
        tifffile.imwrite(str(path), rgb, photometric="rgb")
    else:
        tifffile.imwrite(str(path), rgb, photometric="rgb", ome=True,
                         metadata={"axes": "YXS", "Channel": {"Name": name}})
    return path


def _convert_one(src: Path, out: Path, **kwargs):
    from eubi_bridge.ebridge import EuBIBridge
    EuBIBridge().to_zarr(str(src), str(out), squeeze=False, verbose=False, **kwargs)
    (store,) = out.glob("*.zarr")
    return _omero_channels(store)


RGB_TINTS = ["FF0000", "00FF00", "0000FF"]


@pytest.mark.parametrize("kind, name", [("czi", "Bright"), ("ome.tif", "Brightfield")])
def test_one_named_rgb_channel_end_to_end(tmp_path, kind, name):
    src = (_named_rgb_czi(tmp_path / "rgb.czi", name) if kind == "czi"
           else _rgb_tiff(tmp_path / "rgb.ome.tif", name))
    labels = _convert_one(src, tmp_path / "out")
    assert labels == [(f"{name} {s}", c) for s, c in zip("RGB", RGB_TINTS)]


def test_one_unnamed_rgb_channel_end_to_end(tmp_path):
    labels = _convert_one(_rgb_tiff(tmp_path / "rgb.tif"), tmp_path / "out")
    assert labels == [(f"Channel {i}", c) for i, c in enumerate(RGB_TINTS)]


@requires_full
def test_one_named_rgb_channel_on_the_older_readers(tmp_path):
    """metadata_reader='bfio' (Bio-Formats' metadata, the older pixel reader):
    expand_channel_metadata names it as micro-reader's path does."""
    src = _rgb_tiff(tmp_path / "rgb.ome.tif", "Brightfield")
    labels = _convert_one(src, tmp_path / "out", metadata_reader="bfio")
    assert labels == [(f"Brightfield {s}", c) for s, c in zip("RGB", RGB_TINTS)]


def test_the_users_channel_labels_and_colours_still_win(tmp_path):
    """channel_labels / channel_colors (CLI, GUI, batch settings) over the sample names;
    'auto' and unlisted channels keep them."""
    src = _named_rgb_czi(tmp_path / "rgb.czi")
    labels = _convert_one(src, tmp_path / "out", channel_labels="0,Red;2,auto",
                          channel_colors="1,white;2,auto")
    assert labels == [("Red", "FF0000"), ("Bright G", "FFFFFF"), ("Bright B", "0000FF")]


def test_a_conversion_tables_channel_labels_still_win(tmp_path):
    """A batch / conversion table row's channel_labels and channel_colors win too."""
    src = _named_rgb_czi(tmp_path / "rgb.czi")
    table = tmp_path / "table.csv"
    table.write_text("input_path,channel_labels,channel_colors\n"
                     f"{src.as_posix()},\"1,Green\",\"0,magenta\"\n")
    labels = _convert_one(table, tmp_path / "out")
    assert labels == [("Bright R", "FF00FF"), ("Green", "00FF00"), ("Bright B", "0000FF")]


def test_joined_views_of_a_named_rgb_channel_keep_the_name(tmp_path):
    """Views joined along c (View{v}_<name>): the samples' names, as one view's."""
    from tests._czi_writer import raw_payload, write_czi
    meta = ('<Information><Image><Dimensions><Channels><Channel Id="Channel:0" '
            'Name="Bright"></Channel></Channels></Dimensions></Image></Information>')
    rng = np.random.default_rng(6)
    tiles = [dict(scene=0, plane={"C": 0, "V": v}, x=0, y=0, height=H, width=W, dtype="u1",
                  samples=3, compression=0,
                  payload=raw_payload(rng.integers(0, 255, (H, W, 3)).astype("u1")))
             for v in range(2)]
    write_czi(str(tmp_path / "views.czi"), tiles, metadata=meta)
    labels = _convert_one(tmp_path / "views.czi", tmp_path / "out", view_index="all",
                          concat_views=True)
    assert labels == [(f"View{v}_Bright {s}", c) for v in range(2)
                      for s, c in zip("RGB", RGB_TINTS)]


def test_joined_named_rgb_files_keep_the_name(tmp_path):
    """Aggregative: two named one-channel RGB files along t, named as one file is."""
    src = tmp_path / "set"
    src.mkdir()
    for t in range(2):
        _rgb_tiff(src / f"img_T{t}.ome.tif", "Brightfield")
    from eubi_bridge.ebridge import EuBIBridge
    EuBIBridge().to_zarr(str(src), str(tmp_path / "out"), time_tag="T",
                         concatenation_axes="t", squeeze=False, verbose=False)
    (store,) = (tmp_path / "out").glob("*.zarr")
    assert _omero_channels(store) == [(f"Brightfield {s}", c) for s, c in zip("RGB", RGB_TINTS)]


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
        import dask.array as da
        arr = da.from_array(self.data)
        return arr if order == "TCZYXS" else arr[..., 0]


@requires_full
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


@requires_full
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

REAL_ND2 = Path(os.environ.get("MICROSCOPY_DATA") or os.environ.get("EUBI_TEST_DATA")
                or "C:/Users/oezdemir/Desktop/data/microscopy") \
    / "formats" / "nd2" / "assorted" / "imageJ_test.nd2"


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
