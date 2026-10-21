"""``readers.pixel_reader='micro'``: pixels from micro-reader, everything else
unchanged (eubi_bridge/core/micro_source.py).

The mapping from eubi-bridge's selection (scene rank, tile, as_mosaic) to a
micro-reader image, the older readers handing micro-reader's pixels to the conversion
(``LegacyFile``), and conversions end to end against the standard reader.
"""
from __future__ import annotations

import pickle

import numpy as np
import pytest
from tests.conftest import requires_full  # noqa: E402

pytest.importorskip("micro_reader")
tifffile = pytest.importorskip("tifffile")

from eubi_bridge.core import micro_source  # noqa: E402
from eubi_bridge.core.micro_source import micro_array  # noqa: E402


def _reader_source(arr):
    """The reader's own source behind a DynamicArray: ``dyna_zarr.from_array`` keeps it
    one level down, in its adapter (``arr._source._source``)."""
    src = getattr(arr, "_source", None)
    return getattr(src, "_source", src)


def _legacy_images(path, **kwargs):
    """Every scene's image as the older readers hand it to the conversion
    (conversion/legacy_image.py: ``LegacyFile``, the reader behind it kept open)."""
    from eubi_bridge.conversion.legacy_image import LegacyFile
    from eubi_bridge.core.config_models import ConversionJob
    job = ConversionJob.from_kwargs(str(path), "out", {"series_index": "all", **kwargs})
    f = LegacyFile(job)
    return [f.image(index) for index in f.indices]

RNG = np.random.default_rng(5)


@pytest.fixture(autouse=True)
def _fresh_files():
    yield
    micro_source.close_all()


def _two_series(tmp_path):
    """A TIFF with two series (scenes): Z C Y X and C Y X."""
    a = RNG.integers(0, 4000, (3, 2, 20, 24)).astype("u2")
    b = RNG.integers(0, 4000, (2, 16, 18)).astype("u2")
    path = str(tmp_path / "two.tif")
    with tifffile.TiffWriter(path) as tw:
        tw.write(a, metadata={"axes": "ZCYX"}, photometric="minisblack")
        tw.write(b, metadata={"axes": "CYX"}, photometric="minisblack")
    return path, a, b


def _two_series_ome(tmp_path):
    """Like _two_series, as an OME-TIFF: Bio-Formats (the metadata) also sees
    two images.  (It sees 8 in tifffile's own multi-series TIFF.)"""
    a = RNG.integers(0, 4000, (3, 2, 20, 24)).astype("u2")
    b = RNG.integers(0, 4000, (2, 16, 18)).astype("u2")
    path = str(tmp_path / "two.ome.tif")
    with tifffile.TiffWriter(path, ome=True) as tw:
        tw.write(a, metadata={"axes": "ZCYX"}, photometric="minisblack")
        tw.write(b, metadata={"axes": "CYX"}, photometric="minisblack")
    return path, a, b


def test_scene_rank_maps_to_an_image_as_tczyx(tmp_path):
    path, a, b = _two_series(tmp_path)
    arr, reason = micro_array(path, 0, None)
    assert reason == "" and arr.shape == (1, 2, 3, 20, 24)
    np.testing.assert_array_equal(np.asarray(arr), a.transpose(1, 0, 2, 3)[None])
    arr, _ = micro_array(path, 1, None)
    np.testing.assert_array_equal(np.asarray(arr), b[None, :, None])


def test_tile_0_on_an_untiled_file_is_the_whole_image(tmp_path):
    """eubi-bridge passes mosaic_tile_index=0 by default; set_tile is a no-op
    on a file without tiles."""
    path, a, _ = _two_series(tmp_path)
    arr, reason = micro_array(path, 0, 0)
    assert reason == "" and arr.shape == (1, 2, 3, 20, 24)
    arr, reason = micro_array(path, 0, 1)
    assert arr is None and "no tiles" in reason


def test_tiles_follow_as_mosaic(tmp_path):
    data = RNG.integers(0, 4000, (2, 12, 14)).astype("u2")          # R Y X: 2 tiles
    path = str(tmp_path / "tiles.tif")
    tifffile.imwrite(path, data, metadata={"axes": "RYX"}, photometric="minisblack")
    arr, _ = micro_array(path, 0, 1)
    np.testing.assert_array_equal(np.asarray(arr)[0, 0, 0], data[1])
    arr, reason = micro_array(path, 0, 0, as_mosaic=True)       # TIFF tiles: no composition
    assert arr is None and "mosaic" in reason


def test_out_of_range_and_unreadable_files_give_reasons(tmp_path):
    path, _, _ = _two_series(tmp_path)
    assert "2 series" in micro_array(path, 5, None)[1]
    other = tmp_path / "notes.txt"
    other.write_text("not an image")
    arr, reason = micro_array(str(other), 0, None)
    assert arr is None and "cannot read" in reason


def test_micro_source_survives_pickling_and_deep_copy(tmp_path):
    import copy
    path, a, _ = _two_series(tmp_path)
    arr, _ = micro_array(path, 0, None)
    source = _reader_source(arr)
    source[0, 0, 0, 0:2, 0:2]                                     # opened in this process
    micro_source.close_all()                                      # clones reopen it
    for clone in (pickle.loads(pickle.dumps(source)), copy.deepcopy(source)):
        np.testing.assert_array_equal(clone[0, 1, 2, 3:7, 4:9], a[2, 1, 3:7, 4:9])


def test_the_option_is_validated_and_reaches_the_job():
    from pydantic import ValidationError

    from eubi_bridge.core.config_models import ConversionJob, ReaderConfig
    assert ReaderConfig().pixel_reader == "micro"          # the default (2026-10-04)
    with pytest.raises(ValidationError):
        ReaderConfig(pixel_reader="fast")
    job = ConversionJob.from_kwargs("/in.tif", "/out", {"pixel_reader": "standard"})
    assert job.readers.pixel_reader == "standard"


def test_conversion_writes_the_same_pixels_with_both_readers(tmp_path):
    import zarr

    from eubi_bridge.ebridge import EuBIBridge
    data = RNG.integers(0, 4000, (3, 2, 40, 48)).astype("u2")     # Z C Y X
    source = tmp_path / "img.tif"
    tifffile.imwrite(source, data, imagej=True, metadata={"axes": "ZCYX"})
    written = {}
    for reader in ("standard", "micro"):
        out = tmp_path / reader
        EuBIBridge().to_zarr(str(source), str(out), pixel_reader=reader, verbose=False)
        (store,) = out.glob("*.zarr")
        written[reader] = np.asarray(zarr.open_group(str(store), mode="r")["0"])
    np.testing.assert_array_equal(written["micro"], written["standard"])
    assert written["micro"].size == data.size


@requires_full
@pytest.mark.parametrize("reader,expect_micro", [("micro", True), ("standard", False)])
def test_loaded_scenes_hold_micro_reader_arrays(tmp_path, reader, expect_micro):
    """The layer the conversion reads from: with pixel_reader='micro' each scene's
    array is a micro-reader source, with 'standard' it is not."""
    from eubi_bridge.core.micro_source import MicroSource
    from eubi_bridge.utils.jvm_manager import soft_start_jvm
    soft_start_jvm()                         # bfio metadata, as a conversion worker does
    path, a, _ = _two_series_ome(tmp_path)
    scenes = _legacy_images(path, metadata_reader="bfio", pixel_reader=reader)
    assert len(scenes) == 2
    for image in scenes:
        assert isinstance(_reader_source(image.array), MicroSource) == expect_micro
    np.testing.assert_array_equal(np.asarray(scenes[0].array), a.transpose(1, 0, 2, 3)[None])


# -- the GUI and its config mappings ------------------------------------------------

def test_gui_config_maps_both_ways():
    """The tick is a reader setting, the concurrency a cluster one, in the
    GUI as in the backend (readers.pixel_reader, cluster.decode_threads)."""
    from eubi_bridge.qt_gui.server.config_manager import _config_to_react, _react_to_config
    from eubi_bridge.qt_gui.workers.conversion_worker import _build_kwargs
    gui = {"cluster": {"decodeThreads": 6}, "reader": {"useMicroReader": True}}
    backend = _react_to_config(gui)
    assert backend["readers"]["pixel_reader"] == "micro"
    assert backend["cluster"]["decode_threads"] == 6
    back = _config_to_react(backend)
    assert back["reader"]["useMicroReader"] is True
    assert back["cluster"]["decodeThreads"] == 6
    kwargs = _build_kwargs(gui)
    assert kwargs["pixel_reader"] == "micro" and kwargs["decode_threads"] == 6
    unset = _react_to_config({"cluster": {}, "reader": {}})
    assert unset["readers"]["pixel_reader"] == "micro"          # the default (2026-10-04)
    off = _react_to_config({"cluster": {}, "reader": {"useMicroReader": False}})
    assert off["readers"]["pixel_reader"] == "standard"


@pytest.fixture
def page():
    from tests.conftest import qt_available
    if not qt_available():
        pytest.skip("PyQt6 unavailable or no usable Qt platform plugin")
    from PyQt6.QtWidgets import QApplication
    from eubi_bridge.qt_gui.pages.convert_page import ConvertPage
    app = QApplication.instance() or QApplication([])
    widget = ConvertPage()
    yield widget
    widget.deleteLater()
    app.processEvents()


def test_gui_tick_enables_concurrency_and_is_saved(page):
    # ticked by default (2026-10-04), with its concurrency live from the start
    assert page._use_micro_reader.isChecked()
    assert page._decode_threads.isEnabled()
    page._use_micro_reader.setChecked(False)
    assert not page._decode_threads.isEnabled()
    page._use_micro_reader.setChecked(True)
    assert page._decode_threads.isEnabled()
    page._decode_threads.setValue(8)
    config = page._ui_to_config()
    assert config["reader"]["useMicroReader"] is True
    assert config["cluster"]["decodeThreads"] == 8
    page._use_micro_reader.setChecked(False)
    assert page._ui_to_config()["reader"]["useMicroReader"] is False
    assert not page._decode_threads.isEnabled()


def test_gui_loads_the_tick_from_a_saved_config(page):
    page._load_config_to_ui({"cluster": {"decodeThreads": 5},
                             "reader": {"useMicroReader": True}})
    assert page._use_micro_reader.isChecked()
    assert page._decode_threads.value() == 5
    assert page._decode_threads.isEnabled()


def test_decode_threads_sets_the_decode_pool():
    import micro_reader
    from eubi_bridge.core.micro_source import set_read_concurrency
    old = micro_reader.decode_threads()
    try:
        set_read_concurrency(3)
        assert micro_reader.decode_threads() == 3
    finally:
        micro_reader.set_decode_threads(old)


def test_open_files_are_bounded_and_reopened(tmp_path, monkeypatch):
    """An aggregative input of thousands of files must not hold them all open
    ("Too many open files" at ~500 of 2347): least recently used files close,
    and reading one again reopens it."""
    monkeypatch.setattr(micro_source, "MAX_OPEN_FILES", 4)
    arrays, data = [], []
    for k in range(10):
        d = RNG.integers(0, 4000, (2, 16, 18)).astype("u2")
        path = str(tmp_path / f"f{k}.tif")
        tifffile.imwrite(path, d, photometric="minisblack", metadata={"axes": "CYX"})
        arr, reason = micro_array(path, 0, None)
        assert reason == ""
        arrays.append(arr)
        data.append(d)
        assert len(micro_source._files) <= 4
    for arr, d in zip(arrays, data):                              # all readable, reopened
        np.testing.assert_array_equal(np.asarray(arr)[0, :, 0], d)
        assert len(micro_source._files) <= 4


def test_a_file_in_use_is_not_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(micro_source, "MAX_OPEN_FILES", 1)
    paths = []
    for k in range(3):
        path = str(tmp_path / f"g{k}.tif")
        tifffile.imwrite(path, np.full((8, 9), k, "u1"))
        paths.append(path)
    with micro_source._using(paths[0], "separate") as held:
        for p in paths[1:]:
            with micro_source._using(p, "separate"):
                pass
        assert held.file.images[0][...][0, 0] == 0                # still open under its user
    assert len(micro_source._files) <= 1


def test_aggregative_conversion_reads_with_micro_reader(tmp_path, monkeypatch):
    """Aggregative mode forwarded no reader settings: pixel_reader must reach
    each file's array (micro_source.micro_array), and the z stack must match the
    standard reader's."""
    import zarr

    from eubi_bridge.core import micro_source as ms
    from eubi_bridge.ebridge import EuBIBridge
    folder = tmp_path / "planes"
    folder.mkdir()
    stack = RNG.integers(0, 255, (5, 40, 48)).astype("u1")
    for z, plane in enumerate(stack):
        tifffile.imwrite(folder / f"cell_{z:04d}.tif", plane)
    opened = []
    real = ms.MicroReader.open.__func__
    real_array = ms.micro_array

    def counting(cls, *args, **kwargs):
        out = real(cls, *args, **kwargs)
        opened.append(out is not None)
        return out

    def counting_array(*args, **kwargs):
        out = real_array(*args, **kwargs)
        opened.append(out[0] is not None)
        return out

    monkeypatch.setattr(ms.MicroReader, "open", classmethod(counting))
    monkeypatch.setattr(ms, "micro_array", counting_array)
    written = {}
    for reader in ("standard", "micro"):
        out = tmp_path / reader
        EuBIBridge().to_zarr(str(folder), str(out), concatenation_axes="z", z_tag="cell_",
                             pixel_reader=reader, use_threading=True, max_processes=1,
                             verbose=False)
        (store,) = out.glob("*.zarr")
        written[reader] = np.asarray(zarr.open_group(str(store), mode="r")["0"])
    assert opened.count(True) >= 5                                # every plane, micro-reader
    np.testing.assert_array_equal(written["micro"], written["standard"])
    np.testing.assert_array_equal(np.squeeze(written["micro"]), stack)


# -- micro-reader as the reader: no standard reader opened ---------------------------

def _read(path, **kwargs):
    import asyncio

    from eubi_bridge.core.readers import read_single_image
    return asyncio.run(read_single_image(path, **kwargs))


@requires_full
def test_micro_reader_replaces_the_standard_reader(tmp_path):
    from eubi_bridge.core.micro_source import MicroReader
    path, a, b = _two_series(tmp_path)
    reader = _read(path, pixel_reader="micro")
    assert isinstance(reader, MicroReader) and reader.n_scenes == 2
    np.testing.assert_array_equal(np.asarray(reader.get_image_dask_data()),
                                  a.transpose(1, 0, 2, 3)[None])
    reader.set_scene(1)
    assert reader.series_path.endswith("two.tif_1")
    np.testing.assert_array_equal(np.asarray(reader.get_image_dask_data()), b[None, :, None])
    assert not isinstance(_read(path), MicroReader)               # standard unless asked


def test_rgb_samples_become_channels_with_their_layout(tmp_path):
    rgb = RNG.integers(0, 255, (2, 12, 14, 3)).astype("u1")      # C Y X S
    path = str(tmp_path / "rgb.tif")
    tifffile.imwrite(path, rgb, photometric="rgb", planarconfig="contig",
                     metadata={"axes": "CYXS"})
    reader = _read(path, pixel_reader="micro")
    assert reader.sample_layout == (2, 3)
    arr = np.asarray(reader.get_image_dask_data())
    np.testing.assert_array_equal(arr[0, :, 0], rgb.transpose(0, 3, 1, 2).reshape(6, 12, 14))


@requires_full
def test_tiles_outside_czi_keep_the_standard_reader(tmp_path):
    from eubi_bridge.core.micro_source import MicroReader
    path = str(tmp_path / "tiles.tif")
    tifffile.imwrite(path, np.zeros((2, 12, 14), "u2"), metadata={"axes": "RYX"},
                     photometric="minisblack")
    assert not isinstance(_read(path, pixel_reader="micro"), MicroReader)


def _ome(*images):
    from types import SimpleNamespace as NS
    return NS(images=[NS(pixels=NS(size_t=t, size_c=c, size_z=z, size_y=y, size_x=x,
                                   type=ptype)) for t, c, z, y, x, ptype in images])


def test_metadata_mismatch_compares_scenes_shapes_and_types(tmp_path):
    from eubi_bridge.core.micro_source import metadata_mismatch
    path, _, _ = _two_series(tmp_path)                         # Z C Y X, C Y X; uint16
    good = _ome((1, 2, 3, 20, 24, "uint16"), (1, 2, 1, 16, 18, "uint16"))
    assert metadata_mismatch(_read(path, pixel_reader="micro"), good) == ""
    one = _ome((1, 2, 3, 20, 24, "uint16"))
    assert "2 images" in metadata_mismatch(_read(path, pixel_reader="micro"), one)
    shape = _ome((1, 2, 3, 20, 24, "uint16"), (1, 2, 1, 16, 99, "uint16"))
    assert "matches scene 1" in metadata_mismatch(_read(path, pixel_reader="micro"), shape)
    dtype = _ome((1, 2, 3, 20, 24, "uint8"), (1, 2, 1, 16, 18, "uint16"))
    assert "matches scene 0" in metadata_mismatch(_read(path, pixel_reader="micro"), dtype)
    swapped = _ome((1, 2, 1, 16, 18, "uint16"), (1, 2, 3, 20, 24, "uint16"))
    assert metadata_mismatch(_read(path, pixel_reader="micro"), swapped) == ""   # any order
    assert metadata_mismatch(_read(path, pixel_reader="micro"), None) == ""


def test_rgb_channel_count_may_or_may_not_include_samples(tmp_path):
    from eubi_bridge.core.micro_source import metadata_mismatch
    rgb = np.zeros((2, 12, 14, 3), "u1")
    path = str(tmp_path / "rgb.tif")
    tifffile.imwrite(path, rgb, photometric="rgb", metadata={"axes": "CYXS"})
    for size_c in (6, 2):                                     # C x S, or C
        assert metadata_mismatch(_read(path, pixel_reader="micro"),
                                 _ome((1, size_c, 1, 12, 14, "uint8"))) == ""
    assert "matches scene 0" in metadata_mismatch(_read(path, pixel_reader="micro"),
                                                  _ome((1, 5, 1, 12, 14, "uint8")))


@requires_full
def test_scenes_load_without_opening_the_standard_reader(tmp_path, monkeypatch):
    """The layer the conversion reads from, with the standard TIFF reader made to
    fail: pixel_reader='micro' must not need it."""
    from eubi_bridge.core import tiff_reader
    from eubi_bridge.core.micro_source import MicroSource
    from eubi_bridge.utils.jvm_manager import soft_start_jvm
    soft_start_jvm()

    from eubi_bridge.core import pff_reader

    def no_standard_reader(*args, **kwargs):
        raise AssertionError("the standard reader was opened")

    monkeypatch.setattr(tiff_reader, "read_tiff_image", no_standard_reader)
    monkeypatch.setattr(pff_reader, "read_pff", no_standard_reader)
    path, a, _ = _two_series_ome(tmp_path)
    scenes = _legacy_images(path, metadata_reader="bfio", pixel_reader="micro")
    assert len(scenes) == 2
    assert all(isinstance(_reader_source(image.array), MicroSource) for image in scenes)
    np.testing.assert_array_equal(np.asarray(scenes[0].array), a.transpose(1, 0, 2, 3)[None])


@requires_full
def test_disagreement_with_the_metadata_keeps_the_standard_reader(tmp_path):
    """tifffile's own multi-series TIFF: micro-reader sees 2 scenes, Bio-Formats
    (the metadata) 8.  Who is right is not micro-reader's call: the file is
    read the standard way, as before."""
    from eubi_bridge.core.micro_source import MicroSource
    from eubi_bridge.utils.jvm_manager import soft_start_jvm
    soft_start_jvm()
    path, _, _ = _two_series(tmp_path)
    scene = _legacy_images(path, metadata_reader="bfio", pixel_reader="micro")[0]
    assert not isinstance(_reader_source(scene.array), MicroSource)


def test_views_count_as_images_against_bio_formats(tmp_path):
    """Bio-Formats lists a CZI's views as separate images: 1 scene x 2 views
    must pass against 2 metadata images (Bugra: 25 scenes x 2 views = 50)."""
    from tests._czi_writer import raw_payload, write_czi

    from eubi_bridge.core.micro_source import metadata_mismatch
    path = str(tmp_path / "views.czi")
    tiles = []
    for v in range(2):
        for c in range(2):
            tiles.append(dict(scene=0, plane={"C": c, "V": v}, x=0, y=0, height=12, width=16,
                              dtype="u2", compression=0,
                              payload=raw_payload(np.full((12, 16), v, "u2"))))
    write_czi(path, tiles)
    reader = _read(path, pixel_reader="micro", view_index="all")
    assert reader.n_scenes == 1 and reader.n_views == 2
    two = _ome((1, 2, 1, 12, 16, "uint16"), (1, 2, 1, 12, 16, "uint16"))
    assert metadata_mismatch(reader, two) == ""
    assert "2 images" in metadata_mismatch(reader, _ome((1, 2, 1, 12, 16, "uint16")))


def test_czi_resolution_levels_count_against_bio_formats(tmp_path):
    """Bio-Formats lists a CZI's stored resolution levels as images when its
    own collapse does not apply (S=2_2x2: 2 scenes x 2 levels = 4 images)."""
    from tests._czi_writer import raw_payload, write_czi

    from eubi_bridge.core.micro_source import metadata_mismatch
    path = str(tmp_path / "pyr.czi")
    tiles = [dict(scene=0, mosaic=m, plane={"C": 0}, x=32 * (m % 2), y=32 * (m // 2),
                  height=32, width=32, dtype="u2", compression=0,
                  payload=raw_payload(np.zeros((32, 32), "u2"))) for m in range(4)]
    tiles.append(dict(scene=0, plane={"C": 0}, x=0, y=0, height=64, width=64, stored=(32, 32),
                      pyramid=2, dtype="u2", compression=0,
                      payload=raw_payload(np.zeros((32, 32), "u2"))))
    write_czi(path, tiles)
    reader = _read(path, pixel_reader="micro")                   # separate tiles
    with_level = _ome((1, 1, 1, 64, 64, "uint16"), (1, 1, 1, 32, 32, "uint16"))
    assert metadata_mismatch(reader, with_level) == ""
    three = _ome(*[(1, 1, 1, 64, 64, "uint16")] * 3)
    assert "1 images" in metadata_mismatch(reader, three)


# -- keep_existing_resolutions with micro-reader --------------------------------------

def _pyramid_ome_tiff(path):
    """An OME-TIFF with 2 stored lower levels, altered so that a computed
    level can never pass for a copied one."""
    data = RNG.integers(0, 4000, (2, 256, 320)).astype("u2")
    levels = [data, data[:, ::2, ::2] + 1, data[:, ::4, ::4] + 2]
    with tifffile.TiffWriter(path, ome=True) as tw:
        tw.write(levels[0], subifds=2, metadata={"axes": "CYX"}, photometric="minisblack",
                 tile=(64, 64))
        for level in levels[1:]:
            tw.write(level, subfiletype=1, photometric="minisblack", tile=(64, 64))
    return levels


def _written_levels(out):
    import json

    import zarr
    (store,) = out.glob("*.zarr")
    group = zarr.open_group(str(store), mode="r")
    names = sorted(group.array_keys(), key=int)
    datasets = json.loads((store / ".zattrs").read_text())["multiscales"][0]["datasets"]
    scales = [d["coordinateTransformations"][0]["scale"] for d in datasets]
    return [np.squeeze(np.asarray(group[n])) for n in names], scales


def test_keep_existing_resolutions_copies_the_stored_levels(tmp_path):
    from eubi_bridge.ebridge import EuBIBridge
    source = tmp_path / "pyr.ome.tif"
    levels = _pyramid_ome_tiff(str(source))
    out = tmp_path / "out"
    EuBIBridge().to_zarr(str(source), str(out), pixel_reader="micro",
                         keep_existing_resolutions=True, verbose=False,
                         use_threading=True, max_processes=1)
    written, scales = _written_levels(out)
    assert len(written) == 3
    for mine, stored in zip(written, levels):
        np.testing.assert_array_equal(mine, stored)
    assert [s[-2:] for s in scales] == [[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]]


def test_without_keep_existing_the_pyramid_is_computed(tmp_path):
    from eubi_bridge.ebridge import EuBIBridge
    source = tmp_path / "pyr.ome.tif"
    levels = _pyramid_ome_tiff(str(source))
    out = tmp_path / "out"
    EuBIBridge().to_zarr(str(source), str(out), pixel_reader="micro", verbose=False,
                         use_threading=True, max_processes=1)
    written, _ = _written_levels(out)
    np.testing.assert_array_equal(written[0], levels[0])
    assert not np.array_equal(written[1], levels[1])           # computed, not the stored level


# -- Imaris: micro-reader through IMSImageMeta -----------------------------------------

@requires_full
def test_imaris_reads_with_micro_reader_and_keeps_its_levels(tmp_path):
    """Imaris has its own route (IMSImageMeta, native metadata), which used to
    bypass pixel_reader: the base array and the kept levels must come from
    micro-reader and equal the file's."""
    pytest.importorskip("h5py")
    from tests._ims_writer import write_pyramid_ims

    from eubi_bridge.core.micro_source import MicroSource
    data = RNG.integers(0, 4000, (1, 2, 5, 21, 27)).astype("u2")
    path = str(tmp_path / "pyr.ims")
    write_pyramid_ims(path, data)

    (image,) = _legacy_images(path, metadata_reader="bfio", pixel_reader="micro",
                              keep_existing_resolutions=True)
    levels = [level.array for level in image.resolutions]
    assert len(levels) == 2 and all(isinstance(_reader_source(lv), MicroSource) for lv in levels)
    np.testing.assert_array_equal(np.asarray(levels[0]), data)
    np.testing.assert_array_equal(np.asarray(levels[1]), data[:, :, :, ::2, ::2])
    (standard,) = _legacy_images(path, metadata_reader="bfio", pixel_reader="standard")
    assert not isinstance(_reader_source(standard.array), MicroSource)
