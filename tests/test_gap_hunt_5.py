"""What the fifth gap hunt (2026-11-02) found: joined files that lost their tiles, sizes nobody
measured, and filters that said nothing about a plate converted whole."""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from eubi_bridge.conversion import aggregative_path, dispatcher, legacy_image, metadata_path
from eubi_bridge.conversion import pyramid_path
from eubi_bridge.core import micro_source
from eubi_bridge.core.data_manager import create_ome_xml
from eubi_bridge.utils import path_utils
from eubi_bridge.utils.errors import InputError

pytest.importorskip("micro_reader")
pytest.importorskip("ome_zarr_pyramid")


# -- joined tiled files --------------------------------------------------------------------

def _inputs(as_mosaic=True, scene=0, **readers):
    options = {"mosaic_tile_index": "all", "view_index": "all", "illumination_index": "all",
               "rotation_index": 0, "phase_index": 0, **readers}
    job = SimpleNamespace(readers=SimpleNamespace(as_mosaic=as_mosaic, **options),
                          model_copy=lambda update: job)
    return aggregative_path._Inputs(["a.lif"], job, scene)


def _reads(monkeypatch, answers):
    seen = []

    def micro_array(path, scene, tile, *, as_mosaic=False, multifile=True, **picks):
        seen.append(as_mosaic)
        return answers[as_mosaic]
    monkeypatch.setattr(micro_source, "micro_array", micro_array)
    monkeypatch.setattr(pyramid_path, "_micro_reason", lambda job: "")
    monkeypatch.setattr(aggregative_path._Inputs, "_picks", lambda self, path, tiles: {})
    return seen


def test_joined_files_read_stitched_pixels(monkeypatch):
    seen = _reads(monkeypatch, {True: ("canvas", "")})
    assert _inputs().array("a.lif") == "canvas"
    assert seen == [True]


def test_joined_tiles_that_cannot_be_placed_stay_separate(monkeypatch):
    seen = _reads(monkeypatch, {True: (None, "no positions"), False: ("tile 0", "")})
    assert _inputs().array("a.lif") == "tile 0"
    assert seen == [True, False]


def test_as_mosaic_false_is_kept_when_joining(monkeypatch):
    seen = _reads(monkeypatch, {False: ("tile 0", "")})
    assert _inputs(as_mosaic=False).array("a.lif") == "tile 0"
    assert seen == [False]


def test_joining_a_plate_says_so(monkeypatch):
    plate = SimpleNamespace(indices=[{"well": 0, "field": 0}], close=lambda: None)
    monkeypatch.setattr(pyramid_path, "_open_micro", lambda job: plate)
    with pytest.raises(InputError, match="is a plate"):
        _inputs().image("a.czi")


# -- sizes nobody measured -----------------------------------------------------------------

def test_the_ome_xml_gives_no_size_for_an_axis_without_one():
    ome = create_ome_xml((2, 3, 8, 8), "tzyx", pixel_size_x=0.5, pixel_size_y=0.5,
                         unit_x="micrometer", unit_y="micrometer")
    pixels = ome.images[0].pixels
    assert pixels.physical_size_x == 0.5 and pixels.physical_size_z is None
    assert pixels.time_increment is None


def _store_meta(units, scales):
    axes = "zyx"
    return SimpleNamespace(
        layers={"0": np.zeros((2, 4, 4), "u2")},
        meta=SimpleNamespace(resolution_paths=["0"], axis_order=axes, unit_list=units,
                             get_scale=lambda path: scales, get_translation=lambda path: None,
                             metadata={}, multiscales={"name": "x"}))


def test_an_ome_zarr_axis_without_a_unit_gains_none():
    sizes = legacy_image._ngff_image(_store_meta([None, None, "nanometer"], [1.0, 1.0, 2.0]),
                                     {"series": 0}, None).metadata.pixel_sizes
    assert sizes["z"].unit == "pixel" and sizes["y"].unit == "pixel"
    assert sizes["x"].unit == "nanometer"


# -- tags ----------------------------------------------------------------------------------

def test_a_tag_in_a_folder_above_the_files_does_not_count(tmp_path):
    folder = tmp_path / "Tdata"
    paths = [str(folder / "a_T0.tif"), str(folder / "a_T1.tif"), str(folder / "stray.tif")]
    assert dispatcher._parse_filepaths_with_tags(paths, ["_T"]) == paths[:2]
    assert dispatcher._parse_filepaths_with_tags(paths, ["T"]) == paths[:2]


def test_a_tag_in_a_folder_of_the_input_counts(tmp_path):
    paths = [str(tmp_path / "ch1" / "img.tif"), str(tmp_path / "ch2" / "img.tif")]
    assert dispatcher._parse_filepaths_with_tags(paths, [("ch1", "ch2")]) == paths


# -- plates given in part ------------------------------------------------------------------

def _key(path):
    return path


def _warned(caplog, group, given):
    with caplog.at_level(logging.WARNING):
        path_utils._warn_plate_given_in_part(group, set(given), _key)
    return caplog.text


def test_part_of_a_run_of_nd2_files_warns(caplog):
    group = SimpleNamespace(main="a.nd2", members=("b.nd2", "c.nd2"))
    assert "1 of the 3 files" in _warned(caplog, group, ["a.nd2"])


def test_a_whole_run_does_not_warn(caplog):
    group = SimpleNamespace(main="a.nd2", members=("b.nd2",))
    assert "converted" not in _warned(caplog, group, ["a.nd2", "b.nd2"])


def test_an_index_file_alone_does_not_warn(caplog):
    group = SimpleNamespace(main="Index.xml", members=("a.tiff", "b.tiff"))
    assert "converted" not in _warned(caplog, group, ["Index.xml"])


def test_a_folder_of_planes_follows_the_plate_its_first_plane_opens(tmp_path, monkeypatch):
    files = [str(tmp_path / f"plane_{n}.tif") for n in (1, 2, 3)]
    opens = []

    class Opened:
        path, plates = str(tmp_path / "plate.xml"), [object()]

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def open_(path, plates=True):
        opens.append(path)
        return Opened()
    from micro_reader import _parts
    monkeypatch.setattr(_parts, "main_file", lambda path, *_: path)
    found = path_utils._files_that_open_their_set(files, {}, str, SimpleNamespace(open=open_))
    assert set(found.values()) == {Opened.path} and len(found) == 3
    assert len(opens) == 1                                  # one open per shape of name


def test_a_loose_file_beside_the_planes_stays_itself(tmp_path, monkeypatch):
    # seventh review, F-02: it followed the folder's first plane, or not, by the order of the rows
    plate, loose = str(tmp_path / "plate.xml"), str(tmp_path / "loose.tif")
    planes = [str(tmp_path / f"plane_{n}.tif") for n in (1, 2)]

    class Opened:
        plates = [object()]

        def __init__(self, path):
            self.path = plate if "plane" in path else path

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def open_(path, plates=True):
        return Opened(path)
    from micro_reader import _parts
    monkeypatch.setattr(_parts, "main_file", lambda path, *_: path)
    for files in (planes + [loose], [loose] + planes):
        found = path_utils._files_that_open_their_set(files, {}, str, SimpleNamespace(open=open_))
        assert set(found) == set(planes)


def test_a_file_that_is_a_plate_itself_keeps_its_neighbours(tmp_path, monkeypatch):
    # sixth review, N-05: one-well plates in a folder lost every file after the first
    files = [str(tmp_path / f"{n}.czi") for n in "abc"]

    class Opened:
        plates = [object()]

        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    from micro_reader import _parts
    monkeypatch.setattr(_parts, "main_file", lambda path, *_: path)
    opener = SimpleNamespace(open=lambda path, plates=True: Opened(path))
    assert path_utils._files_that_open_their_set(files, {}, str, opener) == {}


def test_excluded_files_of_a_set_converted_whole_warn(caplog, monkeypatch):
    import micro_reader
    group = SimpleNamespace(main="Index.xml", parts=(), members=("a.tiff", "b.tiff"))
    monkeypatch.setattr(micro_reader, "group_files",
                        lambda paths, plates: SimpleNamespace(groups=[group]))
    with caplog.at_level(logging.WARNING):
        path_utils.warn_excluded_parts(["a.tiff"], ["Index.xml", "b.tiff"])
    assert "converted whole" in caplog.text


# -- metadata updates ----------------------------------------------------------------------

def test_an_rfc8_store_cannot_be_updated_and_says_so(tmp_path):
    store = tmp_path / "x.zarr"
    store.mkdir()
    (store / "zarr.json").write_text(json.dumps({
        "zarr_format": 3, "node_type": "group",
        "attributes": {"ome": {"version": "0.x", "type": "collection", "nodes": []}}}))
    with pytest.raises(InputError, match="RFC 8"):
        metadata_path.update_store(str(store), {"x_scale": 2.0})


def _plate_folder(tmp_path, monkeypatch):
    """A plate's folder: its index and planes, an analysis result (Evaluation1/), and an image
    micro-reader puts in no set (its ``singles``), as in a CV7000 export."""
    import micro_reader
    root = tmp_path / "plate"
    (root / "Evaluation1").mkdir(parents=True)
    index, planes = root / "index.xml", [root / f"plane_{n}.tif" for n in (1, 2)]
    loose, result = root / "unreferenced_auxiliary_image.tif", root / "Evaluation1" / "objects.xml"
    for path in (index, *planes, loose, result):
        path.write_bytes(b"x")

    def group_files(paths, plates=True):
        return SimpleNamespace(groups=[SimpleNamespace(main=str(index), parts=[str(p) for p in planes],
                                                       members=[], format="cv7000")],
                               singles=[str(loose)])

    def no_open(*_, **__):
        raise OSError("not opened in this test")
    monkeypatch.setattr(micro_reader, "group_files", group_files)
    monkeypatch.setattr(micro_reader, "open", no_open)
    return root, index, planes, loose, result


def test_an_image_of_its_own_in_a_plate_folder_stays_itself(tmp_path, monkeypatch):
    # tenth review C-21: it was taken into the plate, and its folder lost it, in silence
    root, index, planes, loose, result = _plate_folder(tmp_path, monkeypatch)
    for given in ([str(index), *map(str, planes), str(loose), str(result)],
                  [str(loose), str(result), *map(str, planes), str(index)],
                  [str(root)]):
        mains = {m for m in path_utils.input_mains(given) if m is not None}
        assert mains == {str(index), str(loose)}, given


def test_the_files_of_one_plate_are_no_input_named_twice(tmp_path, monkeypatch, caplog):
    # tenth review N-04: every plate folder warned "chosen more than once"
    import pandas as pd
    root, index, planes, loose, result = _plate_folder(tmp_path, monkeypatch)
    with caplog.at_level(logging.WARNING):
        path_utils.group_input_rows(pd.DataFrame({"input_path": [str(index), *map(str, planes)]}))
    assert "more than once" not in caplog.text
    with caplog.at_level(logging.WARNING):
        path_utils.group_input_rows(pd.DataFrame({"input_path": [str(index), str(index)]}))
    assert "more than once" in caplog.text


@pytest.mark.parametrize("name, counts, picks, refused", [
    ("sheet.czi", {"series": 1, "view": 2, "illumination": 2}, {}, "2 views and 2 illuminations"),
    ("sheet.czi", {"series": 1, "view": 2, "illumination": 2},
     {"view_index": 0, "illumination_index": 0}, None),                      # the first, as asked
    ("tiles.czi", {"series": 1, "tile": 4}, {}, None),                         # read per tile
    ("mosaic.lif", {"series": 4, "tile": 4}, {}, "4 tiles"),
    ("plain.tif", {"series": 1}, {}, None),
    ("sheet.czi", {"series": 1, "view": 2}, {"force_bioformats": True}, None),   # Bio-Formats' series
])
def test_the_older_readers_refuse_what_they_would_drop(monkeypatch, name, counts, picks, refused):
    # tenth review N-06: --metadata_reader bfio gave view 0 / illumination 0 / tile 0, exit 0
    import micro_reader

    class Opened:
        index_range = counts

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False
    monkeypatch.setattr(micro_reader, "open", lambda path, **_: Opened())
    readers = SimpleNamespace(force_bioformats=False, view_index="all", illumination_index="all",
                              mosaic_tile_index="all")
    for key, value in picks.items():
        setattr(readers, key, value)
    if refused is None:
        legacy_image._refuse_what_is_dropped(name, readers)
    else:
        with pytest.raises(InputError, match=f"has {refused}.*without --metadata_reader bfio"):
            legacy_image._refuse_what_is_dropped(name, readers)


def test_an_image_of_its_own_named_like_a_plane_is_not_folded_into_the_plate(tmp_path, monkeypatch, caplog):
    # review cycle 1, N-03: listed file by file (a folder becomes rows), a TIFF named like a
    # Harmony plane that the index does not list followed its name into the plate
    root, index, planes, loose, result = _plate_folder(tmp_path, monkeypatch)
    from micro_reader import _parts
    monkeypatch.setattr(_parts, "main_file", lambda path, *_: str(index))     # named like a plane
    mains = {m for m in path_utils.input_mains([str(index), *map(str, planes), str(loose)])
             if m is not None}
    assert mains == {str(index), str(loose)}
    assert "belong to" not in caplog.text        # review cycle 2, R2-08: it said it would


def test_a_folder_with_no_image_is_one_line(tmp_path):
    # review cycle 1, N-04: it ended in a 25-line traceback
    import pandas as pd
    (tmp_path / "a.txt").write_text("hi")
    with pytest.raises(InputError, match=r"None of the 1 file\(s\) in .* is an image eubi-bridge reads.*a\.txt"):
        path_utils.group_input_rows(pd.DataFrame({"input_path": [str(tmp_path / "a.txt")] * 1}))


def test_mac_stubs_are_no_input_in_a_list_or_a_table(tmp_path):
    import os
    # review cycle 1, N-06: a GUI batch (a table) or a list kept ._ stubs, and a joined group
    # then failed with "2 files are at one place: x.tiff, ._x.tiff"
    import pandas as pd
    paths = [str(tmp_path / n) for n in ("a_T1.tif", "a_T2.tif", "._a_T1.tif", "._a_T2.tif")]
    kept = path_utils.take_filepaths(paths, output_path=str(tmp_path / "o"))
    assert sorted(map(os.path.basename, kept.input_path)) == ["a_T1.tif", "a_T2.tif"]
    table = path_utils.take_filepaths(pd.DataFrame({"input_path": paths}), output_path=str(tmp_path / "o"))
    assert sorted(map(os.path.basename, table.input_path)) == ["a_T1.tif", "a_T2.tif"]
    with pytest.raises(ValueError, match="all macOS ._ stub files"):
        path_utils.take_filepaths(paths[2:], output_path=str(tmp_path / "o"))
    from eubi_bridge.qt_gui.widgets.sidebar_browser import _is_selectable
    assert not _is_selectable({"name": "._a_T1.tif", "isDirectory": False, "isOmeZarr": False})


def test_bio_formats_tiles_keep_their_plane_positions():
    # review cycle 1, N-10: --force_bioformats wrote a tiled LIF's tiles with no translation
    from ome_types import model
    pixels = model.Pixels(dimension_order="XYCZT", type="uint16", size_x=2, size_y=2, size_z=1,
                          size_c=1, size_t=1,
                          planes=[model.Plane(the_z=0, the_c=0, the_t=0, position_x=5597.99,
                                              position_x_unit="µm", position_y=11310.18,
                                              position_y_unit="µm")])
    position = legacy_image._plane_position(model.Image(pixels=pixels))
    assert {a: (round(q.value, 2), q.unit) for a, q in position.items()} == {
        "x": (5597.99, "micrometer"), "y": (11310.18, "micrometer")}
    assert legacy_image._plane_position(model.Image(pixels=model.Pixels(
        dimension_order="XYCZT", type="uint16", size_x=2, size_y=2, size_z=1, size_c=1, size_t=1))) == {}


def test_a_file_its_set_does_not_list_is_skipped_among_several_inputs():
    # review cycle 1, N-13: a CQ1 measurement folder always exited 1, its calibration images and
    # MIP companion counted as failed tasks; given alone such a file still fails
    calibration = {"status": "error", "error": "UnsupportedFormatError: dark.tif keeps its metadata "
                                              "in ../MeasurementResult.ome.tif, which does not list it"}
    companion = RuntimeError("NotInItsSet: 'MIP.ome.tif' is not part of the set 'MIP.ome.xml' describes")
    dispatcher._report_failures([{"status": "ok"}, calibration, companion])          # no raise
    with pytest.raises(Exception, match="1/1 tasks failed"):
        dispatcher._report_failures([calibration])
    with pytest.raises(Exception, match="1/2 tasks failed"):
        dispatcher._report_failures([calibration, {"status": "error", "error": "ValueError: broken"}])


def test_a_table_rows_output_with_dots_is_the_folder_it_names(tmp_path):
    # review cycle 1, N-20: a CSV row's "sub/../o" failed every row with "Invalid file path"
    import os
    import pandas as pd
    df = path_utils.take_filepaths(pd.DataFrame({"input_path": [str(tmp_path / "a.tif")],
                                                 "output_path": [str(tmp_path / "sub" / ".." / "o")]}))
    assert df.output_path.tolist() == [os.path.abspath(str(tmp_path / "o"))]
    df = path_utils.take_filepaths(pd.DataFrame({"input_path": [str(tmp_path / "a.tif")],
                                                 "output_path": ["s3://bucket/x/../y"]}))
    assert df.output_path.tolist() == ["s3://bucket/x/../y"]                 # a URL as given


def test_channel_wavelengths_reach_the_ome_xml():
    # review cycle 1, N-21: a lambda scan's band wavelengths were only in its channel names
    ome = create_ome_xml(image_shape=(2, 8, 8), axis_order="cyx", dtype="uint16",
                         channel_names=["GFP", "RFP"],
                         channel_wavelengths=[{"emission_wavelength": {"value": 509.0, "unit": "nanometer"},
                                               "excitation_wavelength": {"value": 488.0, "unit": "nanometer"}},
                                              {}])
    gfp, rfp = ome.images[0].pixels.channels
    assert (gfp.emission_wavelength, gfp.excitation_wavelength) == (509.0, 488.0)
    assert str(getattr(gfp.emission_wavelength_unit, "value", gfp.emission_wavelength_unit)) == "nm"
    assert rfp.emission_wavelength is None
    # a list that does not match the channels says nothing (an RGB fold, a channel option)
    ome = create_ome_xml(image_shape=(2, 8, 8), axis_order="cyx", dtype="uint16",
                         channel_wavelengths=[{"emission_wavelength": {"value": 509.0, "unit": "nanometer"}}])
    assert all(c.emission_wavelength is None for c in ome.images[0].pixels.channels)


def test_a_double_star_pattern_takes_every_depth_and_no_store_chunks(tmp_path):
    # review cycle 1, N-22: folder/**/*.tif took the files one folder down only
    import os
    for rel in ("top.tif", "sub/one.tif", "sub/deep/two.tif", "s.zarr/0/c.tif", "note.txt"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")
    found = lambda pattern: sorted(os.path.relpath(p, tmp_path).replace("\\", "/")
                                   for p in path_utils.take_filepaths_from_path(pattern))
    assert found(str(tmp_path / "**" / "*.tif")) == ["sub/deep/two.tif", "sub/one.tif", "top.tif"]
    assert found(str(tmp_path)) == ["note.txt", "s.zarr", "sub/deep/two.tif", "sub/one.tif", "top.tif"]


def test_a_slides_associated_images_carry_their_role(monkeypatch):
    # Bugra 2026-11-04, review cycle 1 N-08: _scene0.._scene4 mixed label and macro photos with
    # the scans; their role (a fixed word micro-reader names them with) is appended
    image = lambda name: SimpleNamespace(name=name)
    assert pyramid_path._slide_role(SimpleNamespace(name_scene=0, image=image("label"))) == "label"
    assert pyramid_path._slide_role(SimpleNamespace(name_scene=4, image=image("Macro"))) == "macro"
    assert pyramid_path._slide_role(SimpleNamespace(name_scene=2, image=image("20x_BF_01"))) is None
    assert pyramid_path._slide_role(SimpleNamespace(name_scene=None, image=image("label"))) is None


def test_an_index_the_file_lacks_is_refused_in_one_line():
    # review cycle 1, N-24: --view_index 3 on a file without views was ignored in silence
    from eubi_bridge.conversion import pyramid_path as pp
    from eubi_bridge.utils.errors import InputError
    with pytest.raises(InputError, match="XYS.lif has no views: --view_index 3"):
        pp._check_positions(3, 1, "view", "XYS.lif", "--view_index")
    with pytest.raises(InputError, match="XYS.lif has no views"):
        pp._check_positions([0, 2], 1, "view", "XYS.lif", "--view_index")
    for fits in (0, "all", [0]):
        pp._check_positions(fits, 1, "view", "XYS.lif", "--view_index")
    pp._check_positions([1, 5], 3, "view", "x.czi", "--view_index")     # 5 skipped, warned
    with pytest.raises(InputError, match="the file has 3 view"):
        pp._check_positions(5, 3, "view", "x.czi", "--view_index")
    with pytest.raises(InputError, match="the file has 12 series"):
        pp._positions(99, 12, "series")


def test_a_series_chosen_in_a_file_read_as_a_plate_is_refused(monkeypatch):
    # review cycle 1, N-23: --series_index 1 on a well-plate CZI wrote the whole plate, no line
    from types import SimpleNamespace as NS
    from eubi_bridge.conversion import pyramid_path as pp
    from eubi_bridge.utils.errors import InputError
    closed = []
    plate = NS(is_plate=True, close=lambda: closed.append(True))
    monkeypatch.setattr(pp, "_open_input", lambda job: (plate, "micro-reader"))
    job = NS(input_path="/d/S=3_CH=2.czi", readers=NS(series_index="1,2"))
    with pytest.raises(InputError, match=r"S=3_CH=2.czi is read as a plate.*--as_plate False"):
        pp._load_input_collection(job)
    assert closed == [True]
    job.readers.series_index = "all"
    f, entries, _ = pp._load_input_collection(job)
    assert entries[0].plate


@pytest.mark.parametrize("option, value", [("view_index", 3), ("illumination_index", "1"),
                                           ("mosaic_tile_index", 9), ("phase_index", 1),
                                           ("rotation_index", 2)])
def test_every_index_option_on_a_plate_is_refused(option, value):
    # review cycle 2, R2-01: --view_index / --illumination_index on a plate were ignored,
    # exit 0; a tile index gave PlateFieldTilesError's advice (as_mosaic=True, the default)
    from types import SimpleNamespace as NS
    from eubi_bridge.conversion import pyramid_path as pp
    from eubi_bridge.utils.errors import InputError
    readers = NS(**{o: d for o, d in pp.INDEX_OPTIONS.items()})
    pp._refuse_indices_on_a_plate(NS(close=lambda: None), NS(readers=readers), "p.czi")
    setattr(readers, option, value)
    with pytest.raises(InputError, match=rf"p.czi is read as a plate.*--{option} {value} .*--as_plate False"):
        pp._refuse_indices_on_a_plate(NS(close=lambda: None), NS(readers=readers), "p.czi")


def test_a_phase_or_rotation_the_file_lacks_is_refused():
    # review cycle 2, R2-02: ignored with exit 0
    from types import SimpleNamespace as NS
    from eubi_bridge.conversion import pyramid_path as pp
    from eubi_bridge.utils.errors import InputError
    keys = [(("series", 0),)]
    assert pp._picked(keys, NS(phase_index=0, rotation_index=0), "x.tif") == keys
    with pytest.raises(InputError, match="x.tif has no rotations: --rotation_index 2"):
        pp._picked(keys, NS(phase_index=0, rotation_index=2), "x.tif")
    keys = [(("series", 0), ("phase", p)) for p in range(2)]
    with pytest.raises(InputError, match=r"No valid phase index in \[5\]: x.tif has 2 phase"):
        pp._picked(keys, NS(phase_index=5, rotation_index=0), "x.tif")


def test_a_side_file_its_set_does_not_list_goes_to_no_other_reader(monkeypatch):
    # review cycle 2, R2-07: a CQ1 calibration TIFF fell back to Bio-Formats (a JVM for
    # nothing on full; on lite NeedsFullEubiBridge failed the run, exit 1)
    import micro_reader
    from types import SimpleNamespace as NS
    from eubi_bridge.conversion import dispatcher, pyramid_path as pp
    from eubi_bridge.utils.errors import InputError

    def refuse(path, job):
        raise micro_reader.UnsupportedFormatError(
            "dark.tif keeps its metadata in ../MeasurementResult.ome.tif, which does not list it")
    monkeypatch.setattr(pp, "_open_tiles", refuse)
    monkeypatch.setattr(pp, "_micro_reason", lambda job, suffix=True: "")
    monkeypatch.setattr("eubi_bridge.core.micro_source.require_micro_reader", lambda: None)
    monkeypatch.setattr("eubi_bridge.core.micro_source.set_read_concurrency", lambda n: None)
    monkeypatch.setattr("eubi_bridge.conversion.conversion_worker._decode_sharers", lambda c: 1)
    job = NS(input_path="/m/Calibration/dark.tif", cluster=NS(decode_threads=1))
    with pytest.raises(InputError, match="which does not list it") as caught:
        pp._open_micro(job)
    assert dispatcher._not_in_its_set(f"InputError: {caught.value}")    # skipped among several


def test_a_list_of_files_takes_includes_and_excludes(tmp_path):
    # review cycle 2, R2-10: a list converted every file it named
    files = [str(tmp_path / n) for n in ("a_nucleus.tif", "b_spots.tif", "c_nucleus_16bit.tif")]
    pick = lambda **kw: [p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]                 # noqa: E731
                         for p in path_utils.take_filepaths(files, **kw).input_path]
    assert pick(includes="*nucleus*") == ["a_nucleus.tif", "c_nucleus_16bit.tif"]
    assert pick(includes="*nucleus*", excludes="*16bit*") == ["a_nucleus.tif"]
    assert pick(excludes="") == pick() == ["a_nucleus.tif", "b_spots.tif", "c_nucleus_16bit.tif"]
    with pytest.raises(InputError, match="None of the 3 listed input path"):
        pick(includes="*nothing*")


@pytest.mark.parametrize("raised, traceback", [("TasksFailedError", False), ("InputError", False),
                                               ("KeyError", True)])
def test_the_gui_batch_log_gets_the_report_not_a_traceback(monkeypatch, raised, traceback):
    # review cycle 2, R2-11: a refused group's report reached the GUI's batch log as a traceback
    import queue
    import sys
    from eubi_bridge import ebridge
    from eubi_bridge.utils import errors
    from eubi_bridge.qt_gui.workers import _conv_subprocess
    error = getattr(errors, raised, None) or KeyError

    class Bridge:
        def to_zarr(self, **kw):
            raise error("1 of 2 aggregative group(s) of the table were not converted")
    monkeypatch.setattr(ebridge, "EuBIBridge", Bridge)
    out, err = sys.stdout, sys.stderr
    results = queue.Queue()
    try:
        _conv_subprocess._conversion_subprocess(
            {"input_path": "t.csv", "output_path": "o", "to_zarr_kwargs": {}}, queue.Queue(), results)
    finally:
        sys.stdout, sys.stderr = out, err
    kind, text = results.get_nowait()
    assert kind == "err" and "1 of 2 aggregative group(s)" in text
    assert ("Traceback" in text) is traceback


def test_a_refused_option_value_is_one_line_per_option(monkeypatch, capsys):
    # review cycle 2, R2-18: --downscale_method gaussian, --x_scale -1 ended in a pydantic traceback
    import sys
    import fire
    from eubi_bridge import cli
    from eubi_bridge.core.config_models import DownscaleConfig

    def refuse(_component):
        DownscaleConfig(downscale_method="gaussian")
    monkeypatch.setattr(fire, "Fire", refuse)
    monkeypatch.setattr(sys, "argv", ["eubi", "to_zarr", "x.czi", "out"])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 1
    err = capsys.readouterr().err
    assert "ERROR: --downscale_method: Input should be" in err and "(given: 'gaussian')" in err
    assert "Traceback" not in err and "pydantic" not in err
