"""Tags, the one reader of them (fileset_io.read_tag), and what a planned run does with them.

Generated names: a grid of files with hostile text around the tags (words holding the tag's
letters, numbers beside it, tag-like folders above and below) must group as the grid says,
in the order of the numbers, whatever the paths' own order, in the planner and in a worker."""
import itertools
import os
import random

import numpy as np
import pytest
import tifffile

from eubi_bridge.conversion.dispatcher import _build_aggregative_plan, _parse_filepaths_with_tags
from eubi_bridge.conversion.fileset_io import FileSet, find_tag, read_tag, reduce_paths_flexible
from eubi_bridge.utils.errors import InputError

DUMMY = (1, 1, 1, 1, 1)


def _plan(paths, axes, **tags):
    tags = {"time_tag": None, "channel_tag": None, "z_tag": None, **tags}
    return _build_aggregative_plan(paths, axes, os.path.commonpath([os.path.dirname(p) for p in paths]),
                                   "/out", **tags)


def _order(fs, axis):
    """The files of the run in the place the slices give them along *axis*."""
    return sorted(fs.slice_dict, key=lambda path: fs.slice_dict[path][axis].start)


# -- the reader ------------------------------------------------------------------------------

@pytest.mark.parametrize("text, tag, number", [
    ("batch3_ch1_t1.tif", "ch", "1"),          # ch3 is inside a word
    ("batch3_ch2_t1.tif", "t", "1"),
    ("C1_T5run_T002.tif", "T", "002"),         # T5 runs on into a word
    ("Dir_T9/a_T1.tif", "T", "1"),             # the file speaks before its folder
    ("s1t2c3.tif", "t", "2"),                  # digits around a tag are no word
    ("imgT0007.tif", "T", "0007"),             # only inside a word: still read
    ("frames/012/a.tif", "frames", "012"),
    ("frames\\012\\a.tif", "frames/", "012"),
    ("frames\\012\\a.tif", "frames\\", "012"),
    ("Thumbnail.tif", "T", None),
    ("batch.tif", "ch", None),
    ("exp_h1.h5", "h", "1"),                   # the extension holds no tag
    ("a_z0.5.tif", "z", "0"),                  # the digits right after the tag, nothing more
    ("x_T00h15m30s.tif", "T", "00"),
    ("s_C001Z001.tif", "C", "001"),
])
def test_where_a_tag_is_read(text, tag, number):
    match = find_tag(text, tag)
    assert (match.group(1) if match else None) == number


def test_a_file_with_the_tag_only_inside_a_word_has_not_got_it_beside_real_ones():
    paths = ["/d/img_t1.tif", "/d/img_t2.tif", "/d/Test1.tif"]
    assert read_tag(paths, "t") == {paths[0]: (1,), paths[1]: (2,)}
    assert read_tag(["/d/Test1.tif", "/d/Test2.tif"], "t") == {"/d/Test1.tif": (1,),
                                                               "/d/Test2.tif": (2,)}


def test_named_tags_take_the_word_and_the_file_name_first():
    paths = ["/d/a_egfp.tif", "/d/a_gfp.tif", "/d/gfp/b_rfp.tif"]
    assert read_tag(paths, ("gfp", "egfp", "rfp")) == {
        paths[0]: "egfp", paths[1]: "gfp", paths[2]: "rfp"}


@pytest.mark.parametrize("tags, name, want", [
    (("GD2", "GM2_GD2"), "x_GM2_GD2.tif", "GM2_GD2"),
    (("GD2", "GM2", "GM2_GD2"), "x_GM2_GD2.tif", "GM2_GD2"),
    (("CFP", "YFP", "YFP_CFP"), "x_YFP_CFP.tif", "YFP_CFP"),
    (("GD2", "GM2_GD2"), "x_GD2.tif", "GD2"),
])
def test_a_tag_inside_another_is_not_read_there(tags, name, want):
    assert read_tag([f"/d/{name}", "/d/other.tif"], tags)[f"/d/{name}"] == want


def test_the_most_unique_name_is_searched_first_and_the_join_keeps_the_order_given():
    paths = ["/d/GD2_x_GM2_GD2.tif", "/d/y_GD2.tif"]              # GD2 also on its own
    assert read_tag(paths, ("GD2", "GM2_GD2")) == {paths[0]: "GM2_GD2", paths[1]: "GD2"}
    fs = FileSet(paths, shapes=[DUMMY] * 2, axis_tag1="GD2,GM2_GD2")
    fs.check_places([1])
    fs.concatenate_along(1)
    assert _order(fs, 1) == [paths[1], paths[0]]                    # GD2 first, as given
    fs = FileSet(["/d/x_GM2_GD2.tif", "/d/x_GD2.tif"], shapes=[DUMMY] * 2, axis_tag1="GD2,GM2_GD2")
    fs.concatenate_along(1)
    assert set(fs.path_dict.values()) == {"/d/x_GD2GM2_GD2_cset.tif"}  # named as any named tags
    with pytest.raises(InputError, match="two of the tags"):        # names apart: no way to tell
        read_tag(["/d/x_GD2_GM2.tif", "/d/y_GD2.tif"], ("GD2", "GM2"))


@pytest.mark.parametrize("names, axis, tag", [
    (["exp_20240116_T01.tif", "exp_20240116_T02.tif", "exp_20240115_T23.tif"], 0, "T"),
    (["d2/a_T1.tif", "d2/a_T2.tif", "d1/a_T8.tif", "d1/a_T9.tif"], 0, "T"),
    (["run7_T1.tif", "run3_T2.tif"], 0, "T"),
    (["01d00h00m.tif", "00d20h00m.tif", "00d22h00m.tif"], 0, "d"),
])
def test_names_that_count_in_more_than_one_number(names, axis, tag):
    paths = [f"/data/{name}" for name in names]
    fs = FileSet(list(reversed(paths)), shapes=[DUMMY] * len(paths), **{f"axis_tag{axis}": tag})
    fs.check_places([axis])
    fs.concatenate_along(axis)
    assert _order(fs, axis) == paths


def test_numbers_ahead_of_the_tag_that_would_order_otherwise_are_said(caplog):
    import logging
    paths = ["/d/exp_20240115_T23.tif", "/d/exp_20240116_T01.tif"]
    fs = FileSet(paths, shapes=[DUMMY] * 2, axis_tag0="T")
    with caplog.at_level(logging.WARNING):
        fs.check_places([0])
    assert "joined in the order of T, from exp_20240116_T01.tif" in caplog.text
    for paths, axes, tags in (
            (["/d/ImageA_T1.tif", "/d/ImageB_T2.tif"], [0], dict(axis_tag0="T")),
            ([f"/d/Channel{c}-T000{t}.tif" for c in (1, 2) for t in (1, 2)], [0, 1],
             dict(axis_tag0="T", axis_tag1="Channel"))):
        caplog.clear()
        fs = FileSet(paths, shapes=[DUMMY] * len(paths), **tags)
        with caplog.at_level(logging.WARNING):
            fs.check_places(axes)
        assert "order" not in caplog.text


@pytest.mark.parametrize("tag, order", [("Blue,Red,Green", ["Blue", "Red", "Green"]),
                                        ("Green,Blue,Red", ["Green", "Blue", "Red"])])
def test_named_tags_join_in_the_order_given_and_name_short(tag, order):
    paths = ["/d/aaa_run1_Red_0.01s.tif", "/d/zzz_run9_Blue_0.09s.tif", "/d/mmm_run5_Green_0.05s.tif"]
    fs = FileSet(paths, shapes=[DUMMY] * 3, axis_tag1=tag)
    fs.check_places([1])
    fs.concatenate_along(1)
    joined = _order(fs, 1)
    assert [p.split("_")[2] for p in joined] == order
    first = os.path.basename(joined[0]).split("_")
    assert set(fs.path_dict.values()) == {
        f"/d/{first[0]}_{first[1]}_{''.join(order)}_cset_{first[3]}"}


def test_named_channel_folders_keep_their_documented_name():
    paths = [f"/d/{c}/T000{t}.tif" for c in ("Blue", "Red") for t in (1, 2)]
    fs = FileSet(paths, shapes=[DUMMY] * 4, axis_tag0="T", axis_tag1="Blue,Red")
    fs.check_places([0, 1])
    fs.concatenate_along(0)
    fs.concatenate_along(1)
    assert set(fs.path_dict.values()) == {"/d/BlueRed_cset/T_tset.tif"}


def test_a_file_named_with_two_tags_is_refused():
    with pytest.raises(InputError, match="two of the tags"):
        read_tag(["/d/a_gfp_rfp.tif", "/d/b_gfp.tif"], ("gfp", "rfp"))


def test_only_the_tag_read_is_replaced_in_the_name():
    assert reduce_paths_flexible(["batch3_ch1_t1.tif", "batch3_ch2_t1.tif"], "ch", "_cset") \
        == "batch3_ch_cset_t1.tif"


def test_a_tag_written_with_a_separator_selects_files_on_every_system():
    paths = ["/d/frames/000/a.tif", "/d/frames/001/a.tif"]
    assert _parse_filepaths_with_tags(paths, ["frames/", None]) == paths
    assert _parse_filepaths_with_tags([p.replace("/", "\\") for p in paths], ["frames/", None])


def test_a_file_must_have_every_tag():
    paths = ["/d/a_T1_C1.tif", "/d/a_T2_C1.tif", "/d/a_T1.tif", "/d/Thumbnail.tif"]
    assert _parse_filepaths_with_tags(paths, ["T", "C"]) == paths[:2]


# -- generated grids -------------------------------------------------------------------------

PREFIXES = ["img", "batch3", "cat7", "T5run", "C1_T5run", "Thumbnail", "ch9x", "z_stack", "run2"]
FOLDERS = ["", "exp", "C1_T5run", "gfp_mcherry", "T9_ch3", "data2024"]
TIME, CHANNEL, Z = ["T", "t", "time"], ["ch", "C", "channel"], ["z", "Z", "slice"]


def _grid(seed):
    """(paths, shape, tags): a t x c x z grid of generated names, shuffled."""
    rng = random.Random(seed)
    top = rng.choice(FOLDERS)
    prefix, tail = rng.choice(PREFIXES), rng.choice(["", "_final", "_v2", "_stitched"])
    tags = {"T": rng.choice(TIME), "C": rng.choice(CHANNEL), "Z": rng.choice(Z)}
    if len({tag.lower()[0] for tag in tags.values()}) < 3:
        tags = {"T": "T", "C": "ch", "Z": "z"}
    order = rng.sample(["T", "C", "Z"], 3)
    pad = rng.choice([1, 2, 4])
    shape = {"T": rng.randint(2, 11), "C": rng.randint(2, 3), "Z": rng.randint(2, 3)}
    paths = {}
    for place in itertools.product(*(range(shape[k]) for k in "TCZ")):
        at = dict(zip("TCZ", place))
        parts = "_".join(f"{tags[k]}{at[k]:0{pad}d}" for k in order)
        paths[place] = "/".join(p for p in ("/d", top, f"{prefix}_{parts}{tail}.tif") if p)
    return paths, shape, tags, rng


@pytest.mark.parametrize("seed", range(120))
def test_a_generated_grid_joins_in_the_order_of_its_numbers(seed):
    paths, shape, tags, rng = _grid(seed)
    shuffled = list(paths.values())
    rng.shuffle(shuffled)
    fs = FileSet(shuffled, shapes=[DUMMY] * len(shuffled), axis_tag0=tags["T"],
                 axis_tag1=tags["C"], axis_tag2=tags["Z"])
    fs.check_places([0, 1, 2])
    for axis in (0, 1, 2):
        fs.concatenate_along(axis)
    assert len(set(fs.path_dict.values())) == 1
    for (t, c, z), path in paths.items():
        assert [s.start for s in fs.slice_dict[path][:3]] == [t, c, z], path


@pytest.mark.parametrize("seed", range(60))
def test_a_generated_grid_gives_one_output_per_tag_not_joined(seed):
    paths, shape, tags, rng = _grid(seed)
    plan = _plan(list(paths.values()), "t", time_tag=tags["T"], channel_tag=tags["C"], z_tag=tags["Z"])
    assert plan.n_outputs == shape["C"] * shape["Z"]
    want = {frozenset(paths[(t, c, z)] for t in range(shape["T"]))
            for c in range(shape["C"]) for z in range(shape["Z"])}
    assert {frozenset(out.source_files) for out in plan.outputs} == want


@pytest.mark.parametrize("seed", range(40))
def test_a_worker_reads_one_group_as_the_plan_read_all(seed):
    paths, shape, tags, rng = _grid(seed)
    kept = {place: path for place, path in paths.items() if place[1] == 0 and place[2] == 0}
    plan = _plan(list(paths.values()), "t", time_tag=tags["T"], channel_tag=tags["C"], z_tag=tags["Z"])
    fs = FileSet(list(kept.values()), shapes=[DUMMY] * len(kept), axis_tag0=tags["T"],
                 axis_tag1=tags["C"], axis_tag2=tags["Z"], tag_folders=plan.tag_folders)
    fs.concatenate_along(0)
    assert [fs.slice_dict[path][0].start for path in kept.values()] == list(range(shape["T"]))


def test_the_order_is_the_tags_not_the_paths():
    paths = ["/d/a_T2.tif", "/d/b_T1.tif", "/d/c_T3.tif"]
    fs = FileSet(paths, shapes=[DUMMY] * 3, axis_tag0="T")
    fs.concatenate_along(0)
    assert _order(fs, 0) == ["/d/b_T1.tif", "/d/a_T2.tif", "/d/c_T3.tif"]
    assert fs.in_order() == _order(fs, 0)


def test_named_channels_in_a_folder_above_a_worker_stay_one_channel_each():
    paths = [f"/d/gfp_mcherry/{c}/a_T{t}.tif" for c in ("gfp", "mcherry") for t in (1, 2, 3)]
    plan = _plan(paths, "t", time_tag="T", channel_tag="gfp,mcherry")
    assert plan.n_outputs == 2
    for out in plan.outputs:
        fs = FileSet(out.source_files, shapes=[DUMMY] * 3, axis_tag0="T", axis_tag1="gfp,mcherry",
                     tag_folders=plan.tag_folders)
        fs.concatenate_along(0)
        assert len(set(fs.path_dict.values())) == 1
        assert sorted(s[0].start for s in fs.slice_dict.values()) == [0, 1, 2]


@pytest.mark.parametrize("paths, axes, tags", [
    (["/d/a_T1.tif", "/d/a_T1_copy.tif", "/d/a_T2.tif"], "t", dict(time_tag="T")),
    (["/d/p1_T1.tif", "/d/p2_T1.tif"], "t", dict(time_tag="T")),
    (["/d/a_T1.tif", "/d/b_T01.tif"], "t", dict(time_tag="T")),
    (["/d/a_gfp.tif", "/d/b_gfp.tif", "/d/a_rfp.tif", "/d/b_rfp.tif"], "c", dict(channel_tag="gfp,rfp")),
])
def test_two_files_at_one_place_are_refused(paths, axes, tags):
    with pytest.raises(InputError, match="at one place"):
        _plan(paths, axes, **tags)


def test_an_incomplete_grid_is_refused_in_a_line():
    paths = ["/d/a_T1_C1.tif", "/d/a_T1_C2.tif", "/d/a_T2_C1.tif"]
    # the empty place named (review cycle 2, R2-19)
    with pytest.raises(InputError, match="leave places empty.*no file for T2 C2"):
        _plan(paths, "tc", time_tag="T", channel_tag="C")


def test_one_file_named_twice_joins_once():
    assert _parse_filepaths_with_tags(["/d/a_T1.tif", "/d/a_T1.tif", "/d/a_T2.tif"], ["T"]) \
        == ["/d/a_T1.tif", "/d/a_T2.tif"]


def test_a_space_after_the_comma_changes_nothing():
    paths = ["/d/a_Blue.tif", "/d/a_Red.tif"]
    assert _parse_filepaths_with_tags(paths, ["Blue, Red"]) == paths
    fs = FileSet(paths, shapes=[DUMMY] * 2, axis_tag1="Blue, Red")
    assert fs.axis_tags[1] == ("Blue", "Red")


def test_a_gap_in_one_joined_tag_is_no_error():
    paths = ["/d/a_T1.tif", "/d/a_T5.tif", "/d/a_T9.tif"]
    assert _plan(paths, "t", time_tag="T").n_outputs == 1


# -- to the pixels ---------------------------------------------------------------------------

def _tif(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.full((8, 8), value, "u2"))


def _values(out):
    import zarr
    stores = sorted(p for p in out.iterdir() if p.suffix == ".zarr")
    return {p.name: [int(v) for v in np.asarray(zarr.open(str(p), mode="r")["0"]).reshape(-1, 64)[:, 0]]
            for p in stores}


def _convert(folder, out, **kwargs):
    from eubi_bridge.ebridge import EuBIBridge
    EuBIBridge().to_zarr(str(folder), str(out), use_threading=True, **kwargs)


def test_pixels_follow_the_numbers_and_strays_stay_out(tmp_path):
    for name, value in (("b_T1", 1), ("a_T2", 2), ("c_T3", 3), ("Thumbnail", 99)):
        _tif(tmp_path / "in" / f"{name}.tif", value)
    _convert(tmp_path / "in", tmp_path / "o", time_tag="T", concatenation_axes="t")
    assert list(_values(tmp_path / "o").values()) == [[1, 2, 3]]


def test_pixels_of_a_tag_inside_a_word(tmp_path):
    for c in (1, 2):
        for t in (1, 2):
            _tif(tmp_path / "in" / f"batch3_ch{c}_t{t}.tif", 10 * c + t)
    _convert(tmp_path / "in", tmp_path / "o", channel_tag="ch", time_tag="t", concatenation_axes="t")
    assert sorted(_values(tmp_path / "o").values()) == [[11, 12], [21, 22]]


def test_pixels_of_tags_in_a_folder_of_that_name(tmp_path):
    for c, offset in (("gfp", 0), ("mcherry", 10)):
        for t in (1, 2, 3):
            _tif(tmp_path / "gfp_mcherry" / c / f"a_T{t}.tif", offset + t)
    _convert(tmp_path / "gfp_mcherry", tmp_path / "o", time_tag="T", channel_tag="gfp,mcherry",
             concatenation_axes="t")
    assert sorted(_values(tmp_path / "o").values()) == [[1, 2, 3], [11, 12, 13]]


def test_pixels_of_a_plan_made_first(tmp_path):
    from eubi_bridge.ebridge import EuBIBridge
    for c, offset in (("gfp", 0), ("mcherry", 10)):
        for t in (1, 2):
            _tif(tmp_path / "gfp_mcherry" / c / f"a_T{t}.tif", offset + t)
    bridge = EuBIBridge()
    kwargs = dict(time_tag="T", channel_tag="gfp,mcherry", concatenation_axes="t")
    plan = bridge.validate_aggregative(str(tmp_path / "gfp_mcherry"), str(tmp_path / "o"), **kwargs)
    bridge.to_zarr(str(tmp_path / "gfp_mcherry"), str(tmp_path / "o"), plan=plan,
                   use_threading=True, **kwargs)
    assert sorted(_values(tmp_path / "o").values()) == [[1, 2], [11, 12]]


def test_a_plan_says_what_it_leaves_out_and_keeps_its_output(tmp_path, caplog):
    import logging
    from eubi_bridge.ebridge import EuBIBridge
    _tif(tmp_path / "in" / "x_T0.tif", 1)
    _tif(tmp_path / "in" / "Thumbnail.tif", 9)
    bridge = EuBIBridge()
    kwargs = dict(time_tag="T", concatenation_axes="t")
    with caplog.at_level(logging.WARNING):
        plan = bridge.validate_aggregative(str(tmp_path / "in"), str(tmp_path / "o"), **kwargs)
    assert "Thumbnail.tif" in caplog.text
    with pytest.raises(Exception, match="make the plan with that output"):
        bridge.to_zarr(str(tmp_path / "in"), str(tmp_path / "elsewhere"), plan=plan,
                       use_threading=True, **kwargs)


def test_joined_ome_zarr_inputs(tmp_path):
    for t in (1, 2):
        _tif(tmp_path / "src" / f"a_T{t}.tif", t)
    from eubi_bridge.ebridge import EuBIBridge
    EuBIBridge().to_zarr(str(tmp_path / "src"), str(tmp_path / "in"), use_threading=True, zarr_format=2)
    _convert(tmp_path / "in", tmp_path / "o", time_tag="T", concatenation_axes="t")
    assert list(_values(tmp_path / "o").values()) == [[1, 2]]


def test_a_plan_of_one_file_gets_a_store_named_after_it(tmp_path):
    from eubi_bridge.ebridge import EuBIBridge
    _tif(tmp_path / "in" / "x_T0.tif", 1)
    plan = EuBIBridge().validate_aggregative(str(tmp_path / "in"), str(tmp_path / "o"),
                                             time_tag="_T", concatenation_axes="t")
    assert os.path.basename(plan.outputs[0].output_path).startswith("x_T0")


# -- tags against the joined axes, before any file is grouped -------------------------------

@pytest.mark.parametrize("axes, tags, says", [
    ("tc", dict(time_tag="T"), "Joining along c needs a tag"),
    ("tt", dict(time_tag="T"), "names t twice"),
    ("tq", dict(time_tag="T"), "q is no axis"),
    ("t", dict(time_tag="T", channel_tag="T"), "are both 'T'"),
    ("c", dict(channel_tag="gfp,rfp", z_tag="rfp"), "are both 'rfp'"),
])
def test_tags_and_axes_that_do_not_fit_are_refused_before_grouping(axes, tags, says):
    with pytest.raises(InputError, match=says):
        _plan(["/d/a_T1_gfp_rfp1.tif", "/d/a_T2_gfp_rfp1.tif"], axes, **tags)


def test_a_tag_of_an_axis_not_joined_splits_the_outputs_and_says_so(caplog):
    import logging
    paths = [f"/d/a_T{t}_C{c}.tif" for t in (1, 2) for c in (1, 2)]
    with caplog.at_level(logging.INFO):
        plan = _plan(paths, "t", time_tag="T", channel_tag="C")
    assert plan.n_outputs == 2
    assert "one output per channel_tag value" in caplog.text


def test_two_axes_join_each_in_its_own_lines():
    paths = [f"/d/Channel{c}-T000{t}.tif" for c in (1, 2) for t in (1, 2, 3)]
    fs = FileSet(list(reversed(paths)), shapes=[DUMMY] * 6, axis_tag0="T", axis_tag1="Channel")
    fs.check_places([0, 1])
    fs.concatenate_along(0)
    fs.concatenate_along(1)
    for path in paths:
        c, t = int(path.split("Channel")[1][0]), int(path[-5])
        assert (fs.slice_dict[path][0].start, fs.slice_dict[path][1].start) == (t - 1, c - 1)


# -- the ninth review: tags that follow each other in one word -------------------------------

VENDOR = {  # name of the file at (t, c, z), the tags, the joined axes
    "fluoview": (lambda t, c, z: f"s_C{c + 1:03d}Z{z + 1:03d}.tif", dict(channel_tag="C", z_tag="Z"), "cz"),
    "cv7000": (lambda t, c, z: f"AssayPlate_B03_T{t + 1:04d}F001L01A01Z{z + 1:02d}C{c + 1:02d}.tif",
               dict(time_tag="T", channel_tag="C", z_tag="Z"), "tcz"),
    "operetta": (lambda t, c, z: f"r01c01f01p{z + 1:02d}-ch{c + 1}sk{t + 1}fk1fl1.tiff",
                 dict(time_tag="sk", channel_tag="ch", z_tag="p"), "tcz"),
    "zen": (lambda t, c, z: f"b0t{t}z{z}c{c}x0-512y0-512.tif", dict(time_tag="t", channel_tag="c", z_tag="z"), "tcz"),
    "nikon": (lambda t, c, z: f"exp_t{t + 1:02d}xy1z{z + 1}c{c + 1}.tif", dict(time_tag="t", channel_tag="c", z_tag="z"), "tcz"),
}


@pytest.mark.parametrize("vendor", VENDOR)
def test_vendor_names_with_tags_in_one_word_join_by_each_tag(vendor):
    name, tags, axes = VENDOR[vendor]
    grid = {(t, c, z): f"/d/{name(t, c, z)}" for t in range(2 if "t" in axes else 1)
            for c in range(2) for z in range(3)}
    shuffled = list(grid.values())
    random.Random(1).shuffle(shuffled)
    plan = _plan(shuffled, axes, **tags)
    assert plan.n_outputs == 1
    axis_tags = {f"axis_tag{'tcz'.index(x)}": tags[t] for x, t in
                 (("t", "time_tag"), ("c", "channel_tag"), ("z", "z_tag")) if t in tags}
    fs = FileSet(shuffled, shapes=[DUMMY] * len(shuffled), **axis_tags)
    for axis in ["tcz".index(x) for x in axes]:
        fs.concatenate_along(axis)
    for (t, c, z), path in grid.items():
        assert [fs.slice_dict[path][a].start for a in range(3)] == [t if "t" in axes else 0, c, z], path


def test_a_tag_not_joined_splits_by_its_own_number_only():
    paths = [f"/d/s_C{c:03d}Z{z:03d}.tif" for c in (1, 2) for z in (1, 2, 3)]
    plan = _plan(paths, "z", channel_tag="C", z_tag="Z")
    assert sorted(len(out.source_files) for out in plan.outputs) == [3, 3]


def test_an_untagged_number_after_a_tag_is_no_part_of_its_place():
    # fields of one time point were joined as time points, exit 0 (also in 0.3.3)
    paths = [f"/d/Plate_B03_T000{t}F00{f}L01A01Z01C01.tif" for t in (1, 2) for f in (1, 2)]
    with pytest.raises(InputError, match="at one place"):
        _plan(paths, "t", time_tag="T")
    paths = [f"/d/r01c01f01p01-ch{c}sk{t}fk1fl1.tiff" for c in (1, 2) for t in (1, 2)]
    with pytest.raises(InputError, match="at one place"):
        _plan(paths, "c", channel_tag="ch")


@pytest.mark.parametrize("name, place", [
    ("x_T00h15m30s.tif", (0,)), ("x_T1h.tif", (1,)), ("x_T5sec.tif", (5,)), ("x_T0001F001.tif", (1,)),
    ("x_T1sk2.tif", (1,)), ("x_T2.5.tif", (2,)), ("x_T3-4.tif", (3,)), ("x_T7_8.tif", (7,)),
])
def test_a_number_tag_is_the_digits_right_after_it_and_nothing_more(name, place):
    assert read_tag([f"/d/{name}"], "T")[f"/d/{name}"] == place


@pytest.mark.parametrize("names, tag", [
    (["a_z0.5.tif", "a_z1.0.tif", "a_z1.5.tif"], "z"),           # z1 twice: no decimals
    (["a_T00h15m.tif", "a_T00h30m.tif"], "T"),                  # T00 twice: no units
])
def test_numbers_the_tag_does_not_read_leave_files_at_one_place(names, tag):
    with pytest.raises(InputError, match="at one place"):
        _plan([f"/d/{name}" for name in names], "z" if tag == "z" else "t",
              **{"z_tag" if tag == "z" else "time_tag": tag})


def test_only_the_tag_and_its_digits_leave_the_name():
    assert reduce_paths_flexible(["s_C001Z001.tif", "s_C002Z001.tif"], "C", "_cset") == "s_C_csetZ001.tif"
    assert reduce_paths_flexible(["x_T00h05m.tif", "x_T01h05m.tif"], "T", "_tset") == "x_T_tseth05m.tif"


@pytest.mark.parametrize("axes, tags, says", [
    ("t", dict(time_tag="frames/", z_tag="frames"), "are both 'frames'"),
    ("c", dict(channel_tag="ch1,ch2", z_tag="ch"), "is read in --channel_tag's name ch1"),
    ("c", dict(channel_tag="Blue,"), "names one channel only"),
    ("c", dict(channel_tag="Blue,Blue,Red"), "'Blue' is named twice"),
])
def test_more_tags_that_do_not_fit_are_refused(axes, tags, says):
    with pytest.raises(InputError, match=says):
        _plan(["/d/frames/000/a_ch1_Blue.tif", "/d/frames/001/a_ch2_Red.tif"], axes, **tags)


def test_names_given_as_a_list_lose_their_spaces_and_one_file_spelled_two_ways_is_one():
    paths = ["/d/x_Blue.tif", "/d/x_Red.tif"]
    assert _parse_filepaths_with_tags(paths, [("Blue", " Red")]) == paths
    assert _parse_filepaths_with_tags(["/d/a_T1.tif", "/d/../d/a_T1.tif", "/d/a_T2.tif"], ["T"]) \
        == ["/d/a_T1.tif", "/d/a_T2.tif"]
    # a backslash separates folders on Windows only: on Linux and macOS it is part of the
    # name, so r"\d\a_T1.tif" there is another file
    if os.sep == "\\":
        assert _parse_filepaths_with_tags(["/d/a_T1.tif", r"\d\a_T1.tif", "/d/a_T2.tif"], ["T"]) \
            == ["/d/a_T1.tif", "/d/a_T2.tif"]


def test_a_plan_is_refused_for_other_settings(tmp_path):
    from eubi_bridge.ebridge import EuBIBridge
    for t in (1, 2):
        _tif(tmp_path / "in" / f"a_T{t}.tif", t)
        _tif(tmp_path / "in2" / f"q_T{t}.tif", 10 + t)
    bridge = EuBIBridge()
    plan = bridge.validate_aggregative(str(tmp_path / "in"), str(tmp_path / "o"),
                                       time_tag="T", concatenation_axes="t")
    for where, kwargs, says in (
            ("in2", dict(time_tag="T", concatenation_axes="t"), "input"),
            ("in", dict(z_tag="T", concatenation_axes="z"), "concatenation_axes t, not z"),
            ("in", dict(), "concatenation_axes t, not none")):
        with pytest.raises(InputError, match=says):
            bridge.to_zarr(str(tmp_path / where), str(tmp_path / "o"), plan=plan,
                           use_threading=True, **kwargs)
    assert not (tmp_path / "o").exists()


def test_folders_named_by_a_tag_are_no_warning(caplog):
    import logging
    paths = [f"/d/{c}/T000{t}.tif" for c in ("Blue", "Red") for t in (1, 2)]
    with caplog.at_level(logging.WARNING):
        _plan(paths, "ct", time_tag="T", channel_tag="Blue,Red")
    assert "do not belong together" not in caplog.text
    with caplog.at_level(logging.WARNING):
        _plan(["/d/run1/a_T1.tif", "/d/run2/a_T2.tif"], "t", time_tag="T")
    assert "do not belong together" in caplog.text


def test_pixels_of_a_vendor_name(tmp_path):
    for c in (1, 2):
        for z in (1, 2, 3):
            _tif(tmp_path / "in" / f"s_C{c:03d}Z{z:03d}.tif", 10 * c + z)
    _convert(tmp_path / "in", tmp_path / "o", channel_tag="C", z_tag="Z", concatenation_axes="z")
    assert _values(tmp_path / "o") == {"s_C001Z_zset.ome.zarr": [11, 12, 13], "s_C002Z_zset.ome.zarr": [21, 22, 23]}


# -- review cycle 1 (2026-11-04) ---------------------------------------------------------------

def test_a_tag_no_file_has_is_not_read_from_the_folders_the_data_is_kept_in():
    # N-02: --time_tag t read t11 from a folder 'hunt11' far above the input
    paths = ["/data/hunt11/x/img_c1.tif", "/data/hunt11/x/img_c2.tif"]
    assert read_tag(paths, "t") == {}
    frames = ["/d/frames/000/a.tif", "/d/frames/001/a.tif"]              # a tag that is a folder
    assert [p[0] for p in read_tag(frames, "frames").values()] == [0, 1]


def test_a_tag_written_in_another_case_is_named(tmp_path):
    for t in (1, 2):
        _tif(tmp_path / "in" / f"a_T{t}.tif", t)
    with pytest.raises(InputError, match=r"did you mean --time_tag T\?"):
        _convert(tmp_path / "in", tmp_path / "o", time_tag="t", concatenation_axes="t")


def test_a_gap_in_the_numbers_is_said(caplog):
    # N-01: a_T1, a_T2, a_T4 were joined as three time points, in silence
    import logging
    with caplog.at_level(logging.WARNING):
        _plan(["/d/a_T1.tif", "/d/a_T2.tif", "/d/a_T4.tif"], "t", time_tag="T")
    assert "no file for T3" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _plan(["/d/a_T0.tif", "/d/a_T10.tif", "/d/a_T20.tif"], "t", time_tag="T")
    assert "no file for" not in caplog.text
