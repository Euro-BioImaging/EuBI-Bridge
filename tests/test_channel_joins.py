"""Channels joined from files, in data and metadata (tenth review N-02 and the cases around it).

Each file's planes hold a known value, so the output's channel order is read from its pixels;
the labels are read from the store's omero block and from the OME-XML beside it."""
import os
import re

import numpy as np
import pytest
import tifffile
import zarr

pytest.importorskip("micro_reader")


def _plane(value):
    return np.full((16, 16), value, "u2")


def _ome(path, array, names, colors=None):
    channel = {"Name": names}
    if colors:
        channel["Color"] = colors
    tifffile.imwrite(path, array, ome=True, photometric="minisblack",
                     metadata={"axes": "CYX", "Channel": channel})


def _read(store):
    """(level 0 as t c z y x, labels, colours, the OME-XML's channel names)"""
    group = zarr.open_group(str(store), mode="r")
    meta = dict(group.attrs)
    meta = meta.get("ome", meta)
    ms = meta["multiscales"][0]
    axes = "".join(a["name"] for a in ms["axes"])
    full = np.asarray(group[ms["datasets"][0]["path"]])
    for ax in "tczyx":
        if ax not in axes:
            full = np.expand_dims(full, "tczyx".index(ax))
    channels = (meta.get("omero") or {}).get("channels") or []
    xml = store / "OME" / "METADATA.ome.xml"
    names = (re.findall(r"<(?:\w+:)?Channel\b[^>]*?\bName=\"([^\"]*)\"", xml.read_text("utf-8"))
             if xml.exists() else None)
    return (full, [c.get("label") for c in channels],
            [str(c.get("color", "")).upper() for c in channels], names)


def _convert(src, out, **kwargs):
    from eubi_bridge.ebridge import EuBIBridge
    EuBIBridge().to_zarr(str(src), str(out), use_threading=True, verbose=False, **kwargs)
    return {p.name: p for p in out.iterdir() if p.suffix == ".zarr"}


def _values(full):
    return [int(np.unique(full[:, c])[0]) for c in range(full.shape[1])]


@pytest.mark.parametrize("order", ["Blue,Red", "Red,Blue"])
@pytest.mark.parametrize("override", [False, True])
def test_named_channels_join_in_the_order_given(tmp_path, order, override):
    (tmp_path / "in").mkdir()
    for value, name in ((10, "Blue"), (20, "Red")):
        tifffile.imwrite(tmp_path / "in" / f"x_{name}.tif", _plane(value))
    names = order.split(",")
    got = _convert(tmp_path / "in", tmp_path / "o", channel_tag=order, concatenation_axes="c",
                   override_channel_names=override)
    full, labels, colors, xml = _read(got[f"x_{''.join(names)}_cset.zarr"])
    assert _values(full) == [10 if n == "Blue" else 20 for n in names]
    if override:
        assert labels == names and xml in (None, names)
    assert len(set(colors)) == 2


def test_each_store_of_a_split_channel_tag_is_named_by_its_own_tag(tmp_path):
    # the documented "Concatenate Along Time Only" example labelled every store "moving"
    (tmp_path / "in").mkdir()
    for base, name in ((10, "growing"), (20, "moving"), (30, "noise")):
        for t in (1, 2, 3):
            tifffile.imwrite(tmp_path / "in" / f"tc_{name}_step{t}_16jan.tif", _plane(base + t))
    got = _convert(tmp_path / "in", tmp_path / "o", channel_tag="growing,moving,noise",
                   time_tag="step", concatenation_axes="t", override_channel_names=True)
    for base, name in ((10, "growing"), (20, "moving"), (30, "noise")):
        full, labels, colors, xml = _read(got[f"tc_{name}_step_tset_16jan.zarr"])
        assert [int(full[t, 0].max()) for t in range(3)] == [base + 1, base + 2, base + 3]
        assert labels == [name] and xml in (None, [name])


@pytest.mark.parametrize("override, labels", [
    (False, ["GFP", "DAPI", "GFP", "DAPI"]),
    (True, ["Blue_0", "Blue_1", "Red_0", "Red_1"]),
])
def test_files_of_several_channels_keep_each_channel(tmp_path, override, labels):
    (tmp_path / "in").mkdir()
    for base, name in ((10, "Blue"), (20, "Red")):
        _ome(tmp_path / "in" / f"x_{name}.ome.tif", np.stack([_plane(base + 1), _plane(base + 2)]),
             ["GFP", "DAPI"], [-16776961, 65535])
    (store,) = _convert(tmp_path / "in", tmp_path / "o", channel_tag="Blue,Red",
                        concatenation_axes="c", override_channel_names=override).values()
    full, got, colors, xml = _read(store)
    assert _values(full) == [11, 12, 21, 22]
    assert got == labels and xml in (None, labels)
    assert len(set(colors)) == 4                      # two of each colour: told apart


def test_rgb_files_join_as_their_samples(tmp_path):
    (tmp_path / "in").mkdir()
    for base, name in ((10, "Blue"), (20, "Red")):
        rgb = np.stack([np.full((16, 16), base + k, "u1") for k in range(3)], axis=-1)
        tifffile.imwrite(tmp_path / "in" / f"x_{name}.tif", rgb, photometric="rgb")
    (store,) = _convert(tmp_path / "in", tmp_path / "o", channel_tag="Blue,Red",
                        concatenation_axes="c").values()
    assert _values(_read(store)[0]) == [10, 11, 12, 20, 21, 22]


@pytest.mark.parametrize("nested", [False, True])
def test_time_and_channels_land_where_their_tags_say(tmp_path, nested):
    want = np.zeros((3, 2, 1, 16, 16), "u2")
    for ci, name in enumerate(("Blue", "Red")):
        for t in (1, 2, 3):
            path = (tmp_path / "in" / name / f"T000{t}.tif" if nested
                    else tmp_path / "in" / f"{name}-T000{t}.tif")
            path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(path, _plane(100 * (ci + 1) + t))
            want[t - 1, ci] = 100 * (ci + 1) + t
    (store,) = _convert(tmp_path / "in", tmp_path / "o", channel_tag="Blue,Red", time_tag="T",
                        concatenation_axes="tc", override_channel_names=True).values()
    full, labels, colors, xml = _read(store)
    assert np.array_equal(full, want)
    assert labels == ["Blue", "Red"]


def test_a_name_holding_another_joins_as_given(tmp_path):
    (tmp_path / "in").mkdir()
    tifffile.imwrite(tmp_path / "in" / "x_GD2.tif", _plane(10))
    tifffile.imwrite(tmp_path / "in" / "x_GM2_GD2.tif", _plane(20))
    got = _convert(tmp_path / "in", tmp_path / "o", channel_tag="GD2,GM2_GD2",
                   concatenation_axes="c", override_channel_names=True)
    full, labels, colors, xml = _read(got["x_GD2GM2_GD2_cset.zarr"])
    assert _values(full) == [10, 20] and labels == ["GD2", "GM2_GD2"]


def test_channel_names_from_the_files_are_kept(tmp_path):
    (tmp_path / "in").mkdir()
    for base, name, label in ((10, "Blue", "Hoechst"), (20, "Red", "mCherry")):
        _ome(tmp_path / "in" / f"x_{name}.ome.tif", _plane(base)[None], [label])
    (store,) = _convert(tmp_path / "in", tmp_path / "o", channel_tag="Blue,Red",
                        concatenation_axes="c").values()
    full, labels, colors, xml = _read(store)
    assert _values(full) == [10, 20] and labels == ["Hoechst", "mCherry"]
    assert xml in (None, ["Hoechst", "mCherry"])
