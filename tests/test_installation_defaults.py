"""The command line and the GUI start from one set of defaults: what an install does must not
depend on how it is started (2026-10-30: the GUI converted every image of a file, the CLI the
first only, so a file silently lost images on the command line)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from eubi_bridge.ebridge import ConfigManager
from eubi_bridge.qt_gui.server import config_manager as gui
from eubi_bridge.utils.path_utils import take_filepaths
from eubi_bridge.conversion.pyramid_path import _warn_uncalibrated

tifffile = pytest.importorskip("tifffile")

#: spelled differently, meaning the same: {} is the codec's own defaults, None the
#: version zarr_format gives
SAME_MEANING = {("conversion", "compressor_params"), ("conversion", "ome_zarr_version")}


def _differences(a: dict, b: dict) -> list:
    return [(section, key, a[section].get(key), b.get(section, {}).get(key))
            for section in a for key in a[section]
            if (section, key) not in SAME_MEANING
            and key not in ("index_defaults_all", "mosaic_default")
            and a[section].get(key) != b.get(section, {}).get(key)]


def test_the_gui_shows_the_installation_defaults():
    root = ConfigManager._ROOT_DEFAULTS
    assert _differences(root, gui._react_to_config(gui._config_to_react(root))) == []


def test_a_gui_without_a_saved_config_has_them_too():
    assert _differences(ConfigManager._ROOT_DEFAULTS, gui._react_to_config({})) == []


def test_every_image_is_converted_by_default():
    readers = ConfigManager._ROOT_DEFAULTS["readers"]
    assert {readers[k] for k in ("series_index", "mosaic_tile_index", "view_index",
                                 "illumination_index")} == {"all"}


def _manager(tmp_path, readers: dict):
    path = tmp_path / ".eubi_config.json"
    path.write_text(json.dumps({"readers": readers}))
    return ConfigManager(str(path)), path


def test_a_config_saved_before_the_change_adopts_them_once(tmp_path):
    manager, path = _manager(tmp_path, {"series_index": 0, "view_index": 0,
                                        "mosaic_tile_index": 0, "illumination_index": 2})
    readers = manager.config["readers"]
    assert readers["series_index"] == readers["view_index"] == readers["mosaic_tile_index"] == "all"
    assert readers["illumination_index"] == 2                   # chosen: kept
    saved = json.loads(path.read_text())["readers"]
    assert saved["index_defaults_all"] is True and saved["series_index"] == "all"


def test_a_zero_chosen_after_that_stays(tmp_path):
    manager, _path = _manager(tmp_path, {"series_index": 0, "index_defaults_all": True})
    assert manager.config["readers"]["series_index"] == 0


def test_a_file_is_only_that_file(tmp_path):
    for name in ("rois.tif", "with_rois.tif", "other.tif"):
        tifffile.imwrite(tmp_path / name, np.zeros((4, 4), np.uint8))
    found = take_filepaths(str(tmp_path / "rois.tif"))["input_path"]
    assert [p.replace("\\", "/").rsplit("/", 1)[-1] for p in found] == ["rois.tif"]


def test_macos_stub_files_are_no_inputs(tmp_path):
    tifffile.imwrite(tmp_path / "a.tif", np.zeros((4, 4), np.uint8))
    (tmp_path / "._a.tif").write_bytes(b"\x00\x05\x16\x07")
    found = take_filepaths(str(tmp_path))["input_path"]
    assert [p.replace("\\", "/").rsplit("/", 1)[-1] for p in found] == ["a.tif"]


def _entry(axes, units, shape):
    from types import SimpleNamespace as NS
    return NS(pyramid=NS(meta=NS(axis_order=axes, unit_list=units),
                         base_array=NS(shape=shape)))


def test_a_warning_names_what_is_not_calibrated(monkeypatch):
    from types import SimpleNamespace as NS
    from eubi_bridge.conversion import pyramid_path
    lines = []
    monkeypatch.setattr(pyramid_path.logger, "warning", lines.append)
    job = NS(input_path="x.tif", metadata=NS(time_scale=None, z_scale=None, y_scale=None, x_scale=None))
    _warn_uncalibrated(job, [_entry("tzyx", [None, None, "micrometer", "micrometer"], (1, 4, 8, 8))])
    assert len(lines) == 1 and "axes z." in lines[0] and "--z_scale" in lines[0]    # t has one value
    lines.clear()
    job.metadata.z_scale = 0.5
    _warn_uncalibrated(job, [_entry("zyx", [None, "micrometer", "micrometer"], (4, 8, 8))])
    assert not [l for l in lines if "NOT CALIBRATED" in l]      # the job gives it
    # given without its unit beside axes in micrometres: said (review cycle 1, N-26)
    assert len(lines) == 1 and "add --z_unit micrometer" in lines[0]


def test_an_unknown_option_is_refused_with_the_nearest_name():
    manager = ConfigManager("unused")
    with pytest.raises(ValueError, match=r"--z_scal \(did you mean --z_scale\?\)"):
        manager.check_option_names({"z_scal": 2})
    with pytest.raises(ValueError, match="--bogus_flag"):
        manager.check_option_names({"bogus_flag": 7, "x_scale": 0.5})


def test_every_known_option_passes():
    manager = ConfigManager("unused")
    manager.check_option_names({key: None for section in ConfigManager._ROOT_DEFAULTS.values()
                                for key in section})
    manager.check_option_names({"scene_index": 0})              # a former name still works


def test_a_narrower_dtype_is_said_to_clip(monkeypatch):
    from types import SimpleNamespace as NS
    from eubi_bridge.conversion import pyramid_path
    lines = []
    monkeypatch.setattr(pyramid_path.logger, "warning", lines.append)
    job = NS(input_path="x.tif", conversion=NS(dtype="uint8"))
    entry = NS(pyramid=NS(base_array=NS(dtype=np.dtype("u2"))))
    pyramid_path._warn_dtype(job, [entry])
    assert len(lines) == 1 and "clipped" in lines[0]
    lines.clear()
    job.conversion.dtype = "uint32"
    pyramid_path._warn_dtype(job, [entry])
    assert not lines


def test_a_long_joined_name_fits_a_path():
    from eubi_bridge.conversion.dispatcher import _short_name
    long, other = "a" * 400, "a" * 399 + "b"
    assert len(_short_name(long)) <= 120 and _short_name(long) != _short_name(other)
    assert _short_name("short") == "short"


def test_a_plane_file_alone_stands_for_its_plate(tmp_path):
    from eubi_bridge.utils.path_utils import input_mains
    import sys
    sys.path.insert(0, "incubation/micro-reader/tests")
    test_incell = pytest.importorskip("test_incell")
    index, _data = test_incell._write(str(tmp_path), [("FITC", [100.0]), ("DAPI", [100.0])])
    planes = sorted(str(p) for p in tmp_path.glob("*.tif"))
    assert len(planes) == 2
    assert {str(m).lower() for m in input_mains(planes)} == {str(index).lower()}


def test_an_uncalibrated_image_among_calibrated_ones_is_named(monkeypatch):
    # review cycle 2, R2-15: a calibrated slide was NOT CALIBRATED because of its macro photo,
    # and the advice (--x_scale ...) would have overwritten every image's sizes
    from types import SimpleNamespace as NS
    from eubi_bridge.conversion import pyramid_path
    lines = []
    monkeypatch.setattr(pyramid_path.logger, "warning", lines.append)
    job = NS(input_path="/d/slide.vsi", metadata=NS(time_scale=None, z_scale=None, y_scale=None, x_scale=None))
    scan = _entry("yx", ["micrometer", "micrometer"], (8, 8))
    macro = _entry("yx", [None, None], (8, 8))
    macro.image = NS(name="macro")
    _warn_uncalibrated(job, [scan, macro])
    assert len(lines) == 1 and "1 of the 2 images of /d/slide.vsi has no physical size" in lines[0]
    assert "'macro': y, x" in lines[0] and "--x_scale" not in lines[0]
