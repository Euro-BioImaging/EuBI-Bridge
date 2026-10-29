"""Every parameter the Convert page can set, against the batch queue.

The queue table, its Edit Cells dialog and the conversion each keep their own idea of
what a parameter is and which values it takes; this is where they drifted apart
(found 2026-10-16): the form offered a codec every conversion refused ('lz4'), Edit
Cells offered a metadata reader the conversion did not know ('bioformats') and a
display range the form did not ('auto', which the form's config then turned into
'from_array'), the writer settings had no per-row editor at all, a number editor
rounded pixel sizes to three decimals, and a row's OME-Zarr version warned that a
derived key was "kept at the batch value". These tests keep the four in step:

- coverage: every value the form sends to a conversion is editable per row, or
  excluded on purpose (cluster-wide, batch-wide, derived, riding with another key);
- choices: Edit Cells offers what the form offers and what the conversion accepts;
- edits: every editable value reaches its conversion job, and every editor gives back
  the value it was set to;
- the batch's Run check reports settings the conversion would refuse.
"""
from __future__ import annotations

import os
from copy import deepcopy

import pytest

from tests.conftest import qt_available

if not qt_available():
    pytest.skip("PyQt6 unavailable or no usable Qt platform plugin", allow_module_level=True)

from eubi_bridge.qt_gui.core import batch as B  # noqa: E402

CONCAT = ("aggregative_group", "concatenation_axes", "time_tag", "channel_tag", "z_tag",
          "y_tag", "x_tag")
#: values a text editor takes in these tests
TEXT = {"series_index": "1", "mosaic_tile_index": "0,2", "view_index": "1",
        "illumination_index": "1", "phase_index": "1", "rotation_index": "1",
        "sample_index": "1", "channel_colors": "0,00FF00", "time_unit": "millisecond",
        "z_unit": "nanometer", "y_unit": "nanometer", "x_unit": "nanometer"}


@pytest.fixture(scope="module")
def page():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from eubi_bridge.qt_gui.pages.convert_page import ConvertPage
    widget = ConvertPage()
    yield widget
    widget.deleteLater()
    app.processEvents()


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Never the user's own config."""
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("EUBI_CONFIG_DIR", str(tmp_path / "home" / ".eubi_bridge"))
    return tmp_path


def _all_toggles_on(page) -> list:
    """The page's config with every conditional setting switched on and off."""
    from eubi_bridge.qt_gui.workers.conversion_worker import _build_kwargs  # noqa: F401
    cfg = page._ui_to_config()
    out = []
    for auto_chunk in (True, False):
        c = deepcopy(cfg)
        c["conversion"]["autoChunk"] = auto_chunk
        c["metadata"]["overridePhysicalScale"] = True
        for ax in ("Time", "Z", "Y", "X"):
            c["metadata"][f"scale{ax}"] = 2.5
            c["metadata"][f"unit{ax}"] = "millisecond" if ax == "Time" else "nanometer"
        out.append(c)
    return out


def _value_for(spec, current):
    if spec.kind == "choice":
        return next(c for c in spec.choices if str(c) != str(B._to_sentinel(spec.key, current)))
    if spec.kind == "bool":
        return not bool(current) if current is not None else True
    if spec.kind == "int":
        return int(spec.minimum) + (3 if (current or 0) != int(spec.minimum) + 3 else 4)
    if spec.kind == "auto_int":
        return 3
    if spec.kind == "float":
        return 0.0947                       # below 0.001 precision: a pixel size
    return TEXT.get(spec.key, "/d/x.tif" if spec.kind in ("file", "directory") else "x")


def test_every_form_setting_is_editable_per_row_or_excluded_on_purpose(page):
    from eubi_bridge.qt_gui.workers.conversion_worker import _build_kwargs
    keys = set().union(*(_build_kwargs(c) for c in _all_toggles_on(page)))
    lost = sorted(k for k in keys if B.uneditable_reason(k) == "not editable per row")
    assert not lost, f"form settings without a per-row editor or a reason: {lost}"
    stale = sorted(s.key for s in B._PARAM_SPECS
                   if s.key not in keys and s.key not in B._PATH_COLUMNS)
    assert not stale, f"Edit Cells editors for settings the form never sets: {stale}"


def test_choices_agree_between_form_edit_cells_and_conversion(page):
    from eubi_bridge.core.config_models import SUPPORTED_COMPRESSORS_ANY
    form = lambda widget: [widget.itemText(i) for i in range(widget.count())]  # noqa: E731
    assert form(page._codec) == list(B.spec_for("compressor").choices)
    assert set(B.spec_for("compressor").choices) - {"none"} <= SUPPORTED_COMPRESSORS_ANY
    assert form(page._downscale_method) == list(B.spec_for("downscale_method").choices)
    assert form(page._ome_zarr_version) == list(B.spec_for("ome_zarr_version").choices)
    assert set(form(page._io_backend)) <= set(B.spec_for("io_backend").choices)
    assert set(form(page._data_type)) <= set(B.spec_for("dtype").choices)
    assert [page._export_acquisition.itemText(i) for i in range(page._export_acquisition.count())] \
        == list(B.spec_for("export_acquisition_metadata").choices)
    # the form's names for the display range, mapped by _build_kwargs
    from eubi_bridge.qt_gui.workers.conversion_worker import _build_kwargs
    mapped = []
    for text in form(page._channel_intensity):
        cfg = page._ui_to_config()
        cfg["metadata"]["channelIntensityLimits"] = text
        mapped.append(_build_kwargs(cfg)["channel_intensity_limits"])
    assert mapped == list(B.spec_for("channel_intensity_limits").choices)
    if B.spec_for("metadata_reader") is not None:
        assert form(page._metadata_reader) == list(B.spec_for("metadata_reader").choices)


def test_a_form_change_shows_in_the_queue_without_a_false_warning(page):
    base = page._ui_to_config()
    for section, key, value, column in (
            ("conversion", "omeZarrVersion", "0.5", "ome_zarr_version"),
            ("conversion", "ioBackend", "zarrista", "io_backend"),
            ("conversion", "exportAcquisitionMetadata", True, "export_acquisition_metadata"),
            ("metadata", "channelIntensityLimits", "auto", "channel_intensity_limits")):
        changed = deepcopy(base)
        changed[section][key] = value
        model = B.BatchModel()
        model.set_baseline(deepcopy(base))
        assert model.add(changed, ["/d/a.tif"], "/out") == [], key
        assert model.cell(model.rows[0], column) not in (None, ""), key


@pytest.mark.parametrize("key", [s.key for s in B._PARAM_SPECS
                                 if s.key not in B._PATH_COLUMNS and s.key not in CONCAT])
def test_an_edited_cell_reaches_its_conversion(page, isolated, monkeypatch, key):
    import eubi_bridge.conversion.dispatcher as dispatcher
    import eubi_bridge.ebridge as ebridge
    from eubi_bridge.ebridge import EuBIBridge
    from eubi_bridge.qt_gui.workers.conversion_worker import _build_kwargs
    captured = {}
    monkeypatch.setattr(dispatcher, "dispatch_unary_jobs", lambda jobs: captured.setdefault("jobs", jobs))
    # nothing is written here, so an io_backend edit to 'zarrista' must reach the job
    # whether or not zarrista is installed (CI has none: the run would be refused)
    monkeypatch.setattr(ebridge, "_zarrista_installed", lambda: True)
    src = isolated / "a.tif"
    src.write_bytes(b"II*\x00")
    base = page._ui_to_config()
    spec = B.spec_for(key)
    value = _value_for(spec, B._to_sentinel(key, _build_kwargs(base).get(key)))
    model = B.BatchModel()
    model.set_baseline(deepcopy(base))
    model.add(base, [str(src)], str(isolated / "out"))
    model.update_cells([0], key, value)
    assert model.cell(model.rows[0], key) is not None
    EuBIBridge().to_zarr(model.to_table(), verbose=False)
    (job,) = captured["jobs"]
    got = next((getattr(getattr(job, section), key)
                for section in ("readers", "conversion", "downscale", "metadata")
                if key in type(getattr(job, section)).model_fields),
               (job.extra or {}).get(key, "<not on the job>"))
    want = B.from_sentinel(key, value)
    norm = lambda v: ",".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)  # noqa: E731
    assert norm(got) == norm(want), f"{key}: set {value!r}, the job has {got!r}"


@pytest.mark.parametrize("spec", B._PARAM_SPECS, ids=lambda s: s.key)
def test_every_edit_cells_editor_gives_back_its_value(page, spec):
    from eubi_bridge.qt_gui.widgets.batch_cell_editor import _Field
    value = _value_for(spec, None)
    field = _Field(spec, None, True, None)
    field.set_value(value)
    back = field.value()
    if spec.kind == "float":
        assert back == pytest.approx(value, abs=1e-6)
    else:
        assert str(back) == str(value)


def test_a_value_edit_cells_does_not_list_is_shown_not_blanked(page):
    from eubi_bridge.qt_gui.widgets.batch_cell_editor import _Field
    field = _Field(B.spec_for("compressor"), "lz4", True, None)     # from an older CSV
    assert field.value() == "lz4"


def test_the_run_check_reports_settings_the_conversion_would_refuse(page, isolated):
    base = page._ui_to_config()
    src = isolated / "a.tif"
    src.write_bytes(b"II*\x00")
    model = B.BatchModel()
    model.set_baseline(deepcopy(base))
    model.add(base, [str(src)] * 3, str(isolated / "out"))
    for row, out in enumerate(("o1", "o2", "o3")):
        model.update_cells([row], "output_path", str(isolated / out))
    model.update_cells([1], "ome_zarr_version", "0.5")
    model.update_cells([1], "compressor", "bz2")                      # zarr v2 only
    model.rows[2]["metadata_reader"] = "bioformats"                   # an older CSV's value
    problems = model.validate()
    assert any(p.startswith("Row 2:") and "bz2" in p for p in problems), problems
    assert any(p.startswith("Row 3:") and "bioformats" in p for p in problems), problems
    assert not any(p.startswith("Row 1:") for p in problems), problems
