"""Automatic channel colours and per-channel overrides.

The first channels use the conventional microscopy palette; beyond that colours
are generated so that a many-channel image stays readable.  The previous
arithmetic fallback produced near-duplicates (channels 7 and 13 were both dark
blue) and very dark colours.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.conftest import qt_available
from eubi_bridge.utils.metadata_utils import (
    DEFAULT_CHANNEL_COLORS, auto_channel_color)


def _rgb(hex_code: str) -> tuple[int, int, int]:
    return tuple(int(hex_code[i:i + 2], 16) for i in (0, 2, 4))


def _distance(a: str, b: str) -> float:
    return sum((x - y) ** 2 for x, y in zip(_rgb(a), _rgb(b))) ** 0.5


@pytest.fixture
def page():
    """A fresh ConvertPage on a QApplication shared by the whole module.

    ConvertPage spawns background helpers, so it is torn down after each test;
    creating one per test without cleanup crashes the interpreter.
    """
    if not qt_available():
        pytest.skip("PyQt6 unavailable or no usable Qt platform plugin")
    from PyQt6.QtWidgets import QApplication
    from eubi_bridge.qt_gui.pages.convert_page import ConvertPage
    app = QApplication.instance() or QApplication([])
    widget = ConvertPage()
    yield widget
    widget.deleteLater()
    app.processEvents()


class TestAutoColors:
    def test_conventional_colors_come_first(self):
        """Users expect channel 0 red, 1 green, 2 blue."""
        for index, expected in enumerate(DEFAULT_CHANNEL_COLORS):
            assert auto_channel_color(index) == expected

    def test_every_color_is_valid_hex(self):
        for index in range(40):
            code = auto_channel_color(index)
            assert len(code) == 6
            assert int(code, 16) >= 0

    def test_no_duplicates_across_many_channels(self):
        colors = [auto_channel_color(i) for i in range(40)]
        assert len(set(colors)) == len(colors)

    def test_colors_stay_visually_separable(self):
        """The old formula put channels 7 and 13 at distance 18."""
        colors = [auto_channel_color(i) for i in range(20)]
        closest = min(_distance(colors[i], colors[j])
                      for i in range(len(colors))
                      for j in range(i + 1, len(colors)))
        assert closest > 40, f"two channels are only {closest:.0f} apart"

    def test_no_near_black_colors(self):
        """A near-black channel is invisible against the usual background."""
        for index in range(len(DEFAULT_CHANNEL_COLORS), 40):
            r, g, b = _rgb(auto_channel_color(index))
            assert 0.2126 * r + 0.7152 * g + 0.0722 * b > 30

    def test_is_deterministic(self):
        assert [auto_channel_color(i) for i in range(15)] == \
               [auto_channel_color(i) for i in range(15)]

    def test_both_metadata_modules_agree(self):
        """Two copies of the palette existed; they must not drift apart."""
        import numpy as np
        from eubi_bridge.utils.metadata_utils import generate_channel_metadata
        from eubi_bridge.ngff.multiscales import (
            generate_channel_metadata as ngff_variant)
        mine = [c["color"] for c in generate_channel_metadata(12, np.uint16)]
        theirs = [c["color"]
                  for c in ngff_variant(12, np.uint16)["omero"]["channels"]]
        assert mine == theirs


class TestColorOverrides:
    """`channel_colors` uses the CLI's "idx,RRGGBB;..." format."""

    def _parse(self, value):
        from eubi_bridge.utils.metadata_utils import ChannelParser
        return ChannelParser(manager=None)._parse_indexed_string(value)

    def test_parses_index_color_pairs(self):
        assert self._parse("0,FF0000;2,00FF00") == {0: "FF0000", 2: "00FF00"}

    def test_empty_means_all_automatic(self):
        assert self._parse("") == {}
        assert self._parse(None) == {}

    def test_unlisted_channels_are_left_alone(self):
        """Only the named indices are overridden; the rest stay automatic."""
        assert 1 not in self._parse("0,FF0000;2,00FF00")


class TestUnaryForwarding:
    """The unary path must forward channel parameters to parse_channels.

    Regression guard: it previously did not, so ``--channel_colors`` and
    ``--channel_labels`` silently did nothing for single-file conversions while
    the aggregative path honoured them.  It also hardcoded
    ``channel_intensity_limits``, overriding the user's choice.
    """

    def _unary_call_source(self) -> str:
        import inspect
        from eubi_bridge.conversion import conversion_worker
        source = inspect.getsource(conversion_worker._process_single_scene)
        start = source.index("parse_channels(")
        return source[start:start + 600]

    def test_channel_kwargs_are_forwarded(self):
        call = self._unary_call_source()
        assert "channel_colors" in call,             "unary path drops channel_colors; the CLI flag would do nothing"
        assert "channel_labels" in call,             "unary path drops channel_labels; the CLI flag would do nothing"

    def test_intensity_limits_are_not_hardcoded(self):
        call = self._unary_call_source()
        assert "meta.channel_intensity_limits" in call,             "unary path ignores the user's channel_intensity_limits setting"
        assert "'from_dtype'" not in call.split("dtype=")[0],             "channel_intensity_limits is hardcoded"

    def test_the_job_carries_the_parameters(self):
        """They must survive job building, wherever they are stored.

        They now belong to MetadataConfig rather than riding in ``extra`` as
        unrecognised keys, so the typed field is the address; a per-row value
        from a conversion table still arrives via ``extra`` and wins.
        """
        from eubi_bridge.core.config_models import ConversionJob
        job = ConversionJob.from_kwargs(
            "/in.tif", "/out",
            {"channel_colors": "0,FF0000", "channel_labels": "0,Red"})
        assert job.metadata.channel_colors == "0,FF0000"
        assert job.metadata.channel_labels == "0,Red"

    def test_both_paths_forward_the_same_keys(self):
        """Aggregative already worked; the two must not diverge again."""
        import inspect
        from eubi_bridge.conversion import conversion_worker
        source = inspect.getsource(conversion_worker)
        # Both parse_channels call sites pass each key explicitly, preferring a
        # per-row override from job.extra over the configured default.
        assert source.count("channel_labels=job.extra.get(") == 2
        assert source.count("channel_colors=job.extra.get(") == 2


class TestGuiSerialisation:
    """The GUI's colour rows serialise to the format the CLI accepts.

    Each row has an "Override existing" toggle, unticked by default: the source
    file's colour wins, or the automatic palette when the file specifies none.
    Ticking it replaces whatever the source said.
    """

    def test_nothing_is_overridden_by_default(self, page):
        assert all(not r["override"].isChecked()
                   for r in page._channel_colour_rows)
        assert page._channel_colours_to_string() == ""

    def test_swatch_is_inactive_until_override_is_ticked(self, page):
        row = page._channel_colour_rows[0]
        assert not row["swatch"].isEnabled()
        row["override"].setChecked(True)
        assert row["swatch"].isEnabled()
        row["override"].setChecked(False)
        assert not row["swatch"].isEnabled()

    def test_only_ticked_rows_are_written(self, page):
        rows = page._channel_colour_rows
        rows[0]["override"].setChecked(True)
        rows[0]["hex"] = "123456"
        rows[2]["override"].setChecked(True)
        rows[2]["hex"] = "ABCDEF"
        assert page._channel_colours_to_string() == "0,123456;2,ABCDEF"

    def test_round_trip_through_config(self, page):
        rows = page._channel_colour_rows
        rows[1]["override"].setChecked(True)
        rows[1]["hex"] = "FF8800"
        before = page._channel_colours_to_string()

        config = page._ui_to_config()
        assert config["metadata"]["channelColors"] == before
        page._load_config_to_ui(config)
        assert page._channel_colours_to_string() == before

    def test_loading_restores_the_toggle_state(self, page):
        page._load_channel_colours("2,00FF00")
        rows = page._channel_colour_rows
        assert rows[2]["override"].isChecked()
        assert rows[2]["swatch"].isEnabled()
        assert not rows[0]["override"].isChecked()
        assert not rows[0]["swatch"].isEnabled()

    def test_loading_grows_rows_for_higher_indices(self, page):
        page._load_channel_colours("11,FF00FF")
        assert len(page._channel_colour_rows) >= 12
        assert page._channel_colours_to_string() == "11,FF00FF"

    def test_untouched_rows_preview_the_automatic_colour(self, page):
        """The greyed swatch must not show a stale or misleading colour."""
        for row in page._channel_colour_rows:
            assert row["hex"] == auto_channel_color(row["index"])


class TestConfigPersistence:
    """Channel colours must survive Save Config and come back on load.

    The GUI config is camelCase and the file is snake_case, so both mapping
    directions have to know the key; a missing entry silently drops it.
    """

    def _react(self, colours):
        return {
            "cluster": {}, "reader": {}, "downscaling": {},
            "concatenation": {}, "conversion": {},
            "metadata": {"channelColors": colours, "metadataReader": "bfio"},
        }

    def test_mapping_round_trips(self):
        from eubi_bridge.qt_gui.core.config import react_to_snake, snake_to_react
        snake = react_to_snake(self._react("0,FF0000;2,00FF00"))
        assert snake["metadata"]["channel_colors"] == "0,FF0000;2,00FF00"
        back = snake_to_react(snake)
        assert back["metadata"]["channelColors"] == "0,FF0000;2,00FF00"

    def test_empty_stays_empty(self):
        from eubi_bridge.qt_gui.core.config import react_to_snake, snake_to_react
        snake = react_to_snake(self._react(""))
        assert snake["metadata"]["channel_colors"] == ""
        assert snake_to_react(snake)["metadata"]["channelColors"] == ""

    def test_written_to_disk_and_reloaded(self, tmp_path):
        import json
        from eubi_bridge.qt_gui.core.config import save_config, load_config
        saved = save_config(self._react("0,FF8800;3,00AAFF"),
                            str(tmp_path / "config.json"))
        on_disk = json.loads(Path(saved["_configPath"]).read_text())
        assert on_disk["metadata"]["channel_colors"] == "0,FF8800;3,00AAFF"
        reloaded = load_config(saved["_configPath"])
        assert reloaded["metadata"]["channelColors"] == "0,FF8800;3,00AAFF"

    def test_survives_a_full_gui_cycle(self, page, tmp_path):
        """Tick two overrides, save, then load into a fresh page."""
        from eubi_bridge.qt_gui.core.config import save_config, load_config
        rows = page._channel_colour_rows
        rows[0]["override"].setChecked(True)
        rows[0]["hex"] = "FF8800"
        rows[3]["override"].setChecked(True)
        rows[3]["hex"] = "00AAFF"
        expected = page._channel_colours_to_string()

        saved = save_config(page._ui_to_config(),
                            str(tmp_path / "config.json"))
        page._load_config_to_ui(load_config(saved["_configPath"]))

        assert page._channel_colours_to_string() == expected
        assert page._channel_colour_rows[0]["override"].isChecked()
        assert not page._channel_colour_rows[1]["override"].isChecked()


class TestPhysicalScaleConfig:
    """Pixel sizes and units are config parameters, not loose kwargs.

    They reached the worker through ``job.extra`` as unrecognised keys, so the
    config file -- which users read to learn the exact key spellings -- never
    mentioned them.  They are typed fields now; ``extra`` still wins so a
    per-row override from a conversion table is not overruled by the default.
    """

    _AXES = ("time", "z", "y", "x")

    def test_every_axis_has_a_scale_and_unit(self):
        from eubi_bridge.core.config_models import MetadataConfig
        fields = set(MetadataConfig.model_fields)
        for axis in self._AXES:
            assert f"{axis}_scale" in fields
            assert f"{axis}_unit" in fields

    def test_channels_have_neither(self):
        """A channel has no physical extent, so the keys would be meaningless."""
        from eubi_bridge.core.config_models import MetadataConfig
        fields = set(MetadataConfig.model_fields)
        assert "channel_scale" not in fields
        assert "channel_unit" not in fields

    def test_they_default_to_none(self):
        """None means "keep what the file says" -- no 'auto' sentinel needed,
        since a blank cell inheriting the config resolves to the same thing."""
        from eubi_bridge.core.config_models import MetadataConfig
        dumped = MetadataConfig().model_dump()
        for axis in self._AXES:
            assert dumped[f"{axis}_scale"] is None
            assert dumped[f"{axis}_unit"] is None

    def test_the_config_file_carries_them(self):
        """The config doubles as the parameter reference, so absent = invisible."""
        from eubi_bridge.ebridge import ConfigManager
        section = ConfigManager._ROOT_DEFAULTS["metadata"]
        for axis in self._AXES:
            assert f"{axis}_scale" in section
            assert f"{axis}_unit" in section

    def test_the_job_carries_them_typed(self):
        from eubi_bridge.core.config_models import ConversionJob
        job = ConversionJob.from_kwargs(
            "/in.tif", "/out", {"z_scale": 0.25, "z_unit": "nanometer"})
        assert job.metadata.z_scale == 0.25
        assert job.metadata.z_unit == "nanometer"
        assert "z_scale" not in job.extra

    def test_the_parser_reads_the_configured_value(self):
        """The parser fed the writer from job.extra alone, so once these became
        typed fields a configured scale reached it as nothing at all."""
        from eubi_bridge.conversion.conversion_worker import _axis_metadata_kwargs
        from eubi_bridge.core.config_models import ConversionJob
        job = ConversionJob.from_kwargs(
            "/in.tif", "/out", {"z_scale": 0.25, "z_unit": "nanometer"})
        parsed = _axis_metadata_kwargs(job)
        assert parsed["z_scale"] == 0.25
        assert parsed["z_unit"] == "nanometer"

    def test_unset_axes_stay_out_of_the_parsed_kwargs(self):
        """Absent means "keep the file's value"; a None would override it."""
        from eubi_bridge.conversion.conversion_worker import _axis_metadata_kwargs
        from eubi_bridge.core.config_models import ConversionJob
        job = ConversionJob.from_kwargs("/in.tif", "/out", {"z_scale": 0.25})
        assert "y_scale" not in _axis_metadata_kwargs(job)

    def test_a_row_override_still_wins(self):
        """job.extra is where a conversion table's per-row value arrives."""
        from eubi_bridge.conversion.conversion_worker import _axis_metadata_kwargs
        from eubi_bridge.core.config_models import ConversionJob
        job = ConversionJob.from_kwargs("/in.tif", "/out", {"z_scale": 0.5})
        job.extra["z_scale"] = 0.125
        assert _axis_metadata_kwargs(job)["z_scale"] == 0.125

    def test_the_channel_axis_is_not_dropped_from_scales(self):
        """Regression: making channel_scale unsettable removed 'c' from the
        scales tuple entirely, leaving it shorter than the axes it describes
        and raising KeyError('x') deep in the writer."""
        from eubi_bridge.conversion.conversion_worker import _parse_axis_params

        class _M:
            axes = "tczyx"
            scaledict = {"t": 1.0, "c": 1.0, "z": 2.0, "y": 3.0, "x": 4.0}

        manager = _M()
        scales = _parse_axis_params(manager, {}, 2, manager.scaledict)
        assert len(scales) == len(manager.axes)
        assert scales == (1.0, 1.0, 2.0, 3.0, 4.0)


class TestPhysicalScaleGuiMapping:
    """Both mapping directions must know the keys, or a saved value vanishes."""

    def _snake(self, **overrides):
        base = {"metadata": {"metadata_reader": "bfio",
                             "channel_intensity_limits": "from_dtype",
                             "channel_colors": "", "channel_labels": ""}}
        base["metadata"].update(overrides)
        return base

    def test_a_stored_scale_comes_back_ticked(self):
        """Otherwise the form shows a value while claiming it is not applied."""
        from eubi_bridge.qt_gui.server.config_manager import _config_to_react
        react = _config_to_react(self._snake(z_scale=0.5))["metadata"]
        assert react["overridePhysicalScale"] is True
        assert react["scaleZ"] == "0.5"

    def test_nothing_stored_stays_unticked(self):
        from eubi_bridge.qt_gui.server.config_manager import _config_to_react
        react = _config_to_react(self._snake())["metadata"]
        assert react["overridePhysicalScale"] is False
        assert react["scaleZ"] == ""

    def _react(self, **overrides):
        """A form payload, as the GUI hands it over on Save Config."""
        meta = {"metadataReader": "bfio", "channelIntensityLimits": "from_datatype",
                "channelColors": "", "channelLabels": "",
                "overridePhysicalScale": False,
                "scaleTime": "", "scaleZ": "", "scaleY": "", "scaleX": "",
                "unitTime": "second", "unitZ": "micrometer",
                "unitY": "micrometer", "unitX": "micrometer"}
        meta.update(overrides)
        return {"cluster": {}, "reader": {}, "conversion": {},
                "downscaling": {}, "concatenation": {}, "metadata": meta}

    def test_a_form_scale_is_written_to_the_config(self):
        """Starts from the form, so it binds the react -> snake direction; a
        round trip that starts from snake would pass without it."""
        from eubi_bridge.qt_gui.server.config_manager import _react_to_config
        snake = _react_to_config(self._react(
            overridePhysicalScale=True, scaleZ="0.5", unitZ="nanometer"))
        assert snake["metadata"]["z_scale"] == 0.5
        assert snake["metadata"]["z_unit"] == "nanometer"
        assert snake["metadata"]["y_scale"] is None

    def test_the_round_trip_preserves_a_scale(self):
        from eubi_bridge.qt_gui.server.config_manager import (
            _config_to_react, _react_to_config)
        back = _react_to_config(
            _config_to_react(self._snake(z_scale=0.5, z_unit="micrometer")))
        assert back["metadata"]["z_scale"] == 0.5
        assert back["metadata"]["z_unit"] == "micrometer"
        assert back["metadata"]["y_scale"] is None

    def test_an_unticked_override_stores_nothing(self):
        """The toggle is the authority: a stale number must not leak through."""
        from eubi_bridge.qt_gui.server.config_manager import _react_to_config
        snake = _react_to_config(self._react(
            overridePhysicalScale=False, scaleZ="0.5"))
        assert snake["metadata"]["z_scale"] is None
