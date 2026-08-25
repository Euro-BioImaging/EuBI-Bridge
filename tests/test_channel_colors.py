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
    pytest.importorskip("PyQt6.QtWidgets")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
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
        assert "conv.channel_intensity_limits" in call,             "unary path ignores the user's channel_intensity_limits setting"
        assert "'from_dtype'" not in call.split("dtype=")[0],             "channel_intensity_limits is hardcoded"

    def test_job_extra_carries_the_parameters(self):
        """The forwarding reads job.extra, so they must survive job building."""
        from eubi_bridge.core.config_models import ConversionJob
        job = ConversionJob.from_kwargs(
            "/in.tif", "/out",
            {"channel_colors": "0,FF0000", "channel_labels": "0,Red"})
        assert job.extra["channel_colors"] == "0,FF0000"
        assert job.extra["channel_labels"] == "0,Red"

    def test_both_paths_forward_the_same_keys(self):
        """Aggregative already worked; the two must not diverge again."""
        import inspect
        from eubi_bridge.conversion import conversion_worker
        source = inspect.getsource(conversion_worker)
        assert source.count(
            "if k in ('channel_labels', 'channel_colors')") == 2


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
        assert snake["conversion"]["channel_colors"] == "0,FF0000;2,00FF00"
        back = snake_to_react(snake)
        assert back["metadata"]["channelColors"] == "0,FF0000;2,00FF00"

    def test_empty_stays_empty(self):
        from eubi_bridge.qt_gui.core.config import react_to_snake, snake_to_react
        snake = react_to_snake(self._react(""))
        assert snake["conversion"]["channel_colors"] == ""
        assert snake_to_react(snake)["metadata"]["channelColors"] == ""

    def test_written_to_disk_and_reloaded(self, tmp_path):
        import json
        from eubi_bridge.qt_gui.core.config import save_config, load_config
        saved = save_config(self._react("0,FF8800;3,00AAFF"),
                            str(tmp_path / "config.json"))
        on_disk = json.loads(Path(saved["_configPath"]).read_text())
        assert on_disk["conversion"]["channel_colors"] == "0,FF8800;3,00AAFF"
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
