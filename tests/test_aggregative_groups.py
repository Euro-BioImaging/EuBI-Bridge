"""Mixed conversion tables: some rows concatenate, others convert alone.

``aggregative_group`` names the group a row belongs to.  Rows sharing a value
become one concatenated output; a blank cell is a plain one-to-one conversion,
which is what every table written before the column existed contains.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eubi_bridge.utils.path_utils import (
    AGGREGATIVE_GROUP_COLUMN, PER_GROUP_CONCAT_KEYS,
    concat_without_group_problems, is_blank_cell, partition_by_group,
    prefix_with_group, resolve_group_concat_params, sanitise_group_name)


class TestSanitiseGroupName:
    """Only the copy that reaches a filename is normalised."""

    def test_plain_names_are_untouched(self):
        assert sanitise_group_name("gr1") == "gr1"
        assert sanitise_group_name("A") == "A"

    def test_path_separators_cannot_create_directories(self):
        assert "/" not in sanitise_group_name("Embryo 1/A")
        assert "\\" not in sanitise_group_name("a\b")

    def test_spaces_become_underscores(self):
        assert sanitise_group_name("Embryo 1") == "Embryo_1"

    def test_surrounding_whitespace_is_dropped(self):
        assert sanitise_group_name("  spaced  ") == "spaced"

    def test_meaningless_names_collapse_to_empty(self):
        assert sanitise_group_name("__") == ""
        assert sanitise_group_name("") == ""

    @pytest.mark.parametrize("blank", [None, float("nan")])
    def test_unset_values_are_not_stringified(self, blank):
        """str(None) is 'None' and str(nan) is 'nan' -- both valid filename
        fragments, so an unset cell must be rejected before stringifying."""
        assert sanitise_group_name(blank) == ""
        assert prefix_with_group("img", blank) == "img"


class TestPrefixWithGroup:
    """The derived name says what was concatenated, so the group goes in front.

    Replacing it would break the common case of one directory yielding several
    outputs (``img_t0_zset``, ``img_t1_zset``), which differ only in that part.
    """

    def test_group_is_prefixed(self):
        assert prefix_with_group("img_t0_zset", "gr1") == "gr1_img_t0_zset"

    def test_sibling_outputs_stay_distinct(self):
        names = [prefix_with_group(n, "M")
                 for n in ("img_t0_zset", "img_t1_zset")]
        assert names == ["M_img_t0_zset", "M_img_t1_zset"]
        assert len(set(names)) == 2

    @pytest.mark.parametrize("group", [None, "", "   ", "__"])
    def test_no_group_leaves_the_name_alone(self, group):
        assert prefix_with_group("img_t0_zset", group) == "img_t0_zset"

    def test_unsafe_group_is_sanitised_in_the_name(self):
        assert prefix_with_group("img", "Embryo 1/A") == "Embryo_1_A_img"


class TestPartitionByGroup:
    def _table(self, groups):
        return pd.DataFrame([
            {"input_path": f"/d/{i}.tif", AGGREGATIVE_GROUP_COLUMN: g}
            for i, g in enumerate(groups)
        ])

    def test_blank_rows_are_unary(self):
        unary, groups = self._table([None, "", "  "]), None
        unary, groups = partition_by_group(unary)
        assert len(unary) == 3
        assert groups == []

    def test_nan_counts_as_blank(self):
        """An empty CSV cell arrives as NaN, not None."""
        unary, groups = partition_by_group(self._table([np.nan]))
        assert len(unary) == 1 and groups == []

    def test_rows_sharing_a_value_form_one_group(self):
        unary, groups = partition_by_group(self._table(["gr1", "gr1", None]))
        assert len(unary) == 1
        assert len(groups) == 1
        assert groups[0][0] == "gr1"
        assert len(groups[0][1]) == 2

    def test_several_groups_are_kept_apart(self):
        _, groups = partition_by_group(self._table(["A", "B", "A", "B"]))
        assert [g for g, _ in groups] == ["A", "B"]
        assert all(len(rows) == 2 for _, rows in groups)

    def test_groups_keep_their_first_appearance_order(self):
        _, groups = partition_by_group(self._table(["B", "A", "B"]))
        assert [g for g, _ in groups] == ["B", "A"]

    def test_a_table_without_the_column_is_all_unary(self):
        """Every table written before the column existed must still work."""
        table = pd.DataFrame([{"input_path": "/d/a.tif"}])
        unary, groups = partition_by_group(table)
        assert len(unary) == 1 and groups == []


class TestJobCarriesTheGroup:
    def test_group_reaches_the_conversion_kwargs(self):
        """The writer needs it to prefix the derived output name."""
        from eubi_bridge.core.config_models import AggregativeConversionJob
        job = AggregativeConversionJob(
            input_path=["/a.tif", "/b.tif"], output_path="/out",
            aggregative_group="gr1")
        assert job.to_conversion_kwargs()["aggregative_group"] == "gr1"

    def test_absent_when_no_group_is_set(self):
        from eubi_bridge.core.config_models import AggregativeConversionJob
        job = AggregativeConversionJob(input_path=["/a.tif"], output_path="/out")
        assert "aggregative_group" not in job.to_conversion_kwargs()


class TestTableWithConcatenationIsAllowed:
    def test_take_filepaths_no_longer_rejects_the_combination(self, tmp_path):
        """Tables and concatenation_axes were once mutually exclusive."""
        from eubi_bridge.utils.path_utils import take_filepaths
        csv = tmp_path / "t.csv"
        csv.write_text("input_path,output_path\n/d/a.tif,/out\n")
        out = take_filepaths(str(csv), concatenation_axes="z")
        assert len(out) == 1


class TestBatchValidationIsGroupAware:
    """Grouped rows share an output on purpose, so the overwrite check skips them."""

    def _model(self, tmp_path, rows):
        from eubi_bridge.qt_gui.core.batch import BatchModel
        model = BatchModel()
        for name, group in rows:
            src = tmp_path / name
            src.write_bytes(b"")
            row = {"input_path": str(src), "output_path": str(tmp_path / "out")}
            if group:
                # A grouped row needs axes too: a group with nothing to
                # concatenate along is refused in its own right, and these
                # tests are about the output-overwrite exemption.
                row[AGGREGATIVE_GROUP_COLUMN] = group
                row["concatenation_axes"] = "z"
            model._rows.append(row)
        return model

    def test_grouped_rows_sharing_an_output_are_accepted(self, tmp_path):
        model = self._model(tmp_path, [("a_z0.tif", "A"), ("a_z1.tif", "A")])
        assert model.validate() == []

    def test_two_groups_may_hold_the_same_filename(self, tmp_path):
        """Their group prefixes keep the outputs apart."""
        (tmp_path / "A").mkdir()
        (tmp_path / "B").mkdir()
        model = self._model(tmp_path, [("A/z0.tif", "A"), ("B/z0.tif", "B")])
        assert model.validate() == []

    def test_ungrouped_duplicates_are_still_caught(self, tmp_path):
        (tmp_path / "A").mkdir()
        (tmp_path / "B").mkdir()
        model = self._model(tmp_path, [("A/z0.tif", None), ("B/z0.tif", None)])
        assert any("overwrite" in p for p in model.validate())


_DEFAULTS = {k: None for k in PER_GROUP_CONCAT_KEYS}


def _group(rows):
    """A group's rows, as columns of concatenation settings."""
    return pd.DataFrame(rows)


class TestPerGroupConcatParams:
    """Concatenation settings belong to a group, so they may differ between
    groups but must agree within one."""

    def test_a_group_uses_its_own_axes(self):
        resolved = resolve_group_concat_params(
            "A", _group([{"concatenation_axes": "z"},
                         {"concatenation_axes": None}]), _DEFAULTS)
        assert resolved["concatenation_axes"] == "z"

    def test_blank_rows_inherit_the_global_value(self):
        """A table that sets nothing behaves as it did before the columns existed."""
        resolved = resolve_group_concat_params(
            "A", _group([{"concatenation_axes": None},
                         {"concatenation_axes": ""}]),
            {**_DEFAULTS, "concatenation_axes": "t"})
        assert resolved["concatenation_axes"] == "t"

    def test_a_row_value_beats_the_global_one(self):
        resolved = resolve_group_concat_params(
            "A", _group([{"concatenation_axes": "z"}]),
            {**_DEFAULTS, "concatenation_axes": "t"})
        assert resolved["concatenation_axes"] == "z"

    def test_a_missing_column_falls_back(self):
        resolved = resolve_group_concat_params(
            "A", _group([{"input_path": "/a.tif"}]),
            {**_DEFAULTS, "concatenation_axes": "z"})
        assert resolved["concatenation_axes"] == "z"

    def test_repeating_the_same_value_is_not_a_conflict(self):
        """Spelling a setting out on every row is natural and must be allowed."""
        resolved = resolve_group_concat_params(
            "A", _group([{"z_tag": "_z"}, {"z_tag": "_z"}]), _DEFAULTS)
        assert resolved["z_tag"] == "_z"

    def test_conflicting_values_are_refused(self):
        with pytest.raises(ValueError) as excinfo:
            resolve_group_concat_params(
                "A", _group([{"concatenation_axes": "z"},
                             {"concatenation_axes": "t"}]), _DEFAULTS)
        message = str(excinfo.value)
        assert "'A'" in message
        assert "concatenation_axes" in message
        assert "'z'" in message and "'t'" in message

    def test_conflicts_are_caught_for_every_concat_key(self):
        for key in PER_GROUP_CONCAT_KEYS:
            with pytest.raises(ValueError, match=key):
                resolve_group_concat_params(
                    "A", _group([{key: "one"}, {key: "two"}]), _DEFAULTS)

    def test_includes_and_excludes_stay_global(self):
        """They filter the input search, which a table has already done."""
        assert "includes" not in PER_GROUP_CONCAT_KEYS
        assert "excludes" not in PER_GROUP_CONCAT_KEYS

    def test_a_conflict_in_one_key_does_not_hide_behind_another(self):
        with pytest.raises(ValueError, match="z_tag"):
            resolve_group_concat_params(
                "A", _group([{"concatenation_axes": "z", "z_tag": "_z"},
                             {"concatenation_axes": "z", "z_tag": "_zz"}]),
                _DEFAULTS)


class TestIsBlankCell:
    @pytest.mark.parametrize("value", [None, "", "   ", np.nan])
    def test_blank_values(self, value):
        assert is_blank_cell(value)

    @pytest.mark.parametrize("value", ["z", 0, "0", False])
    def test_non_blank_values(self, value):
        """0 and False are real settings, not absent ones."""
        assert not is_blank_cell(value)


class TestBatchModelExposesConcatenation:
    """The GUI batch must be able to express grouping, or the CLI support is
    unreachable from the interface."""

    def test_the_group_column_is_a_parameter(self):
        from eubi_bridge.qt_gui.core.batch import _SPEC_BY_KEY
        assert AGGREGATIVE_GROUP_COLUMN in _SPEC_BY_KEY

    def test_every_concat_key_is_editable(self):
        from eubi_bridge.qt_gui.core.batch import _SPEC_BY_KEY
        for key in PER_GROUP_CONCAT_KEYS:
            assert key in _SPEC_BY_KEY, key

    def test_they_share_one_tab(self):
        from eubi_bridge.qt_gui.core.batch import _SPEC_BY_KEY
        tabs = {_SPEC_BY_KEY[k].tab for k in PER_GROUP_CONCAT_KEYS}
        assert tabs == {"Concatenation"}

    def test_concat_settings_are_inert_without_a_group(self):
        """They do nothing on a row that converts on its own, so they grey out."""
        from eubi_bridge.qt_gui.core.batch import BatchModel
        model = BatchModel()
        row = {AGGREGATIVE_GROUP_COLUMN: ""}
        assert model.is_inert(row, "concatenation_axes")
        assert model.is_inert(row, "z_tag")

    def test_they_are_live_once_a_group_is_named(self):
        from eubi_bridge.qt_gui.core.batch import BatchModel
        model = BatchModel()
        row = {AGGREGATIVE_GROUP_COLUMN: "A"}
        assert not model.is_inert(row, "concatenation_axes")
        assert not model.is_inert(row, "z_tag")

    def test_the_greying_reason_is_readable(self):
        """The tooltip must explain the state, not leak the sentinel value."""
        from eubi_bridge.qt_gui.core.batch import BatchModel
        reason = BatchModel().inert_reason({AGGREGATIVE_GROUP_COLUMN: ""}, "z_tag")
        assert reason == "aggregative_group is empty"

    def test_aggregative_conversions_are_batchable(self):
        """The old guard refused any batch with concatenation configured."""
        from eubi_bridge.qt_gui.core.batch import can_batch
        ok, _ = can_batch({"concatenation": {"concatenationAxes": "z"}})
        assert ok


class TestGroupIsAFirstClassParameter:
    """It must exist everywhere a concatenation setting exists, not only in the
    batch table: config file, CLI command, to_zarr argument, and GUI."""

    def test_the_config_model_has_it(self):
        from eubi_bridge.core.config_models import ConcatenationConfig
        assert "aggregative_group" in ConcatenationConfig().model_dump()

    def test_it_defaults_to_none(self):
        """None means no prefix, which is what every earlier run did."""
        from eubi_bridge.core.config_models import ConcatenationConfig
        assert ConcatenationConfig().aggregative_group is None

    def test_the_installation_defaults_have_it(self):
        from eubi_bridge.ebridge import ConfigManager
        assert "aggregative_group" in ConfigManager._ROOT_DEFAULTS["concatenation"]

    def test_the_configure_command_accepts_it(self):
        import inspect
        from eubi_bridge.ebridge import ConfigManager
        params = inspect.signature(
            ConfigManager.configure_concatenation).parameters
        assert "aggregative_group" in params

    def test_to_zarr_accepts_it(self):
        """Both the manager and the entry point users actually call."""
        import inspect
        from eubi_bridge.ebridge import ConversionManager, EuBIBridge
        for cls in (ConversionManager, EuBIBridge):
            params = inspect.signature(cls.to_zarr).parameters
            assert "aggregative_group" in params, cls.__name__

    def test_the_gui_maps_it_both_ways(self):
        """A one-directional mapping silently drops the value on save."""
        import inspect
        import eubi_bridge.qt_gui.server.config_manager as cm
        source = inspect.getsource(cm)
        assert "aggregativeGroup" in source
        assert '"aggregative_group":' in source


class TestConfigBackfill:
    """A config file written before a parameter existed must gain it on load,
    or the parameter stays permanently invisible to existing users."""

    def test_a_missing_key_is_added_to_an_existing_section(self, tmp_path):
        import json
        from eubi_bridge.ebridge import ConfigManager
        path = tmp_path / ".eubi_config.json"
        # A config from before the field existed: the section is there already,
        # so the old "add the section if absent" rule would never fire.
        path.write_text(json.dumps({
            "concatenation": {"concatenation_axes": None, "time_tag": None,
                              "channel_tag": None, "z_tag": None,
                              "y_tag": None, "x_tag": None},
        }))
        manager = ConfigManager(str(path))
        assert "aggregative_group" in manager.config["concatenation"]

    def test_an_existing_value_is_not_overwritten(self, tmp_path):
        import json
        from eubi_bridge.ebridge import ConfigManager
        path = tmp_path / ".eubi_config.json"
        path.write_text(json.dumps({
            "concatenation": {"aggregative_group": "keepme"},
        }))
        manager = ConfigManager(str(path))
        assert manager.config["concatenation"]["aggregative_group"] == "keepme"

    def test_a_missing_section_is_still_added(self, tmp_path):
        import json
        from eubi_bridge.ebridge import ConfigManager
        path = tmp_path / ".eubi_config.json"
        path.write_text(json.dumps({"cluster": {}}))
        manager = ConfigManager(str(path))
        assert "concatenation" in manager.config


class TestConcatWithoutGroupIsRefused:
    """Axes without a group describe a concatenation that will not happen.

    ``aggregative_group`` is what marks a row as aggregative, so a row carrying
    axes but no group converts one-to-one and ignores them -- a wrong result
    rather than a preference, which is why it blocks instead of warning.
    """

    def _table(self, rows):
        return pd.DataFrame(rows)

    def test_axes_without_a_group_are_refused(self):
        problems = concat_without_group_problems(self._table(
            [{AGGREGATIVE_GROUP_COLUMN: None, "concatenation_axes": "z"}]), {})
        assert problems and "no aggregative group" in problems[0]

    def test_axes_with_a_group_are_accepted(self):
        assert concat_without_group_problems(self._table(
            [{AGGREGATIVE_GROUP_COLUMN: "A", "concatenation_axes": "z"}]), {}) == []

    def test_a_plain_unary_row_is_accepted(self):
        """The common case: most rows in a batch concatenate with nothing."""
        assert concat_without_group_problems(self._table(
            [{AGGREGATIVE_GROUP_COLUMN: None, "concatenation_axes": None}]), {}) == []

    def test_a_table_without_the_columns_is_accepted(self):
        """Every table written before the columns existed must still run."""
        assert concat_without_group_problems(
            self._table([{"input_path": "/a.tif"}]), {}) == []

    def test_run_wide_axes_are_caught_too(self):
        """`eubi to_zarr table.csv --concatenation_axes z` is the same mistake."""
        problems = concat_without_group_problems(
            self._table([{AGGREGATIVE_GROUP_COLUMN: None,
                          "concatenation_axes": None}]),
            {"concatenation_axes": "z"})
        assert problems

    def test_run_wide_axes_are_fine_once_grouped(self):
        assert concat_without_group_problems(
            self._table([{AGGREGATIVE_GROUP_COLUMN: "A",
                          "concatenation_axes": None}]),
            {"concatenation_axes": "z"}) == []

    def test_a_stray_tag_alone_is_not_refused(self):
        """Only the axes declare a concatenation; leftover tags are harmless."""
        assert concat_without_group_problems(self._table(
            [{AGGREGATIVE_GROUP_COLUMN: None, "concatenation_axes": None,
              "z_tag": "_z"}]), {}) == []

    def test_a_group_without_axes_is_refused(self):
        """The mirror image, and just as silent: today it dies deep in the
        dispatcher as "commonpath() arg is an empty sequence"."""
        problems = concat_without_group_problems(self._table(
            [{AGGREGATIVE_GROUP_COLUMN: "A", "concatenation_axes": None}]), {})
        assert problems and "no concatenation axes" in problems[0]

    def test_a_group_with_run_wide_axes_is_accepted(self):
        """Axes given once on the command line still count for every group."""
        assert concat_without_group_problems(self._table(
            [{AGGREGATIVE_GROUP_COLUMN: "A", "concatenation_axes": None}]),
            {"concatenation_axes": "z"}) == []

    def test_both_mistakes_are_reported_separately(self):
        problems = concat_without_group_problems(self._table([
            {AGGREGATIVE_GROUP_COLUMN: None, "concatenation_axes": "z"},
            {AGGREGATIVE_GROUP_COLUMN: "B", "concatenation_axes": None},
        ]), {})
        assert len(problems) == 2

    def test_offending_rows_are_named(self):
        problems = concat_without_group_problems(self._table([
            {AGGREGATIVE_GROUP_COLUMN: "A", "concatenation_axes": "z"},
            {AGGREGATIVE_GROUP_COLUMN: None, "concatenation_axes": "z"},
        ]), {})
        assert "Row 2" in problems[0]

    def test_many_offenders_are_summarised(self):
        """A 2000-row batch must not print 2000 row numbers."""
        problems = concat_without_group_problems(self._table(
            [{AGGREGATIVE_GROUP_COLUMN: None, "concatenation_axes": "z"}] * 9), {})
        assert "+4 more" in problems[0]


class TestGuardReachesBothEntryPoints:
    """The same mistake is possible from a hand-written CSV and from the GUI."""

    def test_the_cli_refuses_such_a_table(self, tmp_path):
        from eubi_bridge.ebridge import EuBIBridge
        csv_path = tmp_path / "b.csv"
        csv_path.write_text(
            "input_path,output_path,aggregative_group,concatenation_axes\n"
            f"{tmp_path / 'a.tif'},{tmp_path / 'out'},,z\n")
        with pytest.raises(ValueError, match="no aggregative group"):
            EuBIBridge().to_zarr(str(csv_path), verbose=False)

    def test_the_batch_model_reports_it(self, tmp_path):
        """Covers the edit-time route: valid when added, cleared afterwards."""
        from eubi_bridge.qt_gui.core.batch import BatchModel
        source = tmp_path / "a.tif"
        source.write_bytes(b"")
        model = BatchModel()
        model._rows.append({
            "input_path": str(source), "output_path": str(tmp_path / "out"),
            AGGREGATIVE_GROUP_COLUMN: None, "concatenation_axes": "z",
        })
        assert any("no aggregative group" in p for p in model.validate())
