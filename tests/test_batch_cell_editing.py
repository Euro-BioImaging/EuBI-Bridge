"""Tests for per-cell editing of a queued batch.

Covers ``BatchModel.update_cells`` / ``reset_cells`` / ``common_value`` and the
``ParamSpec`` table behind the Edit Cells dialog.  All of this is pure model
logic, so it runs without Qt.
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eubi_bridge.qt_gui.core.batch import (
    BatchModel, spec_for, uneditable_reason, sort_keys, grouped_specs,
    column_header, parameter_tabs, with_separators, SEPARATOR,
    _PARAM_SPECS, _PATH_COLUMNS, _CLUSTER_KEYS,
)
from eubi_bridge.qt_gui.workers.conversion_worker import _build_kwargs


def _config(**conversion) -> dict:
    """A minimal GUI config dict, with conversion overrides applied."""
    conv = {
        "omeZarrVersion": "0.4",
        "dataType": "auto",
        "autoChunk": True,
        "targetChunkSizeMb": 32,
        "squeezeDimensions": True,
        "saveOmeXml": True,
        "overwrite": False,
    }
    conv.update(conversion)
    return {
        "cluster": {}, "reader": {}, "downscaling": {}, "metadata": {},
        "concatenation": {}, "conversion": conv,
    }


@pytest.fixture
def batch() -> BatchModel:
    """A batch of three rows sharing one baseline."""
    model = BatchModel()
    cfg = _config()
    model.set_baseline(deepcopy(cfg))
    model.add(cfg, ["a.tif", "b.tif", "c.tif"], "/out")
    return model


# ── the spec table ────────────────────────────────────────────────────────────

class TestParamSpecs:
    def test_specs_are_unique(self):
        keys = [s.key for s in _PARAM_SPECS]
        assert len(keys) == len(set(keys))

    def test_every_spec_kind_is_known(self):
        assert {s.kind for s in _PARAM_SPECS} <= {
            "bool", "int", "float", "choice", "text", "auto_int",
            "file", "directory"}

    def test_choice_specs_offer_choices(self):
        for spec in _PARAM_SPECS:
            if spec.kind == "choice":
                assert spec.choices, f"{spec.key} is a choice with no options"

    def test_specs_are_row_overridable(self):
        """Nothing offered as editable may be blocked by the model."""
        for spec in _PARAM_SPECS:
            assert uneditable_reason(spec.key) is None, spec.key


class TestUneditableReason:
    @pytest.mark.parametrize("key", [
        "time_range", "channel_range", "z_range", "y_range", "x_range"])
    def test_dim_ranges_are_batch_wide(self, key):
        assert "whole batch" in uneditable_reason(key)

    @pytest.mark.parametrize("key", ["max_workers", "on_slurm", "jvm_memory"])
    def test_cluster_settings_rejected(self, key):
        assert "cluster" in uneditable_reason(key)

    def test_compressor_params_defers_to_compressor(self):
        assert "compressor" in uneditable_reason("compressor_params")

    def test_editable_key_has_no_reason(self):
        assert uneditable_reason("dtype") is None

    def test_unknown_key_rejected(self):
        assert uneditable_reason("no_such_param") is not None


# ── editing ───────────────────────────────────────────────────────────────────

class TestUpdateCells:
    def test_updates_only_selected_rows(self, batch):
        batch.update_cells([0, 2], "dtype", "uint16")
        assert batch.rows[0]["dtype"] == "uint16"
        assert batch.rows[2]["dtype"] == "uint16"
        assert batch.rows[1]["dtype"] == "auto"

    def test_edited_cell_is_rendered_in_sparse_view(self, batch):
        """A value differing from the baseline must survive sparse rendering."""
        batch.update_cells([1], "dtype", "uint16")
        assert batch.cell(batch.rows[1], "dtype") == "uint16"
        # Untouched rows still inherit, so their cell stays blank.
        assert batch.cell(batch.rows[0], "dtype") is None

    def test_edited_column_appears(self, batch):
        assert "dtype" not in batch.columns()
        batch.update_cells([0], "dtype", "uint16")
        assert "dtype" in batch.columns()

    def test_rejects_non_row_overridable(self, batch):
        with pytest.raises(KeyError):
            batch.update_cells([0], "z_range", "0,10")

    def test_rejects_cluster_key(self, batch):
        with pytest.raises(KeyError):
            batch.update_cells([0], "max_workers", 8)

    def test_rejects_unstorable_value(self, batch):
        with pytest.raises(ValueError):
            batch.update_cells([0], "dtype", object())

    def test_bool_and_int_round_trip(self, batch):
        batch.update_cells([0], "squeeze", False)
        batch.update_cells([0], "z_chunk", 64)
        assert batch.rows[0]["squeeze"] is False
        assert batch.rows[0]["z_chunk"] == 64


def _blosc_config():
    cfg = _config()
    cfg["conversion"]["compression"] = {
        "codec": "blosc", "level": 5,
        "bloscInnerCodec": "lz4", "bloscShuffle": "shuffle",
    }
    return cfg


class TestCoupledKeys:
    """A codec and its parameters must never desynchronise.

    numcodecs raises ``TypeError: GZip.__init__() got an unexpected keyword
    argument 'cname'`` if blosc parameters survive a switch to gzip.
    """

    def _model(self):
        cfg = _blosc_config()
        model = BatchModel()
        model.set_baseline(deepcopy(cfg))
        model.add(cfg, ["a.tif", "b.tif"], "/out")
        return model

    def test_switching_codec_replaces_its_parameters(self):
        model = self._model()
        model.update_cells([1], "compressor", "gzip")
        params = model.rows[1]["compressor_params"]
        assert params == {"level": 5}
        assert "cname" not in params, "blosc params survived a switch to gzip"

    def test_unchanged_codec_keeps_its_parameters(self):
        model = self._model()
        model.update_cells([1], "compressor", "blosc")
        assert model.rows[1]["compressor_params"].get("cname") == "lz4"

    def test_changed_params_reach_the_csv(self):
        """Without a column the row would inherit the baseline's blosc params."""
        model = self._model()
        model.update_cells([1], "compressor", "gzip")
        assert "compressor_params" in model.columns(for_csv=True)

    @pytest.mark.parametrize("codec,expected", [
        ("blosc", {"cname", "clevel", "shuffle"}),
        ("gzip", {"level"}),
        ("zstd", {"level"}),
        ("none", set()),
    ])
    def test_defaults_match_the_codec(self, codec, expected):
        from eubi_bridge.qt_gui.core.batch import default_compressor_params
        assert set(default_compressor_params(codec)) == expected


class TestResetCells:
    def test_reset_restores_baseline(self, batch):
        batch.update_cells([0], "dtype", "uint16")
        batch.reset_cells([0], "dtype")
        assert batch.cell(batch.rows[0], "dtype") is None
        assert batch.rows[0]["dtype"] == "auto"

    def test_reset_drops_column_from_sparse_view(self, batch):
        batch.update_cells([0], "dtype", "uint16")
        batch.reset_cells([0], "dtype")
        assert "dtype" not in batch.columns()

    def test_reset_leaves_other_rows_alone(self, batch):
        batch.update_cells([0, 1], "dtype", "uint16")
        batch.reset_cells([0], "dtype")
        assert batch.cell(batch.rows[1], "dtype") == "uint16"

    def test_reset_rejects_non_overridable(self, batch):
        with pytest.raises(KeyError):
            batch.reset_cells([0], "y_range")


class TestCommonValue:
    def test_agrees_when_rows_match(self, batch):
        batch.update_cells([0, 1], "dtype", "uint16")
        assert batch.common_value([0, 1], "dtype") == ("uint16", True)

    def test_disagrees_when_rows_differ(self, batch):
        batch.update_cells([0], "dtype", "uint16")
        batch.update_cells([1], "dtype", "float32")
        value, agreed = batch.common_value([0, 1], "dtype")
        assert not agreed and value is None

    def test_empty_selection_disagrees(self, batch):
        assert batch.common_value([], "dtype") == (None, False)

    def test_single_row_always_agrees(self, batch):
        value, agreed = batch.common_value([2], "dtype")
        assert agreed and value == "auto"


class TestPersistence:
    def test_edits_survive_save_and_load(self, batch, tmp_path):
        batch.update_cells([1], "dtype", "uint16")
        csv_path = batch.save(str(tmp_path / "batch.csv"))
        reloaded = BatchModel.load(csv_path)
        assert reloaded.rows[1]["dtype"] == "uint16"


# ── presentation order ────────────────────────────────────────────────────────

class TestOrdering:
    """Columns and dialog fields follow the conversion form, not the alphabet."""

    def test_path_columns_lead(self):
        assert sort_keys(["dtype", "output_path", "input_path"])[:2] ==             list(_PATH_COLUMNS)

    def test_form_order_not_alphabetical(self):
        """scene_index (Reader) precedes dtype (Conversion) despite d < s."""
        assert sort_keys(["dtype", "scene_index"]) == ["scene_index", "dtype"]

    def test_related_params_stay_adjacent(self):
        """Chunk axes keep their t,c,z,y,x order rather than sorting by name."""
        keys = [f"{ax}_chunk" for ax in ("x", "z", "time", "y", "channel")]
        assert sort_keys(keys) == [
            "time_chunk", "channel_chunk", "z_chunk", "y_chunk", "x_chunk"]

    def test_unknown_keys_sort_last_but_survive(self):
        out = sort_keys(["zzz_custom", "dtype", "aaa_custom"])
        assert out[0] == "dtype"
        assert set(out[1:]) == {"aaa_custom", "zzz_custom"}

    def test_sort_is_stable_and_total(self):
        keys = [s.key for s in _PARAM_SPECS]
        assert sort_keys(keys) == keys

    def test_queue_columns_follow_form_order(self, batch):
        batch.update_cells([0], "dtype", "uint16")
        batch.update_cells([0], "scene_index", "1")
        columns = [c for c in batch.columns() if c not in _PATH_COLUMNS]
        assert columns == ["scene_index", "dtype"]


class TestGroupedSpecs:
    """Nesting mirrors the conversion form: tab, then group box within it."""

    def test_nests_tab_then_group(self):
        """The example from review: chunk x/y + auto-chunk + auto-detect layers."""
        result = grouped_specs(
            ["x_chunk", "y_chunk", "auto_chunk", "n_layers"])
        assert [tab for tab, _ in result] == ["Conversion", "Downscaling"]

        conversion = dict(result)["Conversion"]
        assert [g for g, _ in conversion] == ["Chunking"]
        assert [s.key for s in dict(conversion)["Chunking"]] == [
            "auto_chunk", "y_chunk", "x_chunk"]

        # n_layers sits loose on the Downscaling tab, as it does on the form.
        downscaling = dict(result)["Downscaling"]
        assert [g for g, _ in downscaling] == [""]
        assert [s.key for s in dict(downscaling)[""]] == ["n_layers"]

    def test_tabs_follow_form_order(self):
        result = grouped_specs(
            ["metadata_reader", "dtype", "scene_index", "y_scale_factor"])
        assert [tab for tab, _ in result] == [
            "Reader", "Conversion", "Downscaling", "Metadata"]

    def test_groups_within_a_tab_follow_form_order(self):
        """Compression precedes Chunking on the Conversion tab."""
        result = dict(grouped_specs(["x_chunk", "compressor"]))["Conversion"]
        assert [g for g, _ in result] == ["Compression", "Chunking"]

    def test_every_spec_declares_a_tab(self):
        for spec in _PARAM_SPECS:
            assert spec.tab, f"{spec.key} has no tab"

    def test_unknown_keys_are_skipped(self):
        assert grouped_specs(["no_such_param"]) == []


class TestColumnHeader:
    """The queue header carries the same hierarchy the dialog nests as boxes."""

    def test_grouped_param_reports_tab_and_group(self):
        assert column_header("x_chunk") == ("Conversion", "Chunking", "Chunk x")

    def test_ungrouped_param_has_empty_group(self):
        tab, group, label = column_header("n_layers")
        assert (tab, group) == ("Downscaling", "")
        assert label == "Resolution layers"

    def test_path_columns_group_under_paths(self):
        """Paths gained a header once they became editable cells."""
        for key in _PATH_COLUMNS:
            tab, group, label = column_header(key)
            assert tab == "Paths"
            assert group == ""
            assert label != key          # a human label, not the raw key

    def test_unknown_key_falls_back_to_its_name(self):
        assert column_header("no_such_param") == ("", "", "no_such_param")

    def test_every_spec_has_a_renderable_header(self):
        for spec in _PARAM_SPECS:
            tab, _, label = column_header(spec.key)
            assert tab and label


# ── no stray columns ──────────────────────────────────────────────────────────

class TestNoStrayColumns:
    """Every column the queue can show must be editable and categorised.

    A column with no ParamSpec renders as a bare snake_case header with no
    tab/group and refuses every edit — the "stray parameter" bug.
    """

    def _full_columns(self, cfg) -> list[str]:
        model = BatchModel(full=True)
        model.set_baseline(deepcopy(cfg))
        model.add(cfg, ["a.tif"], "/out")
        return [c for c in model.columns() if c not in _PATH_COLUMNS]

    def test_every_full_mode_column_is_editable(self):
        for column in self._full_columns(_config()):
            assert uneditable_reason(column) is None, column

    def test_every_full_mode_column_has_a_header(self):
        for column in self._full_columns(_config()):
            tab, _, label = column_header(column)
            assert tab, f"{column} has no tab — renders as a stray column"
            assert label != column, f"{column} has no human label"

    def test_physical_scale_overrides_are_categorised(self):
        """Emitted only when the form enables them, so easy to miss."""
        cfg = _config()
        cfg["metadata"] = {
            "overridePhysicalScale": True,
            "scaleX": "0.5", "unitX": "nanometer",
        }
        columns = self._full_columns(cfg)
        assert "x_scale" in columns and "x_unit" in columns
        for column in columns:
            assert uneditable_reason(column) is None, column

    def test_cluster_keys_are_never_columns(self):
        columns = self._full_columns(_config())
        assert not (_CLUSTER_KEYS & set(columns))

    def test_zarr_format_is_not_a_column(self):
        """Derived from ome_zarr_version; editing it per row is meaningless."""
        assert "zarr_format" not in self._full_columns(_config())

    def test_compressor_params_hidden_from_view_but_kept_in_csv(self, tmp_path):
        model = BatchModel(full=True)
        cfg = _config()
        model.set_baseline(deepcopy(cfg))
        model.add(cfg, ["a.tif"], "/out")
        assert "compressor_params" not in model.columns()
        assert "compressor_params" in model.columns(for_csv=True)
        # And it must survive a real round-trip, or the codec pair desynchronises.
        reloaded = BatchModel.load(model.save(str(tmp_path / "batch.csv")))
        assert "compressor_params" in reloaded.rows[0]


class TestBlockedReporting:
    """Changes a row cannot carry must be reported, never silently dropped."""

    def test_cluster_change_is_reported(self):
        model = BatchModel()
        base = _config()
        model.set_baseline(deepcopy(base))
        model.add(base, ["a.tif"], "/out")

        variant = deepcopy(base)
        variant["cluster"] = {"bfReadConcurrency": 8, "jvmMemory": 8}
        blocked = model.add(variant, ["b.tif"], "/out2")
        assert "bf_read_concurrency" in blocked
        assert "jvm_memory" in blocked

    def test_legitimate_override_is_not_blocked(self):
        model = BatchModel()
        base = _config()
        model.set_baseline(deepcopy(base))
        model.add(base, ["a.tif"], "/out")

        variant = _config(dataType="uint16")
        assert model.add(variant, ["b.tif"], "/out2") == []
        assert model.cell(model.rows[1], "dtype") == "uint16"


# ── dependent parameters ──────────────────────────────────────────────────────

class TestDependentParams:
    """Parameters a parent switch makes inert are declared, not guessed."""

    def test_manual_chunks_depend_on_auto_chunk(self):
        for axis in ("time", "channel", "z", "y", "x"):
            spec = spec_for(f"{axis}_chunk")
            assert spec.depends_on == "auto_chunk"
            assert spec.active_when is False

    def test_target_chunk_depends_on_auto_chunk(self):
        spec = spec_for("target_chunk_mb")
        assert spec.depends_on == "auto_chunk"
        assert spec.active_when is True

    def test_dependencies_point_at_real_specs(self):
        """A typo'd parent would silently never activate the dependant."""
        for spec in _PARAM_SPECS:
            if spec.depends_on:
                assert spec_for(spec.depends_on) is not None, spec.key

    def test_active_when_matches_the_parent_kind(self):
        """A bool parent takes a bool active_when; a choice takes one of its options."""
        for spec in _PARAM_SPECS:
            if not spec.depends_on:
                continue
            parent = spec_for(spec.depends_on)
            if parent.kind == "bool":
                assert isinstance(spec.active_when, bool), spec.key
            elif parent.kind == "choice":
                assert str(spec.active_when) in parent.choices, spec.key
            elif parent.kind == "auto_int":
                # Only the 'auto' state is meaningful as a condition; an exact
                # layer count would be an arbitrary trigger.
                assert spec.active_when == "auto", spec.key
            else:
                raise AssertionError(
                    f"{spec.key} depends on {parent.key} of unsupported "
                    f"kind {parent.kind!r}")

    def test_n_layers_uses_a_sentinel_for_auto(self):
        """'auto' is a value the row states; blank still means "use the config"."""
        assert spec_for("n_layers").kind == "auto_int"

    def test_undeclared_params_have_no_dependency(self):
        assert spec_for("dtype").depends_on == ""


class TestInertCells:
    """Inertness is a per-cell property, shown by greying rather than hiding.

    Hiding the column was wrong: one row switching auto-chunking off makes the
    manual sizes meaningful for that row only, and the column must still appear
    for it while staying visibly inactive everywhere else.
    """

    def _model(self, cfg, full=False):
        model = BatchModel(full=full)
        model.set_baseline(deepcopy(cfg))
        model.add(cfg, ["a.tif", "b.tif", "c.tif"], "/out")
        return model

    def test_manual_chunk_is_inert_while_auto_is_on(self):
        model = self._model(_config(autoChunk=True, chunkZ=96))
        assert model.is_inert(model.rows[0], "z_chunk")

    def test_manual_chunk_is_live_when_auto_is_off(self):
        model = self._model(_config(autoChunk=False, chunkZ=96))
        assert not model.is_inert(model.rows[0], "z_chunk")

    def test_target_chunk_is_the_mirror_case(self):
        model = self._model(_config(autoChunk=False))
        assert model.is_inert(model.rows[0], "target_chunk_mb")

    def test_one_row_off_makes_only_that_row_live(self):
        """The reported bug: per-column hiding cannot express this."""
        model = self._model(_config(autoChunk=True, chunkZ=96))
        model.update_cells([1], "auto_chunk", False)
        model.update_cells([1], "z_chunk", 64)

        assert model.is_inert(model.rows[0], "z_chunk")
        assert not model.is_inert(model.rows[1], "z_chunk")
        assert model.is_inert(model.rows[2], "z_chunk")

    def test_column_stays_visible_for_the_live_row(self):
        model = self._model(_config(autoChunk=True, chunkZ=96))
        model.update_cells([1], "auto_chunk", False)
        model.update_cells([1], "z_chunk", 64)
        assert "z_chunk" in model.columns()
        assert model.cell(model.rows[1], "z_chunk") == 64

    def test_parent_switch_is_reported_for_the_tooltip(self):
        assert BatchModel().parent_switch("z_chunk") == "auto_chunk"
        assert BatchModel().parent_switch("dtype") == ""

    def test_a_parent_is_never_inert(self):
        model = self._model(_config(autoChunk=True))
        assert not model.is_inert(model.rows[0], "auto_chunk")

    def test_inert_values_still_reach_the_csv(self, tmp_path):
        """The CLI applies the same rule, so the value must survive the write."""
        model = self._model(_config(autoChunk=False, chunkZ=96), full=True)
        model.update_cells([0], "auto_chunk", True)
        assert model.is_inert(model.rows[0], "z_chunk")
        assert "z_chunk" in model.columns(for_csv=True)


class TestAutoSentinel:
    """'auto' (compute it) and blank (use the config) are different states."""

    def test_auto_config_renders_as_auto(self):
        cfg = _config()
        cfg["downscaling"] = {"autoDetectLayers": True, "numLayers": 4}
        model = BatchModel(full=True)
        model.set_baseline(deepcopy(cfg))
        model.add(cfg, ["a.tif"], "/out")
        assert model.cell(model.rows[0], "n_layers") == "auto"

    def test_row_can_choose_auto_against_a_manual_config(self, tmp_path):
        """The case a blank cell cannot express."""
        cfg = _config()
        cfg["downscaling"] = {"autoDetectLayers": False, "numLayers": 7}
        model = BatchModel()
        model.set_baseline(deepcopy(cfg))
        model.add(cfg, ["a.tif", "b.tif"], "/out")
        model.update_cells([1], "n_layers", "auto")

        assert model.cell(model.rows[0], "n_layers") is None    # inherit 7
        assert model.cell(model.rows[1], "n_layers") == "auto"

        reloaded = BatchModel.load(model.save(str(tmp_path / "batch.csv")))
        assert reloaded.rows[1]["n_layers"] == "auto"

    def test_cli_normalises_auto_to_none(self):
        from eubi_bridge.ebridge import _normalise_row_overrides
        assert _normalise_row_overrides({"n_layers": "auto"})["n_layers"] is None
        assert _normalise_row_overrides({"n_layers": "3"})["n_layers"] == "3"


class TestShardDependency:
    """Sharding exists only in zarr v3, which OME-Zarr 0.5 selects."""

    def _model(self, version):
        cfg = _config(omeZarrVersion=version)
        model = BatchModel(full=True)
        model.set_baseline(deepcopy(cfg))
        model.add(cfg, ["a.tif"], "/out")
        return model

    @pytest.mark.parametrize("axis", ["time", "channel", "z", "y", "x"])
    def test_shards_inert_under_0_4(self, axis):
        model = self._model("0.4")
        assert model.is_inert(model.rows[0], f"{axis}_shard_coef")

    @pytest.mark.parametrize("axis", ["time", "channel", "z", "y", "x"])
    def test_shards_live_under_0_5(self, axis):
        model = self._model("0.5")
        assert not model.is_inert(model.rows[0], f"{axis}_shard_coef")

    def test_parent_is_reported(self):
        assert BatchModel().parent_switch("z_shard_coef") == "ome_zarr_version"

    def test_non_bool_active_when_is_supported(self):
        """The dependency is on a choice value, not a boolean switch."""
        assert spec_for("z_shard_coef").active_when == "0.5"

    def test_per_row_shard_inertness(self):
        cfg = _config(omeZarrVersion="0.4")
        model = BatchModel()
        model.set_baseline(deepcopy(cfg))
        model.add(cfg, ["a.tif", "b.tif"], "/out")
        model.update_cells([1], "ome_zarr_version", "0.5")
        assert model.is_inert(model.rows[0], "z_shard_coef")
        assert not model.is_inert(model.rows[1], "z_shard_coef")


class TestDownscalingDependencies:
    """Downscaling parameters that a stronger setting makes irrelevant.

    Verified against the code, not assumed: ``update_downscaler`` consults
    ``min_dimension_size`` only when ``n_layers`` is auto, and
    ``keep_existing_resolutions`` routes to ``store_existing_pyramid_async``,
    whose signature has no downscaling parameters at all.
    """

    def _model(self, **downscaling):
        cfg = _config()
        cfg["downscaling"] = downscaling
        model = BatchModel(full=True)
        model.set_baseline(deepcopy(cfg))
        model.add(cfg, ["a.tif"], "/out")
        return model

    def test_min_dimension_live_when_layers_auto(self):
        model = self._model(autoDetectLayers=True)
        assert not model.is_inert(model.rows[0], "min_dimension_size")

    def test_min_dimension_inert_when_layers_explicit(self):
        model = self._model(autoDetectLayers=False, numLayers=7)
        assert model.is_inert(model.rows[0], "min_dimension_size")

    @pytest.mark.parametrize("key", [
        "n_layers", "min_dimension_size", "downscale_method",
        "z_scale_factor", "time_scale_factor"])
    def test_keep_existing_makes_downscaling_inert(self, key):
        model = self._model(keepExistingResolutions=True, autoDetectLayers=True)
        assert model.is_inert(model.rows[0], key)

    @pytest.mark.parametrize("key", [
        "n_layers", "downscale_method", "z_scale_factor"])
    def test_downscaling_live_when_not_keeping(self, key):
        model = self._model(keepExistingResolutions=False, autoDetectLayers=True)
        assert not model.is_inert(model.rows[0], key)

    def test_smart_factors_follow_their_switch(self):
        off = self._model(applySmartDownscaling=False)
        on = self._model(applySmartDownscaling=True)
        assert off.is_inert(off.rows[0], "z_smart_scale_factor")
        assert not on.is_inert(on.rows[0], "z_smart_scale_factor")

    def test_inertness_cascades_through_a_chain(self):
        """keep_existing -> n_layers -> min_dimension_size.

        min_dimension_size's own parent (n_layers) says 'auto', which would
        normally make it live; it must still be inert because n_layers itself is.
        """
        model = self._model(keepExistingResolutions=True, autoDetectLayers=True)
        assert model.is_inert(model.rows[0], "n_layers")
        assert model.is_inert(model.rows[0], "min_dimension_size")

    def test_no_dependency_cycles(self):
        """A cycle would recurse forever; the guard must also not exist by luck."""
        for spec in _PARAM_SPECS:
            seen, key = set(), spec.key
            while key:
                assert key not in seen, f"cycle through {spec.key}"
                seen.add(key)
                parent = spec_for(key)
                key = parent.depends_on if parent else ""


class TestInertReason:
    """The message must name the parent's *disabling* value, not just the parent.

    Naming the parent alone reads backwards for a dependant that is active when
    its parent is True: "apply_smart_downscaling makes X inactive" suggests
    enabling it is what disables X, when the opposite is true.
    """

    def _model(self, conversion=None, downscaling=None):
        cfg = _config(**(conversion or {}))
        cfg["downscaling"] = downscaling or {}
        model = BatchModel(full=True)
        model.set_baseline(deepcopy(cfg))
        model.add(cfg, ["a.tif"], "/out")
        return model

    def test_states_the_disabling_value_for_an_active_when_true_param(self):
        model = self._model(downscaling={"applySmartDownscaling": False})
        assert model.inert_reason(
            model.rows[0], "z_smart_scale_factor") == "apply_smart_downscaling=False"

    def test_states_the_disabling_value_for_an_active_when_false_param(self):
        model = self._model(conversion={"autoChunk": True})
        assert model.inert_reason(model.rows[0], "z_chunk") == "auto_chunk=True"

    def test_mirror_case_reports_the_opposite_value(self):
        model = self._model(conversion={"autoChunk": False})
        assert model.inert_reason(
            model.rows[0], "target_chunk_mb") == "auto_chunk=False"

    def test_choice_parent_uses_inequality(self):
        model = self._model(conversion={"omeZarrVersion": "0.4"})
        assert model.inert_reason(
            model.rows[0], "z_shard_coef") == "ome_zarr_version!=0.5"

    def test_chain_reports_the_root_cause(self):
        """min_dimension_size's own parent says 'auto'; the real cause is upstream."""
        model = self._model(
            downscaling={"keepExistingResolutions": True, "autoDetectLayers": True})
        assert model.inert_reason(
            model.rows[0],
            "min_dimension_size") == "keep_existing_resolutions=True"

    def test_live_parameter_has_no_reason(self):
        model = self._model(conversion={"autoChunk": False})
        assert model.inert_reason(model.rows[0], "z_chunk") == ""

    def test_no_em_dashes_in_user_facing_reasons(self):
        """Project style for these messages: use a colon, not a dash."""
        for spec in _PARAM_SPECS:
            if not spec.depends_on:
                continue
            cfg = _config()
            model = BatchModel(full=True)
            model.set_baseline(deepcopy(cfg))
            model.add(cfg, ["a.tif"], "/out")
            reason = model.inert_reason(model.rows[0], spec.key)
            assert "—" not in reason and "–" not in reason


class TestPathEditing:
    """Input and output paths are editable cells, not fixed row identity.

    They are among the most frequently corrected fields, so they must be
    reachable through Edit Cells like any other parameter.
    """

    @pytest.mark.parametrize("key,kind", [
        ("input_path", "file"), ("output_path", "directory")])
    def test_paths_have_a_browsable_spec(self, key, kind):
        spec = spec_for(key)
        assert spec is not None, f"{key} has no spec, so no editor is built"
        assert spec.kind == kind

    def test_paths_lead_the_column_order(self):
        assert sort_keys(["dtype", "output_path", "input_path"])[:2] ==             list(_PATH_COLUMNS)

    def test_paths_group_under_their_own_tab(self):
        tabs = [tab for tab, _ in grouped_specs(["input_path", "dtype"])]
        assert tabs[0] == "Paths"

    def test_output_can_be_retargeted_in_bulk(self, batch):
        batch.update_cells([0, 2], "output_path", "/mnt/new")
        assert [r["output_path"] for r in batch.rows] == [
            "/mnt/new", "/out", "/mnt/new"]

    def test_input_can_be_corrected(self, batch):
        batch.update_cells([1], "input_path", "b_fixed.tif")
        assert batch.rows[1]["input_path"] == "b_fixed.tif"

    def test_paths_are_never_inert(self, batch):
        for key in _PATH_COLUMNS:
            assert not batch.is_inert(batch.rows[0], key)

    def test_blank_path_is_rejected(self, batch):
        """A blank cell means "inherit" elsewhere; for a path it is unrunnable."""
        for blank in ("", "   "):
            with pytest.raises(ValueError):
                batch.update_cells([0], "output_path", blank)

    def test_paths_cannot_be_reset_to_a_baseline(self, batch):
        """There is no baseline path to fall back to."""
        for key in _PATH_COLUMNS:
            with pytest.raises(KeyError):
                batch.reset_cells([0], key)

    def test_editing_preserves_validation(self, batch):
        """Retargeting two rows onto one output must still be caught."""
        batch.update_cells([1], "input_path", "a.tif")
        problems = batch.validate()
        assert any("overwrite" in p for p in problems)

    def test_edited_paths_survive_a_round_trip(self, batch, tmp_path):
        batch.update_cells([0], "output_path", str(tmp_path / "elsewhere"))
        reloaded = BatchModel.load(batch.save(str(tmp_path / "batch.csv")))
        assert reloaded.rows[0]["output_path"] == batch.rows[0]["output_path"]


class TestCategorySeparators:
    """A thin blank gutter marks each category boundary."""

    def test_separator_between_categories(self):
        out = with_separators(["scene_index", "dtype"])
        assert out == ["scene_index", SEPARATOR, "dtype"]

    def test_no_separator_within_a_category(self):
        out = with_separators(["z_chunk", "dtype"])
        assert SEPARATOR not in out

    def test_never_leading_or_trailing(self):
        out = with_separators(["input_path", "dtype"])
        assert out[0] != SEPARATOR and out[-1] != SEPARATOR

    def test_never_doubled(self):
        out = with_separators(
            ["input_path", "scene_index", "dtype", "n_layers"])
        assert not any(a == SEPARATOR and b == SEPARATOR
                       for a, b in zip(out, out[1:]))

    def test_empty_input_is_unchanged(self):
        assert with_separators([]) == []

    def test_separator_is_not_a_real_parameter(self):
        """It must never collide with a column name."""
        assert spec_for(SEPARATOR) is None
        assert SEPARATOR not in {s.key for s in _PARAM_SPECS}

    def test_real_columns_are_preserved_in_order(self):
        columns = ["input_path", "scene_index", "dtype", "n_layers"]
        out = [c for c in with_separators(columns) if c != SEPARATOR]
        assert out == columns


class TestCategoryToggles:
    """Whole categories can be shown, without hiding any real deviation."""

    def _model(self):
        cfg = _config(autoChunk=False, chunkZ=96)
        model = BatchModel()
        model.set_baseline(deepcopy(cfg))
        model.add(cfg, ["a.tif", "b.tif"], "/out")
        return model

    def test_tabs_exclude_paths(self):
        """Paths are always shown, so they are not a toggleable category."""
        assert "Paths" not in parameter_tabs()

    def test_tabs_follow_form_order(self):
        assert parameter_tabs() == [
            "Reader", "Conversion", "Downscaling", "Metadata"]

    def test_showing_a_tab_adds_its_columns(self):
        model = self._model()
        assert "n_layers" not in model.columns()
        model.shown_tabs = {"Downscaling"}
        assert "n_layers" in model.columns()

    def test_showing_a_tab_adds_nothing_else(self):
        model = self._model()
        model.shown_tabs = {"Downscaling"}
        for key in model.columns():
            if key in _PATH_COLUMNS:
                continue
            assert column_header(key)[0] == "Downscaling"

    def test_deviating_columns_survive_a_tab_being_off(self):
        """The pre-existing behaviour: a deviation is always visible."""
        model = self._model()
        model.update_cells([1], "dtype", "uint16")
        assert "dtype" in model.columns()
        model.shown_tabs = {"Downscaling"}
        assert "dtype" in model.columns()
        model.shown_tabs = set()
        assert "dtype" in model.columns()

    def test_toggling_off_restores_the_narrow_view(self):
        model = self._model()
        narrow = model.columns()
        model.shown_tabs = {"Reader", "Metadata"}
        assert len(model.columns()) > len(narrow)
        model.shown_tabs = set()
        assert model.columns() == narrow

    def test_full_table_still_wins(self):
        model = self._model()
        model.full = True
        with_tab = set(model.columns())
        model.shown_tabs = {"Reader"}
        assert set(model.columns()) == with_tab
