"""A queued batch can run without first being written to a CSV.

Saving a batch is a deliberate act ("keep this for later"), not a step every run
has to perform.  ``BatchModel.to_table()`` produces the same rows ``save()``
writes, and ``take_filepaths`` accepts that table directly, so the queue reaches
``to_zarr`` without touching the filesystem.
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eubi_bridge.qt_gui.core.batch import BatchModel
from eubi_bridge.utils.path_utils import take_filepaths


def _config(**downscaling):
    cfg = {
        "cluster": {}, "reader": {}, "metadata": {}, "concatenation": {},
        "downscaling": downscaling,
        "conversion": {"dataType": "auto"},
    }
    return cfg


def _model(paths=("/d/a.tif", "/d/b.tif"), **downscaling):
    cfg = _config(**downscaling)
    model = BatchModel()
    model.set_baseline(deepcopy(cfg))
    model.add(cfg, list(paths), "/out")
    return model


class TestToTable:
    def test_one_row_per_queued_conversion(self):
        table = _model().to_table()
        assert len(table) == 2
        assert list(table["input_path"]) == ["/d/a.tif", "/d/b.tif"]

    def test_per_row_overrides_are_preserved(self):
        """A path list cannot express this; only a table can."""
        model = _model()
        model.update_cells([0], "dtype", "uint8")
        table = model.to_table()
        assert table.loc[0, "dtype"] == "uint8"
        # An untouched cell is empty; pandas may store that as None or NaN, and
        # to_zarr discards both alike ("v is not None and v == v").
        assert pd.isna(table.loc[1, "dtype"])

    def test_columns_match_the_csv(self, tmp_path):
        model = _model()
        model.update_cells([0], "dtype", "uint8")
        written = pd.read_csv(model.save(str(tmp_path / "b.csv")))
        assert list(model.to_table().columns) == list(written.columns)

    def test_auto_sentinel_is_not_collapsed(self):
        """'auto' (compute it) must stay distinct from blank (use the config).

        Resolving the sentinel here would make the two indistinguishable, since
        both would become None.
        """
        model = _model(autoDetectLayers=False, numLayers=5)
        model.update_cells([1], "n_layers", "auto")
        table = model.to_table()
        assert pd.isna(table.loc[0, "n_layers"])     # inherit the config's 5
        assert table.loc[1, "n_layers"] == "auto"    # compute it

    def test_paths_stay_absolute(self):
        """Unlike the CSV, an in-memory table has no directory to relativise to."""
        table = _model(paths=["/data/deep/a.tif"]).to_table()
        assert table.loc[0, "input_path"] == "/data/deep/a.tif"

    def test_empty_queue_gives_an_empty_table(self):
        model = BatchModel()
        model.set_baseline(_config())
        assert len(model.to_table()) == 0


class TestTakeFilepathsAcceptsTable:
    def _table(self):
        return pd.DataFrame([
            {"input_path": "/d/a.tif", "output_path": "/out", "dtype": "uint8"},
            {"input_path": "/d/b.tif", "output_path": "/out", "dtype": None},
        ])

    def test_table_is_used_as_is(self):
        out = take_filepaths(self._table())
        assert list(out["input_path"]) == ["/d/a.tif", "/d/b.tif"]

    def test_per_row_values_win_over_globals(self):
        """The global value fills only the rows that state nothing."""
        out = take_filepaths(self._table(), n_layers=3)
        assert list(out["dtype"]) == ["uint8", None]
        assert list(out["n_layers"]) == [3, 3]

    def test_filters_still_apply(self):
        out = take_filepaths(self._table(), includes="*a.tif")
        assert list(out["input_path"]) == ["/d/a.tif"]

    def test_filepath_column_is_accepted(self):
        table = pd.DataFrame([{"filepath": "/d/a.tif", "output_path": "/out"}])
        assert "input_path" in take_filepaths(table).columns

    def test_missing_path_column_is_rejected(self):
        with pytest.raises(ValueError):
            take_filepaths(pd.DataFrame([{"nope": 1}]))

    def test_empty_table_is_rejected(self):
        with pytest.raises(ValueError):
            take_filepaths(pd.DataFrame(columns=["input_path"]))

    def test_the_caller_s_table_is_not_mutated(self):
        table = self._table()
        take_filepaths(table, n_layers=3)
        assert "n_layers" not in table.columns

    def test_table_survives_pickling(self):
        """The GUI hands the table to a worker process."""
        import pickle
        table = self._table()
        assert pickle.loads(pickle.dumps(table)).equals(table)


class TestToZarrAcceptsTable:
    def test_output_path_may_be_omitted_for_a_table(self):
        """The rows carry their own output_path, as with a CSV."""
        import inspect
        from eubi_bridge import ebridge
        source = inspect.getsource(ebridge.ConversionManager.to_zarr)
        assert "isinstance(input_path, pd.DataFrame)" in source

    def test_a_bare_path_still_requires_output_path(self):
        from eubi_bridge.ebridge import EuBIBridge
        with pytest.raises(ValueError, match="output_path is required"):
            EuBIBridge().to_zarr("/some/folder", verbose=False)
