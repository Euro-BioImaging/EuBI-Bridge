"""Each saved batch keeps its own settings file (eubi-bridge 0.3.3, Workplan E20).

Until 0.3.2 Save Batch wrote one batch_config.json per folder, so the last batch saved
in a folder gave its settings to every batch there: a batch saved on 2026-10-20 loaded
another batch's zarrista / zarr v2 settings of 2026-10-23, and every row failed. Now
batch.csv keeps batch.config.json. A batch saved before has only the shared file: it is
applied when it was written with the batch's CSV, else not, with a note (decision E-4 b).
"""
from __future__ import annotations

import json
import os
import time

from eubi_bridge.qt_gui.core.batch import (CONFIG_SNAPSHOT_NAME, BatchModel,
                                           config_snapshot_path)


def _batch(folder, name, backend):
    model = BatchModel()
    model._rows.append({"input_path": str(folder / "a.tif")})
    model._base_config = {"conversion": {"ioBackend": backend}}
    return model.save(str(folder / name))


def test_two_batches_of_one_folder_keep_their_own_settings(tmp_path):
    first = _batch(tmp_path, "first.csv", "tensorstore")
    second = _batch(tmp_path, "second.csv", "zarrista")
    assert config_snapshot_path(first).endswith("first.config.json")
    assert not (tmp_path / CONFIG_SNAPSHOT_NAME).exists()
    for csv_path, backend in ((first, "tensorstore"), (second, "zarrista")):
        model = BatchModel.load(csv_path)
        assert model._base_config["conversion"]["ioBackend"] == backend
        assert model.snapshot_path == config_snapshot_path(csv_path)
        assert model.snapshot_note is None


def _old_batch(folder, snapshot_written_later_s):
    """A batch as 0.3.2 saved it: the CSV, the folder's shared batch_config.json."""
    csv_path = folder / "old.csv"
    csv_path.write_text("input_path\na.tif\n")
    shared = folder / CONFIG_SNAPSHOT_NAME
    shared.write_text(json.dumps({"conversion": {"ioBackend": "zarrista"}}))
    now = time.time()
    os.utime(csv_path, (now, now))
    os.utime(shared, (now + snapshot_written_later_s,) * 2)
    return str(csv_path)


def test_an_old_batch_takes_the_shared_file_written_with_it(tmp_path):
    model = BatchModel.load(_old_batch(tmp_path, 1))
    assert model._base_config["conversion"]["ioBackend"] == "zarrista"
    assert model.snapshot_path == str(tmp_path / CONFIG_SNAPSHOT_NAME)


def test_an_old_batch_leaves_another_batchs_shared_file(tmp_path):
    """The shared file written days after the CSV: another batch saved it."""
    model = BatchModel.load(_old_batch(tmp_path, 3 * 24 * 3600))
    assert model._base_config is None and model.snapshot_path is None
    assert CONFIG_SNAPSHOT_NAME in model.snapshot_note
    assert "another batch" in model.snapshot_note and "old.config.json" in model.snapshot_note


def test_an_old_batch_saved_again_gets_its_own_file(tmp_path):
    csv_path = _old_batch(tmp_path, 1)
    model = BatchModel.load(csv_path)
    model.save(csv_path)
    assert os.path.exists(config_snapshot_path(csv_path))
    assert BatchModel.load(csv_path).snapshot_path == config_snapshot_path(csv_path)
