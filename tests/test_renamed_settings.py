"""Renamed cluster settings (2026-10-08, the names the whole ecosystem uses):

    max_workers -> max_processes          max_concurrency -> max_threads
    micro_read_concurrency -> decode_threads
    bf_read_concurrency -> bioformats_threads

The old names keep working everywhere a user can give them (a saved config, the
CLI, a conversion table, a saved GUI batch, a script), with the value kept.
Scenes converted at once share a conversion's max_threads.
"""
from __future__ import annotations

import json
import logging

import pytest

from eubi_bridge.conversion.conversion_worker import _threads_per_scene
from eubi_bridge.core import config_models
from eubi_bridge.core.config_models import ClusterConfig, rename_settings
from eubi_bridge.ebridge import EuBIBridge, _normalise_row_overrides
from eubi_bridge.qt_gui.core.batch import _renamed_snapshot

OLD = {"max_workers": 7, "max_concurrency": 5, "micro_read_concurrency": 3,
       "bf_read_concurrency": 2}
NEW = {"max_processes": 7, "max_threads": 5, "decode_threads": 3, "bioformats_threads": 2}


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch):
    monkeypatch.setattr(config_models, "_renamed_warned", set())


def test_old_names_map_to_the_new_with_one_warning_each(caplog):
    with caplog.at_level(logging.WARNING):
        assert rename_settings(OLD) == NEW
        rename_settings(OLD)
    warned = [r.getMessage() for r in caplog.records if "DEPRECATION" in r.getMessage()]
    assert len(warned) == 4 and any("'max_workers' is now called 'max_processes'" in m
                                     for m in warned)


def test_given_both_the_new_name_wins():
    assert rename_settings({"max_workers": 2, "max_processes": 9}) == {"max_processes": 9}


def test_the_cluster_model_takes_old_names():
    c = ClusterConfig(**OLD)
    assert (c.max_processes, c.max_threads, c.decode_threads, c.bioformats_threads) == (7, 5, 3, 2)


def test_a_saved_config_keeps_its_values_under_the_new_names(tmp_path):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    EuBIBridge(configpath=str(cfg_dir)).config          # writes the defaults
    path = cfg_dir / ".eubi_config.json"
    saved = json.loads(path.read_text())
    for new in NEW:                                     # as an older version wrote it
        saved["cluster"].pop(new)
    saved["cluster"].update(OLD)
    path.write_text(json.dumps(saved))

    cluster = EuBIBridge(configpath=str(cfg_dir)).config["cluster"]
    assert {k: cluster[k] for k in NEW} == NEW          # not the backfilled defaults
    assert not OLD.keys() & cluster.keys()
    on_disk = json.loads(path.read_text())["cluster"]
    assert {k: on_disk[k] for k in NEW} == NEW


def test_configure_cluster_takes_old_names(tmp_path):
    bridge = EuBIBridge(configpath=str(tmp_path))
    bridge.configure.cluster(max_workers=6, micro_read_concurrency=2)
    assert bridge.config["cluster"]["max_processes"] == 6
    assert bridge.config["cluster"]["decode_threads"] == 2
    with pytest.raises(TypeError, match="unknown setting"):
        bridge.configure.cluster(max_wrokers=6)


def test_a_conversion_table_row_with_old_names():
    assert _normalise_row_overrides({"max_concurrency": 2}) == {"max_threads": 2}


def test_a_saved_gui_batch_keeps_its_values():
    snap = {"cluster": {"maxWorkers": 3, "maxConcurrency": 6, "microReadConcurrency": 2,
                        "bfReadConcurrency": 5, "queueSize": 4}}
    assert _renamed_snapshot(snap)["cluster"] == {
        "maxProcesses": 3, "maxThreads": 6, "decodeThreads": 2, "bioformatsThreads": 5,
        "queueSize": 4}


@pytest.mark.parametrize("threads, scenes, each", [(8, 1, 8), (8, 2, 4), (3, 4, 1), (5, 2, 2)])
def test_scenes_share_the_threads(threads, scenes, each):
    assert _threads_per_scene(ClusterConfig(max_threads=threads,
                                            max_concurrent_scenes=scenes)) == each
