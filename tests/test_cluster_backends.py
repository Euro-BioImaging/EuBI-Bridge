"""Where conversions run (Bugra, 2026-10-05).

- The local dask cluster (``on_local_cluster``) is retired: accepted, warned
  about once, ignored -- the process pool does the same job.  Old configs,
  saved batches and scripts keep working.
- SLURM (``on_slurm``, ``slurm_*``) stays in the full eubi-bridge: "Some
  users depend on it."  eubi-bridge-lite has no dask-jobqueue: it refuses
  SLURM clearly (and its GUI hides it, tests/test_lite_gui.py) rather than
  running a conversion meant for a cluster on this machine.
"""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
import tifffile

SLURM = ("on_slurm", "slurm_time", "slurm_account", "slurm_partition",
         "slurm_worker_timeout", "slurm_sif_path")


def _tiff(path, value=0):
    tifffile.imwrite(path, np.full((3, 16, 16), value, "u2"), metadata={"axes": "ZYX"})
    return path


# -- the local dask cluster: retired ----------------------------------------------

def test_on_local_cluster_is_accepted_and_ignored():
    from eubi_bridge.core.config_models import ClusterConfig
    assert ClusterConfig(on_local_cluster=True).on_local_cluster is None


def test_on_local_cluster_is_dropped_from_an_old_config_slurm_is_kept(tmp_path, monkeypatch):
    from eubi_bridge.ebridge import ConfigManager, EuBIBridge
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    path = config_dir / ".eubi_config.json"
    monkeypatch.setattr(ConfigManager, "_get_config_dir", lambda self: config_dir)
    monkeypatch.setattr(ConfigManager, "_get_json_path", lambda self: path)
    EuBIBridge().config                                  # writes defaults
    old = json.loads(path.read_text())
    old["cluster"].update(on_local_cluster=True, on_slurm=True, slurm_partition="gpu")
    path.write_text(json.dumps(old))

    loaded = EuBIBridge().config["cluster"]
    assert "on_local_cluster" not in loaded
    assert "on_local_cluster" not in json.loads(path.read_text())["cluster"]
    assert loaded["on_slurm"] is True and loaded["slurm_partition"] == "gpu"


def test_configure_saves_slurm_not_the_local_cluster(tmp_path):
    from eubi_bridge.ebridge import EuBIBridge
    config = str(tmp_path / "cfg")
    EuBIBridge(configpath=config).configure.cluster(on_local_cluster=True, on_slurm=True,
                                                    slurm_partition="gpu", max_processes=3)
    saved = EuBIBridge(configpath=config).config["cluster"]
    assert "on_local_cluster" not in saved
    assert saved["on_slurm"] is True and saved["slurm_partition"] == "gpu"
    assert saved["max_processes"] == 3


def test_the_cli_with_on_local_cluster_converts_and_warns_once(tmp_path):
    """What a user's old script does: it still converts, and says so once
    -- not once per worker process."""
    source = _tiff(tmp_path / "img.tif")
    out = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, "-c", "import sys; from eubi_bridge.cli import main; "
                               "sys.argv = ['eubi'] + sys.argv[1:]; main()",
         "to_zarr", str(source), str(out), "--on_local_cluster", "True", "--max_processes", "2"],
        capture_output=True, text=True, timeout=600)
    log = result.stdout + result.stderr
    assert result.returncode == 0, log[-3000:]
    assert len(list(out.glob("*.zarr"))) == 1
    assert log.count("'on_local_cluster' is no longer used") == 1, log[-3000:]


def test_an_aggregative_conversion_with_on_local_cluster_runs(tmp_path):
    """on_local_cluster used to refuse concatenation; ignored, nothing refuses."""
    from eubi_bridge.ebridge import EuBIBridge
    folder = tmp_path / "in"
    folder.mkdir()
    _tiff(folder / "img_z0.tif", 1)
    _tiff(folder / "img_z1.tif", 2)
    out = tmp_path / "out"
    EuBIBridge(configpath=str(tmp_path / "cfg")).to_zarr(
        str(folder), str(out), concatenation_axes="z", z_tag="_z",
        on_local_cluster=True, verbose=False)
    assert len(list(out.glob("*.zarr"))) == 1


# -- SLURM: the full eubi-bridge keeps it, lite refuses it ------------------------

def test_slurm_settings_are_kept():
    from eubi_bridge.core.config_models import ClusterConfig
    config = ClusterConfig(on_slurm=True, slurm_time="48:00:00", slurm_account="lab",
                           slurm_partition="gpu", slurm_worker_timeout=600,
                           slurm_sif_path="/apps/eubi.sif")
    assert (config.on_slurm, config.slurm_partition, config.slurm_worker_timeout) == \
        (True, "gpu", 600)


def test_slurm_jobs_go_to_the_slurm_backend(monkeypatch):
    """The dispatch decision (a SLURM cluster cannot run here)."""
    from eubi_bridge.conversion import dispatcher
    from eubi_bridge.core.config_models import ClusterConfig
    sent = []

    async def fake_slurm(jobs):
        sent.extend(jobs)
        return ["ok"] * len(jobs)

    monkeypatch.setattr(dispatcher, "_dispatch_unary_with_slurm", fake_slurm)

    class Job:
        def __init__(self):
            self.cluster = ClusterConfig(on_slurm=True)

        def model_copy(self, update):
            copy = Job()
            copy.cluster = update["cluster"]
            return copy

    assert dispatcher.dispatch_unary_jobs([Job(), Job()]) == ["ok", "ok"]
    assert len(sent) == 2 and all(job.cluster.on_slurm for job in sent)


def test_lite_refuses_slurm_clearly(monkeypatch):
    from eubi_bridge.utils import capabilities
    monkeypatch.setattr(capabilities, "has_slurm", lambda: False)
    with pytest.raises(capabilities.NeedsFullEubiBridge) as info:
        capabilities.check_run(on_slurm=True)
    assert "on_slurm=True" in str(info.value)
    assert "pip uninstall eubi-bridge-lite, then pip install eubi-bridge" in str(info.value)
    capabilities.check_run(on_slurm=False)               # nothing else to refuse


# -- the GUI --------------------------------------------------------------------

def test_the_gui_has_slurm_but_no_local_cluster():
    from tests.conftest import qt_available
    if not qt_available():
        pytest.skip("Qt is not available")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])           # noqa: F841
    from eubi_bridge.qt_gui.pages.convert_page import ConvertPage, _cluster_summary
    from eubi_bridge.utils.capabilities import has_slurm
    page = ConvertPage()
    try:
        assert not hasattr(page, "_use_local_dask")
        assert "useLocalDask" not in page._ui_cluster_config()
        # the memory budget is the run's: always shown, not only with SLURM
        assert not page._memory_budget.isHidden()
        assert not page._memory_budget.parentWidget().isHidden()
        if not has_slurm():
            pytest.skip("lite: SLURM is hidden (tests/test_lite_gui.py)")
        page._use_slurm.setChecked(True)
        page._slurm_partition.setText("gpu")
        assert all(not row.isHidden() for row in page._slurm_rows)
        cluster = page._ui_cluster_config()
        assert cluster["useSlurm"] is True and cluster["slurmPartition"] == "gpu"
        assert "SLURM (partition gpu" in _cluster_summary(cluster)
    finally:
        page.deleteLater()
