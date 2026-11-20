"""eubi's write budget is its own share of the run's memory, so it writes inside dyna's
soft_memory_budget(): a budget too small for dyna's smallest write warns and writes with
it, as scene_mb does below MIN_SCENE_MB, instead of raising MemoryBudgetError
(dyna-zarr 0.2.0+)."""
import importlib
from types import SimpleNamespace

import pytest

from eubi_bridge.conversion import conversion_worker, pyramid_path

dyna_io = importlib.import_module("dyna_zarr.io")

pytestmark = pytest.mark.skipif(not hasattr(dyna_io, "soft_memory_budget"),
                                reason="dyna-zarr before 0.2.0 only warns")


def _write(monkeypatch, budget):
    seen = {}

    def write_pyramid(self, pyr, target, **kwargs):
        seen["soft"] = dyna_io._SOFT_BUDGET.get()
        seen["budget"] = kwargs.get("memory_budget_mb")

    monkeypatch.setattr("ome_zarr_pyramid.IO.write_pyramid", write_pyramid)
    monkeypatch.setattr(pyramid_path, "_system_for_output", lambda pyr: pyr)
    monkeypatch.setattr(conversion_worker, "_threads_per_scene", lambda clus: 2)
    monkeypatch.setattr(conversion_worker, "_write_share", lambda *a, **k: None)
    pyr = SimpleNamespace(meta=SimpleNamespace(zarr_format=2))
    conv = SimpleNamespace(ome_zarr_version="0.5", io_backend="tensorstore",
                           zip_output="off", overwrite=True, compressor="blosc",
                           compressor_params={})
    job = SimpleNamespace(metadata=SimpleNamespace(channel_intensity_limits="from_dtype"))
    pyramid_path._write_pyramid(pyr, "out.zarr", None, conv, SimpleNamespace(), (1, 1),
                                None, "uint16", SimpleNamespace(dtype="uint16"), 8.0,
                                budget, job, None)
    return seen


def test_a_conversion_writes_in_a_soft_budget(monkeypatch):
    seen = _write(monkeypatch, 64.0)
    assert seen == {"soft": True, "budget": 64.0}
    assert dyna_io._SOFT_BUDGET.get() is False          # only for the write


def test_without_a_budget_nothing_changes(monkeypatch):
    """A dask-read source is written without a budget (output_store._memory_budget)."""
    assert _write(monkeypatch, None) == {"soft": False, "budget": None}
