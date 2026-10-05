"""The region budget (``conversion_worker._region_budget_mb``) shares the RAM
among the conversions that actually run at once.

It used to multiply by ``max_workers`` whatever the run: one concatenated
output with ``max_workers=4`` had its region size cut four times further
than its single writer needed (1024 MB -> 30 MB on 3.4 GB free).  The
dispatcher now tells the workers how many conversions run at once
(``ClusterConfig.concurrent_jobs``).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from eubi_bridge.conversion import conversion_worker, dispatcher
from eubi_bridge.core.config_models import ClusterConfig, ConversionJob

AVAILABLE = 3.4 * 1024 ** 3


@pytest.fixture
def three_gb_free(monkeypatch):
    monkeypatch.setattr(conversion_worker.psutil, "virtual_memory",
                        lambda: SimpleNamespace(available=AVAILABLE))


def _cluster(**kw):
    return ClusterConfig(max_workers=4, max_concurrency=4, queue_size=2,
                         region_size_mb=1024, **kw)


def test_the_budget_counts_the_conversions_running(three_gb_free):
    per_writer = 3 * 4 + 2                       # 8 reading, 2 queued, 4 writing
    free_mb = AVAILABLE / 1024 ** 2
    assert conversion_worker._region_budget_mb(_cluster()) == pytest.approx(
        0.5 * free_mb / (4 * per_writer))        # unknown: max_workers at once
    assert conversion_worker._region_budget_mb(_cluster(concurrent_jobs=1)) == pytest.approx(
        0.5 * free_mb / per_writer)              # one conversion: 4x the budget


def test_unary_jobs_are_told_how_many_run_at_once(monkeypatch):
    seen = []

    async def record(jobs, *args):
        seen.extend(jobs)
        return []

    monkeypatch.setattr(dispatcher, "_dispatch", record)
    one = ConversionJob.from_kwargs("/in.tif", "/out", {"max_workers": 4, "use_threading": True})
    dispatcher.dispatch_unary_jobs([one])
    assert [j.cluster.concurrent_jobs for j in seen] == [1]
    seen.clear()
    dispatcher.dispatch_unary_jobs([one] * 6)
    assert {j.cluster.concurrent_jobs for j in seen} == {4}


@pytest.mark.parametrize("timepoints", [1, 2])
def test_aggregative_groups_are_told_how_many_run_at_once(tmp_path, monkeypatch, timepoints):
    """z planes concatenated per time point: one output group per time point,
    each told how many groups are written at once."""
    tifffile = pytest.importorskip("tifffile")
    for t in range(timepoints):
        for z in range(2):
            tifffile.imwrite(tmp_path / f"img_t{t}_z{z:03d}.tif", np.zeros((4, 4), "u2"))
    seen = []
    monkeypatch.setattr(dispatcher, "aggregative_worker_from_paths",
                        lambda paths, out, kwargs: seen.append(kwargs.get("concurrent_jobs")))
    asyncio.run(dispatcher.run_conversions_with_concatenation(
        str(tmp_path), str(tmp_path / "out"), z_tag="_z", time_tag="_t",
        concatenation_axes="z", max_workers=4, use_threading=True))
    assert seen == [timepoints] * timepoints
