"""One memory budget for the whole run (``memory_budget``, 2026-10-08), shared out by
``core/memory.py``:

- per process: the budget over the conversions running at once, resolved ONCE by the
  dispatcher (``process_memory_mb``), or ``slurm_worker_memory`` on SLURM;
- in each process: what it already holds, the reader's share (``decode_memory``) and
  the rest over the scenes converted at once (each scene's write budget);
- ``region_size_mb='auto'`` from that.

Replaces the old region budget, which divided half the free RAM by a model of
eubi-bridge's own writer and ignored ``memory_per_worker``.
"""
from __future__ import annotations

import asyncio
import logging

import numpy as np
import pytest

from eubi_bridge.conversion import dispatcher
from eubi_bridge.core import memory
from eubi_bridge.core.config_models import ClusterConfig, ConversionJob


@pytest.fixture
def free_16gb(monkeypatch):
    monkeypatch.setattr(memory, "available_mb", lambda: 16384.0)


# --- sizes ------------------------------------------------------------------------------

@pytest.mark.parametrize("value, mb", [(512, 512), ("512MB", 512), ("4GB", 4096),
                                       ("4 GiB", 4096), ("1.5g", 1536), ("2048 kb", 2)])
def test_sizes(value, mb):
    assert memory.parse_mb(value) == pytest.approx(mb)


def test_the_config_takes_sizes_and_auto():
    c = ClusterConfig(memory_budget="16 GB", decode_memory="AUTO", region_size_mb="128")
    assert (c.memory_budget, c.decode_memory, c.region_size_mb) == ("16 GB", "auto", 128.0)
    with pytest.raises(ValueError):
        ClusterConfig(memory_budget="lots")
    with pytest.raises(ValueError):
        ClusterConfig(region_size_mb=0)


# --- per process ------------------------------------------------------------------------

def test_auto_is_half_the_free_ram_over_the_conversions_at_once(free_16gb):
    c = ClusterConfig(max_processes=4)
    assert memory.process_mb(c, 4) == pytest.approx(8192 / 4)
    assert memory.process_mb(c, 1) == pytest.approx(8192)          # one job: all of it
    assert memory.process_mb(ClusterConfig(memory_budget="6GB"), 3) == pytest.approx(2048)


def test_slurm_uses_the_worker_memory():
    c = ClusterConfig(on_slurm=True, slurm_worker_memory="8GB", memory_budget="1GB")
    assert memory.process_mb(c, 1) == pytest.approx(8192)


def test_unary_jobs_get_their_share_resolved_once(free_16gb, monkeypatch):
    seen = []

    async def record(jobs, *args):
        seen.extend(jobs)
        return []

    monkeypatch.setattr(dispatcher, "_dispatch", record)
    one = ConversionJob.from_kwargs("/in.tif", "/out", {"max_processes": 4,
                                                        "use_threading": True})
    dispatcher.dispatch_unary_jobs([one])
    assert [(j.cluster.concurrent_jobs, j.cluster.process_memory_mb) for j in seen] == [
        (1, pytest.approx(8192))]
    seen.clear()
    dispatcher.dispatch_unary_jobs([one] * 6)
    assert {(j.cluster.concurrent_jobs, j.cluster.process_memory_mb) for j in seen} == {
        (4, 2048.0)}


@pytest.mark.parametrize("timepoints", [1, 2])
def test_aggregative_groups_get_their_share(tmp_path, monkeypatch, free_16gb, timepoints):
    tifffile = pytest.importorskip("tifffile")
    for t in range(timepoints):
        for z in range(2):
            tifffile.imwrite(tmp_path / f"img_t{t}_z{z:03d}.tif", np.zeros((4, 4), "u2"))
    seen = []
    monkeypatch.setattr(dispatcher, "aggregative_worker_from_paths",
                        lambda paths, out, kw: seen.append((kw.get("concurrent_jobs"),
                                                            kw.get("process_memory_mb"))))
    asyncio.run(dispatcher.run_conversions_with_concatenation(
        str(tmp_path), str(tmp_path / "out"), z_tag="_z", time_tag="_t",
        concatenation_axes="z", max_processes=4, use_threading=True))
    assert seen == [(timepoints, pytest.approx(8192 / timepoints))] * timepoints


# --- the reader's share ---------------------------------------------------------------------

def test_auto_reader_share_follows_the_decode_unit():
    # small tiles or planes: micro-reader's 64 MB floor + the 64 MB shared cache
    assert memory.reader_share(4096, "auto", 4, 1.0) == (64, 64)
    assert memory.reader_share(4096, "auto", 4, 9.0) == (64, 64)
    # 12 threads x 9 MB planes decoded whole
    assert memory.reader_share(4096, "auto", 12, 9.0) == (pytest.approx(108), 64)
    # 100 MB blocks: streamed by rows (16 MB each), but at least one whole block
    assert memory.reader_share(4096, "auto", 4, 100.0) == (pytest.approx(100), 64)


def test_auto_reader_share_is_capped_at_a_quarter_but_never_refuses_a_block(caplog):
    decode, cache = memory.reader_share(400, "auto", 8, 16.0)       # wants 128 + 64
    assert decode + cache == pytest.approx(100)                      # a quarter of 400
    with caplog.at_level(logging.WARNING):
        decode, cache = memory.reader_share(400, "auto", 4, 300.0)  # a 300 MB block
    assert decode >= 300 and "cannot all be streamed" in caplog.text


def test_an_explicit_decode_memory_overrides():
    decode, cache = memory.reader_share(4096, "1GB", 4, 9.0)
    assert decode + cache == pytest.approx(1024) and cache == pytest.approx(64)


# --- scenes and regions ------------------------------------------------------------------------

def test_the_scenes_share_what_is_left(caplog):
    assert memory.scene_mb(4096, 300, 100, 2) == pytest.approx((4096 - 400) / 2)
    with caplog.at_level(logging.WARNING):
        assert memory.scene_mb(500, 300, 100, 4) == memory.MIN_SCENE_MB
    assert "raise memory_budget" in caplog.text


@pytest.mark.parametrize("scene, threads, region", [(3000, 4, 250), (30000, 4, 256),
                                                    (60, 4, 8)])
def test_auto_region_size(scene, threads, region):
    assert memory.region_mb("auto", scene, threads) == pytest.approx(region)


def test_an_explicit_region_size_is_kept():
    assert memory.region_mb(512.0, 100, 4) == 512.0


def test_old_memory_per_worker_becomes_the_slurm_worker_memory():
    c = ClusterConfig(memory_per_worker="12GB")
    assert c.slurm_worker_memory == "12GB" and c.memory_budget == "auto"
