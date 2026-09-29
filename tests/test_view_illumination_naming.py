"""Each view / illumination gets its own output name.

The "_illu1" / "_view0" part of a name exists only inside the manager's
series_path.  Since 0.1.2 the converter passes a resolved basename (used to
keep same-named inputs apart), which replaces that whole stem -- so every view
and illumination got the same name and all but the first were skipped as
"Output already exists".  Seen on a two-illumination MouseBrain CZI.
"""
from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest

from eubi_bridge.conversion import conversion_worker as cw
from eubi_bridge.core.config_models import ConversionJob

SRC = os.path.join("data", "MouseBrain_2Illuminations.czi")


def _vi_manager(suffix):
    stem = os.path.splitext(SRC)[0]
    return SimpleNamespace(series_path=f"{stem}{suffix}.czi", series=0)


def _written_paths(monkeypatch, job, suffixes):
    managers = {s: _vi_manager(s) for s in suffixes}
    loaded = SimpleNamespace(
        _n_scenes=1, _n_tiles=1, loaded_scenes={"s0": None},
        loaded_views_illuminations=managers, loaded_tiles=None)

    async def fake_load(_job):
        return loaded

    written = []

    async def fake_process(man, out_path, _job, _sem):
        written.append(os.path.basename(out_path))

    monkeypatch.setattr(cw, "_load_input_manager", fake_load)
    monkeypatch.setattr(cw, "_process_single_scene_safe", fake_process)
    asyncio.run(cw.unary_worker(job))
    return written


@pytest.mark.parametrize("resolved", [None, "MouseBrain_2Illuminations",
                                      "runA-MouseBrain_2Illuminations"])
def test_each_illumination_gets_its_own_name(monkeypatch, tmp_path, resolved):
    job = ConversionJob.from_kwargs(SRC, str(tmp_path), {},
                                    resolved_basename=resolved)
    written = _written_paths(monkeypatch, job, ["_illu0", "_illu1"])
    base = resolved or "MouseBrain_2Illuminations"
    assert sorted(written) == [f"{base}_illu0.zarr", f"{base}_illu1.zarr"]


def test_views_and_concatenated_names_survive(monkeypatch, tmp_path):
    job = ConversionJob.from_kwargs(SRC, str(tmp_path), {},
                                    resolved_basename="MouseBrain_2Illuminations")
    written = _written_paths(monkeypatch, job,
                             ["_view0_illu_concat", "_view1_illu_concat"])
    assert sorted(written) == ["MouseBrain_2Illuminations_view0_illu_concat.zarr",
                               "MouseBrain_2Illuminations_view1_illu_concat.zarr"]
