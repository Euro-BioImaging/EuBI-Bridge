"""Every image index converts (decision E-1): an index outside series, tile, view and
illumination gives one output per value, its name ending in the index; a backend that
cannot make ``sequence`` an axis keeps it as an index."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_pyramid_path import _convert  # noqa: E402

tifffile = pytest.importorskip("tifffile")
micro_reader = pytest.importorskip("micro_reader")

from eubi_bridge.conversion.pyramid_path import (_full_key, _other_suffix,  # noqa: E402
                                                  _other_values, _picked)
from eubi_bridge.core import micro_source  # noqa: E402


def test_the_values_outside_the_selected_indices():
    keys = [(("series", 0), ("block", 1)), (("series", 0), ("block", 0))]
    assert _other_values(keys) == [(("block", 0),), (("block", 1),)]
    assert _other_values([(("series", 0), ("tile", 2))]) == [()]
    assert _other_suffix((("block", 1), ("lifetime", 2)), {"block", "lifetime"}) == "_block1_lifetime2"
    assert _other_suffix((("plate", 0), ("well", 2)), {"well"}) == "_well2"     # only what varies


def test_an_index_with_an_option_is_picked_not_split():
    readers = SimpleNamespace(phase_index=1, rotation_index=None)
    keys = [(("series", 0), ("phase", 0)), (("series", 0), ("phase", 1))]
    assert _picked(keys, readers, "x.czi") == keys[1:]
    assert _other_suffix((("phase", 1),), {"phase"}) == ""
    with pytest.raises(ValueError, match=r"No valid phase index in \[5\]: x.czi has 2 phase"):
        _picked(keys, SimpleNamespace(phase_index=5, rotation_index=0), "x.czi")


def test_a_key_is_found_in_the_files_own_order():
    keys = [(("series", 0), ("block", 1), ("tile", 2))]
    assert _full_key((("series", 0), ("tile", 2)), (("block", 1),), keys) == keys[0]


def test_one_output_per_value_of_another_index(tmp_path):
    path = tmp_path / "flim.tif"
    tifffile.imwrite(path, np.arange(2 * 3 * 16 * 16, dtype=np.uint16).reshape(2, 3, 16, 16),
                     metadata={"axes": "HZYX"})
    out = tmp_path / "out"
    log = _convert(path, out, [])
    assert sorted(p.name for p in out.glob("*.zarr")) == ["flim_lifetime0.zarr", "flim_lifetime1.zarr"]
    assert "read by micro-reader" in log.replace("\n", " ")


def test_a_backend_that_cannot_rename_keeps_the_index(monkeypatch):
    calls = []

    def fake_open(path, **options):
        calls.append(options)
        if "rename" in options:
            raise micro_reader.UnsupportedFormatError("rename: x: the bdv backend cannot ...")
        return "opened"
    monkeypatch.setattr(micro_reader, "open", fake_open)
    assert micro_source.open_renamed("x", plates=False) == "opened"
    assert calls == [{"rename": micro_source.RENAME, "plates": False}, {"plates": False}]


def test_another_refusal_is_not_swallowed(monkeypatch):
    def fake_open(path, **options):
        raise micro_reader.UnsupportedFormatError("no backend")
    monkeypatch.setattr(micro_reader, "open", fake_open)
    with pytest.raises(micro_reader.UnsupportedFormatError):
        micro_source.open_renamed("x")
