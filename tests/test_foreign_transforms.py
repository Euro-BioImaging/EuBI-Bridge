"""An NGFF 0.6 input's other coordinate systems and transformations go on to the output.

ome_zarr_pyramid reads them with the image (an affine to a registered frame, a scale to an
array system, ...), but eubi builds the output Pyramid from the image's scales alone, so they
were dropped. They are carried while the output still has the input's axes, pixel size and
position; after a squeeze, crop or rescale they would point at the wrong pixels and are
dropped with a warning.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("micro_reader")
pytest.importorskip("ome_zarr_pyramid")
zarr = pytest.importorskip("zarr")

AXES = [{"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"}]
EDGE = {"type": "scale", "scale": [2.0, 2.0],
        "input": {"name": "physical"}, "output": {"name": "array"}}


def _eubi() -> str:
    found = shutil.which("eubi")
    if found:
        return found
    for name in ("eubi.exe", "eubi"):
        for folder in (Path(sys.executable).parent, Path(sys.executable).parent / "Scripts"):
            if (folder / name).exists():
                return str(folder / name)
    pytest.skip("no eubi executable next to this Python")


def _input(path: Path) -> Path:
    """A hand-written 0.6 image (two levels) with a second system and an edge to it."""
    data = np.arange(64 * 64, dtype="u2").reshape(64, 64)
    g = zarr.open_group(str(path), mode="w", zarr_format=3)
    datasets = []
    for i in range(2):
        a = g.create_array(str(i), shape=data[::2 ** i, ::2 ** i].shape, dtype=data.dtype)
        a[...] = data[::2 ** i, ::2 ** i]
        datasets.append({"path": str(i), "coordinateTransformations": [
            {"type": "sequence", "input": {"path": str(i)}, "output": {"name": "physical"},
             "transformations": [{"type": "scale", "scale": [0.5 * 2 ** i] * 2},
                                 {"type": "translation", "translation": [10.0, 20.0]}]}]})
    ms = {"name": "img",
          "coordinateSystems": [{"name": "physical", "axes": AXES},
                                {"name": "array", "axes": [{"name": "y", "type": "space"},
                                                           {"name": "x", "type": "space"}]}],
          "coordinateTransformations": [EDGE], "datasets": datasets}
    g.attrs.put({"ome": {"version": "0.6", "multiscales": [ms]}})
    return path


def _convert(src: Path, out: Path, *args: str) -> str:
    env = dict(os.environ, COLUMNS="250")
    result = subprocess.run([_eubi(), "to_zarr", str(src), str(out), "--overwrite", "True", *args],
                            capture_output=True, text=True, env=env)
    log = " ".join((result.stdout + result.stderr).split())
    assert result.returncode == 0, log[-3000:]
    return log


def _store(out: Path) -> Path:
    (store,) = [p for p in out.iterdir() if p.suffix == ".zarr"]
    return store


def _ome(store: Path) -> dict:
    attrs = dict(zarr.open_group(str(store), mode="r").attrs)
    return attrs["ome"] if isinstance(attrs.get("ome"), dict) else attrs


def test_extra_systems_and_edges_are_written_with_a_06_output(tmp_path):
    src = _input(tmp_path / "in.zarr")
    _convert(src, tmp_path / "out", "--ome_zarr_version", "0.6")
    (ms,) = _ome(_store(tmp_path / "out"))["multiscales"]
    assert [c["name"] for c in ms["coordinateSystems"]] == ["physical", "array"]
    assert ms["coordinateTransformations"] == [EDGE]


def test_they_are_dropped_with_a_warning_when_the_pixel_size_changes(tmp_path):
    src = _input(tmp_path / "in.zarr")
    log = _convert(src, tmp_path / "out", "--ome_zarr_version", "0.6", "--y_scale", "1.0",
                   "--x_scale", "1.0")
    (ms,) = _ome(_store(tmp_path / "out"))["multiscales"]
    assert [c["name"] for c in ms["coordinateSystems"]] == ["physical"]
    assert not ms.get("coordinateTransformations")
    assert "are not written" in log and "array" in log


def test_a_05_output_keeps_them_under_the_custom_key(tmp_path):
    src = _input(tmp_path / "in.zarr")
    _convert(src, tmp_path / "out", "--ome_zarr_version", "0.5")
    attrs = dict(zarr.open_group(str(_store(tmp_path / "out")), mode="r").attrs)
    kept = attrs.get("eubi:ngff06")
    assert kept and kept["coordinateTransformations"] == [EDGE]
