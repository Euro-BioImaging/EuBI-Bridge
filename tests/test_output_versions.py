"""The OME-Zarr versions eubi-bridge writes (0.3.2): 0.4, 0.5, 0.6 and NGFF RFC 8.

- ``--ome_zarr_version 0.6`` writes the image in NGFF 0.6's form (coordinate systems in
  the multiscale, one transformation per level), valid against the official schemas;
- ``--ome_zarr_version rfc8`` writes an RFC 8 collection (a draft, ozp warns) holding the
  image as an NGFF 0.5 multiscale; ozp reads it back with the image's system and position;
- any other version is an error naming the ones eubi writes (before 0.3.2 it fell back to
  zarr_format, i.e. 0.4, with only a log line).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("micro_reader")
ozp = pytest.importorskip("ome_zarr_pyramid")
tifffile = pytest.importorskip("tifffile")

RNG = np.random.default_rng(5)


def _eubi() -> str:
    found = shutil.which("eubi")
    if found:
        return found
    for name in ("eubi.exe", "eubi"):
        for folder in (Path(sys.executable).parent, Path(sys.executable).parent / "Scripts"):
            if (folder / name).exists():
                return str(folder / name)
    pytest.skip("no eubi executable next to this Python")


def _run(src: Path, out: Path, args: list) -> subprocess.CompletedProcess:
    env = dict(os.environ, COLUMNS="250")
    return subprocess.run([_eubi(), "to_zarr", str(src), str(out), *args],
                          capture_output=True, text=True, env=env)


def _convert(src: Path, out: Path, args: list) -> str:
    result = _run(src, out, args)
    log = " ".join((result.stdout + result.stderr).split())
    assert result.returncode == 0, log[-3000:]
    return log


def _tiff(path: Path) -> Path:
    data = RNG.integers(0, 4000, (2, 3, 40, 48)).astype("u2")
    with tifffile.TiffWriter(str(path), ome=True) as tw:
        tw.write(data, photometric="minisblack", metadata={
            "axes": "CZYX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5, "PhysicalSizeZ": 2.0,
            "Plane": {"PositionX": [100.0] * 6, "PositionXUnit": ["µm"] * 6,
                      "PositionY": [50.0] * 6, "PositionYUnit": ["µm"] * 6}})
    return path


def _ome(store: Path) -> dict:
    return json.loads((store / "zarr.json").read_text(encoding="utf-8"))["attributes"]["ome"]


def test_ngff_06_output(tmp_path):
    src = _tiff(tmp_path / "img.ome.tif")
    _convert(src, tmp_path / "out", ["--ome_zarr_version", "0.6"])
    store = tmp_path / "out" / "img.zarr"
    ome = _ome(store)
    assert ome["version"] == "0.6"
    ms = ome["multiscales"][0]
    assert "coordinateSystems" in ms and "axes" not in ms
    if pytest.importorskip("jsonschema"):
        assert ozp.validate(str(store), strict=True) == []
    back = ozp.io.read_pyramid(str(store))
    assert back.meta.version == "0.6" and back.axes == "czyx"


def test_rfc8_output(tmp_path):
    src = _tiff(tmp_path / "img.ome.tif")
    log = _convert(src, tmp_path / "out", ["--ome_zarr_version", "rfc8"])
    assert "RFC 8 output is experimental" in log
    store = tmp_path / "out" / "img.zarr"
    ome = _ome(store)
    assert ome["version"] == "0.x" and ome["type"] == "collection"
    coll = ozp.PyramidCollection.from_zarr(str(store))
    (key,) = list(coll)
    pyr = coll[key]
    assert pyr.axes == "czyx" and pyr.meta.get_translation("0")[-2:] != [0.0, 0.0]


@pytest.mark.parametrize("given, expected", [("v0.6", "0.6"), ("RFC8", "rfc8"), (0.5, "0.5")])
def test_version_spellings(given, expected):
    from eubi_bridge.core.config_models import resolve_ome_zarr_target
    assert resolve_ome_zarr_target(given)[0] == expected


def test_an_unknown_version_is_an_error(tmp_path):
    from eubi_bridge.core.config_models import resolve_ome_zarr_target
    with pytest.raises(ValueError, match=r"ome_zarr_version '0\.7'.*'0\.6', 'rfc8'"):
        resolve_ome_zarr_target("0.7")
    result = _run(_tiff(tmp_path / "img.ome.tif"), tmp_path / "out",
                  ["--ome_zarr_version", "0.7"])
    assert result.returncode != 0
    assert "eubi-bridge writes OME-Zarr" in result.stdout + result.stderr
    assert not (tmp_path / "out" / "img.zarr").exists()
