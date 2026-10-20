"""Every scene and every tile is written with its OWN pixels (eubi-bridge 0.1.6).

Two bugs of 0.1.2-0.1.5, both silent (the conversion succeeded):

- A multi-series file in a format without a native reader (read through bioio's
  Bio-Formats reader) gave every scene the pixels of series 0. Such files are now read
  as with ``--force_bioformats``: bfio, one series per scene.
- Scenes of one file with different numbers of tiles: every scene used scene 0's tile
  count, so a scene with fewer tiles was written several times over and one with more
  lost its last tiles. Each scene now uses its own count.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import requires_full

tifffile = pytest.importorskip("tifffile")
zarr = pytest.importorskip("zarr")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _eubi() -> str:
    for folder in (Path(sys.executable).parent, Path(sys.executable).parent / "Scripts"):
        for name in ("eubi.exe", "eubi"):
            if (folder / name).exists():
                return str(folder / name)
    pytest.skip("no eubi executable next to this Python")


def _convert(src, out, *args):
    result = subprocess.run([_eubi(), "to_zarr", str(src), str(out), *args],
                            capture_output=True, text=True, env=dict(os.environ))
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]


def _values(out) -> dict:
    """{store name: the distinct pixel values of its level 0}."""
    return {os.path.basename(s): np.unique(np.asarray(zarr.open_group(s, mode="r")["0"])).tolist()
            for s in sorted(glob.glob(os.path.join(str(out), "*.zarr")))}


@requires_full
def test_each_series_of_a_bioformats_file_keeps_its_pixels(tmp_path):
    """A 2-series OME-TIFF named .ome.btf has no native reader: Bio-Formats reads it."""
    src = tmp_path / "two.ome.btf"
    with tifffile.TiffWriter(str(src), ome=True) as tw:
        tw.write(np.full((2, 32, 40), 10, "u1"), photometric="minisblack",
                 metadata={"axes": "CYX"})
        tw.write(np.full((2, 32, 40), 200, "u1"), photometric="minisblack",
                 metadata={"axes": "CYX"})
    _convert(src, tmp_path / "out", "--series_index", "all", "--metadata_reader", "bfio",
             "--pixel_reader", "standard")
    assert _values(tmp_path / "out") == {"two_scene0.zarr": [10], "two_scene1.zarr": [200]}


def test_each_scene_writes_its_own_tiles(tmp_path):
    """A CZI whose scene 0 has 2 tiles and scene 1 has 3, every tile its own value."""
    from tests._czi_writer import raw_payload, write_czi
    scaling = ('<Scaling><Items><Distance Id="X"><Value>5E-07</Value></Distance>'
               '<Distance Id="Y"><Value>5E-07</Value></Distance></Items></Scaling>')
    tiles, value = [], 0
    for scene, n in enumerate((2, 3)):
        for m in range(n):
            value += 10
            x, y = 200 * scene + 32 * (m % 2), 32 * (m // 2)
            tiles.append(dict(scene=scene, mosaic=m, plane={"C": 0}, x=x, y=y, height=32,
                              width=32, dtype="u1", compression=0,
                              payload=raw_payload(np.full((32, 32), value, "u1")),
                              tags={"StageXPosition": 1000 + 0.5 * x,
                                    "StageYPosition": 500 + 0.5 * y}))
    src = tmp_path / "mosaic.czi"
    write_czi(str(src), tiles, metadata=scaling)
    _convert(src, tmp_path / "out", "--series_index", "all", "--mosaic_tile_index", "all")
    assert _values(tmp_path / "out") == {
        "mosaic_scene0_tile0.zarr": [10], "mosaic_scene0_tile1.zarr": [20],
        "mosaic_scene1_tile0.zarr": [30], "mosaic_scene1_tile1.zarr": [40],
        "mosaic_scene1_tile2.zarr": [50]}
