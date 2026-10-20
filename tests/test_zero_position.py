"""A tile at stage position (0, 0) is written with its translation, on every level.

Until 0.3.0 eubi wrote the translation itself when ome_zarr_pyramid had written none (ozp
0.2 dropped an all-zero position). ozp 0.3.0 (eubi's floor) writes every position, (0, 0)
included, and the per-level translations of the downscale method; eubi's workaround is
gone (plan 9.4a / 9.4b note), so this checks the result instead.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("micro_reader")
pytest.importorskip("ome_zarr_pyramid")
zarr = pytest.importorskip("zarr")


def _tiles_czi(path):
    """Two 128 x 128 tiles, 0.5 µm pixels, the first with its corner at (0, 0) (a CZI
    stage position is the tile's centre: 64 pixels in, less half a pixel)."""
    from tests._czi_writer import raw_payload, write_czi
    scaling = ('<Scaling><Items><Distance Id="X"><Value>5E-07</Value></Distance>'
               '<Distance Id="Y"><Value>5E-07</Value></Distance></Items></Scaling>')
    rng = np.random.default_rng(3)
    tiles = [dict(scene=0, mosaic=m, plane={"C": 0}, x=128 * m, y=0, height=128, width=128,
                  dtype="u2", compression=0,
                  payload=raw_payload(rng.integers(0, 4000, (128, 128)).astype("u2")),
                  tags={"StageXPosition": 31.75 + 64.0 * m, "StageYPosition": 31.75})
             for m in range(2)]
    write_czi(str(path), tiles, metadata=scaling)
    return path


def _translations(store) -> list:
    attrs = zarr.open_group(str(store), mode="r").attrs.asdict()
    datasets = (attrs.get("ome") or attrs)["multiscales"][0]["datasets"]
    return [next((t["translation"] for t in ds["coordinateTransformations"]
                  if t["type"] == "translation"), None) for ds in datasets]


def test_the_tile_at_the_origin_has_its_translation_on_every_level(tmp_path):
    from eubi_bridge.ebridge import EuBIBridge
    src = _tiles_czi(tmp_path / "tiles.czi")
    out = tmp_path / "out"
    EuBIBridge(configpath=str(tmp_path / "cfg")).to_zarr(
        str(src), str(out), use_threading=True, verbose=False, as_mosaic=False,
        mosaic_tile_index="all", n_layers=3, min_dimension_size=16)
    stores = sorted(out.glob("*.zarr"))
    assert [s.name for s in stores] == ["tiles_tile0.zarr", "tiles_tile1.zarr"]
    first, second = (_translations(s) for s in stores)
    assert len(first) == 3
    assert first == [[0.0, 0.0]] * 3                  # (0, 0): written, on every level
    assert second == [second[0]] * 3 and second[0] != [0.0, 0.0]
