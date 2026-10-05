"""A real multi-tile Zeiss CZI, read tile by tile and stitched.

Extracting individual tiles is the one case still routed to aicspylibczi, and
nothing in CI exercised it on a real file: the real-data suite needs private
data, and the synthetic routing test stubs aicspylibczi (it crashed on the
Windows runners opening a pylibCZIrw-written file that has no attachment
directory, which ZEN-written files do have).  This test downloads a small real
mosaic so every CI platform reads one.

Test data: "CZI (Carl Zeiss Image) dataset with artificial test camera images
with various dimension for testing libraries reading", Sebastian Rhode, Carl
Zeiss Microscopy GmbH, Zenodo record 7015307, CC-BY-4.0.
https://zenodo.org/records/7015307

The file is cached in ``~/.cache/eubi-bridge-test-data`` (or
``EUBI_TEST_CACHE``) and verified by MD5; without network access the tests
skip rather than fail.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pytest

NAME = "S=2_2x2_CH=1.czi"           # 2 scenes x 2x2 tiles, 1 channel
URL = f"https://zenodo.org/api/records/7015307/files/{NAME}/content"
MD5 = "10f8076f58f490b02e8a1a31f94f057b"
CACHE = Path(os.environ.get("EUBI_TEST_CACHE",
                            Path.home() / ".cache" / "eubi-bridge-test-data"))


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def zeiss_mosaic() -> str:
    path = CACHE / NAME
    if path.exists() and _md5(path) == MD5:
        return str(path)
    CACHE.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    try:
        with urllib.request.urlopen(URL, timeout=60) as response, \
                open(partial, "wb") as out:
            shutil.copyfileobj(response, out)
    except Exception as exc:                                # noqa: BLE001
        partial.unlink(missing_ok=True)
        pytest.skip(f"could not download the Zeiss test mosaic: {exc}")
    if _md5(partial) != MD5:
        partial.unlink(missing_ok=True)
        pytest.skip("downloaded Zeiss test mosaic failed its MD5 check")
    partial.replace(path)
    return str(path)


def test_tiles_are_read_individually(zeiss_mosaic):
    """aicspylibczi's native reading, on a real ZEN-written file."""
    from eubi_bridge.core.czi_reader import read_czi

    reader = read_czi(zeiss_mosaic, as_mosaic=False)
    assert reader.n_tiles == 4
    tiles = []
    for index in range(reader.n_tiles):
        reader.set_tile(index)
        tiles.append(np.asarray(reader.get_image_dask_data()))
    assert len({t.shape for t in tiles}) == 1, "tiles differ in shape"
    assert all(t.std() > 0 for t in tiles), "a tile came back blank"
    assert any(not np.array_equal(tiles[0], t) for t in tiles[1:]), \
        "every tile holds the same data"


def test_conversion_writes_one_output_per_tile(zeiss_mosaic, tmp_path):
    from eubi_bridge.ebridge import EuBIBridge

    out = tmp_path / "tiles"
    EuBIBridge().to_zarr(zeiss_mosaic, str(out), as_mosaic=False,
                         scene_index=0, mosaic_tile_index="all", verbose=False)
    assert len(list(out.glob("*.zarr"))) == 4


def test_conversion_stitches_the_mosaic(zeiss_mosaic, tmp_path):
    import zarr
    from eubi_bridge.core.czi_reader import read_czi
    from eubi_bridge.ebridge import EuBIBridge

    out = tmp_path / "stitched"
    EuBIBridge().to_zarr(zeiss_mosaic, str(out), as_mosaic=True,
                         scene_index=0, verbose=False)
    stores = list(out.glob("*.zarr"))
    assert len(stores) == 1
    stitched = zarr.open_group(str(stores[0]), mode="r")["0"]
    # A 2x2 mosaic is larger than one tile in both Y and X.
    tile_shape = read_czi(zeiss_mosaic, as_mosaic=False).get_image_dask_data().shape
    assert stitched.shape[-2] > tile_shape[-2]
    assert stitched.shape[-1] > tile_shape[-1]
