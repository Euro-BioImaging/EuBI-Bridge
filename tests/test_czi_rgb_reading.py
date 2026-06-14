"""RGB / multi-sample CZI reading.

A Zeiss ``Bgr24`` CZI stores three samples per pixel (Blue, Green, Red) under a
single channel.  eubi-bridge must fold those samples into three OME-Zarr
channels in R, G, B order — not collapse them to a single channel (the original
bug reported by an external user) nor leave them in B, G, R order (which would
mis-colour the output).

These tests exercise the reader directly (no JVM / full pipeline needed).  They
are skipped if ``pylibCZIrw``'s write API is unavailable.
"""
import numpy as np
import pytest

pyczi = pytest.importorskip("pylibCZIrw.czi")


def _write_rgb_czi(path, b=50, g=100, r=150, h=32, w=48):
    """Write a uniform Bgr24 CZI; array sample order is (B, G, R)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = b
    img[..., 1] = g
    img[..., 2] = r
    with pyczi.create_czi(str(path)) as czi:
        czi.write(data=img, plane={"T": 0, "Z": 0, "C": 0}, location=(0, 0))
    return str(path)


@pytest.fixture
def rgb_czi(tmp_path):
    return _write_rgb_czi(tmp_path / "rgb.czi")


# Both backends must handle RGB: as_mosaic=False -> aicspylibczi (chunk_dims fix),
# as_mosaic=True -> patched pylibczirw (block-grid ndim fix).
@pytest.mark.parametrize("as_mosaic", [False, True])
def test_rgb_samples_fold_into_three_channels(rgb_czi, as_mosaic):
    from eubi_bridge.core.czi_reader import read_czi

    reader = read_czi(rgb_czi, as_mosaic=as_mosaic)
    data = np.asarray(reader.get_image_dask_data())  # T C Z Y X
    assert data.ndim == 5
    assert data.shape[1] == 3, (
        f"expected 3 channels from RGB samples, got {data.shape[1]}")


@pytest.mark.parametrize("as_mosaic", [False, True])
def test_rgb_channels_are_distinct_and_in_rgb_order(rgb_czi, as_mosaic):
    from eubi_bridge.core.czi_reader import read_czi

    reader = read_czi(rgb_czi, as_mosaic=as_mosaic)
    data = np.asarray(reader.get_image_dask_data())
    means = [round(float(data[:, c].mean())) for c in range(3)]
    # Stored order is B=50, G=100, R=150; output must be reversed to R, G, B
    # so the default per-channel colours (red, green, blue) tint the right data.
    assert means == [150, 100, 50], f"channels not in R, G, B order: {means}"


@pytest.mark.parametrize("as_mosaic", [False, True])
def test_rgb_samples_not_collapsed_to_single_sample(rgb_czi, as_mosaic):
    """Guards against the bioio bugs where every sample read returned sample 0
    (aicspylibczi) or crashed on the block ndim (pylibczirw)."""
    from eubi_bridge.core.czi_reader import read_czi

    reader = read_czi(rgb_czi, as_mosaic=as_mosaic)
    data = np.asarray(reader.get_image_dask_data())
    means = {round(float(data[:, c].mean())) for c in range(3)}
    assert len(means) == 3, f"samples collapsed — channels not distinct: {means}"
