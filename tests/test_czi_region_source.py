"""CZI pylibCZIrw images are read one writer region at a time.

bioio's pylibCZIrw reader makes each dask chunk a full scene plane, and dask
computes whole chunks, so every writer region re-read the whole plane -- for a
stitched mosaic, the entire field of view.  ``_CziRegionSource`` asks pylibCZIrw
for exactly each region's rectangle.  Its output was verified bit-identical to
the bioio path on nine real CZIs (RGB, multi-scene, stitched mosaics,
illuminations separate and concatenated); these tests pin the read arguments.
"""
from __future__ import annotations

import numpy as np
import pytest

from eubi_bridge.core import czi_reader
from eubi_bridge.core.czi_reader import _CziRegionSource


class _FakeCzi:
    """Records read() calls; pixel value = 1000*T + 100*C + 10*Z + sample."""

    def __init__(self, width, height, samples=1):
        self.width, self.height, self.samples = width, height, samples
        self.calls = []

    def read(self, roi, scene, plane):
        self.calls.append((roi, scene, dict(plane)))
        x, y, w, h = roi
        base = 1000 * plane["T"] + 100 * plane["C"] + 10 * plane["Z"]
        out = np.empty((h, w, self.samples), dtype=np.uint16)
        for s in range(self.samples):
            out[:, :, s] = base + s
        return out


@pytest.fixture
def fake(monkeypatch):
    def make(**kw):
        reader = _FakeCzi(**kw)
        monkeypatch.setattr(czi_reader, "_cached_czi", lambda path: reader)
        return reader
    return make


def _source(shape, origin=(500, 300), fixed=None, rgb=False, scene=1):
    return _CziRegionSource("x.czi", scene, origin, shape, np.uint16,
                            fixed or {}, rgb)


def test_reads_exactly_the_requested_rectangle(fake):
    reader = fake(width=400, height=200)
    source = _source((1, 2, 3, 200, 400))

    out = source[0:1, 1:2, 0:3, 50:80, 100:160]

    assert out.shape == (1, 1, 3, 30, 60)
    # Offset by the scene's origin, one read per plane, scene passed through.
    assert [c[0] for c in reader.calls] == [(600, 350, 60, 30)] * 3
    assert {c[1] for c in reader.calls} == {1}
    assert [c[2] for c in reader.calls] == [{"T": 0, "C": 1, "Z": z} for z in range(3)]
    assert out[0, 0, 2, 0, 0] == 100 + 20


def test_view_or_illumination_is_pinned(fake):
    reader = fake(width=10, height=10)
    _source((1, 1, 1, 10, 10), fixed={"I": 1})[...]
    assert reader.calls[0][2] == {"T": 0, "C": 0, "Z": 0, "I": 1}


def test_rgb_samples_become_channels_in_rgb_order(fake):
    """CZI stores B, G, R; output channels are R, G, B, from one read."""
    reader = fake(width=8, height=4, samples=3)
    out = _source((1, 3, 1, 4, 8), rgb=True)[...]
    assert [out[0, c, 0, 0, 0] for c in range(3)] == [2, 1, 0]
    assert len(reader.calls) == 1


def test_numpy_indexing_semantics(fake):
    fake(width=20, height=12)
    source = _source((2, 2, 2, 12, 20))
    full = source[...]
    for key in [(1,), (0, 1, slice(None), 3), (Ellipsis, slice(2, 18, 5)),
                (-1, -1, -1, -1, -1)]:
        np.testing.assert_array_equal(source[key], full[key])


class TestBackendRouting:
    """pylibCZIrw serves everything except individual-tile extraction.

    Requests for views/illuminations (the GUI asks for 'all' of both by
    default) used to force aicspylibczi, so almost no GUI conversion got the
    region reads.  Both backends were verified identical for such files.
    """

    @staticmethod
    def _write(path, tiles):
        pyczi = pytest.importorskip("pylibCZIrw.czi")
        with pyczi.create_czi(str(path)) as czi:
            for x in range(tiles):
                czi.write(data=np.full((16, 24), 7 + x, dtype=np.uint16),
                          plane={"T": 0, "Z": 0, "C": 0}, location=(x * 30, 0))
        return str(path)

    def _backend(self, path, **kw):
        from eubi_bridge.core.czi_reader import read_czi
        data = read_czi(path, **kw).get_image_dask_data()
        return "pylibczirw" if type(data).__name__ == "DynamicArray" else "aics"

    def test_single_tile_with_all_views_uses_pylibczirw(self, tmp_path):
        path = self._write(tmp_path / "one.czi", tiles=1)
        assert self._backend(path, view_index="all",
                             illumination_index="all") == "pylibczirw"

    def test_individual_tiles_still_use_aicspylibczi(self, tmp_path):
        path = self._write(tmp_path / "two.czi", tiles=2)
        assert self._backend(path, as_mosaic=False) == "aics"

    def test_stitched_tiles_use_pylibczirw(self, tmp_path):
        path = self._write(tmp_path / "two.czi", tiles=2)
        assert self._backend(path, as_mosaic=True,
                             illumination_index="all") == "pylibczirw"
