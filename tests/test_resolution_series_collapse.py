"""
Tests for the format-agnostic Bio-Formats resolution-pyramid collapse
(Track B): ``_bioformats_resolution_count`` and ``_collapse_resolution_series``.
"""

from types import SimpleNamespace

import pytest
from ome_types.model import OME, Image, Pixels, Pixels_DimensionOrder, PixelType

from eubi_bridge.core import data_manager
from eubi_bridge.core.data_manager import (_bioformats_resolution_count,
                                            _collapse_resolution_series)


def _make_image(idx: int) -> Image:
    pixels = Pixels(
        dimension_order=Pixels_DimensionOrder.XYZCT,
        type=PixelType.UINT8,
        size_x=4, size_y=4, size_z=1, size_c=1, size_t=1,
    )
    return Image(id=f"Image:{idx}", name=f"Series_{idx}", pixels=pixels)


def _make_omemeta(n_images: int) -> OME:
    return OME(images=[_make_image(i) for i in range(n_images)])


def _fake_cached_reader(resolution_counts):
    """Build a fake bfio.BioReader-like object exposing getCoreMetadataList()."""
    core = [SimpleNamespace(resolutionCount=rc) for rc in resolution_counts]
    rdr = SimpleNamespace(getCoreMetadataList=lambda: core)
    backend = SimpleNamespace(_rdr=rdr)
    return SimpleNamespace(_backend=backend)


def test_bioformats_resolution_count_returns_value(monkeypatch):
    monkeypatch.setattr(data_manager, "_get_cached_reader",
                         lambda path, series: _fake_cached_reader([5, 1, 1, 1, 1]))
    assert _bioformats_resolution_count("dummy.czi") == 5


def test_bioformats_resolution_count_returns_none_on_error(monkeypatch):
    def _raise(path, series):
        raise RuntimeError("boom")
    monkeypatch.setattr(data_manager, "_get_cached_reader", _raise)
    assert _bioformats_resolution_count("dummy.czi") is None


def test_collapse_when_resolution_count_matches_image_count(monkeypatch):
    monkeypatch.setattr(data_manager, "_get_cached_reader",
                         lambda path, series: _fake_cached_reader([5, 1, 1, 1, 1]))
    omemeta = _make_omemeta(5)
    result = _collapse_resolution_series(omemeta, "dummy.czi")
    assert len(result.images) == 1
    assert result.images[0].id == "Image:0"


def test_no_collapse_when_resolution_count_is_one(monkeypatch):
    # Genuine 5-series file: resolutionCount for series 0 is 1, not 5.
    monkeypatch.setattr(data_manager, "_get_cached_reader",
                         lambda path, series: _fake_cached_reader([1, 1, 1, 1, 1]))
    omemeta = _make_omemeta(5)
    result = _collapse_resolution_series(omemeta, "dummy.czi")
    assert len(result.images) == 5


def test_no_collapse_when_single_image(monkeypatch):
    def _raise(path, series):
        raise AssertionError("_get_cached_reader should not be called for a single image")
    monkeypatch.setattr(data_manager, "_get_cached_reader", _raise)
    omemeta = _make_omemeta(1)
    result = _collapse_resolution_series(omemeta, "dummy.czi")
    assert len(result.images) == 1


def test_no_collapse_when_get_cached_reader_raises(monkeypatch):
    def _raise(path, series):
        raise RuntimeError("boom")
    monkeypatch.setattr(data_manager, "_get_cached_reader", _raise)
    omemeta = _make_omemeta(5)
    result = _collapse_resolution_series(omemeta, "dummy.czi")
    assert len(result.images) == 5
