"""Conversion options that no suite exercised.

A coverage sweep before 0.1.3 found two with no test anywhere:

``on_local_cluster``  -- routes the whole conversion through a Dask
LocalCluster instead of the process pool, a different execution backend.
``export_acquisition_metadata`` -- writes acquisition details NGFF has no
field for into a namespaced attrs block, and defaults to *auto*, so its
behaviour changes with the shape of the conversion rather than being fixed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture
def single_tiff(tmp_path):
    source = tmp_path / "img.tif"
    tifffile.imwrite(
        source, np.random.randint(0, 255, (4, 2, 16, 16)).astype(np.uint8),
        imagej=True, metadata={"axes": "ZCYX"})
    return source


def _zattrs(store: Path) -> dict:
    return json.loads((store / ".zattrs").read_text())


def _only_store(out: Path) -> Path:
    stores = list(out.glob("*.zarr"))
    assert stores, f"no store produced under {out}"
    return stores[0]


class TestLocalClusterBackend:
    """The Dask LocalCluster path, chosen by on_local_cluster=True."""

    def test_a_conversion_completes(self, single_tiff, tmp_path):
        from eubi_bridge.ebridge import EuBIBridge
        out = tmp_path / "out"
        EuBIBridge().to_zarr(str(single_tiff), str(out),
                             on_local_cluster=True, verbose=False)
        assert _only_store(out).exists()

    def test_it_produces_the_same_shape_as_the_default_backend(
            self, single_tiff, tmp_path):
        """Choosing a backend must not change the data it writes."""
        import zarr
        from eubi_bridge.ebridge import EuBIBridge

        plain = tmp_path / "plain"
        EuBIBridge().to_zarr(str(single_tiff), str(plain), verbose=False)
        clustered = tmp_path / "clustered"
        EuBIBridge().to_zarr(str(single_tiff), str(clustered),
                             on_local_cluster=True, verbose=False)

        a = zarr.open_array(str(_only_store(plain) / "0"), mode="r")
        b = zarr.open_array(str(_only_store(clustered) / "0"), mode="r")
        assert a.shape == b.shape
        assert np.array_equal(a[...], b[...])

    def test_it_is_refused_with_concatenation(self, single_tiff, tmp_path):
        """ClusterConfig rejects the combination rather than failing mid-run."""
        from eubi_bridge.ebridge import EuBIBridge
        with pytest.raises(Exception):
            EuBIBridge().to_zarr(
                str(single_tiff.parent), str(tmp_path / "out"),
                concatenation_axes="z", z_tag="_z",
                on_local_cluster=True, verbose=False)


class TestExportAcquisitionMetadata:
    """Acquisition details NGFF cannot hold, in a namespaced attrs block.

    The ``eubi_bridge`` key itself is always present -- it carries the version
    stamp on every store we write.  The flag decides whether richer acquisition
    detail is *merged into* it, so the question is what the block contains,
    not whether it exists.
    """

    _KEY = "eubi_bridge"

    def test_the_version_stamp_is_always_written(self, single_tiff, tmp_path):
        """Provenance on every store, independently of this flag."""
        from eubi_bridge.ebridge import EuBIBridge
        out = tmp_path / "out"
        EuBIBridge().to_zarr(str(single_tiff), str(out),
                             export_acquisition_metadata=False, verbose=False)
        block = _zattrs(_only_store(out))[self._KEY]
        assert "version" in block

    def test_the_stamp_records_the_installed_version(self, single_tiff, tmp_path):
        from eubi_bridge import __version__
        from eubi_bridge.ebridge import EuBIBridge
        out = tmp_path / "out"
        EuBIBridge().to_zarr(str(single_tiff), str(out), verbose=False)
        assert _zattrs(_only_store(out))[self._KEY]["version"] == __version__

    def test_off_adds_nothing_beyond_the_stamp(self, single_tiff, tmp_path):
        from eubi_bridge.ebridge import EuBIBridge
        out = tmp_path / "out"
        EuBIBridge().to_zarr(str(single_tiff), str(out),
                             export_acquisition_metadata=False, verbose=False)
        assert set(_zattrs(_only_store(out))[self._KEY]) == {"version"}

    def test_on_is_accepted_and_still_converts(self, single_tiff, tmp_path):
        """A plain single-scene input has no extra detail to record, so the
        block may still hold only the stamp; what matters is it converts."""
        from eubi_bridge.ebridge import EuBIBridge
        out = tmp_path / "out"
        EuBIBridge().to_zarr(str(single_tiff), str(out),
                             export_acquisition_metadata=True, verbose=False)
        assert self._KEY in _zattrs(_only_store(out))

    def test_the_block_does_not_disturb_ngff_metadata(
            self, single_tiff, tmp_path):
        """It is namespaced precisely so a reader can ignore it."""
        from eubi_bridge.ebridge import EuBIBridge
        out = tmp_path / "out"
        EuBIBridge().to_zarr(str(single_tiff), str(out),
                             export_acquisition_metadata=True, verbose=False)
        attrs = _zattrs(_only_store(out))
        assert "multiscales" in attrs
        assert attrs["multiscales"][0]["axes"]

    def test_auto_is_the_default(self):
        """None means "decide from the conversion", not "off"."""
        from eubi_bridge.core.config_models import ConversionConfig
        assert ConversionConfig().export_acquisition_metadata is None

    def test_auto_completes_a_plain_conversion(self, single_tiff, tmp_path):
        from eubi_bridge.ebridge import EuBIBridge
        out = tmp_path / "out"
        EuBIBridge().to_zarr(str(single_tiff), str(out), verbose=False)
        assert _only_store(out).exists()
