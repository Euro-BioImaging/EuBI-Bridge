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


class TestJobsCarryEveryConfigSection:
    """A job built field-by-field must be given every config section.

    ``AggregativeConversionJob`` is constructed with explicit keywords rather
    than ``from_kwargs``, and the model ignores unknown keys -- so omitting
    ``metadata=`` silently produced a default MetadataConfig and
    ``override_channel_names`` was always False.  That shipped, and CI caught
    it on every OS because it is pure object construction with no platform
    component.
    """

    _SECTIONS = ("cluster", "readers", "conversion", "downscale", "metadata")

    def test_every_section_is_a_field(self):
        from eubi_bridge.core.config_models import (AggregativeConversionJob,
                                                    ConversionJob)
        for model in (ConversionJob, AggregativeConversionJob):
            for section in self._SECTIONS:
                assert section in model.model_fields, f"{model.__name__}.{section}"

    def test_from_kwargs_populates_metadata(self):
        """The unary path; it worked, and must keep working."""
        from eubi_bridge.core.config_models import ConversionJob
        job = ConversionJob.from_kwargs(
            "/in.tif", "/out", {"override_channel_names": True})
        assert job.metadata.override_channel_names is True

    def test_an_explicit_metadata_section_survives_flattening(self):
        """The aggregative path: the value has to reach the worker kwargs."""
        from eubi_bridge.core.config_models import (AggregativeConversionJob,
                                                    MetadataConfig)
        job = AggregativeConversionJob(
            input_path=["/a.tif"], output_path="/out",
            metadata=MetadataConfig(override_channel_names=True))
        assert job.to_conversion_kwargs()["override_channel_names"] is True

    def test_a_stray_keyword_does_not_silently_vanish_unnoticed(self):
        """Documents the trap: extra="ignore" drops a misplaced section key.

        Passing override_channel_names at the top level looks reasonable and is
        silently discarded -- which is exactly how the regression happened.  If
        this ever starts raising instead, the trap is gone and the test should
        be updated to match.
        """
        from eubi_bridge.core.config_models import AggregativeConversionJob
        job = AggregativeConversionJob(
            input_path=["/a.tif"], output_path="/out",
            override_channel_names=True)          # wrong level, ignored
        assert job.metadata.override_channel_names is False

    def test_every_construction_site_passes_all_sections(self):
        """Catch the omission where it actually happens: at the call site.

        The behavioural tests above prove a correctly built job works.  This
        one proves nobody builds one incorrectly -- which is the mistake that
        shipped, since a missing section is silently accepted.
        """
        import re
        from pathlib import Path

        pattern = re.compile(
            r"(?:Aggregative)?ConversionJob\(\s*\n(.*?)\n\s*\)", re.S)
        root = Path(__file__).resolve().parents[1] / "eubi_bridge"
        offenders = []
        for path in root.rglob("*.py"):
            if "__pycache__" in str(path):
                continue
            text = path.read_text(encoding="utf-8")
            for match in pattern.finditer(text):
                block = match.group(1)
                if "from_kwargs" in block:
                    continue
                present = {name for name in self._SECTIONS
                           if re.search(r"\b" + name + r"\s*=", block)}
                if present and present != set(self._SECTIONS):
                    missing = sorted(set(self._SECTIONS) - present)
                    line = text[:match.start()].count("\n") + 1
                    offenders.append(f"{path.name}:{line} missing {missing}")
        assert not offenders, (
            "job constructed without every config section: "
            + "; ".join(offenders))
