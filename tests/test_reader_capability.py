"""A codec the install lacks stops a file at its first output (eubi-bridge 0.3.2).

micro-reader says "JPEG XR data needs micro-reader's native module or the [codecs] extra"
for a Zeiss slide scan without imagecodecs. Before, every tile of such a scan was tried,
written and removed again (1500 warnings for one missing package).
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

micro_reader = pytest.importorskip("micro_reader")
pytest.importorskip("ome_zarr_pyramid")

from eubi_bridge.core.config_models import ConversionJob  # noqa: E402
from tests.test_series_index_rename import _czi  # noqa: E402


def test_a_missing_codec_stops_the_file_at_its_first_output(tmp_path, monkeypatch, caplog):
    from eubi_bridge.conversion import pyramid_path
    from eubi_bridge.conversion.pyramid_path import ReaderCapabilityError

    tried = []

    def no_codec(entry, job, output_path):
        tried.append(output_path)
        raise micro_reader.MicroReaderError(
            "JPEG XR data needs micro-reader's native module or the [codecs] extra (imagecodecs)")

    monkeypatch.setattr(pyramid_path, "_prepared", no_codec)
    job = ConversionJob.from_kwargs(str(_czi(tmp_path / "scan.czi", tiles=4)), str(tmp_path / "out"),
                                    {"mosaic_tile_index": "all"})
    with caplog.at_level(logging.WARNING), pytest.raises(ReaderCapabilityError, match=r"\[codecs\] extra"):
        asyncio.run(pyramid_path.convert(job))
    assert len(tried) == 1                                  # the other tiles were not tried
    assert "Skipped a scene" not in caplog.text
    assert not list((tmp_path / "out").glob("*.zarr"))
