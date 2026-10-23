"""A float16 image (Leica FLIM results) converts: OME-XML has no half float, so the XML says
float and the array stays float16; and the partial-conversion notice names the reason."""
import logging

import numpy as np

from eubi_bridge.core.data_manager import _ome_pixeltype_from_dtype, create_ome_xml


def test_half_float_maps_to_the_ome_float_type():
    from ome_types.model import PixelType
    assert _ome_pixeltype_from_dtype(np.float16) == PixelType.FLOAT
    assert _ome_pixeltype_from_dtype(np.dtype("float16")) == PixelType.FLOAT


def test_the_partial_notice_gives_the_reason_not_a_damaged_file(caplog):
    from types import SimpleNamespace
    from eubi_bridge.conversion.conversion_worker import _report_failures
    job = SimpleNamespace(input_path="/data/x.lif")
    results = ["ok", ValueError("Unsupported dtype: float16"), ValueError("Unsupported dtype: float16")]
    with caplog.at_level(logging.WARNING):
        _report_failures(results, job, total_in_file=3)
    text = caplog.text
    assert "Unsupported dtype: float16" in text and "Reason(s)" in text
    assert "DAMAGED FILE" not in text and "Zeiss" not in text
