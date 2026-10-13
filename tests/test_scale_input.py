"""Physical scale overrides accept one notation and refuse everything else loudly.

In 0.1.3 the GUI dropped a scale it could not read as a number (``30 min``, ``0,5``)
without a word, but still applied the unit next to it, so a time step entered as
``30 min`` was written as "1.0 minute". A scale is now digits with '.' as the decimal
point; anything else is an error naming the field, in the GUI, the CLI, the Python API
and conversion tables alike.
"""
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eubi_bridge.core.config_models import (  # noqa: E402
    ConversionJob, MetadataConfig, MetadataUpdateConfig, parse_scale)

ACCEPTED = [("0.26", 0.26), ("1800", 1800.0), ("1000000", 1e6), (".5", 0.5), ("5.", 5.0),
            (" 5 ", 5.0), (0.26, 0.26), (1800, 1800.0), (1e-3, 1e-3)]
#: text, units, commas in either meaning, other separators, other notations, <= 0
REFUSED = ["30 min", "30min", "1800s", "0,5", "1,000,000", "1,5e3", "1 000", "1_000",
           "1e3", "inf", "nan", "-1", "0", "0.0", "abc", "1.2.3", True, float("inf"), -2.0]


@pytest.mark.parametrize("text, value", ACCEPTED)
def test_plain_numbers_are_read(text, value):
    assert parse_scale(text) == pytest.approx(value)


@pytest.mark.parametrize("blank", [None, "", "   ", float("nan")])
def test_blank_means_keep_the_files_value(blank):
    assert parse_scale(blank) is None


@pytest.mark.parametrize("bad", REFUSED)
def test_anything_else_is_an_error_naming_the_field(bad):
    with pytest.raises(ValueError, match="Time scale"):
        parse_scale(bad, "Time scale")


@pytest.mark.parametrize("bad", ["0,5", "1_000", "30 min", "1e3"])
def test_the_config_models_refuse_them(bad):
    with pytest.raises(ValidationError):
        MetadataConfig(time_scale=bad)
    with pytest.raises(ValidationError):
        MetadataUpdateConfig(z_scale=bad)
    assert MetadataConfig(time_scale="1800").time_scale == 1800.0


def test_a_conversion_tables_extra_value_is_checked_too():
    from eubi_bridge.conversion.conversion_worker import _axis_metadata_kwargs
    job = ConversionJob.from_kwargs("/in.tif", "/out", {})
    job.extra["time_scale"] = "0,5"
    with pytest.raises(ValueError, match="time_scale"):
        _axis_metadata_kwargs(job)


def _gui(**scales):
    meta = {"metadataReader": "bfio", "overridePhysicalScale": True,
            "scaleTime": "", "scaleZ": "", "scaleY": "", "scaleX": "",
            "unitTime": "minute", "unitZ": "micrometer",
            "unitY": "micrometer", "unitX": "micrometer"}
    meta.update(scales)
    return {"cluster": {}, "reader": {}, "conversion": {}, "downscaling": {},
            "concatenation": {}, "metadata": meta}


def test_the_gui_refuses_what_it_used_to_drop():
    from eubi_bridge.qt_gui.workers.conversion_worker import (
        _build_kwargs, scale_override_problems)
    cfg = _gui(scaleTime="30 min", scaleZ="0,5", scaleX="0.26")
    problems = scale_override_problems(cfg)
    assert len(problems) == 2
    assert problems[0].startswith("Time scale: '30 min'")
    assert problems[1].startswith("Z scale: '0,5'")
    with pytest.raises(ValueError, match="Time scale"):
        _build_kwargs(cfg)
    kwargs = _build_kwargs(_gui(scaleTime="30", scaleX="0.26"))
    assert (kwargs["time_scale"], kwargs["time_unit"], kwargs["x_scale"]) == (30.0, "minute", 0.26)
    assert scale_override_problems(dict(cfg, metadata=dict(
        cfg["metadata"], overridePhysicalScale=False))) == []      # off: not read at all


def test_saving_a_gui_config_refuses_them():
    from eubi_bridge.qt_gui.server.config_manager import _react_to_config
    with pytest.raises(ValueError, match="X scale"):
        _react_to_config(_gui(scaleX="0,26"))
    assert _react_to_config(_gui(scaleX="0.26"))["metadata"]["x_scale"] == 0.26


def test_a_batch_row_from_a_csv_is_checked():
    from eubi_bridge.qt_gui.core.batch import BatchModel
    cfg = {"cluster": {}, "reader": {}, "downscaling": {}, "metadata": {},
           "concatenation": {}, "conversion": {"omeZarrVersion": "0.5"}}
    model = BatchModel()
    model.set_baseline(cfg)
    model.add(cfg, ["a.tif", "b.tif"], "/out")
    model._rows[1]["z_scale"] = "1,000"                       # as a CSV cell holds it
    problems = model.validate()
    assert any(p.startswith("Row 2: Z scale: '1,000'") for p in problems)
    assert not any(p.startswith("Row 1: Z scale") for p in problems)
