"""One length unit per output (eubi-bridge 0.3.2).

micro-reader reports lengths as the file states them: an Olympus OIB gives its z step in
nanometres and its pixel size in micrometres. eubi writes every length of an image in
x's unit, converting scales and the translations of every level.
"""
from __future__ import annotations

import numpy as np
import pytest

ozp = pytest.importorskip("ome_zarr_pyramid")
pytest.importorskip("micro_reader")

from eubi_bridge.conversion.pyramid_path import harmonise_length_units  # noqa: E402


def _pyramid(units, scale, translation=None):
    pyr = ozp.Pyramid().from_array(np.zeros((4, 8, 8), np.uint16), axis_order="zyx",
                                   unit_list=list(units), scale=list(scale))
    if translation is not None:
        for path in pyr.meta.resolution_paths:
            pyr.meta.update_translation(path, list(translation))
    return pyr


def _state(pyr):
    meta = pyr.meta
    path = meta.resolution_paths[0]
    return (list(meta.unit_list), [round(float(s), 6) for s in meta.get_scale(path)],
            meta.get_translation(path) and [round(float(v), 6) for v in meta.get_translation(path)])


def test_nanometre_z_becomes_micrometres():
    pyr = _pyramid(["nanometer", "micrometer", "micrometer"], [510, 0.207, 0.207],
                   translation=[2000, 5, 5])
    units, scale, translation = _state(harmonise_length_units(pyr))
    assert units == ["micrometer"] * 3
    assert scale == [0.51, 0.207, 0.207] and translation == [2.0, 5, 5]


def test_a_shared_unit_is_left_alone():
    pyr = _pyramid(["micrometer"] * 3, [0.5, 0.2, 0.2], translation=[1, 2, 3])
    before = _state(pyr)
    assert _state(harmonise_length_units(pyr)) == before


def test_an_axis_without_a_unit_is_left_alone():
    pyr = _pyramid([None, "micrometer", "micrometer"], [1.0, 0.2, 0.2])
    before = _state(pyr)
    assert _state(harmonise_length_units(pyr)) == before


def test_x_sets_the_unit():
    pyr = _pyramid(["micrometer", "nanometer", "nanometer"], [0.5, 200, 200])
    units, scale, _ = _state(harmonise_length_units(pyr))
    assert units == ["nanometer"] * 3 and scale == [500.0, 200, 200]


def test_milliseconds_become_seconds():
    pyr = ozp.Pyramid().from_array(np.zeros((3, 8, 8), np.uint16), axis_order="tyx",
                                   unit_list=["millisecond", "micrometer", "micrometer"],
                                   scale=[300000, 1.0, 1.0])
    units, scale, _ = _state(harmonise_length_units(pyr))
    assert units == ["second", "micrometer", "micrometer"] and scale == [300.0, 1.0, 1.0]
