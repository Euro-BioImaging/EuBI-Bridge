"""Channels without a colour of their own (eubi-bridge 0.3.2).

ozp writes a channel without a colour white, and the older readers give every channel
white, so a multichannel image showed all grey. eubi fills them: the colour of the
channel's emission wavelength (the file's, else its dye's by name), else distinct
defaults. Colours the file gives are kept; a colour two channels share is a reader's
default and is replaced; one channel stays as it is.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

ozp = pytest.importorskip("ome_zarr_pyramid")

from eubi_bridge.conversion.pyramid_path import (  # noqa: E402
    fill_channel_colors, wavelength_color)


class _Nm:
    def __init__(self, value):
        self.value = value

    def to(self, unit):
        assert unit == "nanometer"
        return self


def _pyramid(channels: int):
    """As ozp's from_image leaves it: every channel white."""
    pyr = ozp.Pyramid().from_array(np.zeros((channels, 8, 8), np.uint16), axis_order="cyx")
    return pyr.set_channels({i: {"color": "FFFFFF"} for i in range(channels)})


def _image(*channels):
    return SimpleNamespace(metadata=SimpleNamespace(channels=[
        SimpleNamespace(name=name, color=color,
                        emission_wavelength=None if nm is None else _Nm(nm))
        for name, color, nm in channels]))


def _colors(pyr):
    return [ch["color"] for ch in pyr.meta.metadata["omero"]["channels"]]


@pytest.mark.parametrize("nm,expected", [(461, "006BFF"), (519, "21FF00"),
                                         (670, "FF0000"), (1200, None)])
def test_wavelength_colors(nm, expected):
    assert wavelength_color(nm) == expected


def test_reader_white_everywhere_becomes_the_dyes_colors():
    """The DeltaVision case: every channel white, named after its dye."""
    image = _image(("DAPI", "FFFFFF", None), ("FITC", "FFFFFF", None),
                   ("RD-TR-PE", "FFFFFF", None), ("CY-5", "FFFFFF", None))
    assert _colors(fill_channel_colors(_pyramid(4), image)) == [
        wavelength_color(461), wavelength_color(519), wavelength_color(615), "FF0000"]


def test_the_files_wavelengths_win_over_names():
    image = _image(("x", None, 461), ("y", None, 670))
    assert _colors(fill_channel_colors(_pyramid(2), image)) == [wavelength_color(461),
                                                                "FF0000"]


def test_without_names_or_wavelengths_defaults_are_distinct():
    colors = _colors(fill_channel_colors(_pyramid(3), _image(*[(None, None, None)] * 3)))
    assert colors == ["FF0000", "00FF00", "0000FF"]


def test_defaults_avoid_colors_close_to_ones_in_use():
    image = _image(("a", None, 525), ("b", None, None))      # 525 nm: a yellowish green
    colors = _colors(fill_channel_colors(_pyramid(2), image))
    assert colors[0] == wavelength_color(525) and colors[1] == "FF0000"
    image = _image(("a", "FF0000", None), ("b", None, None))
    pyr = _pyramid(2).set_channels({0: {"color": "FF0000"}})
    assert _colors(fill_channel_colors(pyr, image)) == ["FF0000", "00FF00"]


def test_the_files_own_colors_are_kept():
    """from_image wrote them already: nothing is changed."""
    image = _image(("a", "FF00FF", None), ("b", "00FFFF", None))
    pyr = _pyramid(2).set_channels({0: {"color": "FF00FF"}, 1: {"color": "00FFFF"}})
    assert _colors(fill_channel_colors(pyr, image)) == ["FF00FF", "00FFFF"]


def test_transmitted_light_stays_white():
    """An OME file's white (Color -1) is OME's default too and reads as no colour."""
    image = _image(("GFP", "00FF00", None), ("Brightfield", None, None), ("DIC", None, None))
    pyr = _pyramid(3).set_channels({0: {"color": "00FF00"}})
    assert _colors(fill_channel_colors(pyr, image)) == ["00FF00", "FFFFFF", "FFFFFF"]


def test_one_channel_stays_as_it_is():
    pyr = _pyramid(1)
    before = _colors(pyr)
    assert _colors(fill_channel_colors(pyr, _image(("a", None, None)))) == before


def test_many_channels_end_quickly():
    """A 30-channel lambda scan gives every colour of the spectrum: the default search
    must still end (eubi 0.3.2 candidate looped forever on paramecium.lsm)."""
    import time
    n = 40
    image = _image(*[(f"ch{i}", None, 400 + 8 * i) for i in range(n - 8)],
                   *[(None, None, None)] * 8)
    t0 = time.time()
    colors = _colors(fill_channel_colors(_pyramid(n), image))
    assert time.time() - t0 < 5
    assert len(colors) == n and all(len(c) == 6 for c in colors)
