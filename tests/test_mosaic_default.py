"""Tiles are stitched by default, in the CLI and the GUI alike (a tiled slide gave 702 stores
without positions, gap hunt 2026-10-31), and an output carries its image's name."""
from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from eubi_bridge.conversion import pyramid_path
from eubi_bridge.conversion.legacy_image import _recorded
from eubi_bridge.core import micro_source
from eubi_bridge.core.config_models import ConversionJob, ReaderConfig
from eubi_bridge.ebridge import ConfigManager

micro_reader = pytest.importorskip("micro_reader")
pytest.importorskip("ome_zarr_pyramid")
tifffile = pytest.importorskip("tifffile")

from tests.test_series_index_rename import _czi  # noqa: E402


def _convert(src, out, **options) -> list:
    job = ConversionJob.from_kwargs(str(src), str(out), {"dtype": None, **options})
    asyncio.run(pyramid_path.convert(job))
    return sorted(p.name for p in out.glob("*.zarr"))


def test_tiles_are_stitched_by_default(tmp_path):
    assert _convert(_czi(tmp_path / "scan.czi", tiles=4), tmp_path / "out") == ["scan.zarr"]


def test_as_mosaic_false_keeps_the_tiles(tmp_path):
    made = _convert(_czi(tmp_path / "scan.czi", tiles=4), tmp_path / "out", as_mosaic=False)
    assert len(made) == 4


def test_a_chosen_tile_is_a_tile(tmp_path):
    assert ReaderConfig().as_mosaic is True
    assert ReaderConfig(mosaic_tile_index="all").as_mosaic is True
    assert ReaderConfig(mosaic_tile_index=2).as_mosaic is False
    assert ReaderConfig(mosaic_tile_index="0,2").as_mosaic is False
    made = _convert(_czi(tmp_path / "scan.czi", tiles=4), tmp_path / "out", mosaic_tile_index=2)
    assert len(made) == 1 and "tile2" in made[0]


def test_tiles_that_cannot_be_placed_stay_separate_and_it_says_so(tmp_path, monkeypatch, caplog):
    real = micro_source.open_renamed

    def no_positions(path, tiles, **options):
        if tiles == "mosaic":
            raise micro_reader.UnsupportedFormatError("tiles='mosaic': its tiles have no pixel size")
        return real(path, tiles=tiles, **options)

    monkeypatch.setattr(micro_source, "open_renamed", no_positions)
    with caplog.at_level(logging.WARNING):
        made = _convert(_czi(tmp_path / "scan.czi", tiles=4), tmp_path / "out")
    assert len(made) == 4
    assert "tiles cannot be stitched" in caplog.text and "no pixel size" in caplog.text


def _manager(tmp_path, readers: dict):
    path = tmp_path / ".eubi_config.json"
    path.write_text(json.dumps({"readers": readers}))
    return ConfigManager(str(path)), path


def test_a_config_saved_before_the_change_adopts_it_once(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        manager, path = _manager(tmp_path, {"as_mosaic": False})
    assert manager.config["readers"]["as_mosaic"] is True
    saved = json.loads(path.read_text())["readers"]
    assert saved["mosaic_default"] is True and saved["as_mosaic"] is True


def test_false_chosen_after_that_stays(tmp_path):
    manager, _path = _manager(tmp_path, {"as_mosaic": False, "mosaic_default": True})
    assert manager.config["readers"]["as_mosaic"] is False


def test_an_output_carries_its_image_name(tmp_path):
    src = tmp_path / "cells.ome.tif"
    tifffile.imwrite(src, np.zeros((8, 8), np.uint8), ome=True, metadata={"Name": "nuclei"})
    with micro_reader.open(str(src)) as f:
        name = f.images[0].name
    assert name and name != pyramid_path.NAME
    (made,) = _convert(src, tmp_path / "out")
    store = tmp_path / "out" / made
    attrs = dict(zarr.open_group(store, mode="r").attrs)
    assert attrs.get("ome", attrs)["multiscales"][0]["name"] == name
    assert f'Name="{name}"' in (store / "OME" / "METADATA.ome.xml").read_text()


def _klb_like(**sizes):
    pixels = SimpleNamespace(physical_size_x=None, physical_size_y=None, physical_size_z=None,
                             time_increment=None)
    for name, value in sizes.items():
        setattr(pixels, name, value)
    return SimpleNamespace(omemeta=SimpleNamespace(images=[SimpleNamespace(pixels=pixels)]))


def test_a_klb_of_ones_records_no_size():
    img = _klb_like(physical_size_x=1.0, physical_size_y=1.0, physical_size_z=1.0)
    assert _recorded({"x": 1.0, "y": 1.0, "z": 1.0}, img, 0, "a.klb") == {}
    assert _recorded({"x": 1.0, "y": 1.0, "z": 1.0}, img, 0, "a.ome.tif") != {}
    img = _klb_like(physical_size_x=0.4, physical_size_y=0.4)
    assert _recorded({"x": 0.4, "y": 0.4}, img, 0, "a.klb") == {"x": 0.4, "y": 0.4}
