"""CLI commands that no other suite exercised.

A coverage sweep before 0.1.3 found four public commands with no test at all:
``validate_aggregative``, ``update_pixel_meta``, ``show_configs`` and
``version``.  They are all reachable from the terminal, so a break in any of
them is a break a user meets directly.

``update_pixel_meta`` matters most here: it takes the same ``*_scale`` /
``*_unit`` keys that were just promoted to real config fields, so it is the
command most likely to be disturbed by that change.
"""
from __future__ import annotations

import contextlib
import io
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
def bridge(tmp_path, monkeypatch):
    """An EuBIBridge whose config lives in a temp dir, never the user's own.

    ConfigManager resolves its directory itself and does not read
    EUBI_CONFIG_DIR, so the path is injected rather than set in the
    environment -- a lesson from writing into the real config by accident.
    """
    from eubi_bridge.ebridge import ConfigManager, EuBIBridge
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    monkeypatch.setattr(ConfigManager, "_get_config_dir",
                        lambda self: config_dir)
    monkeypatch.setattr(ConfigManager, "_get_json_path",
                        lambda self: config_dir / ".eubi_config.json")
    return EuBIBridge()


@pytest.fixture
def z_stack(tmp_path):
    """Three single-plane TIFFs that concatenate along z."""
    for index in range(3):
        tifffile.imwrite(
            tmp_path / f"img_z{index}.tif",
            np.random.randint(0, 255, (1, 16, 16)).astype(np.uint8),
            imagej=True, metadata={"axes": "ZYX"})
    return tmp_path


def _captured(func, *args, **kwargs) -> str:
    """Run *func*, returning whatever it printed."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        func(*args, **kwargs)
    return buffer.getvalue()


class TestVersion:
    def test_it_reports_the_installed_version(self, bridge):
        from eubi_bridge import __version__
        assert __version__ in _captured(bridge.version)

    def test_the_version_is_not_the_dev_fallback(self):
        """'0.0.0+dev' means the package metadata could not be read."""
        from eubi_bridge import __version__
        assert __version__ != "0.0.0+dev"


class TestValidateAggregative:
    """A dry run: it must describe the conversion without performing it."""

    def test_it_returns_a_plan(self, bridge, z_stack, tmp_path):
        out = tmp_path / "out"
        plan = bridge.validate_aggregative(
            str(z_stack), str(out), concatenation_axes="z", z_tag="_z")
        assert plan.n_outputs == 1
        assert len(plan.outputs[0].source_files) == 3

    def test_it_writes_nothing(self, bridge, z_stack, tmp_path):
        """"Dry run" is the whole point; producing output would be a bug."""
        out = tmp_path / "out"
        bridge.validate_aggregative(
            str(z_stack), str(out), concatenation_axes="z", z_tag="_z")
        assert not out.exists() or not list(out.iterdir())

    def test_the_plan_names_its_output(self, bridge, z_stack, tmp_path):
        plan = bridge.validate_aggregative(
            str(z_stack), str(tmp_path / "out"),
            concatenation_axes="z", z_tag="_z")
        assert plan.outputs[0].output_path
        assert plan.file_workers >= 1

    def test_it_is_printable(self, bridge, z_stack, tmp_path):
        """The CLI shows this to the user, so __str__ must not raise."""
        plan = bridge.validate_aggregative(
            str(z_stack), str(tmp_path / "out"),
            concatenation_axes="z", z_tag="_z")
        assert "output" in str(plan).lower()


class TestShowConfigs:
    def test_it_lists_a_saved_config(self, bridge):
        bridge.save_as("mine")
        assert "mine" in _captured(bridge.show_configs)

    def test_it_runs_with_no_named_configs(self, bridge):
        """An empty config directory must not raise."""
        _captured(bridge.show_configs)

    def test_list_configs_agrees_with_it(self, bridge):
        bridge.save_as("alpha")
        bridge.save_as("beta")
        assert set(bridge.list_configs()) >= {"alpha", "beta"}


class TestUpdatePixelMeta:
    """Rewrites pixel sizes on an existing OME-Zarr, in place."""

    def _converted(self, bridge, tmp_path):
        source = tmp_path / "img.tif"
        tifffile.imwrite(
            source, np.random.randint(0, 255, (4, 16, 16)).astype(np.uint8),
            imagej=True, metadata={"axes": "ZYX"})
        out = tmp_path / "out"
        bridge.to_zarr(str(source), str(out), verbose=False)
        stores = list(out.glob("*.zarr"))
        assert stores, "conversion produced no store"
        return stores[0]

    def _scales(self, store: Path) -> list:
        meta = json.loads((store / ".zattrs").read_text())
        datasets = meta["multiscales"][0]["datasets"]
        return datasets[0]["coordinateTransformations"][0]["scale"]

    def test_it_updates_the_z_scale(self, bridge, tmp_path):
        store = self._converted(bridge, tmp_path)
        before = self._scales(store)
        bridge.update_pixel_meta(str(store), z_scale=0.75)
        after = self._scales(store)
        assert after != before
        assert 0.75 in after

    def test_it_leaves_other_axes_alone(self, bridge, tmp_path):
        """Only the named axis changes; the rest keep the file's own values."""
        store = self._converted(bridge, tmp_path)
        before = self._scales(store)
        bridge.update_pixel_meta(str(store), z_scale=0.75)
        after = self._scales(store)
        assert after[-1] == before[-1]   # x untouched
        assert after[-2] == before[-2]   # y untouched

    def test_it_accepts_a_unit(self, bridge, tmp_path):
        store = self._converted(bridge, tmp_path)
        bridge.update_pixel_meta(str(store), z_scale=2.0, z_unit="nanometer")
        axes = json.loads((store / ".zattrs").read_text())["multiscales"][0]["axes"]
        z_axis = next(a for a in axes if a["name"] == "z")
        assert z_axis.get("unit") == "nanometer"
