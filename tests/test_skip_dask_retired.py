"""skip_dask is retired (2026-10-05): accepted, warned about, ignored.

micro-reader, the default reader, creates no dask arrays, so there was
nothing to skip.  What the option still did was send TIFFs past micro-reader,
to eubi's own TIFF route, overriding the reader the user had chosen.  Old
configs, saved batches and scripts keep working; nothing reads the value.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile


def test_still_accepted_but_ignored():
    from eubi_bridge.core.config_models import ConversionConfig
    assert ConversionConfig(skip_dask=True).skip_dask is None
    assert ConversionConfig().skip_dask is None


def test_dropped_from_an_old_config_on_load(tmp_path, monkeypatch):
    from eubi_bridge.ebridge import ConfigManager, EuBIBridge
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    path = config_dir / ".eubi_config.json"
    monkeypatch.setattr(ConfigManager, "_get_config_dir", lambda self: config_dir)
    monkeypatch.setattr(ConfigManager, "_get_json_path", lambda self: path)
    EuBIBridge().config                                  # writes defaults
    old = json.loads(path.read_text())
    old["conversion"]["skip_dask"] = True                # as 0.1.3 wrote it
    path.write_text(json.dumps(old))

    assert "skip_dask" not in EuBIBridge().config["conversion"]
    assert "skip_dask" not in json.loads(path.read_text())["conversion"]


def test_configure_does_not_save_it(tmp_path):
    from eubi_bridge.ebridge import EuBIBridge
    bridge = EuBIBridge(configpath=str(tmp_path / "cfg"))
    bridge.configure.conversion(skip_dask=True)
    assert "skip_dask" not in EuBIBridge(configpath=str(tmp_path / "cfg")).config["conversion"]


def test_a_conversion_given_it_still_reads_with_micro_reader(tmp_path, monkeypatch):
    """The bug it carried: skip_dask=True sent a TIFF to eubi's own TIFF route
    instead of micro-reader.  Every reader the conversion opens is asked."""
    from eubi_bridge.core import data_manager
    from eubi_bridge.ebridge import EuBIBridge

    asked = []
    original = data_manager.ArrayManager.__init__

    def spy(self, *args, **kwargs):
        asked.append(kwargs.get("skip_dask", False))
        original(self, *args, **kwargs)

    monkeypatch.setattr(data_manager.ArrayManager, "__init__", spy)
    data = np.arange(2 * 3 * 24 * 24, dtype="u2").reshape(2, 3, 24, 24)
    source = tmp_path / "a.ome.tif"
    tifffile.imwrite(source, data, metadata={"axes": "CZYX"})
    out = tmp_path / "out"
    EuBIBridge(configpath=str(tmp_path / "cfg")).to_zarr(
        str(source), str(out), skip_dask=True, use_threading=True, verbose=False)

    assert asked and not any(asked)
    import zarr
    (store,) = out.glob("*.zarr")
    np.testing.assert_array_equal(np.asarray(zarr.open_group(str(store), mode="r")["0"]).squeeze(),
                                  data)


def test_the_gui_offers_it_no_more():
    from eubi_bridge.qt_gui.core.batch import spec_for
    assert spec_for("skip_dask") is None
    from tests.conftest import qt_available
    if not qt_available():
        pytest.skip("Qt is not available")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])           # noqa: F841
    from eubi_bridge.qt_gui.pages.convert_page import ConvertPage
    page = ConvertPage()
    try:
        assert not hasattr(page, "_skip_dask")
        assert "skipDask" not in page._ui_to_config()["conversion"]
    finally:
        page.deleteLater()
