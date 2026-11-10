"""The Inspect page opens zipped OME-Zarr (X.ome.zarr.zip, X.ozx) read-only: the browser
lists it as a dataset, the pyramid loads, TensorStore reads planes inside the zip (its zip
store), and the metadata tab reads the zip's attributes."""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("PyQt6")
dyna_io = pytest.importorskip("dyna_zarr").io
if not hasattr(dyna_io, "open_zip"):
    pytest.skip("dyna-zarr without zip writing", allow_module_level=True)
pytest.importorskip("zarr_zip")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eubi_bridge", "qt_gui",
                                "server"))

DATA = (np.arange(2 * 6 * 48 * 64, dtype="u2").reshape(2, 6, 48, 64) % 1009)


@pytest.mark.parametrize("name, version", [("img.ome.zarr.zip", "0.4"), ("img.ozx", "0.5")])
def test_a_zipped_ome_zarr_opens_in_inspect(tmp_path, name, version):
    from ome_zarr_pyramid import IO, Pyramid
    from zarr_plane_server import pyramid_cache, ts_cache  # type: ignore

    from eubi_bridge.qt_gui.core.file_service import is_ome_zarr_store, list_local
    from eubi_bridge.qt_gui.pages.inspect_page import _read_store_json

    zip_path = str(tmp_path / name)
    pyr = Pyramid().from_arrays([DATA], axis_order="czyx", version=version).downscale(
        n_layers=2, defer=False)
    IO().write_pyramid(pyr, zip_path)

    assert [(e["name"], e["isOmeZarr"]) for e in list_local(str(tmp_path))] == [(name, True)]
    assert is_ome_zarr_store(zip_path)
    loaded = pyramid_cache.get(zip_path)
    assert loaded.meta.resolution_paths == ["0", "1"] and tuple(loaded.shape) == DATA.shape
    store = ts_cache.get(zip_path, "0")
    assert store is not None                                   # TensorStore, not a fallback
    np.testing.assert_array_equal(store[:, 2, 10:20, 30:40].read().result(),
                                  DATA[:, 2, 10:20, 30:40])
    attrs = _read_store_json(zip_path, "zarr.json" if version == "0.5" else ".zattrs")
    held = attrs.get("attributes", attrs).get("ome", attrs.get("attributes", attrs))
    assert "multiscales" in held


def test_the_inspect_page_draws_a_zip_as_it_draws_the_folder(tmp_path):
    """The Inspect page opened on a zipped OME-Zarr renders the same picture as on the
    same image written as a folder (the viewer's own render path, off screen)."""
    import time

    from ome_zarr_pyramid import IO, Pyramid
    from PyQt6.QtWidgets import QApplication

    from tests.conftest import qt_available
    if not qt_available():
        pytest.skip("PyQt6 unavailable or no usable Qt platform plugin")
    app = QApplication.instance() or QApplication([])
    from eubi_bridge.qt_gui.pages.inspect_page import InspectPage

    pyr = Pyramid().from_arrays([DATA], axis_order="czyx", version="0.5").downscale(
        n_layers=2, defer=False)
    IO().write_pyramid(pyr, str(tmp_path / "img.ome.zarr"))
    IO().write_pyramid(pyr, str(tmp_path / "img.ome.zarr.zip"))

    def picture(path):
        page = InspectPage()
        frames = []
        page._viewer.frame_ready.connect(lambda rgb, *_: frames.append(np.array(rgb)))
        page._load_zarr(path)
        start = time.time()
        while time.time() - start < 60:
            app.processEvents()
            if frames and page._viewer.shape:
                time.sleep(0.3)                  # let a coarse preview be replaced
                app.processEvents()
                break
            time.sleep(0.05)
        assert frames, f"no frame drawn for {path}"
        assert page._pyr is not None and list(page._viewer.shape) == list(DATA.shape)
        page.close()
        return frames[-1]

    folder = picture(str(tmp_path / "img.ome.zarr"))
    zipped = picture(str(tmp_path / "img.ome.zarr.zip"))
    assert zipped.shape == folder.shape and zipped.max() > 0
    np.testing.assert_array_equal(zipped, folder)
