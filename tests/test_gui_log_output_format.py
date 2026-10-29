"""The Convert log names the OME-Zarr version the outputs get (eubi-bridge 0.3.3,
Workplan E21). Until 0.3.2 it logged the base settings' zarr_format ('Zarr Format:
v2'), though the batch rows' ome_zarr_version 0.5 wrote zarr v3 stores."""
from __future__ import annotations

import pandas as pd

from eubi_bridge.qt_gui.workers.conversion_worker import _ome_zarr_targets


def test_a_run_names_the_version_and_its_zarr_format():
    assert _ome_zarr_targets({"ome_zarr_version": "0.5", "zarr_format": 3}) == \
        "Output format: OME-Zarr 0.5 (zarr v3)"
    assert _ome_zarr_targets({"ome_zarr_version": None, "zarr_format": 2}) == \
        "Output format: OME-Zarr 0.4 (zarr v2)"


def test_a_batch_names_what_its_rows_get():
    """The Carlos batch: a zarr_format 2 base, every row OME-Zarr 0.5."""
    base = {"ome_zarr_version": None, "zarr_format": 2}
    table = pd.DataFrame({"input_path": ["a.nd2", "b.nd2", "c.nd2"],
                          "ome_zarr_version": ["0.5", "0.5", None]})
    assert _ome_zarr_targets(base, table) == (
        "Output format: OME-Zarr 0.5 (zarr v3): 2 row(s); OME-Zarr 0.4 (zarr v2): 1 row(s)")


def test_a_batch_without_version_cells_takes_the_base():
    table = pd.DataFrame({"input_path": ["a.tif"]})
    assert _ome_zarr_targets({"ome_zarr_version": "0.6", "zarr_format": 3}, table) == \
        "Output format: OME-Zarr 0.6 (zarr v3): 1 row(s)"
