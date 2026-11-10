"""keep_folder_structure (Workplan E28, decision E-7): the folders given as inputs are
mirrored under the output folder, each output where its file was; files given one by one go
into the output folder itself; joined files go to the deepest folder they share. Off by
default (outputs flat, as before)."""
import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

from eubi_bridge.ebridge import EuBIBridge


def _tif(path, value=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.full((16, 16), value, "u2"))


def _tree(root):
    """root/A/x.tif, root/A/sub/y.tif, root/B/x.tif (two files named x)."""
    _tif(root / "A" / "x.tif", 1)
    _tif(root / "A" / "sub" / "y.tif", 2)
    _tif(root / "B" / "x.tif", 3)
    return root


def _outputs(out):
    return sorted(p.relative_to(out).as_posix() for p in Path(out).rglob("*.ome.zarr*")
                  if p.name.endswith((".ome.zarr", ".ome.zarr.zip")))


def _convert(src, out, **kwargs):
    EuBIBridge().to_zarr(src if isinstance(src, list) else str(src), str(out),
                         use_threading=True, **kwargs)


def test_a_folder_is_mirrored(tmp_path):
    src = _tree(tmp_path / "data")
    _convert(src, tmp_path / "out", keep_folder_structure=True)
    assert _outputs(tmp_path / "out") == ["data/A/sub/y.ome.zarr", "data/A/x.ome.zarr",
                                          "data/B/x.ome.zarr"]


def test_off_by_default_outputs_stay_flat(tmp_path):
    src = _tree(tmp_path / "data")
    _convert(src, tmp_path / "out")
    assert _outputs(tmp_path / "out") == ["A_x.ome.zarr", "B_x.ome.zarr", "y.ome.zarr"]


def test_checked_folders_mirror_and_files_picked_by_hand_stay_flat(tmp_path):
    """The GUI's selection: a checked folder and a file picked by hand."""
    src = _tree(tmp_path / "data")
    _tif(tmp_path / "loose" / "z.tif", 4)
    _convert([str(src / "A"), str(tmp_path / "loose" / "z.tif")], tmp_path / "out",
             keep_folder_structure=True)
    assert _outputs(tmp_path / "out") == ["A/sub/y.ome.zarr", "A/x.ome.zarr", "z.ome.zarr"]


def test_a_folder_in_a_list_without_the_option_gives_its_files_flat(tmp_path):
    src = _tree(tmp_path / "data")
    _convert([str(src / "A"), str(src / "B")], tmp_path / "out")
    assert _outputs(tmp_path / "out") == ["A_x.ome.zarr", "B_x.ome.zarr", "y.ome.zarr"]


def test_zipped_outputs_are_mirrored_too(tmp_path):
    if not hasattr(__import__("dyna_zarr").io, "open_zip"):
        pytest.skip("dyna-zarr without zip writing")
    src = _tree(tmp_path / "data")
    _convert(src, tmp_path / "out", keep_folder_structure=True, zip_output="zip")
    assert _outputs(tmp_path / "out") == ["data/A/sub/y.ome.zarr.zip", "data/A/x.ome.zarr.zip",
                                          "data/B/x.ome.zarr.zip"]


def test_joined_files_go_to_the_folder_they_share(tmp_path):
    root = tmp_path / "data"
    for folder in ("A", "B"):
        for t in (1, 2, 3):
            _tif(root / folder / "series" / f"img_T{t}.tif", t)
    _convert(root, tmp_path / "out", keep_folder_structure=True, time_tag="T",
             concatenation_axes="t")
    assert _outputs(tmp_path / "out") == ["data/A/series/img_T_tset.ome.zarr",
                                          "data/B/series/img_T_tset.ome.zarr"]
