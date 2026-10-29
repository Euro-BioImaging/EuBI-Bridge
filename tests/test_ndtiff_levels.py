"""NDTiff datasets with downsampled levels (eubi-bridge 0.3.3).

Micro-Manager Magellan explorations and stitched acquisitions write the image in
``Full resolution/`` and previews of it in ``Downsampled_x<N>/`` beside it, each level a
dataset of its own (its own NDTiff.index). Until 0.3.2 a folder of such a dataset gave
one output per level; now every file of a level stands for the full resolution: one
output per dataset, whichever of its folders or files is chosen.

Real samples from the local microscopy data folder (MICROSCOPY_DATA); a test skips when
its sample is not there.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

micro_reader = pytest.importorskip("micro_reader")

from eubi_bridge.utils.path_utils import group_inputs, ndtiff_full_resolution

DATA = Path(os.environ.get("MICROSCOPY_DATA", os.path.expanduser("~/Desktop/data/microscopy")))
NDTIFF = DATA / "formats" / "tiff" / "ndtiff" / "github-NDTiffStorage"
EXPLORE = NDTIFF / "v3" / "ndtiff3.2_magellan_explore_rgb"         # 5 levels, RGB, 3 tiles
STITCHED = NDTIFF / "v2" / "ndtiffv2.0_stitched_test"              # NDTiff v2, 4 levels
FLAT = NDTIFF / "v3" / "ndtiff3.2_multichannel"                    # no levels


def _needs(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"sample not here: {path}")
    return path


def _full(dataset: Path) -> Path:
    (main,) = [g.main for g in micro_reader.group_files(str(dataset / "Full resolution")).groups]
    return Path(main)


def _files(folder: Path) -> list:
    return [str(p) for p in folder.rglob("*") if p.is_file()]


@pytest.mark.parametrize("dataset", [EXPLORE, STITCHED], ids=["v3 explore", "v2 stitched"])
def test_a_dataset_folder_converts_its_full_resolution_once(dataset):
    full = _full(_needs(dataset))
    assert len(micro_reader.group_files(str(dataset)).groups) > 1       # one per level
    assert [Path(p) for p in group_inputs(_files(dataset))] == [full]


def test_a_level_chosen_alone_converts_the_full_resolution():
    full = _full(_needs(EXPLORE))
    level = EXPLORE / "Downsampled_x4"
    assert [Path(p) for p in group_inputs(_files(level))] == [full]
    assert [Path(p) for p in group_inputs([str(level / "NDTiff.index")])] == [full]


def test_the_full_resolution_stands_for_itself():
    full = _full(_needs(EXPLORE))
    assert ndtiff_full_resolution(str(full)) is None
    assert [Path(p) for p in group_inputs(_files(full.parent))] == [full]


def test_a_dataset_without_levels_is_unchanged():
    folder = _needs(FLAT)
    (main,) = [g.main for g in micro_reader.group_files(str(folder)).groups]
    assert [Path(p) for p in group_inputs(_files(folder))] == [Path(main)]


def test_a_level_without_its_full_resolution_converts_as_it_is(tmp_path):
    """Only the file's folder name is no reason: the full resolution must be there."""
    level = tmp_path / "acq" / "Downsampled_x2"
    level.mkdir(parents=True)
    (level / "NDTiff.index").write_bytes(b"")
    assert ndtiff_full_resolution(str(level / "acq_NDTiffStack.tif")) is None


def test_other_folders_are_not_levels(tmp_path):
    (tmp_path / "Full resolution").mkdir()
    (tmp_path / "Full resolution" / "NDTiff.index").write_bytes(b"")
    for name in ("Downsampled", "downsampled_x2_old", "x2"):
        assert ndtiff_full_resolution(str(tmp_path / name / "a.tif")) is None, name
