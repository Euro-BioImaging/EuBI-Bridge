"""Plate files as NGFF plates (eubi-bridge 0.3.2, as_plate=True by default).

A file micro-reader reads as a plate (Harmony, CellVoyager, MetaXpress, IN Cell, a CZI
holding a plate) is written as ONE NGFF plate store in the OME-Zarr version asked for
(0.4, 0.5, 0.6, RFC 8): every field with the job's settings, each keeping its coordinate
system. A plate whose fields hold several tiles needs as_mosaic in 0.4 / 0.5 / 0.6 (an
NGFF plate holds one image per field); RFC 8 keeps the tiles. as_plate=False writes one
store per image, with a warning. Images outside the plate (a CZI's slide preview) are not
written, with a warning.

Real samples from the local microscopy data folder (MICROSCOPY_DATA); a test skips when
its sample is not there.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

micro_reader = pytest.importorskip("micro_reader")
ozp = pytest.importorskip("ome_zarr_pyramid")

DATA = Path(os.environ.get("MICROSCOPY_DATA", os.path.expanduser("~/Desktop/data/microscopy")))
CV7000 = DATA / "plates" / "cv7000" / "bia-S-BIAD1013-cv7000" / "1093684368.wpi"
HARMONY = (DATA / "plates" / "harmony" / "bia-S-BIAD1013-operetta" / "HMEC_21Hits_Images"
           / "Index.idx.xml")
HS_MEASUREMENT = (DATA / "plates" / "harmony" / "zenodo-20799564"
                  / "Operetta_minimal_dataset_DPC-Brightfield" / "hs"
                  / "3a1d3fc3-c7bc-4a7f-916b-1e07adf426f4")
PHENIX_FULL = DATA / "plates" / "harmony" / "zenodo-7351200-full"
CZI_PLATE = DATA / "formats" / "czi" / "challenge" / "w96_A1+A2.czi"
CZI_TILED = DATA / "formats" / "czi" / "zenodo-7015307" / "W96_B2+B4_S=2_T=1=Z=1_C=1_Tile=5x9.czi"


def _needs(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"sample not here: {path}")
    with micro_reader.open(str(path), plates=True) as f:
        if not getattr(f, "plates", None):
            pytest.skip("this micro-reader does not read the sample as a plate")
    return path


def _eubi() -> str:
    found = shutil.which("eubi")
    if found:
        return found
    for name in ("eubi.exe", "eubi"):
        for folder in (Path(sys.executable).parent, Path(sys.executable).parent / "Scripts"):
            if (folder / name).exists():
                return str(folder / name)
    pytest.skip("no eubi executable next to this Python")


def _run(src: Path, out: Path, args: list):
    result = subprocess.run([_eubi(), "to_zarr", str(src), str(out), *args],
                            capture_output=True, text=True, env=dict(os.environ, COLUMNS="250"))
    return result.returncode, " ".join((result.stdout + result.stderr).split())


def _plate(store: Path):
    coll = ozp.PyramidCollection.from_zarr(str(store))
    assert len(coll.plates) == 1
    return coll, coll.plates[0]


@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6", "rfc8"])
def test_a_plate_file_is_one_ngff_plate(tmp_path, version):
    src = _needs(CV7000)
    code, log = _run(src, tmp_path / "out", ["--ome_zarr_version", version])
    assert code == 0, log[-3000:]
    assert "as a plate" in log
    (store,) = (tmp_path / "out").glob("*.zarr")
    coll, plate = _plate(store)
    assert [w.name for w in plate.wells] == ["B2"] and len(plate.wells[0].fields) == 4
    assert set(coll.system_of.values()) == {"well-B2"}       # each field kept its system
    if version != "rfc8":
        pytest.importorskip("jsonschema")
        assert ozp.validate(str(store)) == []


def test_a_plate_from_a_fixed_index_name_is_named_after_its_folder(tmp_path):
    src = _needs(HARMONY)
    code, log = _run(src, tmp_path / "out", [])
    assert code == 0, log[-3000:]
    assert [p.name for p in (tmp_path / "out").glob("*.zarr")] == ["HMEC_21Hits_Images.zarr"]


def test_a_side_file_of_a_plate_converts_the_plate_once(tmp_path):
    """A Harmony measurement's side files (images/image.index.txt, the measurement's
    .kw.txt) are parts of its index: choosing one, or one beside the index, converts the
    plate once, named after the measurement folder."""
    index = _needs(HS_MEASUREMENT / "images" / "index.xml")
    side = HS_MEASUREMENT / "images" / "image.index.txt"
    from eubi_bridge.utils.path_utils import group_inputs
    assert [Path(p) for p in group_inputs([str(side)])] == [index]
    root = HS_MEASUREMENT.parent                      # the hs archive's own files
    for name in ("_objmeta.txt", "_hs.txt", "_structure.txt"):
        assert [Path(p) for p in group_inputs([str(root / name)])] == [index], name
    every = [str(p) for p in HS_MEASUREMENT.rglob("*") if p.is_file()]
    assert [Path(p) for p in group_inputs(every)] == [index]
    from eubi_bridge.ebridge import EuBIBridge
    out = tmp_path / "out"
    EuBIBridge().to_zarr([str(side), str(index)], str(out))
    assert [p.name for p in out.glob("*.zarr")] == [f"{HS_MEASUREMENT.name}.zarr"]
    _plate(out / f"{HS_MEASUREMENT.name}.zarr")


def test_every_file_of_a_plate_folder_stands_for_the_plate():
    """An Opera Phenix export holds Evaluation1/ (Harmony's analysis: PlateResults.xml,
    ObjectList_*.xml) beside Images/, Assaylayout/ and FFC_Profile/. micro-reader 0.2.0
    names no part of it; eubi took each XML as an input of its own (50 failed tasks)."""
    index = _needs(PHENIX_FULL / "Images" / "Index.xml")
    from eubi_bridge.utils.path_utils import input_mains
    files = [str(p) for p in PHENIX_FULL.rglob("*") if p.is_file()]
    assert any("Evaluation1" in f for f in files)
    assert {Path(m) for m in input_mains(files)} == {index}


def test_a_batch_table_converts_each_plate_once(tmp_path):
    """The GUI's batch passes a table, one row per chosen file: rows of one plate become
    one row (the first's settings), rows that stand for nothing are dropped."""
    import pandas as pd
    from eubi_bridge.utils.path_utils import group_input_rows
    index = _needs(HS_MEASUREMENT / "images" / "index.xml")
    rows = pd.DataFrame({
        "input_path": [str(HS_MEASUREMENT / "images" / "image.index.txt"), str(index),
                       str(next(HS_MEASUREMENT.glob("*.kw.txt")))],
        "ome_zarr_version": ["0.5", "0.4", "0.4"]})
    grouped = group_input_rows(rows)
    assert [Path(p) for p in grouped["input_path"]] == [index]
    assert list(grouped["ome_zarr_version"]) == ["0.5"]


def test_a_czi_plate_is_a_plate_and_says_what_it_leaves_out(tmp_path):
    src = _needs(CZI_PLATE)
    code, log = _run(src, tmp_path / "out", ["--ome_zarr_version", "0.5"])
    assert code == 0, log[-3000:]
    assert "not part of a plate and are not written" in log      # the slide preview
    (store,) = (tmp_path / "out").glob("*.zarr")
    _, plate = _plate(store)
    assert len(plate.wells) == 2


def test_as_plate_false_writes_each_image_and_warns(tmp_path):
    src = _needs(CZI_PLATE)
    code, log = _run(src, tmp_path / "out", ["--as_plate", "False", "--series_index", "all"])
    assert code == 0, log[-3000:]
    assert "converted with as_plate=False" in log
    assert len(list((tmp_path / "out").glob("*.zarr"))) == 3


@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6"])
def test_tiled_fields_need_as_mosaic_in_ngff_plates(tmp_path, version):
    src = _needs(CZI_TILED)
    code, log = _run(src, tmp_path / "out", ["--ome_zarr_version", version])
    assert code != 0
    assert "holds one image per field" in log and "as_mosaic=True" in log
    assert not list((tmp_path / "out").glob("*.zarr"))


def test_tiled_fields_stitched_make_a_plate(tmp_path):
    src = _needs(CZI_TILED)
    code, log = _run(src, tmp_path / "out", ["--ome_zarr_version", "0.6", "--as_mosaic", "True"])
    assert code == 0, log[-3000:]
    (store,) = (tmp_path / "out").glob("*.zarr")
    coll, plate = _plate(store)
    assert len(coll) == 2 and len(plate.wells) == 2


def test_rfc8_keeps_the_tiles_of_a_field(tmp_path):
    src = _needs(CZI_TILED)
    code, log = _run(src, tmp_path / "out", ["--ome_zarr_version", "rfc8"])
    assert code == 0, log[-3000:]
    (store,) = (tmp_path / "out").glob("*.zarr")
    coll, plate = _plate(store)
    assert len(coll) == 90 and len(plate.wells) == 2         # 2 fields of 45 tiles
