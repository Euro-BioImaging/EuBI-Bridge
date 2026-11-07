"""What the fourth gap hunt (2026-11-01) found: outputs that were wrong or misleading without a
word, and errors a user should see as one line."""
from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys

import numpy as np
import pytest

from eubi_bridge.conversion import dispatcher, metadata_path, pyramid_path
from eubi_bridge.core.config_models import ConversionJob
from eubi_bridge.utils import path_utils
from eubi_bridge.utils.errors import InputError

micro_reader = pytest.importorskip("micro_reader")
pytest.importorskip("ome_zarr_pyramid")
tifffile = pytest.importorskip("tifffile")


def _convert(src, out, **options) -> list:
    job = ConversionJob.from_kwargs(str(src), str(out), {"dtype": None, **options})
    asyncio.run(pyramid_path.convert(job))
    return sorted(out.glob("*.zarr"))


def test_a_store_does_not_give_away_the_users_folders(tmp_path):
    src = tmp_path / "private_folder" / "cells.tif"
    src.parent.mkdir()
    tifffile.imwrite(src, np.zeros((8, 8), np.uint8))
    (store,) = _convert(src, tmp_path / "out", export_acquisition_metadata=True)
    written = "".join(f.read_text(errors="replace") for f in store.rglob("*")
                      if f.name in (".zattrs", ".zgroup", "zarr.json") or f.suffix == ".xml")
    assert "private_folder" not in written
    assert '"source_path": "cells.tif"' in written


def test_files_no_tag_takes_are_named(caplog):
    with caplog.at_level(logging.WARNING):
        dispatcher._warn_left_out(["/a/one.tif", "/a/two.tif"], 5)
    assert "2 of 5 file(s) do not have every tag" in caplog.text and "one.tif, two.tif" in caplog.text


def test_files_given_of_a_plate_say_the_whole_plate_is_converted(tmp_path, monkeypatch, caplog):
    plane = tmp_path / "r01c01f01.tiff"
    plane.write_bytes(b"")
    from micro_reader import _parts
    monkeypatch.setattr(_parts, "main_file", lambda *a: str(tmp_path / "Index.idx.xml"))
    with caplog.at_level(logging.WARNING):
        found = path_utils._files_that_open_their_set([str(plane)], {}, str, micro_reader)
        path_utils._say_files_that_open_their_set(found)    # said by the caller (cycle 2, R2-08)
    assert len(found) == 1
    assert "the whole set is converted" in caplog.text and "Index.idx.xml" in caplog.text


def test_binary_side_files_are_skipped():
    assert path_utils.SIDE_FILE_SUFFIXES[-7:] == (".bin", ".mat", ".npz", ".npy", ".pkl", ".pickle",
                                                  ".mxprotocol")


def test_a_missing_input_is_one_line_on_the_command_line(tmp_path):
    done = subprocess.run([sys.executable, "-m", "eubi_bridge.cli", "to_zarr",
                           str(tmp_path / "nothing.tif"), str(tmp_path / "out")],
                          capture_output=True, text=True, errors="replace")
    assert done.returncode == 1
    assert "no such file or folder" in done.stderr and "Traceback" not in done.stderr


def test_a_text_file_is_no_table(tmp_path):
    text = tmp_path / "readme.txt"
    text.write_text("Some notes\nabout, this, data, set\nand a, third\n")
    with pytest.raises(InputError, match="neither an image file nor a conversion table"):
        path_utils.take_filepaths(str(text))


def test_input_errors_are_value_errors():
    assert issubclass(InputError, ValueError)


def test_a_plate_store_is_updated_field_by_field(tmp_path):
    def group(path, attrs):
        path.mkdir(parents=True, exist_ok=True)
        (path / ".zattrs").write_text(json.dumps(attrs))

    plate = tmp_path / "p.zarr"
    group(plate, {"plate": {"wells": [{"path": "A/1"}, {"path": "A/2"}]}})
    group(plate / "A" / "1", {"well": {"images": [{"path": "0"}, {"path": "1"}]}})
    group(plate / "A" / "2", {"well": {"images": [{"path": "0"}]}})
    assert metadata_path._is_plate(plate)
    fields = [p.replace("\\", "/").split("p.zarr/")[1] for p in metadata_path._plate_fields(plate)]
    assert fields == ["A/1/0", "A/1/1", "A/2/0"]
