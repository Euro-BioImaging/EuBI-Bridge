"""A plate stored as one file per well converts once (eubi-bridge 0.3.3).

A NIS-Elements well-plate run writes one ND2 per well (``WellB05_ChannelDIA_Seq0000.nd2``).
micro-reader 0.3.0 opens the whole plate from any of them, and groups them as one set only
when asked for plates (``group_files(..., plates=True)``). eubi asked without it, so each
file was an input of its own and each wrote the whole plate: 24 stores of one plate.
micro-reader 0.2.x takes no ``plates=`` and is asked without it.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

micro_reader = pytest.importorskip("micro_reader")

from eubi_bridge.utils.path_utils import input_mains

DATA = Path(os.environ.get("MICROSCOPY_DATA", os.path.expanduser("~/Desktop/data/microscopy")))
RUN = DATA / "plates" / "nd2" / "bia-S-BIAD3591" / "4x_overview_20250306_171942_094"


def test_a_well_plate_run_is_one_input():
    if not RUN.exists():
        pytest.skip(f"sample not here: {RUN}")
    try:
        micro_reader.group_files([], plates=True)
    except TypeError:
        pytest.skip("this micro-reader does not group plate runs (before 0.3.0)")
    files = sorted(str(p) for p in RUN.glob("*.nd2"))
    assert len(files) > 1
    assert len(set(input_mains(files))) == 1


@pytest.mark.parametrize("off", [False, "off", "False"])
def test_with_as_plate_off_each_file_of_a_run_is_converted(off):
    """as_plate off writes each image of its own: the run's files must stay separate
    inputs (grouped as one plate, only the first file was converted; reviewer finding
    2026-10-28). A table with rows on and off groups each set its own way."""
    import pandas as pd
    from eubi_bridge.utils.path_utils import group_input_rows
    if not RUN.exists():
        pytest.skip(f"sample not here: {RUN}")
    try:
        micro_reader.group_files([], plates=True)
    except TypeError:
        pytest.skip("this micro-reader does not group plate runs (before 0.3.0)")
    files = sorted(str(p) for p in RUN.glob("*.nd2"))
    rows = group_input_rows(pd.DataFrame({"input_path": files, "as_plate": [off] * len(files)}))
    assert sorted(rows["input_path"]) == files
    mixed = group_input_rows(pd.DataFrame({"input_path": files,
                                           "as_plate": [None] * (len(files) - 1) + [off]}))
    assert len(mixed) <= 2 and files[-1] in set(mixed["input_path"])


def test_a_micro_reader_without_plates_is_asked_without_it(monkeypatch, tmp_path):
    asked = []
    real = micro_reader.group_files

    def old_group_files(paths, **kwargs):
        if kwargs:
            raise TypeError(f"group_files() got an unexpected keyword argument {next(iter(kwargs))!r}")
        asked.append(paths)
        return real(paths)

    monkeypatch.setattr(micro_reader, "group_files", old_group_files)
    image = tmp_path / "a.tif"
    import numpy as np
    import tifffile
    tifffile.imwrite(image, np.zeros((4, 4), np.uint8))
    assert input_mains([str(image)]) == [str(image)]
    assert asked


def test_a_folder_row_stands_for_its_one_image_or_plate():
    """A GUI batch row is any entry ticked in the browser, a folder too. A folder
    micro-reader cannot open itself but whose files form one image or plate (a
    NIS-Elements run, an NDTiff dataset with levels) converts that image; a folder it
    opens (an NDTiff dataset folder) stays the input, named as before."""
    if not RUN.exists():
        pytest.skip(f"sample not here: {RUN}")
    try:
        micro_reader.group_files([], plates=True)
    except TypeError:
        pytest.skip("this micro-reader does not group plate runs (before 0.3.0)")
    (main,) = input_mains([str(RUN)])
    assert Path(main).parent == RUN and main.endswith(".nd2")
    ndtiff = DATA / "formats" / "tiff" / "ndtiff" / "github-NDTiffStorage" / "v3"
    if ndtiff.exists():
        (levels,) = input_mains([str(ndtiff / "Magellan_expolore_multi_channel")])
        assert Path(levels).parent.name == "Full resolution"
        assert input_mains([str(ndtiff / "ndtiff3.2_multichannel")]) == [
            str(ndtiff / "ndtiff3.2_multichannel")]


def test_a_folder_of_files_chosen_already_is_skipped():
    """'Select All' in a folder ticks its subfolders too: a subfolder whose files belong
    to the images of the other rows is no input of its own."""
    if not RUN.exists():
        pytest.skip(f"sample not here: {RUN}")
    try:
        micro_reader.group_files([], plates=True)
    except TypeError:
        pytest.skip("this micro-reader does not group plate runs (before 0.3.0)")
    files = sorted(str(p) for p in RUN.glob("*.nd2"))
    mains = input_mains([str(RUN)] + files)
    assert mains[0] is None and len(set(mains[1:])) == 1