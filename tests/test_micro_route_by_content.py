"""Every file micro-reader opens is read by it on the default route, whatever its suffix.

Until 0.4.2 the route was a list of suffixes (pyramid_path.FORMATS): a file micro-reader
reads by its content went to the older readers under any other suffix -- a Luxendo
experiment folder (no suffix), a lone MetaMorph .stk, .tf2 / .tf8 BigTIFFs, a JPEG named
.jfif. Only FORMATS_SINCE (a format that needs a newer micro-reader) keeps a file off now.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("micro_reader")
pytest.importorskip("ome_zarr_pyramid")
tifffile = pytest.importorskip("tifffile")
zarr = pytest.importorskip("zarr")

from eubi_bridge.conversion import pyramid_path  # noqa: E402
from eubi_bridge.conversion.pyramid_path import NotYet, _open_micro  # noqa: E402
from eubi_bridge.core.config_models import ConversionJob  # noqa: E402

DATA = Path(os.environ.get("MICROSCOPY_DATA") or os.environ.get("EUBI_TEST_DATA")
            or r"C:\Users\oezdemir\Desktop\data\microscopy")
LUXENDO = DATA / "formats" / "h5" / "luxendo" / "bia-S-BIAD3686-subset"
#: six pages, a z stack (3-D: a 4-D array tifffile writes as QQYX has two unnamed axes,
#: which need one letter each from the user -- Workplan MR36, micro-reader 0.5.1; with
#: eubi's single 'z' it converts as one output per plane, whatever the suffix)
PIXELS = np.arange(6 * 32 * 48, dtype=np.uint16).reshape(6, 32, 48)


def _job(path, **readers):
    job = ConversionJob.from_kwargs(input_path=str(path), output_path="o", kwargs={})
    if readers:
        job = job.model_copy(update={"readers": job.readers.model_copy(update=readers)})
    return job


def _tiff(path: Path) -> Path:
    tifffile.imwrite(path, PIXELS, photometric="minisblack",
                     bigtiff=path.name.lower().endswith("tf8"))
    return path


@pytest.mark.parametrize("name", ["a.tf8", "a.ome.tf8", "a.tf2", "a.stk", "a.STK", "scan.raw_tif"])
def test_a_tiff_under_a_suffix_the_list_never_named_is_read_by_micro_reader(tmp_path, name):
    path = _tiff(tmp_path / name)
    assert not pyramid_path._listed(path)
    job = _job(path)
    f = _open_micro(job)
    try:
        assert len(f.indices) == 1
        assert str(job.input_path) == str(path)          # the file itself, not a main file
    finally:
        f.close()


def test_a_jpeg_named_jfif_is_read_by_micro_reader(tmp_path):
    image = pytest.importorskip("PIL.Image")
    path = tmp_path / "photo.jfif"
    image.fromarray(np.full((16, 24, 3), 90, np.uint8)).save(path, format="JPEG")
    f = _open_micro(_job(path))
    f.close()


def test_a_file_micro_reader_does_not_recognise_goes_to_the_older_readers(tmp_path):
    path = tmp_path / "a.zvi"
    path.write_bytes(b"\xd0\xcf\x11\xe0" + bytes(4096))      # an OLE header: not micro's
    with pytest.raises(NotYet, match="micro-reader does not read it"):
        _open_micro(_job(path))


def test_the_standard_reader_asked_for_still_goes_to_the_older_readers(tmp_path):
    path = _tiff(tmp_path / "a.tf8")
    with pytest.raises(NotYet, match="the standard reader was asked for"):
        _open_micro(_job(path, pixel_reader="standard"))


def test_a_version_gate_still_keeps_a_format_off(tmp_path, monkeypatch):
    """FORMATS_SINCE: a .klb needs micro-reader 0.4.0 on the route."""
    from eubi_bridge.core import micro_source
    monkeypatch.setattr(micro_source, "micro_reader_version", lambda: (0, 3, 9))
    path = tmp_path / "a.klb"
    path.write_bytes(bytes(64))
    with pytest.raises(NotYet, match=r"\.klb files need micro-reader >=0\.4\.0"):
        _open_micro(_job(path))


def _eubi() -> str:
    found = shutil.which("eubi")
    if found:
        return found
    for name in ("eubi.exe", "eubi"):
        for folder in (Path(sys.executable).parent, Path(sys.executable).parent / "Scripts"):
            if (folder / name).exists():
                return str(folder / name)
    pytest.skip("no eubi executable next to this Python")


def test_a_tf8_converts_through_micro_reader_with_its_pixels(tmp_path):
    src = _tiff(tmp_path / "stack.tf8")
    out = tmp_path / "out"
    env = dict(os.environ, COLUMNS="250")
    result = subprocess.run([_eubi(), "to_zarr", str(src), str(out)],
                            capture_output=True, text=True, env=env)
    log = " ".join((result.stdout + result.stderr).split())
    assert result.returncode == 0, log[-3000:]
    assert "read by micro-reader" in log, log[-3000:]
    stores = sorted(out.rglob("*.zarr"))
    assert len(stores) == 1
    written = np.asarray(zarr.open_group(str(stores[0]), mode="r")["0"])
    assert written.shape == PIXELS.shape                     # the pages as z (squeezed)
    assert np.array_equal(written, PIXELS)


@pytest.mark.skipif(not LUXENDO.is_dir(), reason="Luxendo sample not present")
@pytest.mark.parametrize("given", ["folder", "json part"])
def test_a_luxendo_experiment_folder_is_read_by_micro_reader(given):
    """micro-reader's main file of a Luxendo experiment is the folder itself; until 0.4.2
    it went to the older readers ('format not on micro-reader's route yet')."""
    path = LUXENDO if given == "folder" else Path(glob.glob(
        str(LUXENDO / "raw" / "*" / "Cam_long_00000.json"))[0])
    job = _job(path)
    f = _open_micro(job)          # 89 of its 90 stacks are not held here: a logged warning
    try:
        assert os.path.normcase(str(job.input_path)) == os.path.normcase(str(LUXENDO))
        assert len(f.indices) >= 1
    finally:
        f.close()
