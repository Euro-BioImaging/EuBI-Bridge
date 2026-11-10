"""zip_output (Workplan E15): each output written straight into a zip - X.ome.zarr.zip
holding X.ome.zarr/ (the BioImage Archive's layout) or X.ozx (NGFF RFC-9) - with the files a
folder output holds; outputs are named X.ome.zarr (decision E-3); and the refusals."""
import zipfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

dyna_io = pytest.importorskip("dyna_zarr").io
if not hasattr(dyna_io, "open_zip"):
    pytest.skip("dyna-zarr without zip writing", allow_module_level=True)
pytest.importorskip("zarr_zip")

from eubi_bridge.ebridge import EuBIBridge                               # noqa: E402

DATA = (np.arange(4 * 64 * 80, dtype="u2").reshape(4, 64, 80) % 997)


def _inputs(folder, names=("cells", "nuclei")):
    folder.mkdir(parents=True, exist_ok=True)
    for k, name in enumerate(names):
        tifffile.imwrite(folder / f"{name}.tif", DATA + k, metadata={"axes": "ZYX"})
    return folder


def _convert(src, out, **kwargs):
    EuBIBridge().to_zarr(str(src), str(out), use_threading=True, **kwargs)


def _folder(root):
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in Path(root).rglob("*") if p.is_file()}


def _zipped(path, prefix):
    with zipfile.ZipFile(path) as zf:
        assert zf.testzip() is None
        names = zf.namelist()
        assert len(names) == len(set(names))
        return {n[len(prefix):]: zf.read(n) for n in names if n.startswith(prefix)}


def test_outputs_are_named_ome_zarr(tmp_path):
    _convert(_inputs(tmp_path / "in"), tmp_path / "out")
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["cells.ome.zarr",
                                                                   "nuclei.ome.zarr"]


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_a_zip_holds_what_the_folder_holds(tmp_path, version):
    src = _inputs(tmp_path / "in")
    _convert(src, tmp_path / "folders", ome_zarr_version=version)
    _convert(src, tmp_path / "zips", ome_zarr_version=version, zip_output="zip")
    assert sorted(p.name for p in (tmp_path / "zips").iterdir()) == [
        "cells.ome.zarr.zip", "nuclei.ome.zarr.zip"]                 # nothing else, no .part
    for name in ("cells", "nuclei"):
        assert _zipped(tmp_path / "zips" / f"{name}.ome.zarr.zip", f"{name}.ome.zarr/") == \
            _folder(tmp_path / "folders" / f"{name}.ome.zarr")


def test_rfc9_ozx(tmp_path):
    from ome_zarr_pyramid import IO
    _convert(_inputs(tmp_path / "in", ("cells",)), tmp_path / "out", ome_zarr_version="0.5",
             zip_output="ozx")
    with zipfile.ZipFile(tmp_path / "out" / "cells.ozx") as zf:
        assert zf.namelist()[0] == "zarr.json"
        assert b'"version": "0.5"' in zf.comment
    back = IO().read_pyramid(tmp_path / "out" / "cells.ozx")
    np.testing.assert_array_equal(np.asarray(back.layers["0"][...]).reshape(DATA.shape), DATA)


def test_ozx_needs_ome_zarr_05(tmp_path):
    with pytest.raises(Exception, match="0.5 or newer"):
        _convert(_inputs(tmp_path / "in", ("cells",)), tmp_path / "out",
                 ome_zarr_version="0.4", zip_output="ozx")
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


def test_object_store_outputs_are_refused(tmp_path):
    with pytest.raises(Exception, match="local disk"):
        _convert(_inputs(tmp_path / "in", ("cells",)), "s3://bucket/out", zip_output="zip")


def test_overwrite_replaces_a_zip_and_a_rerun_skips_the_inputs_beside_it(tmp_path):
    src = _inputs(tmp_path / "in", ("cells",))
    _convert(src, src, zip_output="zip")                     # the outputs beside their inputs
    first = (src / "cells.ome.zarr.zip").read_bytes()
    _convert(src, src, zip_output="zip", overwrite=True)     # the zip is no input of its own
    assert sorted(p.name for p in src.iterdir()) == ["cells.ome.zarr.zip", "cells.tif"]
    assert zipfile.ZipFile(src / "cells.ome.zarr.zip").namelist() == \
        zipfile.ZipFile(__import__("io").BytesIO(first)).namelist()


def test_metadata_updates_refuse_a_zip(tmp_path):
    from eubi_bridge.conversion.metadata_path import update_store
    from eubi_bridge.utils.errors import InputError
    _convert(_inputs(tmp_path / "in", ("cells",)), tmp_path / "out", zip_output="zip")
    with pytest.raises(InputError, match="cannot be changed in place"):
        update_store(str(tmp_path / "out" / "cells.ome.zarr.zip"), {})
