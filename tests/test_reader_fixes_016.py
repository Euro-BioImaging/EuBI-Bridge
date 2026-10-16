"""Reader fixes in eubi-bridge 0.1.6.

- The ozp writer (the default writer) reads resource-backed dask arrays, such as
  bioio-nd2's, one region at a time: their shared handle is not thread-safe, and
  eubi-bridge's own writer has always locked those reads.
- File suffixes are matched regardless of case: an upper-case .OME.TIF, .CZI or .ND2
  went to Bio-Formats instead of its own reader.
- phase_index reaches the CZI reader in a unary conversion (phase 0 was always written).
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import full_installed

if not full_installed():
    pytest.skip("needs the full eubi-bridge (dask / bioio)", allow_module_level=True)

tifffile = pytest.importorskip("tifffile")
zarr = pytest.importorskip("zarr")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def test_the_ozp_writer_reads_resource_backed_arrays_one_region_at_a_time():
    import dask.array as da
    rbda = pytest.importorskip("resource_backed_dask_array")

    from eubi_bridge.core.ozp_writer import _read_safely

    class _Ctx:
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    active, peak, guard = [0], [0], threading.Lock()

    def slow(block):
        with guard:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.02)
        with guard:
            active[0] -= 1
        return block

    data = np.arange(8 * 64 * 64, dtype="u2").reshape(8, 64, 64)
    plain = da.from_array(data, chunks=(1, 32, 32))
    assert _read_safely(plain) is plain                     # other arrays as they are
    backed = rbda.resource_backed_dask_array(plain.map_blocks(slow, dtype=data.dtype), _Ctx())
    np.testing.assert_array_equal(
        _read_safely(backed).compute(scheduler="threads", num_workers=8), data)
    assert peak[0] == 1
    peak[0] = 0
    backed.compute(scheduler="threads", num_workers=8)       # unprotected: in parallel
    assert peak[0] > 1


@pytest.mark.parametrize("name", ["img.ome.tif", "IMG.OME.TIF", "Img.Ome.Tiff"])
def test_suffixes_are_matched_regardless_of_case(tmp_path, name):
    from eubi_bridge.core.data_manager import _is_bioformats_backed
    from eubi_bridge.core.readers import read_single_image
    src = tmp_path / name
    tifffile.imwrite(str(src), np.zeros((2, 16, 20), "u1"), ome=True,
                     photometric="minisblack", metadata={"axes": "CYX"})
    reader = asyncio.run(read_single_image(str(src), pixel_reader="standard"))
    assert type(reader.img).__module__.startswith("bioio_ome_tiff"), type(reader.img).__module__
    assert not _is_bioformats_backed(reader)


def _eubi() -> str:
    for folder in (Path(sys.executable).parent, Path(sys.executable).parent / "Scripts"):
        for exe in ("eubi.exe", "eubi"):
            if (folder / exe).exists():
                return str(folder / exe)
    pytest.skip("no eubi executable next to this Python")


def test_a_czi_phase_is_the_one_asked_for(tmp_path):
    from tests._czi_writer import raw_payload, write_czi
    tiles = [dict(scene=0, plane={"C": 0, "H": h}, x=0, y=0, height=24, width=32, dtype="u1",
                  compression=0, payload=raw_payload(np.full((24, 32), 10 + 100 * h, "u1")))
             for h in range(2)]
    src = tmp_path / "phases.czi"
    write_czi(str(src), tiles)
    values = {}
    for phase in (0, 1):
        out = tmp_path / f"phase{phase}"
        result = subprocess.run([_eubi(), "to_zarr", str(src), str(out),
                                 "--phase_index", str(phase)],
                                capture_output=True, text=True, env=dict(os.environ))
        assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
        (store,) = out.glob("*.zarr")
        values[phase] = np.unique(np.asarray(zarr.open_group(str(store), mode="r")["0"])).tolist()
    assert values == {0: [10], 1: [110]}
