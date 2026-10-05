"""A minimal Imaris (.ims) writer for tests: an HDF5 file in Imaris's layout,
each resolution level stored in padded chunks with its real size in the
ImageSize attributes.  Copied from micro-reader's tests (test_ims.py,
test_resolutions.py), which eubi's CI does not have.
"""
from __future__ import annotations

import numpy as np
import pytest


def _char(value):
    return np.array(list(str(value)), dtype="S1")


def write_pyramid_ims(path, data):
    """Level 0 from *data* (T, C, Z, Y, X), then level 1 at half size in Y / X,
    both padded to multiples of 8."""
    h5py = pytest.importorskip("h5py")
    n_t, n_c, nz, ny, nx = data.shape
    with h5py.File(path, "w") as f:
        for r, d in enumerate((data, data[:, :, :, ::2, ::2])):
            real = d.shape[2:]
            for t in range(n_t):
                for c in range(n_c):
                    grp = f.require_group(f"DataSet/ResolutionLevel {r}/TimePoint {t}/Channel {c}")
                    padded = tuple(-(-n // 8) * 8 for n in real)
                    ds = grp.create_dataset("Data", shape=padded, dtype=d.dtype,
                                            chunks=(4, 8, 8), fillvalue=0)
                    ds[:real[0], :real[1], :real[2]] = d[t, c]
                    for axis, size in zip("ZYX", real):
                        grp.attrs[f"ImageSize{axis}"] = _char(size)
        img = f.require_group("DataSetInfo/Image")
        for axis, size in zip("ZYX", data.shape[2:]):
            img.attrs[axis] = _char(size)
