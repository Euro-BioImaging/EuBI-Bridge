"""
Native reader for Imaris (.ims) files.

Imaris files are plain HDF5 with a documented internal resolution-pyramid
layout::

    /DataSet/ResolutionLevel {r}/TimePoint {t}/Channel {c}/Data   # 3D (Z, Y, X)
    /DataSetInfo/Image            .attrs: X, Y, Z, Unit, ExtMin0/1/2, ExtMax0/1/2
    /DataSetInfo/TimeInfo          .attrs: DatasetTimePoints, TimePoint1..N
    /DataSetInfo/Channel {c}        .attrs: Name, Color, ...

By default eubi-bridge reads only ``ResolutionLevel 0`` (full resolution) and
lets its own downscaler rebuild the pyramid. ``get_resolution_level_dask_data``
exposes the other resolution levels for the ``keep_existing_resolutions``
opt-in path.
"""

import h5py
import dask.array as da
import numpy as np

from eubi_bridge.core.reader_interface import ImageReader
from eubi_bridge.utils.logging_config import get_logger

logger = get_logger(__name__)


def _ims_attr_str(attrs, key, default=None):
    """Decode an Imaris HDF5 attribute (array of 1-byte chars) to a string."""
    if key not in attrs:
        return default
    val = attrs[key]
    if isinstance(val, np.ndarray):
        return ''.join(v.decode() if isinstance(v, bytes) else str(v) for v in val)
    return val.decode() if isinstance(val, bytes) else str(val)


class IMSReader(ImageReader):
    """Native reader for Imaris (.ims) HDF5 files.

    Exposes a single scene (ResolutionLevel 0, full resolution) via
    ``get_image_dask_data``. ``get_resolution_level_dask_data`` exposes any
    resolution level for callers that want to preserve the on-disk pyramid.
    """

    def __init__(self, path, h5file, n_timepoints, n_channels, **kwargs):
        self._path = path
        self.h5file = h5file
        self.n_timepoints = n_timepoints
        self.n_channels = n_channels
        self.series = 0
        self._set_series_path()

    @property
    def path(self) -> str:
        return self._path

    @property
    def series_path(self) -> str:
        return self._series_path

    @property
    def n_scenes(self) -> int:
        return 1

    @property
    def n_tiles(self) -> int:
        return 1

    @property
    def n_resolution_levels(self) -> int:
        """Number of ``ResolutionLevel N`` groups under ``/DataSet``."""
        return sum(1 for k in self.h5file['DataSet'].keys()
                   if k.startswith('ResolutionLevel'))

    def _set_series_path(self) -> None:
        self._series_path = f'{self._path}_{self.series}'

    def set_scene(self, scene_index: int) -> None:
        if scene_index != 0:
            raise IndexError(f"Scene index {scene_index} out of range [0, 1)")

    def set_tile(self, tile_index: int) -> None:
        if tile_index != 0:
            logger.warning("Imaris reader does not support tiles. Ignoring set_tile().")

    def get_image_dask_data(self, **kwargs) -> da.Array:
        """Return ResolutionLevel 0 as a T C Z Y X dask array."""
        return self.get_resolution_level_dask_data(0)

    def get_resolution_level_dask_data(self, level: int) -> da.Array:
        """Return the given resolution level as a T C Z Y X dask array."""
        try:
            t_arrays = []
            for t in range(self.n_timepoints):
                c_arrays = []
                for c in range(self.n_channels):
                    ds = self.h5file[
                        f'/DataSet/ResolutionLevel {level}/TimePoint {t}/Channel {c}/Data'
                    ]
                    # Imaris datasets are always HDF5-chunked+compressed; align
                    # dask chunks 1:1 with the native HDF5 chunk grid so reads
                    # don't re-decompress partial chunks. 'auto' is only a
                    # fallback for the (rare) unchunked/contiguous case.
                    c_arrays.append(da.from_array(ds, chunks=ds.chunks or 'auto'))
                t_arrays.append(da.stack(c_arrays, axis=0))
            return da.stack(t_arrays, axis=0)  # -> T C Z Y X
        except Exception as e:
            raise RuntimeError(f"Failed to read image data from {self._path}: {str(e)}") from e


def read_ims(input_path: str, **kwargs) -> IMSReader:
    """Open an Imaris (.ims) file and return an IMSReader.

    Parameters
    ----------
    input_path : str
        Path to the .ims file.
    **kwargs
        Additional keyword arguments (unused, accepted for dispatcher symmetry).

    Returns
    -------
    IMSReader
    """
    f = h5py.File(input_path, 'r')
    res0 = f['DataSet/ResolutionLevel 0']
    n_timepoints = sum(1 for k in res0.keys() if k.startswith('TimePoint'))
    n_channels = sum(1 for k in res0['TimePoint 0'].keys() if k.startswith('Channel'))
    return IMSReader(input_path, f, n_timepoints, n_channels)
