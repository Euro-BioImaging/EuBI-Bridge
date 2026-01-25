"""
Reader for formats supported via bioio (Platform-independent File Format reader).

This is the generic fallback reader that handles any format supported by bioio,
including OME-TIFF, CZI, LIF, ND2, PNG, JPG, and anything bioformats can open.
"""

from typing import Any, Optional

import dask
import dask.array as da
import fsspec
import numpy as np
import zarr
from dask import delayed

from eubi_bridge.core.reader_interface import ImageReader
from eubi_bridge.ngff.multiscales import Pyramid
from eubi_bridge.utils.logging_config import get_logger

logger = get_logger(__name__)


readable_formats = ('.ome.tiff', '.ome.tif', '.czi', '.lif',
                    '.nd2', '.tif', '.tiff', '.lsm',
                    '.png', '.jpg', '.jpeg')


class BioIOReader(ImageReader):
    """
    Generic reader for any format supported by bioio.
    
    This is the fallback reader for formats that don't have specialized readers.
    It uses the appropriate bioio extension reader based on file format.
    """
    
    def __init__(self, path: str, img: Any, **kwargs):
        """
        Initialize BioIO reader.
        
        Parameters
        ----------
        path : str
            Path to the image file.
        img : bioio Reader
            Initialized bioio reader object.
        """
        self._path = path
        self.img = img
        self.series = 0
        self._set_series_path()
    
    @property
    def path(self) -> str:
        """Path to the image file."""
        return self._path
    
    @property
    def series_path(self) -> str:
        """Current series identifier."""
        return self._series_path
    
    @property
    def n_scenes(self) -> int:
        """Number of scenes."""
        return len(self.img.scenes)
    
    @property
    def n_tiles(self) -> int:
        """Number of tiles (most formats don't support this)."""
        return 1
    
    def _set_series_path(self) -> None:
        """Update series path."""
        self._series_path = self._path + f'_{self.series}'
    
    def set_scene(self, scene_index: int) -> None:
        """Set the current scene."""
        if scene_index < 0 or scene_index >= self.n_scenes:
            raise IndexError(f"Scene index {scene_index} out of range [0, {self.n_scenes})")
        self.series = scene_index
        self.img.set_scene(self.series)
        self._set_series_path()
    
    def set_tile(self, tile_index: int) -> None:
        """No-op for most formats (no tile support)."""
        if tile_index != 0:
            logger.warning("This format does not support tiles. Ignoring set_tile().")
    
    def get_image_dask_data(self, **kwargs) -> da.Array:
        """Get image data as dask array."""
        try:
            dimensions_to_read = kwargs.get('dimensions_to_read', 'TCZYX')
            return self.img.get_image_dask_data(dimensions_to_read)
        except Exception as e:
            raise RuntimeError(f"Failed to read image data from {self._path}: {str(e)}") from e


def read_pff(
    input_path: str,
    **kwargs
) -> ImageReader:
    """
    Read a file in any format supported by bioio.
    
    This function selects the appropriate bioio reader based on file extension
    and returns a reader instance implementing the ImageReader interface.
    
    Parameters
    ----------
    input_path : str
        Path to the image file.
    **kwargs
        Additional keyword arguments passed to the reader.
        
    Returns
    -------
    ImageReader
        A reader instance implementing the ImageReader interface.
        May be BioIOReader, TIFFZarrReader, CZIReader, or another specialized reader.
        
    Raises
    ------
    FileNotFoundError
        If the file cannot be found.
    RuntimeError
        If the file cannot be opened.
    """
    logger.info(f"Reading file with format detection: {input_path}")
    
    # Route to specialized readers first
    if input_path.endswith(('.ome.tiff', '.ome.tif')):
        from bioio_ome_tiff.reader import Reader as reader
        logger.info("Using bioio-ome-tiff reader")
        img = reader(input_path, **kwargs)
        return BioIOReader(input_path, img, **kwargs)
    
    elif input_path.endswith(('.tif', '.tiff')):
        from eubi_bridge.core.tiff_reader import read_tiff_image
        logger.info("Using native tifffile reader")
        return read_tiff_image(input_path, aszarr=True, **kwargs)
    
    elif input_path.endswith('.czi'):
        from eubi_bridge.core.czi_reader import read_czi
        logger.info("Using CZI-specific reader")
        return read_czi(input_path, **kwargs)
    
    elif input_path.endswith('.lif'):
        from bioio_lif.reader import Reader as reader
        logger.info("Using bioio-lif reader")
        img = reader(input_path, **kwargs)
        return BioIOReader(input_path, img, **kwargs)
    
    elif input_path.endswith('.nd2'):
        from bioio_nd2.reader import Reader as reader
        logger.info("Using bioio-nd2 reader")
        img = reader(input_path, **kwargs)
        return BioIOReader(input_path, img, **kwargs)
    
    elif input_path.endswith(('.png', '.jpg', '.jpeg')):
        from bioio_imageio.reader import Reader as reader
        logger.info("Using bioio-imageio reader")
        img = reader(input_path, **kwargs)
        return BioIOReader(input_path, img, **kwargs)
    
    else:
        # Default fallback: use bioformats
        from bioio_bioformats.reader import Reader as reader
        logger.info("Using bioio-bioformats reader (fallback)")
        img = reader(input_path, **kwargs)
        return BioIOReader(input_path, img, **kwargs)
# pff.get_image_dask_data()