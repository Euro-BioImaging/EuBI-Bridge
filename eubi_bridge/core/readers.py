"""
Image reader dispatcher and metadata extraction utilities.

This module provides:
- Dispatcher function to select appropriate reader based on file format
- Metadata extraction functions using various backends
- Utilities for handling OME metadata
"""

import fsspec
import numpy as np

from dask import delayed
import dask, zarr, asyncio
from eubi_bridge.utils.jvm_manager import soft_start_jvm

from eubi_bridge.ngff.multiscales import Pyramid
from eubi_bridge.utils.logging_config import get_logger
from eubi_bridge.core.reader_interface import ImageReader
from eubi_bridge.core.metadata_extractors import MetadataExtractorFactory

logger = get_logger(__name__)


async def read_single_image(
    input_path: str,
    aszarr: bool = False,
    **kwargs
) -> ImageReader:
    """
    Read a single image file of any supported format.
    
    This is the main dispatcher function that selects the appropriate reader
    based on file extension and returns an ImageReader instance.

    Parameters
    ----------
    input_path : str
        Path to the image file.
    aszarr : bool, default False
        For formats that support it (TIFF, Zarr), whether to use zarr-backed
        reading for efficiency.
    **kwargs : dict
        Additional keyword arguments such as `verbose`, `scene_index`, etc.

    Returns
    -------
    ImageReader
        Reader instance for the appropriate file format.
        
    Raises
    ------
    FileNotFoundError
        If the file doesn't exist.
    RuntimeError
        If the file cannot be opened.
    """
    logger.info(f"Reading image: {input_path}")
    verbose = kwargs.get('verbose', False)
    
    # Route to appropriate reader
    if input_path.endswith('.zarr'):
        from eubi_bridge.core.pyramid_reader import read_pyramid
        logger.debug("Detected NGFF/Zarr format")
        reader = read_pyramid(input_path, aszarr=aszarr, **kwargs)
    elif input_path.endswith('.h5'):
        from eubi_bridge.core.h5_reader import read_h5
        logger.debug("Detected HDF5 format")
        reader = read_h5(input_path, **kwargs)
    elif input_path.endswith(('.ome.tiff', '.ome.tif')) and not aszarr:
        from eubi_bridge.core.pff_reader import read_pff
        logger.debug("Detected OME-TIFF format (using pff reader)")
        reader = read_pff(input_path, **kwargs)
    elif input_path.endswith(('.tif', '.tiff')):
        from eubi_bridge.core.tiff_reader import read_tiff_image
        logger.debug("Detected TIFF format")
        reader = read_tiff_image(input_path, aszarr=aszarr, **kwargs)
    elif input_path.endswith('.lsm'):
        from eubi_bridge.core.tiff_reader import read_tiff_image
        logger.debug("Detected LSM format")
        reader = read_tiff_image(input_path, aszarr=False, **kwargs)
    else:
        # Default: use pff reader (bioio fallback)
        from eubi_bridge.core.pff_reader import read_pff
        logger.debug("Using pff reader (generic bioio fallback)")
        reader = read_pff(input_path, **kwargs)
    
    if verbose:
        logger.info(f"Reader type: {type(reader).__name__}")
        logger.info(f"Using aszarr: {aszarr}")
    
    # Handle scene setting if specified
    scene_index = kwargs.get('scene_index', None)
    if scene_index is not None:
        reader.set_scene(scene_index)
        if verbose:
            logger.info(f"Set scene to: {scene_index}")
    
    return reader


def read_image_sync(path: str, **kwargs) -> ImageReader:
    """
    Synchronous wrapper for read_single_image.
    
    Parameters
    ----------
    path : str
        Path to the image file.
    **kwargs
        Arguments to pass to read_single_image.
        
    Returns
    -------
    ImageReader
        Reader instance.
    """
    return asyncio.run(read_single_image(path, **kwargs))


@delayed
def read_single_image_delayed(path: str, **kwargs) -> ImageReader:
    """
    Lazy/delayed version of read_single_image for dask workflows.
    
    Parameters
    ----------
    path : str
        Path to the image file.
    **kwargs
        Arguments to pass to read_image_sync.
        
    Returns
    -------
    ImageReader
        Reader instance (returned as dask delayed object).
    """
    return read_image_sync(path, **kwargs)
    

def get_metadata_reader_by_path(input_path: str, **kwargs):
    """
    Get the appropriate bioio Reader class for a file path.
    
    Parameters
    ----------
    input_path : str
        Path to the image file.
    **kwargs
        Unused (for compatibility).
        
    Returns
    -------
    type
        A bioio Reader class appropriate for this file format.
    """
    path_lower = input_path.lower()
    
    if path_lower.endswith(('ome.tiff', 'ome.tif')):
        from bioio_ome_tiff.reader import Reader as reader
    elif path_lower.endswith(('.tif', '.tiff', '.lsm')):
        from bioio_tifffile.reader import Reader as reader
    elif path_lower.endswith('.czi'):
        from bioio_czi.reader import Reader as reader
    elif path_lower.endswith('.lif'):
        from bioio_lif.reader import Reader as reader
    elif path_lower.endswith('.nd2'):
        from bioio_nd2.reader import Reader as reader
    elif path_lower.endswith(('.png','.jpg','.jpeg')):
        from bioio_imageio.reader import Reader as reader
    else:
        from bioio_bioformats.reader import Reader as reader
    return reader


async def read_metadata_via_bioio_bioformats(input_path: str, **kwargs):
    """
    Extract OME metadata using bioio-bioformats reader.
    
    Uses the MetadataExtractorFactory to route to the appropriate extractor.
    This provides better separation of concerns and easier strategy swapping.
    
    Parameters
    ----------
    input_path : str
        Path to the image file.
    **kwargs
        Additional arguments (series, etc.).
        
    Returns
    -------
    ome_types.model.OME
        OME metadata object.
    """
    from eubi_bridge.core.metadata_extractors import BioFormatsMetadataExtractor
    
    series = kwargs.get('series', None)
    extractor = BioFormatsMetadataExtractor()
    return await extractor.extract(input_path, series=series)


async def read_metadata_via_bfio(input_path: str, **kwargs):
    """
    Extract OME metadata using bfio library.
    
    Uses the MetadataExtractorFactory to route to the appropriate extractor.
    
    Parameters
    ----------
    input_path : str
        Path to the image file.
    **kwargs
        Additional arguments (unused).
        
    Returns
    -------
    ome_types.model.OME
        OME metadata object.
    """
    from eubi_bridge.core.metadata_extractors import BFIOMetadataExtractor
    
    extractor = BFIOMetadataExtractor()
    return await extractor.extract(input_path)


async def read_metadata_via_extension(input_path: str, **kwargs):
    """
    Extract OME metadata using format-specific bioio readers.
    
    Uses the MetadataExtractorFactory to intelligently select the best
    metadata extraction strategy based on file extension.
    
    Parameters
    ----------
    input_path : str
        Path to the image file.
    **kwargs
        Additional arguments (series, etc.).
        
    Returns
    -------
    ome_types.model.OME
        OME metadata object or None for formats that don't need extraction.
    """
    series = kwargs.get('series', None)
    factory = MetadataExtractorFactory()
    extractor = factory.get_extractor(input_path)
    
    # Handle formats that don't need metadata extraction (e.g., Zarr)
    if extractor is None:
        logger.debug(f"No metadata extraction for {input_path}")
        return None
    
    return await extractor.extract(input_path, series=series)
