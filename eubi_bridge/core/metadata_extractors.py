"""
Metadata extraction pipeline.

Separates metadata extraction from array reading, allowing independent
strategies for different file formats and metadata sources.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Optional

from eubi_bridge.utils.logging_config import get_logger

logger = get_logger(__name__)


class MetadataExtractor(ABC):
    """
    Abstract base class for metadata extraction strategies.
    
    Each extractor is responsible for reading OME metadata from a specific
    format or source (bioio, bioformats, native libraries, etc.).
    """
    
    @abstractmethod
    async def extract(self, path: str, series: Optional[int] = None) -> Any:
        """
        Extract OME metadata from a file.
        
        Parameters
        ----------
        path : str
            Path to the image file.
        series : int, optional
            Series/scene index. If None, uses series 0.
            
        Returns
        -------
        ome_types.model.OME or dict
            OME metadata object or metadata dictionary.
            
        Raises
        ------
        FileNotFoundError
            If file cannot be found or read.
        RuntimeError
            If metadata extraction fails.
        """
        pass


class BioIOMetadataExtractor(MetadataExtractor):
    """
    Extract metadata using bioio extension-specific readers.
    
    This uses the format-specific bioio packages (bioio-tifffile, bioio-czi, etc.)
    which are faster and more format-native than generic bioformats.
    """
    
    async def extract(self, path: str, series: Optional[int] = None) -> Any:
        """Extract metadata via bioio extension readers."""
        from eubi_bridge.core.readers import get_metadata_reader_by_path
        
        if series is None:
            series = 0
        
        logger.info(f"Extracting metadata via bioio for: {path}")
        
        try:
            Reader = get_metadata_reader_by_path(path)
            img = Reader(path)
            if series is not None:
                img.set_scene(series)
            omemeta = img.ome_metadata
            return omemeta
        except FileNotFoundError as e:
            if ".jgo" in str(e):
                raise RuntimeError("JGO cache may be corrupted. Run `rm -rf ~/.jgo/` and retry.") from e
            raise


class BioFormatsMetadataExtractor(MetadataExtractor):
    """
    Extract metadata using bioio-bioformats (generic fallback).
    
    This is the most compatible option but requires JVM startup and can be slow.
    Used as a fallback when format-specific readers are unavailable.
    """
    
    async def extract(self, path: str, series: Optional[int] = None) -> Any:
        """Extract metadata via bioformats."""
        from bioio_bioformats.reader import Reader
        
        if series is None:
            series = 0
        
        logger.info(f"Extracting metadata via bioformats for: {path}")
        
        try:
            img = Reader(path)
            if series is not None:
                img.set_scene(series)
            omemeta = img.ome_metadata
            return omemeta
        except FileNotFoundError as e:
            if ".jgo" in str(e):
                raise RuntimeError("JGO cache may be corrupted. Run `rm -rf ~/.jgo/` and retry.") from e
            raise


class BFIOMetadataExtractor(MetadataExtractor):
    """
    Extract metadata using bfio library.
    
    An alternative to bioio, provides direct bioformats access.
    """
    
    async def extract(self, path: str, series: Optional[int] = None) -> Any:
        """Extract metadata via bfio."""
        from bfio import BioReader
        
        logger.info(f"Extracting metadata via bfio for: {path}")
        
        try:
            omemeta = BioReader(path, backend='bioformats').metadata
            return omemeta
        except FileNotFoundError as e:
            if ".jgo" in str(e):
                raise RuntimeError("JGO cache may be corrupted. Run `rm -rf ~/.jgo/` and retry.") from e
            raise


class TIFFMetadataExtractor(MetadataExtractor):
    """
    Extract metadata from TIFF files using tifffile library.
    
    Reads TIFF/OME-TIFF metadata directly without bioformats.
    """
    
    async def extract(self, path: str, series: Optional[int] = None) -> Any:
        """Extract metadata from TIFF files."""
        import tifffile
        
        if series is None:
            series = 0
        
        logger.info(f"Extracting metadata from TIFF: {path}")
        
        try:
            tif = tifffile.TiffFile(path)
            # For TIFF, we still need to convert to OME metadata
            # Try to get it from the tiff file first
            if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
                return tif.ome_metadata
            
            # Fallback: use bioio for TIFF to get OME metadata
            from bioio_tifffile.reader import Reader
            img = Reader(path)
            if series is not None:
                img.set_scene(series)
            return img.ome_metadata
        except Exception as e:
            logger.error(f"Failed to extract TIFF metadata: {e}")
            raise RuntimeError(f"Failed to read TIFF metadata: {str(e)}") from e


class H5MetadataExtractor(MetadataExtractor):
    """
    HDF5 metadata extractor (placeholder).
    
    HDF5 files don't have standard OME metadata. The H5ImageMeta class
    handles metadata extraction internally using h5py attributes.
    Returns None since OME metadata extraction is not applicable.
    """
    
    async def extract(self, path: str, series: Optional[int] = None) -> Any:
        """No OME metadata extraction needed for HDF5."""
        logger.debug(f"HDF5 metadata handled internally: {path}")
        return None


class ZarrMetadataExtractor(MetadataExtractor):
    """
    Zarr/NGFF metadata extractor (placeholder).
    
    Zarr/NGFF pyramids don't need traditional OME metadata extraction.
    The pyramid_reader and NGFFImageMeta handle metadata internally.
    Returns None since metadata extraction is not applicable.
    """
    
    async def extract(self, path: str, series: Optional[int] = None) -> Any:
        """No metadata extraction needed for Zarr/NGFF."""
        logger.debug(f"Zarr/NGFF metadata handled internally: {path}")
        return None


class MetadataExtractorFactory:
    """
    Factory for selecting appropriate metadata extractor by file format.
    
    Routes file paths to the best available metadata extraction strategy.
    """
    
    # Map of file extensions to preferred extractors
    _extractor_map = {
        ('.ome.tiff', '.ome.tif'): BioIOMetadataExtractor,
        ('.tif', '.tiff', '.lsm'): TIFFMetadataExtractor,
        ('.h5', '.hdf5'): H5MetadataExtractor,
        ('.czi',): BioIOMetadataExtractor,
        ('.lif',): BioIOMetadataExtractor,
        ('.nd2',): BioIOMetadataExtractor,
        ('.png', '.jpg', '.jpeg'): BioIOMetadataExtractor,
        ('.zarr',): ZarrMetadataExtractor,  # OME-Zarr/NGFF pyramids
    }
    
    @staticmethod
    def get_extractor(path: str) -> MetadataExtractor:
        """
        Select appropriate metadata extractor for a file.
        
        Parameters
        ----------
        path : str
            Path to the image file.
            
        Returns
        -------
        MetadataExtractor
            An appropriate extractor instance for this file type.
            Returns None for formats that don't need metadata extraction (e.g., Zarr).
        """
        path_lower = path.lower()
        
        # Check extension-based routing
        for extensions, extractor_class in MetadataExtractorFactory._extractor_map.items():
            for ext in extensions:
                if path_lower.endswith(ext):
                    if extractor_class is None:
                        logger.debug(f"No metadata extractor needed for {path}")
                        return None
                    logger.debug(f"Using {extractor_class.__name__} for {path}")
                    return extractor_class()
        
        # For unknown files, don't fall back to bioformats to avoid JVM startup
        # Instead, return None to avoid unnecessary metadata extraction
        logger.debug(f"No metadata extractor available for {path} (unknown format)")
        return None


# TODO: Consider deprecating the function below:
# async def extract_metadata(path: str, strategy: str = 'auto', **kwargs) -> Any:
#     """
#     Convenience function to extract metadata with specified strategy.
    
#     Parameters
#     ----------
#     path : str
#         Path to image file.
#     strategy : str, default 'auto'
#         Extraction strategy: 'auto', 'bioio', 'bioformats', 'bfio', 'tiff', 'h5'
#     **kwargs
#         Additional arguments (series, etc.)
        
#     Returns
#     -------
#     Any
#         Extracted metadata.
#     """
#     if strategy == 'auto':
#         extractor = MetadataExtractorFactory.get_extractor(path)
#     elif strategy == 'bioio':
#         extractor = BioIOMetadataExtractor()
#     elif strategy == 'bioformats':
#         extractor = BioFormatsMetadataExtractor()
#     elif strategy == 'bfio':
#         extractor = BFIOMetadataExtractor()
#     elif strategy == 'tiff':
#         extractor = TIFFMetadataExtractor()
#     elif strategy == 'h5':
#         extractor = H5MetadataExtractor()
#     else:
#         raise ValueError(f"Unknown metadata extraction strategy: {strategy}")
    
#     return await extractor.extract(path, **kwargs)
