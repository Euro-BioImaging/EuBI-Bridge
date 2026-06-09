"""
Reader for Zeiss CZI microscopy files.
"""

import os
from typing import Any, Iterable, Optional, Union

import dask.array as da
import numpy as np

from eubi_bridge.core.reader_interface import ImageReader
from eubi_bridge.utils.logging_config import get_logger

logger = get_logger(__name__)


class CZIReader(ImageReader):
    """
    Reader for Zeiss CZI microscopy files.

    Supports single-image, mosaic (tiled), multi-view and multi-illumination
    reading with dimension mapping for non-standard CZI dimensions.
    """

    def __init__(
        self,
        path: str,
        img: Any,
        index_map: dict,
        as_mosaic: bool = False,
        n_views: int = 1,
        n_illuminations: int = 1,
        **kwargs
    ):
        self._path = path
        self.img = img
        self.index_map = index_map
        self.as_mosaic = as_mosaic
        self._n_views = n_views
        self._n_illuminations = n_illuminations
        self.series = 0
        self.tile = 0
        self.view = 0
        self.illumination = 0
        self._set_series_path()

    @property
    def path(self) -> str:
        """Path to the CZI file."""
        return self._path

    @property
    def series_path(self) -> str:
        """Current series identifier."""
        return self._series_path

    @property
    def n_scenes(self) -> int:
        """Number of scenes in the file."""
        try:
            return len(self.img.scenes)
        except (KeyError, Exception):
            return max(1, len(getattr(self.img, '_scenes', None) or [None]))

    @property
    def n_tiles(self) -> int:
        """Number of mosaic tiles in the current scene."""
        if hasattr(self.img.dims, 'M'):
            return self.img.dims.M
        elif hasattr(self.img._dims, 'M'):
            return self.img._dims.M
        else:
            return 1

    @property
    def n_views(self) -> int:
        """Number of views in the file."""
        return self._n_views

    @property
    def n_illuminations(self) -> int:
        """Number of illuminations in the file."""
        return self._n_illuminations

    @property
    def scenes(self):
        """Available scenes."""
        return self.img.scenes

    def _set_series_path(self) -> None:
        """Rebuild the series_path from the current scene/view/illumination indices."""
        parts = [self._path, f'_{self.series}']
        if self._n_views > 1:
            parts.append(f'_view{self.view}')
        if self._n_illuminations > 1:
            parts.append(f'_illu{self.illumination}')
        if self.as_mosaic is False and hasattr(self, 'tile') and self.tile:
            parts.append(f'_tile{self.tile}')
        self._series_path = ''.join(parts)

    def set_scene(self, scene_index: int) -> None:
        """Set the current scene/series."""
        if scene_index < 0 or scene_index >= self.n_scenes:
            raise IndexError(f"Scene index {scene_index} out of range [0, {self.n_scenes})")
        self.index_map['S'] = scene_index
        self.series = scene_index
        self.img.set_scene(scene_index)
        self._set_series_path()

    def set_tile(self, tile_index: int) -> None:
        """Set the current mosaic tile."""
        if tile_index < 0 or tile_index >= self.n_tiles:
            raise IndexError(f"Tile index {tile_index} out of range [0, {self.n_tiles})")
        self.index_map['M'] = tile_index
        self.tile = tile_index
        self._set_series_path()

    def set_view(self, view_index: int) -> None:
        """Set the active view (V dimension)."""
        if view_index < 0 or view_index >= self._n_views:
            raise IndexError(f"View index {view_index} out of range [0, {self._n_views})")
        self.index_map['V'] = view_index
        self.view = view_index
        self._set_series_path()

    def set_illumination(self, illumination_index: int) -> None:
        """Set the active illumination (I dimension)."""
        if illumination_index < 0 or illumination_index >= self._n_illuminations:
            raise IndexError(
                f"Illumination index {illumination_index} out of range "
                f"[0, {self._n_illuminations})"
            )
        self.index_map['I'] = illumination_index
        self.illumination = illumination_index
        self._set_series_path()
    
    def get_image_dask_data(self, **kwargs) -> da.Array:
        """Get image data as dask array with dimension order TCZYX."""
        try:
            return self.img.get_image_dask_data(
                dimension_order_out='TCZYX',
                **self.index_map
            )
        except Exception as e:
            raise RuntimeError(f"Failed to read image data from {self._path}: {str(e)}") from e


def read_czi(
    input_path: str,
    as_mosaic: bool = False,
    view_index: Union[int, str] = 0,
    phase_index: int = 0,
    illumination_index: Union[int, str] = 0,
    scene_index: Union[int, Iterable[int]] = 0,
    rotation_index: int = 0,
    mosaic_tile_index: int = 0,
    sample_index: int = 0,
    **kwargs
) -> 'CZIReader':
    """
    Read a CZI (Zeiss microscopy) file with specified dimension indices.

    Parameters
    ----------
    input_path : str
        Path to the CZI file.
    as_mosaic : bool, default False
        Whether to read as a mosaic (tiled) image.
    view_index : int or 'all', default 0
        View(s) to expose.  Pass ``'all'`` or a comma-separated string
        (e.g. ``'0,2'``) to expose multiple views; each will produce a
        separate output (or be concatenated when ``concat_views=True``).
    phase_index : int, default 0
        Index for the phase dimension (H).
    illumination_index : int or 'all', default 0
        Illumination(s) to expose.  Same multi-value rules as ``view_index``.
    scene_index : int, default 0
        Index for the scene dimension (S).
    rotation_index : int, default 0
        Index for the rotation dimension (R).
    mosaic_tile_index : int, default 0
        Index for the mosaic tile dimension (M).
    sample_index : int, default 0
        Index for the sample dimension (A).
    **kwargs
        Additional keyword arguments passed through to ``CZIReader``.

    Returns
    -------
    CZIReader
        A reader instance implementing the ImageReader interface.
        ``n_views`` and ``n_illuminations`` are set to the number of
        *exposed* views / illuminations so the caller can iterate over them.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    # Import the appropriate reader
    if as_mosaic:
        from bioio_czi.pylibczirw_reader.reader import Reader
    else:
        from bioio_czi.aicspylibczi_reader.reader import Reader

    # Initialize reader and get image metadata
    try:
        img = Reader(input_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CZI file: {str(e)}") from e

    # bioio_czi's pylibczirw reader crashes when a CZI scene XML element has
    # no 'Name' attribute (KeyError: 'Name').  Pre-populate _scenes with
    # fallback string indices so the buggy property is never triggered.
    if hasattr(img, '_scenes') and img._scenes is None:
        try:
            img.scenes  # attempt normal population
        except (KeyError, Exception):
            try:
                import pylibCZIrw.czi as pyczi
                with pyczi.open_czi(input_path) as f:
                    scene_ids = list(f.scenes_bounding_rectangle.keys())
            except Exception:
                scene_ids = [0]
            img._scenes = tuple(str(i) for i in scene_ids) or ("0",)

    # Discover all non-standard dimensions present in the file.
    nonstandard_dims = [
        dim.upper() for dim in img.standard_metadata.dimensions_present
        if dim.upper() not in {"X", "Y", "C", "T", "Z"}
    ]

    # Handle mosaic-specific logic
    if as_mosaic:
        # The pylibczirw reader stitches mosaic tiles transparently into the
        # spatial (Y/X) dimensions, so 'M' never appears in dimensions_present.
        # Remove it from nonstandard_dims only if it happens to be listed.
        nonstandard_dims = [d for d in nonstandard_dims if d != 'M']
        if mosaic_tile_index != 0:
            logger.warning(
                "Mosaic tile index is ignored when reading the entire mosaic. "
                "Set as_mosaic=False to read specific tiles."
            )

    # ── Resolve how many V / I values are available ──────────────────────
    def _dim_size(dim: str) -> int:
        """Return the size of a CZI dimension, or 1 if absent."""
        dims = img.dims
        return getattr(dims, dim, None) or 1

    total_views        = _dim_size('V') if 'V' in nonstandard_dims else 1
    total_illuminations = _dim_size('I') if 'I' in nonstandard_dims else 1

    # Resolve the *initial* (first) index and the *count* exposed by the reader.
    def _resolve_index(value: Union[int, str], total: int, dim_name: str):
        """Return (first_index, n_exposed)."""
        if value == 'all':
            return 0, total
        if isinstance(value, str) and ',' in value:
            # comma-separated list — we only set the first index here;
            # caller is responsible for iterating via set_view/set_illumination
            parts = [int(x.strip()) for x in value.split(',')]
            return parts[0], len(parts)
        idx = int(value)
        return idx, 1

    view_start, n_views_exposed          = _resolve_index(view_index,         total_views,         'view_index')
    illu_start, n_illuminations_exposed  = _resolve_index(illumination_index, total_illuminations, 'illumination_index')

    # ── Build the index_map (locks non-V/I non-standard dims to a fixed index) ──
    czi_dim_map = {
        'V': view_start,
        'H': phase_index,
        'I': illu_start,
        'S': scene_index,
        'R': rotation_index,
        'M': mosaic_tile_index,
        'A': sample_index,
    }

    index_map = {
        dim: czi_dim_map[dim]
        for dim in nonstandard_dims
        if dim in czi_dim_map
    }

    return CZIReader(
        input_path, img, index_map,
        as_mosaic=as_mosaic,
        n_views=n_views_exposed,
        n_illuminations=n_illuminations_exposed,
        **kwargs,
    )

