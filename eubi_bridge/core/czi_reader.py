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

_MOSAIC_READER_CLASS = None


def _get_mosaic_reader_class():
    """Return a pylibczirw ``Reader`` subclass that fixes bioio_czi's BGR bug.

    bioio_czi's pylibczirw ``_read_delayed`` appends a Samples axis to
    ``ordered_dims`` for ``Bgr`` pixel types but builds the dask block grid with
    only two trailing singleton axes.  ``da.block`` then merges the 3-sample
    chunk axis into X whenever any non-YX dimension is present, so the data ends
    up one dimension short of ``ordered_dims`` and xarray raises
    "different number of dimensions on data and dims" — i.e. RGB mosaic CZIs
    cannot be read with ``as_mosaic=True``.  This subclass mirrors the upstream
    method but sizes the trailing singleton grid axes to the block rank (3 for
    ``Bgr``, 2 otherwise); behaviour is identical to upstream for non-RGB data.
    """
    global _MOSAIC_READER_CLASS
    if _MOSAIC_READER_CLASS is not None:
        return _MOSAIC_READER_CLASS

    import dask.array as _da
    import numpy as _np
    import xarray as _xr
    from dask import delayed as _delayed
    import bioio_czi.pylibczirw_reader.reader as _prr

    class _MosaicReader(_prr.Reader):
        def _read_delayed(self) -> "_xr.DataArray":
            DN = _prr.DimensionNames
            size = _prr.size
            dim_bounds = self._total_bounding_box
            if len(self._scenes_bounding_rectangle) > 0:
                rect = self._scenes_bounding_rectangle[self._current_scene_index]
                dim_bounds[DN.SpatialX] = (rect.x, rect.x + rect.w)
                dim_bounds[DN.SpatialY] = (rect.y, rect.y + rect.h)
            coords = self._get_coords(
                self.metadata, self._current_scene_index, dim_bounds)
            ordered_dims = [
                d for d in _prr.DEFAULT_DIMENSION_ORDER_LIST
                if d in coords or size(self._total_bounding_box, d) > 1
            ]
            # bioio's pylibczirw reader only understands T/C/Z/Y/X, so it
            # silently collapses the CZI view ('V'), phase ('H'), illumination
            # ('I') and rotation ('R') dimensions to their first index --
            # merging e.g. a multi-illumination mosaic down to a single output.
            # Expose any that are actually present (size > 1) as leading
            # selectable axes; pylibCZIrw.read() honours these dimension chars
            # in its `plane` argument, so the existing index_map machinery can
            # address a specific view / illumination per output even while the
            # mosaic tiles ('M') are stitched into the spatial plane.
            extra_dims = [
                d for d in ('V', 'H', 'I', 'R')
                if d not in ordered_dims and size(self._total_bounding_box, d) > 1
            ]
            ordered_dims = extra_dims + ordered_dims
            assert ordered_dims[-2:] == [DN.SpatialY, DN.SpatialX]
            non_yx_dims = ordered_dims[:-2]
            shape = tuple(
                len(coords[d]) if d in coords
                else size(self._total_bounding_box, d)
                for d in ordered_dims
            )
            shape_without_yx = shape[:-2]
            chunk_shape = shape[-2:]
            is_bgr = "Bgr" in self._pixel_types[0]
            if is_bgr:
                chunk_shape += (3,)
                ordered_dims.append(DN.Samples)
            # Match the trailing singleton grid axes to the chunk rank so
            # da.block keeps the Samples axis separate instead of merging it.
            trailing = (1, 1, 1) if is_bgr else (1, 1)
            lazy_arrays = _np.ndarray(shape_without_yx + trailing, dtype=object)
            for np_index, _ in _np.ndenumerate(lazy_arrays):
                lazy_arrays[np_index] = _da.from_delayed(
                    _delayed(self._array_builder(non_yx_dims))(np_index),
                    chunk_shape,
                    dtype=_prr.PIXEL_DICT[self._pixel_types[0]],
                )
            return _xr.DataArray(
                data=_da.block(lazy_arrays.tolist()),
                dims=ordered_dims,
                coords=coords,
                attrs={_prr.constants.METADATA_UNPROCESSED: self.metadata},
            )

    _MOSAIC_READER_CLASS = _MosaicReader
    return _MOSAIC_READER_CLASS


def _czi_tile_count(input_path: str) -> int:
    """Return the number of mosaic tiles (CZI ``SizeM``) for a file.

    Reads only the CZI metadata segment via pylibCZIrw, so it works for files of
    any size — including ones too large for aicspylibczi to open.  Returns 1 when
    no mosaic dimension is present or detection fails (i.e. treat as single-tile).
    """
    try:
        import re
        import pylibCZIrw.czi as pyczi
        with pyczi.open_czi(input_path) as f:
            md = f.raw_metadata
        m = re.search(r'<SizeM>\s*(\d+)\s*</SizeM>', md)
        return int(m.group(1)) if m else 1
    except Exception as e:
        logger.debug(f"Could not determine SizeM for {input_path} ({e}); "
                     f"assuming single tile.")
        return 1


# libCZI pixel-type name -> OME PixelType (resolved lazily to avoid import cost).
_CZI_PIXELTYPE_NAMES = {
    'Gray8': 'uint8', 'Gray16': 'uint16', 'Gray32': 'uint32',
    'Gray32Float': 'float', 'Gray64Float': 'double',
    'Bgr24': 'uint8', 'Bgr48': 'uint16', 'Bgr96Float': 'float',
}


def build_czi_omemeta(input_path: str) -> 'OME':  # noqa: F821
    """Build OME metadata for a CZI directly from pylibCZIrw (no binary reads).

    pylibCZIrw reads the metadata segment for files of any size, so this sees
    *all* scenes — unlike Bio-Formats, which silently truncates large CZIs at a
    ~4 GB subblock offset (see the large-CZI offset-truncation note).  One
    ``<Image>`` is produced per CZI scene; per-scene X/Y come from the scene
    bounding rectangle, the remaining dimensions and channel/scale metadata are
    global.
    """
    import re
    import pylibCZIrw.czi as pyczi
    from ome_types.model import (OME, Channel, Image, Pixels,
                                 Pixels_DimensionOrder, PixelType, UnitsLength,
                                 UnitsTime)

    with pyczi.open_czi(input_path) as f:
        md = f.raw_metadata
        tbb = f.total_bounding_box_no_pyramid
        rects = f.scenes_bounding_rectangle_no_pyramid

    def _first_int(pattern, default):
        m = re.search(pattern, md)
        return int(m.group(1)) if m else default

    def _span(dim, fallback_tag):
        if dim in tbb:
            lo, hi = tbb[dim]
            return hi - lo
        return _first_int(rf'<Size{fallback_tag}>(\d+)</Size{fallback_tag}>', 1)

    size_z = _span('Z', 'Z')
    size_c = _span('C', 'C')
    size_t = _span('T', 'T')

    czi_ptype = (re.search(r'<PixelType>([^<]+)</PixelType>', md) or [None, 'Gray16'])
    czi_ptype = czi_ptype.group(1) if hasattr(czi_ptype, 'group') else 'Gray16'
    ome_ptype = PixelType(_CZI_PIXELTYPE_NAMES.get(czi_ptype, 'uint16'))

    # Scaling is stored in metres; OME wants micrometres.
    def _scale(axis):
        m = re.search(rf'<Distance Id="{axis}">\s*<Value>([^<]+)</Value>', md)
        return float(m.group(1)) * 1e6 if m else None

    psx, psy, psz = _scale('X'), _scale('Y'), _scale('Z')

    # Channel names + colours from the DisplaySetting block (Id -> name/colour).
    name_by_id, color_by_id = {}, {}
    disp = re.search(r'<DisplaySetting[^>]*>(.*?)</DisplaySetting>', md, re.S)
    if disp:
        for cid, body in re.findall(r'<Channel Id="([^"]*)"[^>]*>(.*?)</Channel>',
                                    disp.group(1), re.S):
            # Prefer ShortName (matches Bio-Formats) then DyeName then Name.
            dye = re.search(r'<(?:ShortName|DyeName|Name)>([^<]+)</', body)
            if dye:
                name_by_id.setdefault(cid, dye.group(1))
            col = re.search(r'<Color>#?([0-9A-Fa-f]{6,8})</Color>', body)
            if col:
                hexv = col.group(1)
                color_by_id.setdefault(cid, '#' + (hexv[2:] if len(hexv) == 8 else hexv))
    # Fallback names from <Dimensions><Channels>.
    for cid, cname in re.findall(r'<Channel Id="([^"]*)"[^>]*Name="([^"]*)"', md):
        name_by_id.setdefault(cid, cname)

    def _make_channels():
        chans = []
        for i in range(size_c):
            cid = f"Channel:{i}"
            kw = dict(id=cid, name=name_by_id.get(cid, f"Channel {i}"),
                      samples_per_pixel=1)
            if cid in color_by_id:
                try:
                    kw['color'] = color_by_id[cid]
                except Exception:
                    pass
            chans.append(Channel(**kw))
        return chans

    def _pixels(sx, sy):
        kw = dict(
            dimension_order=Pixels_DimensionOrder.XYZCT, type=ome_ptype,
            size_x=int(sx), size_y=int(sy), size_z=size_z, size_c=size_c,
            size_t=size_t, time_increment=1.0,
            time_increment_unit=UnitsTime.SECOND, channels=_make_channels(),
        )
        if psx:
            kw.update(physical_size_x=psx, physical_size_x_unit=UnitsLength.MICROMETER)
        if psy:
            kw.update(physical_size_y=psy, physical_size_y_unit=UnitsLength.MICROMETER)
        if psz:
            kw.update(physical_size_z=psz, physical_size_z_unit=UnitsLength.MICROMETER)
        return Pixels(**kw)

    # Per-scene image name (preserve the acquisition position names, e.g. P1..P25).
    scene_names = {int(i): nm for i, nm in
                   re.findall(r'<Scene Index="(\d+)"[^>]*Name="([^"]*)"', md)}

    images = []
    if rects:
        for i in sorted(rects.keys()):
            r = rects[i]
            name = scene_names.get(i, f"Scene:{i}")
            images.append(Image(id=f"Image:{i}", name=name, pixels=_pixels(r.w, r.h)))
    else:
        x = tbb.get('X', (0, _first_int(r'<SizeX>(\d+)</SizeX>', 1)))
        y = tbb.get('Y', (0, _first_int(r'<SizeY>(\d+)</SizeY>', 1)))
        images.append(Image(id="Image:0", name="Series_0",
                            pixels=_pixels(x[1] - x[0], y[1] - y[0])))
    return OME(images=images)


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
        """Get image data as dask array with dimension order TCZYX.

        For RGB/RGBA CZI files, the Samples ('S') axis is preserved and
        folded into the Channel axis so all samples are retained instead of
        only the first.
        """
        try:
            dims = self.img.dims
            has_samples = hasattr(dims, 'S') and dims.S > 1
            has_channels = hasattr(dims, 'C') and dims.C > 1
            if has_samples and not has_channels:
                data = self.img.get_image_dask_data(
                    dimension_order_out='TCZYXS',
                    **self.index_map
                )
                # 'C' is a padded singleton axis here; fold 'S' into its place.
                data = da.moveaxis(data, -1, 1)   # -> T S C Z Y X
                data = data[:, :, 0, :, :, :]     # -> T S Z Y X  (S acts as C)
                # CZI 'Bgr' samples are stored Blue, Green, Red; reverse to
                # R, G, B so the per-channel default colours (red, green, blue)
                # line up with the data they tint.
                if data.shape[1] == 3:
                    data = data[:, ::-1, :, :, :]
                return data
            if has_samples and has_channels:
                logger.warning(
                    f"{self._path}: both a multi-value 'C' ({dims.C}) and "
                    f"'S' ({dims.S}) dimension are present; only the first "
                    "sample will be read for each channel."
                )
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

    # ── Choose the CZI backend ───────────────────────────────────────────
    # aicspylibczi exposes individual mosaic tiles (needed for per-tile
    # extraction) but cannot open some large CZI files — libCZI's subblock
    # parser hits an offset error ("Invalid SubBlock-magic") on e.g. large
    # CellDiscoverer 7 plates.  pylibczirw stitches tiles into the full field of
    # view and reads large files robustly.  So aicspylibczi is used *only* when
    # individual tiles are actually needed — a genuine multi-tile mosaic read
    # with as_mosaic=False — and pylibczirw is used everywhere else.  Tile count
    # comes from the CZI metadata (SizeM), which is readable for any file size.
    n_tiles = _czi_tile_count(input_path)

    # Views and illuminations live on the CZI 'V' / 'I' dimensions, which the
    # pylibczirw reader collapses (it exposes only X/Y/C/Z/T — so a single-tile
    # multi-view/illumination file would silently merge to one output).
    # aicspylibczi exposes the full BVITCZYX layout, so any request that must
    # address an individual view or illumination has to go through it — even for
    # single-tile files.
    needs_view_illu = (view_index != 0) or (illumination_index != 0)

    def _open_aics():
        """Open via aicspylibczi — exposes V/I and individual mosaic tiles.

        chunk_dims includes the raw libCZI samples axis 'A' so multi-sample
        (RGB/Bgr) pixels read correctly (bioio_czi's default lists the renamed
        'S', which never matches the raw 'A', collapsing all samples to sample 0
        — see CZIReader.get_image_dask_data).
        """
        from bioio_czi.aicspylibczi_reader.reader import Reader
        return Reader(input_path, chunk_dims=['Z', 'Y', 'X', 'A'])

    if as_mosaic:
        if n_tiles <= 1:
            logger.warning(
                f"'{input_path}': as_mosaic=True has no effect — the image has a "
                f"single tile (nothing to stitch)."
            )
        try:
            img = _get_mosaic_reader_class()(input_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read CZI file: {str(e)}") from e
    elif needs_view_illu:
        # Individual view / illumination extraction requested.  Only
        # aicspylibczi exposes the V / I dimensions, so use it regardless of
        # tile count (pylibczirw would collapse them into a single output).
        try:
            img = _open_aics()
        except Exception as e:
            # Large-file subblock-offset limitation — fall back so the
            # conversion still succeeds, but V/I can no longer be separated.
            logger.warning(
                f"The native (aicspylibczi) reader could not open "
                f"'{input_path}' ({e}) to expose views/illuminations. Falling "
                f"back to the pylibczirw reader: individual views and "
                f"illuminations cannot be separated for this file."
            )
            try:
                img = _get_mosaic_reader_class()(input_path)
            except Exception as e2:
                raise RuntimeError(f"Failed to read CZI file: {str(e2)}") from e2
            as_mosaic = True
    elif n_tiles <= 1:
        # Single tile: pylibczirw yields the identical result, is more efficient,
        # and reads large files that aicspylibczi's subblock parser rejects.
        try:
            img = _get_mosaic_reader_class()(input_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read CZI file: {str(e)}") from e
        as_mosaic = True
    else:
        # Multi-tile mosaic with individual-tile extraction requested.
        try:
            img = _open_aics()
        except Exception as e:
            # aicspylibczi could not open this multi-tile file (large-file
            # subblock-offset limitation).  Fall back to pylibczirw so the
            # conversion still succeeds — but tiles are STITCHED into the full
            # FOV and can no longer be extracted individually.
            logger.warning(
                f"The native (aicspylibczi) reader could not open multi-tile "
                f"'{input_path}' ({e}). Falling back to the pylibczirw reader: "
                f"mosaic tiles will be STITCHED into the full field of view per "
                f"scene and individual tiles cannot be extracted."
            )
            try:
                img = _get_mosaic_reader_class()(input_path)
            except Exception as e2:
                raise RuntimeError(f"Failed to read CZI file: {str(e2)}") from e2
            as_mosaic = True

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

