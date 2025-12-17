import zarr, natsort, copy, asyncio

import zarr, psutil, dask, json
import numpy as np, os

from ome_types.model import OME, Image, Pixels, Channel  # TiffData, Plane
from ome_types.model import PixelType, Pixels_DimensionOrder, UnitsLength, UnitsTime

from dask import array as da
from pathlib import Path
from typing import Union, Optional, Any, Dict, List, Iterable

from eubi_bridge.ngff.multiscales import Pyramid
from eubi_bridge.ngff.defaults import unit_map, scale_map, default_axes
from eubi_bridge.utils.path_utils import sensitive_glob, is_zarr_group, is_zarr_array, take_filepaths
from eubi_bridge.utils.array_utils import autocompute_chunk_shape
from eubi_bridge.core.readers import (read_metadata_via_bioio_bioformats,
                                      read_metadata_via_extension,
                                      read_metadata_via_bfio,
                                      # read_single_image_asarray,
                                      read_single_image)
from eubi_bridge.utils.logging_config import get_logger

# Set up logger for this module
logger = get_logger(__name__)


def abbreviate_units(measure: str) -> str:
    """Abbreviate a unit of measurement.

    Given a human-readable unit of measurement, return its abbreviated form.

    Parameters
    ----------
    measure : str
        The human-readable unit of measurement to abbreviate, e.g. "millimeter".

    Returns
    -------
    str
        The abbreviated form of the unit of measurement, e.g. "mm".

    Notes
    -----
    The abbreviations are as follows:

    * Length measurements:
        - millimeter: mm
        - centimeter: cm
        - decimeter: dm
        - meter: m
        - decameter: dam
        - hectometer: hm
        - kilometer: km
        - micrometer: µm
        - nanometer: nm
        - picometer: pm
    * Time measurements:
        - second: s
        - millisecond: ms
        - microsecond: µs
        - nanosecond: ns
        - minute: min
        - hour: h
    """
    if measure is None:
        return None

    abbreviations = {
        # Length measurements
        "millimeter": "mm",
        "centimeter": "cm",
        "decimeter": "dm",
        "meter": "m",
        "decameter": "dam",
        "hectometer": "hm",
        "kilometer": "km",
        "micrometer": "µm",
        "nanometer": "nm",
        "picometer": "pm",

        # Time measurements
        "second": "s",
        "millisecond": "ms",
        "microsecond": "µs",
        "nanosecond": "ns",
        "minute": "min",
        "hour": "h"
    }

    # Return the input if it's already an abbreviation
    if measure.lower() in abbreviations.values():
        return measure.lower()

    return abbreviations.get(measure.lower(), "Unknown")


def expand_units(measure: str) -> str:
    """
    Expand a unit of measurement.

    Given an abbreviated unit of measurement, return its expanded form.

    Parameters
    ----------
    measure : str
        The abbreviated unit of measurement to expand, e.g. "mm".

    Returns
    -------
    str
        The expanded form of the unit of measurement, e.g. "millimeter".
    """
    # Define the abbreviations and their expansions

    if measure is None:
        return None

    expansions = {
        # Length measurements
        "mm": "millimeter",
        "cm": "centimeter",
        "dm": "decimeter",
        "m": "meter",
        "dam": "decameter",
        "hm": "hectometer",
        "km": "kilometer",
        "µm": "micrometer",
        "nm": "nanometer",
        "pm": "picometer",

        # Time measurements
        "s": "second",
        "ms": "millisecond",
        "µs": "microsecond",
        "ns": "nanosecond",
        "min": "minute",
        "h": "hour"
    }

    # Return the input if it's already an expanded form
    if measure.lower() in expansions.values():
        return measure.lower()

    # Return the expanded form if it exists, else return "Unknown"
    return expansions.get(measure.lower(), "Unknown")


def create_ome_xml(  # make 5D omexml
        image_shape: tuple,
        axis_order: str,
        pixel_size_x: float = None,
        pixel_size_y: float = None,
        pixel_size_z: float = None,
        pixel_size_t: float = None,
        unit_x: str = "MICROMETER",
        unit_y: str = None,
        unit_z: str = None,
        unit_t: str = None,
        dtype: str = "uint8",
        image_name: str = "Default Image",
        channel_names: list = None
) -> str:
    fullaxes = 'xyczt'
    if len(axis_order) != len(image_shape):
        raise ValueError("Length of axis_order must match length of image_shape")
    axis_order = axis_order.upper()

    pixel_size_basemap = {
        'time_increment': pixel_size_t,
        'physical_size_z': pixel_size_z,
        'physical_size_y': pixel_size_y,
        'physical_size_x': pixel_size_x
    }

    pixel_size_map = {}
    for ax in 'tzyx':
        if ax == 't':
            if ax in axis_order.lower():
                pixel_size_map['time_increment'] = pixel_size_t or 1
        else:
            if ax in axis_order.lower():
                pixel_size_map[f'physical_size_{ax}'] = pixel_size_basemap[f'physical_size_{ax}'] or 1

    unit_basemap = {
        'time_increment_unit': unit_t,
        'physical_size_z_unit': unit_z,
        'physical_size_y_unit': unit_y,
        'physical_size_x_unit': unit_x,
    }

    unit_map = {}
    for ax in 'tzyx':
        if ax == 't':
            if ax in axis_order.lower():
                unit_map['time_increment_unit'] = unit_t or 'second'
        else:
            if ax in axis_order.lower():
                unit_map[f'physical_size_{ax}_unit'] = unit_basemap[f'physical_size_{ax}_unit'] or 'MICROMETER'
    unit_map = {key: abbreviate_units(value) for key, value in unit_map.items() if value is not None}

    # Map numpy dtype to OME PixelType
    dtype_map = {
        "uint8": PixelType.UINT8,
        "uint16": PixelType.UINT16,
        "uint32": PixelType.UINT32,
        "int8": PixelType.INT8,
        "int16": PixelType.INT16,
        "int32": PixelType.INT32,
        "float32": PixelType.FLOAT,
        "float64": PixelType.DOUBLE,
    }

    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}")

    pixel_type = dtype_map[dtype]

    # Initialize axis sizes
    size_map_ = dict(zip(axis_order.lower(), image_shape))
    size_map = {}
    for ax in fullaxes:
        if ax in size_map_:
            size_map[f'size_{ax}'] = size_map_[ax]
        else:
            size_map[f'size_{ax}'] = 1

    if channel_names is None or len(channel_names) != size_map['size_c']:
        channels = [Channel(id=f"Channel:{idx}",  # TODO: if exists, directly take the channel names
                            samples_per_pixel=1)
                    for idx in range(size_map['size_c'])]
    else:
        channels = [Channel(id=f"Channel:{idx}",  # TODO: if exists, directly take the channel names
                            samples_per_pixel=1,
                            name=channel_names[idx])
                    for idx in range(size_map['size_c'])]

    pixels = Pixels(
        dimension_order=Pixels_DimensionOrder(fullaxes.upper()),
        **size_map,
        type=pixel_type,
        **pixel_size_map,
        **unit_map,
        channels=channels,
    )

    image = Image(id="Image:0", name=image_name, pixels=pixels)

    ome = OME(images=[image])

    return ome


def parse_series(series: Union[Iterable, int]):
    if series is None:
        series = 0
    if np.isscalar(series):
        series = [series]
    return series


class PFFImageMeta:
    essential_omexml_fields = {
        "physical_size_x", "physical_size_x_unit",
        "physical_size_y", "physical_size_y_unit",
        "physical_size_z", "physical_size_z_unit",
        "time_increment", "time_increment_unit",
        "size_x", "size_y", "size_z", "size_t", "size_c"
    }

    def __init__(self,
                 path,
                 meta_reader="bioio",
                 aszarr=False
                 ):
        # series = parse_series(series)
        self.root = path
        self._series = 0
        # images = [self.omemeta.images[series]]
        self.omemeta = None
        self.pyr = None
        self._series = 0
        self._tile = 0
        self._aszarr = aszarr
        self.arraydata = None
        self.reader = None
        self._meta_reader = meta_reader
        self._n_scenes = None
        self._n_tiles = None
        # self._channels = None

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    async def read_omemeta(self):
        if self.root.endswith('ome') or self.root.endswith('xml'):
            from ome_types import OME
            omemeta = OME().from_xml(self.root)
        else:
            if self._meta_reader == 'bioio':
                # Try to read the metadata via bioio
                try:
                    omemeta = await read_metadata_via_extension(self.root, series=self._series)
                except Exception as e:
                    # Fallback to bioformats if bioio extension reader fails
                    logger.debug(f"bioio extension reader failed for {self.root}: {e}. Falling back to bioformats.")
                    omemeta = await read_metadata_via_bioio_bioformats(self.root, series=self._series)
            elif self._meta_reader == 'bfio':
                try:
                    omemeta = await read_metadata_via_bfio(self.root)  # don't pass series, will be handled afterwards
                except Exception as e:
                    # Fallback to bioformats if bfio reader fails
                    logger.debug(f"bfio reader failed for {self.root}: {e}. Falling back to bioformats.")
                    omemeta = await read_metadata_via_bioio_bioformats(self.root, series=self._series)
            else:
                raise ValueError(f"Unsupported metadata reader: {self._meta_reader}")
        self.omemeta = omemeta
        self._n_scenes = len(self.omemeta.images)

        await self.set_scene(self._series)
        await self.set_tile(self._tile)

    async def get_arraydata(self):
        pix = self.pixels
        # img = self.reader.img
        dims = self.reader.img.dims
        shape = (pix.size_t, pix.size_c, pix.size_z, pix.size_y, pix.size_x)

        if not hasattr(dims, 'S'):
            dask_data = self.reader.get_image_dask_data(dimensions_to_read = 'TCZYX')
        elif hasattr(dims, 'S') and not hasattr(dims, 'C'):
            logger.warn(f"As dimension names, 'S' was found but no 'C'. 'S' is being assumed as channel dimension.")
            dask_data = self.reader.get_image_dask_data(dimensions_to_read = 'TSZYX')
            logger.info(f"Current dask data shape: {dask_data.shape}")
        elif hasattr(dims, 'S') and hasattr(dims, 'C'):
            if dims.C != pix.size_c & dims.S == pix.size_c:
                logger.warn(f"As dimension names, both 'S' and 'C' were found but 'S' seems to match the real dimension number. Assuming 'S' as channel dimension.")
                dask_data = self.reader.get_image_dask_data(dimensions_to_read = 'TSZYX')
                logger.info(f"Current dask data shape: {dask_data.shape}")
            else:
                dask_data = self.reader.get_image_dask_data(dimensions_to_read = 'TCZYX')
        else:
            dask_data = self.reader.get_image_dask_data(dimensions_to_read = 'TCZYX')
        return dask_data

    async def set_scene(self, scene_index):
        self._series = scene_index
        if self.reader is not None:
            self.reader.set_scene(self._series)
            self.arraydata = await self.get_arraydata()
        # if self.omemeta is not None:
        #     self.pixels = copy.deepcopy(self.omemeta.images[self._series].pixels)
        #     missing_fields = self.essential_omexml_fields - self.pixels.model_fields_set
        #     self.pixels.model_fields_set.update(missing_fields)

    async def set_tile(self, mosaic_tile_index):
        self._tile = 0
        if self.reader is not None:
            if hasattr(self.reader, 'set_tile') and self.reader.n_tiles > 1:
                self._tile = mosaic_tile_index
                self.reader.set_tile(self._tile)
                self.arraydata = await self.get_arraydata()

        # if self.omemeta is not None:
        #     self.pixels = copy.deepcopy(self.omemeta.images[self._series].pixels)
        #     missing_fields = self.essential_omexml_fields - self.pixels.model_fields_set
        #     self.pixels.model_fields_set.update(missing_fields)

    @property
    def n_tiles(self):
        """Get number of tiles, with fallback to cached value."""
        if self.reader is not None and hasattr(self.reader, 'n_tiles'):
            try:
                return self.reader.n_tiles
            except Exception as e:
                logger.warning(f"Failed to get n_tiles from reader: {e}. Using cached value.")
        return self._n_tiles

    @property
    def n_scenes(self):
        if self.root.endswith('.lsm'):
            return self._n_scenes # use the one from ome metadata.
            # With .lsm files, a downscaled version of the image is stored as a second scene. This is not found in the ome metadata.
        else:
            return self.reader.n_scenes # Use the one from the one from Reader.scenes. Note that this is sometimes (with .lsm images) not consistent with the ome metadata.


    def get_pixels(self):
        """Get pixels metadata from OME metadata, with validation."""
        if self.omemeta is None:
            raise ValueError(f"OME metadata not loaded for path {self.root}")
        if self._series >= len(self.omemeta.images):
            raise ValueError(f"Series {self._series} out of range for path {self.root}")
        
        try:
            pixels = self.omemeta.images[self._series].pixels
            missing_fields = self.essential_omexml_fields - pixels.model_fields_set
            pixels.model_fields_set.update(missing_fields)
        except (AttributeError, IndexError) as e:
            raise ValueError(f"Failed to get pixels from path {self.root} series {self._series}: {e}") from e
        return pixels

    @property
    def pixels(self):
        return self.get_pixels()

    async def read_img(self, **kwargs):
        self.reader = await read_single_image(self.root, aszarr=self._aszarr, **kwargs)
        await self.set_scene(self._series)
        await self.set_tile(self._tile)
        self.arraydata = await self.get_arraydata()

    async def get_pyramid(self, version='0.4'):
        ### add channels from omemeta
        array = await self.get_arraydata()
        pyr = Pyramid().from_array(array=array,
                                   axis_order=self.get_axes(),
                                   unit_list=self.get_units(),
                                   scale=self.get_scales(),
                                   version=version,
                                   name="Series_0"
                                   )
        return pyr

    async def read_dataset(self):
        await self.read_omemeta()
        await self.read_img()

    def get_axes(self):
        return 'tczyx'

    def get_scaledict(self):
        return {
            't': self.pixels.time_increment,
            'z': self.pixels.physical_size_z,
            'y': self.pixels.physical_size_y,
            'x': self.pixels.physical_size_x
        }

    def get_scales(self):
        scaledict = self.get_scaledict()
        caxes = [ax for ax in self.get_axes() if ax != 'c']
        return [scaledict[ax] for ax in caxes]

    def get_unitdict(self):
        return {
            't': self.pixels.time_increment_unit.name.lower(),
            'z': self.pixels.physical_size_z_unit.name.lower(),
            'y': self.pixels.physical_size_y_unit.name.lower(),
            'x': self.pixels.physical_size_x_unit.name.lower()
        }

    def get_units(self):
        unitdict = self.get_unitdict()
        caxes = [ax for ax in self.get_axes() if ax != 'c']
        return [unitdict[ax] for ax in caxes]

    def get_channels(self): #TODO: Here fix 'things_to_solve' point 10
        pixels = self.get_pixels()
        if not hasattr(pixels, 'channels'):
            return None
        if len(pixels.channels) == 0:
            return None
        ###
        if len(pixels.channels) < pixels.size_c:
            chn = ChannelIterator(num_channels=pixels.size_c)
            channels = chn._channels
        elif len(pixels.channels) == pixels.size_c:
            channels = []
            for _, channel in enumerate(pixels.channels):
                color = channel.color.as_hex().upper()
                color = expand_hex_shorthand(color)
                name = channel.name
                channels.append(dict(
                    label=name,
                    color=color
                ))
        return channels

    def snapshot(self):
        """Return a snapshot of safe-to-copy attributes."""
        state = {}
        for key, value in self.__dict__.items():
            if key in ("reader", "omemeta", "pyr"):  # skip non-copyable stuff
                continue
            state[key] = copy.deepcopy(value)
        return state

    def restore(self, snapshot):
        """Restore a previously taken snapshot."""
        for key, value in snapshot.items():
            setattr(self, key, copy.deepcopy(value))

    # async def read_array_data(self):
    #     self._arraydata = await read_single_image_asarray(self.root_path, scene_index=self.series)


class TIFFImageMeta(PFFImageMeta):
    essential_omexml_fields = {
        "physical_size_x", "physical_size_x_unit",
        "physical_size_y", "physical_size_y_unit",
        "physical_size_z", "physical_size_z_unit",
        "time_increment", "time_increment_unit",
        "size_x", "size_y", "size_z", "size_t", "size_c"
    }

    def __init__(self,
                 path,
                 meta_reader="bioio",
                 aszarr=False,
                 ):
        if not path.endswith(('.tif', '.tiff', '.lsm')):
            raise Exception(f"The given path is not a TIFF file: {path}")

        super().__init__(path, meta_reader, aszarr)

    async def read_omemeta(self):
        import tifffile
        tif = tifffile.TiffFile(self.root)
        self.tiffzarrstore = tif.aszarr()
        self._zarrmeta = self.tiffzarrstore._data[self._series]
        self._meta = tif.series[self._series]
        await super().read_omemeta()

    def get_axes(self):
        axes = self._meta.axes.lower()
        # if 's' in axes:
        #     axes = axes.replace('s', 'c')
        # if 'q' in axes:
        #     axes = axes.replace('q', 'c')
        default_axes_cut = default_axes[-len(axes):]
        newaxes = []
        for ax in axes:
            if ax == 's':
                ax = 'c'
            if ax in default_axes:
                newaxes.append(ax)
            else:
                idx = axes.index(ax)
                newaxes.append(default_axes_cut[idx])
        return ''.join(newaxes)

    def get_scaledict(self):
        scaledict = super().get_scaledict()
        axes = self.get_axes()
        return {ax: scaledict[ax]
                for ax in axes
                if ax in scaledict
                }

    def get_unitdict(self):
        unitdict = super().get_unitdict()
        caxes = [ax for ax in self.get_axes() if ax != 'c']
        return {ax: unitdict[ax]
                for ax in caxes
                if ax in unitdict
                }


class H5ImageMeta(PFFImageMeta):
    essential_omexml_fields = {
        "physical_size_x", "physical_size_x_unit",
        "physical_size_y", "physical_size_y_unit",
        "physical_size_z", "physical_size_z_unit",
        "time_increment", "time_increment_unit",
        "size_x", "size_y", "size_z", "size_t", "size_c"
    }

    def __init__(self,
                 path,
                 meta_reader="bioio",  # placeholder
                 **kwargs
                 ):

        if path.endswith('.h5'):
            super().__init__(path, meta_reader,
                             aszarr=False, **kwargs)
        else:
            raise Exception(f"The given path is not an HDF5 file: {path}")

    async def read_omemeta(self, **kwargs):  # This is not real omemeta, just meta.
        import h5py
        f = h5py.File(self.root)
        dset_name = list(f.keys())[self._series]
        ds = f[dset_name]
        self._ds = ds
        self._attrs = dict(ds.attrs)
        self.n_scenes = len(list(f.keys()))

    def get_axes(self):
        attrs = self._attrs
        axistags = attrs.get('axistags', {})
        if isinstance(axistags, str):
            axistags = json.loads(axistags)
        axlist = axistags.get('axes', [])
        axes = []
        for idx, ax in enumerate(axlist):
            if 'key' in ax:
                if ax['key'] in 'tczyx':
                    axes.append(ax['key'])
                else:
                    axes.append(default_axes[idx])
            else:
                axes.append(default_axes[idx])
        axes = ''.join(axes)
        return axes

    def get_scaledict(self):
        attrs = self._attrs
        axistags = attrs.get('axistags', {})
        if isinstance(axistags, str):
            axistags = json.loads(axistags)
        scaledict = {}
        axes = self.get_axes()
        axlist = axistags.get('axes', [])

        for idx, ax in enumerate(axlist):
            if ax['key'] == 'c':
                continue
            if 'key' in ax:
                if ax['key'] in axes:
                    if 'scale' in ax:
                        scaledict[ax['key']] = ax['scale']
                    elif 'resolution' in ax:
                        scaledict[ax['key']] = ax['resolution']
                    else:
                        scaledict[ax['key']] = scale_map[ax['key']]
        return scaledict

    def get_unitdict(self):
        attrs = self._attrs
        axistags = attrs.get('axistags', {})
        if isinstance(axistags, str):
            axistags = json.loads(axistags)
        unitdict = {}
        axes = self.get_axes()
        axlist = axistags.get('axes', [])
        axes = self.get_axes()
        for idx, ax in enumerate(axlist):
            if 'key' in ax:
                if ax['key'] == 'c':
                    continue
                if ax['key'] in axes:
                    if 'unit' in ax:
                        unitdict[ax['key']] = ax['scale']
                    else:
                        unitdict[ax['key']] = unit_map[ax['key']]
        return unitdict

    def get_channels(self):
        return []


# path = f"/home/oezdemir/PycharmProjects/TIM2025/data/h5/ilastik_img/Patient2_002.h5"
# h5 = H5ImageMeta(path)
# await h5.read_omemeta()


class NGFFImageMeta(PFFImageMeta):  ### Maybe set_scene can pick particular resolution layer?
    def __init__(self,
                 path,
                 aszarr=False
                 ):
        if is_zarr_group(path):
            super().__init__(path=path, aszarr=aszarr)
            self._base_path = '0'  ### This will be represented as series later.
        else:
            raise Exception(f"The given path is not an NGFF group: {path}")

    async def read_omemeta(self, **kwargs):  # This is not really omemeta, it is just meta.
        if self.reader is None: # MockImg from read_pyramid
            self.reader = await read_single_image(self.root, # = read_pyramid
                                                aszarr=self._aszarr,
                                                **kwargs)
        self._meta = self.reader.pyr.meta

    async def read_img(self, **kwargs):
        self.reader = await read_single_image(self.root,
                                            aszarr=self._aszarr,
                                            **kwargs)
        # await self.set_scene(self._series)
        self.arraydata = await self.get_arraydata()

    # async def get_arraydata(self):
    #     return self._img.get_image_dask_data()

    async def read_dataset(self):
        await self.read_omemeta()
        await self.read_img()
        self._meta = self.reader.pyr.meta

    def get_axes(self):
        return self._meta.axis_order

    def get_scales(self):
        return self._meta.get_scale(self._base_path)

    def get_scaledict(self):
        return self._meta.get_scaledict(self._base_path)

    def get_units(self):
        return self._meta.unit_list

    def get_unitdict(self):
        return self._meta.unit_dict

    def get_channels(self):
        if not hasattr(self._meta, 'channels'):
            return None
        return self._meta.channels

    async def get_arraydata(self): ###TODO UPDATE LIKE PFFImageMeta
        return self.reader.get_image_dask_data()

    async def get_pyramid(self, version='0.4'):
        ### add channels from omemeta
        return self.reader.pyr


def expand_hex_shorthand(hex_color):
    """
    Expands a shorthand hex color of any valid length (e.g., #abc → #aabbcc, #1234 → #11223344).
    """
    if not hex_color.startswith('#'):
        raise ValueError("Hex color must start with '#'")

    shorthand = hex_color[1:]

    if not all(c in '0123456789ABCDEFabcdef' for c in shorthand):
        raise ValueError("Invalid hex digits")

    expanded = '#' + ''.join([c * 2 for c in shorthand])
    return expanded


class ArrayManager:  ### Unify the classes above.
    """Async-friendly version of ArrayManager.

    Any potentially blocking I/O is offloaded to a thread via ``asyncio.to_thread``.
    CPU / in-memory ops (crop, transpose, etc.) remain synchronous.
    """

    # TODO: Fix the data type change from layer0 to layer 1,2,3..
    essential_omexml_fields = {
        "physical_size_x", "physical_size_x_unit",
        "physical_size_y", "physical_size_y_unit",
        "physical_size_z", "physical_size_z_unit",
        "time_increment", "time_increment_unit",
        "size_x", "size_y", "size_z", "size_t", "size_c"
    }

    # ------------------------------------------------------------------
    # Construction / initialization
    # ------------------------------------------------------------------
    def __init__(self,
                 path: Union[str, Path, None] = None,
                 metadata_reader: str = 'bfio',  # bfio or aicsimageio
                 skip_dask: bool = False,
                 **kwargs: Any):
        self.path = str(path) if path is not None else None
        self.series = kwargs.get('series', 0)
        self.series_path = self.path + f'_{self.series}'
        self.mosaic_tile_index = kwargs.get('mosaic_tile_index', None)
        if self.mosaic_tile_index is not None:
            self.series_path += f'_tile{self.mosaic_tile_index}'

        self._meta_reader = metadata_reader
        self._skip_dask = skip_dask

        # Will be set during init()
        self.img = None
        self.axes: str = ""
        self.array: Optional[Union[da.Array, zarr.Array]] = None
        self.ndim: Optional[int] = None
        self.caxes: str = ""
        self.chunkdict: Dict[str, Any] = {}
        self.shapedict: Dict[str, int] = {}
        self._channels: Optional[List[Dict[str, Any]]] = None
        self.scaledict: Dict[str, Any] = {}
        self.unitdict: Dict[str, Any] = {}
        self.omemeta = None
        self.pyr = None
        self.img = None
        self.loaded_scenes = None
        self.loaded_tiles = None

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    async def init(self):
        """Async initializer: detects image type and loads metadata off-thread."""
        import asyncio
        if not self.path is None:
            # try:
            if await asyncio.to_thread(is_zarr_group, self.path):
                self.img = await asyncio.to_thread(NGFFImageMeta,
                                                   self.path,
                                                   self._skip_dask)
            elif self.path.endswith('.h5'):
                self.img = await asyncio.to_thread(H5ImageMeta,
                                                   self.path,
                                                   self._meta_reader
                                                   )
            elif not self._skip_dask:
                self.img = await asyncio.to_thread(PFFImageMeta,
                                                   self.path,
                                                   self._meta_reader,
                                                   self._skip_dask,
                                                   )
            else:
                if (str(self.path).endswith(('tif', 'tiff'))):
                    # and
                    # not str(self.path).endswith(('ome.tif', 'ome.tiff'))):
                    self.img = await asyncio.to_thread(TIFFImageMeta,
                                                       self.path,
                                                       self._meta_reader,
                                                       self._skip_dask)
                else:
                    logger.warning(f"The given path is not a TIFF file: {self.path}.\n"
                                   f"The 'skip_dask' option is ignored, as its use is currently limited to TIFF files.")
                    self.img = await asyncio.to_thread(PFFImageMeta,
                                                       self.path,
                                                       self._meta_reader,
                                                       self._skip_dask,
                                                       )

            await self.img.read_dataset()
            await self.set_scene(self.series)
            if self.mosaic_tile_index is not None:
                await self.set_tile(self.mosaic_tile_index)
            if self.img.omemeta is not None:
                self.omemeta = self.img.omemeta
        return self

    async def set_scene(self, scene_idx):
        # if self.series == scene_idx:
        #     return self
        if self.img is not None:
            await self.img.set_scene(scene_idx)
            self.pyr = await self.img.get_pyramid()
            # Pull basic properties
            self.axes = self.img.get_axes()
            self.set_arraydata(self.img.arraydata)
            self.series = scene_idx
            self.series_path = self.path + f'_{self.series}'
            self.omemeta = self.img.omemeta
            # self.pixels = copy.copy(self.omemeta.images[self.series].pixels)
            self._channels = self.img.get_channels()
        else:
            raise Exception("Image is missing. An image needs to be read.")
        return self

    async def set_tile(self, mosaic_tile_index):
        if self.img is not None:
            await self.img.set_tile(mosaic_tile_index)
            self.pyr = await self.img.get_pyramid()
            # Pull basic properties
            self.axes = self.img.get_axes()
            self.set_arraydata(self.img.arraydata)
            self.mosaic_tile_index = self.img._tile
            self.series_path = self.path + f'_{self.series}' + f'_tile{self.mosaic_tile_index}'
            self.omemeta = self.img.omemeta
            # self.pixels = copy.copy(self.omemeta.images[self.series].pixels)
            self._channels = self.img.get_channels()
        else:
            raise Exception("Image is missing. An image needs to be read.")
        return self

    async def load_scenes(self,
                          scene_indices: Union[int, str, List[int]]
                          ):
        """TODO: Maybe make sure it checks and loads only available indices."""

        # if self.img.n_scenes is None:
        #     self.loaded_scenes = {self.series_path: self}
        #     return

        if scene_indices == 'all':
            scene_indices = list(range(self.img.n_scenes))
        elif np.isscalar(scene_indices):
            scene_indices = [scene_indices]

        scene_indices_ = []
        for idx in scene_indices:
            if idx < self.img.n_scenes:
                scene_indices_.append(idx)
            else:
                logger.warning(f"Scene index {idx} is out of bounds for the path {self.path}.\n"
                               f"Skipping the nonexistent scene {idx}.")
        scene_indices = scene_indices_
        scenes = []

        async def copy_scene(manager, scene_idx):
            await manager.set_scene(scene_idx)
            return copy.copy(manager)

        for scene_idx in scene_indices:
            scenes.append(copy_scene(self, scene_idx))
        import asyncio
        scenelist = await asyncio.gather(*scenes)
        self.loaded_scenes = {scene.series_path: scene for scene in scenelist}
        return self.loaded_scenes

    async def load_tiles(self, tile_indices: Union[int, str, List[int]]):
        """TODO: Maybe make sure it checks and loads only available indices."""
        n_tiles = self.img.n_tiles or 1
        if tile_indices == 'all':
            tile_indices = list(range(n_tiles))
        elif np.isscalar(tile_indices):
            tile_indices = [tile_indices]
        tile_indices_ = []
        for idx in tile_indices:
            if idx < n_tiles:
                tile_indices_.append(idx)
            else:
                logger.warning(f"Tile index {idx} is out of bounds for the path {self.path}.\n"
                               f"Skipping the nonexistent tile {idx}.")
        tile_indices = tile_indices_
        tiles = []

        async def copy_scene(manager, tile_idx):
            await manager.set_tile(tile_idx)
            return copy.copy(manager)

        for tile_idx in tile_indices:
            tiles.append(copy_scene(self, tile_idx))
        import asyncio
        tilelist = await asyncio.gather(*tiles)
        self.loaded_tiles = {tile.series_path: tile for tile in tilelist}
        return self.loaded_tiles

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def fill_default_meta(self):
        # if self.pyr is not None:
        #     zgroup = self.pyr.meta.zarr_group
        #     omero_copy = copy.copy(self.pyr.meta.omero)
        if self.array is None:
            raise Exception("Array is missing. An array needs to be assigned.")
        new_scaledict: Dict[str, Any] = {}
        new_unitdict: Dict[str, Any] = {}
        values = list(self.scaledict.values())
        if None not in values:
            return self

        for ax, value in self.scaledict.items():
            if value is None:
                if (ax in ('z', 'y')) and self.scaledict.get('x') is not None:
                    new_scaledict[ax] = self.scaledict['x']
                    new_unitdict[ax] = self.unitdict['x']
                else:
                    new_scaledict[ax] = scale_map[ax]
                    new_unitdict[ax] = unit_map[ax]
            else:
                if ax in self.scaledict:
                    new_scaledict[ax] = self.scaledict[ax]
                if ax in self.unitdict:
                    new_unitdict[ax] = self.unitdict[ax]

        new_units = [new_unitdict[ax] for ax in self.axes if ax in new_unitdict]
        new_scales = [new_scaledict[ax] for ax in self.axes if ax in new_scaledict]

        self.set_arraydata(self.array, self.axes, new_units, new_scales)
        return self

    def get_pixel_size_basemap(self, t=1, z=1, y=1, x=1, **kwargs):
        return {'pixel_size_t': t, 'pixel_size_z': z, 'pixel_size_y': y, 'pixel_size_x': x}

    def get_unit_basemap(self, t='second', z='micrometer', y='micrometer', x='micrometer', **kwargs):
        return {'unit_t': t, 'unit_z': z, 'unit_y': y, 'unit_x': x}

    def update_meta(self,
                    new_scaledict: Dict[str, Any] = {},
                    new_unitdict: Dict[str, Any] = {}):
        # await self.set_scene(self.series) # TODO!!! Probably set scene first
        scaledict = self.img.get_scaledict()
        for key, val in new_scaledict.items():
            if key in scaledict and val is not None:
                scaledict[key] = val

        scales = [scaledict[ax] for ax in (self.axes if 'c' in scaledict else self.caxes)]

        unitdict = self.img.get_unitdict()  # TODO: remove this and use set values directly.
        for key, val in new_unitdict.items():
            if key in unitdict and val is not None:
                unitdict[key] = val

        units = [expand_units(unitdict[ax]) for ax in (self.axes if 'c' in unitdict else self.caxes)]

        self.set_arraydata(array=self.array, axes=self.axes, units=units, scales=scales)

    def _ensure_correct_channels(self):
        if self.array is None:
            return
        if self.channels is None:
            return
        shapedict = dict(zip(list(self.axes), self.array.shape))
        csize = shapedict.get('c', None)
        if csize is None:
            return
        channelsize = len(self.channels)
        if channelsize > csize:
            self._channels = [channel for channel in self.channels if channel['label'] is not None]

    def fix_bad_channels(self):
        chn = ChannelIterator()
        for i, channel in enumerate(self.channels):
            if channel.get('label') in (None, ''):
                channel = next(chn)
            self.channels[i] = channel

    def compute_intensity_extrema(self,
                                  dtype
                                  ):
        if 'c' in self.axes:
            c_axis = self.axes.index('c')
            n_channels = self.array.shape[c_axis]
        else:
            c_axis = None
            n_channels = 1
        if dtype is None:
            dtype = self.array.dtype
        if np.issubdtype(dtype, np.integer):
            starts = [np.iinfo(dtype).min for _ in range(n_channels)]
            ends = [np.iinfo(dtype).max for _ in range(n_channels)]
        elif np.issubdtype(dtype, np.floating):
            starts = [np.finfo(dtype).min for _ in range(n_channels)]
            ends = [np.finfo(dtype).max for _ in range(n_channels)]
        else:
            raise ValueError(f"Unsupported dtype {dtype}")
        return starts, ends

    def compute_intensity_limits(self,
                                 from_array=False,
                                 dtype=None,
                                 start_intensity=None,
                                 end_intensity=None,
                                 ):
        if 'c' in self.axes:
            c_axis = self.axes.index('c')
            n_channels = self.array.shape[c_axis]
        else:
            c_axis = None
            n_channels = 1
        starts = None
        ends = None

        if start_intensity is not None:
            starts = [start_intensity for _ in range(n_channels)]
        if end_intensity is not None:
            ends = [end_intensity for _ in range(n_channels)]

        if starts is not None and ends is not None:
            return starts, ends

        if from_array and self.array is None:
            raise Exception(f"Manager needs an array to detect the intensity extrema.")
        elif dtype is None and self.array is None:
            raise Exception(f"Manager needs a dtype to detect the intensity extrema.")
        elif from_array:
            if 'c' in self.axes:
                axes_to_compute = tuple([self.axes.index(ax) for ax in self.axes if ax != 'c'])
            else:
                axes_to_compute = tuple([self.axes.index(ax) for ax in self.axes])
            if isinstance(self.array, zarr.Array):
                arr = da.from_zarr(self.array)
            else:
                arr = self.array
            if starts is None:
                starts = arr.min(axis=axes_to_compute).compute().tolist()
                if np.isscalar(starts):
                    starts = [starts]
            if ends is None:
                ends = arr.max(axis=axes_to_compute).compute().tolist()
                if np.isscalar(ends):
                    ends = [ends]
        else:
            if dtype is None:
                dtype = self.array.dtype
            if np.issubdtype(dtype, np.integer):
                if starts is None:
                    starts = [np.iinfo(dtype).min for _ in range(n_channels)]
                if ends is None:
                    ends = [np.iinfo(dtype).max for _ in range(n_channels)]
            elif np.issubdtype(dtype, np.floating):
                if starts is None:
                    starts = [np.finfo(dtype).min for _ in range(n_channels)]
                if ends is None:
                    ends = [np.finfo(dtype).max for _ in range(n_channels)]
            else:
                raise ValueError(f"Unsupported dtype {dtype}")
        return starts, ends

    # ------------------------------------------------------------------
    # Array + axes state
    # ------------------------------------------------------------------
    def set_arraydata(self,
                      array=None,
                      axes: Optional[str] = None,
                      units: Optional[List[Any]] = None,
                      scales: Optional[List[Any]] = None,
                      **kwargs):
        axes = axes or self.img.get_axes()
        units = units or self.img.get_units()
        scales = scales or self.img.get_scales()

        self.axes = axes
        if array is not None:
            self.array = array
            self.ndim = self.array.ndim
            assert len(self.axes) == self.ndim

        self.caxes = ''.join([ax for ax in axes if ax != 'c'])
        if self.array is not None:
            if isinstance(self.array, zarr.Array):
                chunks = self.array.chunks
            elif isinstance(self.array, da.Array):
                chunks = self.array.chunksize
            else:
                raise Exception(f"Array type {type(self.array)} is not supported.")
            self.chunkdict = dict(zip(list(self.axes), chunks))
            self.shapedict = dict(zip(list(self.axes), self.array.shape))
            if 'c' in self.shapedict:
                self._ensure_correct_channels()

        if len(units) == len(self.axes):
            self.unitdict = dict(zip(list(self.axes), units))
        elif len(units) == len(self.caxes):
            self.unitdict = dict(zip(list(self.caxes), units))
        else:
            raise Exception("Unit length is invalid.")

        if len(scales) == len(self.axes):
            self.scaledict = dict(zip(list(self.axes), scales))
        elif len(scales) == len(self.caxes):
            self.scaledict = dict(zip(list(self.caxes), scales))
            self.scaledict['c'] = 1
        else:
            raise Exception("Scale length is invalid")

    # async def read_arraydata(self):
    #     await self.img.read_array_data()
    #     self.set_arraydata(self.img._arraydata)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def scales(self):
        if len(self.scaledict) < len(self.axes):
            return [self.scaledict[ax] for ax in self.caxes]
        elif len(self.scaledict) == len(self.axes):
            return [self.scaledict[ax] for ax in self.axes]
        else:
            raise ValueError

    @property
    def units(self):
        if len(self.unitdict) < len(self.axes):
            return [self.unitdict[ax] for ax in self.caxes]
        elif len(self.unitdict) == len(self.axes):
            return [self.unitdict[ax] for ax in self.axes]
        else:
            raise ValueError

    @property
    def channels(self):
        if self._channels is not None:
            return self._channels
        return self.img.get_channels()

    @property
    def chunks(self):
        return [self.chunkdict[ax] for ax in self.axes]

    # ------------------------------------------------------------------
    # Pyramid + OME-XML
    # ------------------------------------------------------------------
    async def sync_pyramid(self, create_omexml_if_not_exists: bool = False, save_changes = False):
        """Synchronize scale/unit metadata with pyramid and update/save OME-XML.
        Offloads pyramid + file I/O parts to threads.
        """
        if self.pyr is None:
            raise Exception("No pyramid exists.")

        # Update scales/units on pyramid (likely fast, but be safe if it touches disk)
        await asyncio.to_thread(self.pyr.update_scales, **self.scaledict)
        await asyncio.to_thread(self.pyr.update_units, **self.unitdict)

        ################# Critical ###################
        if isinstance(self.img, NGFFImageMeta):
            for idx,new_channel_meta in enumerate(self.channels):
                channel = self.pyr.meta.metadata['omero']['channels'][idx]
                channel.update(new_channel_meta)
                self.pyr.meta.metadata['omero']['channels'][idx] = channel
        else:
            self.pyr.meta.metadata['omero']['channels'] = self.channels

        self.pyr.meta._pending_changes = True
        ##############################################

        if self.omemeta is None:
            # Lazy import in thread to avoid blocking if heavy
            def _create_ome():
                return create_ome_xml(
                    image_shape=self.pyr.base_array.shape,
                    axis_order=self.pyr.axes,
                    pixel_size_x=self.pyr.meta.scaledict.get('0', {}).get('x'),
                    pixel_size_y=self.pyr.meta.scaledict.get('0', {}).get('y'),
                    pixel_size_z=self.pyr.meta.scaledict.get('0', {}).get('z'),
                    pixel_size_t=self.pyr.meta.scaledict.get('0', {}).get('t'),
                    unit_x=self.pyr.meta.unit_dict.get('x'),
                    unit_y=self.pyr.meta.unit_dict.get('y'),
                    unit_z=self.pyr.meta.unit_dict.get('z'),
                    unit_t=self.pyr.meta.unit_dict.get('t'),
                    dtype=str(self.pyr.base_array.dtype),
                    image_name=self.pyr.meta.multiscales.get('name', 'Default image'),
                    channel_names=[channel['label'] for channel in self.channels],
                )

            self.omemeta = await asyncio.to_thread(_create_ome)

        # Update / write OME XML if exists or requested to create
        try:
            if self.pyr.gr is not None and ('OME' in list(self.pyr.gr.keys()) or create_omexml_if_not_exists):
                await self.save_omexml(self.pyr.gr.store.root, overwrite=True)
        except (OSError, AttributeError) as e:
            # OSError for filesystem issues, AttributeError for store without local root
            logger.warning(f"Failed to save OME-XML for pyramid: {e}")
        if save_changes:
            await asyncio.to_thread(self.pyr.meta.save_changes)

    async def create_omemeta(self):
        self.fill_default_meta()
        pixel_size_basemap = self.get_pixel_size_basemap(**self.scaledict)
        unit_basemap = self.get_unit_basemap(**self.unitdict)

        self.omemeta = create_ome_xml(
            image_shape=self.array.shape,
            axis_order=self.axes,
            **pixel_size_basemap,
            **unit_basemap,
            dtype=str(self.array.dtype),
            channel_names=[channel['label'] for channel in self.channels]
        )
        self.pixels = self.omemeta.images[0].pixels
        missing_fields = self.essential_omexml_fields - self.pixels.model_fields_set
        self.pixels.model_fields_set.update(missing_fields)
        self.omemeta.images[0].pixels = self.pixels
        return self

    async def save_omexml(self, base_path: str, overwrite: bool = False):
        # TODO: Update for s3.
        # if self.img.omemeta is None:
        await self.create_omemeta()
        # else:
        #     self.omemeta = self.img.omemeta
        assert self.omemeta is not None, "No ome-xml exists."
        gr = await asyncio.to_thread(zarr.group, base_path)
        try:
            path = os.path.join(gr.store.root, 'OME', 'METADATA.ome.xml')
        except AttributeError as e:
            # Zarr store doesn't have .root attribute (e.g., cloud storage)
            logger.warning(f"Writing OME-XML is currently only possible with local stores: {e}")
            return
        await asyncio.to_thread(gr.create_group, 'OME', overwrite=overwrite)

        def _write_text_file(p: str, text: str):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, 'w', encoding='utf-8') as f:
                f.write(text)

        await asyncio.to_thread(_write_text_file, path, self.omemeta.to_xml())

        if gr.info._zarr_format == 2:
            gr['OME'].attrs["series"] = [self.series]
        else:  # zarr format 3
            gr['OME'].attrs["ome"] = dict(version="0.5", series=[str(self.series)])

    # ------------------------------------------------------------------
    # Array ops (in-memory / lazy with Dask)
    # ------------------------------------------------------------------
    def squeeze(self):
        if all(n > 1 for n in self.array.shape):
            return
        if self.pyr is not None:
            zgroup = self.pyr.meta.zarr_group
            omero_copy = copy.copy(self.pyr.meta.omero)
            self.pyr = self.pyr.squeeze()
            self.pyr.meta.zarr_group = zgroup
            self.pyr.meta.metadata['omero'] = omero_copy
        else:
            omero_copy = None
        if isinstance(self.array, zarr.Array):
            logger.warn(f"Zarr arrays are not supported for squeeze operation.\n"
                        f"Zarr array for the path {self.series_path} is being converted to dask array.")
            array = da.from_array(self.array)
        else:
            array = self.array
        singlet_axes = [ax for ax, size in self.shapedict.items() if size == 1]
        newaxes = ''.join(ax for ax in self.axes if ax not in singlet_axes)
        newunits, newscales = [], []
        assert (len(self.scaledict) - len(self.unitdict)) <= 1
        for ax in self.axes:
            if ax not in singlet_axes:
                if ax in self.unitdict:
                    newunits.append(self.unitdict[ax])
                if ax in self.scaledict:
                    newscales.append(self.scaledict[ax])
        newarray = da.squeeze(array)
        # print(f"newaxes: {newaxes}")
        # print(f"newunits: {newunits}")
        # print(f"newscales: {newscales}")
        self.set_arraydata(newarray, newaxes, newunits, newscales)

    def transpose(self, newaxes: str):
        newaxes = ''.join(ax for ax in newaxes if ax in self.axes)
        new_ids = [self.axes.index(ax) for ax in newaxes]
        newunits, newscales = [], []
        assert (len(self.scaledict) - len(self.unitdict)) <= 1
        for ax in newaxes:
            if ax in self.unitdict:
                newunits.append(self.unitdict[ax])
            if ax in self.scaledict:
                newscales.append(self.scaledict[ax])
        newarray = self.array.transpose(*new_ids)
        self.set_arraydata(newarray, newaxes, newunits, newscales)

    def crop(self,
             trange=None,
             crange=None,
             zrange=None,
             yrange=None,
             xrange=None):
        omero_copy = copy.copy(self.pyr.meta.omero)
        slicedict = {
            't': slice(*trange) if trange is not None else slice(None),
            'c': slice(*crange) if crange is not None else slice(None),
            'z': slice(*zrange) if zrange is not None else slice(None),
            'y': slice(*yrange) if yrange is not None else slice(None),
            'x': slice(*xrange) if xrange is not None else slice(None),
        }
        slicedict = {ax: r for ax, r in slicedict.items() if ax in self.axes}
        ### slice channels:
        if 'c' in slicedict and 'c' in self.axes:
            c_slice = slicedict.get('c')
            channels = [omero_copy['channels'][i] for i in range(c_slice.start, c_slice.stop)]
            self.pyr.meta.omero['channels'] = channels
        ####
        slices = tuple([slicedict[ax] for ax in self.axes])
        logger.info(f"The array with shape {self.array.shape} is cropped to {slicedict}")
        if isinstance(self.array, zarr.Array):
            logger.warn(f"The crop option is only supported for dask arrays.\n"
                        f"Zarr array for the path {self.series_path} is being converted to dask array.")
            array = da.from_array(self.array)
        else:
            array = self.array
        array = array[slices]
        logger.info(f"The cropped array shape: {array.shape}")
        self.set_arraydata(array, self.axes, self.units, self.scales)


    def to_cupy(self):
        try:
            import cupy
        except Exception:
            raise Exception("Cupy not installed but required for this operation.")
        array = self.array.map_blocks(cupy.asarray)
        self.set_arraydata(array, self.axes, self.units, self.scales)

    def split_series(self):
        """Split each series into a separate ArrayManager using set_scene method.
        Create proper root path representing series number."""

    def split(self):
        # TODO: implement as needed
        pass

    def get_autocomputed_chunks(self, dtype=None):
        array_shape = self.array.shape
        dtype = dtype or self.array.dtype
        axes = self.axes
        chunk_shape = autocompute_chunk_shape(array_shape=array_shape, axes=axes, dtype=dtype)
        return chunk_shape


class ChannelIterator:
    """
    Iterator for generating and managing channel colors.

    This class provides a way to iterate through a sequence of channel colors,
    generating new colors in a visually distinct sequence when needed.
    """
    DEFAULT_COLORS = [
        "FF0000",  # Red
        "00FF00",  # Green
        "0000FF",  # Blue
        "FF00FF",  # Magenta
        "00FFFF",  # Cyan
        "FFFF00",  # Yellow
        "FFFFFF",  # White
    ]

    def __init__(self, num_channels=0):
        """
        Initialize the channel iterator.

        Args:
            num_channels: Initial number of channels to pre-generate
        """
        self._channels = []
        self._current_index = 0
        self._generate_channels(num_channels)

    def _generate_channels(self, count):
        """Generate the specified number of unique channel colors."""
        for i in range(len(self._channels), count):
            if i < len(self.DEFAULT_COLORS):
                color = self.DEFAULT_COLORS[i]
            else:
                # Generate a distinct color by distributing hues
                hue = int((i * 137.5) % 360)  # Golden angle for distinct colors
                r, g, b = self._hsv_to_rgb(hue / 360.0, 1.0, 1.0)
                color = f"{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"
            self._channels.append({"label": f"Channel {i + 1}", "color": color})

    @staticmethod
    def _hsv_to_rgb(h, s, v):
        """Convert HSV color space to RGB color space."""
        h = h * 6.0
        i = int(h)
        f = h - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))

        if i == 0:
            return v, t, p
        elif i == 1:
            return q, v, p
        elif i == 2:
            return p, v, t
        elif i == 3:
            return p, q, v
        elif i == 4:
            return t, p, v
        else:
            return v, p, q

    def __iter__(self):
        """Return the iterator object itself."""
        self._current_index = 0
        return self

    def __next__(self):
        """Return the next channel color."""
        if self._current_index >= len(self._channels):
            self._generate_channels(len(self._channels) + 1)

        if self._current_index < len(self._channels):
            result = self._channels[self._current_index]
            self._current_index += 1
            return result
        raise StopIteration

    def get_channel(self, index):
        """
        Get channel color by index.

        Args:
            index: Index of the channel to retrieve

        Returns:
            dict: Channel information with 'label' and 'color' keys
        """
        if index >= len(self._channels):
            self._generate_channels(index + 1)
        return self._channels[index]

    def __len__(self):
        """Return the number of generated channels."""
        return len(self._channels)


class BatchManager:
    def __init__(self,
                 # managers
                 ):
        pass
        # self.managers = managers

    async def init(self,
                   managers
                   ):
        self.managers = managers
        return self

    async def _collect_scaledict(self, **kwargs):
        """
        Retrieves pixel sizes for image dimensions.

        Args:
            **kwargs: Pixel sizes for time, channel, z, y, and x dimensions.

        Returns:
            list: Pixel sizes.
        """
        t = kwargs.get('time_scale', None)
        c = kwargs.get('channel_scale', None)
        y = kwargs.get('y_scale', None)
        x = kwargs.get('x_scale', None)
        z = kwargs.get('z_scale', None)
        fulldict = dict(zip('tczyx', [t, c, z, y, x]))
        final = {key: val for key, val in fulldict.items() if val is not None}
        return final

    async def _collect_unitdict(self, **kwargs):
        """
        Retrieves unit specifications for image dimensions.

        Args:
            **kwargs: Unit values for time, channel, z, y, and x dimensions.

        Returns:
            list: Unit values.
        """
        t = kwargs.get('time_unit', None)
        c = kwargs.get('channel_unit', None)
        y = kwargs.get('y_unit', None)
        x = kwargs.get('x_unit', None)
        z = kwargs.get('z_unit', None)
        fulldict = dict(zip('tczyx', [t, c, z, y, x]))
        final = {key: val for key, val in fulldict.items() if val is not None}
        return final

    async def _collect_chunks(self, **kwargs):  ###
        """
        Retrieves chunk specifications for image dimensions.

        Args:
            **kwargs: Chunk sizes for time, channel, z, y, and x dimensions.

        Returns:
            list: Chunk shape.
        """
        t = kwargs.get('time_chunk', None)
        c = kwargs.get('channel_chunk', None)
        y = kwargs.get('y_chunk', None)
        x = kwargs.get('x_chunk', None)
        z = kwargs.get('z_chunk', None)
        fulldict = dict(zip('tczyx', [t, c, z, y, x]))
        final = {key: val for key, val in fulldict.items() if val is not None}
        return final

    async def fill_default_meta(self):
        for key, manager in self.managers.items():
            manager.fill_default_meta()

    async def squeeze(self):
        for key, manager in self.managers.items():
            manager.squeeze()

    async def to_cupy(self):
        for key, manager in self.managers.items():
            manager.to_cupy()

    async def crop(self,
                   time_range=None,
                   channel_range=None,
                   z_range=None,
                   y_range=None,
                   x_range=None,
                   **kwargs  # placehold
                   ):
        if any([item is not None
                for item in (time_range,
                             channel_range,
                             z_range,
                             y_range,
                             x_range)]):
            for key, manager in self.managers.items():
                manager.crop(time_range, channel_range, z_range, y_range, x_range)

    async def transpose(self, newaxes):
        for key, manager in self.managers.items():
            manager.transpose(newaxes)

    async def sync_pyramids(self):
        pass