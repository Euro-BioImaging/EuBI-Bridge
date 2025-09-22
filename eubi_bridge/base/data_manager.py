import zarr, natsort

import shutil, time, os, zarr, psutil, dask, gc, json
import numpy as np, os, glob, tempfile, importlib

from ome_types.model import OME, Image, Pixels, Channel  # TiffData, Plane
from ome_types.model import PixelType, Pixels_DimensionOrder, UnitsLength, UnitsTime

from typing import Tuple

from dask import array as da
from pathlib import Path
from typing import Union

from eubi_bridge.ngff.multiscales import Pyramid
from eubi_bridge.ngff.defaults import unit_map, scale_map, default_axes
from eubi_bridge.utils.convenience import sensitive_glob, is_zarr_group, is_zarr_array, take_filepaths, \
    autocompute_chunk_shape
from eubi_bridge.base.readers import read_metadata_via_bioio_bioformats, read_metadata_via_extension, \
    read_metadata_via_bfio
from eubi_bridge.utils.logging_config import get_logger
from eubi_bridge.ngff.defaults import default_axes, unit_map, scale_map

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
                 series,
                 meta_reader="bioio"
                 ):
        if path.endswith('ome') or path.endswith('xml'):
            from ome_types import OME
            omemeta = OME().from_xml(path)
        else:
            if meta_reader == 'bioio':
                # Try to read the metadata via bioio
                try:
                    omemeta = read_metadata_via_extension(path, series=series)
                except:
                    # If not found, try to read the metadata via bioformats
                    omemeta = read_metadata_via_bioio_bioformats(path, series=series)
            elif meta_reader == 'bfio':
                try:
                    omemeta = read_metadata_via_bfio(path)  # don't pass series, will be handled afterwards
                except:
                    # If not found, try to read the metadata via bioformats
                    omemeta = read_metadata_via_bioio_bioformats(path, series=series)
            else:
                raise ValueError(f"Unsupported metadata reader: {meta_reader}")
        if series is None:
            series = 0
        images = [omemeta.images[series]]
        omemeta.images = images
        self.omemeta = omemeta
        self.pixels = self.omemeta.images[0].pixels
        missing_fields = self.essential_omexml_fields - self.pixels.model_fields_set
        self.pixels.model_fields_set.update(missing_fields)
        self.omemeta.images[0].pixels = self.pixels
        self.pyr = None
        # self._channels = None

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

    def get_channels(self):
        if not hasattr(self.pixels, 'channels'):
            return None
        if len(self.pixels.channels) == 0:
            return None
        ###
        if len(self.pixels.channels) < self.pixels.size_c:
            chn = ChannelIterator(num_channels=self.pixels.size_c)
            channels = chn._channels
        elif len(self.pixels.channels) == self.pixels.size_c:
            channels = []
            for _, channel in enumerate(self.pixels.channels):
                color = channel.color.as_hex().upper()
                color = expand_hex_shorthand(color)
                name = channel.name
                channels.append(dict(
                    label=name,
                    color=color
                ))
        return channels


class TIFFImageMeta:
    essential_omexml_fields = {
        "physical_size_x", "physical_size_x_unit",
        "physical_size_y", "physical_size_y_unit",
        "physical_size_z", "physical_size_z_unit",
        "time_increment", "time_increment_unit",
        "size_x", "size_y", "size_z", "size_t", "size_c"
    }

    def __init__(self,
                 path,
                 series,
                 meta_reader="bioio",
                 **kwargs
                 ):
        if not path.endswith('.tif') and not path.endswith('.tiff'):
            raise Exception(f"The given path does not contain a TIFF file.")

        # path = f"/home/oezdemir/PycharmProjects/TIM2025/data/example_images/pff/filament.tif"
        self.path = path
        import tifffile
        tif = tifffile.TiffFile(path)
        self.tiffzarrstore = tif.aszarr()
        self._zarrdata = zarr.open(self.tiffzarrstore, mode='r')
        self._zarrmeta = self.tiffzarrstore._data[series]
        self._meta = tif.series[series]
        self._pff = PFFImageMeta(path, series, meta_reader)
        self.omemeta = self._pff.omemeta
        self.pixels = self.omemeta.images[0].pixels
        missing_fields = self.essential_omexml_fields - self.pixels.model_fields_set
        self.pixels.model_fields_set.update(missing_fields)
        self.omemeta.images[0].pixels = self.pixels
        self.pyr = self._pff.pyr

    def get_axes(self):
        return self._meta.axes.lower()

    def get_scaledict(self):
        scaledict = self._pff.get_scaledict()
        axes = self.get_axes()
        return {ax: scaledict[ax]
                for ax in axes
                if ax in scaledict
                }

    def get_scales(self):
        scaledict = self.get_scaledict()
        caxes = [ax for ax in self.get_axes() if ax != 'c']
        return [scaledict[ax] for ax in caxes]

    def get_unitdict(self):
        unitdict = self._pff.get_unitdict()
        caxes = [ax for ax in self.get_axes() if ax != 'c']
        # print(caxes, self.get_axes())
        return {ax: unitdict[ax]
                for ax in caxes
                if ax in unitdict
                }

    def get_units(self):
        unitdict = self.get_unitdict()
        caxes = [ax for ax in self.get_axes() if ax != 'c']
        return [unitdict[ax] for ax in caxes]

    def get_channels(self):
        return self._pff.get_channels()


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


class NGFFImageMeta:
    def __init__(self,
                 path
                 ):
        if is_zarr_group(path):
            self.pyr = Pyramid().from_ngff(path)
            meta = self.pyr.meta
            self._meta = meta
            self._base_path = self._meta.resolution_paths[0]
        else:
            raise Exception(f"The given path does not contain an NGFF group.")

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


class H5ImageMeta:
    essential_omexml_fields = {
        "physical_size_x", "physical_size_x_unit",
        "physical_size_y", "physical_size_y_unit",
        "physical_size_z", "physical_size_z_unit",
        "time_increment", "time_increment_unit",
        "size_x", "size_y", "size_z", "size_t", "size_c"
    }

    def __init__(self,
                 path,
                 series=0,
                 meta_reader="bioio",  # placeholder
                 **kwargs
                 ):

        if not path.endswith('.h5'):
            raise Exception(f"The given path does not contain a TIFF file.")
        if series is None:
            series = 0
        self.path = path
        import h5py
        f = h5py.File(path)
        dset_name = list(f.keys())[series]
        ds = f[dset_name]
        self._ds = ds
        self._attrs = dict(ds.attrs)

        # self.omemeta = self._pff.omemeta
        # self.pixels = self.omemeta.images[0].pixels
        # missing_fields = self.essential_omexml_fields - self.pixels.model_fields_set
        # self.pixels.model_fields_set.update(missing_fields)
        # self.omemeta.images[0].pixels = self.pixels
        # self.pyr = self._pff.pyr
        self.pyr = None

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

    def get_scales(self):
        scaledict = self.get_scaledict()
        caxes = [ax for ax in self.get_axes() if ax != 'c']
        return [scaledict[ax] for ax in caxes]

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

    def get_units(self):
        unitdict = self.get_unitdict()
        caxes = [ax for ax in self.get_axes() if ax != 'c']
        return [unitdict[ax] for ax in caxes]

    def get_channels(self):
        return []


# path = f"/home/oezdemir/PycharmProjects/TIM2025/data/h5/ilastik_img/Patient1_001.h5"
# h5 = H5ImageMeta(path, series = 0)
# h5.get_axes()
# h5.get_scales()
# h5.get_units()

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


class NGFFImageMeta:
    def __init__(self,
                 path
                 ):
        if is_zarr_group(path):
            self.pyr = Pyramid().from_ngff(path)
            meta = self.pyr.meta
            self._meta = meta
            self._base_path = self._meta.resolution_paths[0]
        else:
            raise Exception(f"The given path does not contain an NGFF group.")

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


class ArrayManager:
    essential_omexml_fields = {
        "physical_size_x", "physical_size_x_unit",
        "physical_size_y", "physical_size_y_unit",
        "physical_size_z", "physical_size_z_unit",
        "time_increment", "time_increment_unit",
        "size_x", "size_y", "size_z", "size_t", "size_c"
    }

    def __init__(self,
                 path: Union[str, Path] = None,
                 series: int = None,
                 metadata_reader='bfio',  # bfio or aicsimageio
                 skip_dask=False,
                 **kwargs
                 ):
        self.path = path
        self.series = series
        if series is not None:
            assert isinstance(self.series, (int,
                                            str)), f"The series parameter must be either an integer or string. Selection of multiple series from the same image is currently not supported."
        if self.series is None:
            self.series = 0
            self._seriesattrs = ""
        else:
            self._seriesattrs = self.series

        self._meta_reader = metadata_reader
        self.omemeta = None
        self._skip_dask = skip_dask

        if not path is None:
            if is_zarr_group(path):
                self.img = NGFFImageMeta(self.path)
            elif self.path.endswith('h5'):
                self.img = H5ImageMeta(self.path, self.series, self._meta_reader)
            elif not self._skip_dask:
                self.img = PFFImageMeta(self.path, self.series, self._meta_reader)
            else:
                if self.path.endswith('tif') or self.path.endswith('tiff'):
                    self.img = TIFFImageMeta(self.path, self.series, self._meta_reader)
                else:
                    logger.warning(f"The given path does not contain a TIFF file. Using PFFImageMeta.")
                    self.img = PFFImageMeta(self.path, self.series, self._meta_reader)

        self.axes = self.img.get_axes()
        self.array = None
        self.pyr = self.img.pyr
        self._channels = None
        if hasattr(self.img, '_zarrdata'):
            self.set_arraydata(self.img._zarrdata)
        else:
            self.set_arraydata()

    def fill_default_meta(self):
        if self.array is None:
            raise Exception(f"Array is missing. An array needs to be assigned.")
        new_scaledict = {}
        new_unitdict = {}
        values = list(self.scaledict.values())
        if not None in values:
            return

        for ax, value in self.scaledict.items():
            if value is None:
                if (ax == 'z' or ax == 'y') and self.scaledict['x'] is not None:
                    new_scaledict[ax] = self.scaledict['x']
                    new_unitdict[ax] = self.unitdict['x']
                else:
                    new_scaledict[ax] = scale_map[ax]
                    new_unitdict[ax] = unit_map[ax]
            else:
                if ax in self.scaledict.keys():
                    new_scaledict[ax] = self.scaledict[ax]
                if ax in self.unitdict.keys():
                    new_unitdict[ax] = self.unitdict[ax]

        new_units = [new_unitdict[ax] for ax in self.axes if ax in new_unitdict]
        new_scales = [new_scaledict[ax] for ax in self.axes if ax in new_scaledict]

        self.set_arraydata(self.array, self.axes, new_units, new_scales)
        return self

    def get_pixel_size_basemap(self,
                               t=1,
                               z=1,
                               y=1,
                               x=1,
                               **kwargs
                               ):
        return {
            'pixel_size_t': t,
            'pixel_size_z': z,
            'pixel_size_y': y,
            'pixel_size_x': x
        }

    def get_unit_basemap(self,
                         t='second',
                         z='micrometer',
                         y='micrometer',
                         x='micrometer',
                         **kwargs
                         ):
        return {
            'unit_t': t,
            'unit_z': z,
            'unit_y': y,
            'unit_x': x
        }

    def update_meta(self,
                    new_scaledict={},
                    new_unitdict={},
                    # new_channels = [],
                    ):

        scaledict = self.img.get_scaledict()
        for key, val in new_scaledict.items():
            if key in scaledict.keys() and val is not None:
                scaledict[key] = val

        if 'c' in scaledict:
            scales = [scaledict[ax] for ax in self.axes]
        else:
            scales = [scaledict[ax] for ax in self.caxes]

        unitdict = self.img.get_unitdict()
        for key, val in new_unitdict.items():
            if key in unitdict.keys() and val is not None:
                unitdict[key] = val

        if 'c' in unitdict:
            units = [expand_units(unitdict[ax]) for ax in self.axes]
        else:
            units = [expand_units(unitdict[ax]) for ax in self.caxes]

        self.set_arraydata(array=self.array,
                           axes=self.axes,
                           units=units,
                           scales=scales)

    def _ensure_correct_channels(self):
        if self.array is None:
            return
        if self.channels is None:
            return
        shapedict = dict(zip(list(self.axes), self.array.shape))
        csize = shapedict['c']
        channelsize = len(self.channels)

        if channelsize > csize:
            self._channels = [channel for
                              channel in self.channels
                              if channel['label'] is not None
                              ]

    def fix_bad_channels(self):
        ### Update channel labels
        chn = ChannelIterator()
        for i, channel in enumerate(self.channels):
            if channel['label'] is None:
                channel = chn.__next__()
            self.channels[i] = channel

    def set_arraydata(self,
                      array=None,
                      axes=None,
                      units=None,
                      scales=None,
                      channels=None,
                      **kwargs  # placehold
                      ):

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
            raise Exception(f"Unit length is invalid.")
        if len(scales) == len(self.axes):
            self.scaledict = dict(zip(list(self.axes), scales))
        elif len(scales) == len(self.caxes):
            self.scaledict = dict(zip(list(self.caxes), scales))
            self.scaledict['c'] = 1
        else:
            raise Exception(f"Scale length is invalid")

        if channels is not None:
            self._channels = channels

    @property
    def scales(self):
        if self.scaledict.__len__() < len(self.axes):
            return [self.scaledict[ax] for ax in self.caxes]
        elif self.scaledict.__len__() == len(self.axes):
            return [self.scaledict[ax] for ax in self.axes]
        else:
            raise ValueError

    @property
    def units(self):
        if self.unitdict.__len__() < len(self.axes):
            return [self.unitdict[ax] for ax in self.caxes]
        elif self.unitdict.__len__() == len(self.axes):
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

    def sync_pyramid(self,
                     create_omexml_if_not_exists=False
                     ):  ### TODO: reconsider for improvement
        ### This is only to be used to update pyramidal metadata in place.
        """
        Synchronizes the scale and unit metadata with the Pyramid (if a Pyramid exists).
        The scale metadata is recalculated for all layers based on the shape of each layer.
        Also updates the ome-xml metadata to the pyramid.
        :return:
        """
        if self.pyr is None:
            raise Exception(f"No pyramid exists.")

        self.pyr.update_scales(**self.scaledict,
                               # hard=True
                               )
        self.pyr.update_units(**self.unitdict,
                              # hard=True
                              )
        if self.omemeta is None:
            self.omemeta = create_ome_xml(
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
                channel_names=[channel['label'] for channel in self.channels]
            )
        if 'OME' in list(self.pyr.gr.keys()) or create_omexml_if_not_exists:
            # If OME-XML exists in the pyramid, update it.
            # Otherwise create a new ome-xml only if create_omexml_if_not_exists is True
            # Otherwise do not create a new ome-xml
            self.save_omexml(self.pyr.gr.store.root, overwrite=True)
        self.pyr.meta.save_changes()

    def create_omemeta(self):
        self.fill_default_meta()

        pixel_size_basemap = self.get_pixel_size_basemap(
            **self.scaledict
        )
        unit_basemap = self.get_unit_basemap(
            **self.unitdict
        )
        self.omemeta = create_ome_xml(image_shape=self.array.shape,
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

    def save_omexml(self,
                    base_path: str,
                    overwrite=False
                    ):
        assert self.omemeta is not None, f"No ome-xml exists."
        gr = zarr.group(base_path)
        gr.create_group('OME', overwrite=overwrite)

        path = os.path.join(gr.store.root, 'OME/METADATA.ome.xml')

        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.omemeta.to_xml())

        if gr.info._zarr_format == 2:
            gr['OME'].attrs["series"] = [self._seriesattrs]
        else:  # zarr format 3
            gr['OME'].attrs["ome"] = dict(version="0.5",
                                          series=[str(self._seriesattrs)]
                                          )

    def squeeze(self):
        singlet_axes = [ax for ax, size in self.shapedict.items() if size == 1]
        newaxes = ''.join(ax for ax in self.axes if ax not in singlet_axes)
        newunits, newscales = [], []
        assert (self.scaledict.__len__() - self.unitdict.__len__()) <= 1
        for ax in self.axes:
            if ax not in singlet_axes:
                if ax in self.unitdict.keys():
                    newunits.append(self.unitdict[ax])
                if ax in self.scaledict.keys():
                    newscales.append(self.scaledict[ax])
        newarray = da.squeeze(self.array)
        self.set_arraydata(newarray, newaxes, newunits, newscales)

    def transpose(self, newaxes):
        newaxes = ''.join(ax for ax in newaxes if ax in self.axes)
        new_ids = [self.axes.index(ax) for ax in newaxes]
        newunits, newscales = [], []
        assert (self.scaledict.__len__() - self.unitdict.__len__()) <= 1
        for ax in newaxes:
            if ax in self.unitdict:
                newunits.append(self.unitdict[ax])
            if ax in self.scaledict.keys():
                newscales.append(self.scaledict[ax])
        newarray = self.array.transpose(*new_ids)
        self.set_arraydata(newarray, newaxes, newunits, newscales)

    def crop(self,
             trange=None,
             crange=None,
             zrange=None,
             yrange=None,
             xrange=None,
             ):
        slicedict = {
            't': slice(*trange) if trange is not None else slice(None),
            'c': slice(*crange) if crange is not None else slice(None),
            'z': slice(*zrange) if zrange is not None else slice(None),
            'y': slice(*yrange) if yrange is not None else slice(None),
            'x': slice(*xrange) if xrange is not None else slice(None),
        }
        slicedict = {ax: r for ax, r in slicedict.items() if ax in self.axes}
        slices = tuple([slicedict[ax] for ax in self.axes])
        array = self.array[slices]
        self.set_arraydata(array, self.axes, self.units, self.scales)

    def to_cupy(self):
        try:
            import cupy
        except:
            raise Exception("Cupy not installed but required for this operation.")
        array = self.array.map_blocks(cupy.asarray)
        self.set_arraydata(array, self.axes, self.units, self.scales)

    def split(self):  ###TODO
        pass

    def get_autocomputed_chunks(self,
                                dtype=None
                                ):
        array_shape = self.array.shape
        dtype = dtype or self.array.dtype
        axes = self.axes
        chunk_shape = autocompute_chunk_shape(array_shape=array_shape,
                                              axes=axes,
                                              dtype=dtype)
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
                 managers
                 ):
        self.managers = managers

    def _collect_scaledict(self, **kwargs):
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

    def _collect_unitdict(self, **kwargs):
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

    def _collect_chunks(self, **kwargs):  ###
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

    def fill_default_meta(self):
        for key, manager in self.managers.items():
            manager.fill_default_meta()

    def squeeze(self):
        for key, manager in self.managers.items():
            manager.squeeze()

    def to_cupy(self):
        for key, manager in self.managers.items():
            manager.to_cupy()

    def crop(self,
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

    def transpose(self, newaxes):
        for key, manager in self.managers.items():
            manager.transpose(newaxes)

    def sync_pyramids(self):
        pass