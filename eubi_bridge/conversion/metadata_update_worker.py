# from concurrent.futures import ProcessPoolExecutor
import copy
from typing import Union
import tensorstore as ts
import zarr, os, asyncio
import numpy as np
from eubi_bridge.conversion.fileset_io import BatchFile, FileSet
from eubi_bridge.utils.convenience import take_filepaths, ChannelMap, is_zarr_group
from eubi_bridge.core.readers import read_single_image_asarray
from eubi_bridge.core.data_manager import ArrayManager
from eubi_bridge.core.writers import write_with_tensorstore_async, _get_or_create_multimeta, _create_zarr_v2_array, CompressorConfig
from eubi_bridge.core.writers import store_multiscale_async
from eubi_bridge.conversion.fileset_io import BatchFile
from eubi_bridge.conversion.aggregative_conversion_base import AggregativeConverter
from eubi_bridge.utils.metadata_utils import generate_channel_metadata, parse_channels
import time
import multiprocessing as mp

from eubi_bridge.utils.logging_config import get_logger
logger = get_logger(__name__)


def _parse_item(kwargs, item_type, item_symbol, defaultitems):
    item = kwargs.get(item_type, None)
    if item is None:
        return defaultitems[item_symbol]
    elif np.isnan(item):
        return defaultitems[item_symbol]
    else:
        try:
            return item
        except:
            raise ValueError(f"Invalid item {item} for {item_type}")

def parse_scales(manager,
                 **kwargs
                 ):

    default_scales = manager.scaledict
    output = {}
    for ax in manager.axes:
        if ax == 't':
            output['t'] = _parse_item(kwargs, 'time_scale', 't', default_scales)
        elif ax == 'c':
            output['c'] = _parse_item(kwargs, 'channel_scale', 'c', default_scales)
        elif ax == 'z':
            output['z'] = _parse_item(kwargs, 'z_scale', 'z', default_scales)
        elif ax == 'y':
            output['y'] = _parse_item(kwargs, 'y_scale', 'y', default_scales)
        elif ax == 'x':
            output['x'] = _parse_item(kwargs, 'x_scale', 'x', default_scales)
    return output

def parse_units(manager,
                **kwargs
                ):
    default_units = manager.unitdict
    output = {}
    for ax in manager.axes:
        if ax == 't':
            output['t'] = _parse_item(kwargs, 'time_unit', 't', default_units)
        elif ax == 'c':
            #output['c'] = kwargs.get('channel_unit', default_units['c'])
            pass
        elif ax == 'z':
            output['z'] = _parse_item(kwargs, 'z_unit', 'z', default_units)
        elif ax == 'y':
            output['y'] = _parse_item(kwargs, 'y_unit', 'y', default_units)
        elif ax == 'x':
            output['x'] = _parse_item(kwargs, 'x_unit', 'x', default_units)
    return output


async def update_worker(input_path: Union[str, ArrayManager],
                         **kwargs):

    max_concurrency = kwargs.get('max_concurrency', 4)
    compute_batch_size = kwargs.get('compute_batch_size', 4)
    memory_limit_per_batch = kwargs.get('memory_limit_per_batch', 1024)
    series = kwargs.get('series', 'all')

    if not isinstance(input_path, ArrayManager):
        from eubi_bridge.utils.convenience import is_zarr_group
        if not is_zarr_group(input_path):
            raise Exception(f"Metadata update only works with OME-Zarr datasets.")
        manager = ArrayManager(
            input_path,
            series=0,
            metadata_reader=kwargs.get('metadata_reader', 'bfio'),
            skip_dask=kwargs.get('skip_dask', True),
        )
        await manager.init()
        # print(f"Conversion initialized for {manager.path}")
        manager.fill_default_meta()
    else:
        manager = input_path

    manager.fill_default_meta()
    manager._channels = parse_channels(manager, **kwargs)
    manager.fix_bad_channels()
    manager.update_meta(new_scaledict = parse_scales(manager, **kwargs),
                    new_unitdict = parse_units(manager, **kwargs)
                    )
    await manager.sync_pyramid(True)


def update_worker_sync(input_path,
                        kwargs:dict):
    # Run the event loop inside the process
    return asyncio.run(update_worker(input_path,
                                      **kwargs))