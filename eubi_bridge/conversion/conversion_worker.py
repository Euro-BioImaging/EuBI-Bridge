# from concurrent.futures import ProcessPoolExecutor
import copy
from typing import Union
import tensorstore as ts
import zarr, os, asyncio
import numpy as np
from eubi_bridge.conversion.fileset_io import BatchFile, FileSet
from eubi_bridge.utils.convenience import take_filepaths, ChannelMap
from eubi_bridge.utils.metadata_utils import generate_channel_metadata
from eubi_bridge.core.readers import read_single_image_asarray
from eubi_bridge.core.data_manager import ArrayManager
from eubi_bridge.core.writers import write_with_tensorstore_async, _get_or_create_multimeta, _create_zarr_v2_array, CompressorConfig
from eubi_bridge.core.writers import store_multiscale_async
from eubi_bridge.conversion.fileset_io import BatchFile
from eubi_bridge.conversion.aggregative_conversion_base import AggregativeConverter
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

def parse_chunks(manager,
                 **kwargs
                 ):
    default_chunks = dict(
        time_chunk = 1,
        channel_chunk = 1,
        z_chunk = 96,
        y_chunk = 96,
        x_chunk = 96
    )
    output = {}
    for ax in manager.axes:
        if ax == 't':
            output['t'] = _parse_item(kwargs, 'time_chunk', 'time_chunk', default_chunks)
        elif ax == 'c':
            output['c'] = _parse_item(kwargs, 'channel_chunk', 'channel_chunk', default_chunks)
        elif ax == 'z':
            output['z'] = _parse_item(kwargs, 'z_chunk', 'z_chunk', default_chunks)
        elif ax == 'y':
            output['y'] = _parse_item(kwargs, 'y_chunk', 'y_chunk', default_chunks)
        elif ax == 'x':
            output['x'] = _parse_item(kwargs, 'x_chunk', 'x_chunk', default_chunks)
    return tuple([output[key] for key in manager.axes])

def parse_shard_coefs(manager,
                      **kwargs
                      ):
    default_shard_coefs = dict(
        time_shard_coef = 1,
        channel_shard_coef = 1,
        z_shard_coef = 5,
        y_shard_coef = 5,
        x_shard_coef = 5
    )
    output = {}
    for ax in manager.axes:
        if ax == 't':
            output['t'] = _parse_item(kwargs, 'time_shard_coef', 'time_shard_coef', default_shard_coefs)
        elif ax == 'c':
            output['c'] = _parse_item(kwargs, 'channel_shard_coef', 'channel_shard_coef', default_shard_coefs)
        elif ax == 'z':
            output['z'] = _parse_item(kwargs, 'z_shard_coef', 'z_shard_coef', default_shard_coefs)
        elif ax == 'y':
            output['y'] = _parse_item(kwargs, 'y_shard_coef', 'y_shard_coef', default_shard_coefs)
        elif ax == 'x':
            output['x'] = _parse_item(kwargs, 'x_shard_coef', 'x_shard_coef', default_shard_coefs)
    return tuple([output[key] for key in manager.axes])

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
    return tuple([output[key] for key in manager.axes])

def parse_scale_factors(manager,
                        **kwargs
                        ):
    default_scale_factors = dict(
        time_scale_factor = 1,
        channel_scale_factor = 1,
        z_scale_factor = 2,
        y_scale_factor = 2,
        x_scale_factor = 2
    )
    output = {}
    for ax in manager.axes:
        if ax == 't':
            output['t'] = _parse_item(kwargs,'time_scale_factor','time_scale_factor', default_scale_factors)
        elif ax == 'c':
            output['c'] = _parse_item(kwargs, 'channel_scale_factor', 'channel_scale_factor', default_scale_factors)
        elif ax == 'z':
            output['z'] = _parse_item(kwargs, 'z_scale_factor', 'z_scale_factor', default_scale_factors)
        elif ax == 'y':
            output['y'] = _parse_item(kwargs, 'y_scale_factor', 'y_scale_factor', default_scale_factors)
        elif ax == 'x':
            output['x'] = _parse_item(kwargs, 'x_scale_factor', 'x_scale_factor', default_scale_factors)
    return tuple([output[key] for key in manager.axes])

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
    return tuple([output[key] for key in manager.caxes])

def parse_channels(manager,
                   **kwargs
                   ):
    # path = f"/home/oezdemir/PycharmProjects/TIM2025/data/example_images1/pff/00001_01.ome.tiff"
    # path = f"/home/oezdemir/PycharmProjects/TIM2025/data/example_images1/pff/17_03_18.lif"
    # manager = ArrayManager(path, skip_dask=True)
    # await manager.init()
    # manager._channels = manager.channels
    # manager.fix_bad_channels()
    # kwargs = dict(
    #     channel_intensity_limits='from_array',
    #     channel_indices='all',
    # )

    dtype = kwargs.get('dtype', None)
    if dtype is None:
        dtype = manager.array.dtype
    if 'c' not in manager.axes:
        channel_count = 1
    else:
        channel_idx = manager.axes.index('c')
        channel_count = manager.array.shape[channel_idx]
        assert channel_count == len(manager.channels), f"Manager constructed incorrectly!"
    default_channels = generate_channel_metadata(num_channels=channel_count,
                                                 dtype=dtype)
    # import pprint
    # pprint.pprint(manager.channels)
    if manager.channels is not None:
        for idx, channel in enumerate(manager.channels):
            default_channels[idx].update(channel)

    output = copy.deepcopy(default_channels)
    assert 'coefficient' in output[0].keys(), f"Channels parsed incorrectly!"

    channel_indices = kwargs.get('channel_indices', [])

    if channel_indices == 'all':
        channel_indices = list(range(len(output)))
    if not hasattr(channel_indices, '__len__'):
        channel_indices = [channel_indices]
    # print(f"Channel indices: {channel_indices}")
    channel_labels = kwargs.get('channel_labels', None)
    channel_colors = kwargs.get('channel_colors', None)
    if channel_labels in ('auto', None):
        channel_labels = [channel_labels] * len(channel_indices)
    if channel_colors in ('auto', None):
        channel_colors = [channel_colors] * len(channel_indices)

    #######
    channel_intensity_limits = kwargs.get('channel_intensity_limits','from_dtype')
    assert channel_intensity_limits in ('from_dtype', 'from_array'), f"Channel intensity limits must be either 'from_dtype' or 'from_array'"
    #######

    try:
        if np.isnan(channel_indices):
            return output
        elif channel_indices is None:
            return output
        elif channel_indices == []:
            return output
    except:
        pass
    if isinstance(channel_indices, str):
        channel_indices = [i for i in channel_indices.split(',')]
    if isinstance(channel_labels, str):
        channel_labels = [i for i in channel_labels.split(',')]
    if isinstance(channel_colors, str):
        channel_colors = [i for i in channel_colors.split(',')]

    channel_indices = [int(i) for i in channel_indices]
    items = [channel_indices, channel_labels, channel_colors, channel_intensity_limits]
    for idx, item in enumerate(items):
        if not isinstance(item, str) and np.isscalar(item):
            item = [item]
        for i, it in enumerate(item):
            try:
                if np.isnan(it):
                    item[i] = 'auto'
            except:
                pass
        items[idx] = item
    channel_indices, channel_labels, channel_colors, channel_intensity_limits = items

    if not len(channel_indices) == len(channel_labels) == len(channel_colors):
        raise ValueError(f"Channel indices, labels, colors, intensity minima and extrema must have the same length. \n"
                         f"So you need to specify --channel_indices, --channel_labels, --channel_colors, --channel_intensity_extrema with the same number of elements. \n"
                         f"To keep specific labels or colors unchanged, add 'auto'. E.g. `--channel_indices 0,1 --channel_colors auto,red`")
    cm = ChannelMap()

    if len(channel_indices) == 0:
        return output

    from_array = channel_intensity_limits == 'from_array'
    start_intensities, end_intensities = manager.compute_intensity_limits(
                                                    from_array = from_array,
                                                    dtype = dtype)
    mins, maxes = manager.compute_intensity_extrema(dtype = dtype)
    # pprint.pprint(output)
    for idx in range(len(channel_indices)):
        channel_idx = channel_indices[idx]
        if channel_idx >= channel_count:
            raise ValueError(f"Channel index {channel_idx} is out of range -> {0}:{channel_count - 1}")
        current_channel = output[channel_idx]
        if channel_labels[idx] not in (None, 'auto'):
            current_channel['label'] = channel_labels[idx]
        colorname = channel_colors[idx]
        if colorname not in (None, 'auto'):
            current_channel['color'] = cm[colorname] if cm[colorname] is not None else colorname
        ###--------------------------------------------------------------------------------###
        # window = {
        #     'min': mins[channel_idx],
        #     'max': maxes[channel_idx],
        #     'start': start_intensities[channel_idx],
        #     'end': end_intensities[channel_idx]
        # }
        # current_channel['window'] = window
        ###--------------------------------------------------------------------------------###
        # Add the parameters that are currently hard-coded
        current_channel['coefficient'] = 1
        current_channel['active'] = True
        current_channel['family'] = "linear"
        current_channel['inverted'] = False
        ###--------------------------------------------------------------------------------###
        output[channel_idx] = current_channel
    # The channel intensity window not controlled by the channel_indices parameter.
    for channel_idx in range(len(output)):
        current_channel = output[channel_idx]
        window = {
            'min': mins[channel_idx],
            'max': maxes[channel_idx],
            'start': start_intensities[channel_idx],
            'end': end_intensities[channel_idx]
        }
        current_channel['window'] = window
    return output

# path = f"/home/oezdemir/PycharmProjects/TIM2025/data/example_images1/pff/PK2_ATH_5to20_20240705_MID_01.czi"
#
# path = f"/home/oezdemir/PycharmProjects/TIM2025/data/example_images1/pff/filament.tif"
#
# path = f"/home/oezdemir/PycharmProjects/TIM2025/data/example_images1/pff/00001_01.ome.tiff"
#
# path = f"/home/oezdemir/PycharmProjects/TIM2025/data/example_images1/pff/17_03_18.lif"
# man = ArrayManager(path, skip_dask=True)
# await man.init()
# # await man.set_scene(30)
# await man.load_scenes((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21))
# 
# scene = man.loaded_scenes['/home/oezdemir/PycharmProjects/TIM2025/data/example_images1/pff/17_03_18.lif_8']
# scene.img.pixels
# scene.img.get_channels()
# img = scene.img
# 
# img.pixels.id
# img._series

# for m in man.loaded_scenes.values():
    # print(m.array.shape)
    # print(m.channels)
# await man.set_scene(4)
# man.channels

# man._channels = man.channels
# man.fix_bad_channels()

# import dask.array as da
#
# chns = parse_channels(man,
#                    channel_intensity_limits='from_dtype',
#                    channel_indices='all')
# man._channels = chns


async def unary_worker(input_path: Union[str, ArrayManager],
                         output_path: str,
                         **kwargs):
    import pprint
    max_concurrency = kwargs.get('max_concurrency', 4)
    compute_batch_size = kwargs.get('compute_batch_size', 4)
    memory_limit_per_batch = kwargs.get('memory_limit_per_batch', 1024)
    series = kwargs.get('scene_index', 'all')

    if not isinstance(input_path, ArrayManager):
        manager = ArrayManager(
            input_path,
            series=0,
            metadata_reader=kwargs.get('metadata_reader', 'bfio'),
            skip_dask=kwargs.get('skip_dask', True),
        )
        await manager.init()
        # pprint.pprint(f"1: {manager.channels}")
        manager.fill_default_meta()
        # pprint.pprint(f"2: {manager.channels}")
        await manager.load_scenes(scene_indices=series)
        # for man in manager.loaded_scenes.values():
            # pprint.pprint(f"3: {man.channels}")
    else:
        manager = input_path
    # Semaphore to limit concurrent scenes
    sem = asyncio.Semaphore(max_concurrency)
    async def run_single_scene(man, output_path):
        async with sem:
            man.fill_default_meta()
            # Add missing metadata to channels:
            man._channels = parse_channels(man, 
                                           channel_indices='all', 
                                           channel_intensity_limits = 'from_dtype')
            # import pprint
            man.fix_bad_channels()
            # pprint.pprint(man.channels)
            ### Additional processing if needed
            if kwargs.get('squeeze'):
                man.squeeze()
            cropping_slices = [kwargs.get(key) for key in kwargs
                               if key in ('time_range', 'channel_range',
                                           'z_range', 'y_range',
                                           'x_range')]
            if any(cropping_slices):
                man.crop(*cropping_slices)
            ###---------------------------------###
            await store_multiscale_async(
                arr=man.array,
                dtype = kwargs.get('dtype', None),
                output_path=output_path,
                zarr_format = kwargs.get('zarr_format', 2),
                axes=man.axes,
                scales=parse_scales(man, **kwargs),
                units=parse_units(man, **kwargs),
                channel_meta = parse_channels(man, **dict(kwargs, channel_intensity_limits = 'from_dtype')), # man.channels,
                auto_chunk=kwargs.get('auto_chunk', True),
                output_chunks=parse_chunks(man, **kwargs),
                output_shard_coefficients=parse_shard_coefs(man, **kwargs),
                overwrite=kwargs.get('overwrite', True),
                n_layers=kwargs.get('n_layers', 3),
                min_dimension_size=kwargs.get('min_dimension_size', 64),
                scale_factors=parse_scale_factors(man, **kwargs),
                max_concurrency=max_concurrency,
                compute_batch_size=compute_batch_size,
                memory_limit_per_batch=memory_limit_per_batch,
            )
            ###--------Handle channel metadata at the end for efficiency---###
            if kwargs.get('channel_intensity_limits', 'from_array'):
                chman = ArrayManager(output_path)
                await chman.init()
                channels = parse_channels(chman, **kwargs)
                meta = chman.pyr.meta
                meta.metadata['omero']['channels'] = channels
                meta._pending_changes = True
                meta.save_changes()
            ###------------------------------------------------------------###
            await man.save_omexml(output_path, overwrite=True)

    tasks = []
    for man in manager.loaded_scenes.values(): ### Careful here!
        # pprint.pprint(man.channels)
        if man.series == 0 and man.img.n_scenes == 1:
            output_path_ = f"{output_path}/{os.path.basename(man.series_path).split('.')[0]}.zarr"
        else:
            output_path_ = f"{output_path}/{os.path.basename(man.series_path).split('.')[0]}_{man.series}.zarr"
        tasks.append(asyncio.create_task(run_single_scene(man, output_path_)))

    await asyncio.gather(*tasks)


async def aggregative_worker(manager: ArrayManager,
                         output_path: str,
                         **kwargs):
    max_concurrency = kwargs.get('max_concurrency', 4)
    compute_batch_size = kwargs.get('compute_batch_size', 4)
    memory_limit_per_batch = kwargs.get('memory_limit_per_batch', 1024)
    series = kwargs.get('series', 0) # Here you can only load one scene per file
    if not isinstance(series, int):
        raise TypeError(f"Aggregative conversion does not support multiple series at the moment. \n"
                        f"Please specify an integer as a single series index.")

    # Semaphore to limit concurrent scenes
    sem = asyncio.Semaphore(max_concurrency)
    async def run_single_scene(man, output_path):
        async with sem:
            man.fill_default_meta()
            await store_multiscale_async(
                arr=man.array,
                output_path=output_path,
                zarr_format = kwargs.get('zarr_format', 2),
                axes=man.axes,
                scales=parse_scales(man, **kwargs),
                units=parse_units(man, **kwargs),
                channel_meta=man.channels if man.channels is not None else 'auto',
                auto_chunk=kwargs.get('auto_chunk', True),
                output_chunks=parse_chunks(man, **kwargs),
                output_shard_coefficients=parse_shard_coefs(man, **kwargs),
                overwrite=kwargs.get('overwrite', True),
                n_layers=kwargs.get('n_layers', 3),
                min_dimension_size=kwargs.get('min_dimension_size', 64),
                scale_factors=parse_scale_factors(man, **kwargs),
                max_concurrency=max_concurrency,
                compute_batch_size=compute_batch_size,
                memory_limit_per_batch=memory_limit_per_batch,
            )
            await man.save_omexml(output_path, overwrite=True)


    tasks = []
    # for man in manager.loaded_scenes.values():
    output_path_ = f"{output_path}.zarr"
    tasks.append(asyncio.create_task(run_single_scene(manager,
                                                      output_path_)))

    await asyncio.gather(*tasks)



# Wrap your existing async function so it can run in a subprocess
def unary_worker_sync(input_path,
                        output_path,
                        kwargs:dict):
    # Run the event loop inside the process
    return asyncio.run(unary_worker(input_path,
                                      output_path,
                                      **kwargs))


def aggregative_worker_sync(input_path,
                        output_path,
                        kwargs:dict):
    # Run the event loop inside the process
    return asyncio.run(aggregative_worker(input_path,
                                      output_path,
                                      **kwargs))



