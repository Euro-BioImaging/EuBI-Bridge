# os.environ["TENSORSTORE_LOCK_DISABLE"] = "1"
import concurrent.futures
import itertools
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import dask
import numcodecs
import numpy as np
import s3fs
import tensorstore as ts
import zarr
from dask import delayed
from distributed import get_client
from zarr import codecs
from zarr.storage import LocalStore

### internal imports
from eubi_bridge.ngff.multiscales import NGFFMetadataHandler, Pyramid
from eubi_bridge.utils.array_utils import (autocompute_chunk_shape,
                                           compute_chunk_batch,
                                           get_chunk_shape,
                                           get_chunksize_from_array)
from eubi_bridge.utils.logging_config import get_logger
from eubi_bridge.utils.path_utils import is_zarr_group

# import logging, warnings

logger = get_logger(__name__)

# logging.getLogger('distributed.diskutils').setLevel(logging.CRITICAL)

# logging.basicConfig(level=logging.INFO,
#                     stream=sys.stdout,
#                     force=True)

ZARR_V2 = 2
ZARR_V3 = 3
DEFAULT_DIMENSION_SEPARATOR = "/"
DEFAULT_COMPRESSION_LEVEL = 5
DEFAULT_COMPRESSION_ALGORITHM = "zstd"


@dataclass
class CompressorConfig:
    name: str = 'blosc'
    params: dict = None

    def __post_init__(self):
        self.params = self.params or {}

def autocompute_color(channel_ix: int):
    default_colors = [
        "FF0000",  # Red
        "00FF00",  # Green
        "0000FF",  # Blue
        "FF00FF",  # Magenta
        "00FFFF",  # Cyan
        "FFFF00",  # Yellow
        "FFFFFF",  # White
    ]
    color = default_colors[channel_ix] if channel_ix < len(default_colors) else f"{channel_ix * 40 % 256:02X}{channel_ix * 85 % 256:02X}{channel_ix * 130 % 256:02X}"
    return color

def create_zarr_array(directory: Union[Path, str, zarr.Group],
                      array_name: str,
                      shape: Tuple[int, ...],
                      chunks: Tuple[int, ...],
                      dtype: Any,
                      overwrite: bool = False) -> zarr.Array:
    """Create a new Zarr array in the specified directory or group.
    
    Parameters
    ----------
    directory : Union[Path, str, zarr.Group]
        Directory path or Zarr group where the array will be created.
    array_name : str
        Name of the array to create.
    shape : Tuple[int, ...]
        Shape of the array.
    chunks : Tuple[int, ...]
        Chunk shape for the array.
    dtype : Any
        Data type of the array.
    overwrite : bool, optional
        If True, overwrite existing array with the same name. Default is False.
        
    Returns
    -------
    zarr.Array
        The created Zarr array.
    """
    chunks = tuple(np.minimum(shape, chunks))

    if not isinstance(directory, zarr.Group):
        path = os.path.join(directory, array_name)
        dataset = zarr.create(shape=shape,
                              chunks=chunks,
                              dtype=dtype,
                              store=path,
                              dimension_separator='/',
                              overwrite=overwrite)
    else:
        dataset = directory.create(name=array_name,
                                   shape=shape,
                                   chunks=chunks,
                                   dtype=dtype,
                                   dimension_separator='/',
                                   overwrite=overwrite)
    return dataset


def get_regions(array_shape: Tuple[int, ...],
                region_shape: Tuple[int, ...],
                as_slices: bool = False) -> list:
    """Generate regions for tiled access to an array.
    
    Divides an array into regions of specified size, returning either
    coordinate tuples or Python slice objects for each region.
    
    Parameters
    ----------
    array_shape : Tuple[int, ...]
        Shape of the full array.
    region_shape : Tuple[int, ...]
        Shape of each region/tile.
    as_slices : bool, optional
        If True, return regions as slice objects. If False, return as
        coordinate tuples. Default is False.
        
    Returns
    -------
    list
        List of regions, either as coordinate tuples or slice objects.
    """
    assert len(array_shape) == len(region_shape)
    steps = []
    for size, inc in zip(array_shape, region_shape):
        seq = np.arange(0, size, inc)
        if size > seq[-1]:
            seq = np.append(seq, size)
        increments = tuple((seq[i], seq[i + 1]) for i in range(len(seq) - 1))
        if as_slices:
            steps.append(tuple(slice(*item) for item in increments))
        else:
            steps.append(increments)
    return list(itertools.product(*steps))


def get_compressor(name,
                   zarr_format = ZARR_V2,
                   **params): ### TODO: continue this, add for zarr3
    name = name.lower()
    assert zarr_format in (ZARR_V2, ZARR_V3)
    compression_dict2 = {
        "blosc": "Blosc",
        "bz2": "BZ2",
        "gzip": "GZip",
        "lzma": "LZMA",
        "lz4": "LZ4",
        "pcodec": "PCodec",
        "zfpy": "ZFPY",
        "zlib": "Zlib",
        "zstd": "Zstd"
    }

    compression_dict3 = {
        "blosc": "BloscCodec",
        "gzip": "GzipCodec",
        "sharding": "ShardingCodec",
        "zstd": "ZstdCodec",
        "crc32ccodec": "CRC32CCodec"
    }

    if zarr_format == ZARR_V2:
        compressor_name = compression_dict2[name]
        compressor_instance = getattr(numcodecs, compressor_name)
    elif zarr_format == ZARR_V3:
        compressor_name = compression_dict3[name]
        compressor_instance = getattr(codecs, compressor_name)
    else:
        raise Exception("Unsupported Zarr format")
    compressor = compressor_instance(**params)
    return compressor

def get_default_fill_value(dtype):
    try:
        dtype = np.dtype(dtype.name)
    except (AttributeError, TypeError):
        # dtype may not have .name attribute or conversion may fail
        pass
    if np.issubdtype(dtype, np.integer):
        return 0
    elif np.issubdtype(dtype, np.floating):
        return 0.0
    elif np.issubdtype(dtype, np.bool_):
        return False
    return None

def _create_zarr_v2_array(
        store_path: Union[Path, str],
        shape: Tuple[int, ...],
        chunks: Tuple[int, ...],
        dtype: Any,
        compressor_config: CompressorConfig,
        dimension_separator: str,
        overwrite: bool,
) -> zarr.Array:
    compressor = get_compressor(compressor_config.name,
                                zarr_format=ZARR_V2,
                                **compressor_config.params)
    return zarr.create(
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        store=store_path,
        compressor=compressor,
        dimension_separator=dimension_separator,
        overwrite=overwrite,
        zarr_format=ZARR_V2,
    )

def _create_zarr_v3_array(
        store: Any,
        shape: Tuple[int, ...],
        chunks: Tuple[int, ...],
        dtype: Any,
        compressor_config: CompressorConfig,
        shards: Optional[Tuple[int, ...]],
        dimension_names: str = None,
        overwrite: bool = False,
        **kwargs
) -> zarr.Array:
    compressors = [get_compressor(compressor_config.name,
                                  zarr_format=ZARR_V3,
                                  **compressor_config.params)
                   ]
    return zarr.create_array(
        store=store,
        shape=shape,
        chunks=chunks,
        shards=shards,
        dimension_names=dimension_names,
        dtype=dtype,
        compressors=compressors,
        overwrite=overwrite,
        zarr_format=ZARR_V3,
        **kwargs
    )

def _create_zarr_array(
        store_path: Union[Path, str],
        shape: Tuple[int, ...],
        chunks: Tuple[int, ...],
        dtype: Any,
        compressor_config: CompressorConfig = None,
        zarr_format: int = ZARR_V2,
        overwrite: bool = False,
        shards: Optional[Tuple[int, ...]] = None,
        dimension_separator: str = DEFAULT_DIMENSION_SEPARATOR,
        dimension_names: str = None,
        **kwargs
) -> zarr.Array:
    """Create a Zarr array with specified format and compression settings."""
    compressor_config = compressor_config or CompressorConfig()
    chunks = tuple(np.minimum(shape, chunks).tolist())
    
    # For sharding: ensure shards are compatible with chunks
    # During downscaling, reshape shards to align with new chunk sizes
    if shards is not None:
        shards = tuple(np.array(shards).flatten().tolist())
        # Adjust shards to be compatible with chunks for this layer
        # If shards don't divide evenly into chunks, scale them proportionally
        adjusted_shards = []
        for shard_size, chunk_size, dim_size in zip(shards, chunks, shape):
            if shard_size % chunk_size != 0 and chunk_size > 0:
                # Find the largest divisor of shard_size that is also a multiple of chunk_size
                # Or just use a shard size that's compatible with the current chunk
                adjusted = (dim_size // max(1, dim_size // shard_size)) if shard_size > 0 else chunk_size
                adjusted_shards.append(min(adjusted, dim_size))
            else:
                adjusted_shards.append(shard_size)
        shards = tuple(adjusted_shards)
    
    store = LocalStore(store_path)

    if zarr_format not in (ZARR_V2, ZARR_V3):
        raise ValueError(f"Unsupported Zarr format: {zarr_format}")

    if zarr_format == ZARR_V2:
        return _create_zarr_v2_array(
            store_path=store_path,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressor_config=compressor_config,
            dimension_separator=dimension_separator,
            overwrite=overwrite,
        )

    return _create_zarr_v3_array(
        store=store,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor_config=compressor_config,
        shards=shards,
        dimension_names=dimension_names,
        overwrite=overwrite,
        # **kwargs
    )

def write_chunk_with_zarrpy(chunk: np.ndarray, zarr_array: zarr.Array, block_info: Dict) -> None:
    if hasattr(chunk, "get"):
        chunk = chunk.get()  # Convert CuPy -> NumPy
    zarr_array[tuple(slice(*b) for b in block_info[0]["array-location"])] = chunk

# def compute_block_slices(arr, block_shape):
#     """Return slices defining large blocks over the array."""
#     slices_per_dim = [range(0, s, b) for
#                       s, b in zip(arr.shape, block_shape)]
#     blocks = []
#     for starts in itertools.product(*slices_per_dim):
#         block_slices = tuple(slice(start, min(start+b, dim)) for start, b, dim in zip(starts, block_shape, arr.shape))
#         blocks.append(block_slices)
#     return blocks

def compute_block_slices(arr, block_shape):
    """Compute block slices for the given array and shard sizes."""
    shards = block_shape
    block_slices = []
    for starts in np.ndindex(*[arr.shape[i] // shards[i] + (1 if arr.shape[i] % shards[i] else 0)
                               for i in range(len(arr.shape))]):
        slices = tuple(slice(start * shard, min((start + 1) * shard, arr.shape[i]))
                       for start, shard, i in zip(starts, shards, range(len(arr.shape))))
        block_slices.append(slices)
    return block_slices

async def write_block_optimized(arr, ts_store, block_slices):
    """Optimized single block write with efficient Dask computation."""
    # Get block data and compute if it's a Dask array
    block = arr[block_slices]
    if hasattr(block, 'compute'):
        block = block.compute()

    # Write and wait for completion
    write_future = ts_store[block_slices].write(block)
    write_future.result()
    return 1


import asyncio
import concurrent.futures
import copy
import gc
import itertools
import math
import os
import threading
import time
from queue import Queue
from typing import Any, Optional, Tuple, Union

import dask.array as da
import numpy as np
import tensorstore as ts
import zarr

from eubi_bridge.utils.storage_utils import make_kvstore


async def write_with_tensorstore_async(
        arr: Union[da.Array, zarr.Array, ts.TensorStore],
        store_path: Union[str, os.PathLike],
        chunks: Optional[Tuple[int, ...]] = None,
        shards: Optional[Tuple[int, ...]] = None,
        dimension_names: str = None,
        dtype: Any = None,
        compressor: str = 'blosc',
        compressor_params: dict = None,
        overwrite: bool = True,
        zarr_format: int = 2,
        pixel_sizes: Optional[Tuple[float, ...]] = None,
        max_concurrency: int = 8,
        compute_batch_size: int = 8,
        memory_limit_per_batch: int = 1024,
        ts_io_concurrency: Optional[int] = None,
        **kwargs
) -> 'ts.TensorStore':
    """
    Hybrid writer: da.compute micro-batches + overlap compute(next) with writes(current).

    - compute_batch_size: number of blocks passed to a single da.compute(...) call
    - max_concurrency: number of parallel write threads (ThreadPoolExecutor)
    - ts_io_concurrency: optional int to set kvstore file_io_concurrency limit (if desired)
    """
    compressor_params = compressor_params or {}
    try:
        dtype = np.dtype(dtype.name)
    except Exception:
        dtype = np.dtype(dtype)
    fill_value = kwargs.get('fill_value', get_default_fill_value(dtype))

    if chunks is None:
        chunks = get_chunk_shape(arr, 
                                 default_chunks=tuple([1,1,96,96,96][:-len(arr.shape)])  # default chunk size if none provided
                                 )
    chunks = tuple(int(size) for size in chunks)

    if shards is None:
        shards = copy.deepcopy(chunks)
    if not np.allclose(np.mod(shards, chunks), 0):
        multiples = np.floor_divide(shards, chunks)
        shards = np.multiply(multiples, chunks)
    shards = tuple(int(size) for size in np.ravel(shards))

    # Optionally tune TensorStore file I/O concurrency inside kvstore spec
    kvstore = make_kvstore(store_path)
    if ts_io_concurrency:
        kvstore["file_io_concurrency"] = {"limit": int(ts_io_concurrency)}

    if zarr_format == 3:
        zarr_metadata = {
            "data_type": np.dtype(dtype).name,
            "shape": arr.shape,
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": shards}},
            "dimension_names": list(dimension_names) if dimension_names else [],
            "codecs": [
                {
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": chunks,
                        "codecs": [
                            {"name": "bytes", "configuration": {"endian": "little"}},
                            {"name": compressor, "configuration": compressor_params or {}}
                        ],
                        "index_codecs": [
                            {"name": "bytes", "configuration": {"endian": "little"}},
                            {"name": "crc32c"}
                        ],
                        "index_location": "end"
                    }
                }
            ],
            "node_type": "array"
        }
    else:
        zarr_metadata = {
            "compressor": {"id": compressor, **compressor_params},
            "dtype": np.dtype(dtype).str,
            "shape": arr.shape,
            "chunks": chunks,
            "fill_value": fill_value,
            "dimension_separator": '/',
        }

    zarr_spec = {
        "driver": "zarr" if zarr_format == 2 else "zarr3",
        "kvstore": kvstore,
        "metadata": zarr_metadata,
        "create": True,
        "delete_existing": overwrite,
    }

    ctx = ts.Context({
        "cache_pool": {"total_bytes_limit": 1_000_000_000},  # 1 GB local cache
        "data_copy_concurrency": {"limit": 64},
        "s3_request_concurrency": {"limit": 32},
        "s3_request_retries": {"max_retries": 5},
    })

    ts_store = ts.open(zarr_spec, context=ctx).result()

    block_size = compute_chunk_batch(arr, dtype, memory_limit_per_batch)
    block_size = tuple([max(bs, cs) for bs, cs in zip(block_size, chunks)])
    block_size = tuple((math.ceil(bs / cs) * cs) for bs, cs in zip(block_size, chunks))
    blocks = compute_block_slices(arr, block_size)
    total_blocks = len(blocks)

    # split blocks into micro-batches
    compute_batches = [blocks[i:i + compute_batch_size]
                       for i in range(0, len(blocks), compute_batch_size)]

    loop = asyncio.get_running_loop()
    write_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, max_concurrency))

    success_count = 0

    # helper to write a single block (runs in threadpool)
    def _write_block(bs, data):
        # keep writer simple: block slice indexing then write and wait for ts future
        return ts_store[bs].write(data).result()

    try:
        # Kick off compute for the first batch in background
        next_compute_future = None
        if compute_batches:
            first_batch_blocks = compute_batches[0]
            # Prepare Dask objects for that batch
            dask_objs = [arr[bs] for bs in first_batch_blocks]
            # run da.compute in executor so we don't block the loop
            next_compute_future = loop.run_in_executor(
                None, lambda *objs: da.compute(*objs), *dask_objs
            )

        # iterate batches, compute current (await previous future) and schedule next compute
        for i, batch_blocks in enumerate(compute_batches):
            # schedule compute for the next batch (if any)
            if i + 1 < len(compute_batches):
                next_batch_blocks = compute_batches[i + 1]
                next_dask_objs = [arr[bs] for bs in next_batch_blocks]
                # Note: None as executor uses default ThreadPoolExecutor from loop.run_in_executor
                next_future = loop.run_in_executor(None, lambda *objs: da.compute(*objs), *next_dask_objs)
            else:
                next_future = None

            # wait for current compute (the previously kicked-off one)
            if next_compute_future is None:
                # This can happen if there were no initial compute; compute inline as fallback
                dask_objs = [arr[bs] for bs in batch_blocks]
                computed = da.compute(*dask_objs) if hasattr(arr, 'compute') else tuple(dask_objs)
            else:
                computed = await next_compute_future  # tuple of numpy arrays

            # Now write all blocks of this batch concurrently using write_executor
            write_futures = []
            for bs, data in zip(batch_blocks, computed):
                # submit to write threadpool
                write_futures.append(loop.run_in_executor(write_executor, _write_block, bs, data))

            # Wait for all writes in this batch to complete
            await asyncio.gather(*write_futures)

            success_count += len(batch_blocks)
            logger.info(
                f"From the array with shape {arr.shape},\n"
                f"with the allocated region size {block_size} for the memory limit {memory_limit_per_batch} and dtype {dtype},\n"
                f"wrote {success_count}/{total_blocks} blocks (batch {i+1}/{len(compute_batches)}) to {store_path}"
            )

            # advance next_compute_future
            next_compute_future = next_future

        # if there is a leftover compute future (for final scheduled compute) wait and discard (shouldn't happen)
        if next_compute_future is not None:
            await next_compute_future

    finally:
        write_executor.shutdown(wait=True)

    # ---- Metadata handling (unchanged) ----
    gr_path = os.path.dirname(store_path)
    arrpath = os.path.basename(store_path)
    gr = zarr.group(gr_path)
    handler = NGFFMetadataHandler()
    handler.connect_to_group(gr)
    handler.read_metadata()
    handler.add_dataset(path=arrpath, scale=pixel_sizes, overwrite=True)
    handler.save_changes()

    return ts_store






async def downscale_with_tensorstore_async(
        base_store: Union[str, Path, 'ts.TensorStore'],
        scale_factor,
        n_layers,
        downscale_method='simple',
        min_dimension_size = None,
        **kwargs
    ):
    try:
        import tensorstore as ts
    except ImportError:
        raise ModuleNotFoundError(
            "The module tensorstore has not been found. "
            "Try 'conda install -c conda-forge tensorstore'"
        )

    if isinstance(base_store, ts.TensorStore):
        base_array_path = base_store.kvstore.path
    else:
        base_array_path = base_store

    gr_path = os.path.dirname(base_array_path)
    pyr = Pyramid(gr_path)

    # min_dimension_size = kwargs.get('min_dimension_size', None)
    # scale_factor = [scale_factor_dict[ax] for ax in pyr.meta.axis_order]

    await pyr.update_downscaler(scale_factor=scale_factor,
                          n_layers=n_layers,
                          downscale_method=downscale_method,
                          min_dimension_size=min_dimension_size,
                          use_tensorstore=True
                          )

    try:
        grpath = pyr.gr.store.root
    except AttributeError:
        # Fallback for stores without .root attribute
        grpath = pyr.gr.store.path
    basepath = pyr.meta.resolution_paths[0]
    base_layer = pyr.layers[basepath]
    zarr_format = pyr.meta.zarr_format

    try:
        compressor_params = base_layer.compressors[0].get_config()
    except (AttributeError, IndexError):
        # Fallback if get_config() not available or no compressors
        compressor_params = base_layer.compressors[0].to_dict()#dict(base_layer.codec.to_json())
    if 'id' in compressor_params:
        compressor_name = compressor_params['id']
        compressor_params.pop('id')
    elif 'name' in compressor_params:
        compressor_name = compressor_params['name']
        compressor_params = compressor_params['configuration']

    # compressor_params = dict(arr.codec.to_json())

    coros = []
    for key, arr in pyr.downscaler.downscaled_arrays.items():
        if key != '0':
            shards = tuple(base_layer.shards) if base_layer.shards is not None else base_layer.chunks
            params = dict(
                arr = arr,
                output_path=os.path.join(grpath, key),
                output_chunks = tuple(base_layer.chunks),
                output_shards = shards,
                compressor = compressor_name,
                compressor_params = compressor_params,
                zarr_format = zarr_format,
                dimension_names = list(pyr.axes),
                pixel_sizes = tuple(pyr.downscaler.dm.scales[int(key)]),
                dtype = np.dtype(arr.dtype.name),
                # **kwargs,
                **{k: v for k, v in kwargs.items() if k not in ('max_concurrency', 'dtype', 'compressor', 'compressor_params', 'zarr_format')}
            )
            #coro = write_with_tensorstore_async(**params)
            #import pprint
            #pprint.pprint(params)
            coro = write_with_queue_async(**params)
            coros.append(coro)

    await asyncio.gather(*coros)
    pyr = Pyramid(gr_path)

    return pyr


def _get_or_create_multimeta(gr: zarr.Group,
                             axis_order: str,
                             unit_list: List[str],
                             version: str) -> NGFFMetadataHandler:
    """
    Read existing or create new metadata handler for zarr group.

    Parameters
    ----------
    gr : zarr.Group
        Zarr group to read metadata from or write metadata to.
    axis_order : str
        String indicating the order of axes in the arrays.
    unit_list : List[str]
        List of strings indicating the units of each axis.
    version : str
        Version of NGFF to create if no metadata exists.

    Returns
    -------
    handler : NGFFMetadataHandler
        Metadata handler for the zarr group.
    """
    handler = NGFFMetadataHandler()
    handler.connect_to_group(gr)
    try:
        handler.read_metadata()
    except (FileNotFoundError, KeyError, ValueError) as e:
        # Metadata doesn't exist or is invalid, create new
        handler.create_new(version=version)
        handler.parse_axes(axis_order=axis_order, units=unit_list)
    return handler


def _read_region(arr, region_slice):
    """
    Unified region reader for both dask.array and DynamicArray.
    
    Parameters
    ----------
    arr : Union[da.Array, DynamicArray, zarr.Array, np.ndarray]
        Array to read from.
    region_slice : tuple of slice
        Region to read.
        
    Returns
    -------
    np.ndarray
        Region data as numpy array.
    """
    try:
        from eubi_bridge.external.dyna_zarr.dynamic_array import DynamicArray
        
        if isinstance(arr, DynamicArray):
            # Zero-copy direct read (optimal for zarr/tensorstore backends)
            return arr._read_direct(region_slice)
        elif hasattr(arr, 'compute') and hasattr(arr, '__dask_graph__'):
            # Dask array: slice then compute
            sliced = arr[region_slice]
            return sliced.compute()
        else:
            # Direct array (zarr.Array, np.ndarray, etc.)
            return np.asarray(arr[region_slice])
    except ImportError:
        # DynamicArray not available, fall back to dask/numpy
        if hasattr(arr, 'compute'):
            return arr[region_slice].compute()
        else:
            return np.asarray(arr[region_slice])


def _compute_region_shape(input_shape, final_chunks, region_size_mb, dtype=None, input_chunks=None):
    """
    Compute optimal region shape with deterministic algorithm.
    
    Uses LCM (Least Common Multiple) to maintain alignment with both
    input chunks and output chunks.
    
    Algorithm:
    1. Start with a single output chunk
    2. Expand dimensions in reverse order (last → first) until region_size_mb reached
    3. Use LCM of input_chunks and output_chunks for expansion increments
    4. Stop when budget exhausted or dimension complete
    
    Parameters
    ----------
    input_shape : tuple of int
        Shape of input array.
    final_chunks : tuple of int
        Output chunk shape (zarr chunks).
    region_size_mb : float or str
        Target size of read regions. Can be a number in MB or a string like '1GB', '512MB'.
    dtype : numpy.dtype, optional
        Data type for computing element size.
    input_chunks : tuple of int, optional
        Input chunk shape (for alignment).
        
    Returns
    -------
    tuple of int
        Optimal region shape.
        
    Example
    -------
    >>> _compute_region_shape((50, 179, 2, 339, 415), (1, 64, 1, 64, 64), 8.0)
    (1, 128, 2, 339, 415)
    """
    # Import here to avoid circular imports
    from eubi_bridge.utils.array_utils import parse_memory
    
    # Convert region_size_mb to MB if it's a string like '1GB'
    region_size_mb = parse_memory(region_size_mb)
    
    if dtype is None:
        element_size = 2
    else:
        try:
            element_size = int(np.dtype(dtype).itemsize)
        except Exception:
            element_size = 2
    
    target_bytes = region_size_mb * 1024 * 1024
    
    input_arr = np.array(input_shape, dtype=np.int64)
    output_chunk_arr = np.array(final_chunks, dtype=np.int64)
    
    if input_chunks is None:
        input_chunk_arr = output_chunk_arr.copy()
    else:
        input_chunk_arr = np.array(input_chunks, dtype=np.int64)
    
    # STEP 1: Start with one output chunk
    region_arr = output_chunk_arr.copy()
    current_bytes = np.prod(region_arr) * element_size
    
    # If single output chunk exceeds target, use it anyway (can't split chunks)
    if current_bytes >= target_bytes:
        return tuple(region_arr.tolist())
    
    # STEP 2: Compute expansion increments using LCM (maintains both alignments)
    expansion_increments = np.zeros(len(region_arr), dtype=np.int64)
    for i in range(len(region_arr)):
        gcd = np.gcd(input_chunk_arr[i], output_chunk_arr[i])
        lcm = (input_chunk_arr[i] * output_chunk_arr[i]) // gcd
        expansion_increments[i] = lcm
    
    # STEP 3: Expand dimensions in reverse order (last → first)
    for dim in reversed(range(len(region_arr))):
        # Expand this dimension until complete or budget exhausted
        while region_arr[dim] < input_arr[dim]:
            increment = expansion_increments[dim]
            remaining = input_arr[dim] - region_arr[dim]
            
            # Determine new size
            if remaining <= increment:
                # Remainder fits in one increment - complete the dimension
                new_size = input_arr[dim]
            else:
                # Add one increment
                new_size = region_arr[dim] + increment
                
                # Check if we should include the remainder now
                # to avoid creating a small partial region later
                future_remaining = input_arr[dim] - new_size
                if 0 < future_remaining < increment:
                    # Next time would be a small remainder - include it now
                    new_size = input_arr[dim]
            
            # Test if this fits in budget
            test_region = region_arr.copy()
            test_region[dim] = new_size
            new_bytes = np.prod(test_region) * element_size
            
            if new_bytes <= target_bytes:
                # Fits in budget - accept it
                region_arr[dim] = new_size
                current_bytes = new_bytes
            else:
                # Doesn't fit - stop expanding this dimension
                break
        
        # After completing this dimension, check if we should continue
        # to the next dimension or stop
        if current_bytes >= target_bytes:
            break
    
    # STEP 4: Verify output chunk alignment for PARTIAL dimensions only
    # Full dimensions don't need alignment (they include all chunks anyway)
    for i in range(len(region_arr)):
        # Skip if dimension is fully enclosed
        if region_arr[i] >= input_arr[i]:
            continue
        
        # For partial dimensions, ensure output chunk alignment
        if output_chunk_arr[i] > 0 and region_arr[i] % output_chunk_arr[i] != 0:
            # Round down to output chunk boundary to avoid cutting inside chunks
            aligned_size = (region_arr[i] // output_chunk_arr[i]) * output_chunk_arr[i]
            # Ensure at least one output chunk
            region_arr[i] = max(output_chunk_arr[i], aligned_size)
    
    return tuple(region_arr.tolist())


def wrap_output_path(output_path):
    if output_path.startswith('https://'):
        endpoint_url = 'https://' + output_path.replace('https://', '').split('/')[0]
        relpath = output_path.replace(endpoint_url, '')
        fs = s3fs.S3FileSystem(
            client_kwargs={
                'endpoint_url': endpoint_url,
            },
            endpoint_url=endpoint_url
        )
        fs.makedirs(relpath, exist_ok=True)
        mapped = fs.get_mapper(relpath)
    else:
        os.makedirs(output_path, exist_ok=True)
        mapped = os.path.abspath(output_path)
    return mapped









async def write_with_queue_async(
    arr: Union[da.Array, zarr.Array],
    output_path: Union[Path, str],
    output_chunks: Tuple[int, ...] = None,
    output_shards: Optional[Tuple[int, ...]] = None,
    zarr_format: int = 2,
    dtype: Optional[np.dtype] = None,
    dimension_names: Optional[List[str]] = None,
    compressor: str = 'blosc',
    compressor_params: Optional[dict] = None,
    pixel_sizes: Optional[Tuple[float, ...]] = None,
    num_readers: Optional[int] = None,
    max_concurrency: Optional[int] = None,
    region_size_mb: float = 8.0,
    queue_size: Optional[int] = None,
    gc_interval: float = 15.0,
    overwrite: bool = False,
    verbose: bool = False,
    **kwargs
) -> 'ts.TensorStore':
    """
    Queue-based writer with producer-consumer threading pattern.
    
    Architecture:
    - Reader threads: Call _read_region() to read from input array, queue (slice, data) tuples
    - Writer threads: Pop from queue, submit TensorStore async writes
    - Monitor thread: Progress logging every 2 seconds
    - Queue buffering: Decouples read/write for pipeline throughput
    
    This unified writer handles both dask.array and DynamicArray through _read_region().
    
    Parameters
    ----------
    arr : Union[da.Array, zarr.Array, DynamicArray]
        Input array to write (supports dask, zarr, or DynamicArray).
    store_path : Union[Path, str]
        Path to output zarr array.
    output_chunks : Tuple[int, ...]
        Output chunk shape.
    zarr_format : int, optional
        Zarr format version (2 or 3). Default is 2.
    dtype : Optional[np.dtype], optional
        Output data type. If None, uses input array dtype.
    dimension_names : Optional[List[str]], optional
        Names for each dimension (e.g., ['t', 'c', 'z', 'y', 'x']).
    compressor : str, optional
        Compression algorithm ('blosc', 'gzip', 'zstd', etc.). Default is 'blosc'.
    compressor_params : Optional[dict], optional
        Parameters for compressor (e.g., {'cname': 'zstd', 'clevel': 1}).
    num_readers : Optional[int], optional
        Number of reader threads. Default is 2 * max_concurrency.
    max_concurrency : Optional[int], optional
        Number of writer threads. Default is 4.
    region_size_mb : float, optional
        Target size of read regions in MB. Default is 8.0.
    queue_size : Optional[int], optional
        Maximum queue size. Default is min(128, max(32, num_readers)).
    gc_interval : float, optional
        Seconds between garbage collections. Default is 15.0.
    overwrite : bool, optional
        If True, delete existing data before writing. Default is False.
    verbose : bool, optional
        Enable verbose logging. Default is False.
    **kwargs
        Additional parameters (e.g., 'output_shards' for zarr v3).
        
    Returns
    -------
    ts.TensorStore
        TensorStore handle to the written array.
        
    Notes
    -----
    - Uses _read_region() for unified reading from dask/DynamicArray
    - Region size computed with _compute_region_shape() for optimal alignment
    - Queue-based streaming avoids memory spikes from overlapping compute/write
    - Progress monitoring logs every 2 seconds via monitor_progress()
    """
    import tensorstore as ts
    
    # === DEFAULTS ===
    if max_concurrency is None:
        max_concurrency = 4
    if num_readers is None:
        num_readers = 2 * max_concurrency
    if queue_size is None:
        queue_size = min(128, max(8, num_readers))
    
    if dtype is None:
        dtype = arr.dtype
    elif isinstance(dtype, str):
        dtype = np.dtype(dtype)
    
    if output_chunks is None:
        output_chunks = get_chunk_shape(arr)
    
    if output_shards is None:
        output_shards = copy.deepcopy(output_chunks)
    if not np.allclose(np.mod(output_shards, output_chunks), 0):
        multiples = np.floor_divide(output_shards, output_chunks)
        output_shards = np.multiply(multiples, output_chunks)
    output_shards = tuple(int(size) for size in np.ravel(output_shards))

    # === CREATE ARRAY WITH ZARR LIBRARY FIRST ===
    # This ensures all compressor parameters are applied correctly
    if compressor_params is None:
        compressor_params = {}
    
    # Clean the output path
    output_path_str = str(output_path)
    if overwrite and os.path.exists(output_path_str):
        shutil.rmtree(output_path_str)
    os.makedirs(output_path_str, exist_ok=True)
    
    compressor_config = CompressorConfig(
        name=compressor,
        params=compressor_params
    )
    z = _create_zarr_array(
        store_path=output_path_str,
        shape=arr.shape,
        chunks=output_chunks,
        shards=output_shards,
        dtype=dtype,
        compressor_config=compressor_config,
        zarr_format=zarr_format,
        dimension_names=dimension_names,
        overwrite=overwrite,
    )

    # === COMPUTE REGION SHAPE ===
    input_chunks = None
    try:
        if hasattr(arr, 'chunks'):
            # Dask array - extract numeric chunks
            input_chunks = tuple(c[0] if isinstance(c, tuple) else c for c in arr.chunks)
        elif hasattr(arr, 'chunk_shape'):
            # DynamicArray
            input_chunks = arr.chunk_shape
    except Exception:
        pass

    region_shape = _compute_region_shape(
        input_shape=arr.shape,
        final_chunks=output_chunks,
        region_size_mb=region_size_mb,
        dtype=dtype,
        input_chunks=input_chunks
    )
    # If arr is a dask array, use dask to write
    if isinstance(arr, da.Array):
        arr = arr.rechunk(region_shape)
    
    # === OPEN WITH TENSORSTORE FOR WRITING ===
    # TensorStore will use the metadata already written by zarr library
    spec_dict = {
        'driver': 'zarr' if zarr_format == 2 else 'zarr3',
        'kvstore': {
            'driver': 'file',
            'path': output_path_str
        },
        'open': True  # Open existing array instead of creating
    }
    
    ts_store = await ts.open(spec_dict)
    
    # # === COMPUTE REGION SHAPE ===
    # input_chunks = None
    # try:
    #     if hasattr(arr, 'chunks'):
    #         # Dask array - extract numeric chunks
    #         input_chunks = tuple(c[0] if isinstance(c, tuple) else c for c in arr.chunks)
    #     elif hasattr(arr, 'chunk_shape'):
    #         # DynamicArray
    #         input_chunks = arr.chunk_shape
    # except Exception:
    #     pass
    
    # region_shape = _compute_region_shape(
    #     input_shape=arr.shape,
    #     final_chunks=output_chunks,
    #     region_size_mb=region_size_mb,
    #     dtype=dtype,
    #     input_chunks=input_chunks
    # )
    
    if verbose:
        logger.info(f"Queue-based writer: {num_readers} readers, {max_concurrency} writers, region_shape={region_shape}")
    
    # === THREADED WRITE FUNCTION ===
    def _run_threaded_write():
        """Synchronous function that runs the threaded write pipeline."""
        # Shared state
        state = {
            'completed': 0,
            'failed': 0,
            'total': 0,
            'lock': threading.Lock(),
            'error': None,
            'done_reading': False
        }
        
        # Compute total regions
        total_regions = 1
        for dim_size, region_size in zip(arr.shape, region_shape):
            total_regions *= int(np.ceil(dim_size / region_size))
        state['total'] = total_regions
        
        # Create queue
        q = Queue(maxsize=queue_size)
        
        # Generate region indices
        region_indices = []
        ranges = [range(0, dim_size, region_size) for dim_size, region_size in zip(arr.shape, region_shape)]
        for idx_tuple in itertools.product(*ranges):
            region_slice = tuple(
                slice(start, min(start + region_size, dim_size))
                for start, region_size, dim_size in zip(idx_tuple, region_shape, arr.shape)
            )
            region_indices.append(region_slice)
        
        # Atomic index counter
        index_lock = threading.Lock()
        index_counter = [0]  # Mutable container for atomic updates
        
        # === READER THREAD ===
        def reader_thread():
            """Read regions and enqueue them."""
            last_gc = time.time()
            while True:
                # Atomically get next index
                with index_lock:
                    if index_counter[0] >= len(region_indices):
                        break
                    idx = index_counter[0]
                    index_counter[0] += 1
                
                region_slice = region_indices[idx]
                if verbose:
                    logger.info(f"Reader thread reading region {idx+1}/{len(region_indices)}: {region_slice}")
                    logger.info(f"arr shape: {arr.shape}, region shape: {[s.stop - s.start for s in region_slice]}")

                try:
                    # Read region using unified abstraction
                    data = _read_region(arr, region_slice)
                    
                    # Enqueue
                    q.put((region_slice, data))
                    
                    # Periodic GC
                    if time.time() - last_gc > gc_interval:
                        gc.collect()
                        last_gc = time.time()
                        
                except Exception as e:
                    with state['lock']:
                        if state['error'] is None:
                            state['error'] = e
                    logger.error(f"Reader thread error at {region_slice}: {e}")
                    break
        
        # === WRITER THREAD ===
        def writer_thread():
            """Write regions from queue."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def _async_writer():
                while True:
                    try:
                        region_slice, data = q.get(timeout=1.0)
                    except Exception:
                        # Check if reading is done and queue is empty
                        if state['done_reading'] and q.empty():
                            break
                        continue
                    
                    try:
                        # Submit async write
                        if verbose:
                            logger.info(f"Writer thread writing region: {region_slice}")
                            logger.info(f"Data shape: {data.shape}, expected shape: {[s.stop - s.start for s in region_slice]}")
                        await ts_store[region_slice].write(data)
                        
                        with state['lock']:
                            state['completed'] += 1
                        
                        q.task_done()
                        
                    except Exception as e:
                        with state['lock']:
                            state['failed'] += 1
                            if state['error'] is None:
                                state['error'] = e
                        logger.error(f"Writer thread error at {region_slice}: {e}")
                        q.task_done()
            
            loop.run_until_complete(_async_writer())
            loop.close()
        
        # === MONITOR THREAD ===
        def monitor_progress():
            """Log progress every 2 seconds."""
            while True:
                time.sleep(2.0)
                with state['lock']:
                    completed = state['completed']
                    total = state['total']
                    if total > 0:
                        pct = 100.0 * completed / total
                        if verbose:
                            logger.info(f"Write progress: {completed}/{total} regions ({pct:.1f}%)")
                    if completed + state['failed'] >= total:
                        break
        
        # === START THREADS ===
        readers = [threading.Thread(target=reader_thread, daemon=True) for _ in range(num_readers)]
        writers = [threading.Thread(target=writer_thread, daemon=True) for _ in range(max_concurrency)]
        monitor = threading.Thread(target=monitor_progress, daemon=True)
        
        for t in readers:
            t.start()
        for t in writers:
            t.start()
        monitor.start()
        
        # Wait for readers to finish
        for t in readers:
            t.join()
        
        # Signal writers that reading is done
        state['done_reading'] = True
        
        # Wait for queue to empty and writers to finish
        q.join()
        for t in writers:
            t.join()
        
        monitor.join(timeout=5.0)
        
        # Check for errors
        if state['error'] is not None:
            raise state['error']
        
        if verbose:
            logger.info(f"Write complete: {state['completed']}/{state['total']} regions")
    
    # === RUN THREADED WRITE IN EXECUTOR ===
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _run_threaded_write)
    
    # === METADATA HANDLING ===
    if pixel_sizes is not None:
        gr_path = os.path.dirname(output_path)
        arrpath = os.path.basename(output_path)
        gr = zarr.group(gr_path)
        handler = NGFFMetadataHandler()
        handler.connect_to_group(gr)
        handler.read_metadata()
        handler.add_dataset(path=arrpath, scale=pixel_sizes, overwrite=True)
        handler.save_changes()
    
    return ts_store


async def store_multiscale_async(
    ### base write params
    arr: Union[da.Array, zarr.Array],
    output_path: Union[Path, str],
    axes: Sequence[str],
    scales: Sequence[Tuple[float, ...]],  # pixel sizes
    units: Sequence[str],
    zarr_format: int = 2,
    auto_chunk: bool = True,
    output_chunks: Optional[Dict[str, Tuple[int, ...]]] = None,
    output_shard_coefficients: Optional[Dict[str, Tuple[int, ...]]] = None,
    compute: bool = False,
    overwrite: bool = False,
    channel_meta: Optional[dict] = None,
    *,
    ### downscale params
    scale_factors: Optional[Tuple[int, ...]] = None,
    n_layers = None,
    min_dimension_size = None,
    downscale_method='simple',
    ### queue-based writer params
    num_readers: Optional[int] = None,      # Number of reader threads (default: 2 * max_concurrency)
    max_concurrency: Optional[int] = None,      # Number of writer threads (default: 4)
    region_size_mb: float = 8.0,            # Target size of read regions in MB
    queue_size: Optional[int] = None,       # Maximum queue size (default: adaptive)
    gc_interval: float = 15.0,              # Seconds between garbage collections
    **kwargs
) -> 'ts.TensorStore':
    # logger.info(f"The array with shape {arr.shape} will be written to {output_path}.")
    import tensorstore as ts
    writer_func = write_with_queue_async
    # Get important kwargs:
    verbose = kwargs.get('verbose', False)
    output_shards = kwargs.get('output_shards', None)
    target_chunk_mb = kwargs.get('target_chunk_mb', 1)
    dtype = kwargs.get('dtype', arr.dtype)
    if dtype is None:
        dtype = arr.dtype
    elif isinstance(dtype, str):
        dtype = np.dtype(dtype)
    compressor = kwargs.get('compressor', 'blosc')
    compressor_params = kwargs.get('compressor_params', {})
    logger.info(f"Compressor selected for output: {compressor} with params: {compressor_params}")
    ###

    dimension_names = list(axes)
    ### Parse chunks
    if auto_chunk or output_chunks is None:     
        if verbose:
            logger.info(f"Auto-computing chunks for {output_path} with target chunk size {target_chunk_mb} MB")     
        chunks = autocompute_chunk_shape(
            arr.shape,
            axes=axes,
            target_chunk_mb=target_chunk_mb,
            dtype=dtype
        )
        if verbose:
            logger.info(f"Auto-chunking {output_path} to {chunks}")
            logger.info(f"Computed chunks: {chunks}")
    else:
        chunks = output_chunks

    chunks = np.minimum(chunks, arr.shape).tolist()
    chunks = tuple(int(item) for item in chunks)
    channels = channel_meta

    ### Parse shards
    if output_shards is not None:
        shards = output_shards
    elif output_shard_coefficients is not None:
        shardcoefs = output_shard_coefficients
        shards = tuple(int(c * s) for c, s in zip(chunks,
                                                  shardcoefs))
    else:
        shards = chunks
    shards = tuple(int(item) for item in shards)
    ###
    # Make (or overwrite) the top-level group

    outpath = wrap_output_path(output_path)
    gr = zarr.group(outpath,
               overwrite=overwrite,
               zarr_version=zarr_format)

    ### Make the base path (use outpath which is the wrapped/resolved path)
    base_store_path = os.path.join(outpath, '0')
    ### Add multiscales metadata
    version = '0.5' if zarr_format == 3 else '0.4'
    meta = _get_or_create_multimeta(
        gr,
        axis_order = axes,
        unit_list = units,
        version = version
    )

    if channels == 'auto':
        if 'c' in axes:
            idx = axes.index('c')
            size = arr.shape[idx]
        else:
            size = 1
        meta.autocompute_omerometa(size, arr.dtype)
    elif channels is not None:
        if verbose:
            logger.info(f"Adding channel metadata: {channels}")
        meta.metadata['omero']['channels'] = channels
    
    meta.save_changes()

    if verbose:
        logger.info(f"Writer function: {writer_func}")
    # Write base layer with progress and error handling
    if verbose:
        logger.info(f"Starting to write base layer to {base_store_path}")
    base_start_time = time.time()
    if verbose:
        logger.info(f"The region_size_mb is set to {region_size_mb} MB for base layer writing.")
    base_ts_store = await writer_func(
        arr=arr,
        output_path=base_store_path,
        output_chunks=chunks,
        zarr_format=zarr_format,
        dtype=dtype,
        dimension_names=list(axes),
        compressor=compressor,
        compressor_params=compressor_params,
        pixel_sizes=scales,  # Base layer scale
        num_readers=num_readers,
        max_concurrency=max_concurrency,
        region_size_mb=region_size_mb,
        queue_size=queue_size,
        gc_interval=gc_interval,
        overwrite=overwrite,
        verbose=verbose,
        output_shards=shards
    )

    base_elapsed = (time.time() - base_start_time) / 60
    logger.info(f"Base layer written in {base_elapsed:.2f} minutes")

    # Add base layer to metadata
    meta.add_dataset(path='0', scale=scales)
    meta.save_changes()

    # Only proceed with downscaling if base layer was successful
    if scale_factors is not None:
        logger.info(f"Starting downscaling...")
        downscale_start = time.time()

        pyr = await downscale_with_tensorstore_async(
            base_store=base_store_path,
            scale_factor=scale_factors,
            n_layers=n_layers,
            min_dimension_size=min_dimension_size,
            downscale_method=downscale_method,
            chunks=chunks,
            shards=shards,
            max_concurrency=max_concurrency,
            queue_size=queue_size,
            region_size_mb=region_size_mb,
            # dtype=dtype,
            **{k: v for k, v in kwargs.items() if k != 'max_concurrency'}
        )
        downscale_elapsed = (time.time() - downscale_start) / 60
        logger.info(f"Downscaling completed in {downscale_elapsed:.2f} minutes")
    else:
        pyr = Pyramid(gr)

    return pyr
