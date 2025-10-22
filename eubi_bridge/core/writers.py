import copy
import os
import time
import itertools
import tempfile
import shutil
import threading
import asyncio
import zarr
import dask
import math
import sys
import numcodecs
from zarr import codecs
from zarr.storage import LocalStore
from dataclasses import dataclass
from dask import delayed
import dask.array as da
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Union, Any, Tuple, Optional, Sequence
from distributed import get_client
import concurrent.futures
import tensorstore as ts
from typing import Union, Optional, Tuple, Any


### internal imports
from eubi_bridge.ngff.multiscales import Pyramid, NGFFMetadataHandler  #Multimeta
from eubi_bridge.utils.convenience import (
    get_chunksize_from_array,
    is_zarr_group,
    autocompute_chunk_shape,
    compute_chunk_batch,
    get_chunk_shape,
)
from eubi_bridge.utils.logging_config import get_logger

import logging, warnings

logger = get_logger(__name__)

# logging.getLogger('distributed.diskutils').setLevel(logging.CRITICAL)

logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    force=True)

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
    color = default_colors[i] if i < len(default_colors) else f"{i * 40 % 256:02X}{i * 85 % 256:02X}{i * 130 % 256:02X}"
    return color

def create_zarr_array(directory: Union[Path, str, zarr.Group],
                      array_name: str,
                      shape: Tuple[int, ...],
                      chunks: Tuple[int, ...],
                      dtype: Any,
                      overwrite: bool = False) -> zarr.Array:
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
    except:
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
    if shards is not None:
        shards = tuple(np.array(shards).flatten().tolist())
        assert np.allclose(np.mod(shards, chunks), 0), f"Shards {shards} must be a multiple of chunks {chunks}"
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

# async def write_with_tensorstore_async(
#         arr: (da.Array, zarr.Array, ts.TensorStore),
#         store_path: Union[str, os.PathLike],
#         chunks: Optional[Tuple[int, ...]] = None,
#         shards: Optional[Tuple[int, ...]] = None,
#         dimension_names: str = None,
#         dtype: Any = None,
#         compressor: str = 'blosc',
#         compressor_params: dict = None,
#         overwrite: bool = True,
#         zarr_format: int = 2,
#         pixel_sizes: Optional[Tuple[float, ...]] = None,
#         max_concurrency: int = 4,
#         compute_batch_size: int = 3,  # Compute this many blocks at once
#         memory_limit_per_batch: int = 1024,  # Limit memory usage to this many MB
#         **kwargs
# ) -> 'ts.TensorStore':
#     """
#     High-performance version that batches Dask computations for maximum efficiency.
#     """
#     compressor_params = compressor_params or {}
#     try:
#         dtype = np.dtype(dtype.name)
#     except:
#         dtype = np.dtype(dtype)
#     fill_value = kwargs.get('fill_value', get_default_fill_value(dtype))
#
#     if chunks is None:
#         chunks = get_chunk_shape(arr)
#     else:
#         pass
#
#     chunks = tuple(int(size) for size in chunks)
#
#     if shards is None:
#         shards = copy.deepcopy(chunks)
#
#     if not np.allclose(np.mod(shards, chunks), 0):
#         multiples = np.floor_divide(shards, chunks)
#         shards = np.multiply(multiples, chunks)
#
#     shards = tuple(int(size) for size in np.ravel(shards))
#
#     # Prepare zarr metadata
#     if zarr_format == 3:
#         zarr_metadata = {
#             "data_type": np.dtype(dtype).name,
#             "shape": arr.shape,
#             "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": shards}},
#             "dimension_names": list(dimension_names) if dimension_names else [],
#             "codecs": [
#                 {
#                     "name": "sharding_indexed",
#                     "configuration": {
#                         "chunk_shape": chunks,
#                         "codecs": [
#                             {"name": "bytes", "configuration": {"endian": "little"}},
#                             {"name": compressor, "configuration": compressor_params or {}}
#                         ],
#                         "index_codecs": [
#                             {"name": "bytes", "configuration": {"endian": "little"}},
#                             {"name": "crc32c"}
#                         ],
#                         "index_location": "end"
#                     }
#                 }
#             ],
#             "node_type": "array"
#         }
#     else:  # zarr_format == 2
#         zarr_metadata = {
#             "compressor": {"id": compressor, **compressor_params},
#             "dtype": np.dtype(dtype).str,
#             "shape": arr.shape,
#             "chunks": chunks,
#             "fill_value": fill_value,
#             "dimension_separator": '/',
#         }
#
#     zarr_spec = {
#         "driver": "zarr" if zarr_format == 2 else "zarr3",
#         "kvstore": {"driver": "file", "path": str(store_path)},
#         "metadata": zarr_metadata,
#         "create": True,
#         "delete_existing": overwrite,
#     }
#
#     ts_store = ts.open(zarr_spec).result()
#
#     block_size = compute_chunk_batch(arr, dtype, memory_limit_per_batch)
#     block_size = tuple([max(bs, cs) for bs, cs in zip(block_size, chunks)])
#     block_size = tuple((math.ceil((bs / cs)) * cs) for bs,cs in zip(block_size, chunks))
#
#     blocks = compute_block_slices(arr, block_size)
#     total_blocks = len(blocks)
#
#     # Process in compute batches for maximum Dask efficiency
#     success_count = 0
#     sem = asyncio.Semaphore(max_concurrency)
#
#     async def process_compute_batch(batch_blocks):
#         nonlocal success_count
#
#         # Prepare all block slices and data in this batch
#         block_data_pairs = [(block_slices, arr[block_slices]) for
#                             block_slices in batch_blocks]
#
#         # Compute all blocks in this batch simultaneously (Dask optimization)
#         if hasattr(arr, 'compute'):
#             computed_data = da.compute(*[data for _, data in block_data_pairs])
#         else:
#             computed_data = tuple([data for _, data in block_data_pairs])
#         # computed_data = tuple([data for _, data in block_data_pairs])
#
#         # Write all computed blocks
#         write_tasks = []
#         for (block_slices, _), computed_block in zip(block_data_pairs, computed_data):
#             async with sem:
#                 write_future = ts_store[block_slices].write(computed_block)
#                 write_tasks.append(asyncio.to_thread(write_future.result))
#                 # write_tasks.append(write_future)
#
#         await asyncio.gather(*write_tasks)
#         success_count += len(batch_blocks)
#         logger.info(f"Wrote {success_count}/{total_blocks} blocks")
#
#     # Split into compute batches
#     compute_batches = [blocks[i:i + compute_batch_size]
#                        for i in range(0, len(blocks), compute_batch_size)]
#
#     # Process all compute batches
#     for batch in compute_batches:
#         await process_compute_batch(batch)
#
#     # ---- Metadata handling last ----
#     gr_path = os.path.dirname(store_path)
#     arrpath = os.path.basename(store_path)
#     gr = zarr.group(gr_path)
#     handler = NGFFMetadataHandler()
#     handler.connect_to_group(gr)
#     handler.read_metadata()
#     handler.add_dataset(path=arrpath,
#                         scale=pixel_sizes,
#                         overwrite=True)
#     handler.save_changes()
#     return ts_store

# async def write_with_tensorstore_async(
#         arr: (da.Array, zarr.Array, ts.TensorStore),
#         store_path: Union[str, os.PathLike],
#         chunks: Optional[Tuple[int, ...]] = None,
#         shards: Optional[Tuple[int, ...]] = None,
#         dimension_names: str = None,
#         dtype: Any = None,
#         compressor: str = 'blosc',
#         compressor_params: dict = None,
#         overwrite: bool = True,
#         zarr_format: int = 2,
#         pixel_sizes: Optional[Tuple[float, ...]] = None,
#         max_concurrency: int = 4,
#         compute_batch_size: int = 3,
#         memory_limit_per_batch: int = 1024,
#         **kwargs
# ) -> 'ts.TensorStore':
#     """
#     High-performance version that batches Dask computations for maximum efficiency.
#     Optimized: uses ThreadPoolExecutor for concurrent I/O writes (optimization #1).
#     """
#     compressor_params = compressor_params or {}
#     try:
#         dtype = np.dtype(dtype.name)
#     except:
#         dtype = np.dtype(dtype)
#     fill_value = kwargs.get('fill_value', get_default_fill_value(dtype))
#
#     if chunks is None:
#         chunks = get_chunk_shape(arr)
#     chunks = tuple(int(size) for size in chunks)
#
#     if shards is None:
#         shards = copy.deepcopy(chunks)
#     if not np.allclose(np.mod(shards, chunks), 0):
#         multiples = np.floor_divide(shards, chunks)
#         shards = np.multiply(multiples, chunks)
#     shards = tuple(int(size) for size in np.ravel(shards))
#
#     # Prepare zarr metadata
#     if zarr_format == 3:
#         zarr_metadata = {
#             "data_type": np.dtype(dtype).name,
#             "shape": arr.shape,
#             "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": shards}},
#             "dimension_names": list(dimension_names) if dimension_names else [],
#             "codecs": [
#                 {
#                     "name": "sharding_indexed",
#                     "configuration": {
#                         "chunk_shape": chunks,
#                         "codecs": [
#                             {"name": "bytes", "configuration": {"endian": "little"}},
#                             {"name": compressor, "configuration": compressor_params or {}}
#                         ],
#                         "index_codecs": [
#                             {"name": "bytes", "configuration": {"endian": "little"}},
#                             {"name": "crc32c"}
#                         ],
#                         "index_location": "end"
#                     }
#                 }
#             ],
#             "node_type": "array"
#         }
#     else:  # zarr_format == 2
#         zarr_metadata = {
#             "compressor": {"id": compressor, **compressor_params},
#             "dtype": np.dtype(dtype).str,
#             "shape": arr.shape,
#             "chunks": chunks,
#             "fill_value": fill_value,
#             "dimension_separator": '/',
#         }
#
#     zarr_spec = {
#         "driver": "zarr" if zarr_format == 2 else "zarr3",
#         "kvstore": {"driver": "file", "path": str(store_path)},
#         "metadata": zarr_metadata,
#         "create": True,
#         "delete_existing": overwrite,
#     }
#
#     ts_store = ts.open(zarr_spec).result()
#
#     block_size = compute_chunk_batch(arr, dtype, memory_limit_per_batch)
#     block_size = tuple([max(bs, cs) for bs, cs in zip(block_size, chunks)])
#     block_size = tuple((math.ceil(bs / cs) * cs) for bs, cs in zip(block_size, chunks))
#
#     blocks = compute_block_slices(arr, block_size)
#     total_blocks = len(blocks)
#
#     success_count = 0
#     sem = asyncio.Semaphore(max_concurrency)
#
#     # Use a persistent ThreadPoolExecutor for I/O operations
#     executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency)
#     loop = asyncio.get_running_loop()
#
#     async def process_compute_batch(batch_blocks):
#         nonlocal success_count
#
#         block_data_pairs = [(block_slices, arr[block_slices]) for block_slices in batch_blocks]
#
#         if hasattr(arr, 'compute'):
#             computed_data = da.compute(*[data for _, data in block_data_pairs])
#         else:
#             computed_data = tuple([data for _, data in block_data_pairs])
#
#         write_tasks = []
#         for (block_slices, _), computed_block in zip(block_data_pairs, computed_data):
#             async with sem:
#                 # Run TensorStore write in ThreadPoolExecutor
#                 fut = loop.run_in_executor(
#                     executor,
#                     lambda bs=block_slices, cb=computed_block: ts_store[bs].write(cb).result()
#                 )
#                 write_tasks.append(fut)
#
#         await asyncio.gather(*write_tasks)
#         success_count += len(batch_blocks)
#         logger.info(f"Wrote {success_count}/{total_blocks} blocks")
#
#     compute_batches = [blocks[i:i + compute_batch_size]
#                        for i in range(0, len(blocks), compute_batch_size)]
#
#     for batch in compute_batches:
#         await process_compute_batch(batch)
#
#     executor.shutdown(wait=True)
#
#     # ---- Metadata handling ----
#     gr_path = os.path.dirname(store_path)
#     arrpath = os.path.basename(store_path)
#     gr = zarr.group(gr_path)
#     handler = NGFFMetadataHandler()
#     handler.connect_to_group(gr)
#     handler.read_metadata()
#     handler.add_dataset(path=arrpath, scale=pixel_sizes, overwrite=True)
#     handler.save_changes()
#
#     return ts_store


import asyncio
import concurrent.futures
import numpy as np
import math
import os
import copy
import dask.array as da
import zarr
import tensorstore as ts
from typing import Union, Optional, Tuple, Any

async def write_with_tensorstore_async(
        arr: (da.Array, zarr.Array, ts.TensorStore),
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
        chunks = get_chunk_shape(arr)
    chunks = tuple(int(size) for size in chunks)

    if shards is None:
        shards = copy.deepcopy(chunks)
    if not np.allclose(np.mod(shards, chunks), 0):
        multiples = np.floor_divide(shards, chunks)
        shards = np.multiply(multiples, chunks)
    shards = tuple(int(size) for size in np.ravel(shards))

    # Optionally tune TensorStore file I/O concurrency inside kvstore spec
    kvstore = {"driver": "file", "path": str(store_path)}
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

    ts_store = ts.open(zarr_spec).result()

    # compute block layout
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
            logger.info(f"With the block size {block_size} for the memory limit {memory_limit_per_batch} and dtype {dtype}")
            logger.info(f"Wrote {success_count}/{total_blocks} blocks (batch {i+1}/{len(compute_batches)})")

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

    grpath = pyr.gr.store.root
    basepath = pyr.meta.resolution_paths[0]
    base_layer = pyr.layers[basepath]
    zarr_format = pyr.meta.zarr_format

    try:
        compressor_params = base_layer.compressors[0].get_config()
    except:
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
            params = dict(
                arr = arr,
                store_path=os.path.join(grpath, key),
                # chunks = tuple(base_layer.chunks),
                # shards = tuple(base_layer.shards),
                compressor = compressor_name,
                compressor_params = compressor_params,
                zarr_format = zarr_format,
                dimension_names = list(pyr.axes),
                pixel_sizes = tuple(pyr.downscaler.dm.scales[int(key)]),
                dtype = np.dtype(arr.dtype.name),
                **kwargs
            )
            coro = write_with_tensorstore_async(**params)
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
    except:
        handler.create_new(version=version)
        handler.parse_axes(axis_order=axis_order, units=unit_list)
    return handler


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
    ### cluster params
    max_concurrency: int = 4,          # limit how many writes run at once
    compute_batch_size: int = 3,  # Compute this many blocks at once
    memory_limit_per_batch: int = 1024,  # Limit memory usage to this many MB
    # executor_kind: str = "processes",    # "threads" for I/O, "processes" for CPU-bound compression
    **kwargs
) -> 'ts.TensorStore':

    logger.info(f"The input array with shape {arr.shape} will be written to {output_path}.")

    import tensorstore as ts
    writer_func = write_with_tensorstore_async
    # Get important kwargs:
    verbose = kwargs.get('verbose', False)
    output_shards = kwargs.get('output_shards', None)
    target_chunk_mb = kwargs.get('target_chunk_mb', 1)
    dtype = kwargs.get('dtype', arr.dtype)
    if dtype is None:
        dtype = arr.dtype
    compressor = kwargs.get('compressor', 'blosc')
    compressor_params = kwargs.get('compressor_params', {})
    ###

    sem = asyncio.Semaphore(max_concurrency)
    tasks: Dict[str, asyncio.Task] = {}
    dimension_names = list(axes)

    ### Parse chunks
    if auto_chunk or output_chunks is None:
        chunks = autocompute_chunk_shape(
            arr.shape,
            axes=axes,
            target_chunk_mb=target_chunk_mb,
            dtype=dtype
        )
        if verbose:
            logger.info(f"Auto-chunking {output_path} to {chunks}")
    else:
        chunks = output_chunks

    channels = channel_meta
    chunks = np.minimum(chunks, arr.shape).tolist()
    chunks = tuple(int(item) for item in chunks)

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
    gr = zarr.group(output_path,
               overwrite=overwrite,
               zarr_version=zarr_format)

    ### Make the base path
    base_store_path = os.path.join(output_path, '0')
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
        for channel in channels:
            meta.add_channel(
                color = channel['color'],
                label = channel['label'],
                dtype = dtype.str,
            )
    meta.save_changes()

    if verbose:
        logger.info(f"Writer function: {writer_func}")

    # Write base layer with progress and error handling
    logger.info(f"Starting to write base layer to {base_store_path}")
    base_start_time = time.time()
    base_ts_store = await write_with_tensorstore_async(
        arr=arr,
        store_path=base_store_path,
        chunks=chunks,
        shards=shards,
        dtype=dtype,
        compressor=compressor,
        compressor_params=compressor_params,
        overwrite=overwrite,
        zarr_format=zarr_format,
        pixel_sizes=scales,
        max_concurrency=max_concurrency,  # Cap concurrency
        compute_batch_size=compute_batch_size,
        memory_limit_per_batch = memory_limit_per_batch,
        dimension_names = dimension_names, ### Consider improving this parameter
        **{k: v for k, v in kwargs.items() if k != 'max_concurrency'}
    )

    base_elapsed = (time.time() - base_start_time) / 60
    logger.info(f"Base layer written in {base_elapsed:.2f} minutes")

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
            compute_batch_size=compute_batch_size,
            memory_limit_per_batch=memory_limit_per_batch,
            # dtype=dtype,
            **{k: v for k, v in kwargs.items() if k != 'max_concurrency'}
        )
        downscale_elapsed = (time.time() - downscale_start) / 60
        logger.info(f"Downscaling completed in {downscale_elapsed:.2f} minutes")
    else:
        pyr = Pyramid(gr)

    return pyr
