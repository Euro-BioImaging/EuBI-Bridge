import asyncio
import concurrent.futures
import gc
import itertools
import math
import os
import shutil
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import dask
import dask.array as da
import numpy as np
import s3fs
import tensorstore as ts
import zarr


def _zarr_group(store, overwrite: bool, zarr_format: int) -> zarr.Group:
    """Create or open a zarr group, handling keyword differences across zarr versions."""
    try:
        return zarr.group(store, overwrite=overwrite, zarr_format=zarr_format)  # type: ignore[arg-type]
    except TypeError:
        try:
            return zarr.group(store, overwrite=overwrite, zarr_version=zarr_format)  # type: ignore[call-arg,arg-type]
        except TypeError:
            return zarr.group(store, overwrite=overwrite)

from dask import delayed
from distributed import get_client
from zarr import codecs
from zarr.storage import LocalStore

### internal imports
from eubi_bridge.core.config_models import CompressorConfig
from eubi_bridge.ngff.multiscales import NGFFMetadataHandler, Pyramid
from eubi_bridge.utils.array_utils import (autocompute_chunk_shape,
                                           compute_chunk_batch,
                                           get_array_chunks,
                                           get_chunk_shape,
                                           get_chunksize_from_array)
from eubi_bridge.utils.logging_config import get_logger
from eubi_bridge.utils.path_utils import is_zarr_group
from eubi_bridge.utils.storage_utils import make_kvstore

logger = get_logger(__name__)

# Per-array locks to serialize reads on non-thread-safe backends (e.g. BioFormats Java readers).
# Keyed by id(arr); cleaned up automatically when new arrays are registered.
_array_read_locks: dict = {}
_array_read_locks_registry_lock = threading.Lock()


def _get_read_lock(arr) -> threading.Lock:
    """Return a per-array Lock, creating one on first access."""
    arr_id = id(arr)
    with _array_read_locks_registry_lock:
        if arr_id not in _array_read_locks:
            _array_read_locks[arr_id] = threading.Lock()
        return _array_read_locks[arr_id]


def _is_resource_backed(arr) -> bool:
    """Return True when *arr* is a resource-backed dask array (BioFormats-backed)."""
    try:
        from resource_backed_dask_array import ResourceBackedDaskArray
        return isinstance(arr, ResourceBackedDaskArray)
    except ImportError:
        return False



ZARR_V2 = 2
ZARR_V3 = 3
DEFAULT_DIMENSION_SEPARATOR = "/"



def _create_zarr_v2_array(
        store_path: Union[Path, str],
        shape: Tuple[int, ...],
        chunks: Tuple[int, ...],
        dtype: Any,
        compressor_config: CompressorConfig,
        dimension_separator: str,
        overwrite: bool,
) -> zarr.Array:
    compressor = compressor_config.build(zarr_format=ZARR_V2)
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
    compressor = compressor_config.build(zarr_format=ZARR_V3)
    # For Zarr v3, only include compressors if not None (no compression)
    compressors = [compressor] if compressor is not None else []
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



# ---------------------------------------------------------------------------
# Shared write helpers
# ---------------------------------------------------------------------------

def _normalize_dtype(dtype, arr) -> np.dtype:
    """Coerce dtype to np.dtype, falling back to arr.dtype when None."""
    if dtype is None:
        return np.dtype(arr.dtype)
    if isinstance(dtype, str):
        return np.dtype(dtype)
    try:
        return np.dtype(dtype.name)
    except Exception:
        return np.dtype(dtype)


def _align_shards(shards, chunks) -> tuple:
    """Return a shard tuple whose every element is a multiple of the matching chunk size."""
    if shards is None:
        return tuple(chunks)
    shards = np.asarray(shards)
    chunks_arr = np.asarray(chunks)
    if not np.allclose(np.mod(shards, chunks_arr), 0):
        shards = np.multiply(np.floor_divide(shards, chunks_arr), chunks_arr)
    return tuple(int(s) for s in np.ravel(shards))


def _write_ngff_metadata(store_path, pixel_sizes) -> None:
    """Attach a single dataset entry to the parent group's NGFF multiscales metadata."""
    if pixel_sizes is None:
        return
    gr_path = os.path.dirname(store_path)
    arrpath = os.path.basename(store_path)
    gr = zarr.group(gr_path)
    handler = NGFFMetadataHandler()
    handler.connect_to_group(gr)
    handler.read_metadata()
    handler.add_dataset(path=arrpath, scale=pixel_sizes, overwrite=True)
    handler.save_changes()



async def downscale_with_tensorstore_async(
        base_store: Union[str, Path, 'ts.TensorStore'],
        scale_factor,
        n_layers,
        downscale_method='simple',
        min_dimension_size = None,
        smart_scale_factor = None,
        max_concurrent_downscale_layers: int = 3,
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

    # Extract region_size_mb from kwargs with default value
    region_size_mb = kwargs.get('region_size_mb', 8.0)
    max_concurrency = kwargs.pop('max_concurrency', None)

    # min_dimension_size = kwargs.get('min_dimension_size', None)
    # scale_factor = [scale_factor_dict[ax] for ax in pyr.meta.axis_order]
    
    logger.info(f"Updating downscaler with scale_factor={scale_factor}, n_layers={n_layers}, smart_scale_factor={smart_scale_factor}")
    await pyr.update_downscaler(scale_factor=scale_factor,
                          n_layers=n_layers,
                          downscale_method=downscale_method,
                          min_dimension_size=min_dimension_size,
                          smart_scale_factor=smart_scale_factor,
                          use_tensorstore=True
                          )
    
    logger.info(f"Downscaler created {len(pyr.downscaler.downscaled_arrays)} layers for writing")

    try:
        grpath = pyr.gr.store.root
    except AttributeError:
        # Fallback for stores without .root attribute
        grpath = pyr.gr.store.path
    basepath = pyr.meta.resolution_paths[0]
    base_layer = pyr.layers[basepath]
    zarr_format = pyr.meta.zarr_format

    # Handle no compression case (empty compressors list)
    if len(base_layer.compressors) == 0:
        compressor_name = None
        compressor_params = {}
    else:
        try:
            compressor_params = base_layer.compressors[0].get_config()
        except (AttributeError, IndexError):
            # Fallback if get_config() not available
            compressor_params = base_layer.compressors[0].to_dict()
        
        if 'id' in compressor_params:
            compressor_name = compressor_params['id']
            compressor_params.pop('id')
        elif 'name' in compressor_params:
            compressor_name = compressor_params['name']
            compressor_params = compressor_params['configuration']
        else:
            compressor_name = None
            compressor_params = {}

    # compressor_params = dict(arr.codec.to_json())

    # Write downscaled layers concurrently with fixed region size
    # Fixed 16 MB regions ensure fast downscaling regardless of chunk size
    downscale_region_size_mb = 16.0
    logger.info(f"Downscaling with fixed region_size_mb={downscale_region_size_mb} MB")
    
    coros = []
    layer_count = 0
    total_layers = len([k for k in pyr.downscaler.downscaled_arrays.keys() if k != '0'])
    
    for key, arr in pyr.downscaler.downscaled_arrays.items():
        if key != '0':
            layer_count += 1
            try:
                logger.info(f"Preparing layer {key} ({layer_count}/{total_layers}) for writing...")
                logger.info(f"Layer {key} shape: {arr.shape}, dtype: {arr.dtype}")
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
                    region_size_mb = downscale_region_size_mb,
                    max_concurrency = max_concurrency,
                    **{k: v for k, v in kwargs.items() if k not in (
                        'max_concurrency', 'dtype', 'compressor', 'compressor_params',
                        'zarr_format', 'region_size_mb',
                    )}
                )
                
                coro = write_with_queue_async(**params)
                coros.append((key, coro))
                
            except Exception as e:
                logger.error(f"Failed to prepare layer {key} for writing: {e}", exc_info=True)
                raise
    
    n_concurrent = max(1, min(max_concurrent_downscale_layers, len(coros)))
    logger.info(
        f"Starting concurrent writes for {len(coros)} downscaled layers "
        f"(max {n_concurrent} at a time)..."
    )

    semaphore = asyncio.Semaphore(n_concurrent)

    async def _bounded(coro):
        async with semaphore:
            return await coro

    try:
        results = await asyncio.gather(
            *[_bounded(coro) for _, coro in coros],
            return_exceptions=True,
        )

        for (key, _), result in zip(coros, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to write layer {key}: {result}", exc_info=result)
                raise result

        logger.info("All downscaled layers written successfully")
    except Exception as e:
        logger.error(f"Error during concurrent downscale writes: {e}", exc_info=True)
        raise
    
    return Pyramid(gr_path)


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
            # Dask array: slice then compute.
            # BioFormats-backed arrays share a single Java reader which is NOT
            # thread-safe, so serialise concurrent reads through a per-array lock.
            if _is_resource_backed(arr):
                with _get_read_lock(arr):
                    sliced = arr[region_slice]
                    return sliced.compute()
            else:
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
    
    # STEP 2: Compute expansion increments using LCM (maintains both alignments).
    # When lcm(input_chunk, output_chunk) >= dim_size the increment is larger
    # than the entire dimension: the first expansion step jumps straight to full
    # extent, making fine-grained region sizing impossible.  This happens when
    # gcd(input_chunk, output_chunk) is very small relative to the chunk sizes
    # (e.g. bfio tile=16384, output_chunk=724 → lcm≈2.96M >> dim_size=50960).
    # Fall back to output_chunk in that case so the budget can be honoured.
    expansion_increments = np.zeros(len(region_arr), dtype=np.int64)
    for i in range(len(region_arr)):
        gcd = np.gcd(input_chunk_arr[i], output_chunk_arr[i])
        lcm = (input_chunk_arr[i] * output_chunk_arr[i]) // gcd
        expansion_increments[i] = output_chunk_arr[i] if lcm >= input_arr[i] else lcm
    
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
    
    dtype = _normalize_dtype(dtype, arr)
    if output_chunks is None:
        output_chunks = get_chunk_shape(arr)
    output_shards = _align_shards(output_shards, output_chunks)

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
    input_chunks = get_array_chunks(arr)

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
    
    _write_ngff_metadata(output_path, pixel_sizes)
    return ts_store


async def store_multiscale_async(
    ### base write params
    arr: Union[da.Array, zarr.Array],
    output_path: Union[Path, str],
    axes: Sequence[str],
    scales: Sequence[float],  # per-axis physical pixel sizes (flat, e.g. [1.0, 1.0, 0.5, 0.25, 0.25])
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
    smart_scale_factor: Optional[Tuple[int, ...]] = None,
    ### queue-based writer params
    num_readers: Optional[int] = None,      # Number of reader threads (default: 2 * max_concurrency)
    max_concurrency: Optional[int] = None,      # Number of writer threads (default: 4)
    region_size_mb: float = 8.0,            # Target size of read regions in MB
    queue_size: Optional[int] = None,       # Maximum queue size (default: adaptive)
    gc_interval: float = 15.0,              # Seconds between garbage collections
    max_concurrent_downscale_layers: int = 3,
    **kwargs
) -> 'ts.TensorStore':
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
    gr = _zarr_group(outpath, overwrite=overwrite, zarr_format=zarr_format)

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
        pixel_sizes=tuple(scales),  # Base layer scale
        num_readers=num_readers,
        max_concurrency=max_concurrency,
        region_size_mb=region_size_mb,
        queue_size=queue_size,
        gc_interval=gc_interval,
        overwrite=overwrite,
        verbose=verbose,
        output_shards=shards,
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
        
        try:
            pyr = await downscale_with_tensorstore_async(
                base_store=base_store_path,
                scale_factor=scale_factors,
                n_layers=n_layers,
                min_dimension_size=min_dimension_size,
                downscale_method=downscale_method,
                smart_scale_factor=smart_scale_factor,
                chunks=chunks,
                shards=shards,
                max_concurrency=max_concurrency,
                queue_size=queue_size,
                region_size_mb=region_size_mb,
                max_concurrent_downscale_layers=max_concurrent_downscale_layers,
                # dtype=dtype,
                **{k: v for k, v in kwargs.items() if k != 'max_concurrency'}
            )
            downscale_elapsed = (time.time() - downscale_start) / 60
            logger.info(f"Downscaling completed in {downscale_elapsed:.2f} minutes")
        except Exception as e:
            downscale_elapsed = (time.time() - downscale_start) / 60
            logger.error(f"Downscaling failed after {downscale_elapsed:.2f} minutes: {e}", exc_info=True)
            raise
    else:
        pyr = Pyramid(gr)

    return pyr
