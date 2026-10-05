"""Array processing utilities for Dask, Zarr, and NumPy arrays."""
from __future__ import annotations

import itertools
import math
import os
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
import zarr

from eubi_bridge.utils.optional_deps import require

if TYPE_CHECKING:
    from dask import array as da

try:
    import cupy as cp
    cupy_available = True
except ImportError:
    cupy_available = False

from eubi_bridge.utils.logging_config import get_logger

logger = get_logger(__name__)


def asdask(data, chunks='auto') -> da.Array:
    """Convert array-like data to Dask array.
    
    Args:
        data: Array-like object (Zarr, NumPy, or Dask array)
        chunks: Chunk specification for Dask (default: 'auto')
    
    Returns:
        Dask array
    """
    da = require("dask.array", "Converting to a dask array")
    assert isinstance(data, (da.Array, zarr.Array, np.ndarray)), \
        f'data must be of type: {da.Array, zarr.Array, np.ndarray}'
    if isinstance(data, zarr.Array):
        return da.from_zarr(data)
    elif isinstance(data, np.ndarray):
        return da.from_array(data, chunks=chunks)
    return data


def as_dask_array(
    array: Union[da.Array, zarr.Array, np.ndarray],
    backend: str = 'numpy',
    **params
) -> da.Array:
    """Convert array-like data to Dask array with optional GPU backend.
    
    Args:
        array: Input array (Dask, Zarr, or NumPy)
        backend: 'numpy' or 'cupy' (requires cupy installed)
        **params: Additional parameters for Dask from_array()
    
    Returns:
        Dask array
    
    Raises:
        TypeError: If array type is not supported
        ValueError: If cupy backend requested but not available
    """
    da = require("dask.array", "Converting to a dask array")
    if cupy_available:
        assert isinstance(array, (da.Array, zarr.Array, np.ndarray, cp.ndarray)), \
            f"The given array type {type(array)} cannot be parsed."
    else:
        assert isinstance(array, (da.Array, zarr.Array, np.ndarray)), \
            f"The given array type {type(array)} cannot be parsed."
    assert backend in ('numpy', 'cupy'), \
        f"Currently, the only supported backends are 'numpy' or 'cupy'."

    if not isinstance(array, da.Array):
        out = da.from_array(array, **params)
    else:
        out = array

    if backend == 'cupy':
        if cupy_available:
            out = out.map_blocks(cp.asarray)
        else:
            raise ValueError("cupy is not available!")

    return out


def _stats_blocks(shape, chunks, itemsize: int, max_bytes: int):
    """Blocks covering *shape*, each a whole number of *chunks* (the array's
    storage chunks, so no chunk is decompressed twice), grown from the last
    axis while one block stays within *max_bytes*."""
    ndim = len(shape)
    chunks = [max(1, min(int(c), int(s))) for c, s in zip(chunks, shape)]
    block = list(chunks)
    for axis in reversed(range(ndim)):
        while block[axis] < shape[axis]:
            grown = min(shape[axis], block[axis] + chunks[axis])
            if int(np.prod(block[:axis] + [grown] + block[axis + 1:])) * itemsize > max_bytes:
                break
            block[axis] = grown
        if block[axis] < shape[axis]:
            break
    starts = [range(0, int(s), int(b)) for s, b in zip(shape, block)]
    for corner in itertools.product(*starts):
        yield tuple(slice(c, min(int(s), c + b)) for c, s, b in zip(corner, shape, block))


def channel_extrema(array, c_index: Optional[int],
                    max_bytes: int = 64 * 1024 ** 2,
                    threads: Optional[int] = None) -> Tuple[list, list]:
    """Exact minimum and maximum of each channel of *array* -- numpy, zarr,
    DynamicArray, or anything that gives numpy through ``np.asarray(a[key])``.

    Read in blocks aligned to the array's own chunks (a plane-by-plane read
    of a 3-D-chunked store decompresses every chunk once per plane it holds:
    101x the work for 101-deep chunks), each of at most *max_bytes*, on
    *threads* threads (default: the CPU count, at most 8).  *c_index*: the
    channel axis, or None (one channel).  Returns ([min per channel], [max
    per channel])."""
    import concurrent.futures
    shape = tuple(int(n) for n in array.shape)
    itemsize = np.dtype(array.dtype).itemsize
    chunks = get_array_chunks(array)
    if chunks is None or len(chunks) != len(shape):
        chunks = (1,) * (len(shape) - 2) + tuple(shape[-2:])     # planes
    n_channels = shape[c_index] if c_index is not None else 1
    reduce_axes = tuple(i for i in range(len(shape)) if i != c_index)

    def extrema(key):
        block = np.asarray(array[key])
        if block.size == 0:
            return key, None, None
        if c_index is None:
            return key, np.atleast_1d(block.min()), np.atleast_1d(block.max())
        return key, block.min(axis=reduce_axes), block.max(axis=reduce_axes)

    lows: list = [None] * n_channels
    highs: list = [None] * n_channels
    workers = threads or min(8, os.cpu_count() or 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for key, low, high in pool.map(extrema, _stats_blocks(shape, chunks, itemsize, max_bytes)):
            if low is None:
                continue
            first = key[c_index].start if c_index is not None else 0
            for k in range(len(low)):
                ch = first + k
                lows[ch] = low[k] if lows[ch] is None else min(lows[ch], low[k])
                highs[ch] = high[k] if highs[ch] is None else max(highs[ch], high[k])
    return [v.item() for v in lows], [v.item() for v in highs]


def get_array_size(array, as_str: bool = True) -> Union[str, int]:
    """Calculate total size of array in bytes or formatted string.
    
    Args:
        array: NumPy, Dask, or Zarr array
        as_str: If True, return formatted string (e.g., "2.5GB"), else return bytes
    
    Returns:
        Formatted string like "2.5GB" or integer bytes
    """
    voxelcount = np.prod(array.shape)
    arraysize = voxelcount * array.dtype.itemsize
    if as_str:
        return f"{arraysize / 1024 ** 3:.2f}GB"
    else:
        return arraysize


def sizeof(array, unit: str = 'gb') -> float:
    """Get array size in specified units.
    
    Args:
        array: NumPy, Dask, or Zarr array
        unit: 'gb', 'mb', or 'kb'
    
    Returns:
        Size in specified units
    """
    unit = unit.lower()
    assert unit in ('gb', 'mb', 'kb'), f"Unit must be 'gb', 'mb', or 'kb', got {unit}"
    
    bytes_size = get_array_size(array, False)
    
    if unit == 'gb':
        return bytes_size / (1024 ** 3)
    elif unit == 'mb':
        return bytes_size / (1024 ** 2)
    elif unit == 'kb':
        return bytes_size / 1024
    else:
        return bytes_size


def get_array_chunks(arr) -> Optional[Tuple[int, ...]]:
    """Extract a flat chunk-shape tuple from any supported array type.

    Handles:
    - ``zarr.Array``       — ``.chunks`` is already a flat tuple.
    - ``dask.Array``       — ``.chunksize`` returns the first-chunk sizes per dim.
    - TensorStore spec     — ``.chunk_layout.read_chunk.shape``.
    - ``DynamicArray``     — ``.chunks`` property (may itself delegate to zarr/ts).

    Returns ``None`` when no chunk information can be retrieved so callers can
    fall back gracefully.
    """
    # TensorStore has chunk_layout; check first to avoid ambiguity with .chunks
    if hasattr(arr, 'chunk_layout'):
        try:
            return tuple(arr.chunk_layout.read_chunk.shape)
        except Exception:
            pass
    # dask.Array: .chunksize gives first-chunk sizes per dim (flat tuple)
    if hasattr(arr, 'chunksize'):
        return tuple(arr.chunksize)
    # zarr.Array and DynamicArray: .chunks is flat tuple of ints
    if hasattr(arr, 'chunks'):
        chunks = arr.chunks
        if chunks is not None:
            # Dask-style tuple-of-tuples? Flatten to first element per dim.
            if chunks and isinstance(chunks[0], tuple):
                return tuple(c[0] for c in chunks)
            return tuple(chunks)
    return None


def get_chunk_shape(arr,
                    default_chunks: Optional[Tuple[int, ...]] = None
                    ) -> Tuple[int, ...]:
    """Extract chunk shape from various array types.

    Args:
        arr: Dask, Zarr, or array-like object

    Returns:
        Tuple of chunk dimensions
    """
    chunks = get_array_chunks(arr)
    if chunks is not None:
        return chunks
    if default_chunks is not None:
        return default_chunks
    logger.warning('Array has no chunk shape, using full shape')
    return tuple(arr.shape)


def get_chunksize_from_array(arr) -> str:
    """Get chunk size as formatted string.
    
    Args:
        arr: Array with chunks attribute
    
    Returns:
        Formatted string like "1.2GB"
    """
    chunks = get_chunk_shape(arr)
    itemsize = arr.dtype.itemsize
    chunk_size = itemsize * np.prod(chunks)
    chunk_size_gb = chunk_size / (1024 ** 3)
    return f'{chunk_size_gb * 1.1:.2f}GB'


def parse_memory(gb_string: Union[str, int, float]) -> float:
    """Convert memory size string to megabytes.
    
    Args:
        gb_string: String like '5GB', '1.2GB', or numeric value
    
    Returns:
        Size in megabytes
    """
    if isinstance(gb_string, (int, float)):
        # Assume already in MB
        return float(gb_string)
    
    if isinstance(gb_string, str):
        gb_string = gb_string.strip().upper()
        if gb_string.isnumeric():
            # Already in MB
            return float(gb_string)
        if not gb_string.endswith('GB'):
            # Assume MB
            return float(gb_string)
        
        # Extract numeric part and convert from GB to MB
        number_part = gb_string[:-2]  # remove 'GB'
        return float(number_part) * 1024
    
    return float(gb_string)


def autocompute_chunk_shape(
    array_shape: Tuple[int, ...],
    axes: str,
    target_chunk_mb: float = 1.0,
    dtype: type = np.uint16,
) -> Tuple[int, ...]:
    """Compute optimal chunk shape for array while respecting memory budget.
    
    For spatial dimensions (x, y, z), aims for isotropic chunks.
    Non-spatial dimensions (t, c, etc.) remain at size 1.
    
    Args:
        array_shape: Shape of the array to chunk
        axes: String describing axes (e.g., 'tzyx')
        target_chunk_mb: Target chunk size in megabytes
        dtype: Data type of array elements
    
    Returns:
        Tuple representing the chunk shape
    
    Raises:
        ValueError: If array_shape length doesn't match axes length
    """
    if len(array_shape) != len(axes):
        raise ValueError("Length of array_shape must match length of axes.")

    chunk_bytes = int(target_chunk_mb * 1024 * 1024)
    element_size = np.dtype(dtype).itemsize
    max_elements = chunk_bytes // element_size

    chunk_shape = [1] * len(array_shape)
    spatial_indices = [i for i, ax in enumerate(axes) if ax in 'xyz']

    if spatial_indices:
        # Estimate isotropic side length
        s = int(np.floor(max_elements ** (1.0 / len(spatial_indices))))
        for i in spatial_indices:
            chunk_shape[i] = min(s, array_shape[i])

        # Safely grow dimensions while staying within the element budget
        while True:
            trial_shape = list(chunk_shape)
            for i in spatial_indices:
                if trial_shape[i] < array_shape[i]:
                    trial_shape[i] += 1

            trial_elements = np.prod([trial_shape[i] for i in spatial_indices])
            if trial_elements <= max_elements and trial_shape != chunk_shape:
                chunk_shape = trial_shape
            else:
                break

        # Final safety trim if somehow over
        while np.prod([chunk_shape[i] for i in spatial_indices]) > max_elements:
            for i in reversed(spatial_indices):  # Trim z first
                if chunk_shape[i] > 1:
                    chunk_shape[i] -= 1

    return tuple(chunk_shape)


def compute_chunk_batch(
    chunked_array,
    dtype: Union[np.dtype, str, type],
    memory_limit_mb: Union[str, int, float],
) -> Tuple[int, ...]:
    """Calculate optimal chunk batch sizes while staying within memory limits.
    
    Maximizes isotropy of chunk batching while respecting memory constraints.
    
    Args:
        chunked_array: Zarr or Dask array with chunks
        dtype: Data type of the array elements
        memory_limit_mb: Memory limit in MB (can be string like '5GB')
    
    Returns:
        Tuple of integers specifying the chunk batch size in each dimension
    """
    memory_limit_mb = parse_memory(memory_limit_mb)

    # Get array properties
    array_shape = chunked_array.shape
    chunk_shape = get_chunk_shape(chunked_array)

    # Convert dtype to numpy dtype and get itemsize
    np_dtype = np.dtype(dtype)
    itemsize = np_dtype.itemsize

    # Calculate memory per chunk
    chunk_size = np.prod(chunk_shape)
    memory_per_chunk = chunk_size * itemsize
    memory_per_chunk_mb = memory_per_chunk / (1024 ** 2)

    if memory_per_chunk_mb > memory_limit_mb:
        logger.warning(
            f"For the array with shape {array_shape} and dtype {dtype},\n"
            f"the size of an input chunk ({memory_per_chunk_mb:.2f} MB) "
            f"exceeds target memory ({memory_limit_mb:.2f} MB).\n"
            f"Target memory is temporarily increased to {memory_per_chunk_mb:.2f} MB."
        )
        memory_limit_mb = memory_per_chunk_mb

    # Maximum number of chunks that fit in memory
    max_chunks = memory_limit_mb // memory_per_chunk_mb

    # Calculate maximum chunks per dimension
    max_chunks_per_dim = []
    for i, (array_dim, chunk_dim) in enumerate(zip(array_shape, chunk_shape)):
        max_chunks_in_dim = math.ceil(array_dim / chunk_dim)
        max_chunks_per_dim.append(max_chunks_in_dim)

    # Find the most isotropic distribution of chunks
    ndims = len(array_shape)
    target_chunks_per_dim = max_chunks ** (1.0 / ndims)

    # Initialize with minimum (1 chunk per dimension)
    batch_chunks = [1] * ndims

    # Greedily increase dimensions to approach isotropy
    while True:
        current_total = np.prod(batch_chunks)
        if current_total >= max_chunks:
            break

        # Find dimension that's furthest from target ratio
        ratios = [batch_chunks[i] / target_chunks_per_dim for i in range(ndims)]
        min_ratio_idx = np.argmin(ratios)

        # Check if we can increase this dimension
        if batch_chunks[min_ratio_idx] < max_chunks_per_dim[min_ratio_idx]:
            test_batch = batch_chunks.copy()
            test_batch[min_ratio_idx] += 1

            if np.prod(test_batch) <= max_chunks:
                batch_chunks[min_ratio_idx] += 1
            else:
                break
        else:
            # This dimension is maxed out, try next best
            available_dims = [i for i in range(ndims)
                              if batch_chunks[i] < max_chunks_per_dim[i]]
            if not available_dims:
                break

            available_ratios = [(ratios[i], i) for i in available_dims]
            _, next_best_idx = min(available_ratios)

            test_batch = batch_chunks.copy()
            test_batch[next_best_idx] += 1

            if np.prod(test_batch) <= max_chunks:
                batch_chunks[next_best_idx] += 1
            else:
                break

    # Convert to actual sizes (multiply by chunk dimensions)
    batch_sizes = tuple(batch_chunks[i] * chunk_shape[i] for i in range(ndims))

    return batch_sizes


def normalise_basic_index(key, shape):
    """numpy basic-index key -> (per-axis (start, stop) spans, squeeze axes, steps).

    The source is read as contiguous spans; integer indices and steps are
    applied to the result in memory afterwards.
    """
    if not isinstance(key, tuple):
        key = (key,)
    if any(k is Ellipsis for k in key):
        at = next(i for i, k in enumerate(key) if k is Ellipsis)
        key = key[:at] + (slice(None),) * (len(shape) - len(key) + 1) + key[at + 1:]
    if len(key) > len(shape):
        raise IndexError(f"too many indices: array is {len(shape)}-D, got {len(key)}")
    key = key + (slice(None),) * (len(shape) - len(key))

    spans, squeeze, steps = [], [], []
    for axis, (k, size) in enumerate(zip(key, shape)):
        if isinstance(k, (int, np.integer)):
            index = int(k) + size if k < 0 else int(k)
            if not 0 <= index < size:
                raise IndexError(f"index {k} is out of bounds for axis {axis} "
                                 f"with size {size}")
            spans.append((index, index + 1))
            squeeze.append(axis)
            steps.append(1)
        elif isinstance(k, slice):
            start, stop, step = k.indices(size)
            if step < 0:
                raise NotImplementedError("negative steps are not supported")
            spans.append((start, max(start, stop)))
            steps.append(step)
        else:
            raise NotImplementedError(
                f"unsupported index type {type(k).__name__} for a Bio-Formats image")
    return spans, tuple(squeeze), steps
