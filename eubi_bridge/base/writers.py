import copy
import os, itertools, tempfile, shutil, threading, asyncio
import zarr, dask, numcodecs
from zarr import codecs
from zarr.storage import LocalStore
from dataclasses import dataclass
from dask import delayed
import dask.array as da
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Union, Any, Tuple, Optional
from distributed import get_client
### internal imports
from eubi_bridge.ngff.multiscales import NGFFMetadataHandler  #Multimeta
from eubi_bridge.utils.convenience import (
    get_chunksize_from_array,
    is_zarr_group,
    autocompute_chunk_shape,
    #, retry_decorator
)
from eubi_bridge.utils.logging_config import get_logger


import logging, warnings

logger = get_logger(__name__)

logging.getLogger('distributed.diskutils').setLevel(logging.CRITICAL)

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

def write_with_zarrpy(arr: da.Array,
                      store_path: Union[str, Path],
                      chunks: Optional[Tuple[int, ...]] = None,
                      shards: Optional[Tuple[int, ...]] = None,
                      dimension_names: str = None,
                      dtype: Any = None,
                      compressor: str = 'blosc',
                      compressor_params: dict = None,
                      rechunk_method: str = 'tasks',
                      overwrite: bool = True,
                      zarr_format: int = 2,
                      **kwargs
                      ) -> zarr.Array:
    """
    Write dask array to zarr storage using either zarr v2 or v3 format.

    Args:
        arr: Dask array to write
        store_path: Path where the Zarr array will be stored
        chunks: Chunk size for each dimension
        shards: Shard size for zarr v3 format
        dtype: Data type of the array (defaults to arr.dtype)
        compressor: Compression algorithm ('blosc' by default)
        compressor_params: Parameters for the compressor
        rechunk_method: Method for rechunking ('tasks' or 'p2p')
        zarr_format: Zarr format version (2 or 3)
        **kwargs: Additional arguments for array creation
    """
    store_path = str(store_path)
    dtype = dtype or arr.dtype
    compressor_params = compressor_params or {}

    if chunks is None:
        chunks = arr.chunksize

    chunks = tuple(int(size) for size in chunks)

    if shards is None:
        shards = copy.deepcopy(chunks)

    if not np.allclose(np.mod(shards, chunks), 0):
        multiples = np.floor_divide(shards, chunks)
        shards = np.multiply(multiples, chunks)

    shards = tuple(int(size) for size in np.ravel(shards))

    if zarr_format == 2:
        if not np.equal(arr.chunksize, chunks).all():
            arr = arr.rechunk(chunks, method=rechunk_method)
    else:  # zarr_format == 3
        if shards is not None and not np.equal(arr.chunksize, shards).all():
            arr = arr.rechunk(shards, method=rechunk_method)

    compressor_config = CompressorConfig(name=compressor,
                                         params=compressor_params)

    zarr_array = _create_zarr_array(
        store_path=store_path,
        shape=arr.shape,
        chunks=chunks,
        dtype=dtype,
        overwrite=overwrite,
        compressor_config = compressor_config,
        zarr_format=zarr_format,
        shards=shards,
        dimension_names=dimension_names,
        **kwargs
    )
    res = arr.map_blocks(write_chunk_with_zarrpy, zarr_array=zarr_array, dtype=dtype)

    return res

def write_chunk_with_tensorstore(chunk: np.ndarray,
                                 ts_store,
                                 block_info: Dict
                                 ) -> None:
    if hasattr(chunk, "get"):
        chunk = chunk.get()  # Convert CuPy -> NumPy
    fut = ts_store[tuple(slice(*b) for b in block_info[0]["array-location"])].write(chunk)
    fut.result()
    return

def write_with_tensorstore(
    arr: da.Array,
    store_path: Union[str, Path],
    chunks: Optional[Tuple[int, ...]] = None,
    shards: Optional[Tuple[int, ...]] = None,
    dimension_names: str = None,
    dtype: Any = None,
    compressor: str = 'blosc',
    compressor_params: dict = None,
    rechunk_method: str = 'tasks',
    overwrite: bool = True,
    zarr_format: int = 2,
    **kwargs
) -> da.Array:
    """
    Write dask array to zarr storage using tensorstore with support for both zarr v2 and v3 formats.
    """
    try:
        import tensorstore as ts
    except ImportError:
        raise ModuleNotFoundError(
            "The module tensorstore has not been found. "
            "Try 'conda install -c conda-forge tensorstore'"
        )

    compressor_params = compressor_params or {}
    # shard_to_chunk_ratio = kwargs.get('shard_to_chunk_ratio', 3)
    dtype = dtype or arr.dtype
    fill_value = kwargs.get('fill_value', get_default_fill_value(dtype))

    if chunks is None:
        chunks = arr.chunksize

    chunks = tuple(int(size) for size in chunks)

    if shards is None:
        shards = copy.deepcopy(chunks)

    if not np.allclose(np.mod(shards, chunks), 0):
        multiples = np.floor_divide(shards, chunks)
        shards = np.multiply(multiples, chunks)

    shards = tuple(int(size) for size in np.ravel(shards))

#    Rechunk if needed
    if zarr_format == 2:
        if not np.equal(arr.chunksize, chunks).all():
            arr = arr.rechunk(shards, method=rechunk_method)
    else:  # zarr_format == 3
        if shards is not None and not np.equal(arr.chunksize, shards).all():
            arr = arr.rechunk(shards, method=rechunk_method)

    # Prepare zarr metadata
    if zarr_format == 3:
        # Only include array-to-array codecs such as blosc
        zarr_metadata = {
            "data_type": np.dtype(dtype).name,
            "shape": arr.shape,
            "chunk_grid": {
                "name": "regular",
                "configuration": {
                    "chunk_shape": shards if zarr_format == 3 else chunks,
                }
            },
            "dimension_names": list(dimension_names),
            "codecs": [
                {
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": chunks,
                        "codecs": [{"name": "bytes",
                                    "configuration": {"endian": "little"}},
                                   {"name": compressor,
                                    "configuration": compressor_params or {}}],
                        "index_codecs": [{"name": "bytes",
                                          "configuration": {"endian": "little"}},
                                         {"name": "crc32c"}],
                        "index_location": "end"
                    }
                }
            ],
            "node_type": "array"
        }

    else:  # zarr_format == 2
        zarr_metadata = {
            "compressor": {
                "id": compressor,
                **compressor_params
            },
            "dtype": np.dtype(dtype).str,
            "shape": arr.shape,
            "chunks": chunks,
            "fill_value": fill_value,
            "dimension_separator": '/',
        }

    zarr_spec = {
        "driver": "zarr" if zarr_format == 2 else "zarr3",
        "kvstore": {
            "driver": "file",
            "path": str(store_path),
        },
        "metadata": zarr_metadata,
        "create": True,
        "delete_existing": overwrite,
    }

    ts_store = ts.open(zarr_spec).result()

    return arr.map_blocks(write_chunk_with_tensorstore,
                          ts_store=ts_store,
                          dtype=dtype)


def compute_blocks(arr, block_shape):
    """Return slices defining large blocks over the array."""
    slices_per_dim = [range(0, s, b) for
                      s, b in zip(arr.shape, block_shape)]
    blocks = []
    for starts in itertools.product(*slices_per_dim):
        block_slices = tuple(slice(start, min(start+b, dim)) for start, b, dim in zip(starts, block_shape, arr.shape))
        blocks.append(block_slices)
    return blocks

@delayed
def count_threads():
    return threading.active_count()


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

def store_arrays(arrays: Dict[str, Dict[str, da.Array]], # flatarrays
                 output_path: Union[Path, str],
                 axes: list, # flataxes
                 scales: Dict[str, Dict[str, Tuple[float, ...]]], # flatscales
                 units: list, # flatunits
                 auto_chunk: bool = True,
                 output_chunks: Dict[str,Tuple[int, ...]] = None,
                 output_shard_coefficients: Tuple[int, ...] = None,
                 compute: bool = False,
                 overwrite: bool = False,
                 channel_meta: dict = None,
                 **kwargs # shard_to_chunk_ratio and zarr_format should specified inside kwargs
                 ) -> Dict[str, da.Array]:

    rechunk_method = kwargs.get('rechunk_method', 'tasks')
    if rechunk_method == 'rechunker':
        raise ValueError(f"This version of EuBI-Bridge does not support rechunker. Choose either of 'tasks' or 'p2p'.")
    assert rechunk_method in ('tasks', 'p2p')
    use_tensorstore = kwargs.get('use_tensorstore', False)
    verbose = kwargs.get('verbose', False)
    zarr_format = kwargs.get('zarr_format', 2)
    output_shards = kwargs.get('output_shards', None)
    target_chunk_mb = kwargs.get('target_chunk_mb', 1)

    writer_func = write_with_tensorstore if use_tensorstore else write_with_zarrpy
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    zarr.group(output_path, 
               overwrite=overwrite, 
               zarr_version = zarr_format)
    results = {}

    for key, arr in arrays.items():
        dtype = kwargs.get('dtype', arr.dtype)
        if dtype is None:
            dtype = arr.dtype
        else:
            if isinstance(dtype, str):
                dtype = np.dtype(dtype)
            arr = arr.astype(dtype)

        flataxes = axes[key]
        flatscale = scales[key]
        flatunit = units[key]
        flatchunks = output_chunks[key]

        if auto_chunk:
            flatchunks = autocompute_chunk_shape(arr.shape,
                                                 axes=flataxes,
                                                 target_chunk_mb=target_chunk_mb,
                                                 dtype=dtype
                                                 )
            if verbose:
                logger.info(f"Auto-chunking {key} to {flatchunks}")

        flatchannels = channel_meta[key] if channel_meta is not None else None

        chunks = np.minimum(flatchunks or arr.chunksize, arr.shape).tolist()
        chunks = tuple([int(item) for item in chunks])

        # if zarr_format == 3:
        if output_shards is not None:
            shards = output_shards[key]
        elif output_shard_coefficients is not None:
            flatshardcoefs = output_shard_coefficients[key]
            shards = np.multiply(chunks, flatshardcoefs)
        else:
            shards = chunks

            shards = tuple([int(item) for item in shards])
        # else:
        #     shards = None

        dirpath = os.path.dirname(key)
        arrpath = os.path.basename(key)

        if is_zarr_group(dirpath):
            gr = zarr.open_group(dirpath, mode='a')
        else:
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            gr = zarr.group(dirpath, 
                            overwrite=overwrite, 
                            zarr_version = zarr_format)

        version = '0.5' if zarr_format == 3 else '0.4'

        meta = _get_or_create_multimeta(gr,
                                        axis_order=flataxes,
                                        unit_list=flatunit,
                                        version=version
                                        )

        meta.add_dataset(path=arrpath,
                         scale=flatscale,
                         overwrite=True)
        meta.retag(os.path.basename(dirpath))

        if flatchannels == 'auto':
            if 'c' in flataxes:
                idx = flataxes.index('c')
                size = arr.shape[idx]
            else:
                size = 1
            meta.autocompute_omerometa(size, arr.dtype)
        elif flatchannels is None:
            pass
        else:
            for channel in flatchannels:
                meta.add_channel(color = channel['color'],
                                 label = channel['label'],
                                 dtype = dtype.str
                                 )
        meta.save_changes()

        if verbose:
            logger.info(f"Writer function: {writer_func}")
            logger.info(f"Rechunk method: {rechunk_method}")

        results[key] = writer_func(arr=arr,
                                   store_path=key,  # compressor = compressor, dtype = dtype,
                                   chunks=chunks,
                                   shards = shards,
                                   dimension_names = flataxes,
                                   overwrite=overwrite,
                                   **kwargs
                                   )

    if compute:
        # try:
            # dask.compute(list(results.values()), retries = 6)

        dask.compute(
            list(results.values()),
            retries=6,
        )
        # except Exception as e:
        #     # print(e)
        #     pass
    else:
        return results
    return results

