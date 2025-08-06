import os, json, glob, time
import warnings

import numpy as np, pandas as pd
try:
    import cupy as cp
    cupy_available = True
except:
    cupy_available = False
import zarr, json, shutil, os, copy, zarr
from dask import array as da
import dask
import numcodecs
import shutil, tempfile
from pathlib import Path
from typing import List

from typing import (
    Union,
    Tuple,
    Dict,
    Any,
    Iterable,
    List,
    Optional
)

def asstr(s):
    if isinstance(s, str):
        return s
    elif isinstance(s, int):
        return str(s)
    else:
        raise TypeError(f"Input must be either of types {str, int}")

def asdask(data, chunks = 'auto'):
    assert isinstance(data, (da.Array, zarr.Array, np.ndarray)), f'data must be of type: {da.Array, zarr.Array, np.ndarray}'
    if isinstance(data, zarr.Array):
        return da.from_zarr(data)
    elif isinstance(data, np.ndarray):
        return da.from_array(data, chunks = chunks)
    return data

def path_has_pyramid(path):
    try:
        store = zarr.storage.LocalStore(path)
        _ = zarr.open_group(store, mode = 'r')
        return True
    except:
        return False


def parse_as_list(path_or_paths: Union[Iterable, str, int, float]
                   ):
    if isinstance(path_or_paths, (str, int, float)):
        inputs = [path_or_paths]
    else:
        inputs = path_or_paths
    return inputs

def includes(group1: Union[Iterable, str, int, float],
             group2: Union[Iterable, str, int, float]
             ):
    """Convenience function that checks if group1 includes group2 completely."""
    gr1 = parse_as_list(group1)
    gr2 = parse_as_list(group2)
    return all([item in gr1 for item in gr2])

def insert_at_indices(iterable1, iterable2, indices):
    if not hasattr(iterable1, '__len__'):
        iterable1 = [iterable1]
    if not hasattr(iterable2, '__len__'):
        iterable2 = [iterable2]
    if not hasattr(indices, '__len__'):
        indices = [indices]
    endlen = (len(iterable1) + len(iterable2))
    end_indices = [None] * endlen
    other_indices = [i for i in range(endlen) if i not in indices]
    for i, j in zip(other_indices, iterable1):
        end_indices[i] = j
    for i, j in zip(indices, iterable2):
        end_indices[i] = j
    return end_indices

def index_nth_dimension(array,
                        dimensions = 2, # a scalar or iterable
                        intervals = None # a scalar, an iterable of scalars, a list of tuple or None
                        ):
    if isinstance(array, zarr.Array):
        array = da.from_zarr(array)
    allinds = np.arange(array.ndim).astype(int)
    if np.isscalar(dimensions):
        dimensions = [dimensions]
    if intervals is None or np.isscalar(intervals):
        intervals = np.repeat(intervals, len(dimensions))
    assert len(intervals) == len(dimensions) ### KALDIM
    interval_dict = {item: interval for item, interval in zip(dimensions, intervals)}
    shape = array.shape
    slcs = []
    for idx, dimlen in zip(allinds, shape):
        if idx not in dimensions:
            slc = slice(dimlen)
        else:
            try:
                slc = slice(interval_dict[idx][0], interval_dict[idx][1])
            except:
                slc = interval_dict[idx]
        slcs.append(slc)
    slcs = tuple(slcs)
    indexed = array[slcs]
    return indexed

def transpose_dict(dictionary):
    keys, values = [], []
    for key, value in dictionary.items():
        keys.append(key)
        values.append(value)
    return keys, values

def argsorter(s):
    return sorted(range(len(s)), key = lambda k: s[k])


def is_zarr_array(path: (str, Path)
                  ):
    try:
        _ = zarr.open_array(path, mode = 'r')
        return True
    except:
        return False

######## GROUP UTILITIES BELOW
def is_zarr_group(path: (str, Path)
                  ):
    try:
        _ = zarr.open_group(path, mode = 'r')
        return True
    except:
        return False

def is_generic_collection(group):
    res = False
    basepath = group.store.path
    basename = os.path.basename(basepath)
    paths = list(group.keys())
    attrs = dict(group.attrs)
    attrkeys, attrvalues = transpose_dict(attrs)
    if basename in attrkeys and (len(paths) > 0):
        if len(attrs[basename]) == len(paths):
            res = True
            for item0, item1 in zip(attrs[basename], paths):
                if item0 != item1:
                    res = False
    return res

def get_collection_paths(directory,
                         return_all = False
                         ):
    gr = zarr.group(directory)
    groupkeys = list(gr.group_keys())
    arraykeys = list(gr.array_keys())
    grouppaths = [os.path.join(directory, item) for item in groupkeys]
    arraypaths = [os.path.join(directory, item) for item in arraykeys]
    collection_paths = []
    multiscales_paths = []
    while len(grouppaths) > 0:
        if is_generic_collection(gr) or 'bioformats2raw.layout' in gr.attrs:
            collection_paths.append(directory)
        if 'multiscales' in list(gr.attrs.keys()):
            multiscales_paths.append(directory)
        directory = grouppaths[0]
        grouppaths.pop(0)
        gr = zarr.group(directory)
        groupkeys = list(gr.group_keys())
        arraykeys = list(gr.array_keys())
        grouppaths += [os.path.join(directory, item) for item in groupkeys]
        arraypaths += [os.path.join(directory, item) for item in arraykeys]
    if is_generic_collection(gr) or 'bioformats2raw.layout' in gr.attrs:
        collection_paths.append(directory)
    if 'multiscales' in list(gr.attrs.keys()):
        multiscales_paths.append(directory)
    out = [item for item in collection_paths]
    for mpath in multiscales_paths:
        s = os.path.dirname(mpath)
        if s in collection_paths:
            pass
        else:
            if mpath not in out:
                out.append(mpath)
    if return_all:
        return out, multiscales_paths, arraypaths
    return out

def convert_np_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def is_valid_json(my_dict):
    try:
        json.dumps(my_dict)
        return True
    except:
        warnings.warn(f"Object is not a valid json!")
        return False

def turn2json(my_dict):
    stringified = json.dumps(my_dict, default = convert_np_types)
    return json.loads(stringified)


def as_store(store: (zarr.storage.StoreLike, zarr.Array, Path, str) # TODO: other scenarios?
                ):
    assert isinstance(store, (zarr.storage.Store, zarr.Array, Path, str)), f"The given store cannot be parsed."
    if isinstance(store, (Path, str)):
        out = zarr.storage.LocalStore(store, dimension_separator = '/')
    else:
        out = store
    return out

def as_dask_array(array: (da.Array, zarr.Array, np.ndarray,
                          # cp.ndarray
                          ),
                  backend = 'numpy',
                  **params
                  ):
    if cupy_available:
        assert isinstance(array, (da.Array, zarr.Array, np.ndarray, cp.ndarray)), f"The given array type {type(array)} cannot be parsed."
    else:
        assert isinstance(array, (da.Array, zarr.Array, np.ndarray)), f"The given array type {type(array)} cannot be parsed."
    assert backend in ('numpy', 'cupy'), f"Currently, the only supported backends are 'numpy' or 'cupy'."

    if not isinstance(array, da.Array):
        out = da.from_array(array, **params)

    if backend == 'cupy':
        if cupy_available:
            out = out.map_blocks(cp.asarray)
        else:
            raise ValueError("cupy is not available!")

    return out


#########################################################

def get_array_size(array, as_str = True):
    voxelcount = np.prod(array.shape)
    arraysize = voxelcount * array.dtype.itemsize
    if as_str:
        return f"{arraysize / 1024 ** 3}GB"
    else:
        return arraysize

def sizeof(array, unit = 'gb'):
    unit = unit.lower()
    assert unit in ('gb', 'mb', 'kb')
    bytes = get_array_size(array, False)
    if unit == 'gb':
        ret = bytes / 1024 ** 3
    elif unit == 'mb':
        ret = bytes / 1024 ** 2
    elif unit == 'kb':
        ret = bytes / 1024 ** 1
    else:
        ret = bytes
    return ret

def get_chunksize_from_array(arr):
    chunks = arr.chunksize if isinstance(arr, da.Array) else arr.chunks
    itemsize = arr.dtype.itemsize
    chunk_size = itemsize * np.prod(chunks)
    chunk_size_gb = chunk_size / (1024 ** 3)
    return f'{chunk_size_gb * 1.1}GB'


def retry_decorator(retries=3, delay=1, exceptions=(Exception,)):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    if attempt < retries - 1:
                        time.sleep(delay)
                    else:
                        raise
        return wrapper
    return decorator



def sensitive_glob(pattern: str,
                   recursive: bool = False,
                   sensitive_to: str = '.zarr'
                   ) -> List[str]:
    results = []

    for start_path in glob.glob(pattern, recursive=recursive):
        def _walk(current_path):
            if os.path.isfile(current_path):
                results.append(current_path)
                return
            if os.path.isdir(current_path):
                if current_path.endswith(sensitive_to):
                    results.append(current_path)
                    return
                for entry in os.listdir(current_path):
                    entry_path = os.path.join(current_path, entry)
                    _walk(entry_path)

        _walk(start_path)

    return results


def take_filepaths_from_csv(input_path: str,
                            includes: bool = None,
                            excludes: bool = None
                            ) -> list:
    """Take filepaths from a CSV file.

    Supports two formats:
    1. With header row containing 'filepath' column:
       filepath,other_column
       /path/to/file.tif,value

    2. Without header (takes first column):
       /path/to/file.tif
       /another/path.tif
    """
    import csv

    with open(input_path, 'r') as f:
        # Try to read with headers first
        start_pos = f.tell()
        try:
            sniffer = csv.Sniffer()
            has_header = sniffer.has_header(f.read(1024))
            f.seek(start_pos)

            if has_header:
                reader = csv.DictReader(f)
                if 'filepath' not in reader.fieldnames:
                    raise ValueError(f"CSV has headers but no 'filepath' column. Found columns: {reader.fieldnames}")
                paths = [row['filepath'] for row in reader]
            else:
                # No headers, read first column
                reader = csv.reader(f)
                paths = [row[0] for row in reader if row]  # Skip empty lines

        except csv.Error as e:
            # Fallback to reading as simple text file if CSV parsing fails
            f.seek(start_pos)
            paths = [line.strip() for line in f if line.strip()]  # Read non-empty lines

    paths = list(filter(lambda path: (includes in path if includes is not None else True)
                                     and
                                     (excludes not in path if excludes is not None else True),
                        paths
                        )
                 )
    paths = list(filter(lambda path: not path.endswith('zarr.json'), paths))
    if len(paths) == 0:
        raise ValueError(f"No valid paths found in {input_path}")
    return sorted(paths)


def take_filepaths(input_path: str,
                   includes: bool = None,
                   excludes: bool = None
                   ):

    if os.path.isfile(input_path) and input_path.endswith('.csv'):
        return take_filepaths_from_csv(input_path, includes, excludes)

    original_input_path = input_path

    if os.path.isfile(input_path) or input_path.endswith('.zarr'):
        dirname = os.path.dirname(input_path)
        basename = os.path.basename(input_path)
        if len(dirname) == 0:
            dirname = '.'
        input_path = f"{dirname}/*{basename}"

    if not '*' in input_path and not input_path.endswith('.zarr'):
        input_path = os.path.join(input_path, '**')

    if not '*' in input_path:
        input_path_ = os.path.join(input_path, '**')
    else:
        input_path_ = input_path
    paths = sensitive_glob(input_path_, recursive=False, sensitive_to='.zarr')

    paths = list(filter(lambda path: (includes in path if includes is not None else True)
                                     and
                                     (excludes not in path if excludes is not None else True),
                        paths
                        )
                 )
    paths = list(filter(lambda path: not path.endswith('zarr.json'), paths))
    if len(paths) == 0:
        raise ValueError(f"No valid paths found for {original_input_path}")
    return sorted(paths)

import numpy as np
from typing import Tuple

def autocompute_chunk_shape(
    array_shape: Tuple[int, ...],
    axes: str,
    target_chunk_mb: float = 1.0,
    dtype: type = np.uint16,
) -> Tuple[int, ...]:
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

