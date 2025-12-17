"""Miscellaneous utility functions for data processing."""

import time
from pathlib import Path
from typing import Callable, List, Tuple, Union

import numpy as np
import zarr

from eubi_bridge.utils.logging_config import get_logger

logger = get_logger(__name__)


def asstr(s: Union[str, int]) -> str:
    """Convert input to string.
    
    Args:
        s: String or integer value
    
    Returns:
        String representation
    
    Raises:
        TypeError: If input is not string or int
    """
    if isinstance(s, str):
        return s
    elif isinstance(s, int):
        return str(s)
    else:
        raise TypeError(f"Input must be either of types {str, int}")


def transpose_dict(dictionary: dict) -> Tuple[List, List]:
    """Separate dictionary into lists of keys and values.
    
    Args:
        dictionary: Dictionary to transpose
    
    Returns:
        Tuple of (keys list, values list)
    
    Examples:
        >>> keys, values = transpose_dict({'a': 1, 'b': 2})
        >>> keys
        ['a', 'b']
        >>> values
        [1, 2]
    """
    keys, values = [], []
    for key, value in dictionary.items():
        keys.append(key)
        values.append(value)
    return keys, values


def argsorter(s: List) -> List[int]:
    """Get indices that would sort a list.
    
    Args:
        s: List to get sort indices for
    
    Returns:
        List of indices that sort the input
    
    Examples:
        >>> argsorter([3, 1, 2])
        [1, 2, 0]
    """
    return sorted(range(len(s)), key=lambda k: s[k])


def insert_at_indices(iterable1, iterable2, indices) -> List:
    """Insert items from iterable2 at specified indices in iterable1.
    
    Args:
        iterable1: Base iterable
        iterable2: Items to insert
        indices: Indices where to insert items
    
    Returns:
        List with items inserted at specified positions
    """
    if not hasattr(iterable1, '__len__'):
        iterable1 = [iterable1]
    if not hasattr(iterable2, '__len__'):
        iterable2 = [iterable2]
    if not hasattr(indices, '__len__'):
        indices = [indices]
    
    endlen = len(iterable1) + len(iterable2)
    end_indices = [None] * endlen
    other_indices = [i for i in range(endlen) if i not in indices]
    
    for i, j in zip(other_indices, iterable1):
        end_indices[i] = j
    for i, j in zip(indices, iterable2):
        end_indices[i] = j
    
    return end_indices


def index_nth_dimension(
    array,
    dimensions: Union[int, List[int]] = 2,
    intervals: Union[None, int, List] = None,
):
    """Index specific dimensions of an array at given intervals.
    
    Args:
        array: Dask, Zarr, or NumPy array
        dimensions: Dimension(s) to index (0-based)
        intervals: Interval or tuple (start, end) for each dimension
    
    Returns:
        Indexed array
    """
    from dask import array as da
    
    if isinstance(array, zarr.Array):
        array = da.from_zarr(array)
    
    allinds = np.arange(array.ndim).astype(int)
    
    if np.isscalar(dimensions):
        dimensions = [dimensions]
    if intervals is None or np.isscalar(intervals):
        intervals = np.repeat(intervals, len(dimensions))
    
    assert len(intervals) == len(dimensions), \
        "Length of intervals must match length of dimensions"
    
    interval_dict = {item: interval for item, interval in zip(dimensions, intervals)}
    shape = array.shape
    slcs = []
    
    for idx, dimlen in zip(allinds, shape):
        if idx not in dimensions:
            slc = slice(dimlen)
        else:
            try:
                slc = slice(interval_dict[idx][0], interval_dict[idx][1])
            except TypeError:
                slc = interval_dict[idx]
        slcs.append(slc)
    
    slcs = tuple(slcs)
    indexed = array[slcs]
    return indexed


def is_generic_collection(group) -> bool:
    """Check if Zarr group is a generic collection.
    
    Args:
        group: Zarr group
    
    Returns:
        True if group is a generic collection
    """
    import os
    
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


def get_collection_paths(directory, return_all: bool = False):
    """Find all collection paths within a Zarr directory structure.
    
    Args:
        directory: Root directory to search
        return_all: If True, also return all individual paths
    
    Returns:
        List of collection paths, or tuple of (collections, multiscales, arrays)
        if return_all=True
    """
    import os
    
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


def as_store(store: Union[zarr.Array, Path, str]):
    """Convert various store formats to Zarr LocalStore.
    
    Args:
        store: Zarr array, path, or string
    
    Returns:
        Zarr LocalStore instance
    
    Raises:
        AssertionError: If store format is not supported
    """
    assert isinstance(store, (zarr.Array, Path, str)), \
        "The given store cannot be parsed."
    
    if isinstance(store, (Path, str)):
        out = zarr.storage.LocalStore(store, dimension_separator='/')
    else:
        out = store
    return out


def retry_decorator(retries: int = 3, delay: float = 1, exceptions: Tuple = (Exception,)):
    """Decorator to retry function with exponential backoff.
    
    Args:
        retries: Number of retries
        delay: Initial delay in seconds
        exceptions: Tuple of exceptions to catch
    
    Returns:
        Decorator function
    
    Examples:
        >>> @retry_decorator(retries=3, delay=1)
        ... def flaky_function():
        ...     pass
    """
    def decorator(func: Callable) -> Callable:
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


class ChannelMap:
    """Mapping of color names to hex color codes."""
    
    DEFAULT_COLORS = {
        'red': "FF0000",
        'green': "00FF00",
        'blue': "0000FF",
        'magenta': "FF00FF",
        'cyan': "00FFFF",
        'yellow': "FFFF00",
        'white': "FFFFFF",
    }
    
    def __getitem__(self, key: str) -> str:
        """Get hex color code for color name.
        
        Args:
            key: Color name (e.g., 'red', 'green')
        
        Returns:
            Hex color code (without #), or None if not found
        """
        if key in self.DEFAULT_COLORS:
            return self.DEFAULT_COLORS[key]
        else:
            return None
