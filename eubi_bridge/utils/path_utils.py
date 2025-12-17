"""Path and file system utilities."""

import glob
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from eubi_bridge.utils.logging_config import get_logger

logger = get_logger(__name__)

TABLE_FORMATS = (".csv", ".tsv", ".txt", ".xls", ".xlsx")


def parse_as_list(path_or_paths: Union[list, str, int, float]) -> list:
    """Convert input to list format.
    
    Args:
        path_or_paths: Single item or iterable
    
    Returns:
        List containing the input(s)
    """
    if isinstance(path_or_paths, (str, int, float)):
        return [path_or_paths]
    else:
        return list(path_or_paths)


def includes(
    group1: Union[list, str, int, float],
    group2: Union[list, str, int, float],
) -> bool:
    """Check if group1 includes all items from group2.
    
    Args:
        group1: Container or single item
        group2: Items to check
    
    Returns:
        True if all items in group2 are in group1
    """
    gr1 = parse_as_list(group1)
    gr2 = parse_as_list(group2)
    return all([item in gr1 for item in gr2])


def path_has_pyramid(path: Union[str, Path]) -> bool:
    """Check if path contains a valid Zarr group with pyramid structure.
    
    Args:
        path: Path to check
    
    Returns:
        True if path is a valid Zarr group
    """
    try:
        import zarr
        store = zarr.storage.LocalStore(path)
        _ = zarr.open_group(store, mode='r')
        return True
    except Exception:
        return False


def is_zarr_array(path: Union[str, Path]) -> bool:
    """Check if path is a valid Zarr array.
    
    Args:
        path: Path to check
    
    Returns:
        True if path is a valid Zarr array
    """
    try:
        import zarr
        _ = zarr.open_array(path, mode='r')
        return True
    except Exception:
        return False


def is_zarr_group(path: Union[str, Path]) -> bool:
    """Check if path is a valid Zarr group.
    
    Args:
        path: Path to check
    
    Returns:
        True if path is a valid Zarr group
    """
    try:
        import zarr
        _ = zarr.open_group(path, mode='r')
        return True
    except Exception:
        return False


def sensitive_glob(
    pattern: str,
    recursive: bool = False,
    sensitive_to: str = '.zarr',
) -> List[str]:
    """Perform glob matching with special handling for directory-like formats.
    
    Args:
        pattern: Glob pattern to match
        recursive: If True, use ** for recursive search
        sensitive_to: Directory format to treat specially (e.g., '.zarr')
    
    Returns:
        List of matching paths
    """
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


def take_filepaths_from_path(
    input_path: str,
    includes: Union[str, tuple, list] = None,
    excludes: Union[str, tuple, list] = None,
    **kwargs,
) -> List[str]:
    """Get list of file paths from directory or file pattern.
    
    Args:
        input_path: Path to file, directory, or glob pattern
        includes: Patterns to include (comma-separated string or list)
        excludes: Patterns to exclude (comma-separated string or list)
        **kwargs: Additional arguments (unused)
    
    Returns:
        Sorted list of matching file paths
    
    Raises:
        ValueError: If no matching paths found
    """
    original_input_path = input_path
    
    if isinstance(includes, str):
        includes = includes.split(',')
    if isinstance(excludes, str):
        excludes = excludes.split(',')

    # Handle file or single zarr path
    if os.path.isfile(input_path) or input_path.endswith('.zarr'):
        dirname = os.path.dirname(input_path)
        basename = os.path.basename(input_path)
        if len(dirname) == 0:
            dirname = '.'
        input_path = f"{dirname}/*{basename}"

    # Ensure glob pattern
    if '*' not in input_path and not input_path.endswith('.zarr'):
        input_path = os.path.join(input_path, '**')

    if '*' not in input_path:
        input_path_ = os.path.join(input_path, '**')
    else:
        input_path_ = input_path
    
    paths = sensitive_glob(input_path_, recursive=False, sensitive_to='.zarr')

    # Filter by includes/excludes
    paths = list(filter(
        lambda path: (
            (
                any(inc in path for inc in includes)
                if isinstance(includes, (tuple, list))
                else (includes in path if includes is not None else True)
            )
            and
            (
                not any(exc in path for exc in excludes)
                if isinstance(excludes, (tuple, list))
                else (excludes not in path if excludes is not None else True)
            )
        ),
        paths
    ))

    # Remove zarr.json files
    paths = list(filter(lambda path: not path.endswith('zarr.json'), paths))
    
    if len(paths) == 0:
        raise ValueError(f"No valid paths found for {original_input_path}")
    
    return sorted(paths)


def take_filepaths(
    input_path: Union[str, os.PathLike],
    **global_kwargs,
) -> pd.DataFrame:
    """Load file paths into a DataFrame, from directory, files, or CSV/Excel table.
    
    Handles multiple input types:
    - Directory path: Finds all files
    - File glob pattern: Matches files
    - CSV/XLSX table: Reads table with 'input_path' column
    
    Args:
        input_path: Directory, file, glob pattern, or table path
        **global_kwargs: Include/exclude filters, column defaults
    
    Returns:
        DataFrame with 'input_path' column and any additional columns from kwargs
    
    Raises:
        ValueError: If input is invalid or no paths found
        Exception: If conflicting parameters provided
    """
    # Normalize include/exclude parameters
    if 'includes' in global_kwargs:
        if global_kwargs['includes'] is None:
            pass
        elif isinstance(global_kwargs['includes'], (tuple, list)):
            global_kwargs['includes'] = tuple([str(member) for member in global_kwargs['includes']])
        elif isinstance(global_kwargs['includes'], str):
            global_kwargs['includes'] = global_kwargs['includes'].split(',')
        elif np.isscalar(global_kwargs['includes']):
            global_kwargs['includes'] = str(global_kwargs['includes'])
        else:
            raise TypeError(f"Unknown type: {type(global_kwargs['includes'])}")

    if 'excludes' in global_kwargs:
        if global_kwargs['excludes'] is None:
            pass
        elif isinstance(global_kwargs['excludes'], (tuple, list)):
            global_kwargs['excludes'] = tuple([str(member) for member in global_kwargs['excludes']])
        elif isinstance(global_kwargs['excludes'], str):
            global_kwargs['excludes'] = global_kwargs['excludes'].split(',')
        elif np.isscalar(global_kwargs['excludes']):
            global_kwargs['excludes'] = str(global_kwargs['excludes'])
        else:
            raise TypeError(f"Unknown type: {type(global_kwargs['excludes'])}")

    # Handle different input types
    if input_path.endswith(TABLE_FORMATS):
        concatenation_axes = global_kwargs.get('concatenation_axes', None)
        if concatenation_axes is not None:
            logger.error(
                "Specifying tables as input is only supported for one-to-one conversions. "
                "With aggregative conversions, specify a directory instead."
            )
            raise Exception(
                "Specifying tables as input is only supported for one-to-one conversions. "
                "With aggregative conversions, specify a directory instead."
            )

        logger.info(f"Loading conversion table from {input_path}")
        if input_path.endswith((".csv", ".tsv", ".txt")):
            df = pd.read_csv(input_path)
        elif input_path.endswith((".xls", ".xlsx")):
            df = pd.read_excel(input_path)
        else:
            raise ValueError("Unsupported file format. Use .csv or .xlsx")
    elif os.path.isdir(input_path) or os.path.isfile(input_path) or '*' in input_path:
        filepaths = take_filepaths_from_path(input_path, **global_kwargs)
        df = pd.DataFrame(filepaths, columns=["input_path"])
    else:
        raise Exception(f"Invalid input path: {input_path}")

    # Normalize input column name
    if "filepath" in df.columns and "input_path" not in df.columns:
        df.rename(columns={"filepath": "input_path"}, inplace=True)

    if "input_path" not in df.columns:
        raise ValueError("Table must include an 'input_path' or 'filepath' column.")
    
    # Filter by includes/excludes
    def should_drop(row):
        inp = row["input_path"]
        includes = global_kwargs.get('includes', [None])
        excludes = global_kwargs.get('excludes', [None])
        if not isinstance(includes, (tuple, list)):
            includes = [includes]
        if not isinstance(excludes, (tuple, list)):
            excludes = [excludes]
        mask1 = any([inc in inp if inc is not None else True for inc in includes])
        mask2 = any([exc not in inp if exc is not None else True for exc in excludes])
        return mask1 and mask2

    mask = df.apply(should_drop, axis=1)
    df = df[mask]
    
    # Apply global defaults for missing parameters
    for k, v in global_kwargs.items():
        if k not in df.columns:
            if hasattr(v, '__len__') and not isinstance(v, str):
                df[k] = [v for _ in range(len(df))]
            else:
                df[k] = v

    return df


def find_common_root(paths: List[Union[str, os.PathLike]]) -> str:
    """Find the common root directory from a list of paths.
    
    Args:
        paths: List of file or directory paths
    
    Returns:
        Common root directory path, or empty string if no common root
    
    Examples:
        >>> find_common_root(['/a/b/c', '/a/b/d', '/a/b/c/e'])
        '/a/b'
    """
    if not paths:
        return ""

    try:
        path_objs = [Path(p).resolve() for p in paths]
    except (TypeError, OSError):
        return ""

    # Get the common prefix of all paths
    common = os.path.commonpath([str(p) for p in path_objs])

    # Verify that common prefix is actually a parent directory
    common_path = Path(common)
    if not all(common_path in p.parents or p == common_path for p in path_objs):
        return ""

    return common


def find_common_root_relative(paths: List[Union[str, os.PathLike]]) -> str:
    """Find common root directory preserving relative path structure.
    
    Works with relative paths without converting to absolute.
    
    Args:
        paths: List of relative or absolute paths
    
    Returns:
        Common root directory path, or empty string if no common root
    """
    if not paths:
        return ""

    # Split all paths into their components
    split_paths = [Path(p).parts for p in paths]

    # Find the common prefix
    common_parts = []
    for parts in zip(*split_paths):
        if len(set(parts)) == 1:
            common_parts.append(parts[0])
        else:
            break

    if not common_parts:
        return ""

    return str(Path(*common_parts))
