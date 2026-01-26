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


def is_ome_zarr(path: Union[str, Path]) -> bool:
    """Check if a path is an OME-Zarr directory using zarr-native detection.
    
    Supports both zarr v2 (NGFF v0.4) and zarr v3 (NGFF v0.5):
    - v0.5 (zarr v3): Has 'ome' attribute in root group
    - v0.4 (zarr v2): Has 'multiscales' attribute in root group
    
    Works with local paths and remote URLs.
    
    Args:
        path: Path to directory or remote URL
    
    Returns:
        True if path is a valid OME-Zarr, False otherwise
    """
    import zarr
    
    try:
        gr = zarr.open_group(path, mode='r')
        # Check for OME-Zarr metadata attributes
        # v0.5 stores metadata under 'ome' attribute
        # v0.4 stores metadata under 'multiscales' attribute
        return 'ome' in gr.attrs or 'multiscales' in gr.attrs
    except Exception:
        return False


def get_ome_zarr_version(path: Union[str, Path]) -> Optional[str]:
    """Get the NGFF version string from an OME-Zarr using zarr-native detection.
    
    Returns the NGFF specification version (e.g., "0.5" or "0.4").
    Uses the zarr group's native format detection which works with remote URLs.
    
    - Zarr format 3 with 'ome' attribute = OME-Zarr v0.5
    - Zarr format 2 with 'multiscales' attribute = OME-Zarr v0.4
    
    Args:
        path: Path to OME-Zarr directory or remote URL
    
    Returns:
        NGFF version string ("0.5" or "0.4"), or None if not an OME-Zarr
    """
    import zarr
    
    try:
        gr = zarr.open_group(path, mode='r')
        zarr_format = gr.info._zarr_format
        
        # Check for OME-Zarr metadata based on zarr format
        if zarr_format == 3:
            # Zarr v3: metadata should be in 'ome' attribute
            if 'ome' in gr.attrs:
                return "0.5"
        elif zarr_format == 2:
            # Zarr v2: metadata can be in 'multiscales' attribute
            if 'multiscales' in gr.attrs:
                return "0.4"
        
        # Fallback: default to 0.4 for v2, 0.5 for v3 if metadata found
        if 'ome' in gr.attrs or 'multiscales' in gr.attrs:
            return "0.5" if zarr_format == 3 else "0.4"
        
        return None
    except Exception:
        return None


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
        
        # Get CSV directory for resolving relative paths
        csv_dir = os.path.dirname(os.path.abspath(input_path))
        
        if input_path.endswith((".csv", ".tsv", ".txt")):
            df = pd.read_csv(input_path)
        elif input_path.endswith((".xls", ".xlsx")):
            df = pd.read_excel(input_path)
        else:
            raise ValueError("Unsupported file format. Use .csv or .xlsx")
        
        # Convert NaN values to None so they don't interfere with parameter handling
        # Empty CSV cells and various NA placeholders become NaN from pandas, but configuration 
        # defaults are designed to handle None values, not NaN
        # Pandas automatically recognizes these as NA/null:
        #   - Empty string (default)
        #   - 'N/A', 'n/a', 'NA', 'nan', 'NaN', '-NaN', '-nan'
        #   - 'NULL', 'null'
        #   - '#N/A', '#NA' (Excel errors)
        #   - '<NA>', '<na>' (Pandas markers)
        #   - Excel infinity variants ('-1.#IND', '1.#IND', etc.)
        # First convert to object dtype, then replace NaN with None
        df = df.astype(object).where(pd.notna(df), None)
        
        # Resolve relative paths in input_path column to be relative to CSV directory
        # This ensures CSV files are portable - paths are always relative to the CSV location
        if 'input_path' in df.columns:
            df['input_path'] = df['input_path'].apply(
                lambda path: os.path.join(csv_dir, path) if path and not os.path.isabs(path) else path
            )
        
        # Also resolve relative output_path if present
        if 'output_path' in df.columns:
            df['output_path'] = df['output_path'].apply(
                lambda path: os.path.join(csv_dir, path) if path and not os.path.isabs(path) else path
            )
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
