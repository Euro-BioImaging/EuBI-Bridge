"""Path and file system utilities."""

import glob
import fnmatch
import os
from collections import Counter
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


def pattern_matches(path: str, pattern: str) -> bool:
    """Match *path* against one include/exclude *pattern*.

    A pattern containing glob metacharacters (``*``, ``?``, ``[``) is matched
    with :mod:`fnmatch`, against both the full path and the basename so that
    ``*.tif`` behaves as users expect on a nested directory.  A pattern without
    them keeps the original substring behaviour, so ``.h5`` or ``Patient1``
    still work.
    """
    if not pattern:
        return True
    if any(ch in pattern for ch in '*?['):
        return (fnmatch.fnmatch(path, pattern)
                or fnmatch.fnmatch(os.path.basename(path), pattern))
    return pattern in path


def _matches_any(path: str, patterns) -> bool:
    """True when *patterns* is empty/None, or any of them matches *path*."""
    if patterns is None:
        return True
    if isinstance(patterns, str):
        patterns = [patterns]
    patterns = [p for p in patterns if p]
    if not patterns:
        return True
    return any(pattern_matches(path, p) for p in patterns)


#: Column naming the aggregative group a conversion-table row belongs to.
#: Rows sharing a value are concatenated into one output; a blank cell is a
#: plain one-to-one conversion.
AGGREGATIVE_GROUP_COLUMN = "aggregative_group"


def is_blank_cell(value) -> bool:
    """True for a cell that means "inherit", i.e. one the user left empty.

    An empty CSV cell arrives as NaN, which is not equal to itself; the GUI
    leaves None or an empty string instead.
    """
    return value is None or value != value or str(value).strip() == ""


def sanitise_group_name(group: object) -> str:
    """Turn an aggregative group id into something usable in a filename.

    The raw value identifies the group and is compared as given, so grouping
    stays exact; only the copy that reaches a path is normalised.  Without this
    a perfectly reasonable id such as ``"Embryo 1/A"`` would silently create a
    subdirectory.
    """
    # str(None) is 'None' and str(nan) is 'nan', both usable in a filename, so
    # an unset cell has to be rejected before the value is stringified.
    if is_blank_cell(group):
        return ""
    text = str(group).strip()
    safe = "".join(ch if (ch.isalnum() or ch in "_.-") else "_" for ch in text)
    return safe.strip("_") or ""


def prefix_with_group(name: str, group: object) -> str:
    """Prefix an output *name* with its aggregative group.

    Aggregative names are derived from what was concatenated
    (``img_t0_zset``), which is what keeps sibling outputs apart when one
    directory yields several groups.  The group is therefore added in front
    rather than replacing anything::

        gr1  +  img_t0_zset  ->  gr1_img_t0_zset

    An empty or missing group leaves the name untouched, so conversions that do
    not use the column keep exactly the names they had.
    """
    safe = sanitise_group_name(group) if group is not None else ""
    if not safe:
        return name
    return f"{safe}_{name}" if name else safe


#: Concatenation settings that describe *how a group is assembled*, so they may
#: differ between groups but must agree within one.  ``includes``/``excludes``
#: are deliberately absent: they filter the input search, which a table has
#: already done by naming its paths explicitly, so they stay global.
PER_GROUP_CONCAT_KEYS = (
    "concatenation_axes", "time_tag", "channel_tag", "z_tag", "y_tag", "x_tag",
)


def resolve_group_concat_params(group_id, group_df, defaults: dict) -> dict:
    """Return one group's concatenation settings, or explain why it cannot.

    Every row of a group describes the *same* output, so a setting that differs
    between its rows has no single meaning.  Taking the first row silently would
    discard the others, so a genuine disagreement is an error naming the group,
    the parameter and the values, rather than a value quietly winning.

    A blank cell inherits *defaults*, which keeps a table that sets nothing
    behaving exactly as it did before the columns existed.
    """
    resolved = dict(defaults)
    for key in PER_GROUP_CONCAT_KEYS:
        if key not in group_df.columns:
            continue
        present = [v for v in group_df[key].tolist() if not is_blank_cell(v)]
        if not present:
            continue
        distinct = list(dict.fromkeys(str(v) for v in present))
        if len(distinct) > 1:
            raise ValueError(
                f"Aggregative group {group_id!r} has conflicting {key}: "
                f"{', '.join(repr(v) for v in distinct)}. Every row of a group "
                f"builds one output, so its concatenation settings must agree; "
                f"use separate groups if they should differ."
            )
        resolved[key] = present[0]
    return resolved


def concat_without_group_problems(df: pd.DataFrame,
                                  defaults: dict | None = None) -> list:
    """Report rows that ask to concatenate but say nothing to concatenate with.

    In a conversion table it is ``aggregative_group`` that marks a row as
    aggregative: rows sharing a value become one output, a blank one converts
    alone.  So a row carrying ``concatenation_axes`` but no group silently
    converts one-to-one, producing something other than what its settings
    describe -- a wrong result rather than a preference, which is why this
    blocks rather than warns.

    Only the axes count as the declaration.  Stray tags left in a form or a
    config are harmless on a unary row, and refusing them would make a mixed
    table needlessly awkward to write.

    *defaults* are the run-wide settings a blank cell inherits, so axes given
    once on the command line are caught as readily as axes written per row.
    """
    defaults = defaults or {}
    default_axes = not is_blank_cell(defaults.get("concatenation_axes"))

    if AGGREGATIVE_GROUP_COLUMN not in df.columns:
        groups = [None] * len(df)
    else:
        groups = df[AGGREGATIVE_GROUP_COLUMN].tolist()

    if "concatenation_axes" in df.columns:
        axes = df["concatenation_axes"].tolist()
    else:
        axes = [None] * len(df)

    def _summarise(positions: list) -> str:
        shown = ", ".join(str(p) for p in positions[:5])
        if len(positions) > 5:
            shown += f", ... (+{len(positions) - 5} more)"
        return shown

    no_group, no_axes = [], []
    for position, (group, axis) in enumerate(zip(groups, axes), start=1):
        has_axes = not is_blank_cell(axis) or default_axes
        has_group = not is_blank_cell(group)
        if has_axes and not has_group:
            no_group.append(position)
        elif has_group and not has_axes:
            no_axes.append(position)

    problems = []
    if no_group:
        problems.append(
            f"Row {_summarise(no_group)}: concatenation axes are set but no "
            f"aggregative group is. Rows are concatenated by sharing a group "
            f"name, so these would convert one-to-one and ignore the "
            f"concatenation settings. Give them a group name to concatenate "
            f"them, or clear the axes to convert them singly."
        )
    if no_axes:
        # The mirror image, and just as silent: the rows are collected into a
        # group and then concatenated along nothing, which currently fails deep
        # in the dispatcher as "commonpath() arg is an empty sequence".
        problems.append(
            f"Row {_summarise(no_axes)}: an aggregative group is set but no "
            f"concatenation axes are. A group has to be told which axis to "
            f"concatenate along, so set the axes (e.g. 'z'), or clear the group "
            f"name to convert these rows one-to-one."
        )
    return problems


def partition_by_group(df: pd.DataFrame):
    """Split a conversion table into unary rows and aggregative groups.

    Returns ``(unary_df, groups)`` where *groups* is a list of
    ``(group_id, group_df)`` pairs, one per aggregative output, in the order the
    groups first appear in the table.

    A blank ``aggregative_group`` cell means the row converts on its own, which
    is the behaviour of every table written before the column existed.  Rows
    sharing a value are concatenated together, so the column decides membership
    explicitly rather than the tags having to be re-derived at run time.
    """
    if AGGREGATIVE_GROUP_COLUMN not in df.columns:
        return df, []

    blank = df[AGGREGATIVE_GROUP_COLUMN].map(is_blank_cell)
    unary = df[blank]

    groups = []
    grouped = df[~blank]
    for group_id in grouped[AGGREGATIVE_GROUP_COLUMN].drop_duplicates():
        groups.append((group_id,
                       grouped[grouped[AGGREGATIVE_GROUP_COLUMN] == group_id]))
    return unary, groups


def disambiguate_output_names(input_paths: List[str]) -> dict:
    """Map each input path to a unique output basename.

    Outputs are named after the input's basename, so two inputs from different
    folders that share a name would target one output — the first written, the
    second either refused (``overwrite=False``) or silently destroying the first
    (``overwrite=True``).  Neither is acceptable in a batch the user cannot split
    by hand, and the Run tab offers only one output folder.

    Colliding names take one parent directory at a time as a prefix, repeating
    until every name is unique::

        A/img.tif  ->  A_img
        B/img.tif  ->  B_img

    Only names that actually collide are changed, so a batch of distinct names
    keeps exactly the output paths it had before.  Paths that remain identical
    after exhausting their parents (the same file listed twice) fall back to a
    numeric suffix, which always terminates.
    """
    def _segments(path: str) -> List[str]:
        """Split *path* into components, treating both separators as separators.

        ``os.path`` only recognises ``\\`` on Windows, so a Windows-style path
        processed on Linux comes back as a single component and the whole string
        is used as the name.  Batch tables travel between machines, so the
        separator is handled explicitly rather than per-platform.
        """
        unified = path.replace('\\', '/').rstrip('/')
        return [part for part in unified.split('/')
                if part not in ('', '.', '..')]

    def stem(path: str) -> str:
        # Matches _generate_output_path: everything before the first dot, so
        # 'image.ome.tiff' -> 'image'.
        segments = _segments(path)
        return segments[-1].split('.')[0] if segments else ''

    def parents(path: str) -> List[str]:
        return _segments(path)[:-1]

    names = {path: stem(path) for path in input_paths}
    depth = 0
    while True:
        clashing = {name for name, count in Counter(names.values()).items()
                    if count > 1}
        if not clashing:
            break
        depth += 1
        progressed = False
        for path, name in list(names.items()):
            if name not in clashing:
                continue
            available = parents(path)
            if depth <= len(available):
                names[path] = f"{available[-depth]}_{stem(path)}"
                progressed = True
        if not progressed:
            # Parents exhausted — distinct inputs cannot be told apart by path
            # alone (or the same path was listed twice).  Numbering terminates.
            for index, (path, name) in enumerate(
                    (p, n) for p, n in names.items() if n in clashing):
                names[path] = f"{name}_{index + 1}" if index else name
            break
    return names


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
    paths = [
        p for p in paths
        if _matches_any(p, includes)
        and not (excludes is not None and _matches_any(p, excludes))
    ]

    # Remove zarr.json files
    paths = list(filter(lambda path: not path.endswith('zarr.json'), paths))
    
    if len(paths) == 0:
        raise ValueError(f"No valid paths found for {original_input_path}")
    
    return sorted(paths)


def _apply_table_filters_and_defaults(df: pd.DataFrame,
                                      global_kwargs: dict) -> pd.DataFrame:
    """Apply include/exclude filters and global column defaults to *df*.

    Shared by every branch of :func:`take_filepaths` so a pattern behaves
    identically whether the rows came from a directory, an explicit file list, a
    conversion table on disk, or a table handed over in memory.
    """
    def _keep(row) -> bool:
        inp = row["input_path"]
        includes = global_kwargs.get('includes')
        excludes = global_kwargs.get('excludes')
        if not _matches_any(inp, includes):
            return False
        if excludes is not None and _matches_any(inp, excludes):
            return False
        return True

    df = df[df.apply(_keep, axis=1)]

    # Global values fill in only the columns the table does not carry itself, so
    # a per-row override always wins over the run-wide setting.
    for k, v in global_kwargs.items():
        if k not in df.columns:
            if hasattr(v, '__len__') and not isinstance(v, str):
                df[k] = [v for _ in range(len(df))]
            else:
                df[k] = v

    return df


def take_filepaths(
    input_path: Union[str, os.PathLike, list, tuple, pd.DataFrame],
    **global_kwargs,
) -> pd.DataFrame:
    """Load file paths into a DataFrame, from directory, files, CSV/Excel table,
    an explicit list of paths (e.g. from GUI multi-select), or a ready-made
    table.

    Handles multiple input types:
    - DataFrame: an already-built conversion table, used as-is
    - List/tuple of paths: Used directly, no globbing or filtering applied
    - Directory path: Finds all files
    - File glob pattern: Matches files
    - CSV/XLSX table: Reads table with 'input_path' column

    Args:
        input_path: Directory, file, glob pattern, table path, list/tuple of
                    explicit file paths, or a DataFrame of rows
        **global_kwargs: Include/exclude filters, column defaults

    Returns:
        DataFrame with 'input_path' column and any additional columns from kwargs

    Raises:
        ValueError: If input is invalid or no paths found
        Exception: If conflicting parameters provided
    """
    # ── An already-built table ────────────────────────────────────────────────
    # A caller that has per-row overrides in hand (the GUI batch queue) can pass
    # them straight through, instead of writing a CSV purely so this function
    # can read it back.  Filtering and column defaults below still apply, so the
    # rows behave exactly as if they had come from a file.
    if isinstance(input_path, pd.DataFrame):
        df = input_path.copy()
        if "filepath" in df.columns and "input_path" not in df.columns:
            df = df.rename(columns={"filepath": "input_path"})
        if "input_path" not in df.columns:
            raise ValueError(
                "A DataFrame input must have an 'input_path' or 'filepath' "
                "column.")
        if df.empty:
            raise ValueError("Empty conversion table provided.")
        # Match the CSV branch: NaN cells become None so that configuration
        # defaults (which expect None, not NaN) take effect.
        df = df.astype(object).where(pd.notna(df), None)
        return _apply_table_filters_and_defaults(df, global_kwargs)

    # ── Explicit list of paths (GUI multi-select or programmatic use) ──────────
    if isinstance(input_path, (list, tuple)):
        fps = [str(p) for p in input_path if p]
        if not fps:
            raise ValueError("Empty list of input paths provided.")
        df = pd.DataFrame({'input_path': fps})
        output_path = global_kwargs.get('output_path', None)
        if output_path is not None:
            df['output_path'] = str(output_path)
        # Propagate any extra kwargs as DataFrame columns (mirrors table-input behaviour)
        for k, v in global_kwargs.items():
            if k not in ('output_path', 'includes', 'excludes') and k not in df.columns:
                if hasattr(v, '__len__') and not isinstance(v, str):
                    df[k] = [v for _ in range(len(df))]
                else:
                    df[k] = v
        return df

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

    return _apply_table_filters_and_defaults(df, global_kwargs)


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
