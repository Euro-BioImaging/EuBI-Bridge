"""Path and file system utilities."""

import glob
import fnmatch
import os
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from eubi_bridge.utils.errors import InputError
from eubi_bridge.utils.logging_config import get_logger

logger = get_logger(__name__)

TABLE_FORMATS = (".csv", ".tsv", ".txt", ".xls", ".xlsx")
#: a .txt that is an image set's own file, never a conversion table: Micro-Manager's
#: metadata.txt (its JSON starts with "{")
IMAGE_TEXT_NAMES = {"metadata.txt"}


def is_table_path(path) -> bool:
    """Is *path* a conversion table (``TABLE_FORMATS``)? A Micro-Manager
    ``metadata.txt`` is not, nor any .txt whose first byte is ``{`` (JSON)."""
    path = str(path)
    if not path.lower().endswith(TABLE_FORMATS):
        return False
    import os
    if os.path.basename(path).lower() in IMAGE_TEXT_NAMES:
        return False
    if path.lower().endswith(".txt") and os.path.isfile(path):
        try:
            with open(path, "rb") as fh:
                return not fh.read(1).startswith(b"{")
        except OSError:
            return True
    return True


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


def ome_zarr_kind(path: Union[str, Path]) -> Optional[str]:
    """What an OME-Zarr group holds, from its root attributes alone (local or remote):
    ``"plate"`` (NGFF 0.4: ``plate`` in the attributes; 0.5 / 0.6: under ``ome``),
    ``"collection"`` (an RFC 8 collection: a plate or a set of images, which only its
    node tree tells), ``"series"`` (a bioformats2raw layout), ``"image"``
    (``multiscales``), or None (not a group, or none of these)."""
    try:
        import zarr
        attrs = zarr.open_group(path, mode='r').attrs.asdict()
    except Exception:
        return None
    ngff = attrs.get("ome") if isinstance(attrs.get("ome"), dict) else attrs
    if isinstance(ngff.get("plate"), dict):
        return "plate"
    if ngff.get("type") == "collection":
        return "collection"
    if "multiscales" in ngff:
        return "image"
    if "bioformats2raw.layout" in ngff:
        return "series"
    return None


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


def _is_image_group(path: Union[str, Path]) -> bool:
    """Whether `path` is a zarr group holding an OME-Zarr image (``multiscales``, under
    ``ome`` in 0.5 / 0.6)."""
    import zarr

    try:
        attrs = zarr.open_group(str(path), mode='r').attrs
        ngff = attrs.get('ome') if isinstance(attrs.get('ome'), dict) else attrs
        return 'multiscales' in ngff
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

    def _in_a_store(path: str) -> bool:
        """Inside a *sensitive_to* store (its chunks): the store is the input, not them."""
        parts = path.replace('\\', '/').rstrip('/').split('/')
        return any(part.endswith(sensitive_to) for part in parts[:-1])

    for start_path in glob.glob(pattern, recursive=recursive):
        if recursive and _in_a_store(start_path):
            continue
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

    # a recursive pattern names every folder on the way, and each is walked: one path once
    return list(dict.fromkeys(results))


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


#: plate index files with a fixed name (Harmony Index.idx.xml / index.xml, CellVoyager
#: MeasurementData.mlf, scanR experiment_descriptor.xml, Columbus ImageIndex /
#: MeasurementIndex.ColumbusIDX.xml): their output is named after the measurement folder
GENERIC_INDEX_NAMES = {"index", "measurementdata", "experiment_descriptor", "imageindex",
                       "measurementindex"}
GENERIC_INDEX_SUFFIXES = (".xml", ".mlf")
#: data files of a plate that has no index file, any one of which opens the plate:
#: Cellomics fields (<plate>_A01f00d0.C01), a NIS-Elements well-plate run's files
#: (WellB05_ChannelDIA_Seq0000.nd2, micro-reader's _nd2_run.RUN_FILE). The plate's
#: output is named after their folder (written per image, each keeps its own name)
PLATE_DATA_FILES = re.compile(
    r"(?:.*\.c01|well[a-z]{1,2}\d{1,3}(?:_point[^_]+_\d+)?_channel.+_seq\d+\.nd2)$",
    re.IGNORECASE)
#: the folder a Harmony measurement keeps its index in (images/ in an hs archive)
INDEX_IMAGES_FOLDERS = {"images"}
#: plane files: never parts of a composite, and too many to ask about one by one
PLANE_SUFFIXES = (".tif", ".tiff")
#: micro-reader's plate formats, whose folder holds one plate: a file in it belongs to
#: the plate (input_mains)
PLATE_FORMATS = {"harmony", "cv7000", "metaxpress", "incell",
                 # micro-reader 0.3.0
                 "flex", "columbus", "scanr", "cellomics", "jdce"}
#: an NDTiff dataset with downsampled levels (Micro-Manager Magellan explorations,
#: stitched acquisitions): the image in "Full resolution/", each level a dataset of its
#: own in "Downsampled_x<N>/" beside it
NDTIFF_FULL_RESOLUTION = "Full resolution"
NDTIFF_LEVEL = re.compile(r"downsampled_x\d+", re.IGNORECASE)


def ndtiff_full_resolution(path: str) -> Optional[str]:
    """For a file of an NDTiff dataset's downsampled level (``<dataset>/Downsampled_x4/``):
    the main file of the dataset's full resolution, whose preview the level is -- not an
    image of its own. None for any other file, and when the full resolution is missing
    (the level is then converted as it is)."""
    folder = os.path.dirname(os.path.abspath(path))
    if not NDTIFF_LEVEL.fullmatch(os.path.basename(folder)):
        return None
    full = os.path.join(os.path.dirname(folder), NDTIFF_FULL_RESOLUTION)
    if not os.path.isfile(os.path.join(full, "NDTiff.index")):
        return None
    import micro_reader
    mains = [g.main for g in micro_reader.group_files(full, plates=False).groups
             if g.main is not None]
    return mains[0] if len(mains) == 1 else None


#: suffixes of two parts, removed whole from an output's name (``a.ome.tif`` -> ``a``)
COMPOUND_SUFFIXES = (".ome.tif", ".ome.tiff", ".ome.btf", ".ome.tf2", ".ome.tf8", ".ome.zarr",
                     ".ome.xml", ".companion.ome", ".idx.xml")


def output_stem(path) -> str:
    """The name an output takes from its input: the file name without its suffix, a
    two-part one (COMPOUND_SUFFIXES) whole. ``Image_V4.1_BF.vsi`` -> ``Image_V4.1_BF``;
    until 0.3.2 everything before the first dot (``Image_V4``), so files differing after
    a dot in their names wrote one output."""
    name = str(path).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    low = name.lower()
    for suffix in COMPOUND_SUFFIXES:
        if low.endswith(suffix) and len(name) > len(suffix):
            return name[:-len(suffix)]
    stem = os.path.splitext(name)[0]
    return stem or name


def named_folder(path: str, plate: bool = True) -> Optional[str]:
    """For an index file with a fixed name: the folder its output is named after -- a
    plate's measurement folder (above ``Images/``), a Micro-Manager image-per-file
    acquisition's folder (``<acquisition>/<position>/metadata.txt``), the folder of a
    plate's data files (PLATE_DATA_FILES) when the plate is written as one (`plate`);
    else None."""
    name = os.path.basename(path)
    if name.lower() in IMAGE_TEXT_NAMES:
        position = os.path.dirname(os.path.abspath(path))
        acquisition = os.path.dirname(position)
        return acquisition if os.path.basename(acquisition) else position
    if plate and PLATE_DATA_FILES.fullmatch(name):
        folder = os.path.dirname(os.path.abspath(path))
        return folder if os.path.basename(folder) else None
    if name.split(".")[0].lower() not in GENERIC_INDEX_NAMES or             not name.lower().endswith(GENERIC_INDEX_SUFFIXES):
        return None
    folder = os.path.dirname(os.path.abspath(path))
    if os.path.basename(folder).lower() in INDEX_IMAGES_FOLDERS:
        folder = os.path.dirname(folder)
    return folder if os.path.basename(folder) else None


#: files that are never images: given among the inputs they are skipped, or -- given on
#: their own in a folder -- stand for the plate in that folder (group_inputs)
TEXT_SUFFIXES = (".txt", ".md", ".log", ".json", ".csv", ".tsv", ".ini", ".cfg",
                 ".yaml", ".yml", ".html", ".htm", ".pdf")


#: side files of instruments and analysis programs, never an image on their own: of
#: several inputs, the full eubi-bridge skips them too when micro-reader reads nothing
#: in them (``_not_an_image``)
SIDE_FILE_SUFFIXES = TEXT_SUFFIXES + (".xml", ".aas", ".xgapp", ".env", ".mes", ".wpp",
                                      ".mrf", ".mlf", ".db", ".mdb", ".tmp", ".bak", ".xslt",
                                      ".xsl", ".pdf", ".xlsx", ".docx", ".mp4", ".avi", ".zip",
                                      ".bin", ".mat", ".npz", ".npy", ".pkl", ".pickle",
                                      ".mxprotocol")             # ImageXpress protocols


#: suffixes of image files: one that micro-reader cannot read is a damaged image, not a side
#: file, so it is kept and the conversion fails on it, with either install
IMAGE_SUFFIXES = (".tif", ".tiff", ".btf", ".lsm", ".nd2", ".lif", ".lof", ".xlef", ".oir",
                  ".oib", ".oif", ".czi", ".ims", ".dv", ".r3d", ".nd", ".ics", ".ids",
                  ".png", ".jpg", ".jpeg", ".h5", ".hdf5", ".mrc", ".ndpi", ".svs", ".vsi",
                  ".klb")


def _not_an_image(path: str) -> bool:
    """True for a file no reader of this install recognises: micro-reader's formats
    find nothing in its first bytes, and -- with Bio-Formats installed (the full
    eubi-bridge), which reads more -- its suffix is a side file's
    (``SIDE_FILE_SUFFIXES``). A file with an image suffix is never skipped (a damaged
    image fails loudly). Unsure (micro-reader's check fails): False."""
    try:
        from micro_reader import _parts, _registry
        head = _parts.read_head(path)
        if any(module.sniff(head, path) for module in _registry.BACKENDS):
            return False
    except Exception:                           # noqa: BLE001 - unsure: keep it
        return False
    if path.lower().endswith(IMAGE_SUFFIXES):
        return False
    from eubi_bridge.utils.capabilities import has_bioformats
    return not has_bioformats() or path.lower().endswith(SIDE_FILE_SUFFIXES)


def _is_text(path: str) -> bool:
    name = os.path.basename(path).lower()
    return name.endswith(TEXT_SUFFIXES) or "." not in name.lstrip(".")


def _walkable_folder(path: str) -> bool:
    """A folder whose files are inputs (grouped by input_mains), not a zarr store or
    group, whose chunks are no inputs."""
    if not os.path.isdir(path) or path.rstrip("/\\").lower().endswith(".zarr"):
        return False
    return not any(os.path.exists(os.path.join(path, name))
                   for name in (".zattrs", ".zgroup", "zarr.json"))


def _files_below(folder: str) -> List[str]:
    """Every file under `folder`, sorted."""
    return sorted(os.path.join(root, name) for root, _, names in os.walk(folder)
                  for name in names)


def _opens_as_one(folder: str, micro_reader) -> bool:
    """Whether micro-reader opens `folder` as its own dataset (a Cellomics, NDTiff or
    Luxendo dataset folder, an OIF storage folder, a NIS-Elements run folder): the file
    it reads is the folder itself, in it, or beside it. Such a folder stays the input,
    named as before. micro-reader 0.4.0 also opens a folder whose one dataset lies in a
    sub-folder (a sample folder over its export): that one converts its dataset's main
    file, named after the dataset, not after the folder above it."""
    try:
        with micro_reader.open(folder, plates=True) as f:
            main = os.path.normcase(os.path.abspath(f.path))
    except Exception:                           # noqa: BLE001 - no: its files decide
        return False
    here = os.path.normcase(os.path.abspath(folder))
    return main == here or os.path.dirname(main) in (here, os.path.dirname(here))


def group_inputs(input_paths: List[str]) -> List[str]:
    """The inputs as the files to convert, each once (``input_mains``)."""
    return list(dict.fromkeys(m for m in input_mains(input_paths) if m is not None))


def per_image_inputs(df: pd.DataFrame) -> List[str]:
    """The inputs of a conversion table whose ``as_plate`` is off (written per image)."""
    if 'as_plate' not in df.columns:
        return []
    return [str(p) for p, v in zip(df['input_path'], df['as_plate']) if _plate_off(v)]


def _plate_off(value) -> bool:
    """Whether a table cell of ``as_plate`` turns plates off (False / "off" / "false"
    / 0); blank, None, "auto" and on keep them."""
    if _blank_cell(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in ("off", "false", "0", "no")
    return value is False or value == 0


def group_input_rows(df: pd.DataFrame) -> pd.DataFrame:
    """A conversion table (one row per input) with each row's ``input_path`` as the file
    to convert (``input_mains``): rows standing for no image are dropped, and a plate
    named by several rows is converted once, with the first of them. Rows with
    ``as_plate`` off are grouped without plates: the files of a set that is a plate
    only as a plate (a NIS-Elements well-plate run, one ND2 per well) stay inputs of
    their own, each written per image."""
    paths = [str(p) for p in df['input_path'].tolist()]
    off = ([_plate_off(v) for v in df['as_plate'].tolist()] if 'as_plate' in df.columns
           else [False] * len(paths))
    if not any(off):
        mains = input_mains(paths)
    else:
        on_paths = [p for p, o in zip(paths, off) if not o]
        off_paths = [p for p, o in zip(paths, off) if o]
        found = {False: iter(input_mains(on_paths) if on_paths else []),
                 True: iter(input_mains(off_paths, plates=False))}
        mains = [next(found[o]) for o in off]
    given = pd.Series([os.path.normcase(os.path.abspath(p)) for p in paths], index=df.index)
    df = df.assign(input_path=mains)
    df = df[df['input_path'].notna()]
    given = given[df.index]
    # a row is what it writes: one input to two outputs, or two series of it, are two
    # conversions. Spellings of a path differ between a row and a plate's main file
    written = pd.DataFrame({'input_path': [os.path.normcase(os.path.abspath(p))
                                           for p in df['input_path']]})
    for column in WHAT_A_ROW_WRITES:
        if column in df.columns:
            written[column] = [_cell(value) for value in df[column]]
    twice = written.duplicated().to_numpy()
    if not twice.any() and mains == paths:
        return df
    # said only of an input named twice: the files of one plate (a folder's, or listed)
    # all stand for its index, which is no choice made twice (tenth review N-04)
    written['given'] = given.to_numpy()
    named_twice = written.duplicated().to_numpy()
    for path in dict.fromkeys(df.loc[named_twice, 'input_path']):
        logger.warning(f"{path}: chosen more than once for one output; converted once (with "
                       f"the first row's settings)")
    df = df[~twice].reset_index(drop=True)
    if not len(df):
        # the user's input, not a program error: one line (review cycle 1, N-04)
        folder = os.path.commonpath([os.path.dirname(os.path.abspath(p)) for p in paths]) \
            if paths else ""
        names = ", ".join(os.path.basename(p) for p in paths[:5])
        raise InputError(
            f"None of the {len(paths)} file(s) in {folder} is an image eubi-bridge reads, or a "
            f"plate's index or side file: {names}{' ...' if len(paths) > 5 else ''}")
    return df


def _plate_it_opens(path: str, micro_reader) -> Optional[str]:
    """The file micro-reader reads the plate of *path* from, when that is another file. A file
    that is a plate by itself (a CZI or ND2 of one well) says nothing about its neighbours."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with micro_reader.open(path, plates=True) as f:
            other = os.path.normcase(os.path.abspath(f.path)) != \
                os.path.normcase(os.path.abspath(path))
            return str(f.path) if f.plates and other else None


def _warn_plate_given_in_part(group, given: set, key) -> None:
    """One line for a plate of files of one kind (a NIS-Elements run, one ND2 per well) when
    only some are given: each opens the whole plate. An index file over its planes is not
    this: giving it alone is how a plate is chosen."""
    suffix = os.path.splitext(str(group.main))[1].lower()
    peers = [f for f in (group.main, *group.members)
             if os.path.splitext(str(f))[1].lower() == suffix]
    held = sum(key(f) in given for f in peers)
    if 0 < held < len(peers) and len(peers) == 1 + len(group.members):
        logger.warning(f"{held} of the {len(peers)} files of {os.path.basename(str(group.main))}'s "
                       f"set are given: the whole set is converted, not only those files: "
                       f"eubi-bridge converts a plate whole. To convert only those files, each as an image of its own, use --as_plate False.")


def warn_excluded_parts(excluded: list, kept: list) -> None:
    """--excludes takes files out of the inputs, not out of a set that is opened whole: one
    line when excluded files belong to a set that is converted all the same."""
    try:
        import micro_reader
        grouping = micro_reader.group_files(excluded + kept, plates=True)
    except Exception as exc:                    # noqa: BLE001 - a warning only
        logger.debug(f"group_files: {exc}")
        return

    def key(path):
        return os.path.normcase(os.path.abspath(path))

    out, stay = {key(p) for p in excluded}, {key(p) for p in kept}
    for group in grouping.groups:
        files = {key(f) for f in (group.main, *group.parts, *group.members)}
        if group.main is not None and files & out and files & stay:
            logger.warning(f"--excludes: {len(files & out)} excluded file(s) belong to "
                           f"{os.path.basename(str(group.main))}, which is converted whole: "
                           f"they are in the output: eubi-bridge converts a plate whole. To "
                           f"convert only the files kept, each as an image, use --as_plate False.")


#: files opened per folder and shape of name, to find whether they are a plate's planes
OPENS_PER_KIND = 3

#: the columns of a table that, with the input, say which store a row writes
WHAT_A_ROW_WRITES = ('output_path', 'series_index', 'mosaic_tile_index', 'view_index',
                     'illumination_index', 'rotation_index', 'phase_index', 'aggregative_group',
                     'as_plate', 'as_mosaic')


def _cell(value) -> str:
    """A cell as a row's choice: blank and "all" are one choice, the default."""
    if _blank_cell(value) or str(value).strip().lower() in ('all', 'none'):
        return 'all'
    return str(value).strip()


def _files_that_open_their_set(files, owner: dict, key, micro_reader) -> dict:
    """{file: main file} for the given *files* group_files cannot place, because their index is
    not among the inputs (plane files picked with --includes): micro-reader opens the plate
    from any of them, so each would convert the whole plate. A scanR plane only says so when
    opened (its OME-XML points to the plate's descriptor): a file of a folder is opened, and
    those named alike follow it, so a plate's thousands of planes open once."""
    try:
        from micro_reader import _parts
        from micro_reader._registry import BACKENDS
    except ImportError:                         # an older micro-reader: planes open alone
        return {}
    found, opened, tries = {}, {}, {}
    for path in files:
        if key(path) in owner:
            continue
        try:
            main = _parts.main_file(path, BACKENDS, True, True)
            if key(main) == key(path):
                kind = (os.path.dirname(key(path)), _name_shape(path))
                # a plain file first says nothing of the planes named alike, and a folder
                # of plain files must not open them all
                if opened.get(kind) is None and tries.get(kind, 0) < OPENS_PER_KIND:
                    tries[kind] = tries.get(kind, 0) + 1
                    opened[kind] = None
                    opened[kind] = _plate_it_opens(path, micro_reader)
                main = opened.get(kind) or path
        except Exception as exc:                # noqa: BLE001 - the conversion reports it
            logger.debug(f"main file of {path}: {exc}")
            continue
        if key(main) != key(path):
            found[key(path)] = main
    return found


def _say_files_that_open_their_set(found: dict) -> None:
    """The line for files given that convert their whole set (_files_that_open_their_set),
    said once the files that stay images of their own are left out (review cycle 2, R2-08:
    a stray plane-named TIFF in a plate folder was named here, then converted as itself)."""
    if found:
        logger.warning(f"{len(found)} file(s) given belong to "
                       f"{os.path.basename(next(iter(found.values())))}: the whole set is "
                       f"converted, not only those files: eubi-bridge converts a plate whole. "
                       f"To convert only those files, each as an image of its own, use --as_plate False.")


def _name_shape(path: str) -> str:
    """A file's name with its numbers left out: the planes of one plate have one shape."""
    return re.sub(r'\d+', '#', os.path.basename(path))


def input_mains(input_paths: List[str], plates: bool = True) -> List[Optional[str]]:
    """For each input, the file to convert (None: none), found with micro-reader's
    ``group_files`` so that a user need not know which file of a plate (or other image
    set) to choose:

    - every given file of a plate -- its index, its side files (a Harmony measurement's
      ``image.index.txt``, ``.kw.txt``, ``Assaylayout`` files), its planes listed beside
      the index -- stands for the plate's index file;
    - a file of an NDTiff dataset's downsampled level stands for the dataset's full
      resolution (``ndtiff_full_resolution``): one conversion per dataset, not per level;
    - a text file that belongs to no image, given with no other input from its folder
      (a Harmony "hs" archive's ``_objmeta.txt``): the one plate in its folder, when the
      folder holds exactly one; else None, with a warning naming what is there;
    - a text file among other inputs from its folder (a readme): None;
    - everything else stays as given (images, OME-Zarr stores).

    Every group stands for its main file, also one eubi reads with the older readers
    (an OME companion, a Leica XLEF, an Olympus OIF folder): one conversion per set.

    Of several inputs (a folder, a file list), a file no reader recognises -- an
    instrument's or analysis program's side file -- is skipped (None), named in one
    warning (``_not_an_image``); a file given on its own is converted, and fails loudly
    if it is not an image."""
    paths = [str(p) for p in input_paths]
    try:
        import micro_reader
    except Exception:                           # no micro-reader: nothing to group
        return paths

    def key(path):
        return os.path.normcase(os.path.abspath(path))

    def usable(group):
        return group.main is not None

    def group_files(paths_):
        """micro-reader's groups as eubi reads the files (*plates*, True unless the
        job's as_plate is off): sets that are plates only then -- a NIS-Elements
        well-plate run, one ND2 per well -- are one group, so the plate is converted
        once, not once per file. micro-reader before 0.3.0 takes no plates= (and
        knows no such sets)."""
        try:
            return micro_reader.group_files(paths_, plates=plates)
        except TypeError:
            return micro_reader.group_files(paths_)

    def owners(paths_):
        """({file: main file}, [(plate folder, main file)], {image of its own}): the plate
        folder of a plate group is the folder its files share, when that is the index's
        folder or the one above it (a Harmony measurement: Images/, Assaylayout/,
        Evaluation1/, ...). An image of its own is one micro-reader puts in no set (its
        ``singles``: a CV7000 folder's unreferenced_auxiliary_image.tif)."""
        try:
            grouping = group_files(paths_)
        except Exception as exc:                # the conversion reports a bad file
            logger.debug(f"group_files: {exc}")
            return {}, [], set()
        found, trees = {}, []
        # singles are every file in no set: a display_settings.txt too; images only
        singles = {key(p) for p in getattr(grouping, "singles", ()) or ()
                   if os.path.isfile(p) and not _is_text(p) and not _not_an_image(p)}
        given = {key(p) for p in paths_}
        for group in grouping.groups:
            if usable(group):
                if plates:
                    _warn_plate_given_in_part(group, given, key)
                try:
                    main = ndtiff_full_resolution(group.main) or group.main
                except Exception as exc:        # the level is converted as it is
                    logger.debug(f"NDTiff full resolution of {group.main}: {exc}")
                    main = group.main
                files_ = (group.main, *group.parts, *group.members)
                for path in files_:
                    found.setdefault(key(path), main)
                if group.format in PLATE_FORMATS:
                    try:
                        root = os.path.commonpath([os.path.abspath(f) for f in files_])
                    except ValueError:          # files on different drives
                        continue
                    index_folder = os.path.dirname(os.path.abspath(group.main))
                    if key(root) in (key(index_folder), key(os.path.dirname(index_folder))):
                        trees.append((key(root) + os.sep, group.main))
        return found, trees, singles - set(found)

    # OME-Zarr stores and image groups are folders of chunks: never walked. Other folders
    # (a GUI batch row is any ticked entry: a plate folder, its subfolders) are grouped
    # with their files
    files = [p for p in paths if os.path.isfile(p)]
    folders = {p: _files_below(p) for p in paths if _walkable_folder(p)}
    below = [f for listed in folders.values() for f in listed]
    together = list(dict.fromkeys(files + below))
    owner, trees, own_images = owners(together) if together else ({}, [], set())
    for path in files:
        # a file in a plate's own folder that micro-reader names no part of it (Harmony's
        # Evaluation1/ analysis results): the plate's too, not an input of its own; an
        # image of its own there stays itself (tenth review C-21)
        if key(path) not in owner and key(path) not in own_images:
            main = next((m for root, m in trees if key(path).startswith(root)), None)
            if main is not None:
                owner[key(path)] = main
    if plates:
        # an image of its own is no plane of a set, however it is named: a TIFF named like
        # a Harmony plane that the index, opened here, does not list (review cycle 1,
        # N-03). A set not opened here (InCell planes given without their .xdce, which
        # group_files cannot place by name) says nothing of its listing: its planes
        # stand for it, as before
        opened_sets = {key(m) for m in owner.values()}
        opening = {f: main for f, main in _files_that_open_their_set(
            files, owner, key, micro_reader).items()
            if not (f in own_images and key(main) in opened_sets)}
        _say_files_that_open_their_set(opening)
        owner.update(opening)
    out, told = [], set()

    def add(path, why=""):
        if why and key(path) not in told:
            told.add(key(path))
            logger.info(why)
        out.append(path)

    many, skipped = len(paths) > 1, []
    chosen = {owner.get(key(p)) for p in files} - {None}      # the file inputs' images
    for path in paths:
        main = owner.get(key(path))
        if main is not None:
            add(main, "" if key(main) == key(path) else
                f"{path}: a file of {main}; converting {main}")
            continue
        if path in folders:
            held = {owner.get(key(f)) for f in folders[path]} - {None}
            # images of their own beside a set the folder opens as (a CV7000 folder's
            # unreferenced image): the folder converts the set, so these go on their own
            loose = [f for f in folders[path] if key(f) in own_images] if held else []
            if many and held and held <= chosen:
                logger.info(f"{path}: a folder of images chosen already; skipped")
                out.append(None)
            elif len(held) == 1 and not _opens_as_one(path, micro_reader):
                (only,) = held
                add(only, f"{path}: a folder of {only}; converting {only}")
            else:                               # one image as micro-reader opens it, or
                add(path)                       # several: the conversion says which
            for image in loose:
                add(image, f"{image}: an image of its own beside {sorted(held)[0]}; "
                           f"converted as itself")
            continue
        if many and os.path.isfile(path) and not _is_text(path) and _not_an_image(path):
            skipped.append(path)
            out.append(None)
            continue
        if not (os.path.isfile(path) and _is_text(path)):
            add(path)
            continue
        folder = os.path.dirname(os.path.abspath(path))
        others = [p for p in paths if p != path and not _is_text(p)
                  and key(p).startswith(key(folder) + os.sep)]
        if others:
            # a file the user named, so a warning, not an info line hidden by default
            # (review cycle 2, R2-10)
            logger.warning(f"{path}: not an image; skipped")
            out.append(None)
            continue
        try:
            grouping = group_files(folder)
        except Exception as exc:
            logger.warning(f"{path}: not an image, and its folder could not be searched "
                           f"for one ({exc}); skipped")
            out.append(None)
            continue
        mains = sorted({g.main for g in grouping.groups if usable(g)})
        if len(mains) == 1:
            add(mains[0], f"{path}: not an image; its folder holds one plate / image "
                          f"set, {mains[0]}: converting it")
            continue
        shown = ", ".join(mains[:5]) + (" ..." if len(mains) > 5 else "")
        logger.warning(
            f"{path}: not an image; skipped. Its folder holds "
            + (f"{len(mains)} plates / image sets ({shown}): choose a file of the one "
               f"to convert" if mains else "no plate or image set"))
        out.append(None)
    if skipped:
        names = ", ".join(os.path.basename(s) for s in skipped[:8])
        logger.warning(f"{len(skipped)} of the {len(paths)} input files are not images "
                       f"eubi-bridge reads and are skipped: {names}"
                       + (" ..." if len(skipped) > 8 else ""))
    return out


def disambiguate_output_names(input_paths: List[str], per_image=()) -> dict:
    """Map each input path to a unique output basename.

    A plate's data file (a Cellomics ``.C01``, a NIS-Elements run's ``.nd2``) is named
    after its folder, the plate's name, unless it is in `per_image` (its job has
    ``as_plate`` off: each of its images is written on its own, under its own name).

    Outputs are named after the input's basename, so two inputs from different
    folders that share a name would target one output — the first written, the
    second either refused (``overwrite=False``) or silently destroying the first
    (``overwrite=True``).  Neither is acceptable in a batch the user cannot split
    by hand, and the Run tab offers only one output folder.

    Colliding names take one parent directory at a time as a prefix, repeating
    until every name is unique::

        A/img.tif  ->  A_img
        B/img.tif  ->  B_img

    Files of one folder that share a name but not a suffix (an Airyscan
    ``x.czi`` and its ``x.ims``) take their suffix instead, since their parents
    cannot tell them apart::

        A/img.czi  ->  img_czi
        A/img.ims  ->  img_ims

    A drive (``C:``) is never a prefix.  Only names that actually collide are
    changed, so a batch of distinct names keeps exactly the output paths it had
    before.  Paths that remain identical after exhausting their parents (the same
    file listed twice) fall back to a numeric suffix, which always terminates.
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

    per_image = {str(p) for p in per_image}

    def named(path: str) -> Optional[str]:
        return named_folder(path, plate=path not in per_image)

    def stem(path: str) -> str:
        # Matches _generate_output_path (output_stem): 'image.ome.tiff' -> 'image',
        # 'img.v2.tif' -> 'img.v2'; a plate index with a fixed name, or a plate's data
        # file, takes its folder's name.
        folder = named(path)
        if folder is not None:
            return _segments(folder)[-1]
        return output_stem(path) if _segments(path) else ''

    def parents(path: str) -> List[str]:
        folder = named(path)
        found = _segments(folder)[:-1] if folder is not None else _segments(path)[:-1]
        # a drive ('C:') is not a folder name, and ':' is no file name character
        return [part for part in found if not part.endswith(':')]

    same_folder = Counter((tuple(parents(p)), stem(p)) for p in set(input_paths))

    def own(path: str) -> str:
        """The stem, with the suffix when files of the same folder share the stem."""
        name = stem(path)
        if same_folder[(tuple(parents(path)), name)] > 1 and named(path) is None:
            suffix = _segments(path)[-1][len(name):].strip('.').lower().replace('.', '_')
            if suffix:
                return f"{name}_{suffix}"
        return name

    # a folder every input shares tells them apart from nothing: prefixes are taken from
    # the folders below the inputs' common root, growing one folder at a time, so
    # coll/a/img and coll/c/a/img are a_img and c_a_img (review cycle 1, N-19: they were
    # coll_img, named after the input folder, and c_img, its own folder dropped)
    trees = [parents(p) for p in set(input_paths)]
    shared = 0
    while trees and all(len(tree) > shared for tree in trees) and             len({tree[shared] for tree in trees}) == 1:
        shared += 1

    def below(path: str) -> List[str]:
        return parents(path)[shared:]

    names = {path: own(path) for path in input_paths}
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
            available = below(path)
            if depth <= len(available):
                names[path] = "_".join(available[-depth:] + [own(path)])
                progressed = True
        if not progressed:
            # Parents exhausted — distinct inputs cannot be told apart by path
            # alone (or the same path was listed twice).  Numbering terminates.
            for name in clashing:
                for index, path in enumerate(p for p, n in list(names.items())
                                             if n == name):
                    names[path] = f"{name}_{index + 1}" if index else name
            break
    return names


def _without_own_outputs(paths: list, output_path, input_path) -> list:
    """Leaves out what a run wrote: a store being written beside another (a killed run), and
    the output folder when it is inside the input folder: a repeated run would convert its
    own outputs."""
    paths = [p for p in paths if not re.search(r'\.(writing|replaced)-[0-9a-f]{8}\.zarr$',
                                               str(p).rstrip('/\\'))]
    input_path = str(input_path).split('*')[0]          # a pattern: the folder before it
    if not output_path or '://' in str(output_path) or not os.path.isdir(input_path):
        return paths
    out = os.path.normcase(os.path.abspath(str(output_path))) + os.sep
    folder = os.path.normcase(os.path.abspath(input_path)) + os.sep

    def inside(path):
        return os.path.normcase(os.path.abspath(path)).startswith(out)

    if out == folder:
        # the output folder is the input folder: a store beside its source is its output
        stems = {os.path.splitext(p)[0] for p in paths if not str(p).endswith('.zarr')}
        kept = [p for p in paths if not (str(p).endswith('.zarr') and p[:-len('.zarr')] in stems)]
    elif out.startswith(folder):
        kept = [p for p in paths if not inside(p)]
    else:
        return paths
    if len(kept) != len(paths):
        logger.warning(f"{len(paths) - len(kept)} input(s) inside the output folder are left "
                       f"out: {output_path}")
    return kept


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

    # A group inside a store that is itself an OME-Zarr image (a well's field, one series
    # of a layout; NGFF 0.6 places such images by a scene on the group above): one input,
    # as named. Its name does not end with .zarr, so it would otherwise be read as a
    # folder of files.
    if (os.path.isdir(input_path) and not input_path.rstrip('/\\').endswith('.zarr')
            and _is_image_group(input_path)):
        return [input_path]

    # One file or one zarr store is that input, and nothing else: a pattern ``*name``
    # would also take every file whose name ends in *name*
    if '*' not in input_path and (os.path.isfile(input_path) or input_path.endswith('.zarr')):
        paths = [input_path]
    else:
        # '**' in the user's pattern is any depth, also none (folder/**/*.tif takes the
        # folder's own files; review cycle 1, N-22: it took those one folder down only). A
        # folder is listed and walked, as before: no recursive glob over a large plate
        recursive = '**' in input_path
        if '*' not in input_path and not input_path.endswith('.zarr'):
            input_path = os.path.join(input_path, '**')
        paths = sensitive_glob(input_path, recursive=recursive, sensitive_to='.zarr')

    # Filter by includes/excludes; macOS's AppleDouble stubs (._name.tiff) hold no image
    paths = [p for p in paths
             if not is_apple_double(p) and _matches_any(p, includes)]
    paths = _without_own_outputs(paths, kwargs.get('output_path'), original_input_path)
    excluded = [p for p in paths if excludes is not None and _matches_any(p, excludes)]
    left_out = set(excluded)
    paths = [p for p in paths if p not in left_out]
    if excluded and paths and not _plate_off(kwargs.get("as_plate")):
        warn_excluded_parts(excluded, paths)

    # Remove zarr.json files
    paths = list(filter(lambda path: not path.endswith('zarr.json'), paths))
    
    if len(paths) == 0:
        raise InputError(f"No valid paths found for {original_input_path}")
    
    return sorted(paths)


def _blank_cell(value) -> bool:
    """True for an empty table cell (None, NaN, NA, '')."""
    if value is None or value is pd.NA or (isinstance(value, str) and not value.strip()):
        return True
    return isinstance(value, float) and value != value


def _renamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """*df* with the old names of renamed settings (a ``scene_index`` column of a
    0.2.x table) under their new names, warned about as ``rename_settings`` warns,
    before the global values fill in the columns the table lacks: else the global
    ``series_index`` would shadow the table's own column. Given both, the new wins;
    for the index (a table eubi-bridge 0.3.0 wrote both into, plan 8.32) per row:
    silently where the two agree or the new cell is blank, else with a
    SeriesIndexConflictWarning naming both (``resolve_index_pair``)."""
    from eubi_bridge.core.config_models import (
        RENAMED_INDEX_SETTINGS, rename_settings, resolve_index_pair)
    for old, new in RENAMED_INDEX_SETTINGS.items():
        if old not in df.columns or new not in df.columns:
            continue
        df = df.copy()
        df[new] = df[new].astype(object)
        for i in df.index:
            scene, series = df.at[i, old], df.at[i, new]
            if _blank_cell(scene):
                continue
            if _blank_cell(series):
                df.at[i, new] = scene
            else:
                name = df.at[i, "input_path"] if "input_path" in df.columns else i
                resolve_index_pair(scene, series,
                                   where=f"conversion table row {i} ({name}): ")
        df = df.drop(columns=[old])
    kept = rename_settings({c: c for c in df.columns})      # {name now: column}
    if len(kept) == len(df.columns) and all(k == c for k, c in kept.items()):
        return df
    dropped = [c for c in df.columns if c not in kept.values()]
    return df.drop(columns=dropped).rename(
        columns={column: name for name, column in kept.items() if name != column})


def is_apple_double(path) -> bool:
    """A macOS AppleDouble stub (``._name.tiff``): a copy's resource fork, no image, beside
    every file that passed through a Mac or a Mac-formatted drive."""
    return os.path.basename(str(path)).startswith("._")


def _apply_table_filters_and_defaults(df: pd.DataFrame,
                                      global_kwargs: dict) -> pd.DataFrame:
    """Apply include/exclude filters and global column defaults to *df*.

    Shared by every branch of :func:`take_filepaths` so a pattern behaves
    identically whether the rows came from a directory, an explicit file list, a
    conversion table on disk, or a table handed over in memory.
    """
    df = _renamed_columns(df)

    def _keep(row) -> bool:
        inp = row["input_path"]
        if is_apple_double(inp):
            return False                # a table row too, not only a folder walk (N-06)
        includes = global_kwargs.get('includes')
        excludes = global_kwargs.get('excludes')
        if not _matches_any(inp, includes):
            return False
        if excludes is not None and _matches_any(inp, excludes):
            return False
        return True

    df = df[df.apply(_keep, axis=1)]

    # the table's own columns, before the run-wide values fill the others in: what a
    # refused group's rows are listed with at the end of a run
    table_columns = list(df.columns)

    # Global values fill in only the columns the table does not carry itself, so
    # a per-row override always wins over the run-wide setting.
    for k, v in global_kwargs.items():
        if k not in df.columns:
            if hasattr(v, '__len__') and not isinstance(v, str):
                df[k] = [v for _ in range(len(df))]
            else:
                df[k] = v

    if 'output_path' in df.columns:
        # a local output as the folder it names, without "..": TensorStore refuses
        # "out/../o" (review cycle 1, N-20: the command line's output was fixed, a table
        # row's was not); an object-store URL stays as it is
        from eubi_bridge.core.output_store import _is_remote_path
        df = df.assign(output_path=[
            os.path.abspath(str(p)) if not is_blank_cell(p) and not _is_remote_path(str(p))
            else p for p in df['output_path']])
    df.attrs['table_columns'] = table_columns
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

    # ── Explicit list of paths (GUI multi-select or programmatic use) ──────────
    if isinstance(input_path, (list, tuple)):
        given = [str(p) for p in input_path if p]
        # macOS's AppleDouble stubs (._name.tiff) hold no image, in a list as in a folder
        fps = [p for p in given if not is_apple_double(p)]
        if not fps:
            raise ValueError("Empty list of input paths provided." if not given else
                             f"The {len(given)} input path(s) are all macOS ._ stub files, "
                             f"which hold no image: {given[:3]}")
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
        # includes / excludes as for a folder or a table (review cycle 2, R2-10: a list
        # converted every file it named)
        includes, excludes = global_kwargs.get('includes'), global_kwargs.get('excludes')
        if includes is not None or excludes is not None:
            dropped = [e for e in ([excludes] if isinstance(excludes, str) else excludes or [])
                       if e]
            kept = df['input_path'].map(
                lambda p: _matches_any(p, includes)
                and not any(pattern_matches(p, e) for e in dropped))
            if not kept.any():
                raise InputError(f"None of the {len(df)} listed input path(s) is left by "
                                 f"--includes {includes} / --excludes {excludes}: "
                                 f"{', '.join(os.path.basename(p) for p in fps[:5])}")
            df = df[kept].reset_index(drop=True)
        return df

    # Handle different input types
    if is_table_path(input_path):

        logger.info(f"Loading conversion table from {input_path}")
        
        # Get CSV directory for resolving relative paths
        csv_dir = os.path.dirname(os.path.abspath(input_path))
        
        if input_path.endswith((".csv", ".tsv", ".txt")):
            try:
                df = pd.read_csv(input_path)
            except (pd.errors.ParserError, UnicodeDecodeError) as exc:
                raise InputError(f"{input_path} is neither an image file nor a conversion "
                                 f"table: {exc}") from exc
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
        raise InputError(f"Invalid input path: {input_path}: no such file or folder")

    # Normalize input column name
    if "filepath" in df.columns and "input_path" not in df.columns:
        df.rename(columns={"filepath": "input_path"}, inplace=True)

    if "input_path" not in df.columns:
        raise InputError(f"Table must include an 'input_path' or 'filepath' column: "
                         f"{input_path} is neither an image file nor a conversion table.")

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
