from __future__ import annotations

from typing import TYPE_CHECKING

import copy
import os
import re
from typing import Dict, Iterable, List, Union

import numpy as np

from eubi_bridge.utils.errors import InputError
from eubi_bridge.utils.logging_config import get_logger
from eubi_bridge.utils.optional_deps import is_dask_array
from dyna_zarr import operations as ops

if TYPE_CHECKING:
    import dask.array as da
                        
logger = get_logger(__name__)

get_numerics = lambda string: list(re.findall(r'\d+', string))


def get_matches(pattern, strings, return_non_matches=False):
    """
    Search for regex pattern matches in a list of strings.

    Args:
        pattern (str): Regular expression pattern to search for.
        strings (list): List of strings to search within.
        return_non_matches (bool): If True, returns all results including None for non-matches.
                                 If False, returns only successful matches.

    Returns:
        list: List of match objects (or None for non-matches if return_non_matches is True).
    """
    matches = [re.search(pattern, string) for string in strings]
    if return_non_matches:
        return matches
    return [match for match in matches if match is not None]


def concatenate_shapes_along_axis(shapes: Iterable,
                                  axis: int
                                  ) -> list:
    """
    Concatenate shapes along a specified axis.

    Args:
        shapes (Iterable): Iterable of shape tuples to concatenate.
        axis (int): Axis along which to concatenate shapes.

    Returns:
        list: New shape after concatenation along the specified axis.

    Raises:
        ValueError: If dimensions other than the concatenation axis don't match.
    """
    reference_shape = shapes[0]
    concatenated_shape = [num for num in reference_shape]
    for shape in shapes[1:]:
        for idx, size in enumerate(shape):
            if idx == axis:
                concatenated_shape[idx] += size
            else:
                assert size == reference_shape[idx], ValueError(
                    "For concatenation to succeed, all dimensions except the dimension of concatenation must match.")
    return concatenated_shape


def accumulate_slices_along_axis(shapes: Iterable,
                                 axis: int,
                                 slices: Union[tuple, list] = None
                                 ) -> list:
    """
    Calculate accumulated slices for concatenation along a specified axis.

    Args:
        shapes (Iterable): Iterable of shape tuples to be concatenated.
        axis (int): Axis along which to calculate slices.
        slices (Union[tuple, list], optional): Initial slices. If None, creates full slices.

    Returns:
        list: List of slice tuples for each input shape.
    """
    reference_shape = shapes[0]
    if slices is None:
        slices = [[slice(None, None) for _ in reference_shape] for _ in shapes]
    assert (len(shapes) == len(slices))
    sizes_per_axis = [shape[axis] for shape in shapes]
    cummulative_sizes = [0] + np.cumsum(sizes_per_axis).tolist()
    slice_tuples = [(cummulative_sizes[idx], cummulative_sizes[idx + 1]) for idx in range(len(sizes_per_axis))]
    for idx, tup in enumerate(slice_tuples):
        slc = slices[idx]
        slclist = list(slc)
        slclist[axis] = slice(*tup)
        slices[idx] = tuple(slclist)
    return slices


def _shared_folder_length(paths: list) -> int:
    """How many leading characters of *paths* are the folders they all share."""
    folders = [re.split(r'[/\\]', path)[:-1] for path in paths]
    shared = 0
    for parts in zip(*folders):
        if len(set(parts)) > 1:
            break
        shared += 1
    return sum(len(part) + 1 for part in folders[0][:shared]) if folders else 0


def find_tag(text: str, tag: str, numbered: bool = True, inside_words: bool = True):
    """The match of *tag* in *text*, None without one. A tag written as a word of its own
    speaks first (ch1 in batch3_ch1: ch3 is inside a word), then the one nearest the file's
    name (a_T1 in Dir_T9/a_T1). *numbered*: a prefix before a number, the digits right after
    it in group 1 and nothing more (T00h15m is 0, z0.5 is 0); else the text itself.
    *inside_words* False leaves out a tag that follows a letter. A file's extension holds no
    tag (h5 in exp_h1.h5)."""
    stripped = str(tag).rstrip('/\\')
    pattern = re.escape(stripped) + (r'[/\\]?(\d+)' if numbered else '')
    best, rank = None, None
    named = len(os.path.splitext(text)[0])
    for match in re.finditer(pattern, text):
        if match.start() >= named:
            break
        before = not text[max(match.start() - 1, 0):match.start()].isalpha()
        after = not text[match.end():match.end() + 1].isalpha()
        if not (before or inside_words):
            continue
        place = (before, after, match.start())
        if rank is None or place > rank:
            best, rank = match, place
    return best


def tag_folder(paths: Iterable[str], tag: str, numbered: bool = True) -> str:
    """The folder *tag* is read below. A tag says something about the files, so it is read
    below the folder they all share: a folder above them (C1_T5run, gfp_mcherry) names no
    time point and no channel. Only when no file has the tag there, the shared folder's own
    name too: the tag can be a folder itself, as in frames/000/a.tif. One folder only: further up
    are the folders the data is kept in (hunt11, Dataset2), which say nothing about the files,
    so a tag no file has is not found there either (it read t11 from hunt11)."""
    paths = list(paths)
    if not paths:
        return ''
    cut = _shared_folder_length(paths)
    if cut and not any(find_tag(path[cut:], tag, numbered) for path in paths):
        start = max(paths[0].rfind('/', 0, cut - 1), paths[0].rfind('\\', 0, cut - 1)) + 1
        if any(find_tag(path[start:], tag, numbered) for path in paths):
            return paths[0][:start]
    return paths[0][:cut]


def _below(path: str, folder: str) -> str:
    return path[len(folder):] if path.startswith(folder) else path


def _shown(tags: list, place: tuple) -> str:
    """A place as the names spell it: T1, gfp."""
    return " ".join(value if isinstance(tag, tuple)
                    else str(tag).rstrip("/\\") + "_".join(str(n) for n in value)
                    for tag, value in zip(tags, place))


class Place(tuple):
    """A file's place along a numbered tag: the digits right after the tag, and nothing else
    of the name (find_tag). *before*: the numbers ahead of the tag (a date, a run). They order
    nothing: the tag does; a line says so when they would order the files otherwise
    (_say_order)."""

    def __new__(cls, text: str, match):
        self = super().__new__(cls, (int(match.group(1)),))
        self.before = tuple(int(n) for n in re.findall(r'\d+', text[:match.start()]))
        return self


def read_tag(paths: Iterable[str], tag: Union[str, tuple, list], folders: dict = None) -> dict:
    """{path: its place along *tag*}: a number for a single tag, the name for several named
    ones (tag_folder, find_tag); a file without the tag is left out. When some file has
    the tag as a word of its own, one that has it only inside a word (Test1.tif beside
    img_t1.tif) has not got it. Of several names, one holding another is searched first
    (GM2_GD2 before GD2), and its files are not searched for the names it holds. *folders*:
    {tag: folder} of a run planned already: one group of its files is read as they were all
    read. The only reader of tags in eubi-bridge."""
    paths = list(paths)
    folders = folders or {}
    if isinstance(tag, (tuple, list)):
        matches = {path: [] for path in paths}
        # the most unique name is searched first, and a file it matches is not searched for
        # the names it holds: GM2_GD2 before GD2. Only the search: the join keeps the order given
        for item in sorted(tag, key=lambda item: -len(str(item))):
            left = [path for path in paths
                    if not any(str(item) in str(other) for _, other, _ in matches[path])]
            folder = folders[str(item)] if str(item) in folders else tag_folder(paths, item, False)
            texts = {path: _below(path, folder) for path in left}
            words = any(find_tag(text, item, False, False) for text in texts.values())
            for path, text in texts.items():
                match = find_tag(text, item, False, not words)
                if match:
                    matches[path].append((match, item, text))
        return {path: _named(path, found) for path, found in matches.items() if found}
    folder = folders[str(tag)] if str(tag) in folders else tag_folder(paths, tag)
    texts = {path: _below(path, folder) for path in paths}
    words = any(find_tag(text, tag, True, False) for text in texts.values())
    found = {path: find_tag(text, tag, True, not words) for path, text in texts.items()}
    return {path: Place(texts[path], match) for path, match in found.items() if match}


def _named(path: str, found: list) -> str:
    """Which named tag the file has, of those *found* in it: a file's name speaks before its
    folders, and two tags in one name say nothing."""
    def rank(match, item, text):
        at = len(path) - len(text)                      # the place in the path
        return (not text[max(match.start() - 1, 0):match.start()].isalpha(),
                not text[match.end():match.end() + 1].isalpha(),
                at + match.start(), len(item), at + match.end())
    ranks = [(rank(match, item, text), item) for match, item, text in found]
    best, name = max(ranks)
    for other, item in ranks:
        apart = other[4] <= best[2] or best[4] <= other[2]
        between = path[min(other[2], best[2]):max(other[4], best[4])]
        if apart and other[:2] == best[:2] and not re.search(r'[/\\]', between):
            raise InputError(
                f"One file matches two of the tags: {os.path.basename(path)} has {item} and "
                f"{name}. Give tags that each match one channel's files.")
    return name


def reduce_paths_flexible(paths: Iterable[str],
                          dimension_tag: Union[str, tuple, list],
                          replace_with: str = 'set') -> str:
    """
    Reduces a list of similar paths by merging over the specified dimension.

    - If `dimension_tag` is a string (e.g., 'T' or 'Channel'), it's assumed to be followed by digits;
      the digits are replaced with `replace_with`.
    - If `dimension_tag` is a tuple/list (e.g., ('Au', 'Apo')), collects the parts containing these
      tokens from all paths and combines them with the replace_with suffix.
    """

    paths = list(paths)
    if not paths:
        return ""
    
    if len(paths) == 1:
        return paths[0]

    if isinstance(dimension_tag, str):
        # Normalize: strip any trailing path separator so "frames" and "frames\" are identical.
        tag_stripped = dimension_tag.rstrip('/\\')

        # Only where the tag was read (find_tag): covers inline patterns (T0001) and
        # directory-component patterns (frames\000). The optional separator goes with the
        # number, so the replacement never leaves a dangling separator before replace_with.
        def replace_tag(path):
            match = find_tag(path, tag_stripped)
            if match is None:
                return path
            return path[:match.start()] + tag_stripped + replace_with + path[match.end():]

        replaced_paths = [replace_tag(p) for p in paths]
        return replaced_paths[0]

    elif isinstance(dimension_tag, (tuple, list)):
        # Categorical case: merge categorical dimension tags
        unique_vals = list(dimension_tag)
        
        # Find which tag appears in each full path and extract before/after parts
        # Search in the full path, not just the basename, to handle nested directories
        before_parts = []
        after_parts = []
        found_tags = []
        tag_indices = []
        
        for fullpath in paths:
            found = False
            for tag in unique_vals:
                if tag in fullpath:
                    idx = fullpath.find(tag)
                    before = fullpath[:idx]
                    after = fullpath[idx + len(tag):]
                    
                    before_parts.append(before)
                    after_parts.append(after)
                    found_tags.append(tag)
                    tag_indices.append(idx)
                    found = True
                    break
            
            if not found:
                raise ValueError(f"No tags {unique_vals} found in {fullpath}")
        
        # Find common prefix of all before_parts
        common_before = os.path.commonprefix(before_parts)
        
        # Find common suffix of all after_parts
        reversed_after_parts = [s[::-1] for s in after_parts]
        common_after_reversed = os.path.commonprefix(reversed_after_parts)
        common_after_suffix = common_after_reversed[::-1]
        
        # Extract variable parts: what's left after removing common prefix/suffix
        variable_before_parts = [b[len(common_before):] for b in before_parts]
        variable_after_parts = [a[:-len(common_after_suffix)] if common_after_suffix else a for a in after_parts]
        
        # Interleave: For each file, merge: variable_before[i] + tag[i] + variable_after[i]
        # Then concatenate all these merged units
        merged_units = []
        for var_before, tag, var_after in zip(variable_before_parts, found_tags, variable_after_parts):
            unit = var_before + tag + var_after
            merged_units.append(unit)
        
        merged_content = ''.join(merged_units)
        
        # Final result = common_before + merged_content + replace_with + common_after_suffix
        # This ensures replace_with is placed right after the merged content, not at the end
        result_path = common_before + merged_content + f'{replace_with}' + common_after_suffix
        
        return result_path

    else:
        raise ValueError("dimension_tag must be a string or a tuple/list of strings")


def parse_channel_tag_from_string(channel_tag: Union[str, tuple, None]) -> Union[tuple, str, None]:
    """
    Parse a channel tag as a tuple of strings.

    Args:
        channel_tag (str): The channel tag to parse.

    Returns:
        tuple: The parsed channel tag as a tuple of strings.
    """
    if isinstance(channel_tag, str) and ',' in channel_tag:
        channel_tag = channel_tag.split(',')
    if isinstance(channel_tag, (tuple, list)):
        return tuple(str(item).strip() for item in channel_tag if str(item).strip())
    return channel_tag


def channel_tag_format_error(channel_tag, filenames=None) -> str:
    """Build an explanatory message for a single channel tag that yields no groups.

    A bare (single) channel tag is interpreted as a prefix immediately followed
    by a number; named/categorical channels must be given as a comma-separated
    list.  This message tells the user exactly how to fix the invocation.
    """
    msg = (
        f"Channel tag '{channel_tag}' was given as a single tag, which is "
        f"interpreted as a prefix immediately followed by a number "
        f"(e.g. 'channel' matches 'channel1', 'channel2'; 'c_' matches 'c_0', "
        f"'c_1'), but no filenames contain '{channel_tag}<number>'. For named "
        f"(categorical) channels that are not numbered, pass a comma-separated "
        f"list of the full per-channel tags instead, e.g. "
        f"\"--channel_tag gfp,mcherry\" or \"--channel_tag c_gfp,c_mcherry\"."
    )
    if filenames:
        msg += f"\nAvailable files: {list(filenames)}"
    return msg


class FileSet:
    """
    A class to manage file paths and their shapes for multi-dimensional data.

    This class handles file paths and their corresponding array data, supporting operations
    like concatenation along specified axes. It's designed to work with up to 5 dimensions (t, c, z, y, x).
    """

    AXIS_DICT = {
        0: 't',
        1: 'c',
        2: 'z',
        3: 'y',
        4: 'x'
    }


    def __init__(self,
                 filepaths: Iterable[str],
                 shapes: Iterable[tuple | list] = None,
                 axis_tag0: Union[str, tuple] = None,
                 axis_tag1: Union[str, tuple] = None,
                 axis_tag2: Union[str, tuple] = None,
                 axis_tag3: Union[str, tuple] = None,
                 axis_tag4: Union[str, tuple] = None,
                 arrays: Iterable[da.Array] = None,
                 tag_folders: dict = None):
        """
        Initialize the FileSet class.

        Args:
            filepaths: The file paths of the arrays.
            shapes: The shapes of the arrays. Required if arrays is not provided.
            axis_tag0-4: Tags for each axis (t, c, z, y, x).
            arrays: The arrays. If provided, shapes can be None.
            tag_folders: {tag: folder} of a run planned already (self.tag_folders there): a
                worker sees one group's files, and must read them as the plan did.
        """
        if shapes is None and arrays is None:
            raise ValueError("Either shapes or arrays must be provided.")

        if arrays is not None:
            self.array_dict = dict(zip(filepaths, arrays))
            shapes = [arr.shape for arr in arrays]
        else:
            self.array_dict = None

        self.shape_dict = dict(zip(filepaths,
                                   shapes)
                               )
        self.axis_tags = [parse_channel_tag_from_string(tag) for tag in
                          (axis_tag0, axis_tag1, axis_tag2, axis_tag3, axis_tag4)]

        # Initialize dimension tags and specified axes
        self.dimension_tags = []
        self.specified_axes = []
        for axis, tag in enumerate(self.axis_tags):
            if tag is not None:
                self.dimension_tags.append(tag)
                self.specified_axes.append(axis)

        self.group = {'': list(filepaths)}
        self.slice_dict = {
            path: tuple(slice(0, size) for size in shape)
            for path, shape in self.shape_dict.items()
        }
        self.path_dict = dict(zip(filepaths, filepaths))
        #: {tag item: the folder it is read below}, over all the files of the run
        self.tag_folders = dict(tag_folders or {})
        for tag in self.dimension_tags:
            items = tag if isinstance(tag, tuple) else (tag,)
            for item in items:
                if str(item) not in self.tag_folders:
                    self.tag_folders[str(item)] = tag_folder(filepaths, item,
                                                             not isinstance(tag, tuple))
        #: {tag: {path: its place along the tag}} (read_tag)
        self._places = {tag: read_tag(filepaths, tag, self.tag_folders)
                        for tag in self.dimension_tags}

    def _rank(self, tag, path):
        """Where *path* comes along *tag*: its number, or the name's place among the names.
        The tag the user names orders the files, nothing else in their names."""
        place = self._places[tag][path]
        return (tag.index(place),) if isinstance(tag, tuple) else tuple(place)

    def in_order(self) -> list:
        """The files in the order of their tags (t, c, z, y, x), not of their paths."""
        tags = [tag for tag in self.dimension_tags]
        return sorted(self.group[''], key=lambda path: tuple(
            self._rank(tag, path) if path in self._places[tag] else -1 for tag in tags))

    def check_places(self, axes: Iterable[int]) -> None:
        """Each output holds one file at each place along the joined *axes*, and every place
        when two axes are joined: two files at one place would become two planes, and which
        is which is a guess."""
        joined = [self.axis_tags[axis] for axis in axes if self.axis_tags[axis] is not None]
        others = [tag for tag in self.dimension_tags if tag not in joined]
        for paths in self._split_by(*others).values():
            places = {}
            for path in paths:
                if all(path in self._places[tag] for tag in joined):
                    places.setdefault(tuple(self._places[tag][path] for tag in joined),
                                      []).append(path)
            for place, same in places.items():
                if len(same) > 1:
                    cut = _shared_folder_length(same)
                    names = [path[cut:] for path in same[:3]]
                    raise InputError(
                        f"{len(same)} files are at one place, {_shown(joined, place)}: "
                        f"{', '.join(names)}{' ...' if len(same) > 3 else ''}. Add a tag for what "
                        f"differs between them, or narrow the input with --includes or --excludes.")
            grid = [list(dict.fromkeys(place[at] for place in places)) for at in range(len(joined))]
            total = np.prod([len(values) for values in grid], dtype=int) if grid else 0
            if len(places) != total:
                # the places with no file, named (review cycle 2, R2-19: the first files that
                # do exist were listed, so the gap had to be found by hand)
                import itertools
                empty = [p for p in itertools.product(*grid) if p not in places]
                shown = ', '.join(_shown(joined, p) for p in empty[:8])
                raise InputError(
                    f"The tags {joined} leave places empty: {len(places)} files for a grid of "
                    f"{' x '.join(str(len(values)) for values in grid)}; no file for "
                    f"{shown}{' ...' if len(empty) > 8 else ''}. Every combination needs its "
                    f"file: add the missing ones, or narrow the input with --includes or "
                    f"--excludes.")
            self._say_order(joined, paths)
        self._say_gaps(joined)

    def _say_gaps(self, joined: list) -> None:
        """One line per numbered joined tag whose numbers leave some out (a_T1, a_T2, a_T4):
        the files are joined in order, one after the other, so a missing time point is
        closed up (tenth review cycle 1, N-01). The step is the smallest between the numbers
        (T0, T10, T20 leave nothing out)."""
        for tag in joined:
            if isinstance(tag, tuple):
                continue
            values = sorted({place[0] for place in self._places[tag].values()})
            if len(values) < 2:
                continue
            step = min(b - a for a, b in zip(values, values[1:]))
            missing = [v for v in range(values[0], values[-1] + 1, step) if v not in set(values)]
            if missing and step > 0:
                name = str(tag).rstrip("/\\")
                shown = ", ".join(f"{name}{v}" for v in missing[:5])
                logger.warning(f"Tag {name}: no file for {shown}{' ...' if len(missing) > 5 else ''}; "
                               f"the {len(values)} values found are joined one after the other, "
                               f"so the gap is closed up (the scale does not know it).")

    def _say_order(self, joined: list, paths: list) -> None:
        """One line when numbers ahead of a tag (a date, a run) would order the files otherwise
        than the tag does: a tag that starts again each day joins the days out of order."""
        for tag in joined:
            if isinstance(tag, tuple):
                continue
            # along one tag the other joined tags stay put: Channel1-T0001 and Channel2-T0001
            # are two channels, their channel numbers order nothing along T
            others = [other for other in joined if other != tag]
            lines = {}
            for path in paths:
                if all(path in self._places[one] for one in joined):
                    lines.setdefault(tuple(self._places[one][path] for one in others),
                                     []).append(path)
            for kept in lines.values():
                rule = sorted(kept, key=lambda path: self._rank(tag, path))
                ahead = sorted(kept, key=lambda path: (self._places[tag][path].before,
                                                        self._rank(tag, path)))
                if rule != ahead:
                    break
            else:
                continue
            if rule != ahead:
                logger.warning(f"Tag {tag}: the files are joined in the order of {tag}, from "
                               f"{os.path.basename(rule[0])}; the numbers ahead of it (a date, a "
                               f"run) would order them otherwise, from "
                               f"{os.path.basename(ahead[0])}.")

    def concatenate(self, arrays, axis: int):
        concatenate = ops.concatenate
        for arr in self.array_dict.values():
            if is_dask_array(arr):
                import dask.array as da
                concatenate = da.concatenate
        return concatenate(arrays, axis=axis)

    def get_numerics_per_dimension_tag(self,
                                       dimension_tag: str
                                       ) -> List[str]:
        """
        Extract numeric values from filepaths for a given dimension tag.

        Args:
            dimension_tag (str): The dimension tag to extract numerics for
                (e.g., 't' for time).

        Returns:
            list: List of numeric strings extracted from the filepaths.

        Example:
            >>> f = FileSet(['file_t0001_channel1.ome.tif', 'file_t0002_channel2.ome.tif'])
            >>> f.get_numerics_per_dimension_tag('t')
            ['0001', '0002']
        """
        filepaths = list(self.group.values())[0]
        folder = self.tag_folders[str(dimension_tag)]
        matches = [find_tag(_below(path, folder), dimension_tag) for path in filepaths]
        # TODO: add an incrementality validator
        return [match.group(1) for match in matches if match]

    def _csplit_by(self, tup: tuple, group: dict) -> dict:
        """
        Split the filepaths of *group* by the given dimension tags.

        Args:
            tup (tuple): A tuple of dimension tags to split by.
            group (dict): The groups so far (a split by another tag stays).

        Returns:
            dict: The split group as a dictionary.
        """
        # a group need not hold every tag (split by channel first, a group of files holds one)
        split = {}
        places = self._places[tup]
        for key, filepaths in group.items():
            for tag in tup:
                paths = [path for path in filepaths if places.get(path) == tag]
                if paths:
                    split[f"{key}-{tag}" if key else tag] = paths
        return split

    def _split_by(self, *args):
        """
        Split the filepaths in the group by the given dimension tags.

        Args:
            *args (str): The dimension tags to split by.

        Returns:
            dict: The split group as a dictionary.
        """
        group = copy.deepcopy(self.group)
        for dim in args:
            if dim not in self.dimension_tags:
                raise ValueError(f"The dimension '{dim}' is not among the given dimension_tags.")
            # If the dimension tag is a tuple or list, split by all of them
            if isinstance(dim, (tuple, list)):
                group = self._csplit_by(dim, group)
            else:
                numeric_dict = {}
                places = self._places[dim]
                for key, filepaths in group.items():
                    # a group need not hold every tag
                    for path in sorted((p for p in filepaths if p in places), key=places.get):
                        num = '_'.join(str(number) for number in places[path])
                        tag_key = ''.join([key, '-', dim, num]) if key else ''.join([dim, num])
                        numeric_dict.setdefault(tag_key, []).append(path)
                group = numeric_dict
        return group

    def concatenate_along(self, axis: int) -> dict:
        """
        Concatenate arrays along the specified axis.

        Args:
            axis: The axis along which to concatenate the arrays.

        Returns:
            dict: The grouped file paths after concatenation.

        Raises:
            ValueError: If the axis is not among the given dimension tags.
        """
        dimension_tag = self.axis_tags[axis]
        if dimension_tag not in self.dimension_tags:
            raise ValueError(f"The dimension '{dimension_tag}' is not among the given dimension_tags.")

        # Split the group by all dimension tags except the one specified by the axis
        to_split = [tag for tag in self.dimension_tags if tag != dimension_tag]
        group = self._split_by(*to_split)
        
        tag_is_tuple = isinstance(dimension_tag, tuple)
        places = self._places[dimension_tag]

        for key, paths in group.items():
            # by the tag, not by the path: b_T1 comes before a_T2
            sorted_paths = sorted((path for path in paths if path in places),
                                  key=lambda path: self._rank(dimension_tag, path))
            if not sorted_paths:
                continue
            logger.debug(f"Sorted paths for concatenation: {sorted_paths}")
            # Get slices and shapes for each path
            group_slices = [self.slice_dict[path] for path in sorted_paths]
            group_shapes = [self.shape_dict[path] for path in sorted_paths]
            group_reduced_paths = [self.path_dict[path] for path in sorted_paths]
            # Calculate new slices and shape after concatenation
            new_slices = accumulate_slices_along_axis(group_shapes, axis, group_slices)

            new_shape = concatenate_shapes_along_axis(group_shapes, axis)

            # Update paths with concatenated version
            # the part tags are read in: the folders above stay as they are
            items = dimension_tag if tag_is_tuple else (dimension_tag,)
            folder = min((self.tag_folders[str(item)] for item in items), key=len)
            cut = len(folder) if all(path.startswith(folder) for path in group_reduced_paths) else 0
            replace_with = f'_{self.AXIS_DICT[axis]}set'
            first = group_reduced_paths[0]
            if tag_is_tuple:
                # the first file's name, its tag replaced by the names joined, in the order
                # given: what else varies between the files stays the first file's, as for a
                # numbered tag (0.3.3 glued every file's varying parts into one long name)
                names = list(dict.fromkeys(places[path] for path in sorted_paths))
                match = find_tag(first[cut:], names[0], False)
                if match is not None:
                    rest = first[cut:]
                    new_reduced_path = (first[:cut] + rest[:match.start()] + ''.join(names)
                                        + replace_with + rest[match.end():])
                else:
                    new_reduced_path = first[:cut] + reduce_paths_flexible(
                        [path[cut:] for path in group_reduced_paths], dimension_tag, replace_with)
            else:
                new_reduced_path = first[:cut] + reduce_paths_flexible(
                    [path[cut:] for path in group_reduced_paths], dimension_tag, replace_with)
            new_reduced_paths = [new_reduced_path] * len(group_reduced_paths)

            # If arrays are present, concatenate them
            if self.array_dict is not None:
                group_arrays = [self.array_dict[path] for path in sorted_paths]
                logger.debug(f"Arrays being concatenated in the order: {sorted_paths}")
                new_array = self.concatenate(group_arrays, axis=axis)

            # Update dictionaries with new values
            for path, slc, reduced_path in zip(sorted_paths,
                                               new_slices,
                                               new_reduced_paths
                                               ):
                self.slice_dict[path] = slc
                self.shape_dict[path] = new_shape
                self.path_dict[path] = reduced_path
                if self.array_dict is not None:
                    self.array_dict[path] = new_array
        #import pprint
        #pprint.pprint(self.path_dict)
        return group

    def get_concatenated_array_paths(self) -> Dict[str, str]:
        """Return ``{input_path: output_path}`` for each unique output."""
        return {k: p for k, (p, _) in self.get_concatenated_arrays().items()}

    def get_concatenated_arrays(self) -> Dict[str, tuple]:
        """Return ``{input_path: (output_path, array)}`` for each unique output."""
        unique_output_paths: list = []
        input_paths: list = []
        rep_keys: list = []

        for key in self.path_dict:
            output_path = self.path_dict[key]
            if output_path not in unique_output_paths:
                input_paths.append(key)
                unique_output_paths.append(output_path)
                rep_keys.append(key)

        if self.array_dict is not None:
            arrays = [self.array_dict[k] for k in rep_keys]
        else:
            arrays = [None] * len(rep_keys)
        return {
            k: (p, a)
            for k, p, a in zip(input_paths, unique_output_paths, arrays)
        }

    def get_concatenated_arrs(self) -> Dict[str, object]:
        """Return ``{input_path: array}`` — convenience wrapper around get_concatenated_arrays."""
        return {k: arr for k, (_, arr) in self.get_concatenated_arrays().items()}
