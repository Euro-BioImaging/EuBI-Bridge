import copy
import os
import re
import tempfile
from typing import Dict, Iterable, List, Union

import dask.array as da
import numpy as np
from dask import delayed
from natsort import natsorted

from eubi_bridge.core.data_manager import (ArrayManager,  # , prune_seriesfix
                                           ChannelIterator)
from eubi_bridge.utils.logging_config import get_logger
from eubi_bridge.external.dyna_zarr import operations as ops, DynamicArray
                        
logger = get_logger(__name__)

transpose_list = lambda l: list(map(list, zip(*l)))
get_numerics = lambda string: list(re.findall(r'\d+', string))
get_alpha = lambda string: ''.join([i for i in string if not i.isnumeric()])


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


def split_by_match(filepaths, *args):
    """
    Group filepaths based on matching patterns.

    Args:
        filepaths (list): List of file paths to search in.
        *args: Variable number of patterns to search for in filepaths.

    Returns:
        dict: Dictionary with patterns as keys and lists of matching filepaths as values.
    """
    ret = dict().fromkeys(args)
    for key in args:
        matches = get_matches(key, filepaths)
        ret[key] = matches
    return ret


def find_match_and_numeric(filepaths, *args):
    """
    Find matches in filepaths and group them by their numeric suffixes.

    Args:
        filepaths (list): List of file paths to search in.
        *args: Variable number of patterns to search for in filepaths.

    Returns:
        dict: Dictionary where keys are matched patterns and values are lists of match objects.
    """
    ret = {}
    for key in args:
        matches = get_matches(key, filepaths)
        for match in matches:
            span = match.string[match.start():match.end()]
            if span not in ret.keys():
                ret[span] = [match]
            else:
                ret[span].append(match)
    return ret


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


def find_consensus_sequence(strings: List[str]) -> tuple:
    """
    Find consensus sequence using column-based majority voting.
    
    Pads all strings to the same length with '_', then identifies
    which positions have differing characters across the strings.
    
    Args:
        strings: List of strings to find consensus from
        
    Returns:
        (consensus_str, variable_positions_list): Consensus string and list of 
        positions where characters differ
    """
    if not strings:
        return '', []
    
    max_len = max(len(s) for s in strings)
    padded_strings = [s.ljust(max_len, '_') for s in strings]
    
    consensus_chars = []
    variable_positions = []
    
    for pos in range(max_len):
        chars_at_pos = [s[pos] for s in padded_strings]
        
        # Find most common character (majority voting)
        char_counts = {}
        for ch in chars_at_pos:
            char_counts[ch] = char_counts.get(ch, 0) + 1
        
        most_common = max(char_counts.items(), key=lambda x: x[1])[0]
        consensus_chars.append(most_common)
        
        # Check if all characters at this position are the same
        if len(set(chars_at_pos)) > 1:
            variable_positions.append(pos)
    
    return ''.join(consensus_chars), variable_positions


def subtract_tags_from_filenames(filenames: List[str], tags: List[str]) -> Dict:
    """
    Subtract tags from filenames, returning before/after parts separately.
    
    Args:
        filenames: List of filenames (basenames, not full paths)
        tags: List of tags to find and remove from each filename
        
    Returns:
        Dict with keys:
            - 'before_parts': Strings before each tag
            - 'after_parts': Strings after each tag
            - 'tag_positions': Index where each tag starts
            - 'mappings': Tuples of (filename, tag, before, after, position)
    """
    before_parts = []
    after_parts = []
    tag_positions = []
    mappings = []
    
    for filename, tag in zip(filenames, tags):
        idx = filename.find(tag)
        if idx == -1:
            raise ValueError(f"Tag '{tag}' not found in filename '{filename}'")
        
        before = filename[:idx]
        after = filename[idx + len(tag):]
        
        before_parts.append(before)
        after_parts.append(after)
        tag_positions.append(idx)
        mappings.append((filename, tag, before, after, idx))
    
    return {
        'before_parts': before_parts,
        'after_parts': after_parts,
        'tag_positions': tag_positions,
        'mappings': mappings
    }


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
        # Numeric case: Match like 'T0001', 'Channel2', etc.
        pattern = re.compile(rf'({re.escape(dimension_tag)})(\d+)')
        def replace_tag(path):
            return pattern.sub(lambda m: m.group(1) + replace_with, path)
        
        replaced_paths = [replace_tag(p) for p in paths]
        return replaced_paths[0]

    elif isinstance(dimension_tag, (tuple, list)):
        # Categorical case: merge categorical dimension tags
        unique_vals = list(dimension_tag)
        
        # Split paths into directory and filename
        common_dir = os.path.dirname(paths[0])
        filenames = [os.path.basename(p) for p in paths]
        
        # Find which tag appears in each filename and extract before/after parts
        before_parts = []
        after_parts = []
        found_tags = []
        tag_indices = []
        
        for filename in filenames:
            found = False
            for tag in unique_vals:
                if tag in filename:
                    idx = filename.find(tag)
                    before = filename[:idx]
                    after = filename[idx + len(tag):]
                    
                    before_parts.append(before)
                    after_parts.append(after)
                    found_tags.append(tag)
                    tag_indices.append(idx)
                    found = True
                    break
            
            if not found:
                raise ValueError(f"No tags {unique_vals} found in {filename}")
        
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
        result_filename = common_before + merged_content + f'{replace_with}' + common_after_suffix
        
        return os.path.join(common_dir, result_filename)

    else:
        raise ValueError("dimension_tag must be a string or a tuple/list of strings")


def parse_channel_tag_from_string(channel_tag: str) -> tuple:
    """
    Parse a channel tag as a tuple of strings.

    Args:
        channel_tag (str): The channel tag to parse.

    Returns:
        tuple: The parsed channel tag as a tuple of strings.
    """
    if isinstance(channel_tag, str) and ',' in channel_tag:
        channel_tag = tuple(channel_tag.split(','))
    return channel_tag

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
                 arrays: Iterable[da.Array] = None):
        """
        Initialize the FileSet class.

        Args:
            filepaths: The file paths of the arrays.
            shapes: The shapes of the arrays. Required if arrays is not provided.
            axis_tag0-4: Tags for each axis (t, c, z, y, x).
            arrays: The arrays. If provided, shapes can be None.
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
        self.axis_tags = [axis_tag0,
                          parse_channel_tag_from_string(axis_tag1), # channel_tag
                          axis_tag2,
                          axis_tag3,
                          axis_tag4
                          ]

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

    def concatenate(self, arrays, axis: int):
        concatenate = ops.concatenate
        for arr in self.array_dict.values():
            if isinstance(arr, da.Array):
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
        matches = get_matches(rf'{dimension_tag}\d+', filepaths)
        spans = [match.string[match.start():match.end()] for match in matches]
        numerics = [get_numerics(span)[0] for span in spans]
        # TODO: add an incrementality validator
        return numerics

    def _csplit_by(self, tup: tuple) -> dict:
        """
        Split the filepaths in the group by the given dimension tags.

        Args:
            tup (tuple): A tuple of dimension tags to split by.

        Returns:
            dict: The split group as a dictionary.
        """
        group = copy.deepcopy(self.group)
        for key, filepaths in group.items():
            # Initialize a dictionary to store the split filepaths
            paths = [path for path in filepaths if
                     any([tag in path for tag in tup])]

            alpha_dict = {key: [] for key in tup}
            for tag in tup:
                # Get matches for the current dimension tag
                matches = get_matches(f'{tag}', paths)
                # Extract the matched spans
                spans = [match.string[match.start():match.end()] for match in matches]
                # Extract the matched paths
                matched_paths = [match.string for match in matches]
                # Create a copy of the spans
                alpha = copy.deepcopy(spans)
                # Get the unique categories for the current dimension tag
                alpha_categories = np.unique(alpha).tolist()
                # Check that there is only one category
                assert len(alpha_categories) == 1, f"Number of categories is not 1: {alpha_categories}"
                # Get the category tag
                alpha_tag = alpha_categories[0]
                # Store the matched paths in the alpha dictionary
                alpha_dict[alpha_tag] = matched_paths
            # Update the group with the split filepaths
            group = alpha_dict
        return group

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
                group = self._csplit_by(dim)
            else:
                numeric_dict = {}
                for key, filepaths in group.items():
                    matches = get_matches(rf'{dim}\d+', filepaths)
                    spans = [match.string[match.start():match.end()] for match in matches]
                    spans = [span.replace(dim, '') for span in spans]  ### remove search term from the spans
                    numerics = [get_numerics(span)[0] for span in spans]
                    numeric_categories = np.unique(numerics).tolist()
                    for idx, num in enumerate(numerics):
                        for i, category in enumerate(numeric_categories):
                            if num == category:
                                if key != '':
                                    tag_key = ''.join([key, '-', dim, num])
                                else:
                                    tag_key = ''.join([dim, num])
                                if not tag_key in numeric_dict:
                                    numeric_dict[tag_key] = []
                                numeric_dict[tag_key].append(filepaths[idx])
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
        
        import sys
        debug_log_path = os.path.join(tempfile.gettempdir(), "eubi_debug.log")
        with open(debug_log_path, "a") as f:
            f.write(f"\n[concatenate_along] axis={axis}, dimension_tag={dimension_tag}\n")
            f.write(f"[concatenate_along] to_split={to_split}\n")
            f.write(f"[concatenate_along] self.axis_tags={self.axis_tags}\n")
            f.write(f"[concatenate_along] Groups created by _split_by:\n")
            for k, v in group.items():
                f.write(f"  Group '{k}': {v}\n")

        tag_is_tuple = False
        if isinstance(dimension_tag, (tuple, list)):
            tag_is_tuple = True
            def get_index_from_filelist(query, filelist):
                for idx, path in enumerate(filelist):
                    if query in path:
                        return idx
                return None

        for key, paths in group.items():
            if tag_is_tuple:
                args = [get_index_from_filelist(item, paths) for item in dimension_tag]
                sorted_paths = [paths[idx] for idx in args]
            else:
                sorted_paths = paths
            logger.info(f"Sorted paths for concatenation: {sorted_paths}")
            debug_log_path = os.path.join(tempfile.gettempdir(), "eubi_debug.log")
            with open(debug_log_path, "a") as f:
                f.write(f"[concatenate_along] Group '{key}': sorted_paths={sorted_paths}\n")
            # Get slices and shapes for each path
            group_slices = [self.slice_dict[path] for path in sorted_paths]
            group_shapes = [self.shape_dict[path] for path in sorted_paths]
            group_reduced_paths = [self.path_dict[path] for path in sorted_paths]
            # Calculate new slices and shape after concatenation
            new_slices = accumulate_slices_along_axis(group_shapes, axis, group_slices)

            new_shape = concatenate_shapes_along_axis(group_shapes, axis)

            # Update paths with concatenated version
            new_reduced_path = reduce_paths_flexible(
                group_reduced_paths,
                dimension_tag,
                replace_with=f'_{self.AXIS_DICT[axis]}set'
            )
            #print(f"Group reduced paths: {group_reduced_paths}")
            #print(f"Dimension tag: {dimension_tag}")
            #print(f"New reduced path: {new_reduced_path}")
            #print(f"Replacing with: _{self.AXIS_DICT[axis]}set")
            new_reduced_paths = [new_reduced_path] * len(group_reduced_paths)

            # If arrays are present, concatenate them
            if self.array_dict is not None:
                group_arrays = [self.array_dict[path] for path in sorted_paths]
                logger.info(f"Arrays being concatenated in the order: {sorted_paths}")
                debug_log_path = os.path.join(tempfile.gettempdir(), "eubi_debug.log")
                with open(debug_log_path, "a") as f:
                    f.write(f"  Concatenating arrays with shapes: {[arr.shape for arr in group_arrays]}\n")
                new_array = self.concatenate(group_arrays, axis=axis)
                debug_log_path = os.path.join(tempfile.gettempdir(), "eubi_debug.log")
                with open(debug_log_path, "a") as f:
                    f.write(f"  Result array shape: {new_array.shape}\n")

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

    def get_concatenated_array_paths(self):
        unique_paths = []
        unique_input_paths = []
        unique_ids = []

        # Process paths in natural sort order
        for key in self.path_dict:
            path = self.path_dict[key]
            if path not in unique_paths:
                unique_input_paths.append(key)
                unique_paths.append(path)
                unique_ids.append(key)

        # Get arrays for unique paths
        return {
            key: path
            for key, path in zip(unique_input_paths, unique_paths)
        }

    def get_concatenated_arrs(self) -> Dict[str, tuple]:
        """
        Get a dictionary of concatenated arrays with their metadata.

        Returns:
            dict: A dictionary where keys are input paths and values are tuples of
                (updated_path, array_data).
        """
        # Get unique paths and their corresponding keys
        unique_paths = []
        unique_input_paths = []
        unique_ids = []

        # Process paths in natural sort order
        for key in self.path_dict:
            path = self.path_dict[key]
            if path not in unique_paths:
                unique_input_paths.append(key)
                unique_paths.append(path)
                unique_ids.append(key)

        # Get arrays for unique paths
        unique_arrays = [self.array_dict[path] for path in unique_ids]

        # Create result dictionary
        return {
            key: arr
            for key, arr in zip(unique_input_paths, unique_arrays)
        }

    def get_concatenated_arrays(self) -> Dict[str, tuple]:
        """
        Get a dictionary of concatenated arrays with their metadata.

        Returns:
            dict: A dictionary where keys are input paths and values are tuples of
                (updated_path, array_data).
        """
        # Get unique paths and their corresponding keys
        unique_paths = []
        unique_input_paths = []
        unique_ids = []

        # Process paths in natural sort order
        for key in self.path_dict:
            path = self.path_dict[key]
            if path not in unique_paths:
                unique_input_paths.append(key)
                unique_paths.append(path)
                unique_ids.append(key)

        # Get arrays for unique paths
        unique_arrays = [self.array_dict[path] for path in unique_ids]

        # Create result dictionary
        return {
            key: (path, arr)
            for key, path, arr in zip(unique_input_paths, unique_paths, unique_arrays)
        }


def prune_seriesfix(path):
    end = path[-1]
    while end.isnumeric():
        path = path[:-1]
        end = path[-1]
    if end == '_':
        path = path[:-1]
    return path
class BatchFile:
    def __init__(self,
                 filepaths: Iterable[str],
                 shapes: Iterable[tuple | list] = None,
                 axis_tag0: Union[str, tuple] = None,
                 axis_tag1: Union[str, tuple] = None,
                 axis_tag2: Union[str, tuple] = None,
                 axis_tag3: Union[str, tuple] = None,
                 axis_tag4: Union[str, tuple] = None,
                 arrays: Dict[str, da.Array] = None,
                 ):

        ### Handle categorical labels. Prefilter files and arrays for the presence of categorical labels
        if shapes is None:
            shapes = [None] * len(filepaths)
            shapedict = dict(zip(filepaths, shapes))
        if arrays is None:
            arrs = [None] * len(filepaths)
            arraydict = dict(zip(filepaths, arrs))
        else:
            assert isinstance(arrays, dict)
            arraydict = arrays

        arrs_sel = []
        shapes_sel = []
        paths = []

        paths = filepaths
        filepaths = natsorted(paths)
        arrs_sel = [arraydict[path] for path in filepaths]
        shapes_sel = [shapedict[path] for path in filepaths]
        self.arrays = dict(zip(filepaths, arrs_sel))
        ###

        self.fileset = FileSet(filepaths,
                          shapes=shapes_sel,
                          axis_tag0=axis_tag0,
                          axis_tag1=axis_tag1,
                          axis_tag2=axis_tag2,
                          axis_tag3=axis_tag3,
                          axis_tag4=axis_tag4,
                          arrays=arrs_sel
                        )
        self.managers = None
        self.channel_managers = None

    def split_channel_groups(self):

        fileset = self.fileset
        sub_filesets = {}
        axis_tags = copy.deepcopy(fileset.axis_tags)

        debug_msg = f"\n[split_channel_groups] fileset.axis_tags = {fileset.axis_tags}\n"
        debug_msg += f"[split_channel_groups] fileset.path_dict keys = {list(fileset.path_dict.keys())}\n"
        debug_msg += f"[split_channel_groups] fileset.group keys = {list(fileset.group.keys())}\n"
        
        if all([item is None for item in axis_tags]):
            debug_msg += f"[split_channel_groups] All axis_tags are None - using path_dict\n"
            groups = copy.deepcopy(fileset.path_dict)
            for key, value in groups.items():
                groups[key] = [value]
        elif fileset.axis_tags[1] is None:
            debug_msg += f"[split_channel_groups] axis_tags[1] is None - using group\n"
            groups = copy.deepcopy(fileset.group)
        else:
            debug_msg += f"[split_channel_groups] Splitting by channel tag: {fileset.axis_tags[1]}\n"
            axis_tags[1] = None
            groups = fileset._split_by(fileset.axis_tags[1])
            debug_msg += f"[split_channel_groups] Groups after _split_by: {list(groups.keys())}\n"
            for k, v in groups.items():
                debug_msg += f"  {k}: {v}\n"
        
        # Write debug output to file
        debug_log_path = os.path.join(tempfile.gettempdir(), "eubi_debug.log")
        with open(debug_log_path, "a") as f:
            f.write(debug_msg)
        
        return groups

    async def _construct_managers(self,
                          axes: Iterable[int] = [],
                          series: int = None,
                          metadata_reader: str = 'bfio',
                          **kwargs
                          ):
        if series is None:
            series = 0

        for axis in axes:
            self.fileset.concatenate_along(axis)
        arrays_ = self.fileset.get_concatenated_arrays()
        self.sample_paths = list(arrays_.keys())
        pruned_paths = list(set(prune_seriesfix(path) for path in self.sample_paths))

        managers = [ArrayManager(path,
                                 series = 0,
                                 metadata_reader = metadata_reader,
                                 **kwargs) for path in pruned_paths]
        self.managers = {} # Speed up this part
        for manager in managers:
            logger.info(f"Reference manager: {manager.path} constructed.")
            await manager.init()
            await manager.load_scenes(series)
            self.managers.update(**manager.loaded_scenes)
        return self.managers

    def _fuse_channels(self): # Concatenates channel metadata actually.
        """Merge channel metadata from all channel files into each output manager.
        
        When concatenating along the channel axis, all input channels should be
        merged into a single channel list, and this list should be assigned to
        the output managers (not the individual channel file managers).
        """
        debug_msg = f"\n[_fuse_channels] START\n"
        debug_msg += f"[_fuse_channels] self.channel_sample_paths = {self.channel_sample_paths}\n"
        debug_msg += f"[_fuse_channels] Number of channel sample paths = {len(self.channel_sample_paths)}\n"
        debug_msg += f"[_fuse_channels] self.channel_managers.keys() = {list(self.channel_managers.keys())}\n"
        debug_msg += f"[_fuse_channels] Number of channel managers = {len(self.channel_managers)}\n"
        debug_msg += f"[_fuse_channels] self.managers.keys() = {list(self.managers.keys())}\n"
        debug_msg += f"[_fuse_channels] Number of output managers = {len(self.managers)}\n"
        
        channelsdict = {
            key: self.channel_managers[key].channels
            for key in self.channel_sample_paths # List that keeps the channel order
        }
        
        debug_msg += f"[_fuse_channels] channelsdict keys = {list(channelsdict.keys())}\n"
        for key, channels in channelsdict.items():
            debug_msg += f"  {key}: {len(channels)} channels\n"
            for i, ch in enumerate(channels):
                debug_msg += f"    Channel {i}: label={ch.get('label', '?')}, color={ch.get('color', '?')}\n"
        
        channelslist = []
        for key in self.channel_sample_paths: # List that keeps the channel order
            # This is where channel metadata are properly sorted.
            channelslist.extend(channelsdict[key])
        
        debug_msg += f"[_fuse_channels] Total merged channels = {len(channelslist)}\n"
        debug_msg += f"[_fuse_channels] Merged channel labels = {[c.get('label', '?') for c in channelslist]}\n"
        
        # Update all output managers with the merged channel list
        # NOT the channel_managers (which are individual files)
        debug_msg += f"[_fuse_channels] Updating {len(self.managers)} output managers with merged channels\n"
        for manager in self.managers.values():
            debug_msg += f"  Manager {manager.series_path}: before={len(manager.channels)} channels, after={len(channelslist)} channels\n"
            manager._channels = channelslist
        
        debug_msg += f"[_fuse_channels] END\n"
        debug_log_path = os.path.join(tempfile.gettempdir(), "eubi_debug.log")
        with open(debug_log_path, "a") as f:
            f.write(debug_msg)

    async def _construct_channel_managers(self,
                      series: int = None,
                      metadata_reader: str = 'bfio',
                      **kwargs
                      ):
        debug_msg = f"\n[_construct_channel_managers] START\n"
        if series is None:
            series = 0
        if np.isscalar(series) and (series != 'all'):
            series = [series]

        grs = self.split_channel_groups()
        debug_msg += f"[_construct_channel_managers] Channel groups from split_channel_groups: {list(grs.keys())}\n"
        debug_msg += f"[_construct_channel_managers] Number of groups: {len(grs)}\n"
        for k, v in grs.items():
            debug_msg += f"  Group '{k}': {v}\n"

        self.channel_sample_paths = [grs[grname][0]
                                        for grname in grs
                                     ]
        debug_msg += f"[_construct_channel_managers] Channel sample paths: {self.channel_sample_paths}\n"
        debug_msg += f"[_construct_channel_managers] Number of channel sample paths: {len(self.channel_sample_paths)}\n"

        channel_managers = {}
        unloaded_paths = []
        for path in self.channel_sample_paths: # List that keeps the channel order
            if path in self.managers: # If loaded already, do not reload!
                channel_managers[path] = self.managers[path]
            else:
                unloaded_paths.append(path)

        debug_msg += f"[_construct_channel_managers] Paths already in managers: {len(self.channel_sample_paths) - len(unloaded_paths)}\n"
        debug_msg += f"[_construct_channel_managers] Paths to load: {len(unloaded_paths)}\n"
        if unloaded_paths:
            debug_msg += f"  Unloaded paths: {unloaded_paths}\n"

        debug_log_path = os.path.join(tempfile.gettempdir(), "eubi_debug.log")
        with open(debug_log_path, "a") as f:
            f.write(debug_msg)

        pruned_paths = list(set(prune_seriesfix(path) for path in unloaded_paths))
        managers = [ArrayManager(path,
                                 series = 0,
                                 metadata_reader = metadata_reader,
                                 **kwargs) for path in pruned_paths
                    ]
        for manager in managers:
            await manager.init()
            await manager.load_scenes(series)
            channel_managers.update(**manager.loaded_scenes)

        self.channel_managers = {key: channel_managers[key] for key in self.channel_sample_paths}
        
        debug_msg2 = f"[_construct_channel_managers] Final channel_managers.keys() = {list(self.channel_managers.keys())}\n"
        debug_msg2 += f"[_construct_channel_managers] Final number of channel managers = {len(self.channel_managers)}\n"
        
        for path in self.channel_sample_paths:
            manager = self.channel_managers[path]
            debug_msg2 += f"[_construct_channel_managers] Before fix_bad_channels - {path}: {len(manager.channels)} channels\n"
            for i, ch in enumerate(manager.channels):
                debug_msg2 += f"    Channel {i}: label={ch.get('label', '?')}, color={ch.get('color', '?')}\n"
            manager._ensure_correct_channels()
            manager.fix_bad_channels()
            debug_msg2 += f"[_construct_channel_managers] After fix_bad_channels - {path}: {len(manager.channels)} channels\n"
            for i, ch in enumerate(manager.channels):
                debug_msg2 += f"    Channel {i}: label={ch.get('label', '?')}, color={ch.get('color', '?')}\n"
        
        debug_msg2 += f"[_construct_channel_managers] END\n"
        debug_log_path = os.path.join(tempfile.gettempdir(), "eubi_debug.log")
        with open(debug_log_path, "a") as f:
            f.write(debug_msg2)
        return self.channel_managers

    async def _complete_process(self,
                          axes: Iterable[int] = [],
                          ):

        if self.managers is None:
            raise ValueError("Managers have not been constructed in advance.")
        if self.channel_managers is None:
            raise ValueError("Channel managers have not been constructed in advance.")

        if 1 in axes:
            self._fuse_channels()

        self.channels_per_output = {
            manager.series_path: manager.channels
            for
            manager in self.managers.values()
        }

    def _update_nonunique_channel_colors(self,
                                         channels
                                         ):
        colors = [channel['color'] for channel in channels]
        if len(set(colors)) < len(colors):
            chn = ChannelIterator(num_channels=len(colors))
            for channel, _channel in zip(channels, chn._channels):
                channel['color'] = _channel['color']
        return channels

    async def get_output_paths_managers(self,
                         root_path,
                         path_separator: str = '-',
                         ):
        # TODO: split the path finding and the array costruction.
        # Make the path finding async, then.
        fileset = self.fileset
        root_path_ = os.path.normpath(root_path).split(os.sep)
        root_path_top = []
        for item in root_path_:
            if '*' in item:
                break
            root_path_top.append(item)

        if os.name == 'nt':
            drive, _ = os.path.splitdrive(root_path)
            root_path = os.path.join(drive + os.sep, *root_path_top)
        else:
            root_path = os.path.join(os.sep, *root_path_top)

        arraypaths = fileset.get_concatenated_array_paths()

        arrays, channels, sample_paths, managers = {}, {}, {}, {}

        for key, updated_key in arraypaths.items():
            new_key = os.path.relpath(updated_key, root_path)
            new_key = os.path.splitext(new_key)[0]
            new_key = new_key.replace(os.sep, path_separator)
            sample_paths[new_key] = key

            ### Update colors if they are not unique
            self.channels_per_output[key] = self._update_nonunique_channel_colors(self.channels_per_output[key])
            ###

            channels[new_key] = self.channels_per_output[key]
            managers[new_key] = self.managers[key]
            managers[new_key]._channels = self.channels_per_output[key]

        return (#arrays,
                sample_paths,
                managers
                )

    async def get_output_arraydicts(self,
                         root_path,
                         path_separator: str = '-',
                         ):
        # TODO: split the path finding and the array costruction.
        # Make the path finding async, then.
        fileset = self.fileset
        root_path_ = os.path.normpath(root_path).split(os.sep)
        root_path_top = []
        for item in root_path_:
            if '*' in item:
                break
            root_path_top.append(item)

        if os.name == 'nt':
            # Use os.path.splitdrive to handle any drive letter
            drive, _ = os.path.splitdrive(root_path)
            root_path = os.path.join(drive + os.sep, *root_path_top)
        else:
            root_path = os.path.join(os.sep, *root_path_top)

        sample_paths, managers = await self.get_output_paths_managers(root_path,
                                                                      path_separator
                                                                      )
        arraydicts = fileset.get_concatenated_arrs()

        arrays, channels, sample_paths, managers = {}, {}, {}, {}

        for key in arraydicts.keys():
            arr = arraydicts[key]
            updated_key = sample_paths[key]
            new_key = os.path.relpath(updated_key, root_path)
            new_key = os.path.splitext(new_key)[0]
            new_key = new_key.replace(os.sep, path_separator)
            arrays[new_key] = key

            ### Update colors if they are not unique
            # self.channels_per_output[key] = self._update_nonunique_channel_colors(self.channels_per_output[key])
            ###

        return (arrays,
                )

    async def get_output_dicts(self,
                         root_path,
                         path_separator: str = '-',
                         ):
        # TODO: split the path finding and the array costruction.
        # Make the path finding async, then.
        fileset = self.fileset
        root_path_ = os.path.normpath(root_path).split(os.sep)
        root_path_top = []
        for item in root_path_:
            if '*' in item:
                break
            root_path_top.append(item)

        if os.name == 'nt':
            # Use os.path.splitdrive to handle any drive letter
            drive, _ = os.path.splitdrive(root_path)
            root_path = os.path.join(drive + os.sep, *root_path_top)
        else:
            root_path = os.path.join(os.sep, *root_path_top)

        arrays_ = fileset.get_concatenated_arrays()  ### ! Careful here

        arrays, channels, sample_paths, managers = {}, {}, {}, {}

        for key, vals in arrays_.items():
            (updated_key, arr, ) = vals
            new_key = os.path.relpath(updated_key, root_path)
            new_key = new_key.replace(os.sep, path_separator)
            arrays[new_key] = arrays_[key][1]
            sample_paths[new_key] = key

            ### Update colors if they are not unique
            self.channels_per_output[key] = self._update_nonunique_channel_colors(self.channels_per_output[key])
            ###

            channels[new_key] = self.channels_per_output[key]
            managers[new_key] = self.managers[key]
            managers[new_key]._channels = self.channels_per_output[key]

        return (arrays,
                sample_paths,
                managers)
