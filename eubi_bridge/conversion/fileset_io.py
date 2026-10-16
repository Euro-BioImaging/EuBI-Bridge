from __future__ import annotations

from typing import TYPE_CHECKING

import copy
import os
import re
from typing import Dict, Iterable, List, Union

import numpy as np

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

        # Match the normalised tag, optionally followed by a path separator, then digits.
        # This covers both inline patterns (T0001) and directory-component patterns (frames\000).
        # group(1) captures only the stripped tag; the optional separator is consumed but not
        # captured, so the replacement never re-introduces a dangling separator before replace_with.
        pattern = re.compile(rf'({re.escape(tag_stripped)})[/\\]?(\d+)')

        def replace_tag(path):
            return pattern.sub(lambda m: m.group(1) + replace_with, path)

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
        channel_tag = tuple(channel_tag.split(','))
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
        matches = get_matches(dimension_tag + r'[/\\]?\d+', filepaths)
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
                    matches = get_matches(dim + r'[/\\]?\d+', filepaths)
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
            new_reduced_paths = [new_reduced_path] * len(group_reduced_paths)

            # If arrays are present, concatenate them
            if self.array_dict is not None:
                group_arrays = [self.array_dict[path] for path in sorted_paths]
                logger.info(f"Arrays being concatenated in the order: {sorted_paths}")
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
