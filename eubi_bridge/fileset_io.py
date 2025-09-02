import copy
import re
import os
from typing import Dict, Iterable, List, Union

import dask.array as da
import numpy as np
from dask import delayed
from natsort import natsorted

from eubi_bridge.base.data_manager import ArrayManager, ChannelIterator

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


def reduce_paths_flexible(paths: Iterable[str],
                          dimension_tag: Union[str, tuple, list],
                          replace_with: str = 'set') -> str:
    """
    Reduces a list of similar paths by merging over the specified dimension.

    - If `dimension_tag` is a string (e.g., 'T' or 'Channel'), it's assumed to be followed by digits;
      the digits are replaced with `replace_with`.
    - If `dimension_tag` is a tuple/list (e.g., ('blue', 'red')), those are treated as categorical tokens
      and replaced with their joined value plus `replace_with`.
    """
    paths = list(paths)
    if not paths:
        return ""

    if isinstance(dimension_tag, str):
        # Match like 'T0001', 'Channel2', etc.
        pattern = re.compile(rf'({re.escape(dimension_tag)})(\d+)')
        def replace_tag(path):
            return pattern.sub(lambda m: m.group(1) + replace_with, path)

    elif isinstance(dimension_tag, (tuple, list)):
        # Categorical case: match only if surrounded by boundaries like /, _, -, ., or start/end of string
        unique_vals = sorted(set(dimension_tag))
        joined_val = ''.join(unique_vals) + replace_with

        # Example: match (^|/|_|-|.)blue(?=$|/|_|-|.)
        pattern = re.compile(
            rf'(?:(?<=^)|(?<=[/_\-.]))({"|".join(map(re.escape, unique_vals))})(?=$|[\/_\-.])'
        )

        def replace_tag(path):
            return pattern.sub(joined_val, path)

    else:
        raise ValueError("dimension_tag must be a string or a tuple/list of strings")

    # Apply replacement
    replaced_paths = [replace_tag(p) for p in paths]

    # Now combine all paths token-wise
    tokenized = [re.split(r'([/_\-.])', p) for p in replaced_paths]
    merged_tokens = []
    for tokens in zip(*tokenized):
        uniq = list(dict.fromkeys(tokens))  # preserve order
        merged_tokens.append(''.join(uniq))

    return ''.join(merged_tokens)


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

        self.shape_dict = dict(zip(filepaths, shapes))
        self.axis_tags = [axis_tag0, axis_tag1, axis_tag2, axis_tag3, axis_tag4]

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
        matches = get_matches(f'{dimension_tag}\d+', filepaths)
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
            alpha_dict = {key: [] for key in tup}
            for tag in tup:
                # Get matches for the current dimension tag
                matches = get_matches(f'{tag}', filepaths)
                # Extract the matched spans
                spans = [match.string[match.start():match.end()] for match in matches]
                # Extract the matched filepaths
                matched_paths = [match.string for match in matches]
                # Create a copy of the spans
                alpha = copy.deepcopy(spans)
                # Get the unique categories for the current dimension tag
                alpha_categories = np.unique(alpha).tolist()
                # Check that there is only one category
                assert len(alpha_categories) == 1, f"Number of categories is not 1: {alpha_categories}"
                # Get the category tag
                alpha_tag = alpha_categories[0]
                # Store the matched filepaths in the alpha dictionary
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
                    matches = get_matches(f'{dim}\d+', filepaths)
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

        for key, paths in group.items():
            # Sort paths naturally
            sorted_paths = natsorted(paths)

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
                new_array = da.concatenate(group_arrays, axis=axis)

            # Update dictionaries with new values
            for path, slc, reduced_path in zip(sorted_paths, new_slices, new_reduced_paths):
                self.slice_dict[path] = slc
                self.shape_dict[path] = new_shape
                self.path_dict[path] = reduced_path
                if self.array_dict is not None:
                    self.array_dict[path] = new_array

        return group

    def get_concatenated_array_paths(self):
        unique_paths = []
        unique_input_paths = []
        unique_ids = []

        # Process paths in natural sort order
        for key in natsorted(self.path_dict):
            path = self.path_dict[key]
            if path not in unique_paths:
                unique_input_paths.append(key)
                unique_paths.append(path)
                unique_ids.append(key)

        # Get arrays for unique paths
        # unique_arrays = [self.array_dict[path] for path in unique_ids]
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
        for key in natsorted(self.path_dict):
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
        for key in natsorted(self.path_dict):
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


# @delayed
# def build_managers_dict(**delayed_managers):
#     return delayed_managers


class BatchFile:
    def __init__(self,
                 filepaths: Iterable[str],
                 shapes: Iterable[tuple | list] = None,
                 axis_tag0: Union[str, tuple] = None,
                 axis_tag1: Union[str, tuple] = None,
                 axis_tag2: Union[str, tuple] = None,
                 axis_tag3: Union[str, tuple] = None,
                 axis_tag4: Union[str, tuple] = None,
                 arrays: Iterable[da.Array] = None,
                 ):

        self.fileset = FileSet(filepaths,
                          shapes=shapes,
                          axis_tag0=axis_tag0,
                          axis_tag1=axis_tag1,
                          axis_tag2=axis_tag2,
                          axis_tag3=axis_tag3,
                          axis_tag4=axis_tag4,
                          arrays=arrays)
        self.managers = None
        self.channel_managers = None


    def split_channel_groups(self):

        fileset = self.fileset
        sub_filesets = {}
        axis_tags = copy.deepcopy(fileset.axis_tags)

        if all([item is None for item in axis_tags]):
            groups = copy.deepcopy(fileset.path_dict)
            for key, value in groups.items():
                groups[key] = [value]
        elif fileset.axis_tags[1] is None:
            groups = copy.deepcopy(fileset.group)
        else:
            axis_tags[1] = None
            groups = fileset._split_by(fileset.axis_tags[1])
        return groups

    async def _construct_managers(self,
                          axes: Iterable[int] = [],
                          series: int = None,
                          metadata_reader: str = 'bfio',
                          **kwargs
                          ):
        for axis in axes:
            self.fileset.concatenate_along(axis)
        arrays_ = self.fileset.get_concatenated_arrays()
        print(f"arrays_: {arrays_}")
        self.sample_paths = natsorted(arrays_.keys())
        print(f"sample_paths: {self.sample_paths}")
        self.managers = {path: ArrayManager(path,
                                            series = series,
                                            metadata_reader = metadata_reader,
                                            **kwargs
                                            ) for
                         path in self.sample_paths
                         }

        # self.managers = build_managers_dict(**managers).compute()
        for name, manager in self.managers.items():
            await manager.init()
        return self.managers

    def _fuse_channels(self):
        channelsdict = {
            key: self.channel_managers[key].channels
            for
            key in natsorted(self.channel_managers.keys())
        }
        channelslist = []
        for key in natsorted(channelsdict.keys()):
            channelslist.extend(channelsdict[key])
        for path,manager in self.managers.items():
            manager._channels = channelslist
            self.managers[path] = manager

    async def _construct_channel_managers(self,
                      series: int = None,
                      metadata_reader: str = 'bfio',
                      **kwargs
                      ):
        grs = self.split_channel_groups()
        self.channel_sample_paths = natsorted([grs[grname][0]
                                        for grname in grs])
        self.channel_managers = {path: ArrayManager(path,
                                            series = series,
                                            metadata_reader = metadata_reader,
                                            **kwargs
                                            ) for
                         path in self.channel_sample_paths
                         }

        # self.channel_managers = build_managers_dict(**managers).compute()
        for name, manager in self.channel_managers.items():
            await manager.init()

        # channels_ = {}
        for path, manager in self.channel_managers.items():
            manager._ensure_correct_channels()
            manager.fix_bad_channels()
            # channels_[manager.path] = manager.channels
        return self.channel_managers


    async def _complete_process(self,
                          axes: Iterable[int] = [],
                          ):

        if self.managers is None:
            raise ValueError("Managers have not been constructed in advance.")
        if self.channel_managers is None:
            raise ValueError("Channel managers have not been constructed in advance.")
        # for axis in axes:
        #     self.fileset.concatenate_along(axis)
        if 1 in axes:
            self._fuse_channels()

        self.channels_per_output = {
            manager.path: manager.channels
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
            # Use os.path.splitdrive to handle any drive letter
            drive, _ = os.path.splitdrive(root_path)
            root_path = os.path.join(drive + os.sep, *root_path_top)
        else:
            root_path = os.path.join(os.sep, *root_path_top)

        # arrays_ = fileset.get_concatenated_arrays()  ### ! Careful here
        arraypaths = fileset.get_concatenated_array_paths()

        arrays, channels, sample_paths, managers = {}, {}, {}, {}

        for key, updated_key in arraypaths.items():
            # (updated_key, arr, ) = vals
            new_key = os.path.relpath(updated_key, root_path)
            new_key = os.path.splitext(new_key)[0]
            new_key = new_key.replace(os.sep, path_separator)
            # arrays[new_key] = arrays_[key][1]
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

        # arrays_ = fileset.get_concatenated_arrays()  ### ! Careful here

        sample_paths, managers = await self.get_output_paths_managers(root_path,
                                                                      path_separator
                                                                      )
        arraydicts = fileset.get_concatenated_arrs()

        arrays, channels, sample_paths, managers = {}, {}, {}, {}

        for key in arraydicts.keys():
            # (updated_key, arr, ) = vals
            arr = arraydicts[key]
            updated_key = sample_paths[key]
            new_key = os.path.relpath(updated_key, root_path)
            new_key = os.path.splitext(new_key)[0]
            new_key = new_key.replace(os.sep, path_separator)
            # arrays[new_key] = arrays_[key][1]
            arrays[new_key] = key

            ### Update colors if they are not unique
            # self.channels_per_output[key] = self._update_nonunique_channel_colors(self.channels_per_output[key])
            ###

            # channels[new_key] = self.channels_per_output[key]
            # managers[new_key] = self.managers[key]
            # managers[new_key]._channels = self.channels_per_output[key]

        return (arrays,
                # sample_paths,
                # managers
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
            new_key = os.path.splitext(new_key)[0]
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
