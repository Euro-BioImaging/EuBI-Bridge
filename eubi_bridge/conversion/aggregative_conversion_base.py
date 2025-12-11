# Standard library imports
import os
from pathlib import Path
from typing import Union

# Third-party imports
import dask
import numpy as np

# Local application imports
from eubi_bridge.core.data_manager import BatchManager
from eubi_bridge.core.readers import read_single_image, read_single_image_delayed
from eubi_bridge.conversion.fileset_io import BatchFile
from eubi_bridge.ngff.multiscales import Pyramid
from eubi_bridge.ngff.defaults import unit_map, scale_map, default_axes
from eubi_bridge.utils.convenience import (
    take_filepaths
)

# Configure logging
# logging.getLogger('distributed.diskutils').setLevel(logging.CRITICAL)
# warnings.filterwarnings('ignore')



def prune_seriesfix(path):
    end = path[-1]
    while end.isnumeric():
        path = path[:-1]
        end = path[-1]
    if end == '_':
        path = path[:-1]
    return path

class AggregativeConverter:
    def __init__(self,
                 input_path: Union[str, Path] = None,  # TODO: add csv option (or a general table option).
                 includes=None,
                 excludes=None,
                 metadata_path=None,
                 series=None,
                 zarr_format=2,
                 verbose=False,
                 override_channel_names = False
                 ):
        """
        Initialize the BridgeBase class. This class is the main entry point for
        converting and processing image data.

        Args:
            input_path (Union[str, Path]): Path to the input file or directory.
            includes (optional): Patterns of filenames to include.
            excludes (optional): Patterns of filenames to exclude.
            metadata_path (optional): Path to metadata file if any.
            series (optional): Series index or name to process.
            client (optional): Dask client for parallel processing.
            zarr_format (int, optional): Zarr format version. Defaults to 2.
            verbose (bool, optional): Enable verbose logging. Defaults to False.
        """

        # Ensure the input path is absolute
        if input_path is not None:
            if not input_path.startswith(os.sep):
                input_path = os.path.abspath(input_path)

        # Initialize instance variables
        self._input_path = input_path
        self._includes = includes
        self._excludes = excludes
        self._metadata_path = metadata_path
        self._series = series
        self._dask_temp_dir = None
        self._zarr_format = zarr_format
        self._verbose = verbose
        self._cluster_params = None
        self.fileset = None
        self.pixel_metadata = None
        self.fileset = None
        self._override_channel_names = override_channel_names

        # Validate the series parameter
        # if self._series is not None:
        #     assert isinstance(self._series, (int, str)), (
        #         "The series parameter must be either an integer or string. "
        #         "Selection of multiple series from the same image is currently not supported."
        #     )

    async def read_dataset(self,
                     chunks_yx = None,
                     readers_params = {},
                     ):
        """
        - If the input path is a directory, can read single or multiple files from it.
        - If the input path is a file, can read a single image from it.
        - If the input path is a file with multiple series, can currently only read one series from it. Reading multiple series is currently not supported.
        - If the input path is a csv file with filepaths and conversion parameters, can read the filepaths and conversion parameters from it.
        :return:
        """
        input_path = self._input_path # todo: make them settable from this method?
        includes = self._includes
        excludes = self._excludes
        metadata_path = self._metadata_path
        series = self._series
        zarr_format = self._zarr_format
        verbose = self._verbose

        _input_is_csv = False
        _input_is_tiff = False
        if input_path is not None and input_path.endswith('.csv'):
            _input_is_csv = True
            df = take_filepaths(input_path,
                                            includes = includes,
                                            excludes = excludes)
            self.filepaths = df.input_path.tolist()
        if input_path is not None and (os.path.isfile(input_path) or
                                       input_path.endswith('.zarr')):
            dirname = os.path.dirname(input_path)
            basename = os.path.basename(input_path)
            input_path = f"{dirname}/*{basename}"
            self._input_path = input_path

        if input_path is not None and not _input_is_csv:
            df = take_filepaths(input_path,
                                            includes = includes,
                                            excludes = excludes)
            self.filepaths = df.input_path.tolist()
        try:
            readers_params.pop('scene_index')
        except:
            pass

        futures = [ ### This must change. Read all series from within the function.
                    ### Function must return a list of series.
                    ### Then flatten the series here.
                    read_single_image_delayed(
                                      path,
                                      chunks_yx=chunks_yx,
                                      # verified_for_cluster=verified_for_cluster,
                                      zarr_format = zarr_format,
                                      verbose = verbose,
                                      scene_index = 0,
                                      **readers_params
                                      )
                    for path in self.filepaths
        ]

        self.imgs = dask.compute(*futures)

        self.arrays = {
            img.series_path: img.get_image_dask_data() for img in self.imgs
        }
        self.arrays = {}
        self.series_filepaths = []
        for img in self.imgs:
            if series == 'all':
                series_ = list(range(img.n_scenes))
            elif np.isscalar(series):
                series_ = [series]
            else:
                series_ = series
            for s in series_:
                img.set_scene(s)
                self.series_filepaths.append(img.series_path)
                self.arrays[img.series_path] = img.get_image_dask_data()

        if metadata_path is None:
            self.metadata_path = self.filepaths[0]
        else:
            self.metadata_path = metadata_path

    async def digest(
            self,
            time_tag: Union[str, tuple] = None,
            channel_tag: Union[str, tuple] = None,
            z_tag: Union[str, tuple] = None,
            y_tag: Union[str, tuple] = None,
            x_tag: Union[str, tuple] = None,
            axes_of_concatenation: Union[int, tuple, str] = None,
            metadata_reader: str = 'bfio',
            skip_dask: bool = True,
            **kwargs
    ):
        """
        Async digest with optional skip_dask.
        """
        axes = 'tczyx'
        tags = (time_tag, channel_tag, z_tag, y_tag, x_tag)

        self.channel_tag = channel_tag # asil burada parse et.
        self._channel_tag_is_tuple = False
        if isinstance(self.channel_tag, str) and ',' in self.channel_tag:
            self.channel_tag = self.channel_tag.split(',')
        if isinstance(self.channel_tag, (tuple, list)):
            self._channel_tag_is_tuple = True

        self.batchfile = BatchFile(
            self.series_filepaths,
            arrays=self.arrays,
            axis_tag0=time_tag,
            axis_tag1=channel_tag,
            axis_tag2=z_tag,
            axis_tag3=y_tag,
            axis_tag4=x_tag,
        )

        axdict = dict(zip(axes, tags))
        if axes_of_concatenation is None:
            axes_of_concatenation = []
        axlist = [axes.index(x)
                  for x in axes_of_concatenation
                  if x in axes
                  ]
        await self.batchfile._construct_managers(
            axes=axlist,
            series=self._series,
            metadata_reader=metadata_reader,
            **kwargs
        )

        await self.batchfile._construct_channel_managers(
            series=self._series,
            metadata_reader=metadata_reader,
            **kwargs
        )
        await self.batchfile._complete_process(axlist)

        output_path = self._input_path or kwargs.get('output_path')

        (self.digested_arrays,
         self.digested_arrays_sample_paths,
         self.managers
         ) = await self.batchfile.get_output_dicts(output_path)

        await self._compute_pixel_metadata(**kwargs)
        return self


    async def _compute_pixel_metadata(self,
                               **kwargs
                               ):
        """Compute and update pixel metadata for the digested arrays.

        Args:
            series: Series identifier
            metadata_reader: Reader to use for metadata (default: 'bfio')
            **kwargs: Additional metadata including units and scales
        """
        assert self.digested_arrays is not None
        assert self.digested_arrays_sample_paths is not None
        assert self.managers is not None

        unit_mapping = {
            'time_unit': 't', 'channel_unit': 'c',
            'z_unit': 'z', 'y_unit': 'y', 'x_unit': 'x'
        }
        scale_mapping = {
            'time_scale': 't', 'channel_scale': 'c',
            'z_scale': 'z', 'y_scale': 'y', 'x_scale': 'x'
        }

        # Process unit and scale updates
        update_unitdict = {unit_mapping[k]: v for k, v in kwargs.items() if k in unit_mapping}
        update_scaledict = {scale_mapping[k]: v for k, v in kwargs.items() if k in scale_mapping}

        # Update arrays and metadata
        for name in self.digested_arrays.keys():
            arr = self.digested_arrays[name]
            path = self.digested_arrays_sample_paths[name]
            manager = self.managers[name]
            manager.set_arraydata(arr)
            manager.update_meta(
                new_unitdict=update_unitdict,
                new_scaledict=update_scaledict
            )

        if self._channel_tag_is_tuple and self._override_channel_names:
            # Do this also for non-tuple channel tags
            for manager in self.managers.values():
                channels = manager.channels
                for idx, (channel, tagitem) in enumerate(zip(channels,
                                                           self.channel_tag
                                                           )
                                                       ):
                    channel['label'] = tagitem
                    channels[idx] = channel
                manager._channels = channels
        self.batchdata = BatchManager()
        await self.batchdata.init(self.managers)
        await self.batchdata.fill_default_meta()

    def squeeze_dataset(self):
        self.batchdata.squeeze()

    async def transpose_dataset(self,
                          dimension_order=Union[str, tuple, list]
                          ):
        """
        Transpose the dataset according to the given dimension order.

        Parameters
        ----------
        dimension_order : Union[str, tuple, list]
            The order of the dimensions in the transposed array.

        """
        self.batchdata.transpose(newaxes = dimension_order)

    async def crop_dataset(self, **kwargs):
        self.batchdata.crop(**kwargs)

    async def to_cupy(self):
        self.batchdata.to_cupy()


    def _prepare_array_metadata(self,
                                batch_manager,
                                sample_path_mapping,
                                autochunk = True
                                ):
        """Prepare metadata dictionaries for array storage.

        Args:
            batch_manager: BatchManager instance containing array data
            sample_path_mapping: Dictionary mapping array names to their file paths

        Returns:
            Tuple containing dictionaries for arrays, scales, axes, units, and chunks
        """
        array_data = {}
        dimension_scales = {}
        dimension_axes = {}
        dimension_units = {}
        chunk_configs = {}
        channel_meta = {} # a dict of lists, each list being the length of channels per image

        for array_name, file_path in sample_path_mapping.items():
            manager = batch_manager.managers[array_name]
            array_data[array_name] = {'0': manager.array}
            dimension_scales[array_name] = {'0': manager.scales}
            dimension_axes[array_name] = {'0': manager.axes}
            dimension_units[array_name] = {'0': manager.units}
            chunk_configs[array_name] = {'0': manager.chunks}
            channel_meta[array_name] = {'0': manager.channels}

        return array_data, dimension_scales, dimension_axes, dimension_units, chunk_configs, channel_meta


    def _create_output_path_mapping(self, output_path,
                                    nested_data, sample_paths):
        """Create flattened path mappings for the given data dictionary.

        Args:
            output_path: Base output directory
            nested_data: Nested dictionary of data to be flattened
            sample_paths: Dictionary mapping array names to their file paths

        Returns:
            Dictionary with output file paths as keys
        """
        return {
            os.path.join(
                output_path,
                f"{array_name}.zarr" if not array_name.endswith('zarr') else array_name,
                str(level)
            ): value
            for array_name, subdict in nested_data.items()
            for level, value in subdict.items()
        }


    def _process_chunking_configurations(self,
                                         chunk_sizes,
                                         shard_coefficients,
                                         axis_mappings,
                                         chunk_mappings):
        """Process chunk and shard configurations for each array.

        Args:
            chunk_sizes: Dictionary of chunk sizes per dimension
            shard_coefficients: Dictionary of shard coefficients per dimension
            axis_mappings: Dictionary mapping output paths to their dimension axes
            chunk_mappings: Dictionary mapping output paths to their chunk configurations

        Returns:
            Tuple of (updated_chunk_sizes, updated_shard_coefficients)
        """
        processed_chunk_sizes = {}
        processed_shard_coeffs = {}

        for output_path, chunk_config in chunk_mappings.items():
            axes = axis_mappings[output_path]
            final_chunk_sizes = []
            final_shard_coeffs = []

            for axis in axes:
                chunk_size = chunk_sizes[axis] or chunk_config[axes.index(axis)]
                final_chunk_sizes.append(chunk_size)
                final_shard_coeffs.append(shard_coefficients[axis])

            processed_chunk_sizes[output_path] = final_chunk_sizes
            processed_shard_coeffs[output_path] = final_shard_coeffs

        return processed_chunk_sizes, processed_shard_coeffs
