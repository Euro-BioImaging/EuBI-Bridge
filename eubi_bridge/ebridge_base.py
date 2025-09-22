# Standard library imports
import copy
import logging
import os
import tempfile
import warnings
from pathlib import Path
from typing import Tuple, Union

# Third-party imports
import dask
import dask.array as da
import numpy as np

# Local application imports
from eubi_bridge.base.data_manager import BatchManager
from eubi_bridge.base.readers import read_single_image_asarray
from eubi_bridge.base.writers import store_arrays
# from eubi_bridge.fileset_io import FileSet
from eubi_bridge.fileset_io import BatchFile
from eubi_bridge.ngff.multiscales import Pyramid
from eubi_bridge.ngff.defaults import unit_map, scale_map, default_axes
from eubi_bridge.utils.convenience import (
    take_filepaths
)

from eubi_bridge.utils.logging_config import get_logger

# Set up logger for this module
logger = get_logger(__name__)

# Configure logging
logging.getLogger('distributed.diskutils').setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')


class BridgeBase:
    def __init__(self,
                 input_path: Union[str, Path],  # TODO: add csv option (or a general table option).
                 includes=None,
                 excludes=None,
                 metadata_path=None,
                 series=None,
                 client=None,
                 zarr_format=2,
                 verbose=False
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
        self.vmeta = None
        self._cluster_params = None
        self.client = client
        self.fileset = None
        self.pixel_metadata = None
        self.fileset = None

        # Validate the series parameter
        if self._series is not None:
            assert isinstance(self._series, (int, str)), (
                "The series parameter must be either an integer or string. "
                "Selection of multiple series from the same image is currently not supported."
            )

    def set_dask_temp_dir(self,
                          temp_dir: Union[str, Path] = 'auto'
                          ):
        """
        Set the temporary directory to store dask intermediate results.

        If the argument is 'auto' or None, the function will create a temporary directory
        in the system's default temporary directory and store the path in
        self._dask_temp_dir. If the argument is a string or Path, the function will create
        a temporary directory at the given path and store the path in
        self._dask_temp_dir.

        Parameters
        ----------
        temp_dir : Union[str, Path], optional
            The name of the temporary directory to store the intermediate results.
            Defaults to 'auto'.

        """
        if isinstance(temp_dir, tempfile.TemporaryDirectory):
            self._dask_temp_dir = temp_dir
            return
        if temp_dir in ('auto', None):
            temp_dir = tempfile.TemporaryDirectory(delete=False)
        elif isinstance(temp_dir, (str, Path)):
            os.makedirs(temp_dir, exist_ok=True)
            temp_dir = tempfile.TemporaryDirectory(dir=temp_dir, delete=False)
        else:
            raise TypeError(f"Invalid temp_dir argument: {temp_dir}")
        self._dask_temp_dir = temp_dir

    def read_dataset(self,
                     verified_for_cluster,
                     chunks_yx=None,
                     readers_params={},
                     # skip_dask = False
                     ):
        """
        - If the input path is a directory, can read single or multiple files from it.
        - If the input path is a file, can read a single image from it.
        - If the input path is a file with multiple series, can currently only read one series from it. Reading multiple series is currently not supported.
        - If the input path is a csv file with filepaths and conversion parameters, can read the filepaths and conversion parameters from it.
        :return:
        """
        input_path = self._input_path  # todo: make them settable from this method?
        includes = self._includes
        excludes = self._excludes
        metadata_path = self._metadata_path
        series = self._series
        zarr_format = self._zarr_format
        verbose = self._verbose

        _input_is_csv = False
        _input_is_tiff = False
        if input_path.endswith('.csv'):
            _input_is_csv = True
            self.filepaths = take_filepaths(input_path, includes, excludes)

        if os.path.isfile(input_path) or input_path.endswith('.zarr'):
            dirname = os.path.dirname(input_path)
            basename = os.path.basename(input_path)
            input_path_ = f"{dirname}/*{basename}"
            self._input_path = input_path_

        print(self._input_path)

        if not _input_is_csv:
            self.filepaths = take_filepaths(input_path, includes, excludes)
        print(self.filepaths)

        if series is None or series == 0:  # TODO: parallelize with cluster setting. Keep serial for no-cluster
            try:
                readers_params.pop('scene_index')
            except:
                pass

        futures = [read_single_image_asarray(path,
                                             chunks_yx=chunks_yx,
                                             verified_for_cluster=verified_for_cluster,
                                             zarr_format=zarr_format,
                                             verbose=verbose,
                                             **readers_params)
                   for path in self.filepaths]
        self.arrays = dask.compute(*futures)

        if metadata_path is None:
            self.metadata_path = self.filepaths[0]
        else:
            self.metadata_path = metadata_path

    def _digest_skip_dask(self,
                          path,
                          metadata_reader='bfio',
                          **kwargs
                          ):
        # n_filepaths = len(self.filepaths)
        # if n_filepaths != 1:
        #     raise Exception(f"Skipping dask digest is only supported for single filepaths. Got {n_filepaths} filepaths.")
        # path = self.filepaths[0]
        # if not path.endswith('.tif') or not path.endswith('.tiff'):
        #     raise Exception(f"Skipping dask digest is only supported for tif files. Got {path}.")
        from eubi_bridge.base.data_manager import ArrayManager
        manager = ArrayManager(path,
                               series=self._series,
                               metadata_reader=metadata_reader,
                               # skip_dask = False,
                               **kwargs
                               )
        import os
        name = os.path.basename(path).split('.')[0]
        self.digested_arrays = {name: manager.array}
        self.digested_arrays_sample_paths = {name: path}
        self.managers = {name: manager}
        self._compute_pixel_metadata(**kwargs)

    def digest(self,  # TODO: refactor to "assimilate_tags" and "concatenate?"
               time_tag: Union[str, tuple] = None,
               channel_tag: Union[str, tuple] = None,
               z_tag: Union[str, tuple] = None,
               y_tag: Union[str, tuple] = None,
               x_tag: Union[str, tuple] = None,
               axes_of_concatenation: Union[int, tuple, str] = None,
               ###
               metadata_reader: str = 'bfio',
               # skip_dask: bool = True,
               **kwargs
               ):
        """
        Digest the input data. This optionally involves
        concatenating multiple images along specified axes.

        Parameters
        ----------
        time_tag : Union[str, tuple], optional
            The tag for the time axis. Defaults to None.
        channel_tag : Union[str, tuple], optional
            The tag for the channel axis. Defaults to None.
        z_tag : Union[str, tuple], optional
            The tag for the z axis. Defaults to None.
        y_tag : Union[str, tuple], optional
            The tag for the y axis. Defaults to None.
        x_tag : Union[str, tuple], optional
            The tag for the x axis. Defaults to None.
        axes_of_concatenation : Union[int, tuple, str], optional
            The axes to concatenate. Defaults to None.


        Examples
        --------
            >>> # For a set of files with patterns in the file path in the following format:
            >>> # ["timepoint0_channel0_slice0.tif",...,"timepoint12_channel2_slice25.tif"] ]
            >>> # Concatenate only along specific axes:
            >>> bridge.digest(
            ...     time_tag='timepoint',
            ...     channel_tag='channel',
            ...     z_tag='slice_',
            ...     axes_of_concatenation='tz'  # Concatenate only time and z dimensions
            ... )
        """

        axes = 'tczyx'
        tags = (time_tag, channel_tag, z_tag, y_tag, x_tag)

        # Create a FileSet object
        self.batchfile = BatchFile(self.filepaths,
                                   arrays=self.arrays,
                                   axis_tag0=time_tag,
                                   axis_tag1=channel_tag,
                                   axis_tag2=z_tag,
                                   axis_tag3=y_tag,
                                   axis_tag4=x_tag,
                                   )

        # TODO: UPDATE WITH CONSTRUCT_MANAGERS

        axdict = dict(zip(axes, tags))

        if axes_of_concatenation is None:
            axes_of_concatenation = []

        axlist = [axes.index(x) for x in axes_of_concatenation if x in axes]

        self.batchfile._construct_managers(axes=axlist,
                                           series=self._series,
                                           metadata_reader=metadata_reader,
                                           **kwargs)
        self.batchfile._construct_channel_managers(series=self._series,
                                                   metadata_reader=metadata_reader,
                                                   **kwargs)
        self.batchfile._complete_process(axlist)
        # Get the refined arrays and sample paths
        (self.digested_arrays,
         self.digested_arrays_sample_paths,
         self.managers
         ) = self.batchfile.get_output_dicts(self._input_path)
        self._compute_pixel_metadata(**kwargs)
        return self

    def _compute_pixel_metadata(self,
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
        for name, arr in self.digested_arrays.items():
            path = self.digested_arrays_sample_paths[name]
            manager = self.managers[name]
            manager.set_arraydata(arr)
            manager.update_meta(
                new_unitdict=update_unitdict,
                new_scaledict=update_scaledict
            )

        self.batchdata = BatchManager(self.managers)
        self.batchdata.fill_default_meta()

    def squeeze_dataset(self):
        self.batchdata.squeeze()

    def transpose_dataset(self,
                          dimension_order=Union[str, tuple, list]
                          ):
        """
        Transpose the dataset according to the given dimension order.

        Parameters
        ----------
        dimension_order : Union[str, tuple, list]
            The order of the dimensions in the transposed array.

        """
        self.batchdata.transpose(newaxes=dimension_order)

    def crop_dataset(self, **kwargs):
        self.batchdata.crop(**kwargs)

    def to_cupy(self):
        self.batchdata.to_cupy()

    def _prepare_array_metadata(self,
                                batch_manager,
                                sample_path_mapping,
                                autochunk=True
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
        channel_meta = {}  # a dict of lists, each list being the length of channels per image

        for array_name, file_path in sample_path_mapping.items():
            manager = batch_manager.managers[array_name]
            array_data[array_name] = {'0': manager.array}
            dimension_scales[array_name] = {'0': manager.scales}
            dimension_axes[array_name] = {'0': manager.axes}
            dimension_units[array_name] = {'0': manager.units}
            chunk_configs[array_name] = {'0': manager.chunks}
            channel_meta[array_name] = {'0': manager.channels}

        return array_data, dimension_scales, dimension_axes, dimension_units, chunk_configs, channel_meta

    def _create_output_path_mapping(self, output_dir,
                                    nested_data, sample_paths):
        """Create flattened path mappings for the given data dictionary.

        Args:
            output_dir: Base output directory
            nested_data: Nested dictionary of data to be flattened
            sample_paths: Dictionary mapping array names to their file paths

        Returns:
            Dictionary with output file paths as keys
        """
        return {
            os.path.join(
                output_dir,
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

    def write_arrays(self,
                     output_dir,
                     compute=True,
                     use_tensorstore=False,
                     rechunk_method='tasks',
                     **kwargs):
        """Write processed arrays to storage.

        Args:
            output_dir: Base output directory
            compute: Whether to compute the result immediately
            use_tensorstore: Whether to use tensorstore for storage
            rechunk_method: Method to use for rechunking ('auto', 'rechunker', etc.)
            **kwargs: Additional arguments for array storage

        Returns:
            Results of the storage operation
        """
        output_dir = os.path.abspath(output_dir)
        storage_options = kwargs.copy()

        # Apply transformations
        if storage_options.get('use_gpu', False):
            self.to_cupy()
        if storage_options.get('squeeze', False):
            self.squeeze_dataset()
        if storage_options.get('dimension_order'):
            if not storage_options.get('dimension_order') in (None, 'auto'):
                self.transpose_dataset(storage_options['dimension_order'])

        # Update storage options with format and verbosity
        storage_options.update({
            'zarr_format': self._zarr_format,
            'verbose': self._verbose
        })

        # Apply cropping (will not do anything if no cropping parameter is provided)
        self.crop_dataset(**storage_options)

        # Prepare data for storage
        batch_manager = self.batchdata
        sample_path_mapping = self.digested_arrays_sample_paths
        assert batch_manager is not None, "The 'batchdata' must be computed before writing arrays"

        # Extract array metadata
        (array_data,
         dimension_scales,
         dimension_axes,
         dimension_units,
         chunk_configs,
         channel_meta) = self._prepare_array_metadata(
            batch_manager, sample_path_mapping
        )

        # Create path mappings for storage
        path_mappings = {
            'arrays': self._create_output_path_mapping(output_dir, array_data, sample_path_mapping),
            'scales': self._create_output_path_mapping(output_dir, dimension_scales, sample_path_mapping),
            'axes': self._create_output_path_mapping(output_dir, dimension_axes, sample_path_mapping),
            'units': self._create_output_path_mapping(output_dir, dimension_units, sample_path_mapping),
            'chunks': self._create_output_path_mapping(output_dir, chunk_configs, sample_path_mapping),
            'channels': self._create_output_path_mapping(output_dir, channel_meta, sample_path_mapping),
        }

        # Configure chunking and sharding
        chunk_sizes = {
            't': storage_options.get('time_chunk'),
            'c': storage_options.get('channel_chunk'),
            'z': storage_options.get('z_chunk'),
            'y': storage_options.get('y_chunk'),
            'x': storage_options.get('x_chunk')
        }

        shard_coefficients = {
            't': storage_options.get('time_shard_coef', 1),
            'c': storage_options.get('channel_shard_coef', 1),
            'z': storage_options.get('z_shard_coef', 3),
            'y': storage_options.get('y_shard_coef', 3),
            'x': storage_options.get('x_shard_coef', 3)
        }

        # Process chunk configurations
        processed_chunk_sizes, processed_shard_coeffs = self._process_chunking_configurations(
            chunk_sizes,
            shard_coefficients,
            path_mappings['axes'],
            path_mappings['chunks']
        )

        # Store arrays
        storage_results = store_arrays(
            path_mappings['arrays'],
            output_dir,
            axes=path_mappings['axes'],
            scales=path_mappings['scales'],
            units=path_mappings['units'],
            output_chunks=processed_chunk_sizes,
            output_shard_coefficients=processed_shard_coeffs,
            use_tensorstore=use_tensorstore,
            compute=compute,
            rechunk_method=rechunk_method,
            channel_meta=path_mappings['channels'] or None,
            **storage_options
        )

        self.flatarrays = path_mappings['arrays']

        # Save OME-XML metadata if requested
        if storage_options.get('save_omexml', False):
            manager_paths = {
                os.path.join(output_dir, f"{name}.zarr" if not name.endswith('zarr') else name):
                    batch_manager.managers[name]
                for name, file_path in sample_path_mapping.items()
            }

            for output_path, manager in manager_paths.items():
                if manager.omemeta is None:
                    manager.create_omemeta()
                manager.save_omexml(output_path)

        return storage_results


def downscale(
        gr_paths,
        time_scale_factor,
        channel_scale_factor,
        z_scale_factor,
        y_scale_factor,
        x_scale_factor,
        n_layers,
        downscale_method='simple',
        **kwargs  # a min_dimension_size parameter added
):
    scale_factor_dict = {
        't': time_scale_factor,
        'c': channel_scale_factor,
        'z': z_scale_factor,
        'y': y_scale_factor,
        'x': x_scale_factor
    }

    if isinstance(gr_paths, dict):
        gr_paths = list(set(os.path.dirname(key) for key in gr_paths.keys()))

    pyrs = [Pyramid(path) for path in gr_paths]  # TODO: add a to_cupy parameter here.
    result_collection = []

    min_dimension_size = kwargs.get('min_dimension_size', None)
    for pyr in pyrs:
        scale_factor = [scale_factor_dict[ax] for ax in pyr.meta.axis_order]

        pyr.update_downscaler(scale_factor=scale_factor,
                              n_layers=n_layers,
                              downscale_method=downscale_method,
                              min_dimension_size=min_dimension_size
                              )
        grpath = pyr.gr.store.root
        grname = os.path.basename(grpath)
        grdict = {grname: {}}
        axisdict = {grname: {}}
        scaledict = {grname: {}}
        unitdict = {grname: {}}
        chunkdict = {grname: {}}
        sharddict = {grname: {}}

        for key, value in pyr.downscaler.downscaled_arrays.items():
            if key != '0':
                grdict[grname][key] = value
                axisdict[grname][key] = tuple(pyr.meta.axis_order)
                scaledict[grname][key] = tuple(pyr.downscaler.dm.scales[int(key)])
                unitdict[grname][key] = tuple(pyr.meta.unit_list)
                chunkdict[grname][key] = tuple(pyr.base_array.chunksize)
                # channeldict[grname][key] = tuple(pyr.meta.channels)
                if pyr.meta.zarr_format == 3:
                    # print(f"pyr: {pyr}")
                    sharddict[grname][key] = tuple(pyr.meta.zarr_group)
                    basepath = pyr.meta.resolution_paths[0]
                    sharddict[grname][key] = tuple(pyr.layers[basepath].shards)

        output_path = os.path.dirname(grpath)
        arrays = {k: {'0': v} if not isinstance(v, dict) else v for k, v in grdict.items()}

        ### TODO: make this a separate function? def flatten_pyramids
        flatarrays = {os.path.join(output_path, f"{key}.zarr"
        if not key.endswith('zarr') else key, str(level)): arr
                      for key, subarrays in arrays.items()
                      for level, arr in subarrays.items()}
        flataxes = {os.path.join(output_path, f"{key}.zarr"
        if not key.endswith('zarr') else key, str(level)): axes
                    for key, subaxes in axisdict.items()
                    for level, axes in subaxes.items()}
        flatscales = {os.path.join(output_path, f"{key}.zarr"
        if not key.endswith('zarr') else key, str(level)): scale
                      for key, subscales in scaledict.items()
                      for level, scale in subscales.items()}
        flatunits = {os.path.join(output_path, f"{key}.zarr"
        if not key.endswith('zarr') else key, str(level)): unit
                     for key, subunits in unitdict.items()
                     for level, unit in subunits.items()}
        flatchunks = {os.path.join(output_path, f"{key}.zarr"
        if not key.endswith('zarr') else key, str(level)): chunk
                      for key, subchunks in chunkdict.items()
                      for level, chunk in subchunks.items()}

        if len(sharddict) > 0:
            # print(f"sharddict: {sharddict}")
            flatshards = {os.path.join(output_path, f"{key}.zarr"
            if not key.endswith('zarr') else key, str(level)): shard
                          for key, subshards in sharddict.items()
                          for level, shard in subshards.items()}
            if len(flatshards) == 0:
                flatshards = None
        else:
            flatshards = None
        ### TODO ends
        # print(f"flatshards: {flatshards}")

        results = store_arrays(flatarrays,
                               output_path=output_path,
                               axes=flataxes,
                               scales=flatscales,
                               units=flatunits,
                               output_chunks=flatchunks,
                               output_shards=flatshards,
                               compute=False,
                               channel_meta=None,
                               **kwargs
                               )

        result_collection += list(results.values())
    if 'rechunk_method' in kwargs:
        if kwargs.get('rechunk_method') == 'rechunker':
            raise NotImplementedError(f"Rechunker is not supported for the downscaling step.")
    if 'max_mem' in kwargs:
        raise NotImplementedError(f"Rechunker is not supported for the downscaling step.")
    try:
        dask.compute(*result_collection)
    except Exception as e:
        # print(e)
        pass
    return results
