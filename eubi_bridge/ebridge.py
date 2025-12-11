import scyjava

# Disable all Maven/JGO endpoints BEFORE bioio/bioformats is imported
scyjava.config.endpoints.clear()
scyjava.config.maven_offline = True
scyjava.config.jgo_disabled = True

from eubi_bridge.utils.convenience import sensitive_glob, is_zarr_group, is_zarr_array, take_filepaths, \
    autocompute_chunk_shape, soft_start_jvm

import warnings
warnings.filterwarnings(
    "ignore",
    message="Dask configuration key 'distributed.p2p.disk' has been deprecated",
    category=FutureWarning,
    module="dask.config",
)
warnings.filterwarnings(
    "ignore",
    message="Could not parse tiff pixel size",
    category=UserWarning,
    module="bioio_tifffile.reader",
)

import shutil, time, zarr, pprint, psutil, dask, s3fs
import numpy as np, os, tempfile

from dask import array as da
from pathlib import Path
from typing import Union

# from eubi_bridge.ngff.multiscales import Pyramid
# from eubi_bridge.ngff import defaults
from eubi_bridge.core.data_manager import BatchManager
from eubi_bridge.utils.convenience import take_filepaths, is_zarr_group
from eubi_bridge.utils.metadata_utils import print_printable, get_printables
from eubi_bridge.utils.logging_config import get_logger
from eubi_bridge.conversion.aggregative_conversion_base import AggregativeConverter
from eubi_bridge.conversion.converter import run_conversions
from eubi_bridge.conversion.updater import run_updates

# Set up logger for this module
logger = get_logger(__name__)



def verify_filepaths_for_cluster(filepaths):
    """Verify that all file extensions are supported for distributed processing."""
    logger.info("Verifying file extensions for distributed setup.")
    formats = ['lif', 'czi', 'lsm',
               'nd2', '.h5'
               'ome.tiff', 'ome.tif',
               'tiff', 'tif', 'zarr',
               'png', 'jpg', 'jpeg',
               'btf']

    for filepath in filepaths:
        verified = any(list(map(lambda path, ext: path.endswith(ext),
                                [filepath] * len(formats), formats)))
        if not verified:
            root, ext = os.path.splitext(filepath)
            logging.warning(f"Distributed execution is not supported for the {ext} format")
            logger.warning(f"Falling back on multithreading.")
            break
    if verified:
        logger.info("File extensions were verified for distributed setup.")
    return verified

# def wrap_output_path(output_path):
#     if output_path.startswith('https://'):
#         endpoint_url = 'https://' + output_path.replace('https://', '').split('/')[0]
#         relpath = output_path.replace(endpoint_url, '')
#         fs = s3fs.S3FileSystem(
#             client_kwargs={
#                 'endpoint_url': endpoint_url,
#             },
#             endpoint_url=endpoint_url
#         )
#         fs.makedirs(relpath, exist_ok=True)
#         mapped = fs.get_mapper(relpath)
#     else:
#         os.makedirs(output_path, exist_ok=True)
#         mapped = os.path.abspath(output_path)
#     return mapped

class EuBIBridge:
    """
    EuBIBridge is a conversion tool for bioimage datasets, allowing for both unary and aggregative conversion of image
    data collections to OME-Zarr format.

    Attributes:
        config_gr (zarr.Group): Configuration settings stored in a Zarr group.
        config (dict): Dictionary representation of configuration settings for cluster, conversion, and downscaling.
        dask_config (dict): Dictionary representation of configuration settings for dask.distributed.
        root_defaults (dict): Installation defaults of configuration settings for cluster, conversion, and downscaling.
        root_dask_defaults (dict): Installation defaults of configuration settings for dask.distributed.
    """
    TABLE_FORMATS = [".csv", ".tsv", ".txt", ".xls", ".xlsx"]

    def __init__(self,
                 configpath=f"{os.path.expanduser('~')}/.eubi_bridge",
                 ):
        """
        Initializes the EuBIBridge class and loads or sets up default configuration.

        Args:
            configpath (str, optional): Path to store configuration settings. Defaults to the home directory.
        """

        self.root_defaults = dict(
            cluster=dict(
                on_local_cluster = False,
                on_slurm=False,
                max_workers=16,  # size of the pool for sync writer
                compute_batch_size = 4,
                memory_limit_per_batch = '1GB',
                max_concurrency = 16,  # limit how many writes run at once
                memory_per_worker = '10GB'
                # executor_kind = "processes",  # "threads" for I/O, "processes" for CPU-bound compression
                ),
            readers=dict(
                as_mosaic=False,
                view_index=0,
                phase_index=0,
                illumination_index=0,
                scene_index=0,
                rotation_index=0,
                mosaic_tile_index=0,
                sample_index=0,
                # use_bioformats_readers=False
            ),
            conversion=dict(
                verbose=False,
                zarr_format=2,
                skip_dask=False,
                auto_chunk=False,
                target_chunk_mb=1,
                time_chunk=1,
                channel_chunk=1,
                z_chunk=96,
                y_chunk=96,
                x_chunk=96,
                time_shard_coef=1,
                channel_shard_coef=1,
                z_shard_coef=3,
                y_shard_coef=3,
                x_shard_coef=3,
                time_range=None,
                channel_range=None,
                z_range=None,
                y_range=None,
                x_range=None,
                dimension_order='tczyx',
                compressor='blosc',
                compressor_params={},
                overwrite=False,
                override_channel_names = False,
                channel_intensity_limits = 'from_dtype',
                # use_tensorstore=False,
                # use_gpu=False,
                # rechunk_method='tasks',
                # trim_memory=False,
                metadata_reader='bfio',
                save_omexml=True,
                squeeze=True,
                dtype='auto'
            ),
            downscale=dict(
                time_scale_factor=1,
                channel_scale_factor=1,
                z_scale_factor=2,
                y_scale_factor=2,
                x_scale_factor=2,
                n_layers=None,
                min_dimension_size=64,
                downscale_method='simple',
            )
        )

        # self.root_defaults = defaults
        # self.root_dask_defaults = root_dask_defaults
        config_gr = zarr.open_group(configpath, mode='a')
        config = config_gr.attrs
        for key in self.root_defaults.keys():
            if key not in config.keys():
                config[key] = {}
                for subkey in self.root_defaults[key].keys():
                    if subkey not in config[key].keys():
                        config[key][subkey] = self.root_defaults[key][subkey]
            config_gr.attrs[key] = config[key]
        self.config = dict(config_gr.attrs)
        ###
        self.config_gr = config_gr
        ###
        self._dask_temp_dir = None

    def _optimize_dask_config(self):
        """Optimize Dask configuration for maximum conversion speed.

        This configuration is tuned for high-performance data processing with Dask,
        focusing on maximizing throughput while maintaining system stability.
        The settings are optimized for I/O and CPU-bound workloads.
        """

        # Get system information for adaptive configuration
        total_memory = psutil.virtual_memory().total
        total_cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 4

        # Calculate memory fractions based on available memory
        memory_target = float(os.getenv('DASK_MEMORY_TARGET', '0.8'))
        memory_spill = float(os.getenv('DASK_MEMORY_SPILL', '0.9'))
        memory_pause = float(os.getenv('DASK_MEMORY_PAUSE', '0.95'))

        dask.config.set({
            # Task scheduling and execution
            'optimization.fuse.active': True,
            'optimization.fuse.ave-width': 10,  # Balanced fusion width
            'optimization.fuse.subgraphs': True,
            'optimization.fuse.rename-keys': True,
            'optimization.culling.active': True,  # Remove unnecessary tasks
            'optimization.rewrite.fuse': True,

        })

    def reset_config(self):
        """
        Resets the cluster, conversion and downscale parameters to the installation defaults.
        """
        self.config_gr.attrs.update(self.root_defaults)
        self.config = dict(self.config_gr.attrs)

    def show_config(self):
        """
        Displays the current cluster, conversion, and downscale parameters.
        """
        pprint.pprint(self.config)

    def show_root_defaults(self):
        """
        Displays the installation defaults for cluster, conversion, and downscale parameters.
        """
        pprint.pprint(self.root_defaults)

    def _collect_params(self, param_type, **kwargs):
        """
        Gathers parameters from the configuration, allowing for overrides.

        Args:
            param_type (str): The type of parameters to collect (e.g., 'cluster', 'conversion', 'downscale').
            **kwargs: Parameter values that may override defaults.

        Returns:
            dict: Collected parameters.
        """
        params = {}
        for key in self.config[param_type].keys():
            if key in kwargs.keys():
                params[key] = kwargs[key]
            else:
                params[key] = self.config[param_type][key]
            if key == 'dtype':
                if params[key] == 'auto':
                    params[key] = None
        return params

    def configure_cluster(self,
                          max_workers: int = 'default',
                          compute_batch_size: int = 'default',
                          memory_limit_per_batch: int = 'default',
                          memory_per_worker: int = 'default',
                          max_concurrency: int = 'default',
                          on_local_cluster: bool = 'default',
                          on_slurm: bool = 'default'
                          ):
        """
        Updates cluster configuration settings. To update the current default value for a parameter, provide that parameter with a value other than 'default'.

        The following parameters can be configured:
            - max_workers (int, optional): Size of the pool for sync writer.
            - compute_batch_size (int, optional): Number of batches to process in parallel.
            - memory_limit_per_batch (int, optional): Memory limit in MB for each batch.
            - max_concurrency (int, optional): Maximum number of concurrent operations.

        Args:
            max_workers (int, optional): Size of the pool for sync writer.
            compute_batch_size (int, optional): Number of batches to process in parallel.
            memory_limit_per_batch (int, optional): Memory limit in MB for each batch.
            max_concurrency (int, optional): Maximum number of concurrent operations.

        Returns:
            None
        """

        params = {
            'max_workers': max_workers,
            'compute_batch_size': compute_batch_size,
            'memory_limit_per_batch': memory_limit_per_batch,
            'memory_per_worker': memory_per_worker,
            'max_concurrency': max_concurrency,
            'on_local_cluster': on_local_cluster,
            'on_slurm': on_slurm
        }

        for key in params:
            if key in self.config['cluster'].keys():
                if params[key] != 'default':
                    self.config['cluster'][key] = params[key]
        self.config_gr.attrs['cluster'] = self.config['cluster']

    def configure_readers(self,
                          as_mosaic: bool = 'default',
                          view_index: int = 'default',
                          phase_index: int = 'default',
                          illumination_index: int = 'default',
                          scene_index: int = 'default',
                          rotation_index: int = 'default',
                          mosaic_tile_index: int = 'default',
                          sample_index: int = 'default',
                          # use_bioformats_readers: bool = 'default'
                          ):
        """
        Updates reader configuration settings. To update the current default value for a parameter, provide that parameter with a value other than 'default'.

        Returns:
            None
        """

        params = {
            'as_mosaic': as_mosaic,
            'view_index': view_index,
            'phase_index': phase_index,
            'illumination_index': illumination_index,
            'scene_index': scene_index,
            'rotation_index': rotation_index,
            'mosaic_tile_index': mosaic_tile_index,
            'sample_index': sample_index,
            'use_bioformats_readers': use_bioformats_readers
        }

        for key in params:
            if key in self.config['readers'].keys():
                if params[key] != 'default':
                    self.config['readers'][key] = params[key]
        self.config_gr.attrs['readers'] = self.config['readers']

    def configure_conversion(self,
                             zarr_format: int = 'default',
                             skip_dask: bool = 'default',
                             auto_chunk: bool = 'default',
                             target_chunk_mb: float = 'default',
                             time_chunk: int = 'default',
                             channel_chunk: int = 'default',
                             z_chunk: int = 'default',
                             y_chunk: int = 'default',
                             x_chunk: int = 'default',
                             time_shard_coef: int = 'default',
                             channel_shard_coef: int = 'default',
                             z_shard_coef: int = 'default',
                             y_shard_coef: int = 'default',
                             x_shard_coef: int = 'default',
                             time_range: int = 'default',
                             channel_range: int = 'default',
                             z_range: int = 'default',
                             y_range: int = 'default',
                             x_range: int = 'default',
                             compressor: str = 'default',
                             compressor_params: dict = 'default',
                             overwrite: bool = 'default',
                             override_channel_names: bool = 'default',
                             channel_intensity_limits = 'default',
                             metadata_reader: str = 'default',
                             save_omexml: bool = 'default',
                             squeeze: bool = 'default',
                             dtype: str = 'default',
                             verbose: bool = 'default',
                             ):
        """
        Updates conversion configuration settings. To update the current default value for a parameter, 
        provide that parameter with a value other than 'default'.

        Args:
            zarr_format (int, optional): Zarr format version (2 or 3).
            skip_dask (bool, optional): Whether to skip using Dask for processing.
            auto_chunk (bool, optional): Whether to automatically determine chunk sizes.
            target_chunk_mb (float, optional): Target chunk size in MB.
            time_chunk (int, optional): Chunk size for time dimension.
            channel_chunk (int, optional): Chunk size for channel dimension.
            z_chunk (int, optional): Chunk size for Z dimension.
            y_chunk (int, optional): Chunk size for Y dimension.
            x_chunk (int, optional): Chunk size for X dimension.
            time_shard_coef (int, optional): Sharding coefficient for time dimension.
            channel_shard_coef (int, optional): Sharding coefficient for channel dimension.
            z_shard_coef (int, optional): Sharding coefficient for Z dimension.
            y_shard_coef (int, optional): Sharding coefficient for Y dimension.
            x_shard_coef (int, optional): Sharding coefficient for X dimension.
            time_range (int, optional): Range for time dimension.
            channel_range (int, optional): Range for channel dimension.
            z_range (int, optional): Range for Z dimension.
            y_range (int, optional): Range for Y dimension.
            x_range (int, optional): Range for X dimension.
            compressor (str, optional): Compression algorithm to use.
            compressor_params (dict, optional): Parameters for the compressor.
            overwrite (bool, optional): Whether to overwrite existing data.
            override_channel_names (bool, optional): Whether to override channel names.
            channel_intensity_limits: Intensity limits for channels.
            metadata_reader (str, optional): Reader to use for metadata.
            save_omexml (bool, optional): Whether to save OME-XML metadata.
            squeeze (bool, optional): Whether to squeeze single-dimensional axes.
            dtype (str, optional): Data type for the output array.
            verbose (bool, optional): Whether to enable verbose output.

        Returns:
            None
        """

        params = {
            'zarr_format': zarr_format,
            'skip_dask': skip_dask,
            'auto_chunk': auto_chunk,
            'target_chunk_mb': target_chunk_mb,
            'time_chunk': time_chunk,
            'channel_chunk': channel_chunk,
            'z_chunk': z_chunk,
            'y_chunk': y_chunk,
            'x_chunk': x_chunk,
            'time_shard_coef': time_shard_coef,
            'channel_shard_coef': channel_shard_coef,
            'z_shard_coef': z_shard_coef,
            'y_shard_coef': y_shard_coef,
            'x_shard_coef': x_shard_coef,
            'time_range': time_range,
            'channel_range': channel_range,
            'z_range': z_range,
            'y_range': y_range,
            'x_range': x_range,
            'compressor': compressor,
            'compressor_params': compressor_params or {},
            'overwrite': overwrite,
            'override_channel_names': override_channel_names,
            'channel_intensity_limits': channel_intensity_limits,
            'metadata_reader': metadata_reader,
            'save_omexml': save_omexml,
            'squeeze': squeeze,
            'dtype': dtype,
            'verbose': verbose
        }

        for key in params:
            if key in self.config['conversion'].keys():
                if params[key] != 'default':
                    self.config['conversion'][key] = params[key]
        self.config_gr.attrs['conversion'] = self.config['conversion']

    def configure_downscale(self,
                            # downscale_method: str = 'default',
                            n_layers: int = 'default',
                            min_dimension_size: int = 'default',
                            time_scale_factor: int = 'default',
                            channel_scale_factor: int = 'default',
                            z_scale_factor: int = 'default',
                            y_scale_factor: int = 'default',
                            x_scale_factor: int = 'default',
                            ):
        """
        Updates downscaling configuration settings. To update the current default value for a parameter, provide that parameter with a value other than 'default'.

        The following parameters can be configured:
            - downscale_method (str, optional): Downscaling algorithm.
            - n_layers (int, optional): Number of downscaling layers.
            - scale_factor (list, optional): Scaling factors for each dimension.

        Args:
            downscale_method (str, optional): Downscaling algorithm.
            n_layers (int, optional): Number of downscaling layers.
            scale_factor (list, optional): Scaling factors for each dimension.

        Returns:
            None
        """

        params = {
            # 'downscale_method': downscale_method,
            'n_layers': n_layers,
            'min_dimension_size': min_dimension_size,
            'time_scale_factor': time_scale_factor,
            "channel_scale_factor": channel_scale_factor,
            "z_scale_factor": z_scale_factor,
            "y_scale_factor": y_scale_factor,
            "x_scale_factor": x_scale_factor,
        }

        for key in params:
            if key in self.config['downscale'].keys():
                if params[key] != 'default':
                    self.config['downscale'][key] = params[key]
        self.config_gr.attrs['downscale'] = self.config['downscale']

    def to_zarr(self,
                input_path,
                output_path=None,
                includes=None,
                excludes=None,
                time_tag: Union[str, tuple] = None,
                channel_tag: Union[str, tuple] = None,
                z_tag: Union[str, tuple] = None,
                y_tag: Union[str, tuple] = None,
                x_tag: Union[str, tuple] = None,
                concatenation_axes: Union[int, tuple, str] = None,
                **kwargs # metadata kwargs such as pixel sizes and channel info
                ):
        """Synchronous wrapper for the async to_zarr_async method."""
        t0 = time.time()
        soft_start_jvm()
        # Get parameters:
        logger.info(f"Conversion starting.")
        if output_path is None:
            assert input_path.endswith(('.csv', '.tsv', '.txt', '.xlsx'))
        self.cluster_params = self._collect_params('cluster', **kwargs)
        self.readers_params = self._collect_params('readers', **kwargs)
        self.conversion_params = self._collect_params('conversion', **kwargs)
        # self.conversion_params = self._collect_params('conversion',
        #                                               channel_intensity_limits = channel_intensity_limits,
        #                                               **kwargs)
        self.downscale_params = self._collect_params('downscale', **kwargs)
        combined = {**self.cluster_params,
                    **self.readers_params,
                    **self.conversion_params,
                    **self.downscale_params}
        extra_kwargs = {key: kwargs[key] for key in kwargs if key not in combined}
        run_conversions(os.path.abspath(input_path),
                        output_path,
                        includes=includes,
                        excludes=excludes,
                        time_tag = time_tag,
                        channel_tag = channel_tag,
                        z_tag = z_tag,
                        y_tag = y_tag,
                        x_tag = x_tag,
                        concatenation_axes = concatenation_axes,
                        **combined,
                        **extra_kwargs
                        )
        t1 = time.time()
        logger.info(f"Conversion complete for all datasets.")
        logger.info(f"Elapsed for conversion + downscaling: {(t1 - t0) / 60} min.")

    def show_pixel_meta(self,
                        input_path: Union[Path, str],
                        includes=None,
                        excludes=None,
                        series: int = None,  # self.readers_params['scene_index'],
                        **kwargs
                        ):
        """
        Print pixel-level metadata for all datasets in the 'input_path'.

        Args:
            input_path (Union[Path, str]): Path to input file or directory.
            output_path (Union[Path, str]): Directory, in which the output OME-Zarrs will be written.
            includes (str, optional): Filename patterns to filter for.
            excludes (str, optional): Filename patterns to filter against.
            **kwargs: Additional configuration overrides.

        Raises:
            Exception: If no files are found in the input path.

        Prints:
            Process logs including conversion and downscaling time.

        Returns:
            None
        """

        # Get parameters:
        self.cluster_params = self._collect_params('cluster', **kwargs)
        self.readers_params = self._collect_params('readers', **kwargs)
        self.conversion_params = self._collect_params('conversion', **kwargs)

        paths = take_filepaths(input_path, includes=includes, excludes=excludes)

        filepaths = sorted(list(paths))

        ###### Start the cluster
        verified_for_cluster = verify_filepaths_for_cluster(filepaths)
        if not verified_for_cluster:
            self.cluster_params['no_distributed'] = True

        self._start_cluster(**self.cluster_params)

        series = self.readers_params['scene_index']

        ###### Read and digest
        base = BridgeBase(input_path,
                          excludes=excludes,
                          includes=includes,
                          series=series
                          )

        base.read_dataset(verified_for_cluster,
                          readers_params=self.readers_params
                          )

        base.digest()

        temp_dir = base._dask_temp_dir
        self.conversion_params['temp_dir'] = temp_dir

        if self.client is not None:
            base.client = self.client
        base.set_dask_temp_dir(self._dask_temp_dir)

        printables = {
            path: get_printables(
                manager.axes,
                manager.shapedict,
                manager.scaledict,
                manager.unitdict
            )
            for path, manager in base.batchdata.managers.items()
        }
        for path, printable in printables.items():
            print('---------')
            print(f"")
            print(f"Metadata for '{path}':")
            print_printable(printable)

        ###### Shutdown and clean up
        if self.client is not None:
            self.client.shutdown()
            self.client.close()

        if isinstance(self._dask_temp_dir, tempfile.TemporaryDirectory):
            shutil.rmtree(self._dask_temp_dir.name)
        else:
            shutil.rmtree(self._dask_temp_dir)

    def update_pixel_meta(self,
                          input_path: Union[Path, str],
                          includes=None,
                          excludes=None,
                          time_scale: (int, float) = None,
                          z_scale: (int, float) = None,
                          y_scale: (int, float) = None,
                          x_scale: (int, float) = None,
                          time_unit: str = None,
                          z_unit: str = None,
                          y_unit: str = None,
                          x_unit: str = None,
                          **kwargs
                          ):
        """
        Updates pixel metadata for image files located at the specified input path.

        Args:
            input_path (Union[Path, str]): Path to input file or directory.
            includes (optional): Filename patterns to include.
            excludes (optional): Filename patterns to exclude.
            series (int, optional): Series index to process.
            time_scale, z_scale, y_scale, x_scale ((int, float), optional): Scaling factors for the respective dimensions.
            time_unit, z_unit, y_unit, x_unit (str, optional): Units for the respective dimensions.
            **kwargs: Additional parameters for cluster and conversion configuration.

        Returns:
            None
        """

        # Collect cluster and conversion parameters
        self.cluster_params = self._collect_params('cluster', **kwargs)
        self.readers_params = self._collect_params('readers', **kwargs)
        self.conversion_params = self._collect_params('conversion', **kwargs)
        self.conversion_params['channel_intensity_limits'] = 'auto'

        combined = {**self.cluster_params,
                    **self.readers_params,
                    **self.conversion_params,
                    }
        extra_kwargs = {key: kwargs[key] for key in kwargs if key not in combined}

        # Collect file paths based on inclusion and exclusion patterns
        # Prepare pixel metadata arguments
        pixel_meta_kwargs_ = dict(time_scale=time_scale,
                                  z_scale=z_scale,
                                  y_scale=y_scale,
                                  x_scale=x_scale,
                                  time_unit=time_unit,
                                  z_unit=z_unit,
                                  y_unit=y_unit,
                                  x_unit=x_unit)
        pixel_meta_kwargs = {key: val for key, val in pixel_meta_kwargs_.items() if val is not None}
        run_updates(
                    input_path,
                    includes=includes,
                    excludes=excludes,
                    **combined,
                    **pixel_meta_kwargs,
                    **extra_kwargs
                    )

    def update_channel_meta(self,
                          input_path: Union[Path, str],
                          channel_indices: list = None,
                          channel_labels: list = None,
                          channel_colors: list = None,
                          channel_intensity_limits = 'from_dtype',
                          includes=None,
                          excludes=None,
                          **kwargs
                          ):
        """
        Updates pixel metadata for image files located at the specified input path.

        Args:
            **kwargs: Additional parameters for cluster and conversion configuration.

        Returns:
            None
        """

        # Collect cluster and conversion parameters
        self.cluster_params = self._collect_params('cluster', **kwargs)
        self.readers_params = self._collect_params('readers', **kwargs)
        self.conversion_params = self._collect_params('conversion',
                                                      channel_intensity_limits = channel_intensity_limits,
                                                      **kwargs)

        combined = {**self.cluster_params,
                    **self.readers_params,
                    **self.conversion_params,
                    }
        extra_kwargs = {key: kwargs[key] for key in kwargs if key not in combined}


        # Collect file paths based on inclusion and exclusion patterns
        # Prepare pixel metadata arguments
        items = [channel_indices, channel_labels, channel_colors]
        if any([item is not None for item in items]):
            assert channel_indices is not None, f"The channel_labels and channel_colors can only be modified if channel_indices is provided."
            if isinstance(channel_indices, (int,str)):
                channel_indices = [channel_indices]
            channel_meta_kwargs_ = dict(channel_indices=channel_indices,
                                       channel_labels=channel_labels,
                                       channel_colors=channel_colors)
            channel_meta_kwargs = {key: val for key, val in channel_meta_kwargs_.items() if val is not None}
            for key, val in channel_meta_kwargs.items():
                if isinstance(val, (int, str)):
                    channel_meta_kwargs[key] = [val]
                assert len(channel_meta_kwargs[key]) == len(channel_indices), f"If channel_labels or channel_colors are provided, they must have the same length as channel_indices."
        else:
            channel_meta_kwargs = {}

        run_updates(
                    os.path.abspath(input_path),
                    includes=includes,
                    excludes=excludes,
                    **combined,
                    **channel_meta_kwargs,
                    **extra_kwargs
                    )


