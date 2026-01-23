import os
import shutil
import tempfile
import time
import warnings
from pathlib import Path
from typing import Union

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from eubi_bridge.utils.logging_config import get_logger

# Set up logger for this module
logger = get_logger(__name__)
_console = Console()

# Heavy imports are deferred to only when needed (config commands don't need them)
# - scyjava, zarr, dask, numpy, psutil, s3fs, etc. are imported on-demand in methods


def _ensure_heavy_imports():
    """Lazy-load heavy modules only when needed for actual conversions."""
    global dask, np, psutil, s3fs, zarr, da, AggregativeConverter, run_conversions, run_updates
    
    if 'dask' in globals():
        return  # Already imported
    
    import scyjava
    scyjava.config.endpoints.clear()
    scyjava.config.maven_offline = True
    scyjava.config.jgo_disabled = True
    
    import dask
    import numpy as np
    import psutil
    import s3fs
    import zarr
    from dask import array as da
    
    # Suppress warnings
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
    
    from eubi_bridge.conversion.aggregative_conversion_base import AggregativeConverter
    from eubi_bridge.conversion.converter import run_conversions
    from eubi_bridge.conversion.updater import run_updates



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
            logger.warning(f"Distributed execution is not supported for the {ext} format")
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
                use_threading=False,
                max_workers=4,  # size of the pool for sync writer
                queue_size = 4,
                region_size_mb = 256,
                max_concurrency = 4,  # limit how many writes run at once
                memory_per_worker = '1GB',
                tensorstore_data_copy_concurrency = 4,  # limit CPU cores for tensorstore data copying in downscaler
                max_retries = 10  # maximum attempts per task (1 initial + retries) with exponential backoff
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
            ),
            conversion=dict(
                verbose=False,
                zarr_format=2,
                skip_dask=False,
                auto_chunk=True,
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

        # Store configpath for lazy loading
        self._configpath = configpath
        self._config = None  # Lazy-loaded config cache
        self._dask_temp_dir = None

    def _get_json_path(self):
        """Get path to JSON config file."""
        return Path(self._configpath) / '.eubi_config.json'

    def _load_config_from_json(self):
        """Load config from JSON file (the single source of truth)."""
        import json
        json_path = self._get_json_path()
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def _save_config_to_json(self, config):
        """Save config to JSON file (the single source of truth)."""
        import json
        json_path = self._get_json_path()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(json_path, 'w') as f:
                json.dump(config, f, indent=2)
        except IOError as e:
            logger.warning(f"Failed to save config JSON: {e}")

    def _ensure_config_loaded(self):
        """Lazy-load config from JSON file (the single source of truth)."""
        if self._config is not None:
            return  # Already loaded
        
        # Try to load from JSON first
        config = self._load_config_from_json()
        
        # If JSON doesn't exist, initialize from defaults and save
        if config is None:
            config = {}
            for key in self.root_defaults.keys():
                config[key] = dict(self.root_defaults[key])
            self._save_config_to_json(config)
        
        self._config = config

    @property
    def config(self):
        """Get config, lazy-loading if needed."""
        self._ensure_config_loaded()
        return self._config

    @config.setter
    def config(self, value):
        """Set config value and persist to JSON."""
        self._config = value
        self._save_config_to_json(value)

    def _optimize_dask_config(self):
        """Optimize Dask configuration for maximum conversion speed and memory efficiency.

        This configuration is tuned for high-performance data processing with Dask,
        focusing on maximizing throughput while maintaining system stability and
        preventing memory overflow.

        Performance Notes:
        - Task fusion reduces graph size and improves scheduler efficiency
        - Memory management prevents OOM errors with large datasets
        - Spill to disk allows processing beyond RAM capacity
        - Culling removes unnecessary task dependencies
        """

        # Get system information for adaptive configuration
        total_memory = psutil.virtual_memory().total
        total_cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 4

        # Calculate memory limits based on available memory
        memory_target_fraction = float(os.getenv('DASK_MEMORY_TARGET', '0.6'))
        memory_spill_fraction = float(os.getenv('DASK_MEMORY_SPILL', '0.8'))
        memory_pause_fraction = float(os.getenv('DASK_MEMORY_PAUSE', '0.95'))

        # Convert fractions to bytes
        memory_target = int(total_memory * memory_target_fraction)
        memory_spill = int(total_memory * memory_spill_fraction)

        dask.config.set({
            # Task scheduling and execution
            'optimization.fuse.active': True,
            'optimization.fuse.ave-width': 10,  # Balanced fusion width
            'optimization.fuse.subgraphs': True,
            'optimization.fuse.rename-keys': True,
            'optimization.culling.active': True,  # Remove unnecessary tasks
            'optimization.rewrite.fuse': True,
            # Memory management
            'dataframe.shuffle-compression': 'lz4',
            'array.chunk-size': '128 MiB',  # Default chunk size
        })

    def _cleanup_temp_dir(self):
        """Clean up temporary directory if it exists.

        Properly handles both tempfile.TemporaryDirectory objects and string paths.
        """
        if self._dask_temp_dir is None:
            return

        try:
            if isinstance(self._dask_temp_dir, tempfile.TemporaryDirectory):
                self._dask_temp_dir.cleanup()
            elif isinstance(self._dask_temp_dir, (str, Path)):
                path = Path(self._dask_temp_dir)
                if path.exists():
                    shutil.rmtree(path)
        except Exception as e:
            logger.warning(f"Error cleaning up temporary directory: {e}")
        finally:
            self._dask_temp_dir = None

    def __enter__(self):
        """Context manager entry for resource management."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        self._cleanup_temp_dir()
        return False

    def __del__(self):
        """Destructor to ensure cleanup on garbage collection."""
        try:
            self._cleanup_temp_dir()
        except Exception:
            pass  # Silence errors during destruction

    def reset_config(self):
        """
        Resets the cluster, conversion and downscale parameters to the installation defaults.
        """
        self.config = dict(self.root_defaults)

    def _print_config_unified(self, title: str, config_sections: dict):
        """Print all config sections in a single table with section headers as rows."""
        table = Table(
            title=title,
            title_style="bold cyan",
            show_header=True,
            header_style="bold white on blue",
            padding=(0, 1),
            border_style="blue"
        )
        table.add_column("Section", style="magenta", width=15)
        table.add_column("Parameter", style="cyan", width=25)
        table.add_column("Value", style="green")
        
        section_list = list(config_sections.items())
        for section_idx, (section_name, section_dict) in enumerate(section_list):
            is_first = True
            for key, value in sorted(section_dict.items()):
                # Format value nicely
                if isinstance(value, bool):
                    val_str = "[bold green]True[/bold green]" if value else "[bold red]False[/bold red]"
                elif isinstance(value, dict):
                    val_str = "[dim]" + str(value) + "[/dim]"
                elif value is None:
                    val_str = "[dim italic]None[/dim italic]"
                else:
                    val_str = str(value)
                
                # Only show section name on first row of each section
                section_display = f"[bold magenta]{section_name.upper()}[/bold magenta]" if is_first else ""
                table.add_row(section_display, key, val_str)
                is_first = False
            
            # Add separator line between sections (but not after the last one)
            if section_idx < len(section_list) - 1:
                table.add_section()
        
        _console.print(table)

    def show_config(self):
        """
        Displays the current cluster, conversion, and downscale parameters with rich formatting.
        Uses JSON file for fast read-only access (no zarr overhead).
        """
        _console.print("\n[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
        _console.print("[bold cyan]Current Configuration[/bold cyan]")
        _console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]\n")
        
        # Try to load config from JSON first (fast, no zarr overhead)
        config = self._load_config_from_json()
        
        # If JSON doesn't exist, load from zarr and cache it
        if config is None:
            config = self.config  # This triggers lazy zarr loading
        
        sections = {k: v for k, v in config.items() if k in ['cluster', 'readers', 'conversion', 'downscale']}
        self._print_config_unified("Configuration", sections)
        _console.print()

    def show_root_defaults(self):
        """
        Displays the installation defaults for cluster, conversion, and downscale parameters with rich formatting.
        """
        _console.print("\n[bold yellow]═══════════════════════════════════════════════════════[/bold yellow]")
        _console.print("[bold yellow]Installation Defaults[/bold yellow]")
        _console.print("[bold yellow]═══════════════════════════════════════════════════════[/bold yellow]\n")
        
        sections = {k: v for k, v in self.root_defaults.items() if k in ['cluster', 'readers', 'conversion', 'downscale']}
        self._print_config_unified("Defaults", sections)
        _console.print()

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
                          queue_size: int = 'default',
                          region_size_mb: int = 'default',
                          memory_per_worker: int = 'default',
                          max_concurrency: int = 'default',
                          on_local_cluster: bool = 'default',
                          on_slurm: bool = 'default',
                          use_threading: bool = 'default',
                          tensorstore_data_copy_concurrency: int = 'default',
                          max_retries: int = 'default'
                          ):
        """
        Updates cluster configuration settings. To update the current default value for a parameter, provide that parameter with a value other than 'default'.

        The following parameters can be configured:
            - max_workers (int, optional): Size of the pool for sync writer.
            - queue_size (int, optional): Number of batches to process in parallel.
            - region_size_mb (int, optional): Memory limit in MB for each batch.
            - max_concurrency (int, optional): Maximum number of concurrent operations.
            - use_threading (bool, optional): Use threading instead of multiprocessing (for cluster compatibility).
            - tensorstore_data_copy_concurrency (int, optional): Limit on CPU cores used concurrently for tensorstore data copying/encoding/decoding during downscaling. Lower values reduce thread contention. Default: 1 (serialized, safest).
            - max_retries (int, optional): Maximum attempts for worker tasks. If a worker crashes, retry up to this many times with exponential backoff. Default: 3 (1 initial + 2 retries).

        Args:
            max_workers (int, optional): Size of the pool for sync writer.
            queue_size (int, optional): Number of batches to process in parallel.
            region_size_mb (int, optional): Memory limit in MB for each batch.
            max_concurrency (int, optional): Maximum number of concurrent operations.
            on_local_cluster (bool, optional): Whether to use local Dask cluster.
            on_slurm (bool, optional): Whether to use SLURM cluster.
            use_threading (bool, optional): Use threading instead of multiprocessing.
            tensorstore_data_copy_concurrency (int, optional): CPU core limit for tensorstore data copying in downscaler. Default: 1.
            max_retries (int, optional): Maximum attempts per task with exponential backoff. Default: 3.

        Returns:
            None
        """

        params = {
            'max_workers': max_workers,
            'queue_size': queue_size,
            'region_size_mb': region_size_mb,
            'memory_per_worker': memory_per_worker,
            'max_concurrency': max_concurrency,
            'on_local_cluster': on_local_cluster,
            'on_slurm': on_slurm,
            'use_threading': use_threading,
            'tensorstore_data_copy_concurrency': tensorstore_data_copy_concurrency,
            'max_retries': max_retries
        }

        for key in params:
            if key in self.config['cluster'].keys():
                if params[key] != 'default':
                    self.config['cluster'][key] = params[key]
        self._save_config_to_json(self.config)

    def configure_readers(self,
                          as_mosaic: bool = 'default',
                          view_index: int = 'default',
                          phase_index: int = 'default',
                          illumination_index: int = 'default',
                          scene_index: int = 'default',
                          rotation_index: int = 'default',
                          mosaic_tile_index: int = 'default',
                          sample_index: int = 'default',
                          # use_bioformats_readers: bool = 'default' # TODO: implement
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
            #'use_bioformats_readers': use_bioformats_readers
        }

        for key in params:
            if key in self.config['readers'].keys():
                if params[key] != 'default':
                    self.config['readers'][key] = params[key]
        self._save_config_to_json(self.config)

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
        self._save_config_to_json(self.config)

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
        self._save_config_to_json(self.config)

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
                max_workers: int = None,
                max_retries: int = None,
                tensorstore_data_copy_concurrency: int = None,
                use_threading: bool = None,
                **kwargs # metadata kwargs such as pixel sizes and channel info
                ):
        """Synchronous wrapper for the async to_zarr_async method."""
        # Ensure heavy modules are loaded (scyjava, zarr, dask, etc.)
        _ensure_heavy_imports()
        
        # Initialize JVM for image reading (needs Java-based Bio-Formats)
        from eubi_bridge.utils.jvm_manager import soft_start_jvm
        soft_start_jvm()
        
        t0 = time.time()
        # Get parameters:
        logger.info(f"Conversion starting.")
        if output_path is None:
            assert input_path.endswith(('.csv', '.tsv', '.txt', '.xlsx'))
        
        # Extract explicit CLI parameters (from function arguments)
        cli_kwargs = {}
        if max_workers is not None:
            cli_kwargs['max_workers'] = max_workers
        if max_retries is not None:
            cli_kwargs['max_retries'] = max_retries
        if tensorstore_data_copy_concurrency is not None:
            cli_kwargs['tensorstore_data_copy_concurrency'] = tensorstore_data_copy_concurrency
        if use_threading is not None:
            cli_kwargs['use_threading'] = use_threading
        
        # Add additional kwargs that are CLI parameters (from **kwargs)
        cli_kwargs.update({k: v for k, v in kwargs.items() if k not in ['channel_intensity_limits']})
        
        # Stage 1 triage: CLI > Config (using _collect_params which already does this)
        # Call _collect_params WITH cli_kwargs so CLI params override config defaults
        merged_params = {
            **self._collect_params('cluster', **cli_kwargs),
            **self._collect_params('readers', **cli_kwargs),
            **self._collect_params('conversion', **cli_kwargs),
            **self._collect_params('downscale', **cli_kwargs)
        }
        
        # Capture any extra kwargs that aren't in known config sections
        # These are metadata parameters like y_scale, x_scale, channel_colors, etc.
        extra_kwargs = {key: kwargs[key] for key in kwargs if key not in merged_params}
        
        # Store for reference
        self.cluster_params = self._collect_params('cluster', **cli_kwargs)
        self.readers_params = self._collect_params('readers', **cli_kwargs)
        self.conversion_params = self._collect_params('conversion', **cli_kwargs)
        self.downscale_params = self._collect_params('downscale', **cli_kwargs)
        
        # Pass merged_params to run_conversions
        # Stage 2 triage will happen in converter: CSV > merged_params
        results = run_conversions(os.path.abspath(input_path),
                                  output_path,
                                  includes=includes,
                                  excludes=excludes,
                                  time_tag = time_tag,
                                  channel_tag = channel_tag,
                                  z_tag = z_tag,
                                  y_tag = y_tag,
                                  x_tag = x_tag,
                                  concatenation_axes = concatenation_axes,
                                  **merged_params,
                                  **extra_kwargs
                                  )
        t1 = time.time()
        logger.info(f"Conversion complete for all datasets.")
        logger.info(f"Elapsed for conversion + downscaling: {(t1 - t0) / 60} min.")

    def show_pixel_meta(self,
                        input_path: Union[Path, str],
                        includes=None,
                        excludes=None,
                        series: int = None,
                        output_file: str = None,
                        **kwargs
                        ):
        """
        Display pixel-level and channel metadata for all datasets in input_path.

        For OME-Zarr inputs: Uses fast ThreadPoolExecutor-based metadata extraction (no JVM).
        For other formats: Uses parallel ProcessPoolExecutor-based collection with Bio-Formats.

        Args:
            input_path (Union[Path, str]): Path to input file or directory.
            includes (str, optional): Filename patterns to filter for (comma-separated).
            excludes (str, optional): Filename patterns to filter against (comma-separated).
            series (int, optional): Series index to read. Defaults to configured scene_index.
            output_file (str, optional): Path to save formatted metadata. Auto-detects format:
                                        - .html extension: Save as HTML file with styling
                                        - .txt extension: Save as plain text file
                                        - None: Print to console only
            **kwargs: Additional configuration overrides.

        Prints:
            Formatted table showing axes, shapes, scales, units, and channels for each file.

        Returns:
            None
        """
        import asyncio
        from eubi_bridge.utils.path_utils import take_filepaths
        
        # Check if all input files are OME-Zarr format
        try:
            df = take_filepaths(input_path, **kwargs)
            all_zarr = all(path.endswith('.zarr') for path in df['input_path'])
            zarr_paths = df['input_path'].tolist() if all_zarr else None
        except (ValueError, KeyError):
            all_zarr = False
            zarr_paths = None
        
        # Fast path: OME-Zarr metadata extraction (no JVM, no worker processes)
        if all_zarr:
            logger.info(f"Fast path: Reading metadata from {len(zarr_paths)} OME-Zarr files (no JVM).")
            from eubi_bridge.utils.metadata_utils import read_ome_zarr_metadata_from_collection
            metadata_list = asyncio.run(read_ome_zarr_metadata_from_collection(input_path))
        
        # Slow path: Bio-Formats metadata extraction (requires JVM, uses threading for efficiency)
        else:
            logger.info("Non-Zarr files detected. Initializing JVM for Bio-Formats image reading.")
            # Ensure heavy modules are loaded (scyjava, zarr, dask, etc.)
            _ensure_heavy_imports()
            
            # Initialize JVM for image reading (needed for Bio-Formats)
            from eubi_bridge.utils.jvm_manager import soft_start_jvm
            soft_start_jvm()

            # Get parameters
            self.cluster_params = self._collect_params('cluster', **kwargs)
            self.readers_params = self._collect_params('readers', **kwargs)
            self.conversion_params = self._collect_params('conversion', **kwargs)

            if series is None:
                series = self.readers_params['scene_index']

            # Combine all parameters for workers
            combined_params = {
                **self.cluster_params,
                **self.readers_params,
                **self.conversion_params,
            }
            combined_params['series'] = series
            
            # Force use_threading=True for metadata collection (IO-bound, no need for multiprocessing)
            # This avoids expensive JVM worker process initialization and is much faster
            combined_params['use_threading'] = True
            # Increase workers for metadata-only collection (IO-bound, not CPU-bound)
            combined_params['max_workers'] = min(16, max(8, len(df['input_path'])))

            # Import and run metadata collection
            from eubi_bridge.conversion.converter import run_metadata_collection_from_filepaths

            metadata_list = asyncio.run(run_metadata_collection_from_filepaths(
                input_path,
                includes=includes,
                excludes=excludes,
                **combined_params
            ))

        # Display results with custom formatting
        _console.print("\n[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
        _console.print("[bold cyan]Image Metadata Summary[/bold cyan]")
        _console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]\n")

        def _format_scale_value(value):
            """Format scale value to 3 significant figures."""
            if isinstance(value, (int, float)):
                if value == 0:
                    return "0"
                val_float = float(value)
                if abs(val_float) >= 0.001 and abs(val_float) < 1000:
                    if val_float >= 1:
                        return f"{val_float:.3g}"
                    else:
                        return f"{val_float:.2g}"
                return f"{val_float:.2e}"
            return str(value)

        def _wrap_text(text, width):
            """Wrap text to specified width, preserving lines."""
            if len(text) <= width:
                return text
            lines = []
            for line in text.split('\n'):
                if len(line) <= width:
                    lines.append(line)
                else:
                    # Break long lines at word boundaries
                    words = line.split()
                    current_line = []
                    for word in words:
                        if len(' '.join(current_line + [word])) <= width:
                            current_line.append(word)
                        else:
                            if current_line:
                                lines.append(' '.join(current_line))
                            current_line = [word]
                    if current_line:
                        lines.append(' '.join(current_line))
            return '\n'.join(lines)

        # Process and display each file's metadata
        for idx, metadata in enumerate(metadata_list):
            if idx > 0:
                _console.print("[cyan]" + "─" * 100 + "[/cyan]")  # Separator line
            
            if metadata.get('status') == 'error':
                input_file = metadata.get('input_path', 'Unknown')
                error_msg = metadata.get('error', 'Unknown error')
                _console.print(f"[red bold]File:[/red bold] {Path(input_file).name}")
                _console.print(f"[red]ERROR:[/red] {error_msg}\n")
                continue

            # Extract metadata fields
            input_file = metadata['input_path']
            axes = metadata.get('axes', 'Unknown')
            shape = metadata.get('shape', {})
            scale = metadata.get('scale', {})
            units = metadata.get('units', {})
            dtype = metadata.get('dtype', 'Unknown')
            channels = metadata.get('channels', [])

            # Print file path (full path, wrapped if needed)
            _console.print(f"[magenta bold]File:[/magenta bold] {_wrap_text(input_file, 90)}")

            # Print shape
            shape_lines = []
            for ax in axes:
                ax_display = {'t': 'time', 'c': 'channels', 'z': 'z', 'y': 'y', 'x': 'x'}.get(ax, ax)
                val = shape.get(ax, '?')
                shape_lines.append(f"  {ax_display}: {val}")
            _console.print(f"[yellow bold]Shape:[/yellow bold]\n" + '\n'.join(shape_lines))

            # Print scale and units
            scale_unit_lines = []
            for ax in axes:
                if ax == 'c':  # Skip channel axis
                    continue
                ax_display = {'t': 'time', 'z': 'z', 'y': 'y', 'x': 'x'}.get(ax, ax)
                scale_val = scale.get(ax, '?')
                unit_val = units.get(ax, '')
                
                if scale_val != '?':
                    formatted_scale = _format_scale_value(scale_val)
                    if unit_val:
                        scale_unit_lines.append(f"  {ax_display}: {formatted_scale} {unit_val}")
                    else:
                        scale_unit_lines.append(f"  {ax_display}: {formatted_scale}")
                else:
                    scale_unit_lines.append(f"  {ax_display}: ?")
            
            _console.print(f"[green bold]Scale & Units:[/green bold]\n" + '\n'.join(scale_unit_lines))

            # Print dtype
            _console.print(f"[white bold]Data Type:[/white bold] {dtype}")

            # Print channels
            if channels:
                _console.print(f"[bright_magenta bold]Channels:[/bright_magenta bold]")
                for i, ch in enumerate(channels):
                    label = ch.get('label', f"Channel {i}")
                    color = ch.get('color', None)
                    color_display = color if color else "None"
                    _console.print(f"  [{i}] Label: {_wrap_text(label, 80)}, Color: {color_display}")
            else:
                _console.print(f"[bright_magenta bold]Channels:[/bright_magenta bold] None")
            
            _console.print()  # Blank line between entries

        _console.print("[cyan]" + "═" * 100 + "[/cyan]")  # Final separator
        _console.print()
        
        # Optionally save to file
        if output_file:
            from rich.console import Console as RichConsole
            
            with open(output_file, 'w') as f:
                file_console = RichConsole(file=f, width=100, force_terminal=True, legacy_windows=False)
                
                # Re-render all output to file
                file_console.print("\n[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
                file_console.print("[bold cyan]Image Metadata Summary[/bold cyan]")
                file_console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]\n")
                
                for idx, metadata in enumerate(metadata_list):
                    if idx > 0:
                        file_console.print("[cyan]" + "─" * 100 + "[/cyan]")
                    
                    if metadata.get('status') == 'error':
                        input_file = metadata.get('input_path', 'Unknown')
                        error_msg = metadata.get('error', 'Unknown error')
                        file_console.print(f"[red bold]File:[/red bold] {Path(input_file).name}")
                        file_console.print(f"[red]ERROR:[/red] {error_msg}\n")
                        continue

                    input_file = metadata['input_path']
                    axes = metadata.get('axes', 'Unknown')
                    shape = metadata.get('shape', {})
                    scale = metadata.get('scale', {})
                    units = metadata.get('units', {})
                    dtype = metadata.get('dtype', 'Unknown')
                    channels = metadata.get('channels', [])

                    file_console.print(f"[magenta bold]File:[/magenta bold] {_wrap_text(input_file, 90)}")

                    shape_lines = []
                    for ax in axes:
                        ax_display = {'t': 'time', 'c': 'channels', 'z': 'z', 'y': 'y', 'x': 'x'}.get(ax, ax)
                        val = shape.get(ax, '?')
                        shape_lines.append(f"  {ax_display}: {val}")
                    file_console.print(f"[yellow bold]Shape:[/yellow bold]\n" + '\n'.join(shape_lines))

                    scale_unit_lines = []
                    for ax in axes:
                        if ax == 'c':
                            continue
                        ax_display = {'t': 'time', 'z': 'z', 'y': 'y', 'x': 'x'}.get(ax, ax)
                        scale_val = scale.get(ax, '?')
                        unit_val = units.get(ax, '')
                        
                        if scale_val != '?':
                            formatted_scale = _format_scale_value(scale_val)
                            if unit_val:
                                scale_unit_lines.append(f"  {ax_display}: {formatted_scale} {unit_val}")
                            else:
                                scale_unit_lines.append(f"  {ax_display}: {formatted_scale}")
                        else:
                            scale_unit_lines.append(f"  {ax_display}: ?")
                    
                    file_console.print(f"[green bold]Scale & Units:[/green bold]\n" + '\n'.join(scale_unit_lines))
                    file_console.print(f"[white bold]Data Type:[/white bold] {dtype}")

                    if channels:
                        file_console.print(f"[bright_magenta bold]Channels:[/bright_magenta bold]")
                        for i, ch in enumerate(channels):
                            label = ch.get('label', f"Channel {i}")
                            color = ch.get('color', None)
                            color_display = color if color else "None"
                            file_console.print(f"  [{i}] Label: {_wrap_text(label, 80)}, Color: {color_display}")
                    else:
                        file_console.print(f"[bright_magenta bold]Channels:[/bright_magenta bold] None")
                    
                    file_console.print()
                
                file_console.print("[cyan]" + "═" * 100 + "[/cyan]")
            
            logger.info(f"Metadata saved to {output_file} (text format)")
        
        # Optionally save to HTML
        if output_file and output_file.lower().endswith('.html'):
            from rich.console import Console as RichConsole
            
            with open(output_file, 'w') as f:
                html_console = RichConsole(
                    file=f,
                    width=100,
                    force_terminal=True,
                    legacy_windows=False,
                    record=True
                )
                
                # Re-render all output to HTML console
                html_console.print("\n[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
                html_console.print("[bold cyan]Image Metadata Summary[/bold cyan]")
                html_console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]\n")
                
                for idx, metadata in enumerate(metadata_list):
                    if idx > 0:
                        html_console.print("[cyan]" + "─" * 100 + "[/cyan]")
                    
                    if metadata.get('status') == 'error':
                        input_file = metadata.get('input_path', 'Unknown')
                        error_msg = metadata.get('error', 'Unknown error')
                        html_console.print(f"[red bold]File:[/red bold] {Path(input_file).name}")
                        html_console.print(f"[red]ERROR:[/red] {error_msg}\n")
                        continue

                    input_file = metadata['input_path']
                    axes = metadata.get('axes', 'Unknown')
                    shape = metadata.get('shape', {})
                    scale = metadata.get('scale', {})
                    units = metadata.get('units', {})
                    dtype = metadata.get('dtype', 'Unknown')
                    channels = metadata.get('channels', [])

                    html_console.print(f"[magenta bold]File:[/magenta bold] {_wrap_text(input_file, 90)}")

                    shape_lines = []
                    for ax in axes:
                        ax_display = {'t': 'time', 'c': 'channels', 'z': 'z', 'y': 'y', 'x': 'x'}.get(ax, ax)
                        val = shape.get(ax, '?')
                        shape_lines.append(f"  {ax_display}: {val}")
                    html_console.print(f"[yellow bold]Shape:[/yellow bold]\n" + '\n'.join(shape_lines))

                    scale_unit_lines = []
                    for ax in axes:
                        if ax == 'c':
                            continue
                        ax_display = {'t': 'time', 'z': 'z', 'y': 'y', 'x': 'x'}.get(ax, ax)
                        scale_val = scale.get(ax, '?')
                        unit_val = units.get(ax, '')
                        
                        if scale_val != '?':
                            formatted_scale = _format_scale_value(scale_val)
                            if unit_val:
                                scale_unit_lines.append(f"  {ax_display}: {formatted_scale} {unit_val}")
                            else:
                                scale_unit_lines.append(f"  {ax_display}: {formatted_scale}")
                        else:
                            scale_unit_lines.append(f"  {ax_display}: ?")
                    
                    html_console.print(f"[green bold]Scale & Units:[/green bold]\n" + '\n'.join(scale_unit_lines))
                    html_console.print(f"[white bold]Data Type:[/white bold] {dtype}")

                    if channels:
                        html_console.print(f"[bright_magenta bold]Channels:[/bright_magenta bold]")
                        for i, ch in enumerate(channels):
                            label = ch.get('label', f"Channel {i}")
                            color = ch.get('color', None)
                            color_display = color if color else "None"
                            html_console.print(f"  [{i}] Label: {_wrap_text(label, 80)}, Color: {color_display}")
                    else:
                        html_console.print(f"[bright_magenta bold]Channels:[/bright_magenta bold] None")
                    
                    html_console.print()
                
                html_console.print("[cyan]" + "═" * 100 + "[/cyan]")
                
                # Export to HTML
                html_content = html_console.export_html()
            
            # Write HTML file
            with open(output_file, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Metadata saved to {output_file} (HTML format)")

    def update_pixel_meta(self,
                          input_path: Union[Path, str],
                          includes=None,
                          excludes=None,
                          time_scale: Union[int, float] = None,
                          z_scale: Union[int, float] = None,
                          y_scale: Union[int, float] = None,
                          x_scale: Union[int, float] = None,
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
            Returns:
            None
        """

        # Ensure heavy modules are loaded (scyjava, zarr, dask, etc.)
        _ensure_heavy_imports()
        
        # Initialize JVM for image reading (needs Java-based Bio-Formats)
        from eubi_bridge.utils.jvm_manager import soft_start_jvm
        soft_start_jvm()

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
                          channel_labels: str = '',
                          channel_colors: str = '',
                          channel_intensity_limits = 'from_dtype',
                          includes=None,
                          excludes=None,
                          **kwargs
                          ):
        """
        Updates pixel metadata for image files located at the specified input path.

        Args:
            input_path: Path to input file(s)
            channel_labels: Channel labels in format "idx1,label1;idx2,label2;..." (e.g., "0,Red;1,Green")
            channel_colors: Channel colors in format "idx1,color1;idx2,color2;..." (e.g., "0,FF0000;1,00FF00")
            channel_intensity_limits: 'from_dtype', 'from_array', or 'auto'
            includes: Include file patterns
            excludes: Exclude file patterns
            **kwargs: Additional parameters for cluster and conversion configuration.
        Returns:
            None
        """

        # Ensure heavy modules are loaded (scyjava, zarr, dask, etc.)
        _ensure_heavy_imports()
        
        # Initialize JVM for image reading (needs Java-based Bio-Formats)
        from eubi_bridge.utils.jvm_manager import soft_start_jvm
        soft_start_jvm()

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

        # Prepare channel metadata arguments
        channel_meta_kwargs = {}
        if channel_labels:
            channel_meta_kwargs['channel_labels'] = channel_labels
        if channel_colors:
            channel_meta_kwargs['channel_colors'] = channel_colors

        run_updates(
                    os.path.abspath(input_path),
                    includes=includes,
                    excludes=excludes,
                    **combined,
                    **channel_meta_kwargs,
                    **extra_kwargs
                    )


