import shutil, ctypes, time, os, zarr, pprint, psutil, dask, copy
import numpy as np, os, glob, tempfile
from multiprocessing.pool import ThreadPool

from dask import array as da
from distributed import LocalCluster, Client
from dask_jobqueue import SLURMCluster
from pathlib import Path
from typing import Union

from eubi_bridge.bigtiff_to_zarr.cli import convert_bigtiff_to_omezarr

# from eubi_bridge.ngff.multiscales import Pyramid
# from eubi_bridge.ngff import defaults
from eubi_bridge.base.data_manager import BatchManager
from eubi_bridge.ebridge_base import BridgeBase, downscale
from eubi_bridge.utils.convenience import take_filepaths, is_zarr_group
from eubi_bridge.utils.metadata_utils import print_printable, get_printables
from eubi_bridge.utils.logging_config import get_logger

import logging, warnings

# Set up logger for this module
logger = get_logger(__name__)

# Suppress noisy logs
logging.getLogger('distributed.diskutils').setLevel(logging.CRITICAL)
logging.getLogger('distributed.worker').setLevel(logging.WARNING)
logging.getLogger('distributed.scheduler').setLevel(logging.WARNING)


def soft_start_jvm():
    """Starts the JVM if it is not already running."""
    import scyjava, jpype

    if not scyjava.jvm_started():
        scyjava.config.endpoints.append("ome:formats-gpl:6.7.0")
        scyjava.start_jvm()
    return


def verify_filepaths_for_cluster(filepaths):
    """Verify that all file extensions are supported for distributed processing."""
    logger.info("Verifying file extensions for distributed setup.")
    formats = ['lif', 'czi', 'lsm',
               'nd2',
               'ome.tiff', 'ome.tif',
               'tiff', 'tif', 'zarr',
               'png', 'jpg', 'jpeg']

    for filepath in filepaths:
        verified = any(list(map(lambda path, ext: path.endswith(ext), [filepath] * len(formats), formats)))
        if not verified:
            root, ext = os.path.splitext(filepath)
            logging.warning(f"Distributed execution is not supported for the {ext} format")
            logger.warning(f"Falling back on multithreading.")
            break
    if verified:
        logger.info("File extensions were verified for distributed setup.")
    return verified


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

    def __init__(self,
                 configpath=f"{os.path.expanduser('~')}/.eubi_bridge",
                 ):
        """
        Initializes the EuBIBridge class and loads or sets up default configuration.

        Args:
            configpath (str, optional): Path to store configuration settings. Defaults to the home directory.
        """

        root_dask_defaults = {'distributed.adaptive.interval': '1s', 'distributed.adaptive.maximum': '.inf',
                              'distributed.adaptive.minimum': 0, 'distributed.adaptive.target-duration': '5s',
                              'distributed.adaptive.wait-count': 3, 'distributed.admin.event-loop': 'tornado',
                              'distributed.admin.large-graph-warning-threshold': '10MB',
                              'distributed.admin.log-format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                              'distributed.admin.log-length': 10000, 'distributed.admin.low-level-log-length': 1000,
                              'distributed.admin.max-error-length': 10000, 'distributed.admin.pdb-on-err': False,
                              'distributed.admin.system-monitor.disk': True,
                              'distributed.admin.system-monitor.gil.enabled': True,
                              'distributed.admin.system-monitor.gil.interval': '1ms',
                              'distributed.admin.system-monitor.host-cpu': False,
                              'distributed.admin.system-monitor.interval': '500ms',
                              'distributed.admin.system-monitor.log-length': 7200,
                              'distributed.admin.tick.cycle': '1s', 'distributed.admin.tick.interval': '20ms',
                              'distributed.admin.tick.limit': '3s', 'distributed.client.heartbeat': '5s',
                              'distributed.client.preload': [], 'distributed.client.preload-argv': [],
                              'distributed.client.scheduler-info-interval': '2s',
                              'distributed.client.security-loader': None,
                              'distributed.comm.compression': False, 'distributed.comm.default-scheme': 'tcp',
                              'distributed.comm.offload': '10MiB', 'distributed.comm.require-encryption': None,
                              'distributed.comm.retry.count': 0, 'distributed.comm.retry.delay.max': '20s',
                              'distributed.comm.retry.delay.min': '1s', 'distributed.comm.shard': '64MiB',
                              'distributed.comm.socket-backlog': 2048, 'distributed.comm.timeouts.connect': '30s',
                              'distributed.comm.timeouts.tcp': '30s', 'distributed.comm.tls.ca-file': None,
                              'distributed.comm.tls.ciphers': None, 'distributed.comm.tls.client.cert': None,
                              'distributed.comm.tls.client.key': None, 'distributed.comm.tls.max-version': None,
                              'distributed.comm.tls.min-version': 1.2, 'distributed.comm.tls.scheduler.cert': None,
                              'distributed.comm.tls.scheduler.key': None, 'distributed.comm.tls.worker.cert': None,
                              'distributed.comm.tls.worker.key': None, 'distributed.comm.ucx.create-cuda-context': None,
                              'distributed.comm.ucx.cuda-copy': None, 'distributed.comm.ucx.environment': {},
                              'distributed.comm.ucx.infiniband': None, 'distributed.comm.ucx.nvlink': None,
                              'distributed.comm.ucx.rdmacm': None, 'distributed.comm.ucx.tcp': None,
                              'distributed.comm.websockets.shard': '8MiB', 'distributed.comm.zstd.level': 3,
                              'distributed.comm.zstd.threads': 0, 'distributed.dashboard.export-tool': False,
                              'distributed.dashboard.graph-max-items': 5000,
                              'distributed.dashboard.link': '{scheme}://{host}:{port}/status',
                              'distributed.dashboard.prometheus.namespace': 'dask',
                              'distributed.deploy.cluster-repr-interval': '500ms',
                              'distributed.deploy.lost-worker-timeout': '15s',
                              'distributed.diagnostics.computations.ignore-files': [r'runpy\.py', 'pytest', r'py\.test',
                                                                                    r'pytest-script\.py', '_pytest',
                                                                                    'pycharm',
                                                                                    'vscode_pytest',
                                                                                    r'get_output_via_markers\.py'],
                              'distributed.diagnostics.computations.ignore-modules': ['asyncio', 'functools',
                                                                                      'threading', 'datashader',
                                                                                      'dask', 'debugpy', 'distributed',
                                                                                      'coiled', 'cudf',
                                                                                      'cuml', 'matplotlib', 'pluggy',
                                                                                      'prefect',
                                                                                      'rechunker', 'xarray', 'xgboost',
                                                                                      'xdist',
                                                                                      '__channelexec__', 'execnet'],
                              'distributed.diagnostics.computations.max-history': 100,
                              'distributed.diagnostics.computations.nframes': 0,
                              'distributed.diagnostics.cudf': False,
                              'distributed.diagnostics.erred-tasks.max-history': 100,
                              'distributed.diagnostics.nvml': True, 'distributed.nanny.environ': {},
                              'distributed.nanny.pre-spawn-environ.MALLOC_TRIM_THRESHOLD_': 65536,
                              'distributed.nanny.pre-spawn-environ.MKL_NUM_THREADS': 1,
                              'distributed.nanny.pre-spawn-environ.OMP_NUM_THREADS': 1,
                              'distributed.nanny.pre-spawn-environ.OPENBLAS_NUM_THREADS': 1,
                              'distributed.nanny.preload': [],
                              'distributed.nanny.preload-argv': [], 'distributed.p2p.comm.buffer': '1 GiB',
                              'distributed.p2p.comm.concurrency': 10,
                              'distributed.p2p.comm.message-bytes-limit': '2 MiB',
                              'distributed.p2p.comm.retry.count': 10, 'distributed.p2p.comm.retry.delay.max': '30s',
                              'distributed.p2p.comm.retry.delay.min': '1s', 'distributed.p2p.storage.buffer': '100 MiB',
                              'distributed.p2p.storage.disk': True, 'distributed.p2p.threads': None,
                              'distributed.rmm.pool-size': None,
                              'distributed.scheduler.active-memory-manager.interval': '2s',
                              'distributed.scheduler.active-memory-manager.measure': 'optimistic',
                              'distributed.scheduler.active-memory-manager.policies': [
                                  {'class': 'distributed.active_memory_manager.ReduceReplicas'}],
                              'distributed.scheduler.active-memory-manager.start': True,
                              'distributed.scheduler.allowed-failures': 3,
                              'distributed.scheduler.allowed-imports': ['dask', 'distributed'],
                              'distributed.scheduler.bandwidth': '100000000',
                              'distributed.scheduler.blocked-handlers': [],
                              'distributed.scheduler.contact-address': None,
                              'distributed.scheduler.dashboard.bokeh-application.allow_websocket_origin': ['*'],
                              'distributed.scheduler.dashboard.bokeh-application.check_unused_sessions_milliseconds': 500,
                              'distributed.scheduler.dashboard.bokeh-application.keep_alive_milliseconds': 500,
                              'distributed.scheduler.dashboard.status.task-stream-length': 1000,
                              'distributed.scheduler.dashboard.tasks.task-stream-length': 100000,
                              'distributed.scheduler.dashboard.tls.ca-file': None,
                              'distributed.scheduler.dashboard.tls.cert': None,
                              'distributed.scheduler.dashboard.tls.key': None,
                              'distributed.scheduler.default-data-size': '1kiB',
                              'distributed.scheduler.default-task-durations.rechunk-split': '1us',
                              'distributed.scheduler.default-task-durations.split-shuffle': '1us',
                              'distributed.scheduler.default-task-durations.split-stage': '1us',
                              'distributed.scheduler.default-task-durations.split-taskshuffle': '1us',
                              'distributed.scheduler.events-cleanup-delay': '1h',
                              'distributed.scheduler.http.routes': ['distributed.http.scheduler.prometheus',
                                                                    'distributed.http.scheduler.info',
                                                                    'distributed.http.scheduler.json',
                                                                    'distributed.http.health', 'distributed.http.proxy',
                                                                    'distributed.http.statics'],
                              'distributed.scheduler.idle-timeout': None,
                              'distributed.scheduler.locks.lease-timeout': '30s',
                              'distributed.scheduler.locks.lease-validation-interval': '10s',
                              'distributed.scheduler.no-workers-timeout': None, 'distributed.scheduler.preload': [],
                              'distributed.scheduler.preload-argv': [], 'distributed.scheduler.rootish-taskgroup': 5,
                              'distributed.scheduler.rootish-taskgroup-dependencies': 5,
                              'distributed.scheduler.unknown-task-duration': '500ms',
                              'distributed.scheduler.validate': False,
                              'distributed.scheduler.work-stealing': True,
                              'distributed.scheduler.work-stealing-interval': '1s',
                              'distributed.scheduler.worker-saturation': 1.1,
                              'distributed.scheduler.worker-ttl': '5 minutes',
                              'distributed.version': 2, 'distributed.worker.blocked-handlers': [],
                              'distributed.worker.connections.incoming': 10,
                              'distributed.worker.connections.outgoing': 50,
                              'distributed.worker.daemon': True,
                              'distributed.worker.http.routes': ['distributed.http.worker.prometheus',
                                                                 'distributed.http.health',
                                                                 'distributed.http.statics'],
                              'distributed.worker.lifetime.duration': None,
                              'distributed.worker.lifetime.restart': False,
                              'distributed.worker.lifetime.stagger': '0 seconds',
                              'distributed.worker.memory.max-spill': False,
                              'distributed.worker.memory.monitor-interval': '100ms',
                              'distributed.worker.memory.pause': 0.8,
                              'distributed.worker.memory.rebalance.measure': 'optimistic',
                              'distributed.worker.memory.rebalance.recipient-max': 0.6,
                              'distributed.worker.memory.rebalance.sender-min': 0.3,
                              'distributed.worker.memory.rebalance.sender-recipient-gap': 0.1,
                              'distributed.worker.memory.recent-to-old-time': '30s',
                              'distributed.worker.memory.spill': 0.7,
                              'distributed.worker.memory.spill-compression': 'auto',
                              'distributed.worker.memory.target': 0.6,
                              'distributed.worker.memory.terminate': 0.95, 'distributed.worker.memory.transfer': 0.1,
                              'distributed.worker.multiprocessing-method': 'spawn', 'distributed.worker.preload': [],
                              'distributed.worker.preload-argv': [], 'distributed.worker.profile.cycle': '1000ms',
                              'distributed.worker.profile.enabled': True, 'distributed.worker.profile.interval': '10ms',
                              'distributed.worker.profile.low-level': False, 'distributed.worker.resources': {},
                              'distributed.worker.transfer.message-bytes-limit': '50MB',
                              'distributed.worker.use-file-locking': True,
                              'distributed.worker.validate': False
                              }

        defaults = dict(
            cluster=dict(
                n_jobs=4,
                threads_per_worker=1,
                memory_limit='auto',
                temp_dir='auto',
                no_worker_restart=False,
                verbose=False,
                no_distributed=False,
                on_slurm=False,
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
                use_bioformats_readers=False
            ),
            conversion=dict(
                zarr_format=2,
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
                use_tensorstore=False,
                use_gpu=False,
                rechunk_method='tasks',
                trim_memory=False,
                metadata_reader='bfio',
                save_omexml=True,
                squeeze=False,
                dtype=None
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

        self.root_defaults = defaults
        self.root_dask_defaults = root_dask_defaults
        config_gr = zarr.open_group(configpath, mode='a')
        config = config_gr.attrs
        for key in defaults.keys():
            if key not in config.keys():
                config[key] = {}
                for subkey in defaults[key].keys():
                    if subkey not in config[key].keys():
                        config[key][subkey] = defaults[key][subkey]
            config_gr.attrs[key] = config[key]
        self.config = dict(config_gr.attrs)
        ###
        if not 'dask_config' in config_gr.keys():
            config_gr.create_group('dask_config')
        dask_config = config_gr['dask_config'].attrs
        for key in root_dask_defaults.keys():
            if key not in dask_config.keys():
                dask_config[key] = root_dask_defaults[key]
        config_gr['dask_config'].attrs.update(dict(dask_config))
        self.dask_config = dict(config_gr['dask_config'].attrs)
        self.config_gr = config_gr
        ###
        self._dask_temp_dir = None
        self.client = None

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

            # Memory management - adaptive based on system memory
            'distributed.worker.memory.target': memory_target,
            'distributed.worker.memory.spill': memory_spill,
            'distributed.worker.memory.pause': memory_pause,
            'distributed.worker.memory.terminate': 0.98,  # Terminate at 98%
            'distributed.worker.memory.monitor-interval': '50ms' if total_memory < 32 * 1024 ** 3 else '100ms',
            'distributed.worker.memory.recent-to-old-time': '3s',  # Faster memory cleanup

            # Communication settings
            'distributed.comm.compression': 'auto',  # Auto compression for large data
            'distributed.comm.retry.count': 2,  # Fewer retries for speed
            'distributed.comm.timeouts.connect': '60s',
            'distributed.comm.timeouts.tcp': '120s',
            'distributed.comm.shard': '64MiB' if total_memory > 64 * 1024 ** 3 else '32MiB',
            'distributed.comm.offload': '2GiB' if total_memory > 128 * 1024 ** 3 else '1GiB',

            # Task scheduling
            'distributed.scheduler.work-stealing': True,
            'distributed.scheduler.work-stealing-interval': '5ms' if total_cores > 8 else '10ms',
            'distributed.scheduler.bandwidth': 5e9,  # 5GB/s network bandwidth
            'distributed.scheduler.default-task-durations': {
                'rechunk-split': '1ms',
                'rechunk-merge': '1ms',
                'from-delayed': '1ms'
            },

            # Worker settings
            'distributed.worker.profile.enabled': False,  # Disable profiling for speed
            'distributed.worker.threads': min(4, max(2, total_cores // 4)),  # Dynamic thread count
            'distributed.worker.memory.rebalance.measure': 'optimistic',
            'distributed.worker.memory.rebalance.recipient-max': 0.8,
            'distributed.worker.memory.rebalance.sender-min': 0.3,  # More aggressive rebalancing

            # Client settings
            'distributed.client.heartbeat': '10s',  # Less frequent heartbeats
            'distributed.client.scheduler-info-interval': '5s',

            # Compression
            # 'distributed.comm.compression': 'lz4',  # Faster compression
            'distributed.comm.zstd.level': 1,  # Faster compression level

            # I/O optimization
            'distributed.worker.use-file-locking': False,  # Disable if NFS not used
            # 'distributed.worker.memory.spill-compression': 'lz4',  # Explicitly use lz4 for spilling
        })

    def _get_optimal_worker_config(self, n_jobs=None, threads_per_worker=None,
                                   memory_limit=None, **kwargs):
        """Calculate optimal worker configuration for conversion speed."""
        import psutil
        import math

        # Get system information
        total_cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 4
        total_memory = psutil.virtual_memory().total

        # Optimize for conversion speed
        if threads_per_worker is None:
            # For I/O-bound tasks, use more threads per worker
            threads_per_worker = min(8, max(2, (total_cores // 2) or 1))

        if n_jobs is None:
            # Leave 2 cores for system and I/O
            n_workers = max(1, min((total_cores - 2) // max(1, threads_per_worker), 32))
        else:
            n_workers = max(1, n_jobs)

        # Memory allocation
        if memory_limit is None:
            # Reserve 10% of memory for system
            reserved_memory = total_memory * 0.1
            memory_per_worker = (total_memory - reserved_memory) / n_workers
            memory_limit = f"{memory_per_worker / (1024 ** 3):.1f}GB"

        # Adjust for conversion tasks
        if isinstance(memory_limit, str) and 'GB' in memory_limit:
            gb = float(memory_limit.replace('GB', ''))
            # Ensure minimum memory per worker
            memory_limit = f"{max(2.0, gb)}GB"  # At least 2GB per worker

        return {
            'n_workers': n_workers,
            'threads_per_worker': threads_per_worker,
            'memory_limit': memory_limit,
            # 'processes': True  # Use processes for CPU-bound work
        }

    def reset_config(self):
        """
        Resets the cluster, conversion and downscale parameters to the installation defaults.
        """
        self.config_gr.attrs.update(self.root_defaults)
        self.config = dict(self.config_gr.attrs)

    def reset_dask_config(self):
        """
        Resets the dask configuration parameters to the installation defaults.
        """
        self.config_gr['dask_config'].attrs.update(self.root_dask_defaults)
        self.dask_config = dict(self.config_gr['dask_config'].attrs)

    def show_config(self):
        """
        Displays the current cluster, conversion, and downscale parameters.
        """
        pprint.pprint(self.config)

    def show_dask_config(self):
        """
        Displays the current dask.distributed parameters.
        """
        pprint.pprint(self.dask_config)

    def show_root_defaults(self):
        """
        Displays the installation defaults for cluster, conversion, and downscale parameters.
        """
        pprint.pprint(self.root_defaults)

    def show_root_dask_defaults(self):
        """
        Displays the installation defaults for dask.distributed.
        """
        pprint.pprint(self.root_dask_defaults)

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
        return params

    def configure_cluster(self,
                          memory_limit: str = 'default',
                          n_jobs: int = 'default',
                          no_worker_restart: bool = 'default',
                          on_slurm: bool = 'default',
                          temp_dir: str = 'default',
                          threads_per_worker: int = 'default',
                          no_distributed: bool = 'default',
                          verbose: bool = 'default'
                          ):
        """
        Updates cluster configuration settings. To update the current default value for a parameter, provide that parameter with a value other than 'default'.

        The following parameters can be configured:
            - memory_limit (str, optional): Memory limit per worker.
            - n_jobs (int, optional): Number of parallel jobs.
            - no_worker_restart (bool, optional): Whether to prevent worker restarts.
            - on_slurm (bool, optional): Whether running on a SLURM cluster.
            - temp_dir (str, optional): Temporary directory for Dask workers.
            - threads_per_worker (int, optional): Number of threads per worker.
            - verbose (bool, optional): Enables detailed logging.

        Args:
            memory_limit (str, optional): Memory limit per worker.
            n_jobs (int, optional): Number of parallel jobs.
            no_worker_restart (bool, optional): Whether to prevent worker restarts.
            on_slurm (bool, optional): Whether running on a SLURM cluster.
            temp_dir (str, optional): Temporary directory for Dask workers.
            threads_per_worker (int, optional): Number of threads per worker.
            verbose (bool, optional): Enables detailed logging.

        Returns:
            None
        """

        params = {
            'memory_limit': memory_limit,
            'n_jobs': n_jobs,
            'no_worker_restart': no_worker_restart,
            'on_slurm': on_slurm,
            'temp_dir': temp_dir,
            'threads_per_worker': threads_per_worker,
            'no_distributed': no_distributed,
            'verbose': verbose
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
                          use_bioformats_readers: bool = 'default'
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
                             auto_chunk: bool = 'default',
                             target_chunk_mb: float = 'default',
                             compressor: str = 'default',
                             compressor_params: dict = 'default',
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
                             dimension_order: str = 'default',
                             overwrite: bool = 'default',
                             rechunk_method: str = 'default',
                             # rechunkers_max_mem: str = 'default',
                             trim_memory: bool = 'default',
                             use_tensorstore: bool = 'default',
                             use_gpu: bool = 'default',
                             metadata_reader: str = 'default',
                             save_omexml: bool = 'default',
                             squeeze: bool = 'default',
                             dtype: str = 'default',
                             ):
        """
        Updates conversion configuration settings. To update the current default value for a parameter, provide that parameter with a value other than 'default'.

        The following parameters can be configured:
            - compressor (str, optional): Compression algorithm.
            - compressor_params (dict, optional): Parameters for the compressor.
            - output_chunks (list, optional): Chunk size for output.
            - overwrite (bool, optional): Whether to overwrite existing data.
            - rechunk_method (str, optional): Method used for rechunking.
            - trim_memory (bool, optional): Whether to trim memory usage.
            - use_tensorstore (bool, optional): Whether to use TensorStore for writing.
            - save_omexml (bool, optional): Whether to create a METADATA.ome.xml file.
        Args:
            compressor (str, optional): Compression algorithm.
            compressor_params (dict, optional): Parameters for the compressor.
            output_chunks (list, optional): Chunk size for output.
            overwrite (bool, optional): Whether to overwrite existing data.
            rechunk_method (str, optional): Method used for rechunking.
            trim_memory (bool, optional): Whether to trim memory usage.
            use_tensorstore (bool, optional): Whether to use TensorStore for storage.
            save_omexml (bool, optional): Whether to create a METADATA.ome.xml file.

        Returns:
            None
        """

        params = {
            'zarr_format': zarr_format,
            'auto_chunk': auto_chunk,
            'target_chunk_mb': target_chunk_mb,
            'compressor': compressor,
            'compressor_params': compressor_params,
            "time_chunk": time_chunk,
            "channel_chunk": channel_chunk,
            "z_chunk": z_chunk,
            "y_chunk": y_chunk,
            "x_chunk": x_chunk,
            "time_shard_coef": time_shard_coef,
            "channel_shard_coef": channel_shard_coef,
            "z_shard_coef": z_shard_coef,
            "y_shard_coef": y_shard_coef,
            "x_shard_coef": x_shard_coef,
            "time_range": time_range,
            "channel_range": channel_range,
            "z_range": z_range,
            "y_range": y_range,
            "x_range": x_range,
            "dimension_order": dimension_order,
            'overwrite': overwrite,
            'rechunk_method': rechunk_method,
            # 'rechunkers_max_mem': rechunkers_max_mem,
            'trim_memory': trim_memory,
            'use_tensorstore': use_tensorstore,
            'use_gpu': use_gpu,
            'metadata_reader': metadata_reader,
            'save_omexml': save_omexml,
            'squeeze': squeeze,
            'dtype': dtype
        }

        for key in params:
            if key in self.config['conversion'].keys():
                if params[key] != 'default':
                    self.config['conversion'][key] = params[key]
        self.config_gr.attrs['conversion'] = self.config['conversion']

    def configure_downscale(self,
                            downscale_method: str = 'default',
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
            'downscale_method': downscale_method,
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

    def _set_dask_temp_dir(self, temp_dir='auto'):
        if self._dask_temp_dir is not None:
            self._dask_temp_dir.cleanup()
        if temp_dir in ('auto', None):
            temp_dir = tempfile.TemporaryDirectory()
        else:
            os.makedirs(temp_dir, exist_ok=True)
            temp_dir = tempfile.TemporaryDirectory(dir=temp_dir)
        self._dask_temp_dir = temp_dir
        return self

    def _start_cluster(self,
                       n_jobs: int = 4,
                       threads_per_worker: int = 1,
                       memory_limit: str = 'auto',
                       temp_dir='auto',
                       verbose=False,
                       on_slurm=False,
                       no_distributed=False,
                       config_kwargs={},
                       **kwargs
                       ):

        config_dict = copy.deepcopy(self.dask_config)
        config_dict.update(**config_kwargs)

        # worker_options = {
        #     "memory_target_fraction": 0.8,
        #     "memory_spill_fraction": 0.9,
        #     "memory_pause_fraction": 0.95,
        #     # "memory_terminate_fraction": 0.98
        # }
        scheduler_options = {
            "allowed_failures": 100,
            "idle_timeout": "1h",
            "worker_ttl": "1d"  # Set to a large value, e.g., 1 day
        }

        self._set_dask_temp_dir(temp_dir)

        dask.config.set(config_dict)  # use dictionary notation here.

        if no_distributed:
            config_dict.update(scheduler='threads',
                               pool=ThreadPool(n_jobs)
                               )
            dask.config.set(config_dict)
            logger.info(f"Process running locally via multithreading.")
        else:
            self._optimize_dask_config()
            if memory_limit == 'auto':
                reserve_fraction = kwargs.get('reserve_memory_fraction', 0.1)
                min_per_worker = kwargs.get('min_memory_per_worker', 1 * 1024 ** 3)

                total_mem = psutil.virtual_memory().total
                reserved_mem = total_mem * reserve_fraction
                available_mem = max(0, total_mem - reserved_mem)
                mem_per_worker = max(available_mem / n_jobs, min_per_worker)
                mem_gb = mem_per_worker / (1 * 1024 ** 3)
                memory_limit = f"{mem_gb} GB"
                logger.info(f"{memory_limit} memory has been allocated per worker.")

            if on_slurm:
                logger.info(f"Process running on Slurm.")
                cluster = SLURMCluster(
                    cores=threads_per_worker,
                    processes=1,
                    nanny=False,
                    scheduler_options=scheduler_options,
                    n_workers=n_jobs,
                    memory=memory_limit,
                    local_directory=f"{self._dask_temp_dir.name}",
                    # **worker_options
                )
            else:
                logger.info(f"Process running on local cluster.")
                cluster = LocalCluster(
                    n_workers=n_jobs,
                    threads_per_worker=threads_per_worker,
                    nanny=False,
                    scheduler_kwargs=scheduler_options,
                    memory_limit=memory_limit,
                    local_directory=f"{self._dask_temp_dir.name}",
                    # **worker_options
                )
            cluster.scale(n_jobs)
            self.client = Client(cluster, asynchronous=False)
            if verbose:
                logger.info(self.client.cluster)
        return self

    # def _start_cluster(
    #         self,
    #         n_jobs: int = 4,
    #         threads_per_worker: int = 1,
    #         memory_limit: str = 'auto',
    #         temp_dir: str = 'auto',
    #         verbose: bool = False,
    #         on_slurm: bool = False,
    #         no_distributed: bool = False,
    #         config_kwargs: dict = None,
    #         **kwargs
    # ):
    #     """Start a Dask cluster with optimized configuration.
    #
    #     Args:
    #         n_jobs: Number of worker processes
    #         threads_per_worker: Number of threads per worker
    #         memory_limit: Memory limit per worker ('auto' or string like '4GB')
    #         temp_dir: Directory for temporary files
    #         verbose: Enable verbose logging
    #         on_slurm: Whether running on a SLURM cluster
    #         no_distributed: Use simple threading backend instead of distributed
    #         config_kwargs: Additional Dask configuration
    #         **kwargs: Additional arguments including:
    #             - reserve_memory_fraction: Fraction of memory to reserve (default: 0.1)
    #             - min_memory_per_worker: Minimum memory per worker in bytes (default: 1GB)
    #     """
    #     # Initialize config with defaults
    #     config_kwargs = config_kwargs or {}
    #     scheduler_options = {
    #         "allowed_failures": 100,
    #         "idle_timeout": "1h",
    #         "worker_ttl": "1d"
    #     }
    #
    #     # Set up temp directory
    #     self._set_dask_temp_dir(temp_dir)
    #
    #     # Configure Dask
    #     if no_distributed:
    #         # Simple threading backend
    #         config = {
    #             'scheduler': 'threads',
    #             'pool': ThreadPool(n_jobs)
    #         }
    #         config.update(config_kwargs)
    #         dask.config.set(config)
    #         logger.info("Process running locally via multithreading")
    #         return self
    #
    #     # Distributed cluster setup
    #     if memory_limit == 'auto':
    #         memory_limit = self._calculate_memory_limit(
    #             n_jobs,
    #             kwargs.get('reserve_memory_fraction', 0.1),
    #             kwargs.get('min_memory_per_worker', 1 * 1024 ** 3)
    #         )
    #         logger.info(f"Allocated {memory_limit} memory per worker")
    #
    #     try:
    #         if on_slurm:
    #             logger.info("Initializing SLURM cluster")
    #             cluster = self._create_slurm_cluster(
    #                 n_jobs, threads_per_worker, memory_limit, scheduler_options
    #             )
    #         else:
    #             logger.info("Initializing local cluster")
    #             cluster = self._create_local_cluster(
    #                 n_jobs, threads_per_worker, memory_limit, scheduler_options
    #             )
    #
    #         cluster.scale(n_jobs)
    #         self.client = Client(cluster)
    #
    #         if verbose:
    #             logger.info("Cluster info: %s", self.client.cluster)
    #
    #         return self
    #
    #     except Exception as e:
    #         logger.error("Failed to start cluster: %s", str(e))
    #         if hasattr(self, 'client'):
    #             self.client.close()
    #         raise
    #
    # def _calculate_memory_limit(self, n_jobs, reserve_fraction, min_per_worker):
    #     """Calculate memory limit per worker."""
    #     total_mem = psutil.virtual_memory().total
    #     reserved_mem = total_mem * reserve_fraction
    #     available_mem = max(0, total_mem - reserved_mem)
    #     mem_per_worker = max(available_mem / n_jobs, min_per_worker)
    #     return f"{mem_per_worker / (1024 ** 3):.1f}GB"
    #
    # def _create_slurm_cluster(self, n_jobs, threads_per_worker, memory_limit, scheduler_options):
    #     """Create a SLURM cluster."""
    #     return SLURMCluster(
    #         cores=threads_per_worker,
    #         processes=1,
    #         nanny=False,
    #         scheduler_options=scheduler_options,
    #         n_workers=n_jobs,
    #         memory=memory_limit,
    #         local_directory=str(self._dask_temp_dir.name),
    #     )
    #
    # def _create_local_cluster(self, n_jobs, threads_per_worker, memory_limit, scheduler_options):
    #     """Create a local cluster."""
    #     return LocalCluster(
    #         n_workers=n_jobs,
    #         threads_per_worker=threads_per_worker,
    #         nanny=False,
    #         scheduler_kwargs=scheduler_options,
    #         memory_limit=memory_limit,
    #         local_directory=str(self._dask_temp_dir.name),
    #     )

    def tiff_to_zarr(self,
                     input_tiff,
                     output_zarr_dir,
                     **kwargs
                     # dtype=None,
                     # compressor="blosc",
                     # x_scale=2,
                     # y_scale=2,
                     # z_scale=2,
                     # time_scale=1,
                     # channel_scale=1,
                     # min_dimension_size=64,
                     # n_layers=None,
                     # auto_chunk=False,
                     # overwrite=False,
                     # save_omexml=False,
                     # zarr_format=2
                     ):
        """Convert a BigTIFF to OME-Zarr format with optional downscaling.

        Args:
            input_tiff: Path to input TIFF (BigTIFF)
            output_zarr_dir: Path to output directory (will be treated as zarr store root)
            dtype: Optional target dtype (e.g. "uint8"). If None, source dtype is used.
            compressor_name: Compression method ("blosc" or "none")
            x_scale: Downscale factor in X (select every n-th pixel)
            y_scale: Downscale factor in Y
            z_scale: Downscale factor in Z
            time_scale: Downscale factor in Time
            channel_scale: Downscale factor in Channel
            min_dimension_size: Stop building pyramid when smallest dimension < this
            n_layers: Max number of pyramid levels (None = unlimited until min size)
            auto_chunk: Let zarr choose chunking if True
            overwrite: Overwrite output directory if exists
            save_omexml: Attempt to copy/save OME-XML metadata if present
            zarr_format: Zarr format version (2 or 3)
        """
        # from eubi_bridge.bigtiff_to_omezarr import convert_bigtiff_to_omezarr
        self.cluster_params = self._collect_params('cluster', **kwargs)
        self.readers_params = self._collect_params('readers', **kwargs)
        self.conversion_params = self._collect_params('conversion', **kwargs)
        self.downscale_params = self._collect_params('downscale', **kwargs)

        print(self.conversion_params)
        convert_bigtiff_to_omezarr(input_tiff=input_tiff,
                                   output_zarr_dir=output_zarr_dir,
                                   # dimension_order='tczyx',
                                   **self.cluster_params,
                                   **self.conversion_params,
                                   **self.downscale_params
                                   )

    def to_zarr(self,
                input_path: Union[Path, str],
                output_path: Union[Path, str],
                includes=None,
                excludes=None,
                time_tag: Union[str, tuple] = None,
                channel_tag: Union[str, tuple] = None,
                z_tag: Union[str, tuple] = None,
                y_tag: Union[str, tuple] = None,
                x_tag: Union[str, tuple] = None,
                concatenation_axes: Union[int, tuple, str] = None,
                **kwargs
                ):
        """
        Converts image data to OME-Zarr format and optionally applies downscaling.

        Args:
            input_path (Union[Path, str]): Path to input file or directory.
            output_path (Union[Path, str]): Directory, in which the output OME-Zarrs will be written.
            includes (str, optional): Filename patterns to filter for.
            excludes (str, optional): Filename patterns to filter against.
            time_tag (Union[str, tuple], optional): Time dimension tag.
            channel_tag (Union[str, tuple], optional): Channel dimension tag.
            z_tag (Union[str, tuple], optional): Z dimension tag.
            y_tag (Union[str, tuple], optional): Y dimension tag.
            x_tag (Union[str, tuple], optional): X dimension tag.
            concatenation_axes (Union[int, tuple, str], optional): Axes, along which the images will be concatenated.
            **kwargs: Additional configuration overrides.

        Raises:
            Exception: If no files are found in the input path.

        Prints:
            Process logs including conversion and downscaling time.

        Returns:
            None
        """
        t0 = time.time()
        # Get parameters:
        self.cluster_params = self._collect_params('cluster', **kwargs)
        self.readers_params = self._collect_params('readers', **kwargs)
        self.conversion_params = self._collect_params('conversion', **kwargs)
        self.downscale_params = self._collect_params('downscale', **kwargs)

        if self.conversion_params['use_gpu'] and self.conversion_params['use_tensorstore']:
            raise ValueError("Tensorstore is not supported for GPU arrays.")

        logger.info(f"Base conversion initiated.")
        ###### Handle input data and metadata
        paths = take_filepaths(input_path, includes=includes, excludes=excludes)

        filepaths = sorted(list(paths))

        ### If a single tiff file, consider special conversion !!!!! TODO
        # self.convert_single_tiff()

        ###### Start the cluster
        verified_for_cluster = verify_filepaths_for_cluster(
            filepaths)  ### Ensure non-bioformats conversion. If bioformats is needed, fall back on local conversion.
        if not verified_for_cluster or self.readers_params['use_bioformats_readers']:
            self.cluster_params['no_distributed'] = True
            # IMPORTANT: If we are here, then bioformats will be needed
            # Then, jvm must be started before importing bioio_bioformats readers
            soft_start_jvm()

        cluster_is_true = not self.cluster_params['no_distributed']

        if cluster_is_true:
            chunks_yx = None
        else:
            chunks_yx = tuple([self.conversion_params['y_chunk'],  # chunks already when reading
                               self.conversion_params['x_chunk']]
                              )

        # self._optimize_dask_config()

        self._start_cluster(**self.cluster_params)

        series = self.readers_params['scene_index']

        ###### Read and concatenate
        base = BridgeBase(input_path,
                          excludes=excludes,
                          includes=includes,
                          series=series,
                          zarr_format=self.conversion_params['zarr_format'],
                          verbose=self.cluster_params['verbose']
                          )

        base.read_dataset(verified_for_cluster=cluster_is_true,
                          chunks_yx=chunks_yx,
                          readers_params=self.readers_params
                          )

        base.digest(
            time_tag=time_tag,
            channel_tag=channel_tag,
            z_tag=z_tag,
            y_tag=y_tag,
            x_tag=x_tag,
            axes_of_concatenation=concatenation_axes,
            # metadata_reader = self.conversion_params['metadata_reader'],
            **kwargs
        )
        logger.info(f"Metadata was extracted")
        verbose = base._verbose

        if 'region_shape' in kwargs.keys():
            self.conversion_params['region_shape'] = kwargs.get('region_shape')
        if verbose:
            print(f"Cluster params:")
            pprint.pprint(self.cluster_params)
            print(f"Readers params:")
            pprint.pprint(self.readers_params)
            print(f"Conversion params:")
            pprint.pprint(self.conversion_params)
            print(f"Downscale params:")
            pprint.pprint(self.downscale_params)

        temp_dir = base._dask_temp_dir
        self.conversion_params['temp_dir'] = temp_dir
        self.downscale_params['temp_dir'] = temp_dir

        if self.client is not None:
            base.client = self.client
        base.set_dask_temp_dir(self._dask_temp_dir)

        ###### Write
        self.base_results = base.write_arrays(output_path,
                                              compute=True,
                                              verbose=verbose,
                                              **self.conversion_params
                                              )
        ###### Downscale
        logger.info(f"Base conversion finished.")
        t1 = time.time()
        logger.info(f"Elapsed for base conversion: {(t1 - t0) / 60} min.")
        n_layers = self.downscale_params['n_layers']

        if n_layers in (None, 'default', 'auto') or n_layers > 1:
            logger.info(f"Downscaling initiated.")
            _ = downscale(
                self.base_results,
                **self.downscale_params,
                auto_chunk=kwargs.get('auto_chunk', self.conversion_params['auto_chunk']),
                target_chunk_mb=kwargs.get('target_chunk_mb', self.conversion_params['target_chunk_mb']),
                zarr_format=self.conversion_params['zarr_format'],
                rechunk_method=self.conversion_params['rechunk_method'],
                use_tensorstore=self.conversion_params['use_tensorstore'],
                compressor=self.conversion_params['compressor'],
                compressor_params=self.conversion_params['compressor_params'],
                verbose=verbose
            )  # TODO: add to_cupy parameter here.

            logger.info(f"Downscaling finished.")

        ###### Shutdown and clean up
        if self.client is not None:
            self.client.shutdown()
            self.client.close()

        if isinstance(self._dask_temp_dir, tempfile.TemporaryDirectory):
            shutil.rmtree(self._dask_temp_dir.name)
        else:
            shutil.rmtree(self._dask_temp_dir)

        t1 = time.time()
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

        # Collect file paths based on inclusion and exclusion patterns
        paths = take_filepaths(input_path, includes=includes, excludes=excludes)

        filepaths = sorted(list(paths))

        # Verify file paths for cluster compatibility
        verified_for_cluster = verify_filepaths_for_cluster(filepaths)
        if not verified_for_cluster:
            self.cluster_params['no_distributed'] = True

        cluster_is_true = not self.cluster_params['no_distributed']

        # Start the processing cluster
        self._start_cluster(**self.cluster_params)

        series = self.readers_params['scene_index']

        base = BridgeBase(input_path,
                          excludes=excludes,
                          includes=includes,
                          series=series
                          )

        # Read and digest the dataset
        base.read_dataset(verified_for_cluster,
                          readers_params=self.readers_params
                          )

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

        base.digest(**pixel_meta_kwargs)
        logger.info(f"Metadata was extracted")

        if self.client is not None:
            base.client = self.client
        base.set_dask_temp_dir(self._dask_temp_dir)

        # Update metadata for each dataset manager
        for path, manager in base.batchdata.managers.items():
            if is_zarr_group(manager.path):
                manager.sync_pyramid(self.conversion_params['save_omexml'])
            else:
                logger.info(f"Cannot update metadata for non-zarr path: {path}")

        # Shutdown the cluster and clean up temporary directories
        if self.client is not None:
            self.client.shutdown()
            self.client.close()

        if isinstance(self._dask_temp_dir, tempfile.TemporaryDirectory):
            shutil.rmtree(self._dask_temp_dir.name)
        else:
            shutil.rmtree(self._dask_temp_dir)



