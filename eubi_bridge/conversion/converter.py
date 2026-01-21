import asyncio
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import s3fs
from natsort import natsorted

from eubi_bridge.conversion.aggregative_conversion_base import \
    AggregativeConverter
from eubi_bridge.conversion.conversion_worker import (aggregative_worker_sync,
                                                      unary_worker_sync)
from eubi_bridge.conversion.worker_init import initialize_worker_process
from eubi_bridge.utils.array_utils import autocompute_chunk_shape
from eubi_bridge.utils.jvm_manager import soft_start_jvm
from eubi_bridge.utils.logging_config import get_logger
from eubi_bridge.utils.path_utils import (is_zarr_array, is_zarr_group,
                                          sensitive_glob, take_filepaths)

logger = get_logger(__name__)


def _warmup_worker():
    """Dummy function to trigger worker initialization without doing any work."""
    return None


async def run_worker_with_retries(loop, pool, worker_func, *args, max_retries=3, base_delay=0.5, task_id=None):
    """
    Execute a worker function with automatic retry on transient failures.
    
    Retries only on BrokenProcessPool exceptions (worker crashes). Other exceptions
    fail immediately without retry.
    
    Args:
        loop: asyncio event loop
        pool: Executor (ProcessPoolExecutor or ThreadPoolExecutor)
        worker_func: Function to execute in worker
        *args: Arguments to pass to worker_func
        max_retries: Maximum number of attempts (default: 3 = 1 initial + 2 retries)
        base_delay: Initial delay in seconds before first retry (default: 0.5)
        task_id: Optional task identifier for logging
    
    Returns:
        Result from worker_func
        
    Raises:
        Exception: Original exception if all retries exhausted
    """
    from concurrent.futures import BrokenProcessPool
    
    task_label = f"Task {task_id}" if task_id is not None else "Task"
    
    for attempt in range(1, max_retries + 1):
        try:
            return await loop.run_in_executor(pool, worker_func, *args)
        except BrokenProcessPool as e:
            if attempt == max_retries:
                logger.error(f"{task_label} failed after {max_retries} attempts: {e}")
                raise
            
            # Calculate exponential backoff delay
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                f"{task_label} attempt {attempt}/{max_retries} failed with BrokenProcessPool. "
                f"Retrying in {delay:.1f}s... Error: {str(e)[:100]}"
            )
            await asyncio.sleep(delay)
    
    # Should never reach here, but for safety
    raise RuntimeError(f"{task_label} exhausted all {max_retries} attempts")


async def run_metadata_collection_from_filepaths(
        input_path,
        **global_kwargs
):
    """
    Collect metadata from image files in parallel.
    
    Similar to run_conversions_from_filepaths but collects and returns
    metadata information instead of converting files.
    
    Args:
        input_path: Path to files or directory or CSV/XLSX file
        **global_kwargs: Global configuration parameters (supports use_threading flag)
    
    Returns:
        List of metadata dictionaries for each file
    """
    df = take_filepaths(input_path, **global_kwargs)

    # --- Setup concurrency ---
    max_workers = int(global_kwargs.get("max_workers", 4))
    use_threading = global_kwargs.get("use_threading", False)
    max_retries = int(global_kwargs.get("max_retries", 3))

    # Import worker function
    from eubi_bridge.conversion.conversion_worker import metadata_reader_sync

    # Choose executor based on use_threading flag
    if use_threading:
        logger.info("Using ThreadPoolExecutor for metadata collection (threading mode)")
        executor_class = ThreadPoolExecutor
        executor_kwargs = {"max_workers": max_workers}
    else:
        logger.info("Using ProcessPoolExecutor for metadata collection (multiprocessing mode)")
        ctx = mp.get_context("spawn")
        executor_class = ProcessPoolExecutor
        executor_kwargs = {
            "max_workers": max_workers,
            "mp_context": ctx,
            "initializer": initialize_worker_process
        }

    with executor_class(**executor_kwargs) as pool:
        loop = asyncio.get_running_loop()
        
        # NOTE: For ProcessPoolExecutor: Submit tasks with staggered timing to serialize
        # worker initialization and prevent race conditions in JVM/native libraries.
        # For ThreadPoolExecutor: Submit all at once (no race conditions, lighter weight).

        tasks = []
        for idx, row in df.iterrows():
            job_kwargs = row.to_dict()
            input_path_job = job_kwargs.pop('input_path')

            # Submit to executor with retry wrapper
            task = run_worker_with_retries(
                loop,
                pool,
                metadata_reader_sync,
                input_path_job,
                job_kwargs,
                max_retries=max_retries,
                task_id=idx
            )
            tasks.append(task)
            
            # For ProcessPoolExecutor: stagger submissions to serialize worker initialization
            # This prevents simultaneous JVM/native library initialization race conditions
            if not use_threading and idx < len(df) - 1:
                await asyncio.sleep(0.2)

        # Gather with return_exceptions to see all failures
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"[Main] Task {i} failed: {result}")
            elif isinstance(result, dict) and result.get('status') == 'error':
                logger.error(f"[Main] Task {i} failed: {result.get('error')}")

    return results




async def run_conversions_from_filepaths(
        input_path,
        **global_kwargs
):
    """
    Run parallel conversions where each job's parameters (including input/output dirs)
    are specified via kwargs or a CSV/XLSX file.

    Supports both multiprocessing (default) and threading modes via use_threading flag.

    Args:
        input_path:
            - list of file paths, OR
            - path to a CSV/XLSX with at least 'input_path' or 'filepath' column.
        **global_kwargs: global defaults for all conversions (supports use_threading flag)
    """
    df = take_filepaths(input_path, **global_kwargs)

    # --- Setup concurrency ---
    max_workers = int(global_kwargs.get("max_workers", 4))
    use_threading = global_kwargs.get("use_threading", False)
    max_retries = int(global_kwargs.get("max_retries", 3))

    # Import worker function
    from eubi_bridge.conversion.conversion_worker import unary_worker_sync

    # Choose executor based on use_threading flag
    if use_threading:
        logger.info("Using ThreadPoolExecutor for conversions (threading mode)")
        executor_class = ThreadPoolExecutor
        executor_kwargs = {"max_workers": max_workers}
    else:
        logger.info("Using ProcessPoolExecutor for conversions (multiprocessing mode)")
        ctx = mp.get_context("spawn")
        executor_class = ProcessPoolExecutor
        executor_kwargs = {
            "max_workers": max_workers,
            "mp_context": ctx,
            "initializer": initialize_worker_process
        }

    with executor_class(**executor_kwargs) as pool:
        loop = asyncio.get_running_loop()
        
        # NOTE: For ProcessPoolExecutor: Submit tasks with staggered timing to serialize
        # worker initialization and prevent race conditions in JVM/native libraries.
        # For ThreadPoolExecutor: Submit all at once (no race conditions, lighter weight).

        tasks = []
        for idx, row in df.iterrows():
            job_kwargs = row.to_dict()
            input_path_job = job_kwargs.pop('input_path')
            output_path = job_kwargs.pop('output_path')

            # Submit to executor with retry wrapper
            task = run_worker_with_retries(
                loop,
                pool,
                unary_worker_sync,
                input_path_job,
                output_path,
                job_kwargs,
                max_retries=max_retries,
                task_id=idx
            )
            tasks.append(task)
            
            # For ProcessPoolExecutor: stagger submissions to serialize worker initialization
            # This prevents simultaneous JVM/native library initialization race conditions
            if not use_threading and idx < len(df) - 1:
                await asyncio.sleep(0.2)

        # Gather without catching exceptions - let them propagate naturally
        # This ensures errors are clearly visible to the user
        results = await asyncio.gather(*tasks)

        # Log results
        for i, result in enumerate(results):
            logger.info(f"[Main] Task {i} succeeded: {result}")

    return results


async def run_conversions_from_filepaths_with_local_cluster(
        input_path,
        **global_kwargs
):
    """
    Run parallel conversions where each job's parameters (including input/output dirs)
    are specified via kwargs or a CSV/XLSX file.

    Args:
        input_path:
            - list of file paths, OR
            - path to a CSV/XLSX with at least 'input_path' or 'filepath' column.
        **global_kwargs: global defaults for all conversions
    """
    from distributed import Client, LocalCluster
    df = take_filepaths(input_path, **global_kwargs)
    # Optionally create dirs before running
    for odir in df["output_path"].unique():
        if odir and not os.path.exists(odir):
            os.makedirs(odir, exist_ok=True)

    # --- Setup concurrency ---
    max_workers = int(global_kwargs.get("max_workers", 4))
    memory = global_kwargs.get("memory_per_worker", "10GB")
    verbose = global_kwargs.get('verbose', False)
    if verbose:
        logger.info(f"Parallelization with {max_workers} workers.")

    job_params = []
    for _, row in df.iterrows():
        job_kwargs = row.to_dict()
        input_path = job_kwargs.get('input_path')
        output_path = job_kwargs.get('output_path')
        job_kwargs.pop('input_path')
        job_kwargs.pop('output_path')
        job_params.append((input_path, output_path, job_kwargs))

    with LocalCluster(n_workers=max_workers,
                      memory_limit=memory
                      ) as cluster:
        with Client(cluster) as client:
            futures = [
                client.submit(unary_worker_sync,
                                    *paramset)
                for paramset in job_params
            ]
            logger.info("Submitted {} jobs to local dask cluster.".format(len(futures)))
            if verbose:
                logger.info("Cluster dashboard: %s", client.dashboard_link)
            results = client.gather(futures)

    return results


async def run_conversions_from_filepaths_with_slurm(
        input_path,
        **global_kwargs
):
    """
    Run parallel conversions using a SLURM cluster.
    Each job's parameters (including input/output dirs) are specified via kwargs or a CSV/XLSX file.

    Args:
        input_path:
            - list of file paths, OR
            - path to a CSV/XLSX with at least 'input_path' or 'filepath' column.
        **global_kwargs: global defaults for all conversions.
            Common SLURM params include:
                slurm_account, slurm_partition, slurm_time, slurm_mem, slurm_cores
    """
    import os

    from dask.distributed import Client
    from dask_jobqueue import SLURMCluster

    from eubi_bridge.conversion.conversion_worker import unary_worker_sync
    from eubi_bridge.utils.logging_config import get_logger
    from eubi_bridge.utils.path_utils import take_filepaths
    logger = get_logger(__name__)

    # --- Prepare file list ---
    df = take_filepaths(input_path, **global_kwargs)

    # Optionally create dirs before running
    for odir in df["output_path"].unique():
        if odir and not os.path.exists(odir):
            os.makedirs(odir, exist_ok=True)

    # --- Cluster configuration ---
    max_workers = int(global_kwargs.get("max_workers", 10))
    # slurm_account = global_kwargs.get("slurm_account", None)
    # slurm_partition = global_kwargs.get("slurm_partition", "general")
    slurm_time = global_kwargs.get("slurm_time", "01:00:00")
    slurm_mem = global_kwargs.get("memory_per_worker", "4GB")
    slurm_cores = int(global_kwargs.get("slurm_cores", 1))
    verbose = global_kwargs.get("verbose", False)

    if verbose:
        logger.info(f"Starting SLURM cluster with up to {max_workers} workers...")

    cluster = SLURMCluster(
        # queue=slurm_partition,
        # account=slurm_account,
        n_workers=max_workers,
        cores = slurm_cores,
        memory = slurm_mem,
        walltime = slurm_time,
        job_extra_directives=[
            f"--job-name=conversion",
            f"--output=slurm-%j.out"
        ],
    )

    cluster.scale(max_workers)

    with Client(cluster) as client:
        job_params = []
        for _, row in df.iterrows():
            job_kwargs = row.to_dict()
            input_path = job_kwargs.pop('input_path', None)
            output_path = job_kwargs.pop('output_path', None)
            job_params.append((input_path, output_path, job_kwargs))

        # Submit jobs
        futures = [
            client.submit(unary_worker_sync, *paramset)
            for paramset in job_params
        ]

        logger.info(f"Submitted {len(futures)} jobs to SLURM cluster.")
        if verbose:
            logger.info("Cluster dashboard: %s", client.dashboard_link)

        results = client.gather(futures)

    if verbose:
        logger.info("All SLURM jobs completed successfully.")

    return results


def _parse_tag(tag):
    if isinstance(tag, str):
        if not ',' in tag:
            return tag
        else:
            return tag.split(',')
    elif isinstance(tag, (tuple, list)):
        return tag
    elif tag is None:
        return None
    else:
        raise ValueError("tag must be a string, tuple, or list")

def _parse_filepaths_with_tags(filepaths, tags): # VIF
    """
    When aggregative conversion is applied, filepaths must contain
    at least one of the existing tags.
    :param filepaths:
    :param tags:
    :return:
    """
    accepted_filepaths = []
    for path in filepaths:
        path_appended = False
        for tag in tags:
            if tag is None:
                continue
            tag = _parse_tag(tag)
            if not isinstance(tag, (tuple, list)):
                tag = [tag]
            for tagitem in tag:
                if tagitem in path:
                    if not path_appended:
                        accepted_filepaths.append(path)
                        path_appended = True
    return accepted_filepaths


async def run_conversions_with_concatenation(
        input_path,
        output_path,
        scene_index: Union[int, tuple] = None,
        time_tag: Union[str, tuple] = None,
        channel_tag: Union[str, tuple] = None,
        z_tag: Union[str, tuple] = None,
        y_tag: Union[str, tuple] = None,
        x_tag: Union[str, tuple] = None,
        concatenation_axes: Union[int, tuple, str] = None,
        metadata_reader: str = 'bfio',
        **kwargs
        ):
    """Convert with concatenation using threads (safe for large Dask arrays)."""

    # Validate that tags are provided for concatenation axes
    if concatenation_axes:
        # Normalize concatenation_axes to a list
        if isinstance(concatenation_axes, str):
            concat_axes_list = list(concatenation_axes.lower())
        else:
            concat_axes_list = list(concatenation_axes)
        
        # Map axes to their corresponding tag parameters
        axis_to_tag = {
            't': time_tag,
            'c': channel_tag,
            'z': z_tag,
            'y': y_tag,
            'x': x_tag,
        }
        
        # Check that each concatenation axis has a corresponding tag
        missing_tags = []
        for axis in concat_axes_list:
            if axis in axis_to_tag and axis_to_tag[axis] is None:
                missing_tags.append(axis)
        
        if missing_tags:
            raise ValueError(
                f"When using --concatenation_axes '{concatenation_axes}', "
                f"you must provide tags for concatenation axes: {', '.join(missing_tags)}. "
                f"For example: --{missing_tags[0]}_tag 'value' "
                f"to identify files belonging to each {missing_tags[0]} group."
            )

    # input_path = f"/home/oezdemir/data/original/steyer/amst"
    # output_path = f"/home/oezdemir/data/zarr/steyer/result"
    # scene_index = 0
    # time_tag = None
    # channel_tag = None
    # z_tag = 'slice_'
    # y_tag = None
    # x_tag = None
    # concatenation_axes = 'z'
    # metadata_reader = 'bfio'
    # kwargs = {'skip_dask': True}

    df = take_filepaths(input_path,
                        scene_index = scene_index,
                        output_path = output_path,
                        metadata_reader = metadata_reader,
                        **kwargs)

    verbose = kwargs.get('verbose', None)
    override_channel_names = kwargs.get('override_channel_names', False)
    if verbose:
        logger.info(f"override_channel_names: {override_channel_names}")

    max_workers = int(kwargs.get("max_workers", 4))
    if verbose:
        logger.info(f"Parallelization with {max_workers} workers.")
    filepaths = df.input_path.to_numpy().tolist()

    tags = [time_tag,channel_tag,z_tag,y_tag,x_tag]
    filepaths_accepted = _parse_filepaths_with_tags(filepaths, tags)

    # Validate that each user-provided tag matches at least some files
    # If the user provides a tag, they expect it to match files
    provided_tags = {
        'time_tag': time_tag,
        'channel_tag': channel_tag,
        'z_tag': z_tag,
        'y_tag': y_tag,
        'x_tag': x_tag,
    }
    # Filter to only tags that were actually provided (non-None)
    provided_tags = {k: v for k, v in provided_tags.items() if v is not None}
    
    for tag_name, tag_value in provided_tags.items():
        # Normalize tag to list
        tag_list = _parse_tag(tag_value)
        if not isinstance(tag_list, (list, tuple)):
            tag_list = [tag_list]
        
        # Check if this tag matches any files
        tag_matches_any = False
        for tag_item in tag_list:
            for filepath in filepaths:
                if str(tag_item) in filepath:
                    tag_matches_any = True
                    break
            if tag_matches_any:
                break
        
        if not tag_matches_any:
            raise ValueError(
                f"Tag '{tag_name}={tag_value}' does not match any files in the input directory.\n"
                f"Available files: {[Path(p).name for p in filepaths]}\n"
                f"Please ensure your tags correctly identify files."
            )

    # --- Initialize AggregativeConverter and digest arrays ---
    base = AggregativeConverter(series = scene_index,
                                override_channel_names = override_channel_names
                                )
    base.filepaths = filepaths_accepted
    common_dir = os.path.commonpath(filepaths_accepted)
    skip_dask = kwargs.get('skip_dask', False)
    await base.read_dataset(readers_params = {'aszarr': skip_dask})
    if verbose:
        logger.info(f"Concatenating along {concatenation_axes}")
    await base.digest( ### Channel concatenation also done here.
        time_tag=time_tag,
        channel_tag=channel_tag,
        z_tag=z_tag,
        y_tag=y_tag,
        x_tag=x_tag,
        axes_of_concatenation=concatenation_axes,
        metadata_reader=metadata_reader,
        output_path=common_dir,
    )

    # --- Collect names and managers ---
    names, managers = [], []
    for key in base.digested_arrays.keys():
        updated_key = os.path.splitext(key)[0]
        names.append(updated_key)
        man = base.managers[key]
        if verbose:
            logger.info(f"Prepared manager for array '{key}' with shape {man.array.shape}")   
        managers.append(man)

    # --- Use ThreadPoolExecutor for aggregative conversion ---
    max_retries = int(kwargs.get("max_retries", 3))
    pool = ThreadPoolExecutor(max_workers=max_workers)
    # pool = ProcessPoolExecutor(max_workers=max_workers,
    #                            mp_context=mp.get_context("spawn")
    #                            )
    try:
        loop = asyncio.get_running_loop()
        tasks = [
            run_worker_with_retries(
                loop,
                pool,
                aggregative_worker_sync,
                manager,
                os.path.join(output_path, name),
                dict(kwargs),
                max_retries=max_retries,
                task_id=idx
            )
            for idx, (manager, name) in enumerate(zip(managers, names))
        ]
        results = await asyncio.gather(*tasks)
    finally:
        pool.shutdown(wait=True)

    return results


def run_conversions(
        filepaths,
        output_path,
        **kwargs
):
    axes_of_concatenation = kwargs.get('concatenation_axes',
                                       None)
    verbose = kwargs.get('verbose',
                         None)
    on_slurm = kwargs.get('on_slurm', False)
    on_local_cluster = kwargs.get('on_local_cluster', False)

    if axes_of_concatenation is None:
        if on_slurm:
            import shutil
            slurm_available = (shutil.which("sinfo") is not None or
                               shutil.which("squeue") is not None)
            if slurm_available:
                runner = run_conversions_from_filepaths_with_slurm
            else:
                logger.warn(f"Slurm is not available. Falling back to local-cluster mode.")
                runner = run_conversions_from_filepaths_with_local_cluster
        elif on_local_cluster:
            runner = run_conversions_from_filepaths_with_local_cluster
        else:
            runner = run_conversions_from_filepaths
    else:
        runner = run_conversions_with_concatenation
    return asyncio.run(runner(filepaths,
                              output_path = output_path,
                              **kwargs
                              )
                       )
