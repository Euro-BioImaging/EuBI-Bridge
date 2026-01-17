import asyncio
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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



async def run_conversions_from_filepaths(
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
    df = take_filepaths(input_path, **global_kwargs)

    # --- Setup concurrency ---
    max_workers = int(global_kwargs.get("max_workers", 4))

    # Get spawn context explicitly
    ctx = mp.get_context("spawn")

    # Import worker function
    from eubi_bridge.conversion.conversion_worker import unary_worker_sync

    # Create executor with spawn context AND initializer
    with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=initialize_worker_process  # Initialize JVM once per worker
    ) as pool:
        loop = asyncio.get_running_loop()
        
        # Pre-warm worker pool: force JVM initialization in all workers
        # This prevents ~30s delay when first tasks are submitted
        warmup_futures = [
            loop.run_in_executor(pool, _warmup_worker)
            for _ in range(max_workers)
        ]
        await asyncio.gather(*warmup_futures)

        tasks = []
        for idx, row in df.iterrows():
            job_kwargs = row.to_dict()
            input_path_job = job_kwargs.pop('input_path')
            output_path = job_kwargs.pop('output_path')

            # Submit to executor
            task = loop.run_in_executor(
                pool,
                unary_worker_sync,
                input_path_job,
                output_path,
                job_kwargs
            )
            tasks.append(task)

        # Gather with return_exceptions to see all failures
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[Main] Task {i} failed: {result}")
            else:
                print(f"[Main] Task {i} succeeded: {result}")

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
    print(f"override_channel_names: {override_channel_names}")

    max_workers = int(kwargs.get("max_workers", 4))
    if verbose:
        logger.info(f"Parallelization with {max_workers} workers.")
    filepaths = df.input_path.to_numpy().tolist()

    tags = [time_tag,channel_tag,z_tag,y_tag,x_tag]
    filepaths_accepted = _parse_filepaths_with_tags(filepaths, tags)

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
        print(f"Manager for {key} has array shape: {man.array.shape}")
        managers.append(man)

    # --- Use ThreadPoolExecutor for aggregative conversion ---
    pool = ThreadPoolExecutor(max_workers=max_workers)
    # pool = ProcessPoolExecutor(max_workers=max_workers,
    #                            mp_context=mp.get_context("spawn")
    #                            )
    try:
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(
                pool,
                aggregative_worker_sync,
                manager,
                os.path.join(output_path, name),
                dict(kwargs),
            )
            for manager, name in zip(managers, names)
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
