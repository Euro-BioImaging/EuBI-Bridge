"""Job dispatcher for EuBI-Bridge conversions.

Receives pre-validated :class:`~eubi_bridge.core.config_models.ConversionJob`
objects from ``ConversionManager`` in ``ebridge.py`` and dispatches them to
the appropriate execution backend (ProcessPool, ThreadPool, LocalCluster,
SLURM).

Responsibilities:
- Pool lifecycle management (create, stagger, gather, shutdown)
- Retry / fallback on BrokenProcessPool
- Failure reporting
- Backend selection (local / SLURM / Dask LocalCluster)

What this module does NOT do:
- Parameter merging or CSV triage  (done in ebridge.py)
- Pydantic validation              (done in ebridge.py)
- Actual conversion work           (done in conversion_worker.py)
"""

import asyncio
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Union


import pandas as pd
from natsort import natsorted

from eubi_bridge.conversion.aggregative_conversion_base import AggregativeConverter
from eubi_bridge.conversion.conversion_worker import (aggregative_worker_from_paths,
                                                      aggregative_worker_sync,
                                                      unary_worker_sync)
from eubi_bridge.conversion.worker_init import initialize_worker_process
from eubi_bridge.core.config_models import (AggregativeConversionJob,
                                             AggregativeOutputInfo,
                                             AggregativePlan,
                                             ConversionJob)
from eubi_bridge.utils.jvm_manager import soft_start_jvm
from eubi_bridge.utils.logging_config import get_logger
from eubi_bridge.utils.path_utils import (is_zarr_array, is_zarr_group,
                                          sensitive_glob, take_filepaths)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _report_failures(results: list, label: str = "Main") -> None:
    """Log and raise RuntimeError if any result signals failure."""
    failed = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"[{label}] Task {i} failed: {result}")
            failed.append((i, str(result)))
        elif isinstance(result, dict) and result.get('status') == 'error':
            logger.error(f"[{label}] Task {i} failed: {result.get('error')}")
            failed.append((i, result.get('error', 'Unknown error')))
        else:
            logger.info(f"[{label}] Task {i} succeeded.")
    if failed:
        summary = "\n".join(f"  Task {i}: {e}" for i, e in failed)
        raise RuntimeError(f"{len(failed)}/{len(results)} tasks failed:\n{summary}")


def _build_executor(cluster, use_threading: bool) -> tuple:
    """Return (executor_cls, executor_kwargs) from a ClusterConfig."""
    if use_threading:
        return ThreadPoolExecutor, {"max_workers": cluster.max_workers}
    ctx = mp.get_context("spawn")
    return ProcessPoolExecutor, {
        "max_workers": cluster.max_workers,
        "mp_context": ctx,
        "initializer": initialize_worker_process,
        "initargs": (cluster.tensorstore_data_copy_concurrency,),
    }


async def _dispatch(
    jobs: list,
    worker_fn: Callable,
    executor_cls,
    executor_kwargs: dict,
    use_threading: bool,
    label: str = "Main",
) -> list:
    """Shared coroutine: submit all jobs to a pool, gather, report failures.

    Staggers ProcessPool submissions by 0.2 s to serialise worker init.
    Falls back to a temporary single-worker pool on BrokenProcessPool.
    """
    tsc  = jobs[0].cluster.tensorstore_data_copy_concurrency if jobs else 1
    loop = asyncio.get_running_loop()

    async def _submit_one(pool, idx: int, job: ConversionJob):
        from concurrent.futures.process import BrokenProcessPool
        try:
            return await loop.run_in_executor(pool, worker_fn, job)
        except BrokenProcessPool:
            if use_threading:
                raise
            logger.warning(f"Task {idx}: main pool broken — retrying in temp pool.")
            ctx = mp.get_context("spawn")
            temp = ProcessPoolExecutor(
                max_workers=1, mp_context=ctx,
                initializer=initialize_worker_process, initargs=(tsc,),
            )
            try:
                return await loop.run_in_executor(temp, worker_fn, job)
            finally:
                temp.shutdown(wait=True)

    pool = None
    try:
        pool  = executor_cls(**executor_kwargs)
        max_w = executor_kwargs.get("max_workers") or 1
        tasks = []
        for idx, job in enumerate(jobs):
            tasks.append(_submit_one(pool, idx, job))
            # Stagger only the first max_workers submissions to serialise worker
            # process init. Staggering every job is O(n_jobs) delay — 288 s for
            # 1 442 files — which makes the GUI appear jammed.
            if not use_threading and idx < max_w - 1:
                await asyncio.sleep(0.2)
        results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        if pool is not None:
            try:
                pool.shutdown(wait=True)
            except Exception as e:
                logger.warning(f"Error shutting down pool: {e}")
                try:
                    pool.shutdown(wait=False)
                except Exception as e2:
                    logger.error(f"Force shutdown failed: {e2}")

    _report_failures(list(results), label)
    return list(results)


# ---------------------------------------------------------------------------
# Unary dispatch entry point  (called from ConversionManager in ebridge.py)
# ---------------------------------------------------------------------------

def dispatch_unary_jobs(jobs: list) -> list:
    """Dispatch a list of pre-validated ConversionJob objects.

    Selects the execution backend from the first job's ClusterConfig.
    All jobs are assumed to share the same cluster configuration.
    """
    if not jobs:
        return []
    cluster       = jobs[0].cluster
    use_threading = cluster.use_threading

    jvm_memory = jobs[0].cluster.jvm_memory if jobs else None
    if jvm_memory:
        os.environ['EUBI_JVM_MEMORY'] = str(jvm_memory)

    if cluster.on_slurm:
        return asyncio.run(_dispatch_unary_with_slurm(jobs))
    elif cluster.on_local_cluster:
        return asyncio.run(_dispatch_unary_with_local_cluster(jobs))
    else:
        exc_cls, exc_kw = _build_executor(cluster, use_threading)
        return asyncio.run(
            _dispatch(jobs, unary_worker_sync, exc_cls, exc_kw, use_threading)
        )


async def _dispatch_unary_with_slurm(jobs: list) -> list:
    """Submit unary jobs to a SLURM cluster via dask-jobqueue."""
    from dask.distributed import Client
    from dask_jobqueue import SLURMCluster

    cfg = jobs[0].cluster
    cluster_kwargs = dict(
        cores=1, processes=1,
        memory=cfg.memory_per_worker, walltime=cfg.slurm_time,
        job_extra_directives=[
            '--job-name=eubi_conversion',
            '--output=slurm-%j.out',
        ],
    )
    if cfg.slurm_account:
        cluster_kwargs['account'] = cfg.slurm_account
    if cfg.slurm_partition:
        cluster_kwargs['queue'] = cfg.slurm_partition
    if cfg.slurm_sif_path:
        cluster_kwargs['python'] = f"apptainer exec --bind /usr/lib64:/usr/lib64 --bind /etc/slurm:/etc/slurm --bind /etc/munge:/etc/munge --bind /run/munge:/run/munge --bind /etc/passwd:/etc/passwd --bind /etc/group:/etc/group {cfg.slurm_sif_path} python"

    slurm_cluster = SLURMCluster(**cluster_kwargs)
    slurm_cluster.scale(cfg.max_workers)
    try:
        with slurm_cluster:
            with Client(slurm_cluster) as client:
                logger.info(f"Cluster dashboard: {client.dashboard_link}")
                client.wait_for_workers(cfg.max_workers,
                                        timeout=cfg.slurm_worker_timeout)
                client.run(_slurm_worker_init,
                           cfg.tensorstore_data_copy_concurrency)
                futures = [
                    client.submit(unary_worker_sync, j, pure=False)
                    for j in jobs
                ]
                logger.info(f"Submitted {len(futures)} jobs to SLURM.")
                results = list(client.gather(futures))
    finally:
        logger.info("SLURM cluster shut down.")

    _report_failures(results, "SLURM")
    return results


async def _dispatch_unary_with_local_cluster(jobs: list) -> list:
    """Submit unary jobs to a local Dask cluster."""
    from dask.distributed import Client, LocalCluster

    cfg = jobs[0].cluster
    with LocalCluster(n_workers=cfg.max_workers,
                      memory_limit=cfg.memory_per_worker) as lc:
        with Client(lc) as client:
            client.run(_slurm_worker_init,
                       cfg.tensorstore_data_copy_concurrency)
            futures = [
                client.submit(unary_worker_sync, j, pure=False)
                for j in jobs
            ]
            results = list(client.gather(futures))

    _report_failures(results, "LocalCluster")
    return results


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Aggregative planning
# ---------------------------------------------------------------------------

def _build_aggregative_plan(
    filepaths: list,
    concatenation_axes,
    common_dir: str,
    output_path: str,
    max_workers: int = 4,
    time_tag=None,
    channel_tag=None,
    z_tag=None,
    y_tag=None,
    x_tag=None,
) -> AggregativePlan:
    """Determine output groups and allocate ``file_workers``.

    Uses dummy shapes for FileSet — grouping is purely tag-based and correct
    regardless of shape values.  Actual shapes are determined inside each
    worker via ``read_dataset()`` + ``digest()``.
    """
    from eubi_bridge.conversion.fileset_io import FileSet

    axes_str = 'tczyx'
    if isinstance(concatenation_axes, str):
        axlist = [axes_str.index(x) for x in concatenation_axes.lower() if x in axes_str]
    elif concatenation_axes is not None:
        axlist = [axes_str.index(x) for x in str(concatenation_axes) if x in axes_str]
    else:
        axlist = []

    dummy      = (1, 1, 1, 1, 1)
    shape_list = [dummy] * len(filepaths)

    fileset = FileSet(
        filepaths,
        shapes=shape_list,
        axis_tag0=time_tag,
        axis_tag1=channel_tag,
        axis_tag2=z_tag,
        axis_tag3=y_tag,
        axis_tag4=x_tag,
    )
    for axis in axlist:
        fileset.concatenate_along(axis)

    file_groups: dict = {}
    for src_path, output_key in fileset.path_dict.items():
        rel      = os.path.relpath(output_key, common_dir)
        rel      = os.sep.join(p for p in rel.split(os.sep) if p != '..')
        name     = os.path.splitext(rel)[0].replace(os.sep, '-')
        full_out = os.path.join(output_path, name)
        if full_out not in file_groups:
            file_groups[full_out] = {'source_files': []}
        file_groups[full_out]['source_files'].append(src_path)

    n_outputs    = len(file_groups)
    file_workers = min(n_outputs, max_workers)

    outputs = [
        AggregativeOutputInfo(output_path=out_path, source_files=grp['source_files'])
        for out_path, grp in file_groups.items()
    ]

    return AggregativePlan(
        n_outputs=n_outputs,
        outputs=outputs,
        file_workers=file_workers,
    )


# ---------------------------------------------------------------------------
# Aggregative dispatch entry point  (called from ConversionManager in ebridge.py)
# ---------------------------------------------------------------------------

def dispatch_aggregative_job(
    job: AggregativeConversionJob,
    plan: 'AggregativePlan | None' = None,
) -> list:
    """Dispatch a pre-validated AggregativeConversionJob.

    If *plan* is supplied (from :meth:`ConversionManager.validate_aggregative`)
    the pre-computed group assignments and worker counts are used directly,
    skipping re-planning.  Without a plan the dispatcher computes both from
    the job's ``max_workers`` and the number of output groups it finds.
    """
    runner = (
        run_aggregative_with_slurm
        if job.cluster.on_slurm
        else run_conversions_with_concatenation
    )
    return asyncio.run(runner(
        job.input_path,
        output_path=job.output_path,
        includes=job.includes,
        excludes=job.excludes,
        time_tag=job.time_tag,
        channel_tag=job.channel_tag,
        z_tag=job.z_tag,
        y_tag=job.y_tag,
        x_tag=job.x_tag,
        concatenation_axes=job.concatenation_axes,
        plan=plan,
        **job.to_conversion_kwargs(),
    ))


# ---------------------------------------------------------------------------
# Metadata collection  (read-only; keeps flat-dict interface)
# ---------------------------------------------------------------------------

async def run_metadata_collection_from_filepaths(input_path, **global_kwargs):
    """Collect metadata from image files in parallel.

    Args:
        input_path: Path to files / directory / CSV-XLSX.
        **global_kwargs: Configuration parameters (supports use_threading flag).

    Returns:
        List of metadata dicts, one per file.
    """
    df = take_filepaths(input_path, **global_kwargs)

    max_workers   = int(global_kwargs.get("max_workers", 4))
    use_threading = global_kwargs.get("use_threading", False)
    tsc           = global_kwargs.get('tensorstore_data_copy_concurrency', 1)

    from eubi_bridge.conversion.conversion_worker import metadata_reader_sync

    if use_threading:
        executor_cls, executor_kwargs = ThreadPoolExecutor, {"max_workers": max_workers}
    else:
        ctx = mp.get_context("spawn")
        executor_cls  = ProcessPoolExecutor
        executor_kwargs = {
            "max_workers": max_workers,
            "mp_context": ctx,
            "initializer": initialize_worker_process,
            "initargs": (tsc,),
        }

    loop = asyncio.get_running_loop()

    async def _submit_meta(pool, idx, ip, job_kw):
        from concurrent.futures.process import BrokenProcessPool
        try:
            return await loop.run_in_executor(pool, metadata_reader_sync, ip, job_kw)
        except BrokenProcessPool:
            if use_threading:
                raise
            logger.warning(f"Meta task {idx}: main pool broken — retrying.")
            ctx2 = mp.get_context("spawn")
            temp = ProcessPoolExecutor(
                max_workers=1, mp_context=ctx2,
                initializer=initialize_worker_process, initargs=(tsc,),
            )
            try:
                return await loop.run_in_executor(temp, metadata_reader_sync, ip, job_kw)
            finally:
                temp.shutdown(wait=True)

    pool = None
    try:
        pool  = executor_cls(**executor_kwargs)
        tasks = []
        for idx, row in df.iterrows():
            kw = row.to_dict()
            ip = kw.pop('input_path')
            tasks.append(_submit_meta(pool, idx, ip, kw))
            if not use_threading and idx < len(df) - 1:
                await asyncio.sleep(0.2)
        results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        if pool is not None:
            try:
                pool.shutdown(wait=True)
            except Exception as e:
                logger.warning(f"Error shutting down meta pool: {e}")
                try:
                    pool.shutdown(wait=False)
                except Exception as e2:
                    logger.error(f"Force shutdown failed: {e2}")

    failed = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.error(f"[Meta] Task {i} failed: {r}")
            failed.append((i, str(r)))
        elif isinstance(r, dict) and r.get('status') == 'error':
            logger.error(f"[Meta] Task {i} failed: {r.get('error')}")
            failed.append((i, r.get('error', 'Unknown error')))
    if failed:
        summary = "\n".join(f"  Task {i}: {e}" for i, e in failed)
        raise RuntimeError(
            f"Metadata collection failed for {len(failed)}/{len(results)} files:\n{summary}"
        )
    return results


# ---------------------------------------------------------------------------
# Aggregative (concatenation) path  — unchanged flat-dict interface
# ---------------------------------------------------------------------------

def _parse_tag(tag):
    if isinstance(tag, str):
        return tag if ',' not in tag else tag.split(',')
    elif isinstance(tag, (tuple, list)):
        return tag
    elif tag is None:
        return None
    else:
        raise ValueError("tag must be a string, tuple, or list")


def _parse_filepaths_with_tags(filepaths, tags):
    accepted, seen = [], set()
    for path in filepaths:
        for tag in tags:
            if tag is None:
                continue
            tag_list = _parse_tag(tag)
            if not isinstance(tag_list, (tuple, list)):
                tag_list = [tag_list]
            for item in tag_list:
                if item in path and path not in seen:
                    accepted.append(path)
                    seen.add(path)
    return accepted


def _slurm_worker_init(tensorstore_data_copy_concurrency='default'):
    """Worker initialiser for Dask SLURM/LocalCluster workers."""
    from eubi_bridge.conversion.worker_init import initialize_worker_process
    initialize_worker_process(tensorstore_data_copy_concurrency)


def _aggregative_slurm_task(input_path, output_path, kwargs_dict):
    """Module-level sync wrapper for aggregative conversion on a Dask worker."""
    import concurrent.futures

    def _run_in_fresh_thread():
        import dask
        from eubi_bridge.utils.jvm_manager import soft_start_jvm
        soft_start_jvm()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with dask.config.set(scheduler='synchronous'):
                return loop.run_until_complete(
                    run_conversions_with_concatenation(
                        input_path, output_path, **kwargs_dict)
                )
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run_in_fresh_thread).result()


async def run_aggregative_with_slurm(input_path, output_path, **kwargs):
    """Submit an aggregative conversion as a single SLURM job."""
    from dask.distributed import Client
    from dask_jobqueue import SLURMCluster

    max_workers     = int(kwargs.get('max_workers', 4))
    slurm_time      = kwargs.get('slurm_time', '24:00:00')
    slurm_mem       = kwargs.get('memory_per_worker', '8GB')
    slurm_account   = kwargs.get('slurm_account', None)
    slurm_partition = kwargs.get('slurm_partition', None)
    slurm_sif_path  = kwargs.get('slurm_sif_path', None)
    tsc             = kwargs.get('tensorstore_data_copy_concurrency', 'default')
    worker_timeout  = int(kwargs.get('slurm_worker_timeout', 300))

    cluster_kwargs = dict(
        cores=max_workers, memory=slurm_mem, walltime=slurm_time,
        job_extra_directives=[
            '--job-name=eubi_aggregative',
            '--output=slurm-%j.out',
        ],
    )
    if slurm_account:
        cluster_kwargs['account'] = slurm_account
    if slurm_partition:
        cluster_kwargs['queue'] = slurm_partition
    if slurm_sif_path:
        cluster_kwargs['python'] = f"apptainer exec --bind /usr/lib64:/usr/lib64 --bind /etc/slurm:/etc/slurm --bind /etc/munge:/etc/munge --bind /run/munge:/run/munge --bind /etc/passwd:/etc/passwd --bind /etc/group:/etc/group {slurm_sif_path} python"

    cluster = SLURMCluster(**cluster_kwargs)
    cluster.scale(1)
    try:
        with Client(cluster) as client:
            logger.info(f"Cluster dashboard: {client.dashboard_link}")
            client.wait_for_workers(1, timeout=worker_timeout)
            client.run(_slurm_worker_init, tsc)
            future = client.submit(
                _aggregative_slurm_task, input_path, output_path,
                dict(kwargs), pure=False, retries=0,
            )
            logger.info("Aggregative SLURM job submitted; waiting...")
            result = client.gather(future)
    finally:
        cluster.close()
        logger.info("SLURM cluster shut down.")
    return result


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
        plan: 'AggregativePlan | None' = None,
        **kwargs,
):
    """Aggregative conversion: plan groups in the main process, write in parallel.

    If *plan* is provided the pre-computed group assignments and worker
    counts are used directly (no re-planning, no header reads).  Without a
    plan the groups are derived via pure filename matching and worker counts
    are auto-computed from ``max_workers`` ÷ ``n_outputs``.

    Each group is dispatched to a pool worker (ProcessPool by default;
    ThreadPool when ``use_threading=True``).  Workers run their own complete
    pipeline (open files → concat → write zarr) so no non-picklable objects
    cross the process boundary.
    """
    from eubi_bridge.core.config_models import ClusterConfig

    verbose = kwargs.get('verbose', None)

    # ── Group assignments ─────────────────────────────────────────────────
    if plan is not None:
        groups       = {out.output_path: out.source_files for out in plan.outputs}
        file_workers = plan.file_workers
        if verbose:
            logger.info(
                f"Using pre-computed plan: {len(groups)} group(s), "
                f"file_workers={file_workers}"
            )
    else:
        df = take_filepaths(
            input_path, scene_index=scene_index,
            output_path=output_path, metadata_reader=metadata_reader, **kwargs,
        )
        filepaths          = df.input_path.to_numpy().tolist()
        filepaths_accepted = _parse_filepaths_with_tags(
            filepaths, [time_tag, channel_tag, z_tag, y_tag, x_tag]
        )

        for tag_name, tag_value in dict(
            time_tag=time_tag, channel_tag=channel_tag,
            z_tag=z_tag, y_tag=y_tag, x_tag=x_tag,
        ).items():
            if tag_value is None:
                continue
            tag_list = _parse_tag(tag_value)
            if not isinstance(tag_list, (list, tuple)):
                tag_list = [tag_list]
            if not any(str(t) in fp for t in tag_list for fp in filepaths):
                raise ValueError(
                    f"Tag '{tag_name}={tag_value}' does not match any files.\n"
                    f"Available: {[Path(p).name for p in filepaths]}"
                )

        common_dir  = os.path.commonpath(filepaths_accepted)
        cluster_cfg = ClusterConfig(**{k: v for k, v in kwargs.items()
                                       if k in ClusterConfig.model_fields})
        inner_plan = _build_aggregative_plan(
            filepaths_accepted,
            concatenation_axes=concatenation_axes,
            common_dir=common_dir,
            output_path=output_path,
            max_workers=cluster_cfg.max_workers,

            time_tag=time_tag, channel_tag=channel_tag,
            z_tag=z_tag, y_tag=y_tag, x_tag=x_tag,
        )
        groups       = {out.output_path: out.source_files for out in inner_plan.outputs}
        file_workers = inner_plan.file_workers

        if verbose:
            logger.info(
                f"Planned {inner_plan.n_outputs} output group(s): {list(groups.keys())}  "
                f"file_workers={file_workers}"
            )

    # ── Build job kwargs (workers receive everything they need) ───────────
    job_kwargs = dict(
        time_tag=time_tag, channel_tag=channel_tag,
        z_tag=z_tag, y_tag=y_tag, x_tag=x_tag,
        concatenation_axes=concatenation_axes,
        metadata_reader=metadata_reader,
        scene_index=scene_index,
        **kwargs,
    )

    tsc = kwargs.get('tensorstore_data_copy_concurrency', 1)
    os.environ['EUBI_TENSORSTORE_DATA_COPY_CONCURRENCY'] = str(tsc)

    jvm_memory = kwargs.get('jvm_memory')
    if jvm_memory:
        os.environ['EUBI_JVM_MEMORY'] = str(jvm_memory)

    use_threading = kwargs.get('use_threading', False)
    # Cap the pool at file_workers.
    pool_cluster = ClusterConfig(**{**{k: v for k, v in kwargs.items()
                                       if k in ClusterConfig.model_fields},
                                   'max_workers': file_workers})
    exc_cls, exc_kw = _build_executor(pool_cluster, use_threading)

    loop = asyncio.get_running_loop()
    pool = None
    try:
        pool = exc_cls(**exc_kw)
        tasks = [
            loop.run_in_executor(
                pool, aggregative_worker_from_paths,
                file_paths, out_path, dict(job_kwargs),
            )
            for out_path, file_paths in groups.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    _report_failures(list(results), "Aggregative")
    return results


def run_conversions(input_path, output_path=None, **kwargs) -> list:
    """Aggregative (concatenation) conversion entry point.

    Thin backward-compat wrapper around :func:`dispatch_aggregative_job`.
    Unary conversions use :func:`dispatch_unary_jobs` instead.
    """
    if kwargs.get('concatenation_axes') is None:
        raise ValueError(
            "run_conversions() is for aggregative (concatenation) conversions only. "
            "For unary conversions call dispatch_unary_jobs() instead."
        )
    job = AggregativeConversionJob.from_kwargs(
        input_path=str(input_path),
        output_path=str(output_path) if output_path is not None else "",
        kwargs=kwargs,
    )
    return dispatch_aggregative_job(job)
