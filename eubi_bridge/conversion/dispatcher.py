"""Job dispatcher for EuBI-Bridge conversions.

Receives pre-validated :class:`~eubi_bridge.core.config_models.ConversionJob`
objects from ``ConversionManager`` in ``ebridge.py`` and dispatches them to
the execution backend: a ProcessPool, a ThreadPool, or SLURM (through
dask-jobqueue; the full eubi-bridge only).  (The dask LocalCluster backend was
retired on 2026-10-05.)

Responsibilities:
- Pool lifecycle management (create, stagger, gather, shutdown)
- Retry / fallback on BrokenProcessPool
- Failure reporting

What this module does NOT do:
- Parameter merging or CSV triage  (done in ebridge.py)
- Pydantic validation              (done in ebridge.py)
- Actual conversion work           (done in conversion_worker.py)
"""

import asyncio
import math
import multiprocessing as mp
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Union


import pandas as pd
from natsort import natsorted

from eubi_bridge.conversion.conversion_worker import (aggregative_worker_from_paths,
                                                      unary_worker_sync)
from eubi_bridge.conversion.worker_init import initialize_worker_process
from eubi_bridge.core.config_models import (AggregativeConversionJob,
                                             AggregativeOutputInfo,
                                             AggregativePlan,
                                             ConversionJob)
from eubi_bridge.utils.jvm_manager import soft_start_jvm
from eubi_bridge.utils.errors import InputError
from eubi_bridge.utils.logging_config import get_logger
from eubi_bridge.utils.path_utils import (is_zarr_array, is_zarr_group,
                                          prefix_with_group, sensitive_glob,
                                          take_filepaths)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _raise_version_error(results: list) -> None:
    """Raise the first MicroReaderVersionError among *results* as itself (a wrong
    micro-reader installed; its message names the fix), not inside the RuntimeError
    that sums up the failed tasks: a script catches it by its type."""
    from eubi_bridge.core.micro_source import MicroReaderVersionError
    for result in results:
        if isinstance(result, MicroReaderVersionError):
            raise result


#: what micro-reader says of a file its set does not list (companion.NotInItsSet, a member
#: whose set "does not list it"): a set's side file, among several inputs no failure
_NOT_IN_ITS_SET = ("NotInItsSet", "which does not list it", "is not part of the set")


def _not_in_its_set(error: str) -> bool:
    return any(words in error for words in _NOT_IN_ITS_SET)


def _report_failures(results: list, label: str = "Main") -> None:
    """Log and raise RuntimeError if any result signals failure (a
    MicroReaderVersionError as itself, ``_raise_version_error``)."""
    failed = []
    for i, result in enumerate(results):
        error = (str(result) if isinstance(result, Exception) else
                 str(result.get('error', 'Unknown error'))
                 if isinstance(result, dict) and result.get('status') == 'error' else None)
        if error is None:
            logger.info(f"[{label}] Task {i} succeeded.")
        elif len(results) > 1 and _not_in_its_set(error):
            # one of several inputs is a file its set does not list (a CQ1 measurement's
            # calibration images, its MIP companion): a side file of a set converted
            # anyway, no failure of the run (review cycle 1, N-13). Given alone, it fails.
            logger.info(f"[{label}] Task {i} skipped: a file its set does not list, no "
                        f"image of its own: {error[:200]}")
        else:
            logger.error(f"[{label}] Task {i} failed: {error[:300]}")
            failed.append((i, error))
    if failed:
        _raise_version_error(results)
        said = [str(e).split(": ", 1)[-1] for _, e in failed if str(e).startswith("InputError: ")]
        if len(said) == len(failed):                # what the user asked for: one line, each
            raise InputError("; ".join(dict.fromkeys(said)))
        from eubi_bridge.utils.errors import TasksFailedError
        summary = "\n".join(f"  Task {i}: {e}" for i, e in failed)
        raise TasksFailedError(f"{len(failed)}/{len(results)} tasks failed:\n{summary}")


#: how long a worker process gets to exit once all work is done, before it is stopped
_WORKER_EXIT_S = 30.0


def _shutdown_pool(pool) -> None:
    """Shut a pool down without waiting forever.

    It is only called once every result is in. A ProcessPool worker that has not exited
    within ``_WORKER_EXIT_S`` is stopped: a worker whose Java VM still runs threads (a
    Bio-Formats read that failed half way leaves its reader open) never exits, and
    ``shutdown(wait=True)`` then waited for it forever - the conversion had already
    reported its error, but the run never ended (eubi-bridge 0.1.4)."""
    if not isinstance(pool, ProcessPoolExecutor):
        pool.shutdown(wait=True)
        return
    workers = list((getattr(pool, "_processes", None) or {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    deadline = time.monotonic() + _WORKER_EXIT_S
    for worker in workers:
        worker.join(max(0.0, deadline - time.monotonic()))
    for worker in workers:
        if worker.is_alive():
            logger.warning(f"Worker process {worker.pid} had not exited {_WORKER_EXIT_S:.0f} s "
                           f"after its work finished (e.g. a Java VM still running after a "
                           f"failed Bio-Formats read); stopping it.")
            worker.kill()
            worker.join(5)


def _build_executor(cluster, use_threading: bool) -> tuple:
    """Return (executor_cls, executor_kwargs) from a ClusterConfig."""
    if use_threading:
        return ThreadPoolExecutor, {"max_workers": cluster.max_processes}
    ctx = mp.get_context("spawn")
    return ProcessPoolExecutor, {
        "max_workers": cluster.max_processes,
        "mp_context": ctx,
        "initializer": initialize_worker_process,
        "initargs": (cluster.tensorstore_data_copy_concurrency,),
    }


#: index-type files: the set they open lies in their folder (its size is the input's)
_SET_INDEX_SUFFIXES = (".xml", ".htd", ".wpi", ".mlf", ".xdce", ".jdce", ".ome", ".nd",
                       "metadata.txt")
#: files looked at when sizing a folder (a huge one is not walked to its end)
_SIZE_WALK_LIMIT = 50_000


def _folder_bytes(folder: str) -> int:
    total, seen = 0, 0
    for root, _dirs, files in os.walk(folder):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
            seen += 1
            if seen >= _SIZE_WALK_LIMIT:
                return total
    return total


def _input_bytes(path) -> int:
    """About how much an input holds, to start the biggest first: a file's size; a
    folder's, or an index file's folder's, files (bounded walk); 0 for a remote store
    or what cannot be read."""
    paths = path if isinstance(path, (list, tuple)) else [path]
    total = 0
    for p in paths:
        p = str(p)
        if "://" in p:
            continue
        try:
            if os.path.isdir(p):
                total += _folder_bytes(p)
            elif p.lower().endswith(_SET_INDEX_SUFFIXES):
                total += _folder_bytes(os.path.dirname(os.path.abspath(p)))
            else:
                total += os.path.getsize(p)
        except OSError:
            pass
    return total


def _largest_first(paths: list) -> list:
    """Indices of *paths* (one input each, or a list of files for one output), the
    biggest first; equal sizes keep their order.  Started in this order, the small
    files fill the other processes while a big one runs, instead of a slide scan
    finishing alone at the end of a batch."""
    sizes = [_input_bytes(p) for p in paths]
    return sorted(range(len(paths)), key=lambda i: -sizes[i])


async def _dispatch(
    jobs: list,
    worker_fn: Callable,
    executor_cls,
    executor_kwargs: dict,
    use_threading: bool,
    label: str = "Main",
    tsc=None,
    order: list = None,
) -> list:
    """Shared coroutine: submit all jobs to a pool, gather, report failures.

    Staggers ProcessPool submissions by 0.2 s to serialise worker init.
    Falls back to a temporary single-worker pool on BrokenProcessPool.
    *order*: the order to start the jobs in (indices into *jobs*, the biggest first);
    results come back, and failures are numbered, in *jobs*' own order.
    """
    if tsc is None:
        tsc = jobs[0].cluster.tensorstore_data_copy_concurrency if jobs else 1
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
                _shutdown_pool(temp)

    pool = None
    try:
        pool  = executor_cls(**executor_kwargs)
        max_w = executor_kwargs.get("max_workers") or 1
        tasks = []
        sequence = list(order) if order is not None else list(range(len(jobs)))
        for k, idx in enumerate(sequence):
            tasks.append(_submit_one(pool, idx, jobs[idx]))
            # Stagger only the first max_workers submissions to serialise worker
            # process init. Staggering every job is O(n_jobs) delay — 288 s for
            # 1 442 files — which makes the GUI appear jammed.
            if not use_threading and k < max_w - 1:
                await asyncio.sleep(0.2)
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        results = [None] * len(jobs)
        for idx, result in zip(sequence, gathered):
            results[idx] = result
    finally:
        if pool is not None:
            try:
                _shutdown_pool(pool)
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
    # the conversions that will run at once: the region budget is shared by
    # these, not by max_processes
    running = min(jobs[0].cluster.max_processes, len(jobs))
    # each conversion's memory, resolved HERE, once, before the pool starts (a worker
    # measuring the free RAM would see the others): the run's memory_budget over the
    # conversions running at once, or on SLURM one worker job's memory each
    from eubi_bridge.core.memory import main_process_mb, process_mb
    first = jobs[0].cluster
    # worker processes: this (main) process's memory comes out of the run's budget
    # first; threads: it is each conversion's own baseline (measured in the worker)
    per_process = process_mb(first, 1 if first.on_slurm else running,
                             0.0 if first.use_threading else main_process_mb())
    if not first.on_slurm:
        # so the conversions still busy can take what the finished ones held
        from eubi_bridge.core.memory_share import plan_tasks
        plan_tasks(len(jobs), running, per_process * running)
    jobs = [job.model_copy(update={
                'cluster': job.cluster.model_copy(update={
                    'concurrent_jobs': running, 'process_memory_mb': per_process})})
            for job in jobs]
    cluster       = jobs[0].cluster
    use_threading = cluster.use_threading

    jvm_memory = jobs[0].cluster.jvm_memory if jobs else None
    if jvm_memory:
        os.environ['EUBI_JVM_MEMORY'] = str(jvm_memory)

    if cluster.on_slurm:
        return asyncio.run(_dispatch_unary_with_slurm(jobs))
    exc_cls, exc_kw = _build_executor(cluster, use_threading)
    return asyncio.run(
        _dispatch(jobs, unary_worker_sync, exc_cls, exc_kw, use_threading,
                  order=_largest_first([job.input_path for job in jobs]))
    )


async def _dispatch_unary_with_slurm(jobs: list) -> list:
    """Submit unary jobs to a SLURM cluster via dask-jobqueue."""
    from dask.distributed import Client
    from dask_jobqueue import SLURMCluster

    cfg = jobs[0].cluster
    cluster_kwargs = dict(
        cores=1, processes=1,
        memory=cfg.slurm_worker_memory, walltime=cfg.slurm_time,
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
    slurm_cluster.scale(cfg.max_processes)
    try:
        with slurm_cluster:
            with Client(slurm_cluster) as client:
                logger.info(f"Cluster dashboard: {client.dashboard_link}")
                client.wait_for_workers(cfg.max_processes,
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


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Aggregative planning
# ---------------------------------------------------------------------------

#: the option naming each axis's tag
TAG_OPTIONS = dict(t="time_tag", c="channel_tag", z="z_tag", y="y_tag", x="x_tag")


def check_tags_and_axes(concatenation_axes, tags: dict, say: bool = True) -> list:
    """The joined axes as positions in t c z y x, after checking them against the tags, before
    any file is grouped. Each joined axis needs its tag, and each axis is joined once; one tag
    names one axis. A tag of an axis that is not joined splits the outputs: one per value,
    with a line saying so (*say*: the planner says it, the checks ahead of it do not).
    Joining along several axes joins each one in its own lines: along t, the files of one
    channel; along c, the files of one time point."""
    axes = 'tczyx'
    letters = [x for x in str(concatenation_axes or '').lower() if x not in " ,;()[]'\""]
    unknown = [x for x in letters if x not in axes]
    if unknown:
        raise InputError(f"--concatenation_axes {concatenation_axes}: {', '.join(unknown)} is no "
                         f"axis; join along t, c, z, y or x.")
    twice = sorted({x for x in letters if letters.count(x) > 1})
    if twice:
        raise InputError(f"--concatenation_axes {concatenation_axes} names {', '.join(twice)} "
                         f"twice: each axis is joined once.")
    given = {x: tag for x, tag in tags.items() if tag is not None}
    untagged = [x for x in letters if x not in given]
    if untagged:
        raise InputError(
            f"Joining along {', '.join(untagged)} needs a tag that numbers or names the files "
            f"along it: --{' --'.join(TAG_OPTIONS[x] for x in untagged)}.")
    seen, named, numbered = {}, {}, {}
    for x, tag in given.items():
        parsed = _parse_tag(tag)
        if isinstance(parsed, list):
            if x in letters and len(parsed) == 1:
                raise InputError(f"--{TAG_OPTIONS[x]} {tag} names one {TAG_OPTIONS[x][:-4]} only: "
                                 f"joining along {x} needs two names or more, with commas and no "
                                 f"space between them, or quoted ('{parsed[0]}, ...').")
            named.update((item, x) for item in parsed)
        else:
            numbered[str(parsed).rstrip('/\\')] = x       # frames/ and frames are one tag
        for item in (parsed if isinstance(parsed, list) else [str(parsed).rstrip('/\\')]):
            if item in seen:
                raise InputError(
                    f"{item!r} is named twice in --{TAG_OPTIONS[x]}: each name is one channel."
                    if seen[item] == x else
                    f"--{TAG_OPTIONS[seen[item]]} and --{TAG_OPTIONS[x]} are both {item!r}: one "
                    f"tag names one axis.")
            seen[item] = x
    for prefix, x in numbered.items():
        inside = next((name for name, other in named.items()
                       if other != x and re.match(re.escape(prefix) + r'[/\\]?\d', name)), None)
        if inside:
            raise InputError(f"--{TAG_OPTIONS[x]} {prefix} is read in --{TAG_OPTIONS[named[inside]]}'s "
                             f"name {inside}: give tags that do not hold each other.")
    split = [x for x in given if x not in letters]
    if say and letters and split:
        logger.info(f"Joined along {', '.join(letters)}; one output per "
                    f"{' and '.join(TAG_OPTIONS[x] for x in split)} value.")
    return [axes.index(x) for x in letters]


def plan_settings(input_path, concatenation_axes, tags: dict, includes=None,
                  excludes=None) -> dict:
    """What a plan is made for, spelled one way: its input, joined axes, tags and filters.
    A run with a plan compares its own (check_plan)."""
    if isinstance(input_path, (list, tuple)):
        where = sorted(os.path.normcase(os.path.abspath(str(path))) for path in input_path)
    elif isinstance(input_path, (str, Path)):
        where = os.path.normcase(os.path.abspath(str(input_path)))
    else:
        where = None                                    # a table built already

    def spelled(tag):
        parsed = _parse_tag(tag)
        if isinstance(parsed, list):
            return ','.join(parsed)
        return None if parsed is None else str(parsed).rstrip('/\\')

    return dict(input=where,
                concatenation_axes=''.join(x for x in str(concatenation_axes or '').lower()
                                           if x in 'tczyx'),
                **{TAG_OPTIONS[x]: spelled(tag) for x, tag in tags.items()},
                includes=includes or None, excludes=excludes or None)


def check_plan(plan, settings: dict) -> None:
    """Refuse a plan for a run it was not made for: its groups would be written whatever the
    run's input, tags or axes say."""
    made = getattr(plan, 'made_for', None) or {}
    differ = [key for key, value in made.items()
              if settings.get(key) != value and not (key == 'input' and None in (value, settings.get(key)))]
    if differ:
        def shown(value):
            return f"{len(value)} files" if isinstance(value, list) else (value or 'none')
        raise InputError(
            "This plan was made for other settings: "
            + "; ".join(f"{key} {shown(made[key])}, not {shown(settings.get(key))}" for key in differ)
            + ". Make the plan with the run's settings, or run it with the plan's.")


def _build_aggregative_plan(
    filepaths: list,
    concatenation_axes,
    common_dir: str,
    output_path: str,
    max_processes: int = 4,
    time_tag=None,
    channel_tag=None,
    z_tag=None,
    y_tag=None,
    x_tag=None,
    aggregative_group=None,
) -> AggregativePlan:
    """Determine output groups and allocate ``file_workers``.

    Uses dummy shapes for FileSet — grouping is purely tag-based and correct
    regardless of shape values.  Each worker reads the actual shapes
    (``aggregative_path.convert_group``).
    """
    from eubi_bridge.conversion.fileset_io import FileSet

    axes_str = 'tczyx'
    axlist = check_tags_and_axes(concatenation_axes, dict(
        t=time_tag, c=channel_tag, z=z_tag, y=y_tag, x=x_tag))

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
    fileset.check_places(axlist)
    for axis in axlist:
        fileset.concatenate_along(axis)

    file_groups: dict = {}
    for src_path, output_key in fileset.path_dict.items():
        rel      = os.path.relpath(output_key, common_dir)
        rel      = os.sep.join(p for p in rel.split(os.sep) if p != '..')
        name     = _short_name(os.path.splitext(rel)[0].replace(os.sep, '-'))
        # The derived name says which axes were concatenated (img_t0_zset), and
        # that is what keeps sibling outputs apart when one run produces several
        # groups.  The group is therefore prefixed rather than replacing it.
        name     = prefix_with_group(name, aggregative_group)
        # store_join, not os.path.join: on Windows the latter put a backslash into
        # a remote key (".../out\\name.zarr"), so an S3 output landed beside, not
        # under, the requested prefix
        from eubi_bridge.core.output_store import store_join
        full_out = store_join(output_path, name)
        if full_out not in file_groups:
            file_groups[full_out] = {'source_files': []}
        file_groups[full_out]['source_files'].append(src_path)

    n_outputs    = len(file_groups)
    file_workers = min(n_outputs, max_processes)

    joined = {f for grp in file_groups.values() for f in grp['source_files']}
    _warn_left_out([f for f in filepaths if f not in joined], len(filepaths))
    from eubi_bridge.conversion.fileset_io import _shared_folder_length, find_tag

    def tag_named(files):
        """The folders the files differ in are a tag's (Blue/, Red/; frames/000/)."""
        cut = _shared_folder_length(files)
        return all(any(find_tag(os.path.dirname(f)[cut:] + os.sep, str(item),
                                not isinstance(tag, tuple))
                       for tag in fileset.dimension_tags
                       for item in (tag if isinstance(tag, tuple) else (tag,)))
                   for f in files)

    spread = next((grp['source_files'] for grp in file_groups.values()
                   if len({os.path.dirname(f) for f in grp['source_files']}) > 1
                   and not tag_named(grp['source_files'])), None)
    if spread:
        folders = sorted({os.path.basename(os.path.dirname(f)) for f in spread})
        logger.warning(f"{len(spread)} files from {len(folders)} folders "
                       f"({', '.join(folders[:4])}{' ...' if len(folders) > 4 else ''}) are "
                       f"joined into one output. If they do not belong together, narrow the "
                       f"input with --includes or --excludes.")

    outputs = [
        AggregativeOutputInfo(output_path=out_path, source_files=grp['source_files'])
        for out_path, grp in file_groups.items()
    ]

    return AggregativePlan(
        n_outputs=n_outputs,
        outputs=outputs,
        file_workers=file_workers,
        tag_folders=fileset.tag_folders,
        output_path=str(output_path),
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
    the job's ``max_processes`` and the number of output groups it finds.
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


def _call(task):
    """Run one pool task: a picklable ``functools.partial`` of a worker function."""
    return task()


def dispatch_mixed_jobs(unary_jobs: list, aggregative_jobs: list,
                        refused: list | None = None) -> list:
    """A conversion table's one-to-one jobs and aggregative groups in ONE pool, running
    at the same time (roadmap step 3): every unary job and every aggregative output is
    one task, up to ``max_processes`` at once, the biggest first (results in the table's
    order). The run's memory
    budget is split over the tasks running at once, as for unary jobs; an aggregative
    output counts as one task however many files it joins. Failures are reported once,
    at the end, for all of them.

    On SLURM the two kinds still run one after the other, as in 0.1.5.

    With *refused* (a list), a group whose plan is refused (two files at one place, a
    tag no file carries) is appended to it as ``(job, reason)`` and the other jobs still
    run; without it, the refusal ends the run as before."""
    import functools

    from eubi_bridge.core.memory import main_process_mb, process_mb
    jobs = list(unary_jobs) + list(aggregative_jobs)
    if not jobs:
        return []
    first = jobs[0].cluster
    if first.on_slurm:
        results = dispatch_unary_jobs(unary_jobs) if unary_jobs else []
        if aggregative_jobs:
            results += dispatch_aggregative_jobs(aggregative_jobs)
        return results

    planned = []
    for job in aggregative_jobs:
        try:
            groups, _, job_kwargs = _aggregative_groups(
                job.input_path, output_path=job.output_path, includes=job.includes,
                excludes=job.excludes, time_tag=job.time_tag, channel_tag=job.channel_tag,
                z_tag=job.z_tag, y_tag=job.y_tag, x_tag=job.x_tag,
                concatenation_axes=job.concatenation_axes, plan=None,
                **job.to_conversion_kwargs())
        except ValueError as exc:             # InputError too: the group's files, not a bug
            if refused is None:
                raise
            refused.append((job, str(exc)))
            continue
        planned += [(paths, out, job_kwargs) for out, paths in groups.items()]
    if not unary_jobs and not planned:
        return []

    running = max(1, min(first.max_processes, len(unary_jobs) + len(planned)))
    use_threading = first.use_threading
    per_process = process_mb(first, running, 0.0 if use_threading else main_process_mb())
    memory = {'concurrent_jobs': running, 'process_memory_mb': per_process}
    if not first.on_slurm:
        # so the conversions still busy can take what the finished ones held
        from eubi_bridge.core.memory_share import plan_tasks
        plan_tasks(len(unary_jobs) + len(planned), running, per_process * running)
    tasks = [functools.partial(unary_worker_sync, job.model_copy(update={
                 'cluster': job.cluster.model_copy(update=memory)}))
             for job in unary_jobs]
    tasks += [functools.partial(aggregative_worker_from_paths, paths, out,
                                {**job_kwargs, **memory})
              for paths, out, job_kwargs in planned]
    logger.info(f"Conversion table: {len(unary_jobs)} one-to-one job(s) and "
                f"{len(planned)} aggregative output(s) in one pool, {running} at once, "
                f"the biggest first.")

    if first.jvm_memory:
        os.environ['EUBI_JVM_MEMORY'] = str(first.jvm_memory)
    exc_cls, exc_kw = _build_executor(first.model_copy(update={'max_processes': running}),
                                      use_threading)
    order = _largest_first([job.input_path for job in unary_jobs]
                           + [paths for paths, _out, _kw in planned])
    return asyncio.run(_dispatch(tasks, _call, exc_cls, exc_kw, use_threading,
                                 label="Conversion table",
                                 tsc=first.tensorstore_data_copy_concurrency, order=order))


def dispatch_aggregative_jobs(jobs: list) -> list:
    """Dispatch several aggregative jobs, one after another.

    Each job concatenates its own set of files, so they are run sequentially
    rather than concurrently: every one already parallelises internally across
    its inputs, and overlapping them would multiply the peak memory by the
    number of groups.

    A failure is reported with the group it came from and does not abandon the
    remaining groups, since one bad group should not discard a long batch.
    """
    results = []
    failures = []
    for index, job in enumerate(jobs):
        label = getattr(job, "aggregative_group", None) or f"group {index + 1}"
        try:
            results.append(dispatch_aggregative_job(job))
        except Exception as exc:                       # noqa: BLE001
            _raise_version_error([exc])                # every group would fail alike
            logger.error(f"Aggregative {label} failed: {exc}")
            failures.append((label, exc))

    if failures:
        from eubi_bridge.utils.errors import TasksFailedError
        summary = "\n".join(f"  {label}: {exc}" for label, exc in failures)
        raise TasksFailedError(
            f"{len(failures)}/{len(jobs)} aggregative group(s) failed:\n{summary}")
    return results


# ---------------------------------------------------------------------------
# Metadata collection  (read-only; keeps flat-dict interface)
# ---------------------------------------------------------------------------

def _short_name(name: str, limit: int = 120) -> str:
    """*name* within what a path can hold (joining 50 channels names them all); the rest goes
    into a short hash, so names stay apart."""
    if len(name) <= limit:
        return name
    import hashlib
    return f"{name[:limit - 9]}-{hashlib.sha1(name.encode()).hexdigest()[:8]}"


async def run_metadata_collection_from_filepaths(input_path, **global_kwargs):
    """Collect metadata from image files in parallel.

    Args:
        input_path: Path to files / directory / CSV-XLSX.
        **global_kwargs: Configuration parameters (supports use_threading flag).

    Returns:
        List of metadata dicts, one per file.
    """
    df = take_filepaths(input_path, **global_kwargs)
    from eubi_bridge.utils.path_utils import input_mains
    df = df.assign(input_path=input_mains(list(df["input_path"])))
    df = df[df["input_path"].notna()].drop_duplicates("input_path").reset_index(drop=True)

    max_workers   = int(global_kwargs.get("max_processes", 4))
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
                _shutdown_pool(temp)

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
                _shutdown_pool(pool)
            except Exception as e:
                logger.warning(f"Error shutting down meta pool: {e}")
                try:
                    pool.shutdown(wait=False)
                except Exception as e2:
                    logger.error(f"Force shutdown failed: {e2}")

    # a file asked for several scenes gives one summary per scene
    results = [x for r in results for x in (r if isinstance(r, list) else [r])]
    failed = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.error(f"[Meta] Task {i} failed: {r}")
            failed.append((i, str(r)))
        elif isinstance(r, dict) and r.get('status') == 'error':
            logger.error(f"[Meta] Task {i} failed: {r.get('error')}")
            failed.append((i, r.get('error', 'Unknown error')))
    if failed:
        _raise_version_error(results)
        from eubi_bridge.utils.errors import TasksFailedError
        summary = "\n".join(f"  Task {i}: {e}" for i, e in failed)
        raise TasksFailedError(
            f"Metadata collection failed for {len(failed)}/{len(results)} files:\n{summary}"
        )
    return results


# ---------------------------------------------------------------------------
# Aggregative (concatenation) path  — unchanged flat-dict interface
# ---------------------------------------------------------------------------

def _parse_tag(tag):
    """A tag as the files are read with it: several names as a list, without spaces and
    empty items (`Blue, Red,`), as FileSet reads them; a prefix or one name as itself."""
    if isinstance(tag, str) and ',' not in tag:
        return tag
    elif isinstance(tag, (str, tuple, list)):
        items = tag.split(',') if isinstance(tag, str) else tag
        return [str(item).strip() for item in items if str(item).strip()]
    elif tag is None:
        return None
    else:
        raise ValueError("tag must be a string, tuple, or list")


def _warn_left_out(left_out: list, total: int) -> None:
    """One warning naming the files an aggregative run does not convert."""
    if left_out:
        shown = ", ".join(os.path.basename(f) for f in left_out[:5])
        logger.warning(f"{len(left_out)} of {total} file(s) do not have every tag and are not "
                       f"converted: {shown}{' ...' if len(left_out) > 5 else ''}")


def _parse_filepaths_with_tags(filepaths, tags):
    """The files with every given tag: a file with one of them has no place in the output."""
    from eubi_bridge.conversion.fileset_io import read_tag
    # one file named twice is one file, also spelled two ways (a/b.tif, a\b.tif)
    filepaths = list({os.path.normcase(os.path.abspath(path)): path
                      for path in reversed(list(filepaths))}.values())[::-1]
    places = [read_tag(filepaths, _parse_tag(tag)) for tag in tags if tag is not None]
    return [path for path in filepaths if all(path in place for place in places)]


def _slurm_worker_init(tensorstore_data_copy_concurrency='default'):
    """Worker initialiser for Dask SLURM/LocalCluster workers."""
    from eubi_bridge.conversion.worker_init import initialize_worker_process
    initialize_worker_process(tensorstore_data_copy_concurrency)


def _aggregative_slurm_task(input_path, output_path, kwargs_dict):
    """Module-level sync wrapper for aggregative conversion on a Dask worker."""
    import concurrent.futures

    def _run_in_fresh_thread():
        from eubi_bridge.utils.jvm_manager import soft_start_jvm
        from eubi_bridge.utils.optional_deps import dask_config
        soft_start_jvm()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with dask_config(scheduler='synchronous'):
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

    max_workers     = int(kwargs.get('max_processes', 4))
    slurm_time      = kwargs.get('slurm_time', '24:00:00')
    slurm_mem       = kwargs.get('slurm_worker_memory', '3GB')
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
            # the whole aggregative run is this one SLURM job: its memory is the run's
            future = client.submit(
                _aggregative_slurm_task, input_path, output_path,
                {**kwargs, 'memory_budget': slurm_mem}, pure=False, retries=0,
            )
            logger.info("Aggregative SLURM job submitted; waiting...")
            result = client.gather(future)
    finally:
        cluster.close()
        logger.info("SLURM cluster shut down.")
    return result


def _aggregative_groups(
        input_path,
        output_path,
        series_index: Union[int, tuple] = None,
        time_tag: Union[str, tuple] = None,
        channel_tag: Union[str, tuple] = None,
        z_tag: Union[str, tuple] = None,
        y_tag: Union[str, tuple] = None,
        x_tag: Union[str, tuple] = None,
        concatenation_axes: Union[int, tuple, str] = None,
        metadata_reader: str = 'bfio',
        plan: 'AggregativePlan | None' = None,
        **kwargs,
) -> tuple:
    """``(groups, file_workers, job_kwargs)`` of an aggregative conversion, planned in
    the main process: ``{output path: its files}``, the outputs written at once, and
    the plain settings each worker gets.

    If *plan* is provided the pre-computed group assignments and worker
    counts are used directly (no re-planning, no header reads).  Without a
    plan the groups are derived via pure filename matching and worker counts
    are auto-computed from ``max_processes`` ÷ ``n_outputs``.
    """
    from eubi_bridge.core.config_models import ClusterConfig

    verbose = kwargs.get('verbose', None)

    # ── Group assignments ─────────────────────────────────────────────────
    if plan is not None:
        if plan.output_path and output_path and (
                os.path.normcase(os.path.abspath(plan.output_path))
                != os.path.normcase(os.path.abspath(str(output_path)))):
            raise InputError(f"This plan writes to {plan.output_path}, not {output_path}: make "
                             f"the plan with that output.")
        groups       = {out.output_path: out.source_files for out in plan.outputs}
        file_workers = plan.file_workers
        folders      = plan.tag_folders
        if verbose:
            logger.info(
                f"Using pre-computed plan: {len(groups)} group(s), "
                f"file_workers={file_workers}"
            )
    else:
        df = take_filepaths(
            input_path, series_index=series_index,
            output_path=output_path, metadata_reader=metadata_reader, **kwargs,
        )
        from eubi_bridge.conversion.fileset_io import channel_tag_format_error, read_tag
        filepaths          = df.input_path.to_numpy().tolist()

        for tag_name, tag_value in dict(
            time_tag=time_tag, channel_tag=channel_tag,
            z_tag=z_tag, y_tag=y_tag, x_tag=x_tag,
        ).items():
            if tag_value is None or read_tag(filepaths, _parse_tag(tag_value)):
                continue
            # A single (non-list) channel tag is treated as a prefix followed by a
            # number.  If it matches filenames but none carry a numeric suffix, the
            # user almost certainly meant a categorical (comma-separated) tag — fail
            # early with an actionable message instead of an opaque worker error.
            names = [Path(p).name for p in filepaths]
            if tag_name == 'channel_tag' and any(str(channel_tag) in p for p in filepaths):
                raise InputError(channel_tag_format_error(channel_tag, names))
            # tags are read as written: say so when the letter case is all that differs
            parsed = _parse_tag(tag_value)
            hint = ""
            if isinstance(parsed, str):
                found = next((m.group(0) for n in names for m in [re.search(
                    re.escape(parsed.rstrip('/\\')) + r"(?=\d)", n, re.IGNORECASE)] if m), None)
                if found:
                    hint = f" Tags are read with their letter case: did you mean --{tag_name} {found}?"
            raise InputError(f"--{tag_name} {tag_value} matches no file below the input.{hint} "
                             f"Files: {', '.join(names[:5])}{' ...' if len(names) > 5 else ''}")

        filepaths_accepted = _parse_filepaths_with_tags(
            filepaths, [time_tag, channel_tag, z_tag, y_tag, x_tag]
        )
        taken = set(filepaths_accepted)
        _warn_left_out([f for f in filepaths if f not in taken], len(filepaths))

        from eubi_bridge.conversion import aggregative_path
        # by name, before the files are grouped (a stray file with tags in its name
        # would otherwise break the grouping): an OME companion is left out, files of
        # mixed formats are refused (roadmap step 3)
        filepaths_accepted = aggregative_path.screen_group(filepaths_accepted)
        if not filepaths_accepted:
            raise InputError(f"None of the {len(filepaths)} file(s) can be joined: joining needs "
                             f"a tag that matches them, such as --z_tag or --channel_tag.")
        # one file: its folder (the common path of one file is the file, and the output's
        # name is taken below it: a store named "..zarr")
        common_dir  = os.path.commonpath([os.path.dirname(p) for p in filepaths_accepted])
        cluster_cfg = ClusterConfig(**{k: v for k, v in kwargs.items()
                                       if k in ClusterConfig.model_fields})
        inner_plan = _build_aggregative_plan(
            filepaths_accepted,
            concatenation_axes=concatenation_axes,
            common_dir=common_dir,
            output_path=output_path,
            max_processes=cluster_cfg.max_processes,

            time_tag=time_tag, channel_tag=channel_tag,
            z_tag=z_tag, y_tag=y_tag, x_tag=x_tag,
            aggregative_group=kwargs.get('aggregative_group'),
        )
        groups       = {out.output_path: out.source_files for out in inner_plan.outputs}
        file_workers = inner_plan.file_workers
        folders      = inner_plan.tag_folders

        if verbose:
            logger.info(
                f"Planned {inner_plan.n_outputs} output group(s): {list(groups.keys())}  "
                f"file_workers={file_workers}"
            )

    # ── Screen each output's files by name (roadmap step 3) ──
    # a plan made beforehand (the GUI's validate_aggregative): an OME companion is left
    # out (its files read it themselves); files of mixed formats are refused before
    # anything is opened
    from eubi_bridge.conversion import aggregative_path
    if plan is not None:
        screened = {out: aggregative_path.screen_group(paths, os.path.basename(out))
                    for out, paths in groups.items()}
        groups = {out: paths for out, paths in screened.items() if paths}
        file_workers = max(1, min(file_workers, len(groups)))

    # ── Build job kwargs (workers receive everything they need) ───────────
    job_kwargs = dict(
        time_tag=time_tag, channel_tag=channel_tag,
        z_tag=z_tag, y_tag=y_tag, x_tag=x_tag,
        concatenation_axes=concatenation_axes,
        metadata_reader=metadata_reader,
        series_index=series_index,
        tag_folders=folders,                # a worker reads its files' tags as the plan did
        **kwargs,
    )

    tsc = kwargs.get('tensorstore_data_copy_concurrency', 1)
    os.environ['EUBI_TENSORSTORE_DATA_COPY_CONCURRENCY'] = str(tsc)

    jvm_memory = kwargs.get('jvm_memory')
    if jvm_memory:
        os.environ['EUBI_JVM_MEMORY'] = str(jvm_memory)
    return groups, file_workers, job_kwargs


async def run_conversions_with_concatenation(input_path, output_path, **kwargs):
    """Aggregative conversion: plan groups in the main process (``_aggregative_groups``,
    same arguments), write in parallel.

    Each group is dispatched to a pool worker (ProcessPool by default;
    ThreadPool when ``use_threading=True``).  Workers run their own complete
    pipeline (open files → concat → write zarr) so no non-picklable objects
    cross the process boundary.
    """
    from eubi_bridge.core.config_models import ClusterConfig

    groups, file_workers, job_kwargs = _aggregative_groups(input_path, output_path, **kwargs)
    use_threading = kwargs.get('use_threading', False)
    if not use_threading and mp.current_process().daemon:
        # A dask worker (SLURM: dask-jobqueue's nanny starts it daemonic) cannot start
        # child processes ("daemonic processes are not allowed to have children"), so
        # the output groups run in threads there.
        logger.info("Running the output groups in threads: this process is a "
                    "daemon (a dask worker) and cannot start a process pool.")
        use_threading = True
    # Cap the pool at file_workers.
    pool_cluster = ClusterConfig(**{**{k: v for k, v in kwargs.items()
                                       if k in ClusterConfig.model_fields},
                                   'max_processes': file_workers})
    exc_cls, exc_kw = _build_executor(pool_cluster, use_threading)
    # each output group's memory: the run's budget over the groups written at once,
    # resolved here, before the pool starts
    from eubi_bridge.core.memory import main_process_mb, process_mb
    per_process = process_mb(pool_cluster.model_copy(update={'on_slurm': False}),
                             file_workers, 0.0 if use_threading else main_process_mb())
    from eubi_bridge.core.memory_share import plan_tasks
    plan_tasks(len(groups), file_workers, per_process * file_workers)

    loop = asyncio.get_running_loop()
    pool = None
    try:
        pool = exc_cls(**exc_kw)
        tasks = [
            loop.run_in_executor(
                pool, aggregative_worker_from_paths,
                # one output group per worker: file_workers write at once
                file_paths, out_path, {**job_kwargs, 'concurrent_jobs': file_workers,
                                       'process_memory_mb': per_process},
            )
            for out_path, file_paths in groups.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        if pool is not None:
            _shutdown_pool(pool)

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
