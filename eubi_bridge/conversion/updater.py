import os
import time
import asyncio
import pickle
import numpy as np, pandas as pd
import multiprocessing as mp
# from distributed import LocalCluster, Client
import logging
from typing import Union, Optional, Any, Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from eubi_bridge.conversion.metadata_update_worker import update_worker_sync
from eubi_bridge.utils.convenience import take_filepaths
from eubi_bridge.utils.logging_config import get_logger

logger = get_logger(__name__)
# Suppress noisy logs
# logging.getLogger('distributed.diskutils').setLevel(logging.CRITICAL)
# logging.getLogger('distributed.worker').setLevel(logging.WARNING)
# logging.getLogger('distributed.scheduler').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)


async def run_updates_on_filepaths(
        input_path,
        **global_kwargs
):
    """
    Run parallel metadata updates where each job's parameters (including input/output dirs)
    are specified via kwargs or a CSV/XLSX file.

    Args:
        input_path:
            - list of file paths, OR
            - path to a CSV/XLSX with at least 'input_path' or 'filepath' column.
        **global_kwargs: global defaults for all updates
    """

    df = take_filepaths(input_path, **global_kwargs)

    # --- Setup concurrency ---
    max_workers = int(global_kwargs.get("max_workers", 4))
    loop = asyncio.get_running_loop()

    def _run_one(row):
        """Run one conversion with unified kwargs."""
        job_kwargs = row.to_dict()
        input_path = job_kwargs.get('input_path')
        job_kwargs.pop('input_path')
        return loop.run_in_executor(
            pool,
            update_worker_sync,
            input_path,
            job_kwargs
        )

    # --- Run all conversions ---
    with ProcessPoolExecutor(max_workers) as pool:
        tasks = [_run_one(row) for _, row in df.iterrows()]
        results = await asyncio.gather(*tasks)

    return results


def run_updates(input_path,
                **kwargs
                ):
    return asyncio.run(run_updates_on_filepaths(
                              input_path,
                              **kwargs
                              )
                       )