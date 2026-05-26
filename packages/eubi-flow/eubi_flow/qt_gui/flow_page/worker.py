"""FlowRunWorker — runs execute_flow in a background QThread.

Mirrors the CLI execution path exactly:
  1. Create a dask LocalCluster (synchronously, before any asyncio)
  2. Connect a Client to it
  3. Run the asyncio event loop with execute_flow

Without a running dask cluster every da.compute() call inside
store_multiscale_async blocks the asyncio thread, causing the flow to
hang silently.  The cluster keeps compute off the event-loop thread.
"""
from __future__ import annotations

import asyncio
import threading
import traceback
from typing import TYPE_CHECKING

from PyQt6.QtCore import QThread, pyqtSignal

if TYPE_CHECKING:
    from eubi_flow.models import FlowSpec


class FlowRunWorker(QThread):
    """Background thread that runs a FlowSpec via execute_flow."""

    wave_status_changed = pyqtSignal(str, str)   # wave_id, status
    flow_finished       = pyqtSignal()
    flow_failed         = pyqtSignal(str)         # full traceback string
    log_line            = pyqtSignal(str)

    def __init__(
        self,
        flow_name: str,
        flow: "FlowSpec",
        max_workers: int = 4,
        **run_kwargs,
    ) -> None:
        super().__init__()
        self._flow_name   = flow_name
        self._flow        = flow
        self._max_workers = max_workers
        self._run_kwargs  = run_kwargs
        self._stop_event  = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------

    def run(self) -> None:
        cluster = None
        client  = None
        try:
            from dask.distributed import Client, LocalCluster
            self.log_line.emit(
                f"Starting dask LocalCluster ({self._max_workers} workers)…"
            )
            cluster = LocalCluster(
                n_workers=self._max_workers,
                threads_per_worker=1,
                dashboard_address=None,   # no web UI
                silence_logs=True,
            )
            client = Client(cluster)
            self.log_line.emit(f"Cluster ready: {cluster.scheduler_address}")
        except Exception:
            self.log_line.emit(
                "WARNING: could not start dask cluster — "
                "falling back to synchronous scheduler.\n" + traceback.format_exc()
            )

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_async(client))
        except Exception:
            self.flow_failed.emit(traceback.format_exc())
        finally:
            self._loop.close()
            self._loop = None
            if client:
                try:
                    client.close()
                except Exception:
                    pass
            if cluster:
                try:
                    cluster.close()
                except Exception:
                    pass

    async def _run_async(self, client) -> None:
        from eubi_flow.eubiflow import _run_flow_async

        def _status_cb(wave_id: str, status: str) -> None:
            self.wave_status_changed.emit(wave_id, status)
            self.log_line.emit(f"  [{wave_id}] {status}")

        def _error_cb(wave_id: str, tb: str) -> None:
            self.log_line.emit(f"  [{wave_id}] ERROR:\n{tb}")

        try:
            results = await _run_flow_async(
                self._flow_name,
                self._flow,
                client=client,
                max_concurrent=self._max_workers,
                on_log=self.log_line.emit,
                on_wave_status=_status_cb,
                on_wave_error=_error_cb,
                **self._run_kwargs,
            )
            n_fail = sum(1 for r in results if r.status == "failed")
            if n_fail:
                self.flow_failed.emit(
                    f"{len(results) - n_fail}/{len(results)} completed — "
                    f"{n_fail} failed.\nSee the log for per-wave tracebacks."
                )
            else:
                self.flow_finished.emit()
        except Exception:
            self.flow_failed.emit(traceback.format_exc())

    def stop(self) -> None:
        self._stop_event.set()
        if self._loop is not None:
            for task in asyncio.all_tasks(self._loop):
                self._loop.call_soon_threadsafe(task.cancel)
