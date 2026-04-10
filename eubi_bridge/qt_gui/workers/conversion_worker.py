"""
Conversion worker — runs EuBIBridge.to_zarr() in a QThread.

Ports run_conversion.py logic; emits Qt signals instead of stdout JSON lines.
"""
from __future__ import annotations

import logging
import multiprocessing
import re as _re
import sys
import threading
import traceback

from PyQt6.QtCore import QThread, pyqtSignal

_ANSI_ESC = _re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-9;]*[ -/]*[@-~])')


# ── Logging bridge ────────────────────────────────────────────────────────────

_SEP = "\x01"  # field separator for structured log lines sent to LogWidget

class _QtLogHandler(logging.Handler):
    """Forwards log records to a Qt signal callback as structured lines.

    Format: ``HH:MM:SS\x01LEVELNAME\x01module.py:lineno\x01message``
    LogWidget recognises the separator and renders each segment in its own colour.
    """

    def __init__(self, emit_fn):
        super().__init__()
        self._emit_fn = emit_fn

    def emit(self, record: logging.LogRecord):
        try:
            import time as _time
            ts = _time.strftime("%H:%M:%S", _time.localtime(record.created))
            module = f"{record.filename}:{record.lineno}"
            msg = record.getMessage()
            structured = f"{ts}{_SEP}{record.levelname}{_SEP}{module}{_SEP}{msg}"
            self._emit_fn(structured)
        except Exception:
            pass


class _StdoutCapture:
    """Redirects print() output to a callback."""

    def __init__(self, emit_fn):
        self._emit_fn = emit_fn

    def write(self, text: str):
        if text and text.strip():
            for line in text.strip().splitlines():
                if line.strip():
                    self._emit_fn(line.rstrip())

    def flush(self):
        pass

    def isatty(self):
        return False


# ── Helper: parse range strings ───────────────────────────────────────────────

def _parse_range(range_str: str):
    if not range_str:
        return None
    parts = range_str.split(",")
    if len(parts) == 2:
        try:
            return (int(parts[0].strip()), int(parts[1].strip()))
        except ValueError:
            pass
    return None


def _parse_index(value) -> int | list:
    if not value:
        return 0
    value = str(value).strip()
    parts = [p.strip() for p in value.split(",")]
    if len(parts) == 1:
        try:
            return int(parts[0])
        except ValueError:
            return 0
    try:
        return [int(p) for p in parts]
    except ValueError:
        return 0


def _glob_to_substring(patterns_str: str):
    if not patterns_str:
        return None
    parts = [p.strip() for p in patterns_str.split(",") if p.strip()]
    result = [p.strip("*") for p in parts if p.strip("*")]
    return result or None


def _build_kwargs(config: dict) -> dict:
    """Build EuBIBridge.to_zarr() kwargs from a camelCase React config dict.

    Mirrors the logic in run_conversion.py.
    """
    concat_config   = config.get("concatenation", {})
    cluster_config  = config.get("cluster", {})
    reader_config   = config.get("reader", {})
    conv_config     = config.get("conversion", {})
    downscale_config = config.get("downscaling", {})
    meta_config     = config.get("metadata", {})
    compression     = conv_config.get("compression", {})

    zarr_format = conv_config.get("zarrFormat", 2)
    compressor = compression.get("codec", "blosc")
    if compressor == "none":
        compressor = None
        compressor_params: dict = {}
    elif compressor == "blosc":
        shuffle_val = compression.get("bloscShuffle", "shuffle")
        if zarr_format == 2:
            shuffle_map = {"noshuffle": 0, "shuffle": 1, "bitshuffle": 2}
            if isinstance(shuffle_val, str):
                shuffle_val = shuffle_map.get(shuffle_val, 1)
        compressor_params = {
            "cname":   compression.get("bloscInnerCodec", "lz4"),
            "clevel":  compression.get("level", 5),
            "shuffle": shuffle_val,
        }
    else:
        compressor_params = {"level": compression.get("level", 5)}

    scene_idx  = "all" if reader_config.get("readAllScenes", True)  else _parse_index(reader_config.get("sceneIndices", "0"))
    mosaic_idx = "all" if reader_config.get("readAllTiles", True)   else _parse_index(reader_config.get("mosaicTileIndices", "0"))

    kwargs: dict = {
        "max_workers":              cluster_config.get("maxWorkers", 4),
        "queue_size":               cluster_config.get("queueSize", 10),
        "region_size_mb":           cluster_config.get("regionSizeMb", 64),
        "max_concurrency":          cluster_config.get("maxConcurrency", 4),
        "memory_per_worker":        cluster_config.get("memoryPerWorker", "4GB"),
        "on_local_cluster":         cluster_config.get("useLocalDask", False),
        "on_slurm":                 cluster_config.get("useSlurm", False),
        "scene_index":              scene_idx,
        "mosaic_tile_index":        mosaic_idx,
        "as_mosaic":                reader_config.get("readAsMosaic", False),
        "view_index":               _parse_index(reader_config.get("viewIndex", "0")),
        "phase_index":              _parse_index(reader_config.get("phaseIndex", "0")),
        "illumination_index":       _parse_index(reader_config.get("illuminationIndex", "0")),
        "rotation_index":           _parse_index(reader_config.get("rotationIndex", "0")),
        "sample_index":             _parse_index(reader_config.get("sampleIndex", "0")),
        "verbose":                  conv_config.get("verbose", False),
        "zarr_format":              zarr_format,
        "auto_chunk":               conv_config.get("autoChunk", True),
        "time_chunk":               conv_config.get("chunkTime", 1),
        "channel_chunk":            conv_config.get("chunkChannel", 1),
        "z_chunk":                  conv_config.get("chunkZ", 1),
        "y_chunk":                  conv_config.get("chunkY", 256),
        "x_chunk":                  conv_config.get("chunkX", 256),
        "time_shard_coef":          conv_config.get("shardTime", 1),
        "channel_shard_coef":       conv_config.get("shardChannel", 1),
        "z_shard_coef":             conv_config.get("shardZ", 1),
        "y_shard_coef":             conv_config.get("shardY", 1),
        "x_shard_coef":             conv_config.get("shardX", 1),
        "time_range":               _parse_range(conv_config.get("dimRangeTime", "")),
        "channel_range":            _parse_range(conv_config.get("dimRangeChannel", "")),
        "z_range":                  _parse_range(conv_config.get("dimRangeZ", "")),
        "y_range":                  _parse_range(conv_config.get("dimRangeY", "")),
        "x_range":                  _parse_range(conv_config.get("dimRangeX", "")),
        "overwrite":                conv_config.get("overwrite", False),
        "override_channel_names":   conv_config.get("overrideChannelNames", False),
        "channel_intensity_limits": "from_dtype" if meta_config.get("channelIntensityLimits", "from_datatype") == "from_datatype" else "from_array",
        "metadata_reader":          meta_config.get("metadataReader", "bioio"),
        "save_omexml":              conv_config.get("saveOmeXml", True),
        "squeeze":                  conv_config.get("squeezeDimensions", True),
        "skip_dask":                conv_config.get("skipDask", False),
        "dtype":                    conv_config.get("dataType", "auto") or "auto",
        "n_layers":                 None if downscale_config.get("autoDetectLayers", True) else downscale_config.get("numLayers", 4),
        "min_dimension_size":       downscale_config.get("minDimSize", 64),
        "downscale_method":         downscale_config.get("downscaleMethod", "simple"),
        "time_scale_factor":        downscale_config.get("scaleTime", 1),
        "channel_scale_factor":     downscale_config.get("scaleChannel", 1),
        "z_scale_factor":           downscale_config.get("scaleZ", 1),
        "y_scale_factor":           downscale_config.get("scaleY", 2),
        "x_scale_factor":           downscale_config.get("scaleX", 2),
        "apply_smart_downscaling":  downscale_config.get("applySmartDownscaling", False),
        "z_smart_scale_factor":     downscale_config.get("smartScaleZ") or None,
        "y_smart_scale_factor":     downscale_config.get("smartScaleY") or None,
        "x_smart_scale_factor":     downscale_config.get("smartScaleX") or None,
        "time_smart_scale_factor":  downscale_config.get("smartScaleTime") or None,
        "compressor":               compressor,
        "compressor_params":        compressor_params,
    }

    if conv_config.get("autoChunk", True):
        kwargs["target_chunk_mb"] = conv_config.get("targetChunkSizeMb", 32)

    # Physical scale overrides
    if meta_config.get("overridePhysicalScale", False):
        for ax in ("time", "z", "y", "x"):
            key = f"scale{ax.capitalize()}"
            ukey = f"unit{ax.capitalize()}"
            if meta_config.get(key, ""):
                try:
                    kwargs[f"{ax}_scale"] = float(meta_config[key])
                except ValueError:
                    pass
            unit_default = "second" if ax == "time" else "micrometer"
            if meta_config.get(ukey, "") and meta_config[ukey] != unit_default:
                kwargs[f"{ax}_unit"] = meta_config[ukey]

    return kwargs


# ── Worker ────────────────────────────────────────────────────────────────────

class ConversionWorker(QThread):
    """Runs EuBIBridge.to_zarr() in a background thread.

    Signals:
        log_line(str)   — each log line from the conversion
        progress(int)   — 0-100 progress (placeholder; ebridge doesn't emit progress yet)
        finished()      — conversion completed successfully
        failed(str)     — conversion failed; argument is the traceback string
    """

    log_line = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config = config
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        config = self._config

        def emit_log(msg: str):
            if not self._cancelled:
                self.log_line.emit(msg)

        # Set up logging capture
        handler = _QtLogHandler(emit_log)
        handler.setLevel(logging.DEBUG)

        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        root_logger.handlers.clear()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)

        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        sys.stdout = _StdoutCapture(emit_log)
        sys.stderr = _StdoutCapture(emit_log)

        # --- subprocess worker log capture via multiprocessing queue ---
        from eubi_bridge.mp_logging_setup import setup_mp_logging_with_worker_init
        import eubi_bridge.conversion.converter as _conv_module

        # Use a spawn-context Queue instead of Manager().Queue().
        # Manager() forks the Qt process on Linux (the default start method),
        # and fork inside a Qt application deadlocks because forked children
        # inherit Qt's locked internal mutexes without the threads that hold them.
        # A spawn-context Queue uses only OS pipes/semaphores and is fully
        # picklable for passing as initargs to spawn ProcessPoolExecutor workers.
        _log_queue = multiprocessing.get_context("spawn").Queue()

        _stop_drain = threading.Event()
        _log_q_ref = _log_queue

        def _drain_worker():
            while not _stop_drain.is_set() or not _log_q_ref.empty():
                try:
                    msg = _log_q_ref.get(timeout=0.1)
                    if isinstance(msg, str):
                        clean = _ANSI_ESC.sub('', msg).strip()
                        if clean:
                            emit_log(clean)
                except Exception:
                    pass

        _drain_thread = threading.Thread(target=_drain_worker, daemon=True)
        _drain_thread.start()

        # Patch converter's executor classes so subprocess workers use queue logging
        _orig_ppe = _conv_module.ProcessPoolExecutor
        _orig_tpe = _conv_module.ThreadPoolExecutor

        class _LoggingPPE(_orig_ppe):
            def __init__(self, max_workers=None, **kw):
                raw = kw.get('initargs', (1,))[0] if 'initargs' in kw else 1
                tsc = raw if isinstance(raw, int) else 1
                kw['initializer'] = setup_mp_logging_with_worker_init
                kw['initargs'] = (_log_q_ref, tsc)
                super().__init__(max_workers, **kw)

        class _LoggingTPE(_orig_tpe):
            def __init__(self, max_workers=None, **kw):
                raw = kw.get('initargs', (1,))[0] if 'initargs' in kw else 1
                tsc = raw if isinstance(raw, int) else 1
                kw['initializer'] = setup_mp_logging_with_worker_init
                kw['initargs'] = (_log_q_ref, tsc)
                super().__init__(max_workers, **kw)

        _conv_module.ProcessPoolExecutor = _LoggingPPE
        _conv_module.ThreadPoolExecutor = _LoggingTPE
        # --- end subprocess log setup ---

        _success = False
        _failure_tb = None

        try:
            emit_log("Initializing EuBI-Bridge...")
            from eubi_bridge.ebridge import EuBIBridge  # type: ignore

            bridge = EuBIBridge()

            input_paths_list = config.get("inputPaths", [])
            input_path = input_paths_list if input_paths_list else config.get("inputPath", "")
            output_path = config.get("outputPath", "")
            include_pattern = config.get("includePattern", "")
            exclude_pattern = config.get("excludePattern", "")

            concat = config.get("concatenation", {})
            kwargs = _build_kwargs(config)

            includes = _glob_to_substring(include_pattern)
            excludes = _glob_to_substring(exclude_pattern)

            if isinstance(input_path, list):
                emit_log(f"Input: {len(input_path)} file(s) selected")
            else:
                emit_log(f"Input: {input_path}")
            emit_log(f"Output: {output_path}")
            emit_log(f"Zarr Format: v{kwargs.get('zarr_format', 2)}")
            emit_log(f"Max Workers: {kwargs.get('max_workers', 4)}")
            emit_log("Starting conversion...")

            bridge.to_zarr(
                input_path=input_path,
                output_path=output_path or None,
                includes=includes,
                excludes=excludes,
                time_tag=concat.get("timeTag") or None,
                channel_tag=concat.get("channelTag") or None,
                z_tag=concat.get("zTag") or None,
                y_tag=concat.get("yTag") or None,
                x_tag=concat.get("xTag") or None,
                concatenation_axes=concat.get("concatenationAxes") or None,
                **kwargs,
            )

            _success = not self._cancelled

        except Exception:
            _failure_tb = traceback.format_exc()
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            root_logger.handlers.clear()
            for h in original_handlers:
                root_logger.addHandler(h)
            # Restore patched executors and stop drainer
            _conv_module.ProcessPoolExecutor = _orig_ppe
            _conv_module.ThreadPoolExecutor = _orig_tpe
            _stop_drain.set()
            _drain_thread.join(timeout=5.0)
            try:
                _log_queue.close()
            except Exception:
                pass
            # Emit finished/failed AFTER the drain thread has joined so that all
            # log_line signals are already queued before the main thread processes
            # the completion signal (which sets _worker=None and breaks connections).
            if _failure_tb is not None and not self._cancelled:
                self.failed.emit(_failure_tb)
            elif _success:
                self.finished.emit()
