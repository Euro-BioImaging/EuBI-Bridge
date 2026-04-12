"""
Conversion worker — manages a spawned conversion subprocess from a QThread.

The actual conversion (bridge.to_zarr) runs in a dedicated child process so
that cancel() can reliably kill the entire process tree — including all
ProcessPoolExecutor workers spawned by converter.py — without depending on
cooperation from asyncio.gather or concurrent.futures.
"""
from __future__ import annotations

import multiprocessing
import re as _re
import sys
import traceback

from PyQt6.QtCore import QThread, pyqtSignal

from eubi_bridge.qt_gui.workers._conv_subprocess import (
    _conversion_subprocess,
    _kill_process_tree,
)

_ANSI_ESC = _re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-9;]*[ -/]*[@-~])')


# ── Config helpers ────────────────────────────────────────────────────────────

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
    """Build EuBIBridge.to_zarr() kwargs from a camelCase config dict."""
    concat_config    = config.get("concatenation", {})
    cluster_config   = config.get("cluster", {})
    reader_config    = config.get("reader", {})
    conv_config      = config.get("conversion", {})
    downscale_config = config.get("downscaling", {})
    meta_config      = config.get("metadata", {})
    compression      = conv_config.get("compression", {})

    zarr_format = conv_config.get("zarrFormat", 2)
    compressor  = compression.get("codec", "blosc")
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
        "max_concurrent_scenes":    cluster_config.get("maxConcurrentScenes", 4),
        "memory_per_worker":        cluster_config.get("memoryPerWorker", "4GB"),
        "on_local_cluster":         cluster_config.get("useLocalDask", False),
        "on_slurm":                 cluster_config.get("useSlurm", False),
        "slurm_partition":          cluster_config.get("slurmPartition") or None,
        "slurm_account":            cluster_config.get("slurmAccount") or None,
        "slurm_time":               cluster_config.get("slurmTime", "24:00:00"),
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

    if meta_config.get("overridePhysicalScale", False):
        for ax in ("time", "z", "y", "x"):
            key  = f"scale{ax.capitalize()}"
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
    """Manages a spawned conversion subprocess.

    Signals:
        log_line(str)   — structured log line (forwarded from child process)
        progress(int)   — 0-100 progress placeholder
        finished()      — conversion completed successfully
        failed(str)     — conversion failed; argument is the traceback string
    """

    log_line = pyqtSignal(str)
    finished = pyqtSignal()
    failed   = pyqtSignal(str)

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config     = config
        self._cancelled  = False
        self._conv_proc: multiprocessing.Process | None = None

    def cancel(self):
        """Kill the conversion subprocess and its entire process tree."""
        self._cancelled = True
        proc = self._conv_proc
        if proc is not None and proc.is_alive():
            _kill_process_tree(proc.pid)

    def run(self):
        config = self._config
        input_paths_list = config.get("inputPaths", [])
        input_path  = input_paths_list if input_paths_list else config.get("inputPath", "")
        output_path = config.get("outputPath", "")
        concat      = config.get("concatenation", {})

        call_args = {
            "input_path":         input_path,
            "output_path":        output_path or None,
            "includes":           _glob_to_substring(config.get("includePattern", "")),
            "excludes":           _glob_to_substring(config.get("excludePattern", "")),
            "time_tag":           concat.get("timeTag")           or None,
            "channel_tag":        concat.get("channelTag")        or None,
            "z_tag":              concat.get("zTag")              or None,
            "y_tag":              concat.get("yTag")              or None,
            "x_tag":              concat.get("xTag")              or None,
            "concatenation_axes": concat.get("concatenationAxes") or None,
            "to_zarr_kwargs":     _build_kwargs(config),
        }

        # Emit startup info before spawning
        if isinstance(input_path, list):
            self.log_line.emit(f"Input: {len(input_path)} file(s) selected")
        else:
            self.log_line.emit(f"Input: {input_path}")
        self.log_line.emit(f"Output: {output_path}")
        self.log_line.emit(f"Zarr Format: v{call_args['to_zarr_kwargs'].get('zarr_format', 2)}")
        self.log_line.emit(f"Max Workers: {call_args['to_zarr_kwargs'].get('max_workers', 4)}")
        self.log_line.emit("Starting conversion...")

        ctx          = multiprocessing.get_context("spawn")
        log_queue    = ctx.Queue()
        result_queue = ctx.Queue()

        try:
            self._conv_proc = ctx.Process(
                target=_conversion_subprocess,
                args=(call_args, log_queue, result_queue),
                daemon=False,   # must be False — daemon processes cannot spawn children
            )
            self._conv_proc.start()
        except Exception:
            self.failed.emit(traceback.format_exc())
            return

        # Drain log queue while the process runs
        while self._conv_proc.is_alive():
            try:
                msg = log_queue.get(timeout=0.1)
                if not self._cancelled:
                    self.log_line.emit(msg)
            except Exception:
                pass

        # Drain any remaining messages after process exits
        while True:
            try:
                msg = log_queue.get_nowait()
                if not self._cancelled:
                    self.log_line.emit(msg)
            except Exception:
                break

        self._conv_proc.join(timeout=5)

        if self._cancelled:
            return

        # Read result
        try:
            status, data = result_queue.get_nowait()
        except Exception:
            self.failed.emit(
                f"Conversion process exited unexpectedly "
                f"(exit code {self._conv_proc.exitcode})"
            )
            return

        if status == "ok":
            self.finished.emit()
        else:
            self.failed.emit(data or "Conversion failed")
