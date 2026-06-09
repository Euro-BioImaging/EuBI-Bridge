#!/usr/bin/env python3
"""
Config manager helper for the EuBI-Bridge React GUI.

Actions
-------
  get [path]        - Read config file (or defaults if absent), emit camelCase JSON.
  save <path> <json>- Accept camelCase React config JSON, write canonical snake_case to file.
  reset             - Reset file to root_defaults, emit camelCase JSON of the defaults.

All output is a single JSON line on stdout.
Errors are emitted as {"error": "<message>"} with exit code 1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# snake_case (ebridge.py) → camelCase (React GUI) mapping
# ---------------------------------------------------------------------------

def _config_to_react(cfg: dict) -> dict:
    """Convert ebridge.py config dict (snake_case, 4 sections) to React GUI shape."""
    cluster = cfg.get("cluster", {})
    readers = cfg.get("readers", {})
    conv    = cfg.get("conversion", {})
    down    = cfg.get("downscale", {})
    concat  = cfg.get("concatenation", {})

    # Reconstruct compression dict from compressor + compressor_params
    compressor = conv.get("compressor", "blosc")
    cp = conv.get("compressor_params", {})
    if compressor is None:
        codec = "none"
        level = 0
        blosc_inner = "lz4"
        blosc_shuffle = "shuffle"
    elif compressor == "blosc":
        codec = "blosc"
        level = cp.get("clevel", 5)
        blosc_inner = cp.get("cname", "lz4")
        raw_shuffle = cp.get("shuffle", 1)
        # normalise numeric shuffle to string
        if isinstance(raw_shuffle, int):
            raw_shuffle = {0: "noshuffle", 1: "shuffle", 2: "bitshuffle"}.get(raw_shuffle, "shuffle")
        blosc_shuffle = raw_shuffle
    else:
        codec = compressor
        level = cp.get("level", 5)
        blosc_inner = "lz4"
        blosc_shuffle = "shuffle"

    n_layers = down.get("n_layers", None)

    return {
        "cluster": {
            "maxWorkers":           cluster.get("max_workers", 4),
            "queueSize":            cluster.get("queue_size", 4),
            "maxConcurrency":       cluster.get("max_concurrency", 4),
            "maxConcurrentScenes":  cluster.get("max_concurrent_scenes", 1),
            "regionSizeMb":         float(cluster.get("region_size_mb", 256)),
            "memoryPerWorker":      _parse_memory_gb(cluster.get("memory_per_worker", "4GB"), 4.0),
            "useLocalDask":         cluster.get("on_local_cluster", False),
            "useSlurm":             cluster.get("on_slurm", False),
            "bfTileSizeMb":         cluster.get("bf_tile_size_mb", 512.0),
            "jvmMemory":            _parse_jvm_gb(cluster.get("jvm_memory", "2g"), 2.0),
            "bfReadConcurrency":    cluster.get("bf_read_concurrency", 4),
        },
        "reader": {
            "readAsMosaic":         readers.get("as_mosaic", False),
            "readAllScenes":        True,
            "sceneIndices":         str(readers.get("scene_index", 0)) if readers.get("scene_index") not in (None, 0, "all") else "",
            "readAllTiles":         True,
            "mosaicTileIndices":    str(readers.get("mosaic_tile_index", 0)) if readers.get("mosaic_tile_index") not in (None, 0, "all") else "",
            "readAllViews":         True,
            "viewIndices":          str(readers.get("view_index", 0)) if readers.get("view_index") not in (None, 0, "all") else "",
            "concatViews":          readers.get("concat_views", False),
            "phaseIndex":           str(readers.get("phase_index", 0)),
            "readAllIlluminations": True,
            "illuminationIndices":  str(readers.get("illumination_index", 0)) if readers.get("illumination_index") not in (None, 0, "all") else "",
            "concatIlluminations":  readers.get("concat_illuminations", False),
            "rotationIndex":        str(readers.get("rotation_index", 0)),
            "sampleIndex":          str(readers.get("sample_index", 0)),
            "forceBioformats":      readers.get("force_bioformats", False),
        },
        "conversion": {
            "zarrFormat":           conv.get("zarr_format", 2),
            "dataType":             conv.get("dtype", "auto") or "auto",
            "verbose":              conv.get("verbose", False),
            "overwrite":            conv.get("overwrite", False),
            "squeezeDimensions":    conv.get("squeeze", True),
            "saveOmeXml":           conv.get("save_omexml", True),
            "overrideChannelNames": conv.get("override_channel_names", False),
            "skipDask":             conv.get("skip_dask", False),
            "autoChunk":            conv.get("auto_chunk", True),
            "targetChunkSizeMb":    conv.get("target_chunk_mb", 1),
            "chunkTime":            conv.get("time_chunk", 1),
            "chunkChannel":         conv.get("channel_chunk", 1),
            "chunkZ":               conv.get("z_chunk", 96),
            "chunkY":               conv.get("y_chunk", 96),
            "chunkX":               conv.get("x_chunk", 96),
            "shardTime":            conv.get("time_shard_coef", 1),
            "shardChannel":         conv.get("channel_shard_coef", 1),
            "shardZ":               conv.get("z_shard_coef", 3),
            "shardY":               conv.get("y_shard_coef", 3),
            "shardX":               conv.get("x_shard_coef", 3),
            "dimRangeTime":    "",
            "dimRangeChannel": "",
            "dimRangeZ":       "",
            "dimRangeY":       "",
            "dimRangeX":       "",
            "compression": {
                "codec":          codec,
                "level":          level,
                "bloscInnerCodec": blosc_inner,
                "bloscShuffle":   blosc_shuffle,
            },
        },
        "downscaling": {
            "autoDetectLayers":     n_layers is None,
            "numLayers":            n_layers if n_layers is not None else 4,
            "minDimSize":           down.get("min_dimension_size", 64),
            "downscaleMethod":      down.get("downscale_method", "simple"),
            "scaleTime":            down.get("time_scale_factor", 1),
            "scaleChannel":         down.get("channel_scale_factor", 1),
            "scaleZ":               down.get("z_scale_factor", 2),
            "scaleY":               down.get("y_scale_factor", 2),
            "scaleX":               down.get("x_scale_factor", 2),
            "applySmartDownscaling": down.get("apply_smart_downscaling", False),
            "smartScaleZ":          down.get("z_smart_scale_factor", None),
            "smartScaleY":          down.get("y_smart_scale_factor", None),
            "smartScaleX":          down.get("x_smart_scale_factor", None),
            "smartScaleTime":       down.get("time_smart_scale_factor", None),
        },
        "metadata": {
            "metadataReader":         conv.get("metadata_reader", "bfio"),
            "channelIntensityLimits": "from_datatype" if conv.get("channel_intensity_limits", "from_dtype") == "from_dtype" else "from_array",
            # Physical scale overrides cannot be stored in the config file
            "overridePhysicalScale": False,
            "scaleTime": "", "unitTime": "second",
            "scaleZ": "",    "unitZ": "micrometer",
            "scaleY": "",    "unitY": "micrometer",
            "scaleX": "",    "unitX": "micrometer",
        },
        "concatenation": {
            "concatenationAxes": concat.get("concatenation_axes", "") or "",
            "timeTag":           concat.get("time_tag", "")    or "",
            "channelTag":        concat.get("channel_tag", "") or "",
            "zTag":              concat.get("z_tag", "")       or "",
            "yTag":              concat.get("y_tag", "")       or "",
            "xTag":              concat.get("x_tag", "")       or "",
        },
    }


def _react_to_config(data: dict) -> dict:
    """Convert React GUI camelCase payload to ebridge.py snake_case config dict."""
    cluster_d  = data.get("cluster", {})
    conv_d     = data.get("conversion", {})
    down_d     = data.get("downscaling", {})
    reader_d   = data.get("reader", {})
    meta_d     = data.get("metadata", {})
    concat_d   = data.get("concatenation", {})
    comp_d     = conv_d.get("compression", {})

    # Compression
    codec = comp_d.get("codec", "blosc")
    if codec == "none":
        compressor = None
        compressor_params: dict = {}
    elif codec == "blosc":
        compressor = "blosc"
        shuffle_val = comp_d.get("bloscShuffle", "shuffle")
        compressor_params = {
            "cname":   comp_d.get("bloscInnerCodec", "lz4"),
            "clevel":  comp_d.get("level", 5),
            "shuffle": shuffle_val,
        }
    else:
        compressor = codec
        compressor_params = {"level": comp_d.get("level", 5)}

    n_layers = None if down_d.get("autoDetectLayers", True) else down_d.get("numLayers", 4)

    ci_limits = "from_dtype" if meta_d.get("channelIntensityLimits", "from_datatype") == "from_datatype" else "from_array"

    return {
        "cluster": {
            "on_local_cluster":                cluster_d.get("useLocalDask", False),
            "on_slurm":                        cluster_d.get("useSlurm", False),
            "use_threading":                   False,
            "max_workers":                     cluster_d.get("maxWorkers", 4),
            "queue_size":                      cluster_d.get("queueSize", 4),
            "region_size_mb":                  float(cluster_d.get("regionSizeMb", 256)),
            "max_concurrency":                 cluster_d.get("maxConcurrency", 4),
            "max_concurrent_scenes":           cluster_d.get("maxConcurrentScenes", 1),
            "memory_per_worker":               _gb_to_memory_str(cluster_d.get("memoryPerWorker", 4)),
            "tensorstore_data_copy_concurrency": 4,
            "max_retries":                     10,
            "bf_tile_size_mb":                 cluster_d.get("bfTileSizeMb", 512.0),
            "jvm_memory":                      _gb_to_jvm_str(cluster_d.get("jvmMemory", 2)),
            "bf_read_concurrency":             cluster_d.get("bfReadConcurrency", 4),
        },
        "readers": {
            "as_mosaic":            reader_d.get("readAsMosaic", False),
            "scene_index":          "all" if reader_d.get("readAllScenes", True) else _parse_int(reader_d.get("sceneIndices", "0")),
            "mosaic_tile_index":    None if reader_d.get("readAllTiles", True) else _parse_int(reader_d.get("mosaicTileIndices", "0")),
            "view_index":           "all" if reader_d.get("readAllViews", True) else _parse_int(reader_d.get("viewIndices", "0")),
            "concat_views":         reader_d.get("concatViews", False),
            "phase_index":          _parse_int(reader_d.get("phaseIndex", "0")),
            "illumination_index":   "all" if reader_d.get("readAllIlluminations", True) else _parse_int(reader_d.get("illuminationIndices", "0")),
            "concat_illuminations": reader_d.get("concatIlluminations", False),
            "rotation_index":       _parse_int(reader_d.get("rotationIndex", "0")),
            "sample_index":         _parse_int(reader_d.get("sampleIndex", "0")),
            "force_bioformats":     reader_d.get("forceBioformats", False),
        },
        "conversion": {
            "zarr_format":           conv_d.get("zarrFormat", 2),
            "auto_chunk":            conv_d.get("autoChunk", True),
            "target_chunk_mb":       conv_d.get("targetChunkSizeMb", 1),
            "time_chunk":            conv_d.get("chunkTime", 1),
            "channel_chunk":         conv_d.get("chunkChannel", 1),
            "z_chunk":               conv_d.get("chunkZ", 96),
            "y_chunk":               conv_d.get("chunkY", 96),
            "x_chunk":               conv_d.get("chunkX", 96),
            "time_shard_coef":       conv_d.get("shardTime", 1),
            "channel_shard_coef":    conv_d.get("shardChannel", 1),
            "z_shard_coef":          conv_d.get("shardZ", 3),
            "y_shard_coef":          conv_d.get("shardY", 3),
            "x_shard_coef":          conv_d.get("shardX", 3),
            "compressor":            compressor,
            "compressor_params":     compressor_params,
            "overwrite":             conv_d.get("overwrite", False),
            "override_channel_names": conv_d.get("overrideChannelNames", False),
            "channel_intensity_limits": ci_limits,
            "metadata_reader":       meta_d.get("metadataReader", "bfio"),
            "save_omexml":           conv_d.get("saveOmeXml", True),
            "squeeze":               conv_d.get("squeezeDimensions", True),
            "skip_dask":             conv_d.get("skipDask", False),
            "verbose":               conv_d.get("verbose", False),
            "dimension_order":       "tczyx",
            "dtype":                 conv_d.get("dataType", "auto") or "auto",
            "time_range":            None,
            "channel_range":         None,
            "z_range":               None,
            "y_range":               None,
            "x_range":               None,
        },
        "downscale": {
            "time_scale_factor":        down_d.get("scaleTime", 1),
            "channel_scale_factor":     down_d.get("scaleChannel", 1),
            "z_scale_factor":           down_d.get("scaleZ", 2),
            "y_scale_factor":           down_d.get("scaleY", 2),
            "x_scale_factor":           down_d.get("scaleX", 2),
            "n_layers":                 n_layers,
            "min_dimension_size":       down_d.get("minDimSize", 64),
            "downscale_method":         down_d.get("downscaleMethod", "simple"),
            "apply_smart_downscaling":  down_d.get("applySmartDownscaling", False),
            "z_smart_scale_factor":     down_d.get("smartScaleZ") or None,
            "y_smart_scale_factor":     down_d.get("smartScaleY") or None,
            "x_smart_scale_factor":     down_d.get("smartScaleX") or None,
            "time_smart_scale_factor":  down_d.get("smartScaleTime") or None,
        },
        "concatenation": {
            "concatenation_axes": concat_d.get("concatenationAxes") or None,
            "time_tag":           concat_d.get("timeTag")    or None,
            "channel_tag":        concat_d.get("channelTag") or None,
            "z_tag":              concat_d.get("zTag")       or None,
            "y_tag":              concat_d.get("yTag")       or None,
            "x_tag":              concat_d.get("xTag")       or None,
        },
    }


def _parse_int(value) -> int:
    try:
        return int(str(value).strip().split(",")[0])
    except (ValueError, TypeError):
        return 0


def _parse_memory_gb(value, default: float = 4.0) -> float:
    """Parse a memory string like '4GB', '4gb', '0.5GB', '512MB' → float GB.
    If *value* is already numeric it is returned as-is (assumed GB)."""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower()
    for suffix, divisor in [("gb", 1), ("mb", 1024), ("g", 1), ("m", 1024), ("b", 1)]:
        if s.endswith(suffix):
            try:
                return float(s[: -len(suffix)]) / divisor
            except ValueError:
                break
    try:
        return float(s)
    except ValueError:
        return default


def _parse_jvm_gb(value, default: float = 2.0) -> float:
    """Parse a JVM heap string like '2g', '4G', '2048m' → float GB.
    If *value* is already numeric it is returned as-is (assumed GB)."""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower()
    if s.endswith("m"):
        try:
            return float(s[:-1]) / 1024
        except ValueError:
            return default
    for suffix in ("gb", "g", "b"):
        if s.endswith(suffix):
            try:
                return float(s[: -len(suffix)])
            except ValueError:
                break
    try:
        return float(s)
    except ValueError:
        return default


def _gb_to_memory_str(val, default: str = "4GB") -> str:
    """Float GB → '4GB' or '4.5GB' string for Dask memory_per_worker."""
    try:
        gb = float(val)
        return f"{int(gb)}GB" if gb == int(gb) else f"{gb:.1f}GB"
    except (TypeError, ValueError):
        return str(val) if val else default


def _gb_to_jvm_str(val, default: str = "2g") -> str:
    """Float GB → '4g' string for JVM -Xmx (rounded to nearest integer GB)."""
    try:
        return f"{max(1, round(float(val)))}g"
    except (TypeError, ValueError):
        return str(val) if val else default


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def _resolve_configpath(config_path: str | None) -> str | None:
    """
    Expand ~ and normalise the path.  EuBIBridge._get_json_path() now handles
    both cases: a directory → appends .eubi_config.json; a .json file → uses
    it directly.  So we just expand and pass through.
    """
    if not config_path:
        return None
    return str(Path(config_path).expanduser())


def action_get(config_path: str | None) -> None:
    from eubi_bridge.ebridge import EuBIBridge

    resolved = _resolve_configpath(config_path)
    kwargs: dict = {}
    if resolved:
        kwargs["configpath"] = resolved

    bridge = EuBIBridge(**kwargs)
    cfg = bridge.config  # lazy-loads (or creates) the JSON file

    result = _config_to_react(cfg)
    # Also report which file was actually used
    result["_configPath"] = str(bridge._get_json_path())
    print(json.dumps(result))


def action_save(config_path: str, react_json_str: str) -> None:
    from eubi_bridge.ebridge import EuBIBridge

    react_data = json.loads(react_json_str)
    new_cfg = _react_to_config(react_data)

    resolved = _resolve_configpath(config_path) or config_path
    bridge = EuBIBridge(configpath=resolved)
    bridge.config = new_cfg  # setter writes to JSON immediately

    result = _config_to_react(bridge.config)
    result["_configPath"] = str(bridge._get_json_path())
    print(json.dumps(result))


def action_reset(config_path: str | None) -> None:
    from eubi_bridge.ebridge import EuBIBridge

    resolved = _resolve_configpath(config_path)
    kwargs: dict = {}
    if resolved:
        kwargs["configpath"] = resolved

    bridge = EuBIBridge(**kwargs)
    bridge.reset_config()

    result = _config_to_react(bridge.config)
    result["_configPath"] = str(bridge._get_json_path())
    print(json.dumps(result))


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    All input is read as a single JSON object from stdin:
        { "action": "get",   "configPath": "<optional dir or .json file>" }
        { "action": "save",  "configPath": "<optional>", "config": { ... } }
        { "action": "reset", "configPath": "<optional>" }

    This avoids ALL Windows command-line argument quoting / backslash escaping
    issues since nothing is passed via argv.
    """
    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({"error": "No input on stdin"}))
        sys.exit(1)

    try:
        req = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"Invalid JSON on stdin: {exc}"}))
        sys.exit(1)

    action = req.get("action", "")
    config_path = req.get("configPath") or None   # None if missing or empty string

    try:
        if action == "get":
            action_get(config_path)
        elif action == "save":
            react_data = req.get("config")
            if react_data is None:
                raise ValueError("save requires a 'config' key in the stdin JSON")
            action_save(config_path or "", json.dumps(react_data))
        elif action == "reset":
            action_reset(config_path)
        else:
            raise ValueError(f"Unknown action: {action!r}. Use: get | save | reset")
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
