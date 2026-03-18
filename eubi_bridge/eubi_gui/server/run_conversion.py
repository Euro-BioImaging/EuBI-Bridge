#!/usr/bin/env python3
import sys
import json
import logging
import os
import signal
import traceback

_raw_stdout = sys.__stdout__

def emit_json(msg_type, **kwargs):
    payload = {"type": msg_type, **kwargs}
    _raw_stdout.write(json.dumps(payload) + "\n")
    _raw_stdout.flush()

class JsonLineHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            emit_json("log", message=msg)
        except Exception:
            pass

class StdoutCapture:
    def __init__(self, is_stderr=False):
        self.is_stderr = is_stderr

    def write(self, text):
        if text and text.strip():
            for line in text.strip().splitlines():
                if line.strip():
                    emit_json("log", message=line.rstrip())

    def flush(self):
        pass

    def isatty(self):
        return False

def main():
    if len(sys.argv) < 2:
        emit_json("error", message="No configuration provided")
        sys.exit(1)

    try:
        config = json.loads(sys.argv[1])
    except json.JSONDecodeError as e:
        emit_json("error", message=f"Invalid JSON configuration: {e}")
        sys.exit(1)

    handler = JsonLineHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                                  datefmt="%H:%M:%S")
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    eubi_logger = logging.getLogger("eubi_bridge")
    eubi_logger.setLevel(logging.INFO)
    eubi_logger.propagate = True

    sys.stdout = StdoutCapture()
    sys.stderr = StdoutCapture(is_stderr=True)

    def handle_signal(signum, frame):
        emit_json("log", message="Conversion cancelled by user.")
        emit_json("cancelled")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        emit_json("log", message="Initializing EuBI-Bridge...")

        from eubi_bridge.ebridge import EuBIBridge

        bridge = EuBIBridge()

        input_path = config.get("inputPath", "")
        output_path = config.get("outputPath", "")
        include_pattern = config.get("includePattern", "")
        exclude_pattern = config.get("excludePattern", "")

        concat_config = config.get("concatenation", {})
        cluster_config = config.get("cluster", {})
        reader_config = config.get("reader", {})
        conv_config = config.get("conversion", {})
        downscale_config = config.get("downscaling", {})
        meta_config = config.get("metadata", {})

        def parse_range(range_str):
            if not range_str:
                return None
            parts = range_str.split(",")
            if len(parts) == 2:
                try:
                    return (int(parts[0].strip()), int(parts[1].strip()))
                except ValueError:
                    return None
            return None

        def parse_index_value(value):
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

        compression = conv_config.get("compression", {})
        compressor = compression.get("codec", "blosc")
        zarr_format = conv_config.get("zarrFormat", 2)

        if compressor == "none":
            compressor = None
            compressor_params = {}
        elif compressor == "blosc":
            shuffle_val = compression.get("bloscShuffle", "shuffle")
            if zarr_format == 2:
                shuffle_map = {"noshuffle": 0, "shuffle": 1, "bitshuffle": 2}
                if isinstance(shuffle_val, str):
                    shuffle_val = shuffle_map.get(shuffle_val, 1)
            compressor_params = {
                "cname": compression.get("bloscInnerCodec", "lz4"),
                "clevel": compression.get("level", 5),
                "shuffle": shuffle_val,
            }
        else:
            compressor_params = {
                "level": compression.get("level", 5),
            }

        scene_idx = "all" if reader_config.get("readAllScenes", True) else parse_index_value(reader_config.get("sceneIndices", "0"))
        mosaic_idx = "all" if reader_config.get("readAllTiles", True) else parse_index_value(reader_config.get("mosaicTileIndices", "0"))

        metadata_reader = meta_config.get("metadataReader", "bioio")

        kwargs = {
            "max_workers": cluster_config.get("maxWorkers", 4),
            "queue_size": cluster_config.get("queueSize", 10),
            "region_size_mb": cluster_config.get("regionSizeMb", 64),
            "max_concurrency": cluster_config.get("maxConcurrency", 4),
            "memory_per_worker": cluster_config.get("memoryPerWorker", "4GB"),
            "on_local_cluster": cluster_config.get("useLocalDask", False),
            "on_slurm": cluster_config.get("useSlurm", False),
            "scene_index": scene_idx,
            "mosaic_tile_index": mosaic_idx,
            "as_mosaic": reader_config.get("readAsMosaic", False),
            "view_index": parse_index_value(reader_config.get("viewIndex", "0")),
            "phase_index": parse_index_value(reader_config.get("phaseIndex", "0")),
            "illumination_index": parse_index_value(reader_config.get("illuminationIndex", "0")),
            "rotation_index": parse_index_value(reader_config.get("rotationIndex", "0")),
            "sample_index": parse_index_value(reader_config.get("sampleIndex", "0")),
            "verbose": conv_config.get("verbose", False),
            "zarr_format": conv_config.get("zarrFormat", 2),
            "auto_chunk": conv_config.get("autoChunk", True),
            "time_chunk": conv_config.get("chunkTime", 1),
            "channel_chunk": conv_config.get("chunkChannel", 1),
            "z_chunk": conv_config.get("chunkZ", 1),
            "y_chunk": conv_config.get("chunkY", 256),
            "x_chunk": conv_config.get("chunkX", 256),
            "time_shard_coef": conv_config.get("shardTime", 1),
            "channel_shard_coef": conv_config.get("shardChannel", 1),
            "z_shard_coef": conv_config.get("shardZ", 1),
            "y_shard_coef": conv_config.get("shardY", 1),
            "x_shard_coef": conv_config.get("shardX", 1),
            "time_range": parse_range(conv_config.get("dimRangeTime", "")),
            "channel_range": parse_range(conv_config.get("dimRangeChannel", "")),
            "z_range": parse_range(conv_config.get("dimRangeZ", "")),
            "y_range": parse_range(conv_config.get("dimRangeY", "")),
            "x_range": parse_range(conv_config.get("dimRangeX", "")),
            "overwrite": conv_config.get("overwrite", False),
            "override_channel_names": conv_config.get("overrideChannelNames", False),
            "channel_intensity_limits": "from_dtype" if meta_config.get("channelIntensityLimits", "from_datatype") == "from_datatype" else "from_array",
            "metadata_reader": metadata_reader,
            "save_omexml": conv_config.get("saveOmeXml", True),
            "squeeze": conv_config.get("squeezeDimensions", True),
            "skip_dask": conv_config.get("skipDask", False),
            "dtype": conv_config.get("dataType", "auto") if conv_config.get("dataType", "auto") != "auto" else "auto",
            "n_layers": None if downscale_config.get("autoDetectLayers", True) else downscale_config.get("numLayers", 4),
            "min_dimension_size": downscale_config.get("minDimSize", 64),
            "time_scale_factor": downscale_config.get("scaleTime", 1),
            "channel_scale_factor": downscale_config.get("scaleChannel", 1),
            "z_scale_factor": downscale_config.get("scaleZ", 1),
            "y_scale_factor": downscale_config.get("scaleY", 2),
            "x_scale_factor": downscale_config.get("scaleX", 2),
            "compressor": compressor,
            "compressor_params": compressor_params,
        }

        if conv_config.get("autoChunk", True):
            kwargs["target_chunk_mb"] = conv_config.get("targetChunkSizeMb", 32)

        time_tag = concat_config.get("timeTag", "") or None
        channel_tag = concat_config.get("channelTag", "") or None
        z_tag = concat_config.get("zTag", "") or None
        y_tag = concat_config.get("yTag", "") or None
        x_tag = concat_config.get("xTag", "") or None
        concatenation_axes = concat_config.get("concatenationAxes", "") or None

        if meta_config.get("overridePhysicalScale", False):
            if meta_config.get("scaleTime", ""):
                try:
                    kwargs["time_scale"] = float(meta_config["scaleTime"])
                except ValueError:
                    pass
            if meta_config.get("scaleZ", ""):
                try:
                    kwargs["z_scale"] = float(meta_config["scaleZ"])
                except ValueError:
                    pass
            if meta_config.get("scaleY", ""):
                try:
                    kwargs["y_scale"] = float(meta_config["scaleY"])
                except ValueError:
                    pass
            if meta_config.get("scaleX", ""):
                try:
                    kwargs["x_scale"] = float(meta_config["scaleX"])
                except ValueError:
                    pass

            unit_default = "second"
            if meta_config.get("unitTime", "") and meta_config["unitTime"] != unit_default:
                kwargs["time_unit"] = meta_config["unitTime"]
            space_default = "micrometer"
            if meta_config.get("unitZ", "") and meta_config["unitZ"] != space_default:
                kwargs["z_unit"] = meta_config["unitZ"]
            if meta_config.get("unitY", "") and meta_config["unitY"] != space_default:
                kwargs["y_unit"] = meta_config["unitY"]
            if meta_config.get("unitX", "") and meta_config["unitX"] != space_default:
                kwargs["x_unit"] = meta_config["unitX"]

        def glob_to_substring(patterns_str):
            if not patterns_str:
                return None
            parts = [p.strip() for p in patterns_str.split(",") if p.strip()]
            if not parts:
                return None
            result = []
            for p in parts:
                p = p.strip("*")
                if p:
                    result.append(p)
            return result if result else None

        includes = glob_to_substring(include_pattern)
        excludes = glob_to_substring(exclude_pattern)

        emit_json("log", message=f"Input: {input_path}")
        emit_json("log", message=f"Output: {output_path}")
        if includes:
            emit_json("log", message=f"Include filter: {includes}")
        if excludes:
            emit_json("log", message=f"Exclude filter: {excludes}")
        emit_json("log", message=f"Zarr Format: v{kwargs.get('zarr_format', 2)}")
        emit_json("log", message=f"Max Workers: {kwargs.get('max_workers', 4)}")
        emit_json("log", message="Calling EuBI-Bridge to_zarr()...")

        bridge.to_zarr(
            input_path=input_path,
            output_path=output_path if output_path else None,
            includes=includes,
            excludes=excludes,
            time_tag=time_tag,
            channel_tag=channel_tag,
            z_tag=z_tag,
            y_tag=y_tag,
            x_tag=x_tag,
            concatenation_axes=concatenation_axes,
            **kwargs
        )

        emit_json("complete")
    except Exception as e:
        tb = traceback.format_exc()
        emit_json("error", message=str(e), traceback=tb)
        sys.exit(1)

if __name__ == "__main__":
    main()
