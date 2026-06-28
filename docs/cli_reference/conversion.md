# Conversion

Click any command card below to expand its full parameter reference.

!!! attention "Info" 
    Note that all cards below are auto-generated from `EuBIBridge` methods and Pydantic config models. Types, defaults, and valid ranges are extracted directly from the source.


!!! tip "Tip: Named-config chaining"
    Prefix any command with `with_config NAME` to use a saved config profile:
    ```shell
    eubi with_config hpc to_zarr /data/input /data/output --ome_zarr_version 0.5
    ```


??? command "**`to_zarr`**&ensp;—&ensp;Convert image data to OME-Zarr format"

    **Usage:**
    ```shell
    eubi to_zarr INPUT_PATH [OPTIONS]
    eubi with_config NAME to_zarr INPUT_PATH [OPTIONS]
    ```

    <details>
    <summary>Required arguments</summary>

    <details>
    <summary><code>--input_path</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `Path`</p>
    <p><strong>Default:</strong>&nbsp; *required*</p>
    <p>File path, directory, or table (.csv/.tsv/.xlsx) of paths.</p>
    <pre><code>eubi to_zarr /data/input /data/output
    </code></pre>
    </details>


    </details>

    <details>
    <summary>CLI-only options</summary>

    <details>
    <summary><code>--output_path</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `Path` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Destination directory.  Required unless input_path is a table that already contains an output_path column.</p>
    <pre><code>eubi to_zarr /data/input /data/output
    </code></pre>
    </details>

    <details>
    <summary><code>--includes</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Comma-separated filename patterns to include (glob-style).</p>
    <pre><code># Include only files whose name contains "EMPIAR"
    eubi to_zarr /data/input /data/output --includes "EMPIAR"
    </code></pre>
    <pre><code># Multiple patterns — comma-separated, no spaces around commas
    eubi to_zarr /data/input /data/output --includes "EMPIAR,scan_001,confocal"
    </code></pre>
    </details>

    <details>
    <summary><code>--excludes</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Comma-separated filename patterns to exclude (glob-style).</p>
    <pre><code># Exclude all files in a subdirectory named 'backup'
    eubi to_zarr /data/input /data/output --excludes "backup"
    </code></pre>
    <pre><code># Exclude multiple patterns at once
    eubi to_zarr /data/input /data/output --excludes "backup,temp,preview"
    </code></pre>
    </details>

    <details>
    <summary><code>--plan</code></summary>
    <p><strong>Type:</strong>&nbsp; `AggregativePlan` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Pre-computed AggregativePlan from validate_aggregative().</p>
    <pre><code># Step 1 — validate without converting to inspect the plan
    eubi validate_aggregative /data/input /data/output --concatenation_axes c --channel_tag raw,mask
    </code></pre>
    <pre><code># Step 2 — pass the plan object to skip re-planning on the actual run
    eubi to_zarr /data/input /data/output --concatenation_axes c --channel_tag raw,mask --plan &lt;plan&gt;
    </code></pre>
    </details>


    </details>

    <details>
    <summary>Cluster overrides</summary>

    <details>
    <summary><code>--on_local_cluster</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `False`</p>
    <p><strong>Valid values:</strong>&nbsp; `--on_local_cluster` to enable &nbsp;·&nbsp; `--on_local_cluster False` to disable</p>
    <p>Use a Dask LocalCluster.</p>
    <pre><code>eubi to_zarr /data/input /data/output --on_local_cluster
    </code></pre>
    </details>

    <details>
    <summary><code>--on_slurm</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `False`</p>
    <p><strong>Valid values:</strong>&nbsp; `--on_slurm` to enable &nbsp;·&nbsp; `--on_slurm False` to disable</p>
    <p>Submit jobs to a SLURM cluster.</p>
    <pre><code># Submit all file workers to SLURM
    eubi to_zarr /data/input /data/output --on_slurm --slurm_account myaccount --slurm_partition gpu
    </code></pre>
    </details>

    <details>
    <summary><code>--use_threading</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `False`</p>
    <p><strong>Valid values:</strong>&nbsp; `--use_threading` to enable &nbsp;·&nbsp; `--use_threading False` to disable</p>
    <p>Use a ThreadPool instead of a ProcessPool.</p>
    <pre><code>eubi to_zarr /data/input /data/output --use_threading
    </code></pre>
    </details>

    <details>
    <summary><code>--max_workers</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `4`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 1, ≤ 256</p>
    <p>Override the configured max_workers for this run.</p>
    <pre><code>eubi to_zarr /data/input /data/output --max_workers 8
    </code></pre>
    </details>

    <details>
    <summary><code>--queue_size</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `4`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 1, ≤ 4096</p>
    <p>Internal write-queue depth per worker.</p>
    <pre><code>eubi to_zarr /data/input /data/output --queue_size 8
    </code></pre>
    </details>

    <details>
    <summary><code>--region_size_mb</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `256`</p>
    <p><strong>Valid values:</strong>&nbsp; > 0</p>
    <p>Region size in MB for spatial partitioning.</p>
    <pre><code>eubi to_zarr /data/input /data/output --region_size_mb 512
    </code></pre>
    </details>

    <details>
    <summary><code>--max_concurrency</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `4`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 1</p>
    <p>TensorStore write concurrency per worker.</p>
    <pre><code>eubi to_zarr /data/input /data/output --max_concurrency 8
    </code></pre>
    </details>

    <details>
    <summary><code>--max_concurrent_scenes</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `1`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 1</p>
    <p>Parallel scenes per file (default 1).</p>
    <pre><code>eubi to_zarr /data/input /data/output --max_concurrent_scenes 2
    </code></pre>
    </details>

    <details>
    <summary><code>--memory_per_worker</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; `3GB`</p>
    <p>Memory limit for SLURM / LocalCluster workers.</p>
    <pre><code>eubi to_zarr /data/input /data/output --memory_per_worker 8GB
    </code></pre>
    </details>

    <details>
    <summary><code>--tensorstore_data_copy_concurrency</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `4`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 1</p>
    <p>TensorStore internal copy threads.</p>
    <pre><code>eubi to_zarr /data/input /data/output --tensorstore_data_copy_concurrency 8
    </code></pre>
    </details>

    <details>
    <summary><code>--max_retries</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `10`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 0, ≤ 100</p>
    <p>Override the configured max_retries.</p>
    <pre><code>eubi to_zarr /data/input /data/output --max_retries 3
    </code></pre>
    </details>

    <details>
    <summary><code>--bf_read_concurrency</code></summary>
    <p><strong>Type:</strong>&nbsp; `int` or `None`</p>
    <p><strong>Default:</strong>&nbsp; `4`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 1</p>
    <p>—</p>
    </details>

    <details>
    <summary><code>--bf_tile_size_mb</code></summary>
    <p><strong>Type:</strong>&nbsp; `float`</p>
    <p><strong>Default:</strong>&nbsp; `512.0`</p>
    <p><strong>Valid values:</strong>&nbsp; > 0.0</p>
    <p>—</p>
    </details>

    <details>
    <summary><code>--jvm_memory</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `None`</p>
    <p><strong>Default:</strong>&nbsp; `1g`</p>
    <p>—</p>
    </details>

    <details>
    <summary><code>--slurm_time</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; `24:00:00`</p>
    <p>SLURM wall-clock limit, e.g. '24:00:00'.</p>
    <pre><code>eubi to_zarr /data/input /data/output --on_slurm --slurm_time 48:00:00
    </code></pre>
    </details>

    <details>
    <summary><code>--slurm_account</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>SLURM account name.</p>
    <pre><code>eubi to_zarr /data/input /data/output --on_slurm --slurm_account myaccount
    </code></pre>
    </details>

    <details>
    <summary><code>--slurm_partition</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>SLURM partition / queue.</p>
    <pre><code>eubi to_zarr /data/input /data/output --on_slurm --slurm_partition gpu
    </code></pre>
    </details>

    <details>
    <summary><code>--slurm_worker_timeout</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `300`</p>
    <p><strong>Valid values:</strong>&nbsp; > 0</p>
    <p>Seconds to wait for SLURM workers to start.</p>
    <pre><code>eubi to_zarr /data/input /data/output --on_slurm --slurm_worker_timeout 600
    </code></pre>
    </details>

    <details>
    <summary><code>--slurm_sif_path</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Path to an Apptainer/Singularity `.sif` image to run SLURM workers inside (optional).</p>
    <pre><code>eubi to_zarr /data/input /data/output --on_slurm --slurm_sif_path /apps/eubi.sif
    </code></pre>
    </details>

    <details>
    <summary><code>--max_concurrent_downscale_layers</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `3`</p>
    <p><strong>Valid values:</strong>&nbsp; > 0</p>
    <p>How many pyramid levels are downscaled in parallel per file (default 1 = sequential, lowest memory).</p>
    <pre><code>eubi to_zarr /data/input /data/output --max_concurrent_downscale_layers 2
    </code></pre>
    </details>


    </details>

    <details>
    <summary>Conversion overrides</summary>

    <details>
    <summary><code>--verbose</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `False`</p>
    <p><strong>Valid values:</strong>&nbsp; `--verbose` to enable &nbsp;·&nbsp; `--verbose False` to disable</p>
    <p>Log verbose per-chunk progress output.</p>
    <pre><code>eubi to_zarr /data/input /data/output --verbose
    </code></pre>
    </details>

    <details>
    <summary><code>--ome_zarr_version</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>OME-Zarr (NGFF) version to write: `0.4` (backed by Zarr v2) or `0.5` (backed by Zarr v3, which enables sharding). This is the preferred control and supersedes `zarr_format` — the underlying zarr container format is derived from it.</p>
    <pre><code># Write OME-Zarr 0.4 (Zarr v2, the default)
    eubi to_zarr /data/input /data/output --ome_zarr_version 0.4
    </code></pre>
    <pre><code># Write OME-Zarr 0.5 (Zarr v3 — required for sharding)
    eubi to_zarr /data/input /data/output --ome_zarr_version 0.5
    </code></pre>
    </details>

    <details>
    <summary><code>--zarr_format</code></summary>
    <p><strong>Type:</strong>&nbsp; `2` or `3`</p>
    <p><strong>Default:</strong>&nbsp; `2`</p>
    <p>**Deprecated** — use `ome_zarr_version` instead. Zarr container version: `2` = OME-Zarr 0.4 (classic chunk-based); `3` = OME-Zarr 0.5 (sharding support).</p>
    <pre><code>eubi to_zarr /data/input /data/output --zarr_format 3
    </code></pre>
    </details>

    <details>
    <summary><code>--skip_dask</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `False`</p>
    <p><strong>Valid values:</strong>&nbsp; `--skip_dask` to enable &nbsp;·&nbsp; `--skip_dask False` to disable</p>
    <p>Read TIFF files via zarr's native aszarr backend instead of dask. Faster for large TIFFs; ignored for non-TIFF formats.</p>
    <pre><code>eubi to_zarr /data/input /data/output --skip_dask
    </code></pre>
    </details>

    <details>
    <summary><code>--auto_chunk</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `True`</p>
    <p><strong>Valid values:</strong>&nbsp; `--auto_chunk` to enable &nbsp;·&nbsp; `--auto_chunk False` to disable</p>
    <p>Auto-compute chunk shape to approximate `target_chunk_mb`. When `False`, the manual `*_chunk` values below are used.</p>
    <pre><code>eubi to_zarr /data/input /data/output --auto_chunk False --z_chunk 64 --y_chunk 256 --x_chunk 256
    </code></pre>
    </details>

    <details>
    <summary><code>--target_chunk_mb</code></summary>
    <p><strong>Type:</strong>&nbsp; `float`</p>
    <p><strong>Default:</strong>&nbsp; `1.0`</p>
    <p><strong>Valid values:</strong>&nbsp; > 0.0</p>
    <p>Target uncompressed chunk size in MB when `auto_chunk=True`.</p>
    <pre><code>eubi to_zarr /data/input /data/output --target_chunk_mb 4.0
    </code></pre>
    </details>

    <details>
    <summary><code>--time_chunk</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `1`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 1</p>
    <p>Chunk size along the time axis. Applies only when `auto_chunk=False`.</p>
    <pre><code>eubi to_zarr /data/input /data/output --auto_chunk False --time_chunk 1
    </code></pre>
    </details>

    <details>
    <summary><code>--channel_chunk</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `1`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 1</p>
    <p>Chunk size along the channel axis. Applies only when `auto_chunk=False`.</p>
    <pre><code>eubi to_zarr /data/input /data/output --auto_chunk False --channel_chunk 1
    </code></pre>
    </details>

    <details>
    <summary><code>--z_chunk</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `96`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 1</p>
    <p>Chunk size along the z axis. Applies only when `auto_chunk=False`.</p>
    <pre><code>eubi to_zarr /data/input /data/output --auto_chunk False --z_chunk 64
    </code></pre>
    </details>

    <details>
    <summary><code>--y_chunk</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `96`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 1</p>
    <p>Chunk size along the y axis. Applies only when `auto_chunk=False`.</p>
    <pre><code>eubi to_zarr /data/input /data/output --auto_chunk False --y_chunk 256
    </code></pre>
    </details>

    <details>
    <summary><code>--x_chunk</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `96`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 1</p>
    <p>Chunk size along the x axis. Applies only when `auto_chunk=False`.</p>
    <pre><code>eubi to_zarr /data/input /data/output --auto_chunk False --x_chunk 256
    </code></pre>
    </details>

    <details>
    <summary><code>--time_shard_coef</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `1`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 1</p>
    <p>Shard size = chunk × coef for the time axis. Zarr v3 only — ignored for v2.</p>
    <pre><code>eubi to_zarr /data/input /data/output --ome_zarr_version 0.5 --time_shard_coef 1
    </code></pre>
    </details>

    <details>
    <summary><code>--channel_shard_coef</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `1`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 1</p>
    <p>Shard size = chunk × coef for the channel axis. Zarr v3 only — ignored for v2.</p>
    <pre><code>eubi to_zarr /data/input /data/output --ome_zarr_version 0.5 --channel_shard_coef 1
    </code></pre>
    </details>

    <details>
    <summary><code>--z_shard_coef</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `3`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 1</p>
    <p>Shard size = chunk × coef for the z axis. Zarr v3 only — ignored for v2.</p>
    <pre><code>eubi to_zarr /data/input /data/output --ome_zarr_version 0.5 --z_shard_coef 5
    </code></pre>
    </details>

    <details>
    <summary><code>--y_shard_coef</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `3`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 1</p>
    <p>Shard size = chunk × coef for the y axis. Zarr v3 only — ignored for v2.</p>
    <pre><code>eubi to_zarr /data/input /data/output --ome_zarr_version 0.5 --y_shard_coef 5
    </code></pre>
    </details>

    <details>
    <summary><code>--x_shard_coef</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `3`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 1</p>
    <p>Shard size = chunk × coef for the x axis. Zarr v3 only — ignored for v2.</p>
    <pre><code>eubi to_zarr /data/input /data/output --ome_zarr_version 0.5 --x_shard_coef 5
    </code></pre>
    </details>

    <details>
    <summary><code>--time_range</code></summary>
    <p><strong>Type:</strong>&nbsp; `Any` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Crop the time axis before writing. Pass as a `"start,stop"` string (e.g. `"0,10"` to keep frames 0–9).</p>
    <pre><code>eubi to_zarr /data/input /data/output --time_range "0,10"
    </code></pre>
    </details>

    <details>
    <summary><code>--channel_range</code></summary>
    <p><strong>Type:</strong>&nbsp; `Any` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Crop the channel axis. Same format as `time_range`, e.g. `"0,2"` keeps the first two channels.</p>
    <pre><code>eubi to_zarr /data/input /data/output --channel_range "0,2"
    </code></pre>
    </details>

    <details>
    <summary><code>--z_range</code></summary>
    <p><strong>Type:</strong>&nbsp; `Any` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Crop the z axis. Same format as `time_range`, e.g. `"5,50"` keeps z-slices 5–49.</p>
    <pre><code>eubi to_zarr /data/input /data/output --z_range "5,50"
    </code></pre>
    </details>

    <details>
    <summary><code>--y_range</code></summary>
    <p><strong>Type:</strong>&nbsp; `Any` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Crop the y axis. Same format as `time_range`.</p>
    <pre><code>eubi to_zarr /data/input /data/output --y_range "0,512"
    </code></pre>
    </details>

    <details>
    <summary><code>--x_range</code></summary>
    <p><strong>Type:</strong>&nbsp; `Any` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Crop the x axis. Same format as `time_range`.</p>
    <pre><code>eubi to_zarr /data/input /data/output --x_range "0,512"
    </code></pre>
    </details>

    <details>
    <summary><code>--dimension_order</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; `tczyx`</p>
    <p>Axis order string for the output array (default `tczyx`).</p>
    </details>

    <details>
    <summary><code>--compressor</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `None`</p>
    <p><strong>Default:</strong>&nbsp; `blosc`</p>
    <p><strong>Valid values:</strong>&nbsp; v2: `blosc`, `bz2`, `gzip`, `none`, `zstd`; v3: `blosc`, `crc32ccodec`, `gzip`, `none`, `sharding`, `zstd`</p>
    <p>Compression codec.</p>
    <pre><code>eubi to_zarr /data/input /data/output --compressor zstd
    </code></pre>
    </details>

    <details>
    <summary><code>--compressor_params</code></summary>
    <p><strong>Type:</strong>&nbsp; `dict`</p>
    <p><strong>Default:</strong>&nbsp; *(auto)*</p>
    <p>Codec-specific parameters dict. When omitted, sensible defaults are used (e.g. for blosc: `{'cname': 'lz4', 'clevel': 5, 'shuffle': 1}`).</p>
    <pre><code>eubi to_zarr /data/input /data/output --compressor blosc --compressor_params '{"cname": "lz4", "clevel": 9}'
    </code></pre>
    </details>

    <details>
    <summary><code>--overwrite</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `False`</p>
    <p><strong>Valid values:</strong>&nbsp; `--overwrite` to enable &nbsp;·&nbsp; `--overwrite False` to disable</p>
    <p>Overwrite existing output zarr if it already exists.</p>
    <pre><code>eubi to_zarr /data/input /data/output --overwrite
    </code></pre>
    </details>

    <details>
    <summary><code>--override_channel_names</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `False`</p>
    <p><strong>Valid values:</strong>&nbsp; `--override_channel_names` to enable &nbsp;·&nbsp; `--override_channel_names False` to disable</p>
    <p>Replace output channel labels with the `channel_tag` values. For aggregative conversions with a tuple `channel_tag` only.</p>
    <pre><code>eubi to_zarr /data/input /data/output --concatenation_axes c --channel_tag raw,mask --override_channel_names
    </code></pre>
    </details>

    <details>
    <summary><code>--channel_intensity_limits</code></summary>
    <p><strong>Type:</strong>&nbsp; `from_dtype` or `from_array` or `auto`</p>
    <p><strong>Default:</strong>&nbsp; `from_dtype`</p>
    <p><strong>Valid values:</strong>&nbsp; `from_dtype` · `from_array` · `auto`</p>
    <p>Strategy for OMERO display-window limits.</p>
    <pre><code>eubi to_zarr /data/input /data/output --channel_intensity_limits from_array
    </code></pre>
    </details>

    <details>
    <summary><code>--metadata_reader</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; `bfio`</p>
    <p><strong>Valid values:</strong>&nbsp; `bfio` · `bioformats`</p>
    <p>Backend used to read OME-XML pixel metadata.</p>
    <pre><code>eubi to_zarr /data/input /data/output --metadata_reader bioformats
    </code></pre>
    </details>

    <details>
    <summary><code>--save_omexml</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `True`</p>
    <p><strong>Valid values:</strong>&nbsp; `--save_omexml` to enable &nbsp;·&nbsp; `--save_omexml False` to disable</p>
    <p>Write a companion OME-XML sidecar file alongside the zarr.</p>
    <pre><code>eubi to_zarr /data/input /data/output --save_omexml False
    </code></pre>
    </details>

    <details>
    <summary><code>--squeeze</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `True`</p>
    <p><strong>Valid values:</strong>&nbsp; `--squeeze` to enable &nbsp;·&nbsp; `--squeeze False` to disable</p>
    <p>Remove length-1 dimensions before writing.</p>
    <pre><code>eubi to_zarr /data/input /data/output --squeeze False
    </code></pre>
    </details>

    <details>
    <summary><code>--dtype</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `None`</p>
    <p><strong>Default:</strong>&nbsp; `auto`</p>
    <p>`auto` preserves the source dtype; pass any NumPy dtype string (e.g. `uint16`, `float32`) to cast on write.</p>
    <pre><code>eubi to_zarr /data/input /data/output --dtype uint8
    </code></pre>
    </details>


    </details>

    <details>
    <summary>Downscale overrides</summary>

    <details>
    <summary><code>--time_scale_factor</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `1`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 1</p>
    <p>Downscale factor for the time axis per pyramid level.</p>
    <pre><code>eubi to_zarr /data/input /data/output --time_scale_factor 1
    </code></pre>
    </details>

    <details>
    <summary><code>--channel_scale_factor</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `1`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 1</p>
    <p>Downscale factor for the channel axis.</p>
    <pre><code>eubi to_zarr /data/input /data/output --channel_scale_factor 1
    </code></pre>
    </details>

    <details>
    <summary><code>--z_scale_factor</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `2`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 1</p>
    <p>Downscale factor for the z axis.</p>
    <pre><code># Isotropic downscaling on all spatial axes
    eubi to_zarr /data/input /data/output --z_scale_factor 2 --y_scale_factor 2 --x_scale_factor 2
    </code></pre>
    <pre><code># Anisotropic — skip z downscaling for thick sections
    eubi to_zarr /data/input /data/output --z_scale_factor 1 --y_scale_factor 2 --x_scale_factor 2
    </code></pre>
    </details>

    <details>
    <summary><code>--y_scale_factor</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `2`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 1</p>
    <p>Downscale factor for the y axis.</p>
    <pre><code>eubi to_zarr /data/input /data/output --y_scale_factor 2
    </code></pre>
    </details>

    <details>
    <summary><code>--x_scale_factor</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `2`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 1</p>
    <p>Downscale factor for the x axis.</p>
    <pre><code>eubi to_zarr /data/input /data/output --x_scale_factor 2
    </code></pre>
    </details>

    <details>
    <summary><code>--n_layers</code></summary>
    <p><strong>Type:</strong>&nbsp; `int` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Number of pyramid levels (None = auto).</p>
    <pre><code>eubi to_zarr /data/input /data/output --n_layers 5
    </code></pre>
    </details>

    <details>
    <summary><code>--min_dimension_size</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `64`</p>
    <p><strong>Valid values:</strong>&nbsp; > 0</p>
    <p>Stop downscaling when smallest dimension reaches this size.</p>
    <pre><code>eubi to_zarr /data/input /data/output --min_dimension_size 32
    </code></pre>
    </details>

    <details>
    <summary><code>--downscale_method</code></summary>
    <p><strong>Type:</strong>&nbsp; `simple` or `mean` or `median` or `min` or `max` or `mode`</p>
    <p><strong>Default:</strong>&nbsp; `simple`</p>
    <p><strong>Valid values:</strong>&nbsp; `simple` · `mean` · `median` · `min` · `max` · `mode`</p>
    <p>`simple` = stride/nearest (fastest). `mean` / `median` / `min` / `max` / `mode` = aggregation methods passed to TensorStore.</p>
    <pre><code># Use mean aggregation (recommended for fluorescence images)
    eubi to_zarr /data/input /data/output --downscale_method mean
    </code></pre>
    <pre><code># Use nearest-neighbour striding (fastest, recommended for label images)
    eubi to_zarr /data/input /data/output --downscale_method simple
    </code></pre>
    </details>

    <details>
    <summary><code>--keep_existing_resolutions</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `False`</p>
    <p><strong>Valid values:</strong>&nbsp; `--keep_existing_resolutions` to enable &nbsp;·&nbsp; `--keep_existing_resolutions False` to disable</p>
    <p>If the input already carries its own multiscale pyramid (e.g. `.ims`, `.zarr`), write each existing resolution level straight to the output OME-Zarr instead of rebuilding the pyramid (default False).</p>
    <pre><code># Copy an .ims / .zarr input's own pyramid levels instead of rebuilding them
    eubi to_zarr /data/input /data/output --keep_existing_resolutions
    </code></pre>
    </details>

    <details>
    <summary><code>--apply_smart_downscaling</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `False`</p>
    <p><strong>Valid values:</strong>&nbsp; `--apply_smart_downscaling` to enable &nbsp;·&nbsp; `--apply_smart_downscaling False` to disable</p>
    <p>Choose per-axis downscale factors automatically from the pixel anisotropy so the pyramid approaches isotropy, instead of using the fixed `*_scale_factor` values (default False).</p>
    <pre><code># Let EuBI-Bridge pick per-axis factors from the voxel anisotropy
    eubi to_zarr /data/input /data/output --apply_smart_downscaling
    </code></pre>
    </details>

    <details>
    <summary><code>--time_smart_scale_factor</code></summary>
    <p><strong>Type:</strong>&nbsp; `int` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Override the smart-downscaling factor for the time axis (used when `apply_smart_downscaling=True`).</p>
    <pre><code>eubi to_zarr /data/input /data/output --apply_smart_downscaling --time_smart_scale_factor 1
    </code></pre>
    </details>

    <details>
    <summary><code>--z_smart_scale_factor</code></summary>
    <p><strong>Type:</strong>&nbsp; `int` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Override the smart-downscaling factor for the z axis (used when `apply_smart_downscaling=True`).</p>
    <pre><code>eubi to_zarr /data/input /data/output --apply_smart_downscaling --z_smart_scale_factor 1
    </code></pre>
    </details>

    <details>
    <summary><code>--y_smart_scale_factor</code></summary>
    <p><strong>Type:</strong>&nbsp; `int` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Override the smart-downscaling factor for the y axis (used when `apply_smart_downscaling=True`).</p>
    <pre><code>eubi to_zarr /data/input /data/output --apply_smart_downscaling --y_smart_scale_factor 2
    </code></pre>
    </details>

    <details>
    <summary><code>--x_smart_scale_factor</code></summary>
    <p><strong>Type:</strong>&nbsp; `int` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Override the smart-downscaling factor for the x axis (used when `apply_smart_downscaling=True`).</p>
    <pre><code>eubi to_zarr /data/input /data/output --apply_smart_downscaling --x_smart_scale_factor 2
    </code></pre>
    </details>


    </details>

    <details>
    <summary>Reader overrides</summary>

    <details>
    <summary><code>--as_mosaic</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `False`</p>
    <p><strong>Valid values:</strong>&nbsp; `--as_mosaic` to enable &nbsp;·&nbsp; `--as_mosaic False` to disable</p>
    <p>Stitch all mosaic tiles into a single full field-of-view output (instead of one OME-Zarr per tile).</p>
    <pre><code># Stitch all mosaic tiles into a single full field-of-view output
    eubi to_zarr /data/input /data/output --as_mosaic
    </code></pre>
    </details>

    <details>
    <summary><code>--view_index</code></summary>
    <p><strong>Type:</strong>&nbsp; `int` or `str`</p>
    <p><strong>Default:</strong>&nbsp; `0`</p>
    <p>View(s) to read. Pass an integer, `all`, or comma-separated integers; each selected view becomes a separate OME-Zarr (named `_view{N}`) unless `--concat_views` stacks them along the channel axis.</p>
    <pre><code># Write each view as its own OME-Zarr
    eubi to_zarr /data/input /data/output --view_index all
    </code></pre>
    <pre><code># Concatenate all views along the channel axis into one output
    eubi to_zarr /data/input /data/output --view_index all --concat_views
    </code></pre>
    </details>

    <details>
    <summary><code>--phase_index</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `0`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 0</p>
    <p>Phase index.</p>
    <pre><code>eubi to_zarr /data/input /data/output --phase_index 1
    </code></pre>
    </details>

    <details>
    <summary><code>--illumination_index</code></summary>
    <p><strong>Type:</strong>&nbsp; `int` or `str`</p>
    <p><strong>Default:</strong>&nbsp; `0`</p>
    <p>Illumination(s) to read. Pass an integer, `all`, or comma-separated integers; each selected illumination becomes a separate OME-Zarr (named `_illu{N}`) unless `--concat_illuminations` stacks them along the channel axis.</p>
    <pre><code># Write each illumination as its own OME-Zarr
    eubi to_zarr /data/input /data/output --illumination_index all
    </code></pre>
    <pre><code># Concatenate all illuminations along the channel axis into one output
    eubi to_zarr /data/input /data/output --illumination_index all --concat_illuminations
    </code></pre>
    </details>

    <details>
    <summary><code>--scene_index</code></summary>
    <p><strong>Type:</strong>&nbsp; `int` or `str`</p>
    <p><strong>Default:</strong>&nbsp; `0`</p>
    <p>Scene / series index to read. Pass an integer for a single scene, `all` to convert each scene to a separate zarr group, or comma-separated integers (e.g. `0,2,4`) for a subset of scenes.</p>
    <pre><code># Convert only scene 2 from a multi-scene file
    eubi to_zarr /data/input /data/output --scene_index 2
    </code></pre>
    <pre><code># Convert every scene to a separate OME-Zarr group
    eubi to_zarr /data/input /data/output --scene_index all
    </code></pre>
    <pre><code># Convert scenes 0, 2 and 4 only
    eubi to_zarr /data/input /data/output --scene_index 0,2,4
    </code></pre>
    </details>

    <details>
    <summary><code>--rotation_index</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `0`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 0</p>
    <p>Rotation index.</p>
    <pre><code>eubi to_zarr /data/input /data/output --rotation_index 0
    </code></pre>
    </details>

    <details>
    <summary><code>--mosaic_tile_index</code></summary>
    <p><strong>Type:</strong>&nbsp; `int` or `str` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Mosaic tile(s) to read when **not** stitching. Pass an integer, `all`, or comma-separated integers; each selected tile becomes a separate OME-Zarr (named `_tile{N}`). Use `--as_mosaic` instead to stitch tiles into one output. Composes with scene / view / illumination selection (cartesian product).</p>
    <pre><code># Write every tile as its own OME-Zarr (default — each tile separate)
    eubi to_zarr /data/input /data/output --mosaic_tile_index all
    </code></pre>
    <pre><code># Write only tiles 0 and 2 (each as a separate output)
    eubi to_zarr /data/input /data/output --mosaic_tile_index 0,2
    </code></pre>
    <pre><code># Stitch all tiles into one mosaic instead of writing them separately
    eubi to_zarr /data/input /data/output --as_mosaic
    </code></pre>
    </details>

    <details>
    <summary><code>--sample_index</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `0`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 0</p>
    <p>Sample index.</p>
    <pre><code>eubi to_zarr /data/input /data/output --sample_index 0
    </code></pre>
    </details>

    <details>
    <summary><code>--force_bioformats</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `False`</p>
    <p><strong>Valid values:</strong>&nbsp; `--force_bioformats` to enable &nbsp;·&nbsp; `--force_bioformats False` to disable</p>
    <p>Force the Java Bio-Formats reader even for formats EuBI-Bridge reads natively (CZI, ND2, LIF, IMS…). Useful as a fallback when a native read fails.</p>
    <pre><code>eubi to_zarr /data/input /data/output --force_bioformats
    </code></pre>
    </details>

    <details>
    <summary><code>--concat_views</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `False`</p>
    <p><strong>Valid values:</strong>&nbsp; `--concat_views` to enable &nbsp;·&nbsp; `--concat_views False` to disable</p>
    <p>When reading multiple views, stack them along the channel axis into one output (cartesian product with existing channels) instead of writing one OME-Zarr per view.</p>
    <pre><code># Stack every view onto the channel axis (cartesian with existing channels)
    eubi to_zarr /data/input /data/output --view_index all --concat_views
    </code></pre>
    </details>

    <details>
    <summary><code>--concat_illuminations</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `False`</p>
    <p><strong>Valid values:</strong>&nbsp; `--concat_illuminations` to enable &nbsp;·&nbsp; `--concat_illuminations False` to disable</p>
    <p>When reading multiple illuminations, stack them along the channel axis into one output (cartesian product with existing channels) instead of writing one OME-Zarr per illumination.</p>
    <pre><code># Stack every illumination onto the channel axis (cartesian with existing channels)
    eubi to_zarr /data/input /data/output --illumination_index all --concat_illuminations
    </code></pre>
    </details>


    </details>

    <details>
    <summary>Concatenation overrides</summary>

    <details>
    <summary><code>--concatenation_axes</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `int` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Axes to concatenate across files, e.g. `'tc'`. If omitted the config value is used; `None` means unary conversion.</p>
    <pre><code># Concatenate files along the channel axis
    eubi to_zarr /data/input /data/output --concatenation_axes c --channel_tag raw,mask
    </code></pre>
    <pre><code># Concatenate along both z and channel axes simultaneously
    eubi to_zarr /data/input /data/output --concatenation_axes zc --z_tag slices --channel_tag raw,mask
    </code></pre>
    </details>

    <details>
    <summary><code>--time_tag</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `List` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Filename tag identifying the time axis (aggregative only).</p>
    <pre><code># Three time points identified by substrings in each filename
    eubi to_zarr /data/input /data/output --concatenation_axes t --time_tag t001,t002,t003
    </code></pre>
    </details>

    <details>
    <summary><code>--channel_tag</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `List` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Filename tag identifying the channel axis.</p>
    <pre><code># Two channels — raw fluorescence and segmentation mask
    eubi to_zarr /data/input /data/output --concatenation_axes c --channel_tag raw,mask
    </code></pre>
    <pre><code># Single channel — assign a label without concatenating
    eubi to_zarr /data/input /data/output --concatenation_axes c --channel_tag DAPI
    </code></pre>
    </details>

    <details>
    <summary><code>--z_tag</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `List` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Filename tag identifying the z axis.</p>
    <pre><code># Multiple z-slices acquired as separate files
    eubi to_zarr /data/input /data/output --concatenation_axes z --z_tag slice001,slice002,slice003
    </code></pre>
    </details>

    <details>
    <summary><code>--y_tag</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `List` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Filename tag identifying the y axis.</p>
    <pre><code>eubi to_zarr /data/input /data/output --concatenation_axes y --y_tag row_top,row_bottom
    </code></pre>
    </details>

    <details>
    <summary><code>--x_tag</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `List` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Filename tag identifying the x axis.</p>
    <pre><code>eubi to_zarr /data/input /data/output --concatenation_axes x --x_tag col_left,col_right
    </code></pre>
    </details>


    </details>


??? command "**`validate_aggregative`**&ensp;—&ensp;Dry-run an aggregative conversion and print the resource plan"

    **Usage:**
    ```shell
    eubi validate_aggregative INPUT_PATH [OPTIONS]
    eubi with_config NAME validate_aggregative INPUT_PATH [OPTIONS]
    ```

    Determines output groups and worker allocation without reading any pixel data.  Prints a human-readable summary and returns the plan object so it can be passed to :meth:`to_zarr` to skip re-planning.

    <details>
    <summary>Required arguments</summary>

    <details>
    <summary><code>--input_path</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `Path`</p>
    <p><strong>Default:</strong>&nbsp; *required*</p>
    <p>File path, directory, or table of paths.</p>
    <pre><code>eubi validate_aggregative /data/input /data/output --concatenation_axes c --channel_tag raw,mask
    </code></pre>
    </details>


    </details>

    <details>
    <summary>CLI-only options</summary>

    <details>
    <summary><code>--output_path</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `Path` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Destination directory.</p>
    <pre><code>eubi validate_aggregative /data/input /data/output --concatenation_axes c --channel_tag raw,mask
    </code></pre>
    </details>

    <details>
    <summary><code>--includes</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Comma-separated filename patterns to include.</p>
    <pre><code># Plan the conversion but only include files matching "scan"
    eubi validate_aggregative /data/input /data/output --concatenation_axes c --channel_tag raw,mask --includes "scan"
    </code></pre>
    </details>

    <details>
    <summary><code>--excludes</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Comma-separated filename patterns to exclude.</p>
    <pre><code># Exclude preview files from the plan
    eubi validate_aggregative /data/input /data/output --concatenation_axes c --channel_tag raw,mask --excludes "preview"
    </code></pre>
    </details>


    </details>

    <details>
    <summary>Cluster overrides</summary>

    <details>
    <summary><code>--on_local_cluster</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `False`</p>
    <p><strong>Valid values:</strong>&nbsp; `--on_local_cluster` to enable &nbsp;·&nbsp; `--on_local_cluster False` to disable</p>
    <p>Use a Dask LocalCluster backend.</p>
    <pre><code>eubi to_zarr /data/input /data/output --on_local_cluster
    </code></pre>
    </details>

    <details>
    <summary><code>--on_slurm</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `False`</p>
    <p><strong>Valid values:</strong>&nbsp; `--on_slurm` to enable &nbsp;·&nbsp; `--on_slurm False` to disable</p>
    <p>Submit to a SLURM cluster.</p>
    <pre><code># Submit all file workers to SLURM
    eubi to_zarr /data/input /data/output --on_slurm --slurm_account myaccount --slurm_partition gpu
    </code></pre>
    </details>

    <details>
    <summary><code>--use_threading</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `False`</p>
    <p><strong>Valid values:</strong>&nbsp; `--use_threading` to enable &nbsp;·&nbsp; `--use_threading False` to disable</p>
    <p>Use ThreadPool instead of ProcessPool.</p>
    <pre><code>eubi to_zarr /data/input /data/output --use_threading
    </code></pre>
    </details>

    <details>
    <summary><code>--max_workers</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `4`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 1, ≤ 256</p>
    <p>Number of parallel file-level worker processes.</p>
    <pre><code>eubi to_zarr /data/input /data/output --max_workers 8
    </code></pre>
    </details>

    <details>
    <summary><code>--queue_size</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `4`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 1, ≤ 4096</p>
    <p>Internal write-queue depth per worker.</p>
    <pre><code>eubi to_zarr /data/input /data/output --queue_size 8
    </code></pre>
    </details>

    <details>
    <summary><code>--region_size_mb</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `256`</p>
    <p><strong>Valid values:</strong>&nbsp; > 0</p>
    <p>Region size in MB for spatial partitioning.</p>
    <pre><code>eubi to_zarr /data/input /data/output --region_size_mb 512
    </code></pre>
    </details>

    <details>
    <summary><code>--max_concurrency</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `4`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 1</p>
    <p>TensorStore write concurrency per worker.</p>
    <pre><code>eubi to_zarr /data/input /data/output --max_concurrency 8
    </code></pre>
    </details>

    <details>
    <summary><code>--max_concurrent_scenes</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `1`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 1</p>
    <p>Parallel scenes per file (default 1).</p>
    <pre><code>eubi to_zarr /data/input /data/output --max_concurrent_scenes 2
    </code></pre>
    </details>

    <details>
    <summary><code>--memory_per_worker</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; `3GB`</p>
    <p>Memory limit for SLURM / LocalCluster workers.</p>
    <pre><code>eubi to_zarr /data/input /data/output --memory_per_worker 8GB
    </code></pre>
    </details>

    <details>
    <summary><code>--tensorstore_data_copy_concurrency</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `4`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 1</p>
    <p>TensorStore internal copy threads.</p>
    <pre><code>eubi to_zarr /data/input /data/output --tensorstore_data_copy_concurrency 8
    </code></pre>
    </details>

    <details>
    <summary><code>--max_retries</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `10`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 0, ≤ 100</p>
    <p>Retries on a broken worker process.</p>
    <pre><code>eubi to_zarr /data/input /data/output --max_retries 3
    </code></pre>
    </details>

    <details>
    <summary><code>--bf_read_concurrency</code></summary>
    <p><strong>Type:</strong>&nbsp; `int` or `None`</p>
    <p><strong>Default:</strong>&nbsp; `4`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 1</p>
    <p>—</p>
    </details>

    <details>
    <summary><code>--bf_tile_size_mb</code></summary>
    <p><strong>Type:</strong>&nbsp; `float`</p>
    <p><strong>Default:</strong>&nbsp; `512.0`</p>
    <p><strong>Valid values:</strong>&nbsp; > 0.0</p>
    <p>—</p>
    </details>

    <details>
    <summary><code>--jvm_memory</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `None`</p>
    <p><strong>Default:</strong>&nbsp; `1g`</p>
    <p>—</p>
    </details>

    <details>
    <summary><code>--slurm_time</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; `24:00:00`</p>
    <p>SLURM wall-clock limit, e.g. '24:00:00'.</p>
    <pre><code>eubi to_zarr /data/input /data/output --on_slurm --slurm_time 48:00:00
    </code></pre>
    </details>

    <details>
    <summary><code>--slurm_account</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>SLURM account name.</p>
    <pre><code>eubi to_zarr /data/input /data/output --on_slurm --slurm_account myaccount
    </code></pre>
    </details>

    <details>
    <summary><code>--slurm_partition</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>SLURM partition / queue.</p>
    <pre><code>eubi to_zarr /data/input /data/output --on_slurm --slurm_partition gpu
    </code></pre>
    </details>

    <details>
    <summary><code>--slurm_worker_timeout</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `300`</p>
    <p><strong>Valid values:</strong>&nbsp; > 0</p>
    <p>Seconds to wait for SLURM workers to start.</p>
    <pre><code>eubi to_zarr /data/input /data/output --on_slurm --slurm_worker_timeout 600
    </code></pre>
    </details>

    <details>
    <summary><code>--slurm_sif_path</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Path to an Apptainer/Singularity `.sif` image to run SLURM workers inside (optional).</p>
    <pre><code>eubi to_zarr /data/input /data/output --on_slurm --slurm_sif_path /apps/eubi.sif
    </code></pre>
    </details>

    <details>
    <summary><code>--max_concurrent_downscale_layers</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `3`</p>
    <p><strong>Valid values:</strong>&nbsp; > 0</p>
    <p>How many pyramid levels are downscaled in parallel per file (default 1 = sequential, lowest memory).</p>
    <pre><code>eubi to_zarr /data/input /data/output --max_concurrent_downscale_layers 2
    </code></pre>
    </details>


    </details>

    <details>
    <summary>Reader overrides</summary>

    <details>
    <summary><code>--as_mosaic</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `False`</p>
    <p><strong>Valid values:</strong>&nbsp; `--as_mosaic` to enable &nbsp;·&nbsp; `--as_mosaic False` to disable</p>
    <p>Stitch all mosaic tiles into a single full field-of-view output (instead of one OME-Zarr per tile).</p>
    <pre><code># Stitch all mosaic tiles into a single full field-of-view output
    eubi to_zarr /data/input /data/output --as_mosaic
    </code></pre>
    </details>

    <details>
    <summary><code>--view_index</code></summary>
    <p><strong>Type:</strong>&nbsp; `int` or `str`</p>
    <p><strong>Default:</strong>&nbsp; `0`</p>
    <p>View(s) to read. Pass an integer, `all`, or comma-separated integers; each selected view becomes a separate OME-Zarr (named `_view{N}`) unless `--concat_views` stacks them along the channel axis.</p>
    <pre><code># Write each view as its own OME-Zarr
    eubi to_zarr /data/input /data/output --view_index all
    </code></pre>
    <pre><code># Concatenate all views along the channel axis into one output
    eubi to_zarr /data/input /data/output --view_index all --concat_views
    </code></pre>
    </details>

    <details>
    <summary><code>--phase_index</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `0`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 0</p>
    <p>Phase index.</p>
    <pre><code>eubi to_zarr /data/input /data/output --phase_index 1
    </code></pre>
    </details>

    <details>
    <summary><code>--illumination_index</code></summary>
    <p><strong>Type:</strong>&nbsp; `int` or `str`</p>
    <p><strong>Default:</strong>&nbsp; `0`</p>
    <p>Illumination(s) to read. Pass an integer, `all`, or comma-separated integers; each selected illumination becomes a separate OME-Zarr (named `_illu{N}`) unless `--concat_illuminations` stacks them along the channel axis.</p>
    <pre><code># Write each illumination as its own OME-Zarr
    eubi to_zarr /data/input /data/output --illumination_index all
    </code></pre>
    <pre><code># Concatenate all illuminations along the channel axis into one output
    eubi to_zarr /data/input /data/output --illumination_index all --concat_illuminations
    </code></pre>
    </details>

    <details>
    <summary><code>--scene_index</code></summary>
    <p><strong>Type:</strong>&nbsp; `int` or `str`</p>
    <p><strong>Default:</strong>&nbsp; `0`</p>
    <p>Scene / series index to read. Pass an integer for a single scene, `all` to convert each scene to a separate zarr group, or comma-separated integers (e.g. `0,2,4`) for a subset of scenes.</p>
    <pre><code># Convert only scene 2 from a multi-scene file
    eubi to_zarr /data/input /data/output --scene_index 2
    </code></pre>
    <pre><code># Convert every scene to a separate OME-Zarr group
    eubi to_zarr /data/input /data/output --scene_index all
    </code></pre>
    <pre><code># Convert scenes 0, 2 and 4 only
    eubi to_zarr /data/input /data/output --scene_index 0,2,4
    </code></pre>
    </details>

    <details>
    <summary><code>--rotation_index</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `0`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 0</p>
    <p>Rotation index.</p>
    <pre><code>eubi to_zarr /data/input /data/output --rotation_index 0
    </code></pre>
    </details>

    <details>
    <summary><code>--mosaic_tile_index</code></summary>
    <p><strong>Type:</strong>&nbsp; `int` or `str` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Mosaic tile(s) to read when **not** stitching. Pass an integer, `all`, or comma-separated integers; each selected tile becomes a separate OME-Zarr (named `_tile{N}`). Use `--as_mosaic` instead to stitch tiles into one output. Composes with scene / view / illumination selection (cartesian product).</p>
    <pre><code># Write every tile as its own OME-Zarr (default — each tile separate)
    eubi to_zarr /data/input /data/output --mosaic_tile_index all
    </code></pre>
    <pre><code># Write only tiles 0 and 2 (each as a separate output)
    eubi to_zarr /data/input /data/output --mosaic_tile_index 0,2
    </code></pre>
    <pre><code># Stitch all tiles into one mosaic instead of writing them separately
    eubi to_zarr /data/input /data/output --as_mosaic
    </code></pre>
    </details>

    <details>
    <summary><code>--sample_index</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `0`</p>
    <p><strong>Valid values:</strong>&nbsp; ≥ 0</p>
    <p>Sample index.</p>
    <pre><code>eubi to_zarr /data/input /data/output --sample_index 0
    </code></pre>
    </details>

    <details>
    <summary><code>--force_bioformats</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `False`</p>
    <p><strong>Valid values:</strong>&nbsp; `--force_bioformats` to enable &nbsp;·&nbsp; `--force_bioformats False` to disable</p>
    <p>Force the Java Bio-Formats reader even for formats EuBI-Bridge reads natively (CZI, ND2, LIF, IMS…). Useful as a fallback when a native read fails.</p>
    <pre><code>eubi to_zarr /data/input /data/output --force_bioformats
    </code></pre>
    </details>

    <details>
    <summary><code>--concat_views</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `False`</p>
    <p><strong>Valid values:</strong>&nbsp; `--concat_views` to enable &nbsp;·&nbsp; `--concat_views False` to disable</p>
    <p>When reading multiple views, stack them along the channel axis into one output (cartesian product with existing channels) instead of writing one OME-Zarr per view.</p>
    <pre><code># Stack every view onto the channel axis (cartesian with existing channels)
    eubi to_zarr /data/input /data/output --view_index all --concat_views
    </code></pre>
    </details>

    <details>
    <summary><code>--concat_illuminations</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `False`</p>
    <p><strong>Valid values:</strong>&nbsp; `--concat_illuminations` to enable &nbsp;·&nbsp; `--concat_illuminations False` to disable</p>
    <p>When reading multiple illuminations, stack them along the channel axis into one output (cartesian product with existing channels) instead of writing one OME-Zarr per illumination.</p>
    <pre><code># Stack every illumination onto the channel axis (cartesian with existing channels)
    eubi to_zarr /data/input /data/output --illumination_index all --concat_illuminations
    </code></pre>
    </details>


    </details>

    <details>
    <summary>Concatenation overrides</summary>

    <details>
    <summary><code>--concatenation_axes</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `int` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Axes to concatenate, e.g. `'c'`.</p>
    <pre><code># Plan a channel concatenation
    eubi validate_aggregative /data/input /data/output --concatenation_axes c --channel_tag raw,mask
    </code></pre>
    <pre><code># Plan a z + channel concatenation
    eubi validate_aggregative /data/input /data/output --concatenation_axes zc --z_tag slices --channel_tag raw,mask
    </code></pre>
    </details>

    <details>
    <summary><code>--time_tag</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `List` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Filename substring (or tuple of substrings) identifying the time axis.</p>
    <pre><code># Plan concatenation of three time points
    eubi validate_aggregative /data/input /data/output --concatenation_axes t --time_tag t001,t002,t003
    </code></pre>
    </details>

    <details>
    <summary><code>--channel_tag</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `List` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Filename substring (or tuple of substrings) identifying the channel axis.</p>
    <pre><code># Two channels — raw fluorescence and segmentation mask
    eubi validate_aggregative /data/input /data/output --concatenation_axes c --channel_tag raw,mask
    </code></pre>
    </details>

    <details>
    <summary><code>--z_tag</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `List` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Filename substring (or tuple of substrings) identifying the z axis.</p>
    <pre><code># Multiple z-slices acquired as separate files
    eubi validate_aggregative /data/input /data/output --concatenation_axes z --z_tag slice001,slice002,slice003
    </code></pre>
    </details>

    <details>
    <summary><code>--y_tag</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `List` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Filename substring (or tuple of substrings) identifying the y axis.</p>
    <pre><code>eubi validate_aggregative /data/input /data/output --concatenation_axes y --y_tag row_top,row_bottom
    </code></pre>
    </details>

    <details>
    <summary><code>--x_tag</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `List` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Filename substring (or tuple of substrings) identifying the x axis.</p>
    <pre><code>eubi validate_aggregative /data/input /data/output --concatenation_axes x --x_tag col_left,col_right
    </code></pre>
    </details>


    </details>


