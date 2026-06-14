# Configuration

Auto-generated from `EuBIBridge` methods and Pydantic config models.
Types, defaults, and valid ranges are extracted directly from the source —
click any command card to expand its full parameter reference.

??? command "**`configure cluster`**&ensp;—&ensp;Update cluster / concurrency parameters. Omitted arguments keep their current values"

    **Usage:**
    ```shell
    eubi configure cluster [OPTIONS]
    ```

    <details>
    <summary>Optional arguments</summary>

    <details>
    <summary><code>--max_workers</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Number of parallel file-level worker processes (default 4).</p>
    <pre><code># Set the default number of parallel worker processes
    eubi configure cluster --max_workers 8
    </code></pre>
    </details>

    <details>
    <summary><code>--queue_size</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Internal write-queue depth per worker (default 4).</p>
    <pre><code># Allow each worker to buffer more write tasks
    eubi configure cluster --queue_size 8
    </code></pre>
    </details>

    <details>
    <summary><code>--region_size_mb</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Region size in MB for spatial partitioning (default 256).</p>
    <pre><code># Process larger spatial regions per task (reduces overhead on large arrays)
    eubi configure cluster --region_size_mb 512
    </code></pre>
    </details>

    <details>
    <summary><code>--memory_per_worker</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Memory limit accepted by SLURM / LocalCluster, e.g. `'8GB'` (default `'1GB'`).</p>
    <pre><code># Allocate 8 GB per worker (relevant for SLURM / LocalCluster)
    eubi configure cluster --memory_per_worker 8GB
    </code></pre>
    </details>

    <details>
    <summary><code>--max_concurrency</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>TensorStore write concurrency per worker (default 4).</p>
    <pre><code># Allow TensorStore to issue 8 write operations concurrently per worker
    eubi configure cluster --max_concurrency 8
    </code></pre>
    </details>

    <details>
    <summary><code>--max_concurrent_scenes</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Parallel scenes within one file (default 1). Increase only when writing a multi-scene file to a large store.</p>
    <pre><code># Convert 2 scenes within a single file simultaneously
    eubi configure cluster --max_concurrent_scenes 2
    </code></pre>
    </details>

    <details>
    <summary><code>--max_concurrent_downscale_layers</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Number of downscaled pyramid layers written concurrently per scene (default 3).</p>
    <pre><code>eubi to_zarr /data/input /data/output --max_concurrent_downscale_layers 2
    </code></pre>
    </details>

    <details>
    <summary><code>--on_local_cluster</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p><strong>Valid values:</strong>&nbsp; `--on_local_cluster` to enable &nbsp;·&nbsp; `--on_local_cluster False` to disable</p>
    <p>Use a Dask LocalCluster backend (default False).</p>
    <pre><code># Use a Dask LocalCluster as the scheduler
    eubi configure cluster --on_local_cluster
    </code></pre>
    </details>

    <details>
    <summary><code>--on_slurm</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p><strong>Valid values:</strong>&nbsp; `--on_slurm` to enable &nbsp;·&nbsp; `--on_slurm False` to disable</p>
    <p>Submit jobs to a SLURM cluster (default False).</p>
    <pre><code># Submit workers to SLURM and set partition + account
    eubi configure cluster --on_slurm --slurm_account myaccount --slurm_partition gpu
    </code></pre>
    </details>

    <details>
    <summary><code>--use_threading</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p><strong>Valid values:</strong>&nbsp; `--use_threading` to enable &nbsp;·&nbsp; `--use_threading False` to disable</p>
    <p>Use ThreadPool instead of ProcessPool (default False).</p>
    <pre><code># Switch from ProcessPool to ThreadPool (useful when GIL is not a bottleneck)
    eubi configure cluster --use_threading
    </code></pre>
    </details>

    <details>
    <summary><code>--tensorstore_data_copy_concurrency</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>TensorStore internal copy threads (default 4).</p>
    <pre><code># Allow TensorStore to copy data with 8 internal threads
    eubi configure cluster --tensorstore_data_copy_concurrency 8
    </code></pre>
    </details>

    <details>
    <summary><code>--max_retries</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Retries on broken worker process (default 10).</p>
    <pre><code># Retry failed workers up to 5 times before aborting
    eubi configure cluster --max_retries 5
    </code></pre>
    </details>

    <details>
    <summary><code>--bf_read_concurrency</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Dask thread count for parallel bfio tile reads (default 4).  `None` lets dask choose (cpu_count).</p>
    </details>

    <details>
    <summary><code>--bf_tile_size_mb</code></summary>
    <p><strong>Type:</strong>&nbsp; `float`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Tile size budget in MB for bfio tiled reading (default 512).</p>
    </details>

    <details>
    <summary><code>--jvm_memory</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Maximum JVM heap for Bio-Formats, e.g. `'8GB'`, `'4GB'`. Accepts `'NGB'` / `'NMB'` (like memory_per_worker) and normalises internally to JVM format (`'Ng'` / `'Nm'`).  Default `'2g'`.</p>
    </details>

    <details>
    <summary><code>--slurm_time</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>SLURM job wall-clock limit, e.g. `'24:00:00'` (used only when on_slurm=True).</p>
    <pre><code># Set SLURM wall-clock limit to 48 hours
    eubi configure cluster --on_slurm --slurm_time 48:00:00
    </code></pre>
    </details>

    <details>
    <summary><code>--slurm_account</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>SLURM account/allocation to charge (default None).</p>
    <pre><code># Charge SLURM jobs to 'myproject'
    eubi configure cluster --on_slurm --slurm_account myproject
    </code></pre>
    </details>

    <details>
    <summary><code>--slurm_partition</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>SLURM partition/queue to submit to (default None).</p>
    <pre><code># Submit to the 'gpu' partition
    eubi configure cluster --on_slurm --slurm_partition gpu
    </code></pre>
    </details>

    <details>
    <summary><code>--slurm_worker_timeout</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Seconds to wait for SLURM workers to start (default 300).</p>
    <pre><code># Wait up to 10 minutes for SLURM workers to start
    eubi configure cluster --on_slurm --slurm_worker_timeout 600
    </code></pre>
    </details>

    <details>
    <summary><code>--slurm_sif_path</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Path to an Apptainer/Singularity SIF image to run workers in (default None).</p>
    <pre><code>eubi to_zarr /data/input /data/output --on_slurm --slurm_sif_path /apps/eubi.sif
    </code></pre>
    </details>


    </details>


??? command "**`configure conversion`**&ensp;—&ensp;Update Zarr conversion parameters. Omitted arguments keep their current values"

    **Usage:**
    ```shell
    eubi configure conversion [OPTIONS]
    ```

    <details>
    <summary>Optional arguments</summary>

    <details>
    <summary><code>--ome_zarr_version</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>OME-Zarr (NGFF) version to write, e.g. `'0.4'` or `'0.5'`.  This is the preferred control and supersedes `zarr_format` (the zarr container format is derived from it).</p>
    <pre><code># Default to OME-Zarr 0.5 (Zarr v3, enables sharding)
    eubi configure conversion --ome_zarr_version 0.5
    </code></pre>
    <pre><code># Default to OME-Zarr 0.4 (Zarr v2)
    eubi configure conversion --ome_zarr_version 0.4
    </code></pre>
    </details>

    <details>
    <summary><code>--zarr_format</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>DEPRECATED zarr container version — `2` (default) or `3`.  Used only when `ome_zarr_version` is unset/unrecognised; prefer `ome_zarr_version`.</p>
    <pre><code># Deprecated — prefer 'ome_zarr_version'. Switch to Zarr v3 (supports sharding)
    eubi configure conversion --zarr_format 3
    </code></pre>
    </details>

    <details>
    <summary><code>--skip_dask</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p><strong>Valid values:</strong>&nbsp; `--skip_dask` to enable &nbsp;·&nbsp; `--skip_dask False` to disable</p>
    <p>Read TIFF files via zarr's native aszarr backend instead of dask. Faster for large TIFFs; ignored for non-TIFF formats.</p>
    <pre><code># Use zarr's native TIFF backend instead of dask (faster for large TIFFs)
    eubi configure conversion --skip_dask
    </code></pre>
    </details>

    <details>
    <summary><code>--auto_chunk</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p><strong>Valid values:</strong>&nbsp; `--auto_chunk` to enable &nbsp;·&nbsp; `--auto_chunk False` to disable</p>
    <p>Auto-compute chunk shape from array size (default True).</p>
    <pre><code># Disable auto-chunking and set chunk sizes manually
    eubi configure conversion --auto_chunk False --z_chunk 64 --y_chunk 256 --x_chunk 256
    </code></pre>
    </details>

    <details>
    <summary><code>--target_chunk_mb</code></summary>
    <p><strong>Type:</strong>&nbsp; `float`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Target chunk size in MB when auto_chunk is True.</p>
    <pre><code># Target 4 MB uncompressed chunks when auto_chunk is on
    eubi configure conversion --target_chunk_mb 4.0
    </code></pre>
    </details>

    <details>
    <summary><code>--time_chunk</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p> Manual per-axis chunk sizes (used when auto_chunk is False).</p>
    <pre><code># Process one time point per chunk
    eubi configure conversion --auto_chunk False --time_chunk 1
    </code></pre>
    </details>

    <details>
    <summary><code>--channel_chunk</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Chunk size along the channel axis. Applies only when `auto_chunk=False`.</p>
    <pre><code># Process one channel per chunk
    eubi configure conversion --auto_chunk False --channel_chunk 1
    </code></pre>
    </details>

    <details>
    <summary><code>--z_chunk</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Chunk size along the z axis. Applies only when `auto_chunk=False`.</p>
    <pre><code># Set manual z chunk size (requires auto_chunk False)
    eubi configure conversion --auto_chunk False --z_chunk 64
    </code></pre>
    </details>

    <details>
    <summary><code>--y_chunk</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Chunk size along the y axis. Applies only when `auto_chunk=False`.</p>
    <pre><code># Set manual y chunk size (requires auto_chunk False)
    eubi configure conversion --auto_chunk False --y_chunk 256
    </code></pre>
    </details>

    <details>
    <summary><code>--x_chunk</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Chunk size along the x axis. Applies only when `auto_chunk=False`.</p>
    <pre><code># Set manual x chunk size (requires auto_chunk False)
    eubi configure conversion --auto_chunk False --x_chunk 256
    </code></pre>
    </details>

    <details>
    <summary><code>--time_shard_coef</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p> Shard-to-chunk multipliers for Zarr v3 sharding (default 3 on spatial axes).</p>
    <pre><code># Shard = 1 × time chunk (no sharding on time axis)
    eubi configure conversion --ome_zarr_version 0.5 --time_shard_coef 1
    </code></pre>
    </details>

    <details>
    <summary><code>--channel_shard_coef</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Shard size = chunk × coef for the channel axis. Zarr v3 only — ignored for v2.</p>
    <pre><code># Shard = 1 × channel chunk
    eubi configure conversion --ome_zarr_version 0.5 --channel_shard_coef 1
    </code></pre>
    </details>

    <details>
    <summary><code>--z_shard_coef</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Shard size = chunk × coef for the z axis. Zarr v3 only — ignored for v2.</p>
    <pre><code># Shard = 5 × z chunk on the z axis
    eubi configure conversion --ome_zarr_version 0.5 --z_shard_coef 5
    </code></pre>
    </details>

    <details>
    <summary><code>--y_shard_coef</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Shard size = chunk × coef for the y axis. Zarr v3 only — ignored for v2.</p>
    <pre><code># Shard = 5 × y chunk on the y axis
    eubi configure conversion --ome_zarr_version 0.5 --y_shard_coef 5
    </code></pre>
    </details>

    <details>
    <summary><code>--x_shard_coef</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Shard size = chunk × coef for the x axis. Zarr v3 only — ignored for v2.</p>
    <pre><code># Shard = 5 × x chunk on the x axis
    eubi configure conversion --ome_zarr_version 0.5 --x_shard_coef 5
    </code></pre>
    </details>

    <details>
    <summary><code>--time_range</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p> Crop ranges as `"start,stop"` strings applied before writing.</p>
    <pre><code># Keep only time frames 0–9 (start inclusive, stop exclusive)
    eubi configure conversion --time_range "0,10"
    </code></pre>
    </details>

    <details>
    <summary><code>--channel_range</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Crop the channel axis. Same format as `time_range`, e.g. `"0,2"` keeps the first two channels.</p>
    <pre><code># Keep only the first two channels
    eubi configure conversion --channel_range "0,2"
    </code></pre>
    </details>

    <details>
    <summary><code>--z_range</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Crop the z axis. Same format as `time_range`, e.g. `"5,50"` keeps z-slices 5–49.</p>
    <pre><code># Keep z-slices 5 through 49
    eubi configure conversion --z_range "5,50"
    </code></pre>
    </details>

    <details>
    <summary><code>--y_range</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Crop the y axis. Same format as `time_range`.</p>
    <pre><code># Crop y to the first 512 pixels
    eubi configure conversion --y_range "0,512"
    </code></pre>
    </details>

    <details>
    <summary><code>--x_range</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Crop the x axis. Same format as `time_range`.</p>
    <pre><code># Crop x to the first 512 pixels
    eubi configure conversion --x_range "0,512"
    </code></pre>
    </details>

    <details>
    <summary><code>--compressor</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Compression codec — `'blosc'` (default), `'gzip'`, `'zstd'`, `'bz2'`, or `'none'`.</p>
    <pre><code># Use zstd compression (good balance of speed and ratio)
    eubi configure conversion --compressor zstd
    </code></pre>
    <pre><code># Disable compression entirely
    eubi configure conversion --compressor none
    </code></pre>
    </details>

    <details>
    <summary><code>--compressor_params</code></summary>
    <p><strong>Type:</strong>&nbsp; `dict`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Dict of codec parameters, e.g. `{'cname': 'lz4', 'clevel': 5}`.</p>
    <pre><code># Use blosc with lz4 codec at compression level 9
    eubi configure conversion --compressor blosc --compressor_params '{"cname": "lz4", "clevel": 9}'
    </code></pre>
    </details>

    <details>
    <summary><code>--overwrite</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p><strong>Valid values:</strong>&nbsp; `--overwrite` to enable &nbsp;·&nbsp; `--overwrite False` to disable</p>
    <p>Overwrite existing output zarr (default False).</p>
    <pre><code># Allow overwriting an existing OME-Zarr at the output path
    eubi configure conversion --overwrite
    </code></pre>
    </details>

    <details>
    <summary><code>--override_channel_names</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p><strong>Valid values:</strong>&nbsp; `--override_channel_names` to enable &nbsp;·&nbsp; `--override_channel_names False` to disable</p>
    <p>Replace output channel labels with the `channel_tag` values. For aggregative conversions with a tuple `channel_tag` only.</p>
    <pre><code># Replace channel labels with the channel_tag values from filenames
    eubi configure conversion --override_channel_names
    </code></pre>
    </details>

    <details>
    <summary><code>--channel_intensity_limits</code></summary>
    <p><strong>Type:</strong>&nbsp; `Literal['from_dtype', 'from_array', 'auto']`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>How to set OMERO window limits — `'from_dtype'` (default) uses dtype min/max, `'from_array'` computes per-channel min/max from pixel data, `'auto'` lets the viewer decide.</p>
    <pre><code># Compute display window limits from actual pixel data
    eubi configure conversion --channel_intensity_limits from_array
    </code></pre>
    <pre><code># Let the viewer compute limits automatically
    eubi configure conversion --channel_intensity_limits auto
    </code></pre>
    </details>

    <details>
    <summary><code>--metadata_reader</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Metadata backend — `'bfio'` (default) or `'bioformats'`.</p>
    <pre><code># Use the Java BioFormats backend for metadata (more formats supported)
    eubi configure conversion --metadata_reader bioformats
    </code></pre>
    </details>

    <details>
    <summary><code>--save_omexml</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p><strong>Valid values:</strong>&nbsp; `--save_omexml` to enable &nbsp;·&nbsp; `--save_omexml False` to disable</p>
    <p>Write a companion OME-XML file alongside the zarr (default True).</p>
    <pre><code># Disable OME-XML sidecar file generation
    eubi configure conversion --save_omexml False
    </code></pre>
    </details>

    <details>
    <summary><code>--squeeze</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p><strong>Valid values:</strong>&nbsp; `--squeeze` to enable &nbsp;·&nbsp; `--squeeze False` to disable</p>
    <p>Remove singleton dimensions before writing (default True).</p>
    <pre><code># Keep length-1 dimensions instead of removing them
    eubi configure conversion --squeeze False
    </code></pre>
    </details>

    <details>
    <summary><code>--dtype</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Output dtype — `'auto'` keeps the source dtype, or any NumPy dtype string such as `'uint16'`.</p>
    <pre><code># Cast pixel values to uint8 on write (reduces file size)
    eubi configure conversion --dtype uint8
    </code></pre>
    <pre><code># Keep the source dtype
    eubi configure conversion --dtype auto
    </code></pre>
    </details>

    <details>
    <summary><code>--verbose</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p><strong>Valid values:</strong>&nbsp; `--verbose` to enable &nbsp;·&nbsp; `--verbose False` to disable</p>
    <p>Log verbose progress output (default False).</p>
    <pre><code># Enable verbose per-chunk progress logging
    eubi configure conversion --verbose
    </code></pre>
    </details>


    </details>


??? command "**`configure downscale`**&ensp;—&ensp;Update downscale parameters. Omitted arguments keep their current values"

    **Usage:**
    ```shell
    eubi configure downscale [OPTIONS]
    ```

    <details>
    <summary>Optional arguments</summary>

    <details>
    <summary><code>--n_layers</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Number of downscale pyramid levels to generate. `None` (default) auto-derives the count from `min_dimension_size`.</p>
    <pre><code># Generate 5 pyramid levels
    eubi configure downscale --n_layers 5
    </code></pre>
    </details>

    <details>
    <summary><code>--min_dimension_size</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Stop adding levels when the smallest spatial dimension falls below this pixel count (default 64).</p>
    <pre><code># Stop downscaling when the smallest spatial dimension is below 32 px
    eubi configure downscale --min_dimension_size 32
    </code></pre>
    </details>

    <details>
    <summary><code>--time_scale_factor</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Downscale factor along the time axis (default 1, i.e. no downscaling).</p>
    <pre><code># Disable time-axis downscaling (value 1 = no downscaling)
    eubi configure downscale --time_scale_factor 1
    </code></pre>
    </details>

    <details>
    <summary><code>--channel_scale_factor</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Downscale factor along the channel axis (default 1).</p>
    <pre><code># Disable channel-axis downscaling
    eubi configure downscale --channel_scale_factor 1
    </code></pre>
    </details>

    <details>
    <summary><code>--z_scale_factor</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Downscale factor along the z axis (default 2).</p>
    <pre><code># Isotropic downscaling on all spatial axes
    eubi configure downscale --z_scale_factor 2 --y_scale_factor 2 --x_scale_factor 2
    </code></pre>
    <pre><code># Anisotropic — skip z downscaling for thick sections
    eubi configure downscale --z_scale_factor 1 --y_scale_factor 2 --x_scale_factor 2
    </code></pre>
    </details>

    <details>
    <summary><code>--y_scale_factor</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Downscale factor along the y axis (default 2).</p>
    <pre><code>eubi configure downscale --y_scale_factor 2
    </code></pre>
    </details>

    <details>
    <summary><code>--x_scale_factor</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Downscale factor along the x axis (default 2).</p>
    <pre><code>eubi configure downscale --x_scale_factor 2
    </code></pre>
    </details>

    <details>
    <summary><code>--downscale_method</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Pixel aggregation method for downscaling — one of 'simple', 'mean', 'median', 'min', 'max', 'mode' (default 'simple').</p>
    <pre><code># Use mean aggregation (recommended for fluorescence images)
    eubi configure downscale --downscale_method mean
    </code></pre>
    <pre><code># Use nearest-neighbour striding (fastest, recommended for label images)
    eubi configure downscale --downscale_method simple
    </code></pre>
    </details>

    <details>
    <summary><code>--keep_existing_resolutions</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p><strong>Valid values:</strong>&nbsp; `--keep_existing_resolutions` to enable &nbsp;·&nbsp; `--keep_existing_resolutions False` to disable</p>
    <p>If the input format already carries its own multiscale pyramid (e.g. `.ims`, `.zarr`), write each existing resolution level straight to the output OME-Zarr instead of rebuilding the pyramid with the above scale factors (default False).</p>
    <pre><code># Copy a source pyramid (.ims / .zarr) verbatim instead of rebuilding it
    eubi configure downscale --keep_existing_resolutions
    </code></pre>
    </details>

    <details>
    <summary><code>--apply_smart_downscaling</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p><strong>Valid values:</strong>&nbsp; `--apply_smart_downscaling` to enable &nbsp;·&nbsp; `--apply_smart_downscaling False` to disable</p>
    <p>Make the first pyramid level use automatically-computed factors that drive the voxels toward isotropy (default False).</p>
    <pre><code># Pick per-axis downscale factors automatically from voxel anisotropy
    eubi configure downscale --apply_smart_downscaling
    </code></pre>
    </details>

    <details>
    <summary><code>--time_smart_scale_factor</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Per-axis override for smart downscaling on the time axis (default None = auto).</p>
    <pre><code>eubi to_zarr /data/input /data/output --apply_smart_downscaling --time_smart_scale_factor 1
    </code></pre>
    </details>

    <details>
    <summary><code>--z_smart_scale_factor</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Per-axis smart-downscaling override for z.</p>
    <pre><code>eubi to_zarr /data/input /data/output --apply_smart_downscaling --z_smart_scale_factor 1
    </code></pre>
    </details>

    <details>
    <summary><code>--y_smart_scale_factor</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Per-axis smart-downscaling override for y.</p>
    <pre><code>eubi to_zarr /data/input /data/output --apply_smart_downscaling --y_smart_scale_factor 2
    </code></pre>
    </details>

    <details>
    <summary><code>--x_smart_scale_factor</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Per-axis smart-downscaling override for x.</p>
    <pre><code>eubi to_zarr /data/input /data/output --apply_smart_downscaling --x_smart_scale_factor 2
    </code></pre>
    </details>


    </details>


??? command "**`configure readers`**&ensp;—&ensp;Update file-reader parameters. Omitted arguments keep their current values"

    **Usage:**
    ```shell
    eubi configure readers [OPTIONS]
    ```

    <details>
    <summary>Optional arguments</summary>

    <details>
    <summary><code>--as_mosaic</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p><strong>Valid values:</strong>&nbsp; `--as_mosaic` to enable &nbsp;·&nbsp; `--as_mosaic False` to disable</p>
    <p>Treat tiled acquisitions as a stitched mosaic (default False).</p>
    <pre><code># Stitch tiled acquisitions into a single mosaic array on read
    eubi configure readers --as_mosaic
    </code></pre>
    </details>

    <details>
    <summary><code>--view_index</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>View index for multi-view formats (default 0). Pass `'all'` or a comma-separated list (e.g. `'0,2'`) to expose multiple views.</p>
    <pre><code># Write each view separately
    eubi configure readers --view_index all
    </code></pre>
    <pre><code># Select only the second view
    eubi configure readers --view_index 1
    </code></pre>
    </details>

    <details>
    <summary><code>--phase_index</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Phase index (default 0).</p>
    <pre><code># Select phase index 1
    eubi configure readers --phase_index 1
    </code></pre>
    </details>

    <details>
    <summary><code>--illumination_index</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Illumination index (default 0). Pass `'all'` or a comma-separated list to expose multiple illuminations.</p>
    <pre><code># Write each illumination separately
    eubi configure readers --illumination_index all
    </code></pre>
    <pre><code># Select only the second illumination
    eubi configure readers --illumination_index 1
    </code></pre>
    </details>

    <details>
    <summary><code>--scene_index</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Scene / series index to read.  Pass an integer, `'all'`, or a comma-separated list such as `'0,2,4'` (default 0).</p>
    <pre><code># Always read scene 2 from multi-scene files
    eubi configure readers --scene_index 2
    </code></pre>
    <pre><code># Convert each scene to a separate OME-Zarr group
    eubi configure readers --scene_index all
    </code></pre>
    <pre><code># Convert only scenes 0, 2 and 4
    eubi configure readers --scene_index 0,2,4
    </code></pre>
    </details>

    <details>
    <summary><code>--rotation_index</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Rotation index (default 0).</p>
    <pre><code># Select rotation index 0
    eubi configure readers --rotation_index 0
    </code></pre>
    </details>

    <details>
    <summary><code>--mosaic_tile_index</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Mosaic tile index.  Pass an integer, `'all'`, or a comma-separated list (default None = all tiles).</p>
    <pre><code># Write every tile separately (default)
    eubi configure readers --mosaic_tile_index all
    </code></pre>
    <pre><code># Write only tiles 0 and 2
    eubi configure readers --mosaic_tile_index 0,2
    </code></pre>
    </details>

    <details>
    <summary><code>--sample_index</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Sample index (default 0).</p>
    <pre><code># Select sample index 0
    eubi configure readers --sample_index 0
    </code></pre>
    </details>

    <details>
    <summary><code>--force_bioformats</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p><strong>Valid values:</strong>&nbsp; `--force_bioformats` to enable &nbsp;·&nbsp; `--force_bioformats False` to disable</p>
    <p>Force bfio tiled path even for natively-supported formats.</p>
    <pre><code># Always route reads through the Java Bio-Formats reader
    eubi configure readers --force_bioformats
    </code></pre>
    </details>

    <details>
    <summary><code>--concat_views</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p><strong>Valid values:</strong>&nbsp; `--concat_views` to enable &nbsp;·&nbsp; `--concat_views False` to disable</p>
    <p>Stack multiple views along the channel axis instead of writing separate OME-Zarr outputs (default False).</p>
    <pre><code># Always stack views onto the channel axis
    eubi configure readers --concat_views
    </code></pre>
    </details>

    <details>
    <summary><code>--concat_illuminations</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p><strong>Valid values:</strong>&nbsp; `--concat_illuminations` to enable &nbsp;·&nbsp; `--concat_illuminations False` to disable</p>
    <p>Stack multiple illuminations along the channel axis instead of writing separate OME-Zarr outputs (default False).</p>
    <pre><code># Always stack illuminations onto the channel axis
    eubi configure readers --concat_illuminations
    </code></pre>
    </details>


    </details>


??? command "**`configure concatenation`**&ensp;—&ensp;Update aggregative (concatenation) parameters. Omitted arguments keep their current values"

    **Usage:**
    ```shell
    eubi configure concatenation [OPTIONS]
    ```

    <details>
    <summary>Optional arguments</summary>

    <details>
    <summary><code>--concatenation_axes</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `int` or `None`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Axes along which to concatenate files.  Pass a string of axis letters such as `'tc'` or `'z'`, or an integer axis index.  `None` disables aggregative mode entirely.</p>
    <pre><code># Concatenate files along the channel axis
    eubi configure concatenation --concatenation_axes c --channel_tag raw,mask
    </code></pre>
    <pre><code># Concatenate along both z and channel axes simultaneously
    eubi configure concatenation --concatenation_axes zc --z_tag slices --channel_tag raw,mask
    </code></pre>
    </details>

    <details>
    <summary><code>--time_tag</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `tuple` or `None`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Filename substring (or tuple of substrings) that identifies a file as contributing to the time axis.</p>
    <pre><code># Identify three time-point files by substrings in their names
    eubi configure concatenation --concatenation_axes t --time_tag t001,t002,t003
    </code></pre>
    </details>

    <details>
    <summary><code>--channel_tag</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `tuple` or `None`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Filename substring (or tuple of substrings) identifying the channel axis.</p>
    <pre><code># Two channels — raw fluorescence and segmentation mask
    eubi configure concatenation --concatenation_axes c --channel_tag raw,mask
    </code></pre>
    <pre><code># Single channel — DAPI only
    eubi configure concatenation --concatenation_axes c --channel_tag DAPI
    </code></pre>
    </details>

    <details>
    <summary><code>--z_tag</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `tuple` or `None`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Filename substring (or tuple of substrings) identifying the z axis.</p>
    <pre><code># Multiple z-slices acquired as separate files
    eubi configure concatenation --concatenation_axes z --z_tag slice001,slice002,slice003
    </code></pre>
    </details>

    <details>
    <summary><code>--y_tag</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `tuple` or `None`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Filename substring (or tuple of substrings) identifying the y axis.</p>
    <pre><code># Two rows of a tile scan
    eubi configure concatenation --concatenation_axes y --y_tag row_top,row_bottom
    </code></pre>
    </details>

    <details>
    <summary><code>--x_tag</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `tuple` or `None`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Filename substring (or tuple of substrings) identifying the x axis.</p>
    <pre><code># Two columns of a tile scan
    eubi configure concatenation --concatenation_axes x --x_tag col_left,col_right
    </code></pre>
    </details>


    </details>


