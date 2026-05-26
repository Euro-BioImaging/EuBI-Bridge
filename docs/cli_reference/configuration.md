# Configuration

Auto-generated from `EuBIBridge` methods and Pydantic config models.
Types, defaults, and valid ranges are extracted directly from the source —
click any command card to expand its full parameter reference.

??? command "**`configure_cluster`**&ensp;—&ensp;Update cluster / concurrency parameters. Omitted arguments keep their current values"

    **Usage:**
    ```shell
    eubi configure_cluster [OPTIONS]
    ```

    <details>
    <summary>Optional arguments</summary>

    <details>
    <summary><code>--max_workers</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Number of parallel file-level worker processes (default 4).</p>
    <pre><code># Set the default number of parallel worker processes
    eubi configure_cluster --max_workers 8
    </code></pre>
    </details>

    <details>
    <summary><code>--queue_size</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Internal write-queue depth per worker (default 4).</p>
    <pre><code># Allow each worker to buffer more write tasks
    eubi configure_cluster --queue_size 8
    </code></pre>
    </details>

    <details>
    <summary><code>--region_size_mb</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Region size in MB for spatial partitioning (default 256).</p>
    <pre><code># Process larger spatial regions per task (reduces overhead on large arrays)
    eubi configure_cluster --region_size_mb 512
    </code></pre>
    </details>

    <details>
    <summary><code>--memory_per_worker</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Memory limit accepted by SLURM / LocalCluster, e.g. ``'8GB'`` (default ``'1GB'``).</p>
    <pre><code># Allocate 8 GB per worker (relevant for SLURM / LocalCluster)
    eubi configure_cluster --memory_per_worker 8GB
    </code></pre>
    </details>

    <details>
    <summary><code>--max_concurrency</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>TensorStore write concurrency per worker (default 4).</p>
    <pre><code># Allow TensorStore to issue 8 write operations concurrently per worker
    eubi configure_cluster --max_concurrency 8
    </code></pre>
    </details>

    <details>
    <summary><code>--max_concurrent_scenes</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Parallel scenes within one file (default 1). Increase only when writing a multi-scene file to a large store.</p>
    <pre><code># Convert 2 scenes within a single file simultaneously
    eubi configure_cluster --max_concurrent_scenes 2
    </code></pre>
    </details>

    <details>
    <summary><code>--on_local_cluster</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p><strong>Valid values:</strong>&nbsp; `--on_local_cluster` to enable &nbsp;·&nbsp; `--on_local_cluster False` to disable</p>
    <p>Use a Dask LocalCluster backend (default False).</p>
    <pre><code># Use a Dask LocalCluster as the scheduler
    eubi configure_cluster --on_local_cluster
    </code></pre>
    </details>

    <details>
    <summary><code>--on_slurm</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p><strong>Valid values:</strong>&nbsp; `--on_slurm` to enable &nbsp;·&nbsp; `--on_slurm False` to disable</p>
    <p>Submit jobs to a SLURM cluster (default False).</p>
    <pre><code># Submit workers to SLURM and set partition + account
    eubi configure_cluster --on_slurm --slurm_account myaccount --slurm_partition gpu
    </code></pre>
    </details>

    <details>
    <summary><code>--use_threading</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p><strong>Valid values:</strong>&nbsp; `--use_threading` to enable &nbsp;·&nbsp; `--use_threading False` to disable</p>
    <p>Use ThreadPool instead of ProcessPool (default False).</p>
    <pre><code># Switch from ProcessPool to ThreadPool (useful when GIL is not a bottleneck)
    eubi configure_cluster --use_threading
    </code></pre>
    </details>

    <details>
    <summary><code>--tensorstore_data_copy_concurrency</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>TensorStore internal copy threads (default 4).</p>
    <pre><code># Allow TensorStore to copy data with 8 internal threads
    eubi configure_cluster --tensorstore_data_copy_concurrency 8
    </code></pre>
    </details>

    <details>
    <summary><code>--max_retries</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Retries on broken worker process (default 10).</p>
    <pre><code># Retry failed workers up to 5 times before aborting
    eubi configure_cluster --max_retries 5
    </code></pre>
    </details>


    </details>


??? command "**`configure_conversion`**&ensp;—&ensp;Update Zarr conversion parameters. Omitted arguments keep their current values"

    **Usage:**
    ```shell
    eubi configure_conversion [OPTIONS]
    ```

    <details>
    <summary>Optional arguments</summary>

    <details>
    <summary><code>--zarr_format</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Zarr version — ``2`` (default) or ``3``.</p>
    <pre><code># Switch to Zarr v3 (supports sharding)
    eubi configure_conversion --zarr_format 3
    </code></pre>
    </details>

    <details>
    <summary><code>--skip_dask</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p><strong>Valid values:</strong>&nbsp; `--skip_dask` to enable &nbsp;·&nbsp; `--skip_dask False` to disable</p>
    <p>Read TIFF files via zarr's native aszarr backend instead of dask. Faster for large TIFFs; ignored for non-TIFF formats.</p>
    <pre><code># Use zarr's native TIFF backend instead of dask (faster for large TIFFs)
    eubi configure_conversion --skip_dask
    </code></pre>
    </details>

    <details>
    <summary><code>--auto_chunk</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p><strong>Valid values:</strong>&nbsp; `--auto_chunk` to enable &nbsp;·&nbsp; `--auto_chunk False` to disable</p>
    <p>Auto-compute chunk shape from array size (default True).</p>
    <pre><code># Disable auto-chunking and set chunk sizes manually
    eubi configure_conversion --auto_chunk False --z_chunk 64 --y_chunk 256 --x_chunk 256
    </code></pre>
    </details>

    <details>
    <summary><code>--target_chunk_mb</code></summary>
    <p><strong>Type:</strong>&nbsp; `float`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Target chunk size in MB when auto_chunk is True.</p>
    <pre><code># Target 4 MB uncompressed chunks when auto_chunk is on
    eubi configure_conversion --target_chunk_mb 4.0
    </code></pre>
    </details>

    <details>
    <summary><code>--time_chunk</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p> Manual per-axis chunk sizes (used when auto_chunk is False).</p>
    <pre><code># Process one time point per chunk
    eubi configure_conversion --auto_chunk False --time_chunk 1
    </code></pre>
    </details>

    <details>
    <summary><code>--channel_chunk</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Chunk size along the channel axis. Applies only when `auto_chunk=False`.</p>
    <pre><code># Process one channel per chunk
    eubi configure_conversion --auto_chunk False --channel_chunk 1
    </code></pre>
    </details>

    <details>
    <summary><code>--z_chunk</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Chunk size along the z axis. Applies only when `auto_chunk=False`.</p>
    <pre><code># Set manual z chunk size (requires auto_chunk False)
    eubi configure_conversion --auto_chunk False --z_chunk 64
    </code></pre>
    </details>

    <details>
    <summary><code>--y_chunk</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Chunk size along the y axis. Applies only when `auto_chunk=False`.</p>
    <pre><code># Set manual y chunk size (requires auto_chunk False)
    eubi configure_conversion --auto_chunk False --y_chunk 256
    </code></pre>
    </details>

    <details>
    <summary><code>--x_chunk</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Chunk size along the x axis. Applies only when `auto_chunk=False`.</p>
    <pre><code># Set manual x chunk size (requires auto_chunk False)
    eubi configure_conversion --auto_chunk False --x_chunk 256
    </code></pre>
    </details>

    <details>
    <summary><code>--time_shard_coef</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p> Shard-to-chunk multipliers for Zarr v3 sharding (default 3 on spatial axes).</p>
    <pre><code># Shard = 1 × time chunk (no sharding on time axis)
    eubi configure_conversion --zarr_format 3 --time_shard_coef 1
    </code></pre>
    </details>

    <details>
    <summary><code>--channel_shard_coef</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Shard size = chunk × coef for the channel axis. Zarr v3 only — ignored for v2.</p>
    <pre><code># Shard = 1 × channel chunk
    eubi configure_conversion --zarr_format 3 --channel_shard_coef 1
    </code></pre>
    </details>

    <details>
    <summary><code>--z_shard_coef</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Shard size = chunk × coef for the z axis. Zarr v3 only — ignored for v2.</p>
    <pre><code># Shard = 5 × z chunk on the z axis
    eubi configure_conversion --zarr_format 3 --z_shard_coef 5
    </code></pre>
    </details>

    <details>
    <summary><code>--y_shard_coef</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Shard size = chunk × coef for the y axis. Zarr v3 only — ignored for v2.</p>
    <pre><code># Shard = 5 × y chunk on the y axis
    eubi configure_conversion --zarr_format 3 --y_shard_coef 5
    </code></pre>
    </details>

    <details>
    <summary><code>--x_shard_coef</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Shard size = chunk × coef for the x axis. Zarr v3 only — ignored for v2.</p>
    <pre><code># Shard = 5 × x chunk on the x axis
    eubi configure_conversion --zarr_format 3 --x_shard_coef 5
    </code></pre>
    </details>

    <details>
    <summary><code>--time_range</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p> Crop ranges as ``"start,stop"`` strings applied before writing.</p>
    <pre><code># Keep only time frames 0–9 (start inclusive, stop exclusive)
    eubi configure_conversion --time_range "0,10"
    </code></pre>
    </details>

    <details>
    <summary><code>--channel_range</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Crop the channel axis. Same format as `time_range`, e.g. `"0,2"` keeps the first two channels.</p>
    <pre><code># Keep only the first two channels
    eubi configure_conversion --channel_range "0,2"
    </code></pre>
    </details>

    <details>
    <summary><code>--z_range</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Crop the z axis. Same format as `time_range`, e.g. `"5,50"` keeps z-slices 5–49.</p>
    <pre><code># Keep z-slices 5 through 49
    eubi configure_conversion --z_range "5,50"
    </code></pre>
    </details>

    <details>
    <summary><code>--y_range</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Crop the y axis. Same format as `time_range`.</p>
    <pre><code># Crop y to the first 512 pixels
    eubi configure_conversion --y_range "0,512"
    </code></pre>
    </details>

    <details>
    <summary><code>--x_range</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Crop the x axis. Same format as `time_range`.</p>
    <pre><code># Crop x to the first 512 pixels
    eubi configure_conversion --x_range "0,512"
    </code></pre>
    </details>

    <details>
    <summary><code>--compressor</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Compression codec — ``'blosc'`` (default), ``'gzip'``, ``'zstd'``, ``'bz2'``, or ``'none'``.</p>
    <pre><code># Use zstd compression (good balance of speed and ratio)
    eubi configure_conversion --compressor zstd
    </code></pre>
    <pre><code># Disable compression entirely
    eubi configure_conversion --compressor none
    </code></pre>
    </details>

    <details>
    <summary><code>--compressor_params</code></summary>
    <p><strong>Type:</strong>&nbsp; `dict`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Dict of codec parameters, e.g. ``{'cname': 'lz4', 'clevel': 5}``.</p>
    <pre><code># Use blosc with lz4 codec at compression level 9
    eubi configure_conversion --compressor blosc --compressor_params '{"cname": "lz4", "clevel": 9}'
    </code></pre>
    </details>

    <details>
    <summary><code>--overwrite</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p><strong>Valid values:</strong>&nbsp; `--overwrite` to enable &nbsp;·&nbsp; `--overwrite False` to disable</p>
    <p>Overwrite existing output zarr (default False).</p>
    <pre><code># Allow overwriting an existing OME-Zarr at the output path
    eubi configure_conversion --overwrite
    </code></pre>
    </details>

    <details>
    <summary><code>--override_channel_names</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p><strong>Valid values:</strong>&nbsp; `--override_channel_names` to enable &nbsp;·&nbsp; `--override_channel_names False` to disable</p>
    <p>Replace output channel labels with the `channel_tag` values. For aggregative conversions with a tuple `channel_tag` only.</p>
    <pre><code># Replace channel labels with the channel_tag values from filenames
    eubi configure_conversion --override_channel_names
    </code></pre>
    </details>

    <details>
    <summary><code>--channel_intensity_limits</code></summary>
    <p><strong>Type:</strong>&nbsp; `Literal['from_dtype', 'from_array', 'auto']`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>How to set OMERO window limits — ``'from_dtype'`` (default) uses dtype min/max, ``'from_array'`` computes per-channel min/max from pixel data, ``'auto'`` lets the viewer decide.</p>
    <pre><code># Compute display window limits from actual pixel data
    eubi configure_conversion --channel_intensity_limits from_array
    </code></pre>
    <pre><code># Let the viewer compute limits automatically
    eubi configure_conversion --channel_intensity_limits auto
    </code></pre>
    </details>

    <details>
    <summary><code>--metadata_reader</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Metadata backend — ``'bfio'`` (default) or ``'bioformats'``.</p>
    <pre><code># Use the Java BioFormats backend for metadata (more formats supported)
    eubi configure_conversion --metadata_reader bioformats
    </code></pre>
    </details>

    <details>
    <summary><code>--save_omexml</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p><strong>Valid values:</strong>&nbsp; `--save_omexml` to enable &nbsp;·&nbsp; `--save_omexml False` to disable</p>
    <p>Write a companion OME-XML file alongside the zarr (default True).</p>
    <pre><code># Disable OME-XML sidecar file generation
    eubi configure_conversion --save_omexml False
    </code></pre>
    </details>

    <details>
    <summary><code>--squeeze</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p><strong>Valid values:</strong>&nbsp; `--squeeze` to enable &nbsp;·&nbsp; `--squeeze False` to disable</p>
    <p>Remove singleton dimensions before writing (default True).</p>
    <pre><code># Keep length-1 dimensions instead of removing them
    eubi configure_conversion --squeeze False
    </code></pre>
    </details>

    <details>
    <summary><code>--dtype</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Output dtype — ``'auto'`` keeps the source dtype, or any NumPy dtype string such as ``'uint16'``.</p>
    <pre><code># Cast pixel values to uint8 on write (reduces file size)
    eubi configure_conversion --dtype uint8
    </code></pre>
    <pre><code># Keep the source dtype
    eubi configure_conversion --dtype auto
    </code></pre>
    </details>

    <details>
    <summary><code>--verbose</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p><strong>Valid values:</strong>&nbsp; `--verbose` to enable &nbsp;·&nbsp; `--verbose False` to disable</p>
    <p>Log verbose progress output (default False).</p>
    <pre><code># Enable verbose per-chunk progress logging
    eubi configure_conversion --verbose
    </code></pre>
    </details>


    </details>


??? command "**`configure_downscale`**&ensp;—&ensp;Update downscale parameters. Omitted arguments keep their current values"

    **Usage:**
    ```shell
    eubi configure_downscale [OPTIONS]
    ```

    <details>
    <summary>Optional arguments</summary>

    <details>
    <summary><code>--n_layers</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Number of downscale pyramid levels to generate (default 5).</p>
    <pre><code># Generate 5 pyramid levels
    eubi configure_downscale --n_layers 5
    </code></pre>
    </details>

    <details>
    <summary><code>--min_dimension_size</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Stop adding levels when the smallest spatial dimension falls below this pixel count (default 64).</p>
    <pre><code># Stop downscaling when the smallest spatial dimension is below 32 px
    eubi configure_downscale --min_dimension_size 32
    </code></pre>
    </details>

    <details>
    <summary><code>--time_scale_factor</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Downscale factor along the time axis (default 1, i.e. no downscaling).</p>
    <pre><code># Disable time-axis downscaling (value 1 = no downscaling)
    eubi configure_downscale --time_scale_factor 1
    </code></pre>
    </details>

    <details>
    <summary><code>--channel_scale_factor</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Downscale factor along the channel axis (default 1).</p>
    <pre><code># Disable channel-axis downscaling
    eubi configure_downscale --channel_scale_factor 1
    </code></pre>
    </details>

    <details>
    <summary><code>--z_scale_factor</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Downscale factor along the z axis (default 2).</p>
    <pre><code># Isotropic downscaling on all spatial axes
    eubi configure_downscale --z_scale_factor 2 --y_scale_factor 2 --x_scale_factor 2
    </code></pre>
    <pre><code># Anisotropic — skip z downscaling for thick sections
    eubi configure_downscale --z_scale_factor 1 --y_scale_factor 2 --x_scale_factor 2
    </code></pre>
    </details>

    <details>
    <summary><code>--y_scale_factor</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Downscale factor along the y axis (default 2).</p>
    <pre><code>eubi configure_downscale --y_scale_factor 2
    </code></pre>
    </details>

    <details>
    <summary><code>--x_scale_factor</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Downscale factor along the x axis (default 2).</p>
    <pre><code>eubi configure_downscale --x_scale_factor 2
    </code></pre>
    </details>


    </details>


??? command "**`configure_readers`**&ensp;—&ensp;Update file-reader parameters. Omitted arguments keep their current values"

    **Usage:**
    ```shell
    eubi configure_readers [OPTIONS]
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
    eubi configure_readers --as_mosaic
    </code></pre>
    </details>

    <details>
    <summary><code>--view_index</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>View index for multi-view formats (default 0).</p>
    <pre><code># Select the second view in a multi-view dataset
    eubi configure_readers --view_index 1
    </code></pre>
    </details>

    <details>
    <summary><code>--phase_index</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Phase index (default 0).</p>
    <pre><code># Select phase index 1
    eubi configure_readers --phase_index 1
    </code></pre>
    </details>

    <details>
    <summary><code>--illumination_index</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Illumination index (default 0).</p>
    <pre><code># Select illumination index 1
    eubi configure_readers --illumination_index 1
    </code></pre>
    </details>

    <details>
    <summary><code>--scene_index</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Scene / series index to read.  Pass an integer, ``'all'``, or a comma-separated list such as ``'0,2,4'`` (default 0).</p>
    <pre><code># Always read scene 2 from multi-scene files
    eubi configure_readers --scene_index 2
    </code></pre>
    <pre><code># Convert each scene to a separate OME-Zarr group
    eubi configure_readers --scene_index all
    </code></pre>
    <pre><code># Convert only scenes 0, 2 and 4
    eubi configure_readers --scene_index 0,2,4
    </code></pre>
    </details>

    <details>
    <summary><code>--rotation_index</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Rotation index (default 0).</p>
    <pre><code># Select rotation index 0
    eubi configure_readers --rotation_index 0
    </code></pre>
    </details>

    <details>
    <summary><code>--mosaic_tile_index</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Mosaic tile index.  Pass an integer, ``'all'``, or a comma-separated list (default None = all tiles).</p>
    <pre><code># Read only the first tile of each mosaic acquisition
    eubi configure_readers --as_mosaic --mosaic_tile_index 0
    </code></pre>
    <pre><code># Read all tiles (default behaviour)
    eubi configure_readers --as_mosaic --mosaic_tile_index all
    </code></pre>
    </details>

    <details>
    <summary><code>--sample_index</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Sample index (default 0).</p>
    <pre><code># Select sample index 0
    eubi configure_readers --sample_index 0
    </code></pre>
    </details>


    </details>


??? command "**`configure_concatenation`**&ensp;—&ensp;Update aggregative (concatenation) parameters. Omitted arguments keep their current values"

    **Usage:**
    ```shell
    eubi configure_concatenation [OPTIONS]
    ```

    <details>
    <summary>Optional arguments</summary>

    <details>
    <summary><code>--concatenation_axes</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `int` or `None`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Axes along which to concatenate files.  Pass a string of axis letters such as ``'tc'`` or ``'z'``, or an integer axis index.  ``None`` disables aggregative mode entirely.</p>
    <pre><code># Concatenate files along the channel axis
    eubi configure_concatenation --concatenation_axes c --channel_tag raw,mask
    </code></pre>
    <pre><code># Concatenate along both z and channel axes simultaneously
    eubi configure_concatenation --concatenation_axes zc --z_tag slices --channel_tag raw,mask
    </code></pre>
    </details>

    <details>
    <summary><code>--time_tag</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `tuple` or `None`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Filename substring (or tuple of substrings) that identifies a file as contributing to the time axis.</p>
    <pre><code># Identify three time-point files by substrings in their names
    eubi configure_concatenation --concatenation_axes t --time_tag t001,t002,t003
    </code></pre>
    </details>

    <details>
    <summary><code>--channel_tag</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `tuple` or `None`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Filename substring (or tuple of substrings) identifying the channel axis.</p>
    <pre><code># Two channels — raw fluorescence and segmentation mask
    eubi configure_concatenation --concatenation_axes c --channel_tag raw,mask
    </code></pre>
    <pre><code># Single channel — DAPI only
    eubi configure_concatenation --concatenation_axes c --channel_tag DAPI
    </code></pre>
    </details>

    <details>
    <summary><code>--z_tag</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `tuple` or `None`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Filename substring (or tuple of substrings) identifying the z axis.</p>
    <pre><code># Multiple z-slices acquired as separate files
    eubi configure_concatenation --concatenation_axes z --z_tag slice001,slice002,slice003
    </code></pre>
    </details>

    <details>
    <summary><code>--y_tag</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `tuple` or `None`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Filename substring (or tuple of substrings) identifying the y axis.</p>
    <pre><code># Two rows of a tile scan
    eubi configure_concatenation --concatenation_axes y --y_tag row_top,row_bottom
    </code></pre>
    </details>

    <details>
    <summary><code>--x_tag</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `tuple` or `None`</p>
    <p><strong>Default:</strong>&nbsp; `default`</p>
    <p>Filename substring (or tuple of substrings) identifying the x axis.</p>
    <pre><code># Two columns of a tile scan
    eubi configure_concatenation --concatenation_axes x --x_tag col_left,col_right
    </code></pre>
    </details>


    </details>


