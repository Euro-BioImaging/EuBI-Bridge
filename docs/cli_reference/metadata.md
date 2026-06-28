# Metadata

Auto-generated from `EuBIBridge` methods and Pydantic config models.
Types, defaults, and valid ranges are extracted directly from the source —
click any command card to expand its full parameter reference.

??? command "**`show_pixel_meta`**&ensp;—&ensp;Display pixel-level and channel metadata for all files in input_path"

    **Usage:**
    ```shell
    eubi show_pixel_meta INPUT_PATH [OPTIONS]
    eubi with_config NAME show_pixel_meta INPUT_PATH [OPTIONS]
    ```

    <details>
    <summary>Required arguments</summary>

    <details>
    <summary><code>--input_path</code></summary>
    <p><strong>Type:</strong>&nbsp; `Path` or `str`</p>
    <p><strong>Default:</strong>&nbsp; *required*</p>
    <p>File path or directory.</p>
    <pre><code>eubi show_pixel_meta /data/input
    </code></pre>
    </details>


    </details>

    <details>
    <summary>Optional arguments</summary>

    <details>
    <summary><code>--includes</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Comma-separated filename patterns to include.</p>
    <pre><code># Show metadata only for files whose name contains "EMPIAR"
    eubi show_pixel_meta /data/input --includes "EMPIAR"
    </code></pre>
    <pre><code># Multiple patterns — comma-separated
    eubi show_pixel_meta /data/input --includes "EMPIAR,scan_001,confocal"
    </code></pre>
    </details>

    <details>
    <summary><code>--excludes</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Comma-separated filename patterns to exclude.</p>
    <pre><code># Skip preview files
    eubi show_pixel_meta /data/input --excludes "preview"
    </code></pre>
    <pre><code># Skip multiple patterns
    eubi show_pixel_meta /data/input --excludes "preview,temp"
    </code></pre>
    </details>

    <details>
    <summary><code>--series</code></summary>
    <p><strong>Type:</strong>&nbsp; `int`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Series / scene index (default: configured scene_index).</p>
    <pre><code>eubi show_pixel_meta /data/input --series 2
    </code></pre>
    </details>

    <details>
    <summary><code>--output_file</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Write output to this path; use `.html` for HTML output.</p>
    <pre><code># Save as an HTML report you can open in a browser
    eubi show_pixel_meta /data/input --output_file report.html
    </code></pre>
    <pre><code># Save as plain text
    eubi show_pixel_meta /data/input --output_file report.txt
    </code></pre>
    </details>


    </details>


??? command "**`update_pixel_meta`**&ensp;—&ensp;Update pixel-scale and unit metadata on existing OME-Zarr files"

    **Usage:**
    ```shell
    eubi update_pixel_meta INPUT_PATH [OPTIONS]
    eubi with_config NAME update_pixel_meta INPUT_PATH [OPTIONS]
    ```

    <details>
    <summary>Required arguments</summary>

    <details>
    <summary><code>--input_path</code></summary>
    <p><strong>Type:</strong>&nbsp; `Path` or `str`</p>
    <p><strong>Default:</strong>&nbsp; *required*</p>
    <p>OME-Zarr path or directory of OME-Zarr files.</p>
    <pre><code>eubi update_pixel_meta /data/output
    </code></pre>
    </details>


    </details>

    <details>
    <summary>Optional arguments</summary>

    <details>
    <summary><code>--includes</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Comma-separated filename patterns to include.</p>
    <pre><code># Update only files whose name contains "EMPIAR"
    eubi update_pixel_meta /data/output --includes "EMPIAR"
    </code></pre>
    </details>

    <details>
    <summary><code>--excludes</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Comma-separated filename patterns to exclude.</p>
    <pre><code># Skip files whose name contains 'backup'
    eubi update_pixel_meta /data/output --excludes "backup"
    </code></pre>
    </details>

    <details>
    <summary><code>--time_scale</code></summary>
    <p><strong>Type:</strong>&nbsp; `float`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Physical pixel size along the time axis.</p>
    <pre><code>eubi update_pixel_meta /data/output --time_scale 1.0 --time_unit second
    </code></pre>
    </details>

    <details>
    <summary><code>--z_scale</code></summary>
    <p><strong>Type:</strong>&nbsp; `float`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Physical pixel size along the z axis (e.g. `0.5` for 0.5 µm z-steps).</p>
    <pre><code># Set z-step size and unit together
    eubi update_pixel_meta /data/output --z_scale 0.5 --z_unit micrometer
    </code></pre>
    </details>

    <details>
    <summary><code>--y_scale</code></summary>
    <p><strong>Type:</strong>&nbsp; `float`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Physical pixel size along the y axis.</p>
    <pre><code>eubi update_pixel_meta /data/output --y_scale 0.25 --y_unit micrometer
    </code></pre>
    </details>

    <details>
    <summary><code>--x_scale</code></summary>
    <p><strong>Type:</strong>&nbsp; `float`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Physical pixel size along the x axis.</p>
    <pre><code>eubi update_pixel_meta /data/output --x_scale 0.25 --x_unit micrometer
    </code></pre>
    </details>

    <details>
    <summary><code>--time_unit</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Physical unit for the time axis (e.g. `second`, `millisecond`).</p>
    <pre><code>eubi update_pixel_meta /data/output --time_scale 1.0 --time_unit second
    </code></pre>
    </details>

    <details>
    <summary><code>--z_unit</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Physical unit for the z axis (e.g. `micrometer`, `nanometer`).</p>
    <pre><code>eubi update_pixel_meta /data/output --z_scale 0.5 --z_unit micrometer
    </code></pre>
    </details>

    <details>
    <summary><code>--y_unit</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Physical unit for the y axis.</p>
    <pre><code>eubi update_pixel_meta /data/output --y_scale 0.25 --y_unit micrometer
    </code></pre>
    </details>

    <details>
    <summary><code>--x_unit</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Physical unit for the x axis.</p>
    <pre><code>eubi update_pixel_meta /data/output --x_scale 0.25 --x_unit micrometer
    </code></pre>
    </details>

    <details>
    <summary><code>--max_workers</code></summary>
    <p><strong>Type:</strong>&nbsp; `int` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Number of files to process in parallel.</p>
    <pre><code># Update 8 OME-Zarr stores in parallel
    eubi update_pixel_meta /data/output --max_workers 8
    </code></pre>
    </details>


    </details>


??? command "**`update_channel_meta`**&ensp;—&ensp;Update channel label, color, and intensity metadata"

    **Usage:**
    ```shell
    eubi update_channel_meta INPUT_PATH [OPTIONS]
    eubi with_config NAME update_channel_meta INPUT_PATH [OPTIONS]
    ```

    <details>
    <summary>Required arguments</summary>

    <details>
    <summary><code>--input_path</code></summary>
    <p><strong>Type:</strong>&nbsp; `Path` or `str`</p>
    <p><strong>Default:</strong>&nbsp; *required*</p>
    <p>OME-Zarr path or directory of OME-Zarr files.</p>
    <pre><code>eubi update_channel_meta /data/output
    </code></pre>
    </details>


    </details>

    <details>
    <summary>Optional arguments</summary>

    <details>
    <summary><code>--channel_labels</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; `""`</p>
    <p>Index/name pairs, e.g. `"0,DAPI;1,GFP"`.</p>
    <pre><code># Rename channels 0 and 1
    eubi update_channel_meta /data/output --channel_labels "0,DAPI;1,GFP"
    </code></pre>
    <pre><code># Three channels
    eubi update_channel_meta /data/output --channel_labels "0,DAPI;1,GFP;2,mCherry"
    </code></pre>
    </details>

    <details>
    <summary><code>--channel_colors</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; `""`</p>
    <p>Index/hex pairs, e.g. `"0,0000FF;1,00FF00"`.</p>
    <pre><code># Set display colours in hex (blue DAPI, green GFP)
    eubi update_channel_meta /data/output --channel_colors "0,0000FF;1,00FF00"
    </code></pre>
    <pre><code># Three channels — blue, green, red
    eubi update_channel_meta /data/output --channel_colors "0,0000FF;1,00FF00;2,FF0000"
    </code></pre>
    </details>

    <details>
    <summary><code>--channel_intensity_limits</code></summary>
    <p><strong>Type:</strong>&nbsp; `Literal['from_dtype', 'from_array', 'auto']`</p>
    <p><strong>Default:</strong>&nbsp; `from_dtype`</p>
    <p>`'from_dtype'` (default), `'from_array'`, or `'auto'`.</p>
    <pre><code># Compute per-channel min/max from the pixel data
    eubi update_channel_meta /data/output --channel_intensity_limits from_array
    </code></pre>
    <pre><code># Let the viewer decide (OMERO-style auto-scaling)
    eubi update_channel_meta /data/output --channel_intensity_limits auto
    </code></pre>
    </details>

    <details>
    <summary><code>--includes</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Comma-separated filename patterns to include.</p>
    <pre><code># Update only files whose name contains "EMPIAR"
    eubi update_channel_meta /data/output --includes "EMPIAR"
    </code></pre>
    </details>

    <details>
    <summary><code>--excludes</code></summary>
    <p><strong>Type:</strong>&nbsp; `str` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Comma-separated filename patterns to exclude.</p>
    <pre><code># Skip files whose name contains 'backup'
    eubi update_channel_meta /data/output --excludes "backup"
    </code></pre>
    </details>

    <details>
    <summary><code>--max_workers</code></summary>
    <p><strong>Type:</strong>&nbsp; `int` or `None`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Number of files to process in parallel.</p>
    <pre><code># Update 8 OME-Zarr stores in parallel
    eubi update_channel_meta /data/output --max_workers 8
    </code></pre>
    </details>


    </details>


