# Named Configs

Auto-generated from `EuBIBridge` methods and Pydantic config models.
Types, defaults, and valid ranges are extracted directly from the source —
click any command card to expand its full parameter reference.

??? command "**`with_config`**&ensp;—&ensp;Return a new EuBIBridge backed by the named config file"

    **Usage:**
    ```shell
    eubi with_config NAME [OPTIONS]
    ```

    All parameters in the named config are used as defaults; any arguments passed directly to ``to_zarr()`` (or other commands) still take priority.  Designed for Fire chaining:: eubi with_config hpc to_zarr input/ output/

    <details>
    <summary>Required arguments</summary>

    <details>
    <summary><code>--name</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; *required*</p>
    <p>—</p>
    <pre><code># Use the 'hpc' profile for one conversion run
    eubi with_config hpc to_zarr /data/input /data/output
    </code></pre>
    <pre><code># Combine a named config with an inline override
    eubi with_config hpc to_zarr /data/input /data/output --zarr_format 3
    </code></pre>
    </details>


    </details>


??? command "**`save_as`**&ensp;—&ensp;Save the current config as a named config file"

    **Usage:**
    ```shell
    eubi save_as NAME [OPTIONS]
    ```

    <details>
    <summary>Required arguments</summary>

    <details>
    <summary><code>--name</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; *required*</p>
    <p>—</p>
    <pre><code># Save the current configuration as a profile named 'hpc'
    eubi save_as hpc
    </code></pre>
    </details>


    </details>


??? command "**`update_config`**&ensp;—&ensp;Copy the current config into a named config or an explicit file path"

    **Usage:**
    ```shell
    eubi update_config NAME_OR_PATH [OPTIONS]
    ```

    Typical use: propagate default-config edits to a named config:: eubi configure_cluster --max_workers 64   # edits .eubi_config.json eubi update_config hpc                     # copies it to hpc.json

    <details>
    <summary>Required arguments</summary>

    <details>
    <summary><code>--name_or_path</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; *required*</p>
    <p>—</p>
    <pre><code># Propagate edits from the active config into the 'hpc' profile
    eubi update_config hpc
    </code></pre>
    <pre><code># Write to an explicit file path instead of a named profile
    eubi update_config /data/configs/hpc.json
    </code></pre>
    </details>


    </details>

    <details>
    <summary>Optional arguments</summary>

    <details>
    <summary><code>--create</code></summary>
    <p><strong>Type:</strong>&nbsp; boolean flag</p>
    <p><strong>Default:</strong>&nbsp; `False`</p>
    <p><strong>Valid values:</strong>&nbsp; `--create` to enable &nbsp;·&nbsp; `--create False` to disable</p>
    <p>—</p>
    <pre><code># Create the target profile if it does not exist yet
    eubi update_config hpc --create
    </code></pre>
    </details>


    </details>


??? command "**`list_configs`**&ensp;—&ensp;Return ``{name: path}`` for all named configs in the config directory"

    **Usage:**
    ```shell
    eubi list_configs [OPTIONS]
    ```


??? command "**`show_configs`**&ensp;—&ensp;List all named config files found in the config directory"

    **Usage:**
    ```shell
    eubi show_configs [OPTIONS]
    ```

    Prints a table of ``{name: path}`` entries.  Use ``with_config NAME`` to activate one of them for a single command, or ``save_as`` / ``update_config`` to manage them.


??? command "**`delete_config`**&ensp;—&ensp;Permanently delete a named config file from the config directory"

    **Usage:**
    ```shell
    eubi delete_config NAME [OPTIONS]
    ```

    The active (default) config is not affected.  Only named configs created with ``save_as`` or ``update_config --create`` can be deleted this way.

    <details>
    <summary>Required arguments</summary>

    <details>
    <summary><code>--name</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; *required*</p>
    <p>Short name of the config to delete (without the ``.json`` extension), e.g. ``hpc``.</p>
    <pre><code># Permanently remove the 'hpc' profile
    eubi delete_config hpc
    </code></pre>
    </details>


    </details>


