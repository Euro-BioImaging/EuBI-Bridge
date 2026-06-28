# Display & Info

Click any command card below to expand its full parameter reference.

!!! attention "Info"
    Note that all cards below are auto-generated from `EuBIBridge` methods and Pydantic config models. Types, defaults, and valid ranges are extracted directly from the source.

??? command "**`show_config`**&ensp;—&ensp;Display the active configuration, or a specific named config"

    **Usage:**
    ```shell
    eubi show_config [OPTIONS]
    ```

    With no arguments, prints the current `.eubi_config.json`. Pass a name to inspect a saved config without activating it.

    <details>
    <summary>Optional arguments</summary>

    <details>
    <summary><code>--name</code></summary>
    <p><strong>Type:</strong>&nbsp; `str`</p>
    <p><strong>Default:</strong>&nbsp; —</p>
    <p>Named config to display (e.g. `hpc`).  If omitted, the active (default) config is shown.</p>
    <pre><code># Inspect the 'hpc' profile without activating it
    eubi show_config --name hpc
    </code></pre>
    </details>


    </details>


??? command "**`show_root_defaults`**&ensp;—&ensp;Display the installation defaults for all configuration parameters"

    **Usage:**
    ```shell
    eubi show_root_defaults [OPTIONS]
    ```

    These are the values that `reset_config` will restore.  Useful as a reference when tuning cluster or conversion settings.


??? command "**`version`**&ensp;—&ensp;Display the installed EuBI-Bridge version"

    **Usage:**
    ```shell
    eubi version [OPTIONS]
    ```


