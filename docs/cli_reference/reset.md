# Reset

Auto-generated from `EuBIBridge` methods and Pydantic config models.
Types, defaults, and valid ranges are extracted directly from the source —
click any command card to expand its full parameter reference.

??? command "**`reset_config`**&ensp;—&ensp;Reset all parameters in the active config to installation defaults"

    **Usage:**
    ```shell
    eubi reset_config [OPTIONS]
    ```

    Overwrites the active `.eubi_config.json` with the bundled root defaults.  Named configs (`hpc.json`, etc.) are not affected. Use `show_root_defaults` to inspect the values that will be written.


