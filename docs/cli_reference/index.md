# CLI Reference

Auto-generated from `EuBIBridge` methods and Pydantic config models.
Types, defaults, and valid ranges are extracted directly from the source —
click any command card to expand its full parameter reference.

!!! tip "Named-config chaining"
    Prefix any command with `with_config NAME` to use a saved config profile:
    ```shell
    eubi with_config hpc to_zarr /data/input /data/output --zarr_format 3
    ```
    See [CLI Usage Guide](../cli.md) for the full workflow.

| Section | Commands |
|---------|----------|
| [Conversion](conversion.md) | `to_zarr`, `validate_aggregative` |
| [Metadata](metadata.md) | `show_pixel_meta`, `update_pixel_meta`, `update_channel_meta` |
| [Configuration](configuration.md) | `configure_cluster`, `configure_conversion`, `configure_downscale`, `configure_readers`, `configure_concatenation` |
| [Named Configs](named_configs.md) | `with_config`, `save_as`, `update_config`, `list_configs`, `show_configs`, `delete_config` |
| [Display & Info](display_info.md) | `show_config`, `show_root_defaults`, `version` |
| [Reset](reset.md) | `reset_config` |
