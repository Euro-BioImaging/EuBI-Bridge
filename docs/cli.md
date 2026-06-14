
# CLI Usage

After installing **EuBI-Bridge**, the CLI command `eubi` becomes globally available.

---

## Quick Start

### Unary vs aggregative conversion

**Unary conversion:** conversion of each input file to a single output OME-Zarr container.

**Aggregative conversion:** conversion that concatenates multiple input files along user-specified dimensions. 
 

Below are examples for both of these conversion modes:

### Examples

**Simple unary conversion:**

Convert each file in `input_dir` into an OME-Zarr container, saving the result in `output_dir`:

```bash
eubi to_zarr /path/to/input_dir /path/to/output_dir
```

Convert to OME-Zarr **version 0.5** (backed by Zarr v3, which enables sharding):

```bash
eubi to_zarr /path/to/input_dir /path/to/output_dir --ome_zarr_version 0.5
```

The default is OME-Zarr **0.4** (Zarr v2). Pass `--ome_zarr_version 0.4` or `0.5` to choose
explicitly. (The legacy `--zarr_format 2/3` flag still works but is deprecated in favour of
`--ome_zarr_version`.)

**Excluding files:**

Exclude files with `thumbs` in the filename:

```bash
eubi to_zarr /path/to/input_dir /path/to/output_dir --excludes 'thumbs'
```

Note that this option is especially useful to filter out non-image files that may be present in the input directory.

**Wildcard filtering:**

Convert only `tiff` files using wildcards:

```bash
eubi to_zarr "/path/to/input_dir/*tiff" /path/to/output_dir
```

**Aggregative conversion:**

Perform an aggregative conversion that concatenates input files along the z axis:

```bash
eubi to_zarr /path/to/input_dir /path/to/output_dir --z_tag slice_ --concatenation_axes z
```

Note that the pattern corresponding to the z axis is provided to the command via the `--z_tag` and the concatenation is activated by supplying `--concatenation_axes`. 

> ℹ️ To better understand aggregative conversion, see the [conversion tutorial](conversion_tutorial.md#tutorial).


---


## Main Commands

--- 
### `eubi to_zarr`

Performs data conversion from supported input formats (including most BioFormats-compatible formats) to OME-Zarr. 
It supports both **unary** and **aggregative** conversion modes, with options for filtering, metadata specification, downscaling, and distributed processing.


#### Non-configurable Parameters

These must be provided directly via the CLI:

| Argument               | Type                        | Description                     |
|------------------------|-----------------------------|---------------------------------|
| `input_path`           | `str` or `Path` (mandatory) | Path to input file or folder    |
| `output_path`          | `str` or `Path` (mandatory) | Path to output folder           |
| `--includes`           | `str`                       | Include filter for filenames    |
| `--excludes`           | `str`                       | Exclude filter for filenames    |
| `--scene_index`        | `int` or `tuple`            | Scene(s) to be extracted        |
| `--time_tag`           | `str` or `tuple`            | Time dimension tag              |
| `--channel_tag`        | `str` or `tuple`            | Channel dimension tag           |
| `--z_tag`              | `str` or `tuple`            | Z-dimension tag                 |
| `--y_tag`              | `str` or `tuple`            | Y-dimension tag                 |
| `--x_tag`              | `str` or `tuple`            | X-dimension tag                 |
| `--concatenation_axes` | `int`, `tuple`, or `str`    | Axes for concatenating datasets |
| `--time_scale`         | `int`, `float`              | Temporal increment              |
| `--z_scale`            | `int`, `float`              | Z spatial increment             |
| `--y_scale`            | `int`, `float`              | Y spatial increment             |
| `--x_scale`            | `int`, `float`              | X spatial increment             |
| `--time_unit`          | `str`                       | Temporal unit                   |
| `--z_unit`             | `str`                       | Z spatial unit                  |
| `--y_unit`             | `str`                       | Y spatial unit                  |
| `--x_unit`             | `str`                       | X spatial unit                  |

#### Examples

**Override metadata for unary conversion:**

```bash
eubi to_zarr /path/to/input_dir /path/to/output_dir --z_scale 2.5 --z_unit micrometer --time_scale 1.5 --time_unit second
```

**Perform aggregative conversion along the time and channel axes:**

```bash
eubi to_zarr /path/to/input_dir /path/to/output_dir --time_tag T --channel_tag Channel --concatenation_axes tc
```

> ℹ️ Tag arguments corresponding to the axes `tczyx` are: `--time_tag`, `--channel_tag`, `--z_tag` `--y_tag`, `--x_tag`, respectively.

> ℹ️ To better understand aggregative conversion, see the [conversion tutorial](conversion_tutorial.md#tutorial).


### Configurable Parameters

These are stored in the configuration file but can also be supplied directly to the command. The values supplied to the command will override the values from the configuration file. 

#### Cluster Parameters

| Argument               | Type     | Description                                  |
|------------------------|----------|----------------------------------------------|
| `--max_concurrency`       | `str`    | Maximum concurrent writes **per OME-Zarr group** |
| `--max_retries`             | `int`    | Number of retries attempted for certain types of errors    |
| `--max_workers`     | `int`   | Maximum number of processors for **input collection** |
| `--memory_per_worker`  | `int`   | Memory allocated per worker |
| `--on_local_cluster`           | `bool`   | Use local dask cluster for parallelisation |
| `--on_slurm`           | `bool`   | Enable SLURM-based execution                 |
| `--queue_size`           | `int`   | Queue size for the concurrent writes |
| `--region_size_mb`           | `int`   | Megabytes to calculate unit write-region |
| `--tensorstore_data_copy_concurrency`  | `int`   | Tensorstore context parameter |
| `--use_threading`           | `bool`   | Use multi-threading rather than multi-processing |


#### Readers Parameters

| Parameter            | Type   | Description                        |
|----------------------|--------|------------------------------------|
| `as_mosaic`          | `bool` | Whether to stitch the mosaic tiles |
| `view_index`         | `int`  | Index for view selection           |
| `phase_index`        | `int`  | Index for phase selection          |
| `illumination_index` | `int`  | Index for illumination selection   |
| `scene_index`        | `int`  | Index for scene selection          |
| `rotation_index`     | `int`  | Index for rotation selection       |
| `mosaic_tile_index`  | `int`  | Index for mosaic tile selection    |
| `sample_index`       | `int`  | Index for sample selection         |


#### Conversion Parameters

| Parameter              | Type   | Description                                              |
|------------------------|--------|----------------------------------------------------------|
| `--ome_zarr_version`   | `str`  | OME-Zarr (NGFF) version to write: `0.4` (Zarr v2) or `0.5` (Zarr v3). Preferred control. |
| `--zarr_format`        | `int`  | **Deprecated** — Zarr container version (`2`=OME-Zarr 0.4, `3`=OME-Zarr 0.5). Use `--ome_zarr_version` instead. |
| `--compressor`         | `str`  | Compression algorithm                                    |
| `--compressor_params`  | `dict` | Compressor parameters                                    |
| `--time_chunk`         | `int`  | Output Zarr chunk size in the time dimension             |
| `--channel_chunk`      | `int`  | Output Zarr chunk size in the channel dimension          |
| `--z_chunk`            | `int`  | Output Zarr chunk size in the z dimension                |
| `--y_chunk`            | `int`  | Output Zarr chunk size in the y dimension                |
| `--x_chunk`            | `int`  | Output Zarr chunk size in the x dimension                |
| `--time_shard_coef`    | `int`  | Sharding coefficient for the time dimension              |
| `--channel_shard_coef` | `int`  | Sharding coefficient for the channel dimension           |
| `--z_shard_coef`       | `int`  | Sharding coefficient for the z dimension                 |
| `--y_shard_coef`       | `int`  | Sharding coefficient for the y dimension                 |
| `--x_shard_coef`       | `int`  | Sharding coefficient for the x dimension                 |
| `--time_range`         | `int`  | Range of pixels to crop in the time dimension            |
| `--channel_range`      | `int`  | Range of pixels to crop in the channel dimension         |
| `--z_range`            | `int`  | Range of pixels to crop in the z dimension               |
| `--y_range`            | `int`  | Range of pixels to crop in the y dimension               |
| `--x_range`            | `int`  | Range of pixels to crop in the x dimension               |
| `--squeeze`            | `bool` | Drop the singlet dimensions from the output array        |
| `--overwrite`          | `bool` | Overwrite existing Zarr data                             |
| `--dtype`              | `str`  | `auto` keeps the source dtype; pass a NumPy dtype (e.g. `uint8`) to cast on write |
| `--metadata_reader`    | `str`  | Metadata extraction backend (`bfio` or `bioformats`)     |
| `--save_omexml`        | `bool` | Save a companion OME-XML sidecar file                    |

#### Downscale Parameters

| Parameter                    | Type    | Description                                  |
|------------------------------|---------|----------------------------------------------|
| `--downscale_method`         | `str`   | Downscale algorithm (`simple`, `mean`, `median`, `min`, `max`, `mode`) |
| `--n_layers`                 | `int`   | Number of pyramid levels (omit for auto)     |
| `--min_dimension_size`       | `int`   | Stop downscaling once the smallest axis reaches this size |
| `--time_scale_factor`        | `int`   | Downscaling factor for the time dimension    |
| `--z_scale_factor`           | `int`   | Downscaling factor for the z dimension       |
| `--y_scale_factor`           | `int`   | Downscaling factor for the y dimension       |
| `--x_scale_factor`           | `int`   | Downscaling factor for the x dimension       |
| `--keep_existing_resolutions`| `bool`  | For inputs that already carry a pyramid (`.ims`, `.zarr`), copy existing levels instead of rebuilding |


#### Examples

**Run with 8 workers and limit memory per worker:**

```bash
eubi to_zarr /path/to/input_dir /path/to/output_dir --max_workers 8 --memory_per_worker 10GB
```

**Specify output chunk size:**

```bash
eubi to_zarr /path/to/input_dir /path/to/output_dir --z_chunk 128 --y_chunk 128 --x_chunk 128
```

**Convert to OME-Zarr 0.5 (Zarr v3) and specify the shard size:**

```bash
eubi to_zarr /path/to/input_dir /path/to/output_dir --ome_zarr_version 0.5 --y_shard_coef 4 --x_shard_coef 4 --y_chunk 128 --x_chunk 128
```

Note that this will create a zarr dataset with a chunk size of 128x128 and a shard size of 512x512 (by multiplying the chunk size by the shard coefficient). Sharding requires OME-Zarr 0.5 (Zarr v3).

**Specify downscaling layers and scale factor:**

```bash
eubi to_zarr /path/to/input_dir /path/to/output_dir --n_layers 6 --z_scale_factor 2 --y_scale_factor 3 --x_scale_factor 3
```

**Remove singlet dimensions:**

```bash
eubi to_zarr /path/to/input_dir /path/to/output_dir --squeeze True
```

Note that supplying `False` to `--squeeze` will guarantee a 5-dimensional output, creating singletons for dimensions that do not exist in the input images.

**Crop a subset of the array:**

```bash
eubi to_zarr /path/to/input_dir /path/to/output_dir --time_range 0,100 --z_range 15,125
```

This will convert a subset of the input datasets (0-100 in the time range and 15-125 in the z range) to OME-Zarr.


> ℹ️ For more examples, see the [conversion tutorial](conversion_tutorial.md#tutorial).

---

### `eubi update_pixel_meta`

Performs in-place metadata update in the OME-Zarr datasets. Currently, pixel units and pixel scales are updateable metadata elements. 

#### Parameters

Each of these parameters can be supplied to the `eubi update_pixel_meta` command:

| Argument       | Type                        | Description                                                 |
|:---------------|-----------------------------|-------------------------------------------------------------|
| `input_path`   | `str` or `Path` (mandatory) | Path to an **OME-Zarr** or a directory of **OME-Zarrs**     |
| `--includes`   | `str`                       | Filename include filter                                     |
| `--excludes`   | `str`                       | Filename exclude filter                                     |
| `--time_scale` | `int`, `float`              | Time increment                                              |
| `--z_scale`    | `int`, `float`              | Z-axis increment                                            |
| `--y_scale`    | `int`, `float`              | Y-axis increment                                            |
| `--x_scale`    | `int`, `float`              | X-axis increment                                            |
| `--time_unit`  | `str`                       | Time unit                                                   |
| `--z_unit`     | `str`                       | Z-axis unit                                                 |
| `--y_unit`     | `str`                       | Y-axis unit                                                 |
| `--x_unit`     | `str`                       | X-axis unit                                                 |


Note that, input to `update_pixel_meta` **must** be in the OME-Zarr format. Updating pixel metadata is not supported with other file formats. 

#### Example

```bash
eubi update_pixel_meta /path/to/input_dir --z_scale 5 --z_unit nanometer
```

### `eubi show_pixel_meta`

Displays basic pixel metadata for all images in the input directory. 

#### Example

```bash
eubi show_pixel_meta /path/to/input_dir
```

The command `show_pixel_meta` supports inputs with diverse file formats. Note that for pyramidal images, currently metadata is displayed only for the highest resolution layer.

---

## Configuration Commands

---

Configuration is managed through the `configure` command group. Run `eubi configure`
with no subcommand to list the available sections:

```bash
eubi configure
# COMMANDS: cluster · conversion · downscale · readers · concatenation
```

### `eubi configure cluster`

Set cluster defaults using any of the [cluster parameters](#cluster-parameters).

#### Example

```bash
eubi configure cluster --memory_per_worker 10GB
```

### `eubi configure conversion`

Set conversion defaults using any of the [conversion parameters](#conversion-parameters).

#### Example

```bash
eubi configure conversion --ome_zarr_version 0.5
```

### `eubi configure downscale`

Set downscale defaults using any of the [downscale parameters](#downscale-parameters).

#### Example

```bash
eubi configure downscale --z_scale_factor 1 --y_scale_factor 2 --x_scale_factor 2
```

### `eubi configure readers`

Set reader defaults (scene / view / illumination / mosaic selection) using any of the
[reader parameters](#readers-parameters).

### `eubi configure concatenation`

Set aggregative defaults (concatenation axes and filename tags).

---

## Reset and Inspect Configuration

<table>
  <thead>
    <tr>
      <th>Command</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code><nobr>eubi reset_config</nobr></code></td>
      <td>Reset cluster/conversion/downscale parameters to installation defaults</td>
    </tr>
    <tr>
      <td><code><nobr>eubi show_config</nobr></code></td>
      <td>Show current cluster/conversion/downscale settings</td>
    </tr>
    <tr>
      <td><code><nobr>eubi show_root_defaults</nobr></code></td>
      <td>Show installation defaults for cluster/conversion/downscale parameters</td>
    </tr>
    <tr>
      <td><code><nobr>eubi show_configs</nobr></code></td>
      <td>List all saved named configuration profiles</td>
    </tr>
  </tbody>
</table>