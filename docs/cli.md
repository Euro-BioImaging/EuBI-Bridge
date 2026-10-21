
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

Performs data conversion from supported input formats to OME-Zarr. Pixels and metadata are
read with micro-reader by default; a file it does not read falls back to the standard
readers (Bio-Formats among them). It supports both **unary** and **aggregative** conversion
modes, with options for filtering, metadata specification, downscaling, and distributed processing.


#### Command Arguments

These are given to the command only:

| Argument               | Type                        | Description                     |
|------------------------|-----------------------------|---------------------------------|
| `input_path`           | `str` or `Path` (mandatory) | Path to an input file or folder, or a conversion table (`.csv`, `.tsv`, `.txt`, `.xlsx`) |
| `output_path`          | `str` or `Path`             | Path to the output folder. Optional when `input_path` is a conversion table whose rows carry their own `output_path` |
| `--includes`           | `str`                       | Include filter for filenames    |
| `--excludes`           | `str`                       | Exclude filter for filenames    |

Everything else (series selection, tags, scales and units, ...) is a setting: it can be
passed to the command too, and is described under [Configurable Parameters](#configurable-parameters).

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
| `--max_processes`     | `int`   | Images converted at once, each in its own worker process (formerly `--max_workers`) |
| `--max_threads`       | `int`    | Threads per conversion, each reading and writing one region at a time; scenes converted at once share them (formerly `--max_concurrency`) |
| `--max_retries`             | `int`    | Number of retries attempted for certain types of errors    |
| `--memory_budget`  | `str`   | The memory the whole run may use on this machine, e.g. `16GB`, or `auto` (default: half the free RAM) |
| `--max_concurrent_scenes` | `int` | Scenes of one file converted at once (default 1); they share `max_threads` |
| `--decode_threads`     | `int`    | micro-reader's decode threads per worker process (formerly `--micro_read_concurrency`) |
| `--decode_memory`      | `str`    | micro-reader's decoding memory per process, e.g. `1GB`, or `auto` (default) |
| `--bioformats_threads` | `int`    | Threads for parallel Bio-Formats tile reads (formerly `--bf_read_concurrency`) |
| `--jvm_memory`         | `str`    | Maximum JVM heap for Bio-Formats, e.g. `8GB` (default `1g`) |
| `--on_slurm`           | `bool`   | Enable SLURM-based execution                 |
| `--region_size_mb`           | `float` or `auto` | Size of one read region in MB; `auto` (default) sizes it from the memory budget (8 to 256 MB) |
| `--use_threading`           | `bool`   | Use multi-threading rather than multi-processing |


#### Readers Parameters

| Parameter            | Type   | Description                        |
|----------------------|--------|------------------------------------|
| `pixel_reader`       | `str`  | Where the pixels come from: `micro` (default; a file micro-reader does not read falls back to the standard reader, with a warning) or `standard` |
| `force_bioformats`   | `bool` | Use the Java Bio-Formats reader even for formats read natively; a fallback when a native read fails |
| `as_mosaic`          | `bool` | Whether to stitch the mosaic tiles |
| `series_index`       | `int`, `all` or a list | Series to convert (some formats call them scenes), e.g. `0`, `all` or `0,2,4`; each becomes its own output, still named `_scene{N}`. Formerly `scene_index`, which still works, with a warning |
| `view_index`         | `int`, `all` or a list | View(s) to convert |
| `illumination_index` | `int`, `all` or a list | Illumination(s) to convert |
| `phase_index`        | `int`  | Index for phase selection          |
| `rotation_index`     | `int`  | Index for rotation selection       |
| `mosaic_tile_index`  | `int`, `all` or a list | Mosaic tile(s) to convert when not stitching; each becomes its own output |
| `sample_index`       | `int`  | Index for sample selection         |
| `concat_views`       | `bool` | Stack the views along the channel axis into one output instead of one output per view |
| `concat_illuminations` | `bool` | Stack the illuminations along the channel axis into one output instead of one output per illumination |


#### Conversion Parameters

| Parameter              | Type   | Description                                              |
|------------------------|--------|----------------------------------------------------------|
| `--ome_zarr_version`   | `str`  | OME-Zarr (NGFF) version to write: `0.4` (Zarr v2) or `0.5` (Zarr v3). Preferred control. |
| `--zarr_format`        | `int`  | **Deprecated** — Zarr container version (`2`=OME-Zarr 0.4, `3`=OME-Zarr 0.5). Use `--ome_zarr_version` instead. |
| `--compressor`         | `str`  | Compression algorithm. Avoid blosc's `snappy` codec: few OME-Zarr readers can read it, and eubi-bridge warns about it (or refuses it, with dyna-zarr 0.1.0) |
| `--compressor_params`  | `dict` | Compressor parameters                                    |
| `--auto_chunk`         | `bool` | Choose the chunk shape automatically, near `--target_chunk_mb` (default `True`); `False` uses the `*_chunk` values below |
| `--target_chunk_mb`    | `float` | Target uncompressed chunk size in MB with `--auto_chunk` (default 1) |
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
| `--save_omexml`        | `bool` | Save a companion OME-XML sidecar file                    |
| `--export_acquisition_metadata` | `bool` | Write acquisition metadata NGFF has no field for (view / illumination indices, objective NA, wavelengths) into the `eubi_bridge` attributes; unset = on when one acquisition is split into several outputs |
| `--io_backend`         | `str`  | The bytes backend ome_zarr_pyramid writes with: `tensorstore` (default) or `zarrista` (`pip install "dyna-zarr[zarrista]"`) |

#### Metadata Parameters

| Parameter                    | Type    | Description                                  |
|------------------------------|---------|----------------------------------------------|
| `--metadata_reader`          | `str`   | Where the metadata comes from: `micro` (default; a file micro-reader does not read falls back to the standard reader), `bfio` or `bioio` |
| `--channel_labels`           | `str`   | Per-channel name overrides, e.g. `"0,DAPI;1,GFP"` |
| `--channel_colors`           | `str`   | Per-channel colour overrides, e.g. `"0,FF0000;1,00FF00"`; channels left out keep their source colour |
| `--channel_intensity_limits` | `str`   | Display-window limits: `from_dtype` (default), `from_array` (per-channel min/max of the data) or `auto` (0.1–99.9 percentiles of the smallest resolution level; the dtype's range when that level is over 64 MB) |
| `--override_channel_names`   | `bool`  | Name the output channels after the `channel_tag` values (aggregative conversions) |
| `--time_scale`, `--z_scale`, `--y_scale`, `--x_scale` | `float` | Physical size of a time step / pixel; unset keeps the source value |
| `--time_unit`, `--z_unit`, `--y_unit`, `--x_unit` | `str` | Units for those scales, e.g. `second`, `micrometer` |

#### Concatenation Parameters

| Parameter              | Type                     | Description                     |
|------------------------|--------------------------|---------------------------------|
| `--concatenation_axes` | `str`                    | Axes to concatenate along, e.g. `z` or `tc`; enables aggregative conversion |
| `--time_tag`, `--channel_tag`, `--z_tag`, `--y_tag`, `--x_tag` | `str` or a list | File-name patterns that give each file's place along that axis |
| `--aggregative_group`  | `str`                    | Name prefixed to the concatenated output; in a conversion table, rows sharing it are concatenated together |

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

**Run 8 conversions at once within 10 GB of memory:**

```bash
eubi to_zarr /path/to/input_dir /path/to/output_dir --max_processes 8 --memory_budget 10GB
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

Performs in-place metadata update in the OME-Zarr datasets: pixel units and pixel scales.
Channel names and colours are updated with `eubi update_channel_meta` (see the
[CLI reference](cli_reference/metadata.md)).

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
# COMMANDS: cluster · conversion · downscale · readers · metadata · concatenation
```

### `eubi configure cluster`

Set cluster defaults using any of the [cluster parameters](#cluster-parameters).

#### Example

```bash
eubi configure cluster --memory_budget 10GB
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

Set reader defaults (series / view / illumination / mosaic selection) using any of the
[reader parameters](#readers-parameters).

### `eubi configure metadata`

Set metadata defaults (metadata reader, channel names / colours / display limits, pixel
scales and units) using any of the [metadata parameters](#metadata-parameters).

#### Example

```bash
eubi configure metadata --channel_intensity_limits from_array
```

### `eubi configure concatenation`

Set aggregative defaults using any of the [concatenation parameters](#concatenation-parameters).

#### Example

```bash
eubi configure concatenation --concatenation_axes c --channel_tag raw,mask
```

Named configuration profiles (`save_as`, `with_config`, `list_configs`, ...),
`validate_aggregative` and `version` are described in the [CLI reference](cli_reference/index.md).

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
      <td>Reset all settings (cluster, readers, conversion, downscale, metadata, concatenation) to installation defaults</td>
    </tr>
    <tr>
      <td><code><nobr>eubi show_config</nobr></code></td>
      <td>Show the current settings of all six sections</td>
    </tr>
    <tr>
      <td><code><nobr>eubi show_root_defaults</nobr></code></td>
      <td>Show the installation defaults of all six sections</td>
    </tr>
    <tr>
      <td><code><nobr>eubi show_configs</nobr></code></td>
      <td>List all saved named configuration profiles</td>
    </tr>
  </tbody>
</table>