# EuBI-Bridge Conversion Examples

Command examples loosely organized by use case. Copy and paste these commands, adjusting input/output paths for your data.

## Table of Contents

1. [Basic Unary Conversion](#basic-unary-conversion)
2. [Filtering & Pattern Matching](#filtering--pattern-matching)
3. [Metadata Inspection & Override](#metadata-inspection--override)
4. [Aggregative Conversion](#aggregative-conversion)
5. [Advanced Aggregation](#advanced-aggregation)
6. [CSV Based Processing](#csv-batch-processing)
7. [Multi-Series Datasets](#multi-series-datasets)
8. [Zarr-to-Zarr Conversion](#zarr-to-zarr-conversion)
9. [Configuration Management](#configuration-management)

---

## Basic Unary Conversion

### Simple Conversion: Single Input → Single Output

```bash
eubi to_zarr small_dataset ~/codash_output/small_dataset_zarr
```

Verify the output:
```bash
eubi show_pixel_meta ~/codash_output/small_dataset_zarr
```

### Custom Chunk Sizes

```bash
eubi to_zarr small_dataset ~/codash_output/small_dataset_zarr_new_chunks \
  --auto_chunk False \
  --y_chunk 64 \
  --x_chunk 64
```

### Multiple Resolution Layers (Pyramid)

```bash
eubi to_zarr small_dataset ~/codash_output/small_dataset_zarr_6_layers \
  --n_layers 6
```

### NGFF v0.5 with Sharding

```bash
eubi to_zarr small_dataset ~/codash_output/small_dataset_zarr_v0.5 \
  --auto_chunk False \
  --y_chunk 64 \
  --x_chunk 64 \
  --zarr_format 3 \
  --y_shard_coef 5 \
  --x_shard_coef 5
```

---

## Filtering & Pattern Matching

### Include Only TIFF Files

```bash
eubi to_zarr small_dataset ~/codash_output/small_dataset_zarr_inc_filtered \
  --includes .tiff
```

### Exclude JPG Files

```bash
eubi to_zarr small_dataset ~/codash_output/small_dataset_zarr_ex_filtered \
  --excludes .jpg
```

### Include Multiple Formats

```bash
eubi to_zarr small_dataset ~/codash_output/small_dataset_zarr_multi_inc_filtered \
  --includes .oir,.lsm
```

### Exclude Multiple Formats

```bash
eubi to_zarr small_dataset ~/codash_output/small_dataset_zarr_multi_ex_filtered \
  --excludes .oir,.lsm
```

### Glob Pattern Matching

```bash
eubi to_zarr "small_dataset/*.tif" ~/codash_output/small_dataset_zarr_wildcards
```

---

## Metadata Inspection & Override

### Inspect Source Metadata

```bash
eubi show_pixel_meta "small_dataset/*oir"
```

### Convert with Pixel Scale & Units

```bash
eubi to_zarr \
  small_dataset/23052022_D3_0001.oir \
  ~/codash_output/small_image_with_metadata \
  --y_scale 0.5 \
  --y_unit micrometer \
  --x_scale 0.5 \
  --x_unit micrometer \
  --channel_labels "0,GFP;1,mCherry;2,BF" \
  --channel_colors "0,yellow;1,magenta;2,white"
```

### Verify Overridden Metadata

```bash
eubi show_pixel_meta "~/codash_output/small_image_with_metadata/*zarr"
```

### Update Metadata In Place
```bash
eubi update_pixel_meta \
  "~/codash_output/small_image_with_metadata/*zarr" \
  --y_scale 0.2 \
  --x_scale 0.2
```

### Verify Updated Metadata

```bash
eubi show_pixel_meta "~/codash_output/small_image_with_metadata/*zarr"
```

---

## Aggregative Conversion

Concatenate multiple input files into a single output file along specified dimensions.

### Concatenate Along All Specified Axes

Combine files into channels and time points:

```bash
eubi to_zarr \
  multidim_dataset \
  ~/codash_output/multidim_dataset_concat_zarr \
  --channel_tag growing,moving,noise \
  --time_tag step \
  --concatenation_axes ct
```

### With Nested Directory Structure

```bash
eubi to_zarr \
  multidim_dataset_nested \
  ~/codash_output/multidim_dataset_nested_concat_zarr \
  --channel_tag growing,moving,noise \
  --time_tag step \
  --concatenation_axes ct
```

### Specify Channel Order

```bash
eubi to_zarr \
  multidim_dataset_nested \
  ~/codash_output/multidim_dataset_nested_concat_zarr \
  --channel_tag noise,growing,moving \
  --time_tag step \
  --concatenation_axes ct \
  --override_channel_names
```

---

## Advanced Aggregation

### Concatenate Along Time Only

```bash
eubi to_zarr \
  multidim_dataset \
  ~/codash_output/multidim_dataset_concat_along_time_zarr \
  --channel_tag moving,noise,growing \
  --time_tag step \
  --concatenation_axes t \
  --override_channel_names
```

### Aggregation with Pattern Filter

```bash
eubi to_zarr \
  "medium_dataset/PK2*.ome.tiff" \
  ~/codash_output/multidim_dataset_concat_filtered_zarr \
  --channel_tag ch \
  --concatenation_axes c \
  --skip_dask
```

### Aggregation with Full Metadata Override

```bash
eubi to_zarr \
  multidim_dataset \
  ~/codash_output/multidim_dataset_concat_zarr_with_metadata \
  --channel_tag moving,noise,growing \
  --time_tag step \
  --concatenation_axes ct \
  --time_scale 18 \
  --time_unit millisecond \
  --y_scale 0.5 \
  --y_unit micrometer \
  --x_scale 0.5 \
  --x_unit micrometer \
  --override_channel_names \
  --channel_colors "0,red;1,green;2,white"
```

### Aggregation with Time Range Limit

```bash
eubi to_zarr \
  multidim_dataset \
  ~/codash_output/multidim_dataset_concat_zarr_with_metadata \
  --channel_tag moving,noise,growing \
  --time_tag step \
  --concatenation_axes ct \
  --time_scale 18 \
  --time_unit millisecond \
  --y_scale 0.5 \
  --y_unit micrometer \
  --x_scale 0.5 \
  --x_unit micrometer \
  --override_channel_names \
  --channel_colors "0,red;1,green;2,white" \
  --time_range 0,5
```

---

## CSV Based Processing

### Create a CSV File

Find or create `selection.csv`. Its convent should be similar to:

```csv
input_path,output_path,y_scale,x_scale,y_unit,x_unit,channel_labels
small_dataset/image1.tif,~/output/image1_zarr,0.5,0.5,micrometer,micrometer,"0,Red;1,Green"
small_dataset/image2.czi,~/output/image2_zarr,1.0,1.0,micrometer,micrometer,"0,DAPI;1,GFP"
```

### Convert All Files in CSV

```bash
eubi to_zarr selection.csv
```

### Verify Results

```bash
eubi show_pixel_meta ~/codash_output/from_csv_zarr
```

---

## Multi-Series Datasets

### Convert All Series from CZI Files

```bash
eubi to_zarr "medium_dataset/*.czi" \
  ~/codash_output/multiseries_dataset \
  --scene_index all
```

### Convert Selected Series Only

```bash
eubi to_zarr medium_dataset/19_07_19_Lennard.lif \
  ~/codash_output/multiseries_dataset \
  --scene_index 2,4,6
```

---

## Zarr-to-Zarr Conversion

Re-chunk and reformat existing OME-Zarr files:

```bash
eubi to_zarr \
  ~/codash_output/multidim_dataset_concat_filtered_zarr/PK2_ATH_5to20_20240705_MID_01_ch_cset.ome.zarr \
  ~/codash_output/multidim_dataset_concat_filtered_zarr_newchunks_newshards_v0.5 \
  --auto_chunk False \
  --zarr_format 3 \
  --y_shard_coef 5 \
  --x_shard_coef 5
```

---

## Configuration Management

### Display Current Configuration

```bash
eubi show_config
```

### Modify Cluster Settings

```bash
eubi configure_cluster --max_workers 3
eubi show_config
```

### Reset to Installation Defaults

```bash
eubi reset_config
eubi show_config
```

---



## Quick Reference: Parameter Combinations

### Standard Unary + Metadata

```bash
eubi to_zarr INPUT_DIR OUTPUT_PATH \
  --y_scale 0.5 \
  --x_scale 0.5 \
  --y_unit micrometer \
  --x_unit micrometer
```

### Optimized for Cloud Storage

```bash
eubi to_zarr INPUT_DIR OUTPUT_PATH \
  --zarr_format 3 \
  --y_shard_coef 3 \
  --x_shard_coef 3
```

### Small Datasets (Memory Efficient)

```bash
eubi to_zarr INPUT_DIR OUTPUT_PATH \
  --auto_chunk False \
  --y_chunk 32 \
  --x_chunk 32 \
  --z_chunk 16
```

### Large Datasets (Maximum Performance)

```bash
eubi to_zarr INPUT_DIR OUTPUT_PATH \
  --n_layers 5 \
  --auto_chunk False \
  --y_chunk 128 \
  --x_chunk 128 \
  --z_chunk 64 \
  --zarr_format 3
```

### Multi-File Aggregation (Complete Example)

```bash
eubi to_zarr \
  dataset_directory \
  ~/output/combined_dataset \
  --channel_tag ch0,ch1,ch2 \
  --time_tag t \
  --z_tag slice \
  --concatenation_axes ctz \
  --y_scale 0.5 \
  --x_scale 0.5 \
  --override_channel_names \
  --channel_labels "0,DAPI;1,GFP;2,RFP" \
  --channel_colors "0,0000FF;1,00FF00;2,FF0000"
```

