"""
EuBI-Bridge Comprehensive Test Suite

A complete test suite for the eubi-bridge API functionality covering most major parameters, conversion modes, and edge cases.

## Scope & Current Coverage

**Current Test Coverage:**
- **Input Formats:** TIFF and OME-TIFF only
- **API Functionality:** Core conversion, metadata handling, and parameter interactions
- **Test Count:** 58 passing tests across 4 test modules

**Note:** This test suite currently validates eubi-bridge functionality using TIFF inputs.
Future expansion will include testing with proprietary microscopy formats (see Future Testing below).

## Test Organization

### 1. Fixture Infrastructure (conftest_fixtures.py)
Pytest fixtures generating test data on-the-fly using tifffile and numpy.

**ImageJ TIFF Fixtures:**
- `imagej_tiff_zyx` — 3D Z-stack (uint8)
- `imagej_tiff_zyx_uint16` — 3D Z-stack (uint16)
- `imagej_tiff_czyx` — Multi-channel 4D image
- `imagej_tiff_tczyx` — Time-series 5D image

**OME-TIFF Fixtures (with embedded channel metadata):**
- `ome_tiff_3ch` — 3 channels with names and colors
- `ome_tiff_2ch_categorical` — Categorical channel tags (gfp, mcherry)
- `ome_tiff_single_ch` — Single channel with metadata
- `ome_tiff_tczyx` — 5D with channel metadata

**Aggregative Test Sets:**
- `aggregative_z_concat_files` — Multiple Z-slices per timepoint
- `aggregative_channel_categorical_files` — Files tagged with channel names (gfp, mcherry)
- `aggregative_channel_numerical_files` — Files with numerical indices (channel1, 2, 3)
- `aggregative_zc_concat_files` — Multi-axis Z and C concatenation
- `aggregative_ome_channel_merge_files` — OME-TIFF time series for metadata merging

### 2. Validation Utilities (validation_utils.py)
Shared helper functions for validating zarr output:

- `validate_zarr_exists()` — Check output is valid zarr group
- `validate_zarr_format()` — Check zarr v2 vs v3
- `validate_base_array_shape()` — Check output array dimensions
- `validate_dtype()` — Verify data type preservation
- `validate_chunk_size()` — Check chunking configuration
- `validate_multiscale_metadata()` — Validate NGFF pyramid metadata
- `validate_channel_metadata()` — Check OMERO channel info
- `validate_pixel_scales()` — Verify pixel scale values
- `validate_pixel_units()` — Check unit strings
- `validate_downscaling_pyramid()` — Verify pyramid layers
- `validate_compression()` — Check compressor codec
- `compare_pixel_data()` — Compare arrays with tolerance

### 3. Test Modules

#### test_unary_conversions.py (~12 tests)
Single-file conversion tests:

**TestZarrFormat**
- `test_zarr_format_v2` — Verify zarr v2 output structure
- `test_zarr_format_v3` — Verify zarr v3 output structure

**TestChunking**
- `test_auto_chunk` — Automatic chunk size computation
- `test_manual_chunks` — Manual chunk specification
- (Tests for v3 sharding in future)

**TestDownscaling**
- `test_default_downscaling` — Default pyramid creation
- `test_no_downscaling` — Single layer (n_layers=1)
- `test_custom_n_layers` — Multiple downscaling layers

**TestPixelMetadata**
- `test_default_pixel_scales` — Default scale values
- `test_custom_pixel_scales` — Custom x/y/z scales
- `test_custom_units` — Custom unit strings

**TestDataType**
- `test_preserve_uint8` — uint8 preservation
- `test_preserve_uint16` — uint16 preservation

**TestCompression**
- `test_default_compressor` — Default compression
- `test_custom_compressor` — Custom compressor selection
- `test_compressor_params` — Compressor parameter passing

**TestSqueeze**
- `test_squeeze_enabled_by_default` — Verify squeeze=True default
- `test_squeeze_disabled` — Disable squeezing

**TestOverwrite**
- `test_overwrite_existing` — Overwrite with flag
- `test_no_overwrite_fails` — Fail without overwrite

#### test_ome_metadata_parsing.py (~8 tests)
OME-TIFF channel metadata handling:

**TestOMEMetadataReading**
- `test_read_ome_channel_names` — Read channel names
- `test_read_ome_channel_colors` — Read channel colors
- `test_read_categorical_channel_names` — Read categorical tags
- `test_single_channel_metadata` — Single channel metadata

**TestChannelMetadataInAggregative**
- `test_merge_channels_from_multiple_omes` — Merge OME metadata
- `test_override_channel_names_with_tag` — Override with categorical tags
- `test_override_with_numerical_channel_tags` — Override with numerical indices

**TestChannelMetadataPreservation**
- `test_5d_ome_metadata_preserved` — Preserve in 5D
- `test_metadata_survives_downscaling` — Metadata through pyramid
- `test_metadata_with_custom_scales` — Metadata with scale changes

#### test_aggregative_conversions.py (~11 tests)
Multi-file aggregative conversions:

**TestZConcatenation**
- `test_z_concat_basic` — Z-axis file concatenation

**TestTConcatenation**
- `test_t_concat_from_z_files` — Time-point concatenation

**TestChannelConcatenationCategorical**
- `test_channel_concat_categorical` — Categorical channel tags
- `test_channel_concat_categorical_with_override` — Override names

**TestChannelConcatenationNumerical**
- `test_channel_concat_numerical` — Numerical channel indices
- `test_channel_concat_numerical_with_override` — Override numerical

**TestMultiAxisConcatenation**
- `test_zc_concat` — Z and C simultaneous concatenation

**TestTagFiltering**
- `test_tag_filtering_with_exclude` — Include/exclude patterns
- `test_partial_tag_match` — Substring matching

**TestNoMatchingTags**
- `test_tag_no_matches_fails` — Error on no matches

**TestOMEMetadataMerge**
- `test_ome_merge_channels_from_multiple_files` — OME merge
- `test_ome_metadata_preserved_with_time_concat` — Preserve on T-concat

#### test_parameter_interactions.py (~13 tests)
Parameter combinations, edge cases, and constraints:

**TestSmallArrays**
- `test_tiny_2d_array` — Very small 2D arrays
- `test_minimum_chunk_size` — Chunking small arrays

**TestLargeNumericValues**
- `test_uint16_full_range` — Full uint16 range
- `test_float32_preservation` — float32 type preservation

**TestConflictingChunkParameters**
- `test_chunk_larger_than_array` — Handle oversized chunks
- `test_auto_chunk_overrides_manual` — Parameter precedence

**TestDownscaleSmallArrays**
- `test_downscale_small_array` — Downscale small arrays
- `test_min_dimension_size_constraint` — Respect min_dimension_size

**TestConcurrencyParameters**
- `test_single_worker` — Single-worker sequential processing
- `test_multiple_workers` — Multi-worker parallelism
- `test_max_concurrency` — Concurrent write limits

**TestMemoryConstraints**
- `test_region_size_limit` — Memory region size control
- `test_skip_dask` — Direct array processing

**TestAxisOrdering**
- `test_5d_to_4d_via_squeeze` — Dimension reduction via squeeze

**TestParameterCombinations**
- `test_combined_zarr_v3_with_sharding` — v3 + sharding
- `test_combined_downscale_with_custom_scales` — Pyramid + scales
- `test_combined_compression_with_chunking` — Compression + chunks

## Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run specific test class
```bash
pytest tests/test_unary_conversions.py::TestZarrFormat -v
```

### Run specific test
```bash
pytest tests/test_unary_conversions.py::TestZarrFormat::test_zarr_format_v2 -v
```

### Run with coverage
```bash
pytest tests/ --cov=eubi_bridge --cov-report=html -v
```

### Run specific test module
```bash
pytest tests/test_ome_metadata_parsing.py -v
```

### Run only aggregative tests
```bash
pytest tests/test_aggregative_conversions.py -v
```

## Test Data Generation

All test data is generated on-the-fly using fixtures:

- **Synthetic Images**: Generated with numpy in configurable shapes and dtypes
- **ImageJ TIFF**: Created with tifffile.imwrite()
- **OME-TIFF**: Created with tifffile.TiffWriter() and XML metadata
- **File Sets**: Created in temporary directories, cleaned up after tests

No pre-baked test data files are stored in the repository.

## CI/CD Integration

### GitHub Actions Workflow (.github/workflows/test.yml)

**Triggers:**
- Push to main/develop branches
- Pull requests to main/develop
- Daily schedule (2 AM UTC)

**Matrices:**
- OS: Ubuntu-latest, macOS-latest
- Python: 3.11, 3.12

**Jobs:**
1. **test** — Run pytest suite on all OS/Python combinations
2. **lint** — Pylint checks on Python 3.12
3. **integration-test** — Full integration tests on Ubuntu

**Coverage:**
- Codecov upload on successful test completion
- Coverage reports: console + XML format

### Local Testing Before Commit

```bash
# Install test dependencies
pip install pytest pytest-cov numpy tifffile imageio

# Run full test suite
pytest tests/ -v --tb=short

# Run with coverage
pytest tests/ --cov=eubi_bridge --cov-report=term-missing

# Run specific category
pytest tests/test_unary_conversions.py -v
```

## Parameter Coverage Matrix

### Zarr Configuration
- ✅ zarr_format (2, 3)
- ✅ auto_chunk (True/False)
- ✅ Manual chunks (time_chunk, channel_chunk, z_chunk, y_chunk, x_chunk)
- ✅ Zarr v3 sharding coefficients (planned)

### Downscaling
- ✅ n_layers (single to multiple)
- ✅ min_dimension_size (constraint testing)
- ✅ Scale factors per dimension

### Pixel Metadata
- ✅ Custom pixel scales (x, y, z)
- ✅ Custom units (micrometer, nanometer, etc.)
- ✅ Default value handling

### Data Type
- ✅ uint8 preservation
- ✅ uint16 preservation
- ✅ float32 preservation (edge case)

### Compression
- ✅ Compressor selection (blosc, zstd)
- ✅ Compressor parameters (clevel, etc.)

### Aggregative Mode
- ✅ Z-axis concatenation
- ✅ T-axis concatenation
- ✅ C-axis (categorical tags)
- ✅ C-axis (numerical indices)
- ✅ Multi-axis (Z+C)
- ✅ Tag filtering (include/exclude)
- ✅ Error handling (no matches)

### Metadata
- ✅ OME-TIFF channel names
- ✅ OME-TIFF channel colors
- ✅ Channel metadata merging
- ✅ Channel name override (aggregative)
- ✅ Metadata through pyramid

### Edge Cases
- ✅ Very small arrays (16x16)
- ✅ Chunk size > array size
- ✅ Downscale with min_dimension_size
- ✅ Full numeric range (uint16)
- ✅ 5D → 4D squeeze

### Concurrency & Memory
- ✅ Single vs multiple workers
- ✅ max_concurrency parameter
- ✅ region_size_mb limit
- ✅ skip_dask option

## What is not included here:

1. **Proprietary Microscopy Formats** - No input formats other than TIFF and OME-TIFF tested
2. **Sharding Validation** — Detailed sharding codec inspection
3. **Memory Profiling** — Memory usage isn't explicitly validated, only parameters
4. **Performance** — No performance benchmarks (CI optimized for speed, not realism)
5. **Remote Storage** — S3 paths not tested (would require S3 mock or credentials)

## Future Enhancements

### Format Support Testing
Future test suites will expand to include proprietary microscopy formats with their specificities.

These formats will enable comprehensive validation of:
- Metadata extraction and preservation across different vendors
- Complex multi-scene and multi-resolution handling
- Format-specific quirks and edge cases
- Reader robustness and error handling

### Other Future Enhancements
- [ ] Add sharding-specific zarr v3 tests
- [ ] Performance benchmarking suite
- [ ] Remote storage (S3) integration tests with actual S3 or S3-compatible backend
- [ ] Complex aggregative scenarios (>2 axes)
- [ ] Distributed execution tests (dask cluster)
- [ ] Stress tests (very large images)

## Debugging Tests

### Enable verbose output
```bash
pytest tests/ -vv  # Very verbose
pytest tests/ -s   # Show print statements
```

### Run single test with debugger
```bash
pytest tests/test_unary_conversions.py::TestZarrFormat::test_zarr_format_v2 -vv -s
```

### Keep temporary files for inspection
Modify conftest.py to not auto-cleanup tmp_path (inspect /tmp for artifacts)

### Check actual output
```python
# In a test, add:
print(f"Output created at: {output}")
import zarr
gr = zarr.open_group(output)
print(f"Metadata: {gr.attrs}")
```

## Test Statistics

- **Total Tests**: ~48
- **Test Modules**: 5
- **Test Classes**: 20+
- **Fixture Functions**: 13
- **Validation Functions**: 20+
- **CI Platforms**: 2 (Ubuntu, macOS)
- **Python Versions**: 2 (3.11, 3.12)

## Authors

Created as comprehensive CI test suite for EuBI-Bridge project.
Validates all major `eubi to_zarr` parameters and conversion modes.
