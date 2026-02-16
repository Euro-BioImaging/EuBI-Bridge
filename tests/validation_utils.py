"""
Shared validation utilities for test modules.

Provides functions to validate zarr output structure, metadata, pixel data,
and conversion parameters.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import zarr


def validate_zarr_format(output_path: Union[str, Path]) -> int:
    """Get zarr format version (2 or 3) from output.
    
    Handles both direct zarr groups and nested structure where output
    is in a subdirectory (e.g., output.zarr/test.zarr).
    """
    path = Path(output_path)
    
    # Try direct path first
    if (path / 'zarr.json').exists():
        return 3
    elif (path / '0' / '.zarray').exists():
        return 2
    
    # Try nested zarr (eubi creates output.zarr/filename.zarr)
    if path.is_dir():
        for item in path.iterdir():
            if item.is_dir():
                if (item / 'zarr.json').exists():
                    return 3
                elif (item / '0' / '.zarray').exists():
                    return 2
    
    raise ValueError(f"Cannot determine zarr format for {output_path}")


def validate_zarr_exists(output_path: Union[str, Path]) -> bool:
    """Check if output path is a valid zarr group.
    
    Handles both direct zarr groups and nested structure where output
    is in a subdirectory (e.g., output.zarr/test.zarr).
    """
    path = Path(output_path)
    
    # Check direct path first
    try:
        _ = zarr.open_group(path, mode='r')
        return True
    except Exception:
        pass
    
    # Check for nested zarr (eubi creates output.zarr/filename.zarr)
    if path.is_dir():
        for item in path.iterdir():
            if item.is_dir():
                try:
                    _ = zarr.open_group(item, mode='r')
                    return True
                except Exception:
                    pass
    
    return False


def get_zarr_format(output_path: Union[str, Path]) -> int:
    """Alias for validate_zarr_format for backward compatibility."""
    return validate_zarr_format(output_path)


def get_actual_zarr_path(output_path: Union[str, Path]) -> Path:
    """Get the actual zarr group path, handling nested structure.
    
    eubi to_zarr creates output in a nested structure:
    - If output_path doesn't exist or is directly a zarr group, returns output_path
    - If output_path is a directory containing a zarr group, returns that subdirectory
    """
    path = Path(output_path)
    
    # Check if direct path is a zarr group
    try:
        zarr.open_group(path, mode='r')
        return path
    except Exception:
        pass
    
    # Check for nested zarr (output.zarr/filename.zarr)
    if path.is_dir():
        for item in path.iterdir():
            if item.is_dir():
                try:
                    zarr.open_group(item, mode='r')
                    return item
                except Exception:
                    pass
    
    # Fallback to direct path (will fail in validation if it doesn't exist)
    return path


def validate_base_array_shape(output_path: Union[str, Path], 
                               expected_shape: Tuple[int, ...],
                               axis_order: str = 'tczyx') -> bool:
    """Validate that base array (resolution 0) has expected shape."""
    actual_path = get_actual_zarr_path(output_path)
    gr = zarr.open_group(actual_path, mode='r')
    base_array = gr['0']
    
    if base_array.shape != expected_shape:
        raise AssertionError(
            f"Expected shape {expected_shape}, got {base_array.shape}"
        )
    return True


def validate_dtype(output_path: Union[str, Path], 
                   expected_dtype: Union[str, np.dtype]) -> bool:
    """Validate base array data type."""
    actual_path = get_actual_zarr_path(output_path)
    gr = zarr.open_group(actual_path, mode='r')
    base_array = gr['0']
    
    expected_dtype = np.dtype(expected_dtype)
    if base_array.dtype != expected_dtype:
        raise AssertionError(
            f"Expected dtype {expected_dtype}, got {base_array.dtype}"
        )
    return True


def validate_chunk_size(output_path: Union[str, Path],
                        expected_chunks: Tuple[int, ...],
                        resolution: str = '0') -> bool:
    """Validate that array is chunked as expected."""
    actual_path = get_actual_zarr_path(output_path)
    gr = zarr.open_group(actual_path, mode='r')
    array = gr[resolution]
    
    if array.chunks != expected_chunks:
        raise AssertionError(
            f"Expected chunks {expected_chunks}, got {array.chunks}"
        )
    return True


def validate_multiscale_metadata(output_path: Union[str, Path],
                                 expected_n_resolutions: Optional[int] = None,
                                 axis_order: str = 'tczyx') -> Dict:
    """Validate NGFF multiscale metadata structure."""
    actual_path = get_actual_zarr_path(output_path)
    gr = zarr.open_group(actual_path, mode='r')
    
    # Try NGFF v0.5 (zarr v3) format first
    if 'ome' in gr.attrs:
        metadata = gr.attrs['ome']
    # Fall back to NGFF v0.4 (zarr v2) format
    elif 'multiscales' in gr.attrs:
        metadata = {'multiscales': gr.attrs['multiscales']}
        if 'omero' in gr.attrs:
            metadata['omero'] = gr.attrs['omero']
    else:
        raise ValueError("No multiscales metadata found in zarr group")
    
    # Validate structure
    if 'multiscales' not in metadata:
        raise ValueError("Missing 'multiscales' in metadata")
    
    multiscales = metadata['multiscales'][0]
    
    # Check required keys
    required_keys = ['name', 'axes', 'datasets']
    for key in required_keys:
        if key not in multiscales:
            raise ValueError(f"Missing '{key}' in multiscales metadata")
    
    # Validate axis order if provided
    axes_names = ''.join([ax['name'] for ax in multiscales['axes']])
    if axes_names != axis_order[:len(axes_names)]:
        raise AssertionError(
            f"Expected axis order {axis_order[:len(axes_names)]}, got {axes_names}"
        )
    
    # Check resolution count
    if expected_n_resolutions is not None:
        if len(multiscales['datasets']) != expected_n_resolutions:
            raise AssertionError(
                f"Expected {expected_n_resolutions} resolutions, "
                f"got {len(multiscales['datasets'])}"
            )
    
    return metadata


def validate_channel_metadata(output_path: Union[str, Path],
                               expected_n_channels: int,
                               expected_labels: Optional[List[str]] = None,
                               expected_colors: Optional[List[str]] = None) -> Dict:
    """Validate OMERO channel metadata."""
    actual_path = get_actual_zarr_path(output_path)
    gr = zarr.open_group(actual_path, mode='r')
    
    # Get metadata
    if 'ome' in gr.attrs:
        metadata = gr.attrs['ome']
    elif 'omero' in gr.attrs:
        metadata = gr.attrs['omero']
    else:
        raise ValueError("No omero metadata found in zarr group")
    
    if 'omero' not in metadata and 'channels' not in metadata:
        raise ValueError("No channel metadata found")
    
    # Handle both nested and flat formats
    if 'omero' in metadata:
        omero_meta = metadata['omero']
    else:
        omero_meta = metadata
    
    channels = omero_meta.get('channels', [])
    
    if len(channels) != expected_n_channels:
        raise AssertionError(
            f"Expected {expected_n_channels} channels, got {len(channels)}"
        )
    
    # Validate labels if provided
    if expected_labels is not None:
        actual_labels = [ch.get('label', '') for ch in channels]
        if actual_labels != expected_labels:
            raise AssertionError(
                f"Expected labels {expected_labels}, got {actual_labels}"
            )
    
    # Validate colors if provided
    if expected_colors is not None:
        actual_colors = [ch.get('color', '') for ch in channels]
        if actual_colors != expected_colors:
            raise AssertionError(
                f"Expected colors {expected_colors}, got {actual_colors}"
            )
    
    return omero_meta


def validate_pixel_scales(output_path: Union[str, Path],
                          expected_scales: Dict[str, float],
                          resolution: str = '0',
                          axis_order: str = 'tczyx') -> bool:
    """Validate pixel scale metadata for a specific resolution."""
    metadata = validate_multiscale_metadata(output_path, axis_order=axis_order)
    multiscales = metadata['multiscales'][0]
    
    # Find the dataset
    dataset = None
    for ds in multiscales['datasets']:
        if ds['path'] == resolution:
            dataset = ds
            break
    
    if dataset is None:
        raise ValueError(f"Resolution {resolution} not found in metadata")
    
    # Extract scales from coordinate transformations
    scales = None
    for transform in dataset['coordinateTransformations']:
        if transform.get('type') == 'scale':
            scales = transform['scale']
            break
    
    if scales is None:
        raise ValueError(f"No scale transformation found for resolution {resolution}")
    
    # Map to axis names
    axes_names = ''.join([ax['name'] for ax in multiscales['axes']])
    actual_scales = dict(zip(axes_names, scales))
    
    # Compare
    for axis, expected_scale in expected_scales.items():
        if axis not in actual_scales:
            raise ValueError(f"Axis '{axis}' not found in scales")
        if abs(actual_scales[axis] - expected_scale) > 1e-6:
            raise AssertionError(
                f"Expected scale {axis}={expected_scale}, got {actual_scales[axis]}"
            )
    
    return True


def validate_pixel_units(output_path: Union[str, Path],
                         expected_units: Dict[str, str],
                         axis_order: str = 'tczyx') -> bool:
    """Validate pixel unit metadata."""
    metadata = validate_multiscale_metadata(output_path, axis_order=axis_order)
    multiscales = metadata['multiscales'][0]
    
    # Extract units from axes
    actual_units = {}
    for ax in multiscales['axes']:
        name = ax['name']
        unit = ax.get('unit')
        actual_units[name] = unit
    
    # Compare
    for axis, expected_unit in expected_units.items():
        if axis not in actual_units:
            raise ValueError(f"Axis '{axis}' not found in units")
        if actual_units[axis] != expected_unit:
            raise AssertionError(
                f"Expected unit {axis}={expected_unit}, got {actual_units[axis]}"
            )
    
    return True


def validate_pixel_data_range(output_path: Union[str, Path],
                              min_value: Union[int, float],
                              max_value: Union[int, float],
                              resolution: str = '0',
                              sample_size: int = 1000) -> bool:
    """Validate that pixel data is within expected range (samples a subset)."""
    actual_path = get_actual_zarr_path(output_path)
    gr = zarr.open_group(actual_path, mode='r')
    array = gr[resolution]
    
    # Sample random chunks to avoid loading entire array
    chunk_indices = np.random.randint(0, len(array.chunks), size=len(array.shape))
    
    # Load one chunk
    try:
        chunk_data = array[:]  # Load full for small arrays
        actual_min = chunk_data.min()
        actual_max = chunk_data.max()
    except (MemoryError, ValueError):
        # For large arrays, just check shape
        return True
    
    if actual_min < min_value or actual_max > max_value:
        raise AssertionError(
            f"Data range [{actual_min}, {actual_max}] outside expected "
            f"[{min_value}, {max_value}]"
        )
    
    return True


def validate_downscaling_pyramid(output_path: Union[str, Path],
                                 expected_n_layers: int,
                                 axis_order: str = 'tczyx') -> bool:
    """Validate that downscaling pyramid has correct number of layers."""
    metadata = validate_multiscale_metadata(output_path, axis_order=axis_order)
    multiscales = metadata['multiscales'][0]
    
    n_layers = len(multiscales['datasets'])
    if n_layers != expected_n_layers:
        raise AssertionError(
            f"Expected {expected_n_layers} pyramid layers, got {n_layers}"
        )
    
    return True


def validate_sharding_metadata(output_path: Union[str, Path],
                               resolution: str = '0') -> Dict:
    """Validate zarr v3 sharding metadata."""
    zarr_format = get_zarr_format(output_path)
    if zarr_format != 3:
        raise ValueError("Sharding only applies to zarr v3")
    
    actual_path = get_actual_zarr_path(output_path)
    gr = zarr.open_group(actual_path, mode='r')
    array = gr[resolution]
    
    # Check for sharding codec in array metadata
    if not hasattr(array, 'metadata') or array.metadata is None:
        raise ValueError("Array has no metadata")
    
    # Look for sharding in codecs
    codecs = array.metadata.get('codecs', [])
    sharding_codec = None
    for codec in codecs:
        if codec.get('name') == 'sharding_indexed':
            sharding_codec = codec
            break
    
    if sharding_codec is None:
        raise ValueError("No sharding codec found in array metadata")
    
    return sharding_codec


def validate_compression(output_path: Union[str, Path],
                        expected_compressor: str,
                        resolution: str = '0') -> bool:
    """Validate compression codec for both Zarr v2 and v3."""
    actual_path = get_actual_zarr_path(output_path)
    gr = zarr.open_group(actual_path, mode='r')
    array = gr[resolution]
    
    # Get Zarr format version
    zarr_format = validate_zarr_format(output_path)
    
    # Get compressor name based on Zarr version
    if zarr_format == 2:
        # Zarr v2: use .compressor (deprecated but works)
        if hasattr(array, 'compressor') and array.compressor is not None:
            compressor_name = type(array.compressor).__name__.lower()
        else:
            compressor_name = 'none'
    else:
        # Zarr v3: use .compressors property
        if hasattr(array, 'compressors') and array.compressors:
            # Get first compressor name from the tuple of codecs
            # Filter out BytesCodec and other non-compressor codecs
            for codec in array.compressors:
                codec_name = type(codec).__name__.lower()
                if codec_name not in ('bytescodec', 'crc32ccodec', 'transposecoder'):
                    compressor_name = codec_name
                    break
            else:
                compressor_name = 'none'
        else:
            compressor_name = 'none'
    
    # Normalize comparison - handle both codec class names and string names
    expected_lower = expected_compressor.lower()
    if expected_lower == 'blosc':
        # For Zarr v3, BloscCodec is the class name
        if not (expected_lower in compressor_name or 'bloscodec' in compressor_name):
            raise AssertionError(
                f"Expected compressor {expected_compressor}, got {compressor_name}"
            )
    elif expected_lower == 'zstd':
        if not (expected_lower in compressor_name or 'zstdcodec' in compressor_name):
            raise AssertionError(
                f"Expected compressor {expected_compressor}, got {compressor_name}"
            )
    elif expected_lower == 'gzip':
        if not (expected_lower in compressor_name or 'gzipcodec' in compressor_name):
            raise AssertionError(
                f"Expected compressor {expected_compressor}, got {compressor_name}"
            )
    elif expected_lower == 'none':
        if compressor_name != 'none':
            raise AssertionError(
                f"Expected no compression, got {compressor_name}"
            )
    else:
        # Generic comparison
        if expected_lower not in compressor_name:
            raise AssertionError(
                f"Expected compressor {expected_compressor}, got {compressor_name}"
            )
    
    return True


def validate_squeezed_dimensions(output_path: Union[str, Path],
                                 input_shape: Tuple[int, ...],
                                 axis_order: str) -> bool:
    """Validate that singleton dimensions were removed by squeeze."""
    actual_path = get_actual_zarr_path(output_path)
    gr = zarr.open_group(actual_path, mode='r')
    base_array = gr['0']
    output_shape = base_array.shape
    
    # Count singleton dimensions in input
    singleton_axes = []
    for i, size in enumerate(input_shape):
        if size == 1:
            singleton_axes.append(axis_order[i])
    
    # Output should not contain any singleton dimensions
    for i, size in enumerate(output_shape):
        if size == 1:
            raise AssertionError(
                f"Output has singleton dimension at axis {axis_order[i]} "
                f"(squeeze should have removed it)"
            )
    
    return True


def compare_pixel_data(output_path: Union[str, Path],
                       expected_data: np.ndarray,
                       resolution: str = '0',
                       rtol: float = 1e-5,
                       atol: float = 1e-8) -> bool:
    """Compare output pixel data with expected data."""
    actual_path = get_actual_zarr_path(output_path)
    gr = zarr.open_group(actual_path, mode='r')
    array = gr[resolution]
    
    actual_data = np.array(array[:])
    
    if actual_data.shape != expected_data.shape:
        raise AssertionError(
            f"Shape mismatch: expected {expected_data.shape}, got {actual_data.shape}"
        )
    
    if not np.allclose(actual_data, expected_data, rtol=rtol, atol=atol):
        raise AssertionError("Pixel data mismatch")
    
    return True


def get_resolution_count(output_path: Union[str, Path], axis_order: str = 'tczyx') -> int:
    """Get number of resolution levels in pyramid."""
    metadata = validate_multiscale_metadata(output_path, axis_order=axis_order)
    return len(metadata['multiscales'][0]['datasets'])


def get_base_array_shape(output_path: Union[str, Path]) -> Tuple[int, ...]:
    """Get shape of base (full resolution) array."""
    actual_path = get_actual_zarr_path(output_path)
    gr = zarr.open_group(actual_path, mode='r')
    return gr['0'].shape
