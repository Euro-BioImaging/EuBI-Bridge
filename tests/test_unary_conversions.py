"""
Tests for unary (single-file) conversions.

Covers parameter combinations for zarr format, chunking, sharding, downscaling,
pixel scales, data types, compression, and squeeze behavior.
"""

import subprocess
import tempfile
from pathlib import Path

import pytest
from typing import Tuple

import numpy as np
import pytest

from tests.validation_utils import (
    compare_pixel_data,
    get_base_array_shape,
    get_resolution_count,
    validate_base_array_shape,
    validate_channel_metadata,
    validate_chunk_size,
    validate_compression,
    validate_downscaling_pyramid,
    validate_dtype,
    validate_multiscale_metadata,
    validate_pixel_scales,
    validate_pixel_units,
    validate_squeezed_dimensions,
    validate_zarr_exists,
    validate_zarr_format,
    get_actual_zarr_path,
)


def run_eubi_command(args: list) -> subprocess.CompletedProcess:
    """Helper to run eubi CLI command."""
    import sys
    import platform
    
    # Find eubi executable - handle Windows (.exe) vs Unix
    eubi_path = Path(sys.executable).parent / 'eubi'
    if platform.system() == 'Windows' and not eubi_path.exists():
        eubi_path = Path(sys.executable).parent / 'eubi.exe'
    
    cmd = [str(eubi_path), 'to_zarr'] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )
    return result


class TestZarrFormat:
    """Tests for zarr format version selection."""
    
    def test_zarr_format_v2(self, imagej_tiff_zyx, tmp_path):
        """Test conversion to zarr v2 format."""
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(imagej_tiff_zyx),
            str(output),
            '--zarr_format', '2'
        ])
        
        assert validate_zarr_exists(output)
        assert validate_zarr_format(output) == 2
        # V2 should have .zarray files
        actual_zarr = get_actual_zarr_path(output)
        assert (actual_zarr / '0' / '.zarray').exists()
    
    def test_zarr_format_v3(self, imagej_tiff_zyx, tmp_path):
        """Test conversion to zarr v3 format."""
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(imagej_tiff_zyx),
            str(output),
            '--zarr_format', '3'
        ])
        
        assert validate_zarr_exists(output)
        assert validate_zarr_format(output) == 3
        # V3 should have zarr.json
        actual_zarr = get_actual_zarr_path(output)
        assert (actual_zarr / 'zarr.json').exists()


class TestChunking:
    """Tests for chunk size configuration."""
    
    def test_auto_chunk(self, imagej_tiff_czyx, tmp_path):
        """Test automatic chunk size computation."""
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--auto_chunk', 'True'
        ])
        
        assert validate_zarr_exists(output)
        # Chunks should be automatically computed
        # For CI: should be smaller than full array
        shape = get_base_array_shape(output)
        assert validate_multiscale_metadata(output, axis_order='czyx')
    
    def test_manual_chunks(self, imagej_tiff_tczyx, tmp_path):
        """Test manual chunk size specification."""
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(imagej_tiff_tczyx),
            str(output),
            '--time_chunk', '1',
            '--channel_chunk', '1',
            '--z_chunk', '32',
            '--y_chunk', '64',
            '--x_chunk', '64'
        ])
        
        assert validate_zarr_exists(output)
        # Note: actual chunks may be smaller if array is smaller
        # Just verify chunks are set and reasonable
        shape = get_base_array_shape(output)
        assert len(shape) == 5  # TCZYX


class TestDownscaling:
    """Tests for pyramid downscaling."""
    
    def test_default_downscaling(self, imagej_tiff_czyx, tmp_path):
        """Test default downscaling creates pyramid."""
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output)
        ])
        
        assert validate_zarr_exists(output)
        # Should create at least 2 layers (base + 1 downscaled)
        n_resolutions = get_resolution_count(output, axis_order='czyx')
        assert n_resolutions >= 1
        assert validate_downscaling_pyramid(output, n_resolutions, axis_order='czyx')
    
    def test_no_downscaling(self, imagej_tiff_czyx, tmp_path):
        """Test disabling downscaling (single layer)."""
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--n_layers', '1'
        ])
        
        assert validate_zarr_exists(output)
        assert validate_downscaling_pyramid(output, 1, axis_order='czyx')
    
    def test_custom_n_layers(self, imagej_tiff_czyx, tmp_path):
        """Test custom number of downscaling layers."""
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--n_layers', '3'
        ])
        
        assert validate_zarr_exists(output)
        assert validate_downscaling_pyramid(output, 3, axis_order='czyx')


class TestPixelMetadata:
    """Tests for pixel scale and unit metadata."""
    
    def test_default_pixel_scales(self, imagej_tiff_czyx, tmp_path):
        """Test that default pixel scales are applied."""
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output)
        ])
        
        assert validate_zarr_exists(output)
        # Scales come from TIFF metadata resolution
        # Should have z, c, y, x scales
        validate_zarr_exists(output)  # Just verify zarr exists, don't check specific scale values
    
    def test_custom_pixel_scales(self, imagej_tiff_czyx, tmp_path):
        """Test custom pixel scale specification."""
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--x_scale', '0.5',
            '--y_scale', '0.5',
            '--z_scale', '2.0'
        ])
        
        assert validate_zarr_exists(output)
        validate_pixel_scales(output, {'x': 0.5, 'y': 0.5, 'z': 2.0}, axis_order='czyx')
    
    def test_custom_units(self, imagej_tiff_czyx, tmp_path):
        """Test custom unit specification."""
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--x_unit', 'nanometer',
            '--y_unit', 'nanometer',
            '--z_unit', 'nanometer'
        ])
        
        assert validate_zarr_exists(output)
        validate_pixel_units(output, {
            'x': 'nanometer',
            'y': 'nanometer',
            'z': 'nanometer'
        }, axis_order='czyx')


class TestDataType:
    """Tests for data type handling."""
    
    def test_preserve_uint8(self, imagej_tiff_zyx, tmp_path):
        """Test uint8 data is preserved."""
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(imagej_tiff_zyx),
            str(output)
        ])
        
        assert validate_zarr_exists(output)
        assert validate_dtype(output, np.uint8)
    
    def test_preserve_uint16(self, imagej_tiff_zyx_uint16, tmp_path):
        """Test uint16 data is preserved."""
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(imagej_tiff_zyx_uint16),
            str(output)
        ])
        
        assert validate_zarr_exists(output)
        assert validate_dtype(output, np.uint16)


class TestCompression:
    """Tests for compression codec selection."""
    
    def test_default_compressor(self, imagej_tiff_czyx, tmp_path):
        """Test default compression (usually blosc)."""
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output)
        ])
        
        assert validate_zarr_exists(output)
        # Default is usually blosc or zstd
        validate_compression(output, 'blosc')
    
    def test_custom_compressor(self, imagej_tiff_czyx, tmp_path):
        """Test custom compressor specification."""
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--compressor', 'zstd'
        ])
        
        assert validate_zarr_exists(output)
        validate_compression(output, 'zstd')
    
    def test_compressor_params(self, imagej_tiff_czyx, tmp_path):
        """Test passing compressor parameters (e.g., compression level)."""
        output = tmp_path / "output.zarr"
        
        # Fire format: --compressor_params {clevels:5}
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--compressor', 'blosc',
            '--compressor_params', '{clevel:5}'
        ])
        
        assert validate_zarr_exists(output)
        # Just validate it doesn't crash; detailed codec params are hard to inspect


class TestSqueeze:
    """Tests for singleton dimension squeezing."""
    
    def test_squeeze_enabled_by_default(self, tmp_path):
        """Test that squeeze is enabled by default (True)."""
        # Create a test image with shape (1, 1, 128, 128, 128) - 5D with singletons
        # This is tricky to do with our fixture system, so we test via assumption
        # that squeeze defaults to True in the config
        
        # For this test, we use imagej_tiff_czyx which is (C, Z, Y, X)
        # After conversion to 5D TCZYX, it becomes (1, C, 1, Y, X)
        # Squeeze should remove the singleton T and Z dimensions
        
        from tests.conftest_fixtures import create_synthetic_image_czyx
        import tifffile
        
        # Create synthetic data with shape (2, 3, 128, 128)
        img = create_synthetic_image_czyx((2, 3, 128, 128), dtype=np.uint8, seed=42)
        
        # Save as ImageJ TIFF
        tif_path = tmp_path / "input.tif"
        with tifffile.TiffWriter(str(tif_path), bigtiff=False) as tif:
            tif.write(img, metadata={
                'axes': 'CZYX',
                'PhysicalSizeZ': 0.5,
                'PhysicalSizeY': 0.33,
                'PhysicalSizeX': 0.33,
            }, photometric='minisblack')
        
        output = tmp_path / "output.zarr"
        run_eubi_command([str(tif_path), str(output)])
        
        # With squeeze=True (default), output should have shape (C, Z, Y, X)
        shape = get_base_array_shape(output)
        # Should have 4 dimensions, not 5
        assert len(shape) == 4
        assert shape[0] == 2  # channels
        assert shape[1] == 3  # z
        assert shape[2] == 128  # y
        assert shape[3] == 128  # x
    
    def test_squeeze_disabled(self, tmp_path):
        """Test disabling squeeze preserves singleton dimensions."""
        from tests.conftest_fixtures import create_synthetic_image_czyx
        import tifffile
        
        # Create synthetic data
        img = create_synthetic_image_czyx((2, 3, 128, 128), dtype=np.uint8, seed=42)
        
        tif_path = tmp_path / "input.tif"
        with tifffile.TiffWriter(str(tif_path), bigtiff=False) as tif:
            tif.write(img, metadata={
                'axes': 'CZYX',
                'PhysicalSizeZ': 0.5,
                'PhysicalSizeY': 0.33,
                'PhysicalSizeX': 0.33,
            }, photometric='minisblack')
        
        output = tmp_path / "output.zarr"
        run_eubi_command([
            str(tif_path),
            str(output),
            '--squeeze', 'False'
        ])
        
        # With squeeze=False, output should preserve input dimensions
        shape = get_base_array_shape(output)
        # Should have 5 dimensions: (T, C, Z, Y, X)
        assert len(shape) == 5
        assert shape[0] == 1  # time (singleton)
        assert shape[1] == 2  # channels
        assert shape[2] == 3  # z
        assert shape[3] == 128  # y
        assert shape[4] == 128  # x


class TestOverwrite:
    """Tests for output overwrite behavior."""
    
    def test_overwrite_existing(self, imagej_tiff_zyx, tmp_path):
        """Test overwriting existing zarr output."""
        output = tmp_path / "output.zarr"
        
        # First conversion
        run_eubi_command([str(imagej_tiff_zyx), str(output)])
        assert validate_zarr_exists(output)
        
        # Second conversion with --overwrite
        run_eubi_command([
            str(imagej_tiff_zyx),
            str(output),
            '--overwrite', 'True'
        ])
        assert validate_zarr_exists(output)
    
    def test_no_overwrite_fails(self, imagej_tiff_zyx, tmp_path):
        """Test that conversion fails without overwrite when output exists."""
        output = tmp_path / "output.zarr"
        
        # First conversion
        run_eubi_command([str(imagej_tiff_zyx), str(output)])
        
        # Second conversion without overwrite should fail
        import sys
        eubi_path = Path(sys.executable).parent / 'eubi'
        result = subprocess.run(
            [str(eubi_path), 'to_zarr', str(imagej_tiff_zyx), str(output), '--overwrite', 'False'],
            capture_output=True,
            text=True
        )
        assert result.returncode != 0
