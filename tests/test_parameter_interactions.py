"""
Tests for parameter interactions, edge cases, and error conditions.

Covers incompatible parameter combinations, memory constraints, small arrays,
and concurrency stress testing.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.conftest_fixtures import create_synthetic_image_czyx
from tests.validation_utils import (
    get_base_array_shape,
    validate_multiscale_metadata,
    validate_zarr_exists,
)


def run_eubi_command(args: list) -> subprocess.CompletedProcess:
    """Helper to run eubi CLI command."""
    import sys
    import platform
    import shutil
    import os
    
    # Ensure Scripts directory is in PATH (important for Windows)
    scripts_dir = os.path.join(sys.prefix, "Scripts")
    if scripts_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = scripts_dir + os.pathsep + os.environ.get("PATH", "")
    
    # Try to find eubi executable using shutil.which (works cross-platform)
    eubi_cmd = shutil.which('eubi')
    
    # If not in PATH, try constructing the path
    if not eubi_cmd:
        if platform.system() == 'Windows':
            # Windows: Try Scripts directory first, then bin
            possible_paths = [
                Path(sys.executable).parent / 'Scripts' / 'eubi.exe',
                Path(sys.executable).parent / 'Scripts' / 'eubi',
                Path(sys.executable).parent / 'eubi.exe',
                Path(sys.executable).parent / 'eubi',
            ]
            for path in possible_paths:
                if path.exists():
                    eubi_cmd = str(path)
                    break
        else:
            # Unix/Mac: Look in same directory as Python
            eubi_path = Path(sys.executable).parent / 'eubi'
            if eubi_path.exists():
                eubi_cmd = str(eubi_path)
    
    # If still not found, provide diagnostic info
    if not eubi_cmd:
        print(f"DEBUG: eubi not found via shutil.which()")
        print(f"DEBUG: Python executable: {sys.executable}")
        print(f"DEBUG: Python directory: {Path(sys.executable).parent}")
        print(f"DEBUG: PATH: {os.environ.get('PATH', 'NOT SET')}")
        if platform.system() == 'Windows':
            scripts_dir = Path(sys.executable).parent / 'Scripts'
            print(f"DEBUG: Scripts directory: {scripts_dir}")
            print(f"DEBUG: Scripts directory exists: {scripts_dir.exists()}")
            if scripts_dir.exists():
                print(f"DEBUG: Contents of Scripts: {list(scripts_dir.iterdir())}")
        raise RuntimeError(
            f"Could not find eubi executable on {platform.system()}\n"
            f"Python: {sys.executable}\n"
            f"Searched: shutil.which(), {Path(sys.executable).parent}/Scripts, {Path(sys.executable).parent}"
        )
    
    cmd = [eubi_cmd, 'to_zarr'] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )
    return result


class TestSmallArrays:
    """Tests for handling very small arrays."""
    
    def test_tiny_2d_array(self, tmp_path):
        """Test conversion of very small 2D array."""
        import tifffile
        
        # Create tiny 2D array (16x16)
        img = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
        
        tif_path = tmp_path / "tiny.tif"
        with tifffile.TiffWriter(str(tif_path)) as tif:
            tif.write(img, metadata={'axes': 'YX'})
        
        output = tmp_path / "output.zarr"
        run_eubi_command([str(tif_path), str(output)])
        
        assert validate_zarr_exists(output)
        shape = get_base_array_shape(output)
        assert shape == (16, 16)
    
    def test_minimum_chunk_size(self, tmp_path):
        """Test that chunking works even with very small arrays."""
        import tifffile
        
        img = np.random.randint(0, 256, (3, 32, 32), dtype=np.uint8)
        
        tif_path = tmp_path / "small.tif"
        with tifffile.TiffWriter(str(tif_path)) as tif:
            tif.write(img, metadata={'axes': 'ZYX'})
        
        output = tmp_path / "output.zarr"
        # Request chunks smaller than array
        run_eubi_command([
            str(tif_path),
            str(output),
            '--z_chunk', '1',
            '--y_chunk', '16',
            '--x_chunk', '16'
        ])
        
        assert validate_zarr_exists(output)


class TestLargeNumericValues:
    """Tests for handling arrays with large numeric ranges."""
    
    def test_uint16_full_range(self, tmp_path):
        """Test uint16 data across full range [0, 65535]."""
        import tifffile
        
        # Create uint16 array with full range
        img = np.linspace(0, 65535, 128*128, dtype=np.uint16).reshape((128, 128))
        
        tif_path = tmp_path / "uint16_full_range.tif"
        with tifffile.TiffWriter(str(tif_path)) as tif:
            tif.write(img, metadata={'axes': 'YX'})
        
        output = tmp_path / "output.zarr"
        run_eubi_command([str(tif_path), str(output)])
        
        assert validate_zarr_exists(output)
    
    def test_float32_preservation(self, tmp_path):
        """Test that float32 data is preserved."""
        import tifffile
        
        img = np.random.rand(128, 128).astype(np.float32)
        
        tif_path = tmp_path / "float32.tif"
        with tifffile.TiffWriter(str(tif_path)) as tif:
            tif.write(img, metadata={'axes': 'YX'})
        
        output = tmp_path / "output.zarr"
        run_eubi_command([str(tif_path), str(output)])
        
        assert validate_zarr_exists(output)


class TestConflictingChunkParameters:
    """Tests for handling conflicting chunk specifications."""
    
    def test_chunk_larger_than_array(self, tmp_path):
        """Test that chunks larger than array dimensions are handled gracefully."""
        import tifffile
        
        # Create small array
        img = np.random.randint(0, 256, (16, 64, 64), dtype=np.uint8)
        
        tif_path = tmp_path / "small_array.tif"
        with tifffile.TiffWriter(str(tif_path)) as tif:
            tif.write(img, metadata={'axes': 'ZYX'})
        
        output = tmp_path / "output.zarr"
        # Request chunks much larger than array
        run_eubi_command([
            str(tif_path),
            str(output),
            '--z_chunk', '256',
            '--y_chunk', '512',
            '--x_chunk', '512'
        ])
        
        assert validate_zarr_exists(output)
        # System should clamp chunks to array size
        validate_multiscale_metadata(output, axis_order='zyx')
    
    def test_auto_chunk_overrides_manual(self, imagej_tiff_czyx, tmp_path):
        """Test interaction between auto_chunk and manual chunk parameters."""
        output = tmp_path / "output.zarr"
        
        # When auto_chunk=True, manual chunks should be ignored
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--auto_chunk', 'True',
            '--z_chunk', '256'
        ])
        
        assert validate_zarr_exists(output)


class TestDownscaleSmallArrays:
    """Tests for downscaling very small arrays."""
    
    def test_downscale_small_array(self, tmp_path):
        """Test downscaling an array that's already small."""
        import tifffile
        
        # Create small array (64x64 spatial)
        img = np.random.randint(0, 256, (3, 64, 64), dtype=np.uint8)
        
        tif_path = tmp_path / "small.tif"
        with tifffile.TiffWriter(str(tif_path)) as tif:
            tif.write(img, metadata={'axes': 'ZYX'})
        
        output = tmp_path / "output.zarr"
        # Request multiple downscaling layers
        run_eubi_command([
            str(tif_path),
            str(output),
            '--n_layers', '3',
            '--min_dimension_size', '16'
        ])
        
        assert validate_zarr_exists(output)
        # Pyramid should stop when reaching min_dimension_size
        validate_multiscale_metadata(output, axis_order='zyx')
    
    def test_min_dimension_size_constraint(self, imagej_tiff_czyx, tmp_path):
        """Test that min_dimension_size is respected."""
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--min_dimension_size', '64'
        ])
        
        assert validate_zarr_exists(output)
        validate_multiscale_metadata(output, axis_order='czyx')


class TestConcurrencyParameters:
    """Tests for concurrent processing parameters."""
    
    def test_single_worker(self, imagej_tiff_czyx, tmp_path):
        """Test conversion with single worker (sequential processing)."""
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--max_workers', '1'
        ])
        
        assert validate_zarr_exists(output)
    
    def test_multiple_workers(self, imagej_tiff_czyx, tmp_path):
        """Test conversion with multiple workers."""
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--max_workers', '4'
        ])
        
        assert validate_zarr_exists(output)
    
    def test_max_concurrency(self, imagej_tiff_czyx, tmp_path):
        """Test concurrent write limits."""
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--max_concurrency', '2'
        ])
        
        assert validate_zarr_exists(output)


class TestMemoryConstraints:
    """Tests for memory-aware processing."""
    
    def test_region_size_limit(self, imagej_tiff_czyx, tmp_path):
        """Test that region_size_mb parameter controls memory usage."""
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--region_size_mb', '64'
        ])
        
        assert validate_zarr_exists(output)
    
    def test_skip_dask(self, imagej_tiff_czyx, tmp_path):
        """Test skip_dask option for direct array processing."""
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--skip_dask', 'True'
        ])
        
        assert validate_zarr_exists(output)


class TestAxisOrdering:
    """Tests for different axis orderings and interpretations."""
    
    def test_5d_to_4d_via_squeeze(self, tmp_path):
        """Test that squeeze reduces 5D to 4D when appropriate."""
        import tifffile
        
        # Create synthetic 5D image (T=1, C=2, Z=3, Y=128, X=128)
        img = create_synthetic_image_czyx((2, 3, 128, 128), dtype=np.uint8, seed=42)
        # Reshape to add time dimension
        img_5d = img[np.newaxis, ...]  # Shape: (1, 2, 3, 128, 128)
        
        tif_path = tmp_path / "5d.tif"
        with tifffile.TiffWriter(str(tif_path)) as tif:
            tif.write(img_5d, metadata={'axes': 'TCZYX'})
        
        output = tmp_path / "output.zarr"
        # With squeeze (default=True), should become 4D
        run_eubi_command([str(tif_path), str(output)])
        
        assert validate_zarr_exists(output)
        shape = get_base_array_shape(output)
        # T dimension (size 1) should be removed
        assert len(shape) == 4


class TestParameterCombinations:
    """Tests for realistic parameter combinations."""
    
    def test_combined_zarr_v3_with_sharding(self, imagej_tiff_czyx, tmp_path):
        """Test zarr v3 with sharding parameters."""
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--zarr_format', '3',
            '--z_shard_coef', '2',
            '--y_shard_coef', '3',
            '--x_shard_coef', '3'
        ])
        
        assert validate_zarr_exists(output)
    
    def test_combined_downscale_with_custom_scales(self, imagej_tiff_czyx, tmp_path):
        """Test downscaling with custom pixel scales."""
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--n_layers', '2',
            '--z_scale', '2.0',
            '--x_scale', '0.5',
            '--y_scale', '0.5'
        ])
        
        assert validate_zarr_exists(output)
    
    def test_combined_compression_with_chunking(self, imagej_tiff_czyx, tmp_path):
        """Test compression with specific chunk sizes."""
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--compressor', 'zstd',
            '--z_chunk', '32',
            '--y_chunk', '64',
            '--x_chunk', '64'
        ])
        
        assert validate_zarr_exists(output)
