"""
Tests for aggregative (multi-file) conversions.

Covers concatenating multiple files along different axes (Z, T, C),
tag-based filtering, categorical vs numerical channel tags, and error handling.
"""

import subprocess
from pathlib import Path

import pytest

from tests.validation_utils import (
    get_base_array_shape,
    validate_channel_metadata,
    validate_multiscale_metadata,
    validate_zarr_exists,
)


def run_eubi_command(args: list) -> subprocess.CompletedProcess:
    """Helper to run eubi CLI command."""
    import sys
    import platform
    import shutil
    import os
    
    print(f"\n[DEBUG] Running eubi command with args: {args}")
    
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
                try:
                    print(f"DEBUG: Contents of Scripts: {list(scripts_dir.iterdir())}")
                except (OSError, FileNotFoundError) as e:
                    print(f"DEBUG: Could not list Scripts directory: {e}")
        raise RuntimeError(
            f"Could not find eubi executable on {platform.system()}\n"
            f"Python: {sys.executable}\n"
            f"Searched: shutil.which(), {Path(sys.executable).parent}/Scripts, {Path(sys.executable).parent}"
        )
    
    cmd = [eubi_cmd, 'to_zarr'] + args
    print(f"[DEBUG] Full command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(f"[DEBUG] Command return code: {result.returncode}")
    if result.stdout:
        print(f"[DEBUG] STDOUT:\n{result.stdout[:500]}")
    if result.stderr:
        print(f"[DEBUG] STDERR:\n{result.stderr[:500]}")
    
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )
    return result


class TestZConcatenation:
    """Tests for Z-axis concatenation."""
    
    def test_z_concat_basic(self, aggregative_z_concat_files, tmp_path):
        """Test concatenating Z slices from multiple files."""
        tmpdir, files = aggregative_z_concat_files
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(tmpdir),
            str(output),
            '--concatenation_axes', 'z',
            '--time_tag', 't',
            '--z_tag', 'z'
        ])
        
        assert validate_zarr_exists(output)
        shape = get_base_array_shape(output)
        
        # Should have concatenated Z slices
        # Images are 2D (128, 128), read as (1, 1, 1, 128, 128) [T,C,Z,Y,X]
        # After Z concatenation: (1, 1, 3, 128, 128)
        # After squeeze: (3, 128, 128) [Z, Y, X]
        assert shape[0] == 3  # Z dimension should have 3 slices


class TestTConcatenation:
    """Tests for T (time) axis concatenation."""
    
    def test_t_concat_from_z_files(self, aggregative_z_concat_files, tmp_path):
        """Test concatenating time points from Z-organized files."""
        tmpdir, files = aggregative_z_concat_files
        output = tmp_path / "output.zarr"
        
        # Organize files have t0, t1 timepoints - concatenate along T
        # Must also provide z_tag to properly group Z slices within each timepoint
        run_eubi_command([
            str(tmpdir),
            str(output),
            '--concatenation_axes', 't',
            '--time_tag', 't',
            '--z_tag', 'z'
        ])
        
        assert validate_zarr_exists(output)
        shape = get_base_array_shape(output)
        
        # Should have time concatenation
        # Expected time dimension with 2 timepoints
        assert shape[0] == 2  # T dimension


class TestChannelConcatenationCategorical:
    """Tests for channel concatenation with categorical tags."""
    
    def test_channel_concat_categorical(self, aggregative_channel_categorical_files, tmp_path):
        """Test concatenating channels with categorical tag matching."""
        tmpdir, files = aggregative_channel_categorical_files
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(tmpdir),
            str(output),
            '--concatenation_axes', 'c',
            '--time_tag', 't',
            '--channel_tag', 'gfp,mcherry'
        ])
        
        assert validate_zarr_exists(output)
        shape = get_base_array_shape(output)
        
        # Should have 2 channels (gfp and mcherry)
        assert shape[0] == 2  # C dimension after squeeze
    
    def test_channel_concat_categorical_with_override(self, aggregative_channel_categorical_files, tmp_path):
        """Test channel concatenation with name override."""
        tmpdir, files = aggregative_channel_categorical_files
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(tmpdir),
            str(output),
            '--concatenation_axes', 'c',
            '--time_tag', 't',
            '--channel_tag', 'gfp,mcherry',
            '--override_channel_names', 'True'
        ])
        
        assert validate_zarr_exists(output)
        # Channel names should be gfp, mcherry from tags
        validate_channel_metadata(
            output,
            expected_n_channels=2,
            expected_labels=['gfp', 'mcherry']
        )


class TestChannelConcatenationNumerical:
    """Tests for channel concatenation with numerical indices."""
    
    def test_channel_concat_numerical(self, aggregative_channel_numerical_files, tmp_path):
        """Test concatenating channels with numerical tag matching (channel1, channel2, etc)."""
        tmpdir, files = aggregative_channel_numerical_files
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(tmpdir),
            str(output),
            '--concatenation_axes', 'c',
            '--time_tag', 't',
            '--channel_tag', 'channel'
        ])
        
        assert validate_zarr_exists(output)
        shape = get_base_array_shape(output)
        
        # Should have 3 channels
        assert shape[0] == 3  # C dimension
    
    def test_channel_concat_numerical_with_override(self, aggregative_channel_numerical_files, tmp_path):
        """Test numerical channels with name override."""
        tmpdir, files = aggregative_channel_numerical_files
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(tmpdir),
            str(output),
            '--concatenation_axes', 'c',
            '--time_tag', 't',
            '--channel_tag', 'channel',
            '--override_channel_names', 'True'
        ])
        
        assert validate_zarr_exists(output)
        # Should create channels, possibly with numerical indices in names
        validate_channel_metadata(output, expected_n_channels=3)


class TestMultiAxisConcatenation:
    """Tests for multi-axis concatenation (Z and C together)."""
    
    def test_zc_concat(self, aggregative_zc_concat_files, tmp_path):
        """Test concatenating both Z and C axes from separate files."""
        tmpdir, files = aggregative_zc_concat_files
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(tmpdir),
            str(output),
            '--concatenation_axes', 'zc',
            '--z_tag', 'z',
            '--channel_tag', 'c_'
        ])
        
        assert validate_zarr_exists(output)
        shape = get_base_array_shape(output)
        
        # Should have concatenated Z slices and channels
        # Expected: 3 Z × 2 C × 128 × 128 = shape (6, 128, 128) or similar
        # depending on organization
        assert len(shape) >= 3


class TestTagFiltering:
    """Tests for tag-based file matching and filtering."""
    
    def test_tag_filtering_with_exclude(self, aggregative_channel_categorical_files, tmp_path):
        """Test that only files matching tags are included."""
        tmpdir, files = aggregative_channel_categorical_files
        output = tmp_path / "output.zarr"
        
        # Include only gfp, exclude mcherry
        run_eubi_command([
            str(tmpdir),
            str(output),
            '--concatenation_axes', 'c',
            '--time_tag', 't',
            '--channel_tag', 'gfp',
            '--excludes', 'mcherry'
        ])
        
        assert validate_zarr_exists(output)
        # Should only have 1 channel (gfp)
        validate_channel_metadata(output, expected_n_channels=1)
    
    def test_partial_tag_match(self, aggregative_channel_categorical_files, tmp_path):
        """Test that tag matching uses substring containment."""
        tmpdir, files = aggregative_channel_categorical_files
        output = tmp_path / "output.zarr"
        
        # Tag "gfp" should match files containing "gfp"
        run_eubi_command([
            str(tmpdir),
            str(output),
            '--concatenation_axes', 'c',
            '--channel_tag', 'gfp'
        ])
        
        assert validate_zarr_exists(output)


class TestNoMatchingTags:
    """Tests for error handling when tags don't match any files."""
    
    def test_tag_no_matches_fails(self, aggregative_channel_categorical_files, tmp_path):
        """Test that conversion fails with explicit error when tag matches no files."""
        tmpdir, files = aggregative_channel_categorical_files
        output = tmp_path / "output.zarr"
        
        # Use tag that matches nothing
        import sys
        eubi_path = Path(sys.executable).parent / 'eubi'
        result = subprocess.run(
            [str(eubi_path), 'to_zarr',
             str(tmpdir), str(output),
             '--concatenation_axes', 'c',
             '--time_tag', 't',
             '--channel_tag', 'nonexistent'],
            capture_output=True,
            text=True
        )
        
        # Must fail with explicit error message
        assert result.returncode != 0, f"Expected failure but got success. stderr: {result.stderr}"
        # Verify error message is explicit about tag mismatch
        assert 'does not match any files' in result.stderr, f"Expected error about tag not matching files. Got: {result.stderr}"


class TestOMEMetadataMerge:
    """Tests for merging OME-TIFF metadata in aggregative mode."""
    
    def test_ome_merge_channels_from_multiple_files(self, aggregative_ome_channel_merge_files, tmp_path):
        """Test merging channel metadata from multiple OME-TIFF files."""
        tmpdir, files = aggregative_ome_channel_merge_files
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(tmpdir),
            str(output),
            '--concatenation_axes', 't',
            '--time_tag', 't'
        ])
        
        assert validate_zarr_exists(output)
        # Should have 2 channels from OME metadata
        validate_channel_metadata(output, expected_n_channels=2)
    
    def test_ome_metadata_preserved_with_time_concat(self, aggregative_ome_channel_merge_files, tmp_path):
        """Test that OME channel metadata is preserved when concatenating over time."""
        tmpdir, files = aggregative_ome_channel_merge_files
        output = tmp_path / "output.zarr"
        
        run_eubi_command([
            str(tmpdir),
            str(output),
            '--concatenation_axes', 't',
            '--time_tag', 't'
        ])
        
        assert validate_zarr_exists(output)
        # Verify structure is valid
        validate_multiscale_metadata(output)
        # Should have preserved channel metadata
        validate_channel_metadata(output, expected_n_channels=2)
