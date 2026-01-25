"""
Tests for OME-TIFF channel metadata parsing and handling.

Covers reading channel names/colors from OME-TIFF files, merging in aggregative mode,
and overriding channel names with tag-based names.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.validation_utils import (
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


class TestOMEMetadataReading:
    """Tests for reading OME-TIFF channel metadata."""
    
    def test_read_ome_channel_names(self, ome_tiff_3ch, tmp_path):
        """Test that channel names are read from OME-TIFF metadata."""
        tiff_path, expected_names, expected_colors = ome_tiff_3ch
        output = tmp_path / "output.zarr"
        
        run_eubi_command([str(tiff_path), str(output)])
        
        assert validate_zarr_exists(output)
        # Validate channel metadata was preserved
        validate_channel_metadata(
            output,
            expected_n_channels=3,
            expected_labels=expected_names
        )
    
    def test_read_ome_channel_colors(self, ome_tiff_3ch, tmp_path):
        """Test that channel colors are read from OME-TIFF metadata."""
        tiff_path, expected_names, expected_colors = ome_tiff_3ch
        output = tmp_path / "output.zarr"
        
        run_eubi_command([str(tiff_path), str(output)])
        
        assert validate_zarr_exists(output)
        # Validate channel colors were preserved
        validate_channel_metadata(
            output,
            expected_n_channels=3,
            expected_colors=expected_colors
        )
    
    def test_read_categorical_channel_names(self, ome_tiff_2ch_categorical, tmp_path):
        """Test reading categorical channel names (gfp, mcherry)."""
        tiff_path, expected_names, expected_colors = ome_tiff_2ch_categorical
        output = tmp_path / "output.zarr"
        
        run_eubi_command([str(tiff_path), str(output)])
        
        assert validate_zarr_exists(output)
        validate_channel_metadata(
            output,
            expected_n_channels=2,
            expected_labels=expected_names,
            expected_colors=expected_colors
        )
    
    def test_single_channel_metadata(self, ome_tiff_single_ch, tmp_path):
        """Test single-channel OME-TIFF metadata preservation."""
        tiff_path, expected_names, expected_colors = ome_tiff_single_ch
        output = tmp_path / "output.zarr"
        
        run_eubi_command([str(tiff_path), str(output)])
        
        assert validate_zarr_exists(output)
        validate_channel_metadata(
            output,
            expected_n_channels=1,
            expected_labels=expected_names,
            expected_colors=expected_colors
        )


class TestChannelMetadataInAggregative:
    """Tests for channel metadata handling in aggregative conversions."""
    
    def test_merge_channels_from_multiple_omes(self, aggregative_ome_channel_merge_files, tmp_path):
        """Test merging channel metadata from multiple OME-TIFF files in time axis."""
        tmpdir, files = aggregative_ome_channel_merge_files
        output = tmp_path / "output.zarr"
        
        # Convert with time concatenation (files named ome_t01, ome_t02 use 't' tag)
        run_eubi_command([
            str(tmpdir),
            str(output),
            '--concatenation_axes', 't',
            '--time_tag', 't'
        ])
        
        assert validate_zarr_exists(output)
        # Should have 2 channels (merged from both files)
        validate_channel_metadata(output, expected_n_channels=2)
    
    def test_override_channel_names_with_tag(self, aggregative_channel_categorical_files, tmp_path):
        """Test overriding channel names with tag-based names."""
        tmpdir, files = aggregative_channel_categorical_files
        output = tmp_path / "output.zarr"
        
        # Concatenate channels with categorical tags, override names with tag values
        # Must provide both time_tag and channel_tag for proper grouping by timepoint
        run_eubi_command([
            str(tmpdir),
            str(output),
            '--concatenation_axes', 'c',
            '--time_tag', 't',
            '--channel_tag', 'gfp,mcherry',
            '--override_channel_names', 'True'
        ])
        
        assert validate_zarr_exists(output)
        # Channel names should be overridden with the tag values
        validate_channel_metadata(
            output,
            expected_n_channels=2,
            expected_labels=['gfp', 'mcherry']
        )
    
    def test_override_with_numerical_channel_tags(self, aggregative_channel_numerical_files, tmp_path):
        """Test overriding with numerical channel indices."""
        tmpdir, files = aggregative_channel_numerical_files
        output = tmp_path / "output.zarr"
        
        # Use numerical channel tags
        # Must provide both time_tag and channel_tag for proper grouping by timepoint
        run_eubi_command([
            str(tmpdir),
            str(output),
            '--concatenation_axes', 'c',
            '--time_tag', 't',
            '--channel_tag', 'channel',
            '--override_channel_names', 'True'
        ])
        
        assert validate_zarr_exists(output)
        # Should have 3 channels with numerical indices in names
        validate_channel_metadata(output, expected_n_channels=3)


class TestChannelMetadataPreservation:
    """Tests ensuring channel metadata is correctly preserved through conversion."""
    
    def test_5d_ome_metadata_preserved(self, ome_tiff_tczyx, tmp_path):
        """Test that 5D OME-TIFF channel metadata is preserved."""
        tiff_path, expected_names, expected_colors = ome_tiff_tczyx
        output = tmp_path / "output.zarr"
        
        run_eubi_command([str(tiff_path), str(output)])
        
        assert validate_zarr_exists(output)
        validate_channel_metadata(
            output,
            expected_n_channels=2,
            expected_labels=expected_names,
            expected_colors=expected_colors
        )
    
    def test_metadata_survives_downscaling(self, ome_tiff_3ch, tmp_path):
        """Test that channel metadata survives pyramid downscaling."""
        tiff_path, expected_names, expected_colors = ome_tiff_3ch
        output = tmp_path / "output.zarr"
        
        # Convert with multiple downscaling layers
        run_eubi_command([
            str(tiff_path),
            str(output),
            '--n_layers', '3'
        ])
        
        assert validate_zarr_exists(output)
        # Metadata should be at zarr group level, not per-resolution
        validate_channel_metadata(
            output,
            expected_n_channels=3,
            expected_labels=expected_names,
            expected_colors=expected_colors
        )
    
    def test_metadata_with_custom_scales(self, ome_tiff_3ch, tmp_path):
        """Test channel metadata preservation with custom pixel scales."""
        tiff_path, expected_names, expected_colors = ome_tiff_3ch
        output = tmp_path / "output.zarr"
        
        # Modify pixel scales but preserve channel metadata
        run_eubi_command([
            str(tiff_path),
            str(output),
            '--x_scale', '0.25',
            '--y_scale', '0.25'
        ])
        
        assert validate_zarr_exists(output)
        validate_channel_metadata(
            output,
            expected_n_channels=3,
            expected_labels=expected_names
        )
