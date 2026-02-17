"""
Comprehensive compression codec tests for Zarr v2 and v3.

Tests various compression codecs with different parameters:
- Zarr v2: blosc (with cname: lz4, lz4hc, blosclz, snappy, zlib, zstd), zstd, gzip, lz4, bz2, lzma, none
  * Blosc shuffle modes: 0=noshuffle, 1=shuffle, 2=bitshuffle
  * GZip levels: 0-9 (default 9)
  * Zstd levels: -131072 to 22 (default 0)
  
- Zarr v3: blosc (with cname: lz4, lz4hc, blosclz, zlib, zstd, snappy), zstd, gzip, none
  * Blosc shuffle modes: enum (noshuffle, shuffle, bitshuffle)
  * GZip level: 0-9 (default 5, different from v2!)
  * Zstd level: -131072 to 22 (default 0)

Key Differences:
- V2-only: lz4 (direct), bz2, lzma
- V3-only: TransposeCodec, ShardingCodec, Crc32cCodec
- Shared: blosc, gzip, zstd (but with parameter/type differences)
- Blosc in v3: Uses enums for cname and shuffle instead of strings/ints
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

import pytest
import numpy as np

from tests.validation_utils import (
    validate_zarr_exists,
    validate_zarr_format,
    validate_compression,
    get_actual_zarr_path,
    compare_pixel_data,
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


class TestCompressionZarrV2:
    """Tests for Zarr v2 compression codecs and parameters."""
    
    def test_v2_default_compression(self, imagej_tiff_czyx, tmp_path):
        """Test Zarr v2 with default compression."""
        output = tmp_path / "output_v2_default.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--zarr_format', '2'
        ])
        
        assert validate_zarr_exists(output)
        assert validate_zarr_format(output) == 2
    
    def test_v2_blosc_default(self, imagej_tiff_czyx, tmp_path):
        """Test Zarr v2 with Blosc compression (default settings)."""
        output = tmp_path / "output_v2_blosc.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--zarr_format', '2',
            '--compressor', 'blosc'
        ])
        
        assert validate_zarr_exists(output)
        assert validate_zarr_format(output) == 2
        validate_compression(output, 'blosc')
    
    @pytest.mark.parametrize("clevel", [1, 5, 9])
    def test_v2_blosc_compression_levels(self, imagej_tiff_czyx, tmp_path, clevel):
        """Test Zarr v2 Blosc with different compression levels."""
        output = tmp_path / f"output_v2_blosc_l{clevel}.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--zarr_format', '2',
            '--compressor', 'blosc',
            '--compressor_params', f'{{clevel:{clevel}}}'
        ])
        
        assert validate_zarr_exists(output)
        assert validate_zarr_format(output) == 2
        validate_compression(output, 'blosc')
    
    @pytest.mark.parametrize("shuffle", [0, 1, 2])
    def test_v2_blosc_shuffle_modes(self, imagej_tiff_czyx, tmp_path, shuffle):
        """Test Zarr v2 Blosc with different shuffle modes."""
        shuffle_names = {0: 'noshuffle', 1: 'shuffle', 2: 'bitshuffle'}
        output = tmp_path / f"output_v2_blosc_shuffle{shuffle}.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--zarr_format', '2',
            '--compressor', 'blosc',
            '--compressor_params', f'{{shuffle:{shuffle}}}'
        ])
        
        assert validate_zarr_exists(output)
        assert validate_zarr_format(output) == 2
        validate_compression(output, 'blosc')
    
    @pytest.mark.parametrize("cname", ['lz4', 'zstd', 'zlib', 'snappy', 'blosclz'])
    def test_v2_blosc_inner_codecs(self, imagej_tiff_czyx, tmp_path, cname):
        """Test Zarr v2 Blosc with different inner compression libraries.
        
        Note: 'snappy' is v2-specific (not supported in v3 BloscCodec)
        """
        output = tmp_path / f"output_v2_blosc_{cname}.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--zarr_format', '2',
            '--compressor', 'blosc',
            '--compressor_params', f'{{cname:{cname}}}'
        ])
        
        assert validate_zarr_exists(output)
        assert validate_zarr_format(output) == 2
        validate_compression(output, 'blosc')
    
    def test_v2_zstd_default(self, imagej_tiff_czyx, tmp_path):
        """Test Zarr v2 with Zstd compression."""
        output = tmp_path / "output_v2_zstd.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--zarr_format', '2',
            '--compressor', 'zstd'
        ])
        
        assert validate_zarr_exists(output)
        assert validate_zarr_format(output) == 2
        validate_compression(output, 'zstd')
    
    @pytest.mark.parametrize("level", [1, 5, 10, 15])
    def test_v2_zstd_compression_levels(self, imagej_tiff_czyx, tmp_path, level):
        """Test Zarr v2 Zstd with different compression levels."""
        output = tmp_path / f"output_v2_zstd_l{level}.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--zarr_format', '2',
            '--compressor', 'zstd',
            '--compressor_params', f'{{level:{level}}}'
        ])
        
        assert validate_zarr_exists(output)
        assert validate_zarr_format(output) == 2
        validate_compression(output, 'zstd')
    
    def test_v2_gzip(self, imagej_tiff_czyx, tmp_path):
        """Test Zarr v2 with GZip compression."""
        output = tmp_path / "output_v2_gzip.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--zarr_format', '2',
            '--compressor', 'gzip'
        ])
        
        assert validate_zarr_exists(output)
        assert validate_zarr_format(output) == 2
        validate_compression(output, 'gzip')
    
    @pytest.mark.parametrize("level", [1, 5, 9])
    def test_v2_gzip_levels(self, imagej_tiff_czyx, tmp_path, level):
        """Test Zarr v2 GZip with different compression levels."""
        output = tmp_path / f"output_v2_gzip_l{level}.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--zarr_format', '2',
            '--compressor', 'gzip',
            '--compressor_params', f'{{level:{level}}}'
        ])
        
        assert validate_zarr_exists(output)
        assert validate_zarr_format(output) == 2
        validate_compression(output, 'gzip')
    
    @pytest.mark.xfail(reason="LZ4 codec not registered in tensorstore backend")
    def test_v2_lz4(self, imagej_tiff_czyx, tmp_path):
        """Test Zarr v2 with LZ4 compression.
        
        Known issue: LZ4 is not registered in tensorstore's zarr driver.
        This is an environment/backend configuration issue, not a code issue.
        """
        output = tmp_path / "output_v2_lz4.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--zarr_format', '2',
            '--compressor', 'lz4'
        ])
        
        assert validate_zarr_exists(output)
        assert validate_zarr_format(output) == 2
        validate_compression(output, 'lz4')
    
    def test_v2_bz2(self, imagej_tiff_czyx, tmp_path):
        """Test Zarr v2 with BZ2 compression."""
        output = tmp_path / "output_v2_bz2.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--zarr_format', '2',
            '--compressor', 'bz2'
        ])
        
        assert validate_zarr_exists(output)
        assert validate_zarr_format(output) == 2
        validate_compression(output, 'bz2')
    
    def test_v2_no_compression(self, imagej_tiff_czyx, tmp_path):
        """Test Zarr v2 with no compression."""
        output = tmp_path / "output_v2_none.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--zarr_format', '2',
            '--compressor', 'none'
        ])
        
        assert validate_zarr_exists(output)
        assert validate_zarr_format(output) == 2
    
    @pytest.mark.xfail(reason="LZMA codec not registered in tensorstore backend")
    def test_v2_lzma(self, imagej_tiff_czyx, tmp_path):
        """Test Zarr v2 with LZMA compression (v2-only, not in v3).
        
        Known issue: LZMA is not registered in tensorstore's zarr driver.
        This is an environment/backend configuration issue, not a code issue.
        """
        output = tmp_path / "output_v2_lzma.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--zarr_format', '2',
            '--compressor', 'lzma'
        ])
        
        assert validate_zarr_exists(output)
        assert validate_zarr_format(output) == 2
        validate_compression(output, 'lzma')
    
    def test_v2_blosc_snappy(self, imagej_tiff_czyx, tmp_path):
        """Test Zarr v2 Blosc with snappy inner codec (v2-only, not guaranteed in v3).
        
        Note: 'snappy' is supported in numcodecs Blosc for v2, but v3 support is uncertain.
        This test documents v2 snappy availability.
        """
        output = tmp_path / "output_v2_blosc_snappy.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--zarr_format', '2',
            '--compressor', 'blosc',
            '--compressor_params', '{cname:snappy}'
        ])
        
        assert validate_zarr_exists(output)
        assert validate_zarr_format(output) == 2
        validate_compression(output, 'blosc')


class TestCompressionZarrV3:
    """Tests for Zarr v3 compression codecs and parameters."""
    
    def test_v3_default_compression(self, imagej_tiff_czyx, tmp_path):
        """Test Zarr v3 with default compression."""
        output = tmp_path / "output_v3_default.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--zarr_format', '3'
        ])
        
        assert validate_zarr_exists(output)
        assert validate_zarr_format(output) == 3
    
    def test_v3_blosc_default(self, imagej_tiff_czyx, tmp_path):
        """Test Zarr v3 with Blosc compression (default settings)."""
        output = tmp_path / "output_v3_blosc.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--zarr_format', '3',
            '--compressor', 'blosc'
        ])
        
        assert validate_zarr_exists(output)
        assert validate_zarr_format(output) == 3
        validate_compression(output, 'blosc')
    
    @pytest.mark.parametrize("clevel", [1, 5, 9])
    def test_v3_blosc_compression_levels(self, imagej_tiff_czyx, tmp_path, clevel):
        """Test Zarr v3 Blosc with different compression levels."""
        output = tmp_path / f"output_v3_blosc_l{clevel}.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--zarr_format', '3',
            '--compressor', 'blosc',
            '--compressor_params', f'{{clevel:{clevel}}}'
        ])
        
        assert validate_zarr_exists(output)
        assert validate_zarr_format(output) == 3
        validate_compression(output, 'blosc')
    
    @pytest.mark.parametrize("shuffle", [0, 1, 2])
    def test_v3_blosc_shuffle_modes(self, imagej_tiff_czyx, tmp_path, shuffle):
        """Test Zarr v3 Blosc with different shuffle modes. System converts int to proper enum."""
        output = tmp_path / f"output_v3_blosc_shuffle{shuffle}.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--zarr_format', '3',
            '--compressor', 'blosc',
            '--compressor_params', f'{{shuffle:{shuffle}}}'
        ])
        
        assert validate_zarr_exists(output)
        assert validate_zarr_format(output) == 3
        validate_compression(output, 'blosc')
    
    @pytest.mark.parametrize("cname", ['lz4', 'lz4hc', 'zstd', 'zlib', 'blosclz'])
    def test_v3_blosc_inner_codecs(self, imagej_tiff_czyx, tmp_path, cname):
        """Test Zarr v3 Blosc with different inner compression libraries.
        
        Zarr v3 BloscCodec supports: lz4, lz4hc, zstd, zlib, blosclz, snappy
        (lz4hc tests codec variant support; snappy can be tested separately if needed)
        """
        output = tmp_path / f"output_v3_blosc_{cname}.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--zarr_format', '3',
            '--compressor', 'blosc',
            '--compressor_params', f'{{cname:{cname}}}'
        ])
        
        assert validate_zarr_exists(output)
        assert validate_zarr_format(output) == 3
        validate_compression(output, 'blosc')
    
    def test_v3_zstd_default(self, imagej_tiff_czyx, tmp_path):
        """Test Zarr v3 with Zstd compression."""
        output = tmp_path / "output_v3_zstd.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--zarr_format', '3',
            '--compressor', 'zstd'
        ])
        
        assert validate_zarr_exists(output)
        assert validate_zarr_format(output) == 3
        validate_compression(output, 'zstd')
    
    @pytest.mark.parametrize("level", [1, 5, 10, 15])
    def test_v3_zstd_compression_levels(self, imagej_tiff_czyx, tmp_path, level):
        """Test Zarr v3 Zstd with different compression levels."""
        output = tmp_path / f"output_v3_zstd_l{level}.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--zarr_format', '3',
            '--compressor', 'zstd',
            '--compressor_params', f'{{level:{level}}}'
        ])
        
        assert validate_zarr_exists(output)
        assert validate_zarr_format(output) == 3
        validate_compression(output, 'zstd')
    
    def test_v3_gzip(self, imagej_tiff_czyx, tmp_path):
        """Test Zarr v3 with GZip compression."""
        output = tmp_path / "output_v3_gzip.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--zarr_format', '3',
            '--compressor', 'gzip'
        ])
        
        assert validate_zarr_exists(output)
        assert validate_zarr_format(output) == 3
        validate_compression(output, 'gzip')
    
    @pytest.mark.parametrize("level", [1, 5, 9])
    def test_v3_gzip_levels(self, imagej_tiff_czyx, tmp_path, level):
        """Test Zarr v3 GZip with different compression levels."""
        output = tmp_path / f"output_v3_gzip_l{level}.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--zarr_format', '3',
            '--compressor', 'gzip',
            '--compressor_params', f'{{level:{level}}}'
        ])
        
        assert validate_zarr_exists(output)
        assert validate_zarr_format(output) == 3
        validate_compression(output, 'gzip')
    
    def test_v3_no_compression(self, imagej_tiff_czyx, tmp_path):
        """Test Zarr v3 with no compression."""
        output = tmp_path / "output_v3_none.zarr"
        
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output),
            '--zarr_format', '3',
            '--compressor', 'none'
        ])
        
        assert validate_zarr_exists(output)
        assert validate_zarr_format(output) == 3


class TestCompressionV2V3Parity:
    """Tests comparing compression behavior between Zarr v2 and v3."""
    
    def test_blosc_v2_vs_v3(self, imagej_tiff_czyx, tmp_path):
        """Test Blosc compression produces valid output for both v2 and v3."""
        output_v2 = tmp_path / "output_v2.zarr"
        output_v3 = tmp_path / "output_v3.zarr"
        
        # Convert to v2 with Blosc
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output_v2),
            '--zarr_format', '2',
            '--compressor', 'blosc',
            '--compressor_params', '{clevel:5}'
        ])
        
        # Convert to v3 with Blosc
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output_v3),
            '--zarr_format', '3',
            '--compressor', 'blosc',
            '--compressor_params', '{clevel:5}'
        ])
        
        assert validate_zarr_exists(output_v2)
        assert validate_zarr_format(output_v2) == 2
        assert validate_zarr_exists(output_v3)
        assert validate_zarr_format(output_v3) == 3
        
        # Both should have valid compression
        validate_compression(output_v2, 'blosc')
        validate_compression(output_v3, 'blosc')
        
        # Data should match between v2 and v3
        from tests.validation_utils import compare_zarr_v2_vs_v3
        compare_zarr_v2_vs_v3(output_v2, output_v3)
    
    def test_zstd_v2_vs_v3(self, imagej_tiff_czyx, tmp_path):
        """Test Zstd compression produces valid output for both v2 and v3."""
        output_v2 = tmp_path / "output_v2.zarr"
        output_v3 = tmp_path / "output_v3.zarr"
        
        # Convert to v2 with Zstd
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output_v2),
            '--zarr_format', '2',
            '--compressor', 'zstd',
            '--compressor_params', '{level:10}'
        ])
        
        # Convert to v3 with Zstd
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output_v3),
            '--zarr_format', '3',
            '--compressor', 'zstd',
            '--compressor_params', '{level:10}'
        ])
        
        assert validate_zarr_exists(output_v2)
        assert validate_zarr_format(output_v2) == 2
        assert validate_zarr_exists(output_v3)
        assert validate_zarr_format(output_v3) == 3
        
        # Both should have valid compression
        validate_compression(output_v2, 'zstd')
        validate_compression(output_v3, 'zstd')
        
        # Data should match between v2 and v3
        from tests.validation_utils import compare_zarr_v2_vs_v3
        compare_zarr_v2_vs_v3(output_v2, output_v3)
    
    def test_gzip_v2_vs_v3(self, imagej_tiff_czyx, tmp_path):
        """Test GZip compression produces valid output for both v2 and v3."""
        output_v2 = tmp_path / "output_v2.zarr"
        output_v3 = tmp_path / "output_v3.zarr"
        
        # Convert to v2 with GZip
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output_v2),
            '--zarr_format', '2',
            '--compressor', 'gzip',
            '--compressor_params', '{level:6}'
        ])
        
        # Convert to v3 with GZip
        run_eubi_command([
            str(imagej_tiff_czyx),
            str(output_v3),
            '--zarr_format', '3',
            '--compressor', 'gzip',
            '--compressor_params', '{level:6}'
        ])
        
        assert validate_zarr_exists(output_v2)
        assert validate_zarr_format(output_v2) == 2
        assert validate_zarr_exists(output_v3)
        assert validate_zarr_format(output_v3) == 3
        
        # Both should have valid compression
        validate_compression(output_v2, 'gzip')
        validate_compression(output_v3, 'gzip')
        
        # Data should match between v2 and v3
        from tests.validation_utils import compare_zarr_v2_vs_v3
        compare_zarr_v2_vs_v3(output_v2, output_v3)
