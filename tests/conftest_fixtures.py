"""
Pytest fixtures for generating test data (ImageJ TIFF and OME-TIFF files).
These fixtures create various synthetic image datasets on-the-fly for testing.
"""

import numpy as np
import tifffile
from pathlib import Path
import pytest

# === Monkey patch for bioio-ome-tiff-fork-by-bugra ===
# The fork package has a different dist name, but OmeTiffWriter hard-codes 'bioio_ome_tiff'
# in importlib.metadata.version() lookup. Patch it to handle both names gracefully.
import importlib.metadata
_original_version = importlib.metadata.version

def _patched_version(package_name):
    if package_name == 'bioio_ome_tiff':
        try:
            return _original_version('bioio_ome_tiff')
        except importlib.metadata.PackageNotFoundError:
            try:
                return _original_version('bioio-ome-tiff-fork-by-bugra')
            except importlib.metadata.PackageNotFoundError:
                return "0.0.1-fork"
    return _original_version(package_name)

importlib.metadata.version = _patched_version


@pytest.fixture
def tmp_test_data(tmp_path):
    """Fixture that provides a temporary directory for test data."""
    return tmp_path


def create_synthetic_image_zyx(shape=(8, 128, 128), dtype=np.uint8, seed=42):
    """Create synthetic 3D image (ZYX format)."""
    np.random.seed(seed)
    if dtype == np.uint16:
        return (np.random.rand(*shape) * 65535).astype(dtype)
    else:
        return (np.random.rand(*shape) * 255).astype(dtype)


def create_synthetic_image_czyx(shape=(3, 4, 128, 128), dtype=np.uint8, seed=42):
    """Create synthetic 4D image (CZYX format)."""
    np.random.seed(seed)
    if dtype == np.uint16:
        return (np.random.rand(*shape) * 65535).astype(dtype)
    else:
        return (np.random.rand(*shape) * 255).astype(dtype)


def create_synthetic_image_tczyx(shape=(2, 3, 4, 128, 128), dtype=np.uint8, seed=42):
    """Create synthetic 5D image (TCZYX format)."""
    np.random.seed(seed)
    if dtype == np.uint16:
        return (np.random.rand(*shape) * 65535).astype(dtype)
    else:
        return (np.random.rand(*shape) * 255).astype(dtype)


# ============================================================================
# ImageJ TIFF Fixtures
# ============================================================================

@pytest.fixture
def imagej_tiff_zyx(tmp_test_data):
    """Create single ImageJ TIFF with ZYX format."""
    shape = (8, 128, 128)
    img = create_synthetic_image_zyx(shape, dtype=np.uint8)
    
    path = tmp_test_data / "imagej_zyx.tif"
    tifffile.imwrite(
        path,
        img,
        imagej=True,
        resolution=(3.03, 3.03),  # 1/0.33 pixels per micrometer
        metadata={
            'spacing': 0.66,
            'unit': 'um',
            'axes': 'ZYX'
        }
    )
    return path


@pytest.fixture
def imagej_tiff_zyx_uint16(tmp_test_data):
    """Create ImageJ TIFF with ZYX format and uint16 dtype."""
    shape = (8, 128, 128)
    img = create_synthetic_image_zyx(shape, dtype=np.uint16)
    
    path = tmp_test_data / "imagej_zyx_uint16.tif"
    tifffile.imwrite(
        path,
        img,
        imagej=True,
        resolution=(3.03, 3.03),
        metadata={
            'spacing': 0.66,
            'unit': 'um',
            'axes': 'ZYX'
        }
    )
    return path


@pytest.fixture
def imagej_tiff_czyx(tmp_test_data):
    """Create ImageJ TIFF with CZYX format (3 channels, 4 Z-slices)."""
    shape = (3, 4, 256, 256)
    img = create_synthetic_image_czyx(shape, dtype=np.uint8)
    
    path = tmp_test_data / "imagej_czyx.tif"
    tifffile.imwrite(
        path,
        img,
        imagej=True,
        resolution=(3.03, 3.03),
        metadata={
            'spacing': 0.66,
            'unit': 'um',
            'finterval': 0.1,
            'axes': 'ZCYX'
        }
    )
    return path


@pytest.fixture
def imagej_tiff_tczyx(tmp_test_data):
    """Create ImageJ TIFF with TCZYX format."""
    shape = (2, 3, 4, 256, 256)  # T, C, Z, Y, X
    img = create_synthetic_image_tczyx(shape, dtype=np.uint8)
    
    path = tmp_test_data / "imagej_tczyx.tif"
    tifffile.imwrite(
        path,
        img,
        imagej=True,
        resolution=(3.03, 3.03),
        metadata={
            'spacing': 0.66,
            'unit': 'um',
            'finterval': 0.1,
            'fps': 10.0,
            'axes': 'TZCYX'
        }
    )
    return path


# ============================================================================
# OME-TIFF Fixtures (with Channel Metadata)
# ============================================================================

@pytest.fixture
def ome_tiff_3ch(tmp_test_data):
    """Create OME-TIFF with 3 channels (GFP, mCherry, Hoechst)."""
    from bioio_ome_tiff.writers import OmeTiffWriter
    from bioio_base.types import PhysicalPixelSizes
    
    # Create image data: CZYX format
    shape = (3, 4, 128, 128)  # C=3, Z=4, Y=128, X=128
    img = create_synthetic_image_czyx(shape, dtype=np.uint8)
    
    channel_names = ['GFP', 'mCherry', 'Hoechst']
    channel_colors_hex = [0x00FF00, 0xFF0000, 0x0000FF]  # Green, Red, Blue
    # Convert hex to RGB triplets for OmeTiffWriter
    channel_colors = [[((c >> 16) & 0xFF), ((c >> 8) & 0xFF), (c & 0xFF)] for c in channel_colors_hex]
    
    path = tmp_test_data / "ome_3ch.ome.tif"
    
    # Use OmeTiffWriter to properly create OME-TIFF with metadata
    OmeTiffWriter.save(
        img,
        str(path),
        dim_order='CZYX',
        channel_names=channel_names,
        channel_colors=channel_colors,
        physical_pixel_sizes=PhysicalPixelSizes(Z=0.66, Y=0.33, X=0.33),
    )
    
    return path, channel_names, [f"{c:06X}" for c in channel_colors_hex]


@pytest.fixture
def ome_tiff_2ch_categorical(tmp_test_data):
    """Create OME-TIFF with 2 channels using categorical names (gfp, mcherry)."""
    from bioio_ome_tiff.writers import OmeTiffWriter
    from bioio_base.types import PhysicalPixelSizes
    
    shape = (2, 3, 256, 256)  # C=2, Z=3, Y=256, X=256
    img = create_synthetic_image_czyx(shape, dtype=np.uint8)
    
    channel_names = ['gfp', 'mcherry']
    channel_colors_hex = [0x00FF00, 0xFF0000]  # Green, Red
    # Convert hex to RGB triplets for OmeTiffWriter
    channel_colors = [[((c >> 16) & 0xFF), ((c >> 8) & 0xFF), (c & 0xFF)] for c in channel_colors_hex]
    
    path = tmp_test_data / "ome_2ch_categorical.ome.tif"
    
    OmeTiffWriter.save(
        img,
        str(path),
        dim_order='CZYX',
        channel_names=channel_names,
        channel_colors=channel_colors,
        physical_pixel_sizes=PhysicalPixelSizes(Z=0.5, Y=0.25, X=0.25),
    )
    
    return path, channel_names, [f"{c:06X}" for c in channel_colors_hex]


@pytest.fixture
def ome_tiff_single_ch(tmp_test_data):
    """Create single-channel OME-TIFF."""
    from bioio_ome_tiff.writers import OmeTiffWriter
    from bioio_base.types import PhysicalPixelSizes
    
    shape = (1, 5, 256, 256)  # C=1, Z=5, Y=256, X=256
    img = create_synthetic_image_czyx(shape, dtype=np.uint16)
    
    channel_names = ['ProLong']
    channel_colors_hex = [0x808080]  # Gray
    # Convert hex to RGB triplets for OmeTiffWriter
    channel_colors = [[((c >> 16) & 0xFF), ((c >> 8) & 0xFF), (c & 0xFF)] for c in channel_colors_hex]
    
    path = tmp_test_data / "ome_1ch.ome.tif"
    
    OmeTiffWriter.save(
        img,
        str(path),
        dim_order='CZYX',
        channel_names=channel_names,
        channel_colors=channel_colors,
        physical_pixel_sizes=PhysicalPixelSizes(Z=1.0, Y=0.5, X=0.5),
    )
    
    return path, channel_names, [f"{c:06X}" for c in channel_colors_hex]


@pytest.fixture
def ome_tiff_tczyx(tmp_test_data):
    """Create 5D OME-TIFF with time dimension."""
    from bioio_ome_tiff.writers import OmeTiffWriter
    from bioio_base.types import PhysicalPixelSizes
    
    shape = (2, 2, 3, 256, 256)  # T=2, C=2, Z=3, Y=256, X=256
    img = create_synthetic_image_tczyx(shape, dtype=np.uint8)
    
    channel_names = ['GFP', 'Brightfield']
    channel_colors_hex = [0x00FF00, 0xFFFFFF]  # Green, White
    # Convert hex to RGB triplets for OmeTiffWriter
    channel_colors = [[((c >> 16) & 0xFF), ((c >> 8) & 0xFF), (c & 0xFF)] for c in channel_colors_hex]
    
    path = tmp_test_data / "ome_tczyx.ome.tif"
    
    OmeTiffWriter.save(
        img,
        str(path),
        dim_order='TCZYX',
        channel_names=channel_names,
        channel_colors=channel_colors,
        physical_pixel_sizes=PhysicalPixelSizes(Z=0.5, Y=0.2, X=0.2),
    )
    
    return path, channel_names, [f"{c:06X}" for c in channel_colors_hex]


# ============================================================================
# Aggregative Conversion Test Fixtures
# ============================================================================

@pytest.fixture
def aggregative_z_concat_files(tmp_test_data):
    """Create files for Z-axis concatenation (3 Z-slices, 2 timepoints)."""
    files = []
    for t in range(2):
        for z in range(3):
            img = create_synthetic_image_zyx((128, 128), dtype=np.uint8, 
                                           seed=t*10+z)
            filename = f"scan_t{t:03d}_c01_z{z:02d}.tif"
            path = tmp_test_data / filename
            tifffile.imwrite(path, img, imagej=True, resolution=(3.03, 3.03))
            files.append(path)
    return tmp_test_data, files


@pytest.fixture
def aggregative_channel_categorical_files(tmp_test_data):
    """Create files for channel concatenation with categorical names (gfp, mcherry)."""
    files = []
    channels = [
        ('gfp', '00FF00'),      # Green
        ('mcherry', 'FF0000'),  # Red
    ]
    
    for t in range(2):
        for ch_name, ch_color in channels:
            img = create_synthetic_image_zyx((128, 128), dtype=np.uint8,
                                           seed=t*10)
            filename = f"exp_t{t:03d}_{ch_name}.tif"
            path = tmp_test_data / filename
            tifffile.imwrite(path, img, imagej=True, resolution=(3.03, 3.03))
            files.append((path, ch_name, ch_color))
    return tmp_test_data, files


@pytest.fixture
def aggregative_channel_numerical_files(tmp_test_data):
    """Create files for channel concatenation with numerical indices (channel1, channel2, channel3)."""
    files = []
    for t in range(2):
        for ch_idx in range(1, 4):  # channel1, channel2, channel3
            img = create_synthetic_image_zyx((128, 128), dtype=np.uint8,
                                           seed=t*10+ch_idx)
            filename = f"scan_t{t:03d}_channel{ch_idx}.tif"
            path = tmp_test_data / filename
            tifffile.imwrite(path, img, imagej=True, resolution=(3.03, 3.03))
            files.append((path, f"channel{ch_idx}"))
    return tmp_test_data, files


@pytest.fixture
def aggregative_zc_concat_files(tmp_test_data):
    """Create files for multi-axis concatenation (Z + Channel)."""
    files = []
    for z in range(3):
        for ch_name in ['gfp', 'mcherry']:
            img = create_synthetic_image_zyx((128, 128), dtype=np.uint8,
                                           seed=z*10)
            filename = f"data_c_{ch_name}_z{z:02d}.tif"
            path = tmp_test_data / filename
            tifffile.imwrite(path, img, imagej=True, resolution=(3.03, 3.03))
            files.append(path)
    return tmp_test_data, files


@pytest.fixture
def aggregative_ome_channel_merge_files(tmp_test_data):
    """Create OME-TIFF files for testing channel metadata merging in aggregative mode."""
    from bioio_ome_tiff.writers import OmeTiffWriter
    from bioio_base.types import PhysicalPixelSizes
    
    files = []
    
    # File 1: 2 channels (GFP, mCherry) at timepoint 1
    shape1 = (2, 3, 128, 128)  # CZYX
    img1 = create_synthetic_image_czyx(shape1, dtype=np.uint8, seed=1)
    
    channel_names1 = ['GFP', 'mCherry']
    channel_colors1_hex = [0x00FF00, 0xFF0000]
    channel_colors1 = [[(c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF] for c in channel_colors1_hex]
    
    path1 = tmp_test_data / "ome_t01.ome.tif"
    OmeTiffWriter.save(
        img1, str(path1),
        dim_order='CZYX',
        channel_names=channel_names1,
        channel_colors=channel_colors1,
        physical_pixel_sizes=PhysicalPixelSizes(Z=0.5, Y=0.33, X=0.33),
    )
    files.append(path1)
    
    # File 2: 2 channels (GFP, mCherry) at timepoint 2
    img2 = create_synthetic_image_czyx(shape1, dtype=np.uint8, seed=2)
    
    path2 = tmp_test_data / "ome_t02.ome.tif"
    OmeTiffWriter.save(
        img2, str(path2),
        dim_order='CZYX',
        channel_names=channel_names1,
        channel_colors=channel_colors1,
        physical_pixel_sizes=PhysicalPixelSizes(Z=0.5, Y=0.33, X=0.33),
    )
    files.append(path2)
    
    return tmp_test_data, files
