# -*- coding: utf-8 -*-
"""
@author: bugra

Setup configuration for EuBI-Bridge with platform-specific JDK handling.

JDK binaries are not bundled in the package but downloaded from GitHub at install time.
This keeps PyPI wheels small while maintaining offline availability on HPC clusters.
"""

import setuptools
import os
import platform
import sys
import urllib.request
import tarfile
import shutil
from pathlib import Path

# GitHub repository details for JDK downloads
GITHUB_REPO = "Euro-BioImaging/EuBI-Bridge"
GITHUB_BRANCH = "bf_bundled"  # Change to "main" for production release
GITHUB_RAW_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}"
JDK_BASE_PATH = Path(__file__).parent / "bioformats" / "jdk"


def get_platform_identifier():
    """
    Determine the platform identifier for JDK download.
    
    Returns
    -------
    str
        Platform identifier: 'darwin', 'linux', or 'win32'
    """
    system = platform.system()
    
    if system == 'Darwin':
        return 'darwin'
    elif system == 'Linux':
        return 'linux'
    elif system == 'Windows':
        return 'win32'
    else:
        raise RuntimeError(
            f"Unsupported platform: {system}. "
            "EuBI-Bridge is only supported on macOS (Darwin), Linux, and Windows."
        )


def download_and_extract_jdk():
    """
    Download the platform-specific JDK from GitHub and extract it locally.
    
    This function is called during setup to prepare the JDK for the current platform.
    The JDK MUST be downloaded and extracted at installation time.
    
    This is CRITICAL for HPC environments where runtime downloads are unstable.
    
    Raises
    ------
    RuntimeError
        If download or extraction fails (installation will fail)
    """
    platform_id = get_platform_identifier()
    jdk_platform_path = JDK_BASE_PATH / platform_id
    
    # Check if JDK already exists locally
    if jdk_platform_path.exists() and list(jdk_platform_path.glob('**/*')):
        print(f"✓ JDK for {platform_id} already present at {jdk_platform_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"DOWNLOADING JDK FOR {platform_id.upper()}")
    print(f"{'='*70}")
    
    # Create directory structure if needed
    JDK_BASE_PATH.mkdir(parents=True, exist_ok=True)
    
    # Download from GitHub
    jdk_tar_name = f"jdk_{platform_id}.tar.gz"
    github_url = f"{GITHUB_RAW_URL}/bioformats/jdk/{platform_id}/{jdk_tar_name}"
    
    print(f"GitHub URL: {github_url}")
    
    import tempfile
    import os
    
    # Download to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tar.gz') as tmp:
        tmp_path = tmp.name
    
    try:
        print(f"Downloading JDK (this may take a minute)...")
        urllib.request.urlretrieve(github_url, tmp_path)
        print(f"✓ Downloaded JDK to {tmp_path}")
        
        # Extract to target directory
        print(f"Extracting JDK...")
        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(path=jdk_platform_path.parent)
        print(f"✓ Extracted JDK to {jdk_platform_path}")
        print(f"{'='*70}\n")
        
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(
            f"\n{'='*70}\n"
            f"CRITICAL: JDK DOWNLOAD FAILED\n"
            f"{'='*70}\n"
            f"Platform: {platform_id}\n"
            f"URL: {github_url}\n"
            f"Error: {e}\n\n"
            f"INSTALLATION FAILED: JDK must be downloaded at installation time.\n"
            f"This is required for HPC and production environments.\n\n"
            f"Solutions:\n"
            f"1. Check your internet connection\n"
            f"2. Verify GitHub is accessible\n"
            f"3. For offline installation: Pre-download from {github_url}\n"
            f"   and place at: {jdk_platform_path}\n"
            f"4. If using git clone, ensure: git lfs pull\n"
            f"{'='*70}\n"
        ) from e
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def get_requirements():
    """Get requirements separated by use case."""
    core_requires = [
        "aicspylibczi>=0.0.0",
        "asciitree>=0.3.3",
        "bfio>=0.0.0",
        "bioformats_jar>=0.0.0",
        "bioio-base>=0.0.0",
        "bioio-bioformats==1.1.0",
        "bioio-czi==2.1.0",
        "bioio-imageio==1.1.0",
        "bioio-lif==1.1.0",
        "bioio-nd2==1.1.0",
        "bioio-ome-tiff-fork-by-bugra==0.0.1b2",
        "bioio-tifffile-fork-by-bugra>=0.0.1b2",
        "cmake==4.0.2",
        "dask>=2024.12.1",
        "dask-jobqueue>=0.0.0",
        "distributed>=2024.12.1",
        "elementpath==5.0.1",
        "fasteners==0.19",
        "imageio==2.27.0",
        "imageio-ffmpeg==0.6.0",
        "install-jdk",
        "lz4>=4.4.4",
        "natsort>=0.0.0",
        "nd2>=0.0.0",
        "numpy>=2.3.2",
        "pydantic>=2.11.7",
        "pylibczirw>=0.0.0",
        "readlif==0.6.5",
        "s3fs>=0.0.0",
        "scipy>=1.8",
        "tensorstore>=0.0.0",
        "tifffile>=2025.5.21",
        "validators==0.35.0",
        "xarray>=0.0.0",
        "xmlschema>=0.0.0",
        "xmltodict==0.14.2",
        "zarr>=3.0",
        "zstandard>=0.0.0",
        "blosc2>=3.7.1",
        "aiofiles>=24.1.0",
        "psutil>=7.0.0",
        "rich>=14.1.0",
        "h5py",
    ]

    extras_require = {
        "cli": [
            "fire>=0.0.0",
        ],
        "gui": [
            "fire>=0.0.0",
            "streamlit>=1.28.0",
            "matplotlib>=3.5.0",
        ],
        "all": [
            "fire>=0.0.0",
            "streamlit>=1.28.0",
            "matplotlib>=3.5.0",
        ],
    }

    # Optionally still try to read from requirements.txt if it exists
    # if os.path.exists('../requirements.txt'):
    #     with open('../requirements.txt', encoding='utf-8') as f:
    #         requirements = [
    #             line.strip() for line in f
    #             if line.strip() and not line.startswith('#')
    #         ]
    return core_requires, extras_require


def readme():
    """Read the README file."""
    for filename in ['README.md', 'README.rst', 'README.txt']:
        if os.path.exists(filename):
            with open(filename, encoding='utf-8') as f:
                return f.read()
    return ""


# Prepare JDK during setup - THIS MUST SUCCEED
# Installation fails if JDK cannot be downloaded
download_and_extract_jdk()


setuptools.setup(
    name='eubi_bridge',
    version='0.1.0b3',
    author='Bugra Özdemir',
    author_email='bugraa.ozdemir@gmail.com',
    description='A package for converting datasets to OME-Zarr format.',
    long_description=readme(),
    long_description_content_type="text/markdown",
    url='https://github.com/Euro-BioImaging/EuBI-Bridge',
    license='MIT',
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={
        "eubi_bridge": [
            "bioformats/*.jar",
            "bioformats/*.xml",
            "bioformats/*.txt",
            "bioformats/jdk/**/*",
        ],
    },
    install_requires=get_requirements()[0],
    extras_require=get_requirements()[1],
    python_requires='>=3.11,<3.13',
    entry_points={
        'console_scripts': [
            "eubi = eubi_bridge.cli:main",
            "eubi-gui = eubi_bridge.app:main"
        ]
    },
)
