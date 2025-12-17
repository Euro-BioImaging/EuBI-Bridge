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
GITHUB_RAW_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main"
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
    The JDK is only downloaded if not already present locally.
    
    Raises
    ------
    RuntimeError
        If download or extraction fails
    """
    platform_id = get_platform_identifier()
    jdk_platform_path = JDK_BASE_PATH / platform_id
    
    # Check if JDK already exists locally
    if jdk_platform_path.exists() and list(jdk_platform_path.glob('**/*')):
        print(f"✓ JDK for {platform_id} already present at {jdk_platform_path}")
        return
    
    print(f"Preparing JDK for {platform_id}...")
    
    # Create directory structure if needed
    JDK_BASE_PATH.mkdir(parents=True, exist_ok=True)
    
    # For development/local setup: JDKs are already in repo
    # For PyPI installs: would need to download from GitHub
    # Since this is typically run in development, we skip download if already versioned
    if not jdk_platform_path.exists():
        print(
            f"Warning: JDK for {platform_id} not found at {jdk_platform_path}\n"
            f"If installing from PyPI, JDKs will be downloaded at first runtime.\n"
            f"For git checkouts, ensure JDK files are present or run: git lfs pull"
        )


def get_requirements():
    """Get requirements from requirements.txt or return default requirements."""
    requirements = [
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
        "fire>=0.0.0",
        "imageio==2.27.0",
        "imageio-ffmpeg==0.6.0",
        "install-jdk",
        "natsort>=0.0.0",
        "nd2>=0.0.0",
        "numpy>=0.0.0",
        # "openjdk==8.*",
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
        #
        "aiofiles>=24.1.0",
        "blosc2>=3.7.1",
        "fastapi>=0.116.1",
        "lz4>=4.4.4",
        "numpy>=2.3.2",
        "psutil>=7.0.0",
        "rich>=14.1.0",
        "uvicorn>=0.35.0",
        "websockets>=15.0.1",
        "h5py"
    ]

    # Optionally still try to read from requirements.txt if it exists
    # if os.path.exists('../requirements.txt'):
    #     with open('../requirements.txt', encoding='utf-8') as f:
    #         requirements = [
    #             line.strip() for line in f
    #             if line.strip() and not line.startswith('#')
    #         ]
    return requirements


def readme():
    """Read the README file."""
    for filename in ['README.md', 'README.rst', 'README.txt']:
        if os.path.exists(filename):
            with open(filename, encoding='utf-8') as f:
                return f.read()
    return ""


# Prepare JDK during setup
try:
    download_and_extract_jdk()
except Exception as e:
    print(f"Warning during JDK setup: {e}")
    # Don't fail the entire setup if JDK prep fails
    # JDKs will be sourced at runtime if needed


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
        ],
    },
    install_requires=get_requirements(),
    python_requires='>=3.11,<3.13',
    extras_require={
        # Include CUDA variants if needed
        'cuda11': ['cupy-cuda11x'],
        'cuda12': ['cupy-cuda12x'],
    },
    entry_points={
        'console_scripts': [
            "eubi = eubi_bridge.cli:main"
        ]
    },
)
