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
# JDK archives are hosted on GitHub Releases due to their size (170-185 MB each)
GITHUB_REPO = "Euro-BioImaging/EuBI-Bridge"
GITHUB_RELEASE_TAG = "jdk-v11"  # Release tag containing JDK archives
GITHUB_RELEASES_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{GITHUB_RELEASE_TAG}"
# JDK must be inside eubi_bridge package to be included in the wheel
JDK_BASE_PATH = Path(__file__).parent / "eubi_bridge" / "bioformats" / "jdk"


def get_platform_identifier():
    """
    Determine the platform identifier and architecture for JDK download.
    
    Returns
    -------
    tuple[str, str]
        (platform_id, jdk_archive_name) tuple
        platform_id: 'darwin', 'linux', or 'win32'
        jdk_archive_name: filename of the JDK tar.gz archive
    """
    system = platform.system()
    machine = platform.machine().lower()
    
    if system == 'Darwin':
        # macOS: distinguish between Apple Silicon (arm64) and Intel (x86_64)
        if machine in ('arm64', 'aarch64'):
            return 'darwin', 'jdk_darwin_arm64.tar.gz'
        else:
            return 'darwin', 'jdk_darwin_x86_64.tar.gz'
    elif system == 'Linux':
        return 'linux', 'jdk_linux.tar.gz'
    elif system == 'Windows':
        return 'win32', 'jdk_win32.tar.gz'
    else:
        raise RuntimeError(
            f"Unsupported platform: {system}. "
            "EuBI-Bridge is only supported on macOS (Darwin), Linux, and Windows."
        )


def download_and_extract_jdk():
    """
    Download the platform-specific JDK from GitHub Releases and extract it.
    
    This function is called during setup to prepare the JDK for the current platform.
    The JDK MUST be downloaded and extracted at installation time.
    
    This is CRITICAL for HPC environments where runtime downloads are unstable.
    
    JDK archives are hosted on GitHub Releases:
    - jdk_darwin_arm64.tar.gz (~178 MB) - macOS Apple Silicon
    - jdk_darwin_x86_64.tar.gz (~172 MB) - macOS Intel
    - jdk_linux.tar.gz (~184 MB) - Linux x86_64
    - jdk_win32.tar.gz - Windows (if available)
    
    Raises
    ------
    RuntimeError
        If download or extraction fails (installation will fail)
    """
    platform_id, jdk_archive_name = get_platform_identifier()
    jdk_platform_path = JDK_BASE_PATH / platform_id
    
    # Architecture-specific subdirectory for darwin
    machine = platform.machine().lower()
    if platform_id == 'darwin':
        if machine in ('arm64', 'aarch64'):
            jdk_platform_path = JDK_BASE_PATH / platform_id / 'arm64'
        else:
            jdk_platform_path = JDK_BASE_PATH / platform_id / 'x86_64'
    
    # Check if JDK already exists locally
    if jdk_platform_path.exists() and list(jdk_platform_path.glob('**/*')):
        print(f"✓ JDK for {platform_id} already present at {jdk_platform_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"DOWNLOADING JDK FOR {platform_id.upper()}")
    print(f"{'='*70}")
    
    # Create directory structure if needed
    jdk_platform_path.mkdir(parents=True, exist_ok=True)
    
    # Download from GitHub Releases
    github_url = f"{GITHUB_RELEASES_URL}/{jdk_archive_name}"
    
    print(f"Platform: {platform_id}")
    print(f"Architecture: {machine}")
    print(f"Archive: {jdk_archive_name}")
    print(f"URL: {github_url}")
    
    import tempfile
    import os
    
    # Download to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tar.gz') as tmp:
        tmp_path = tmp.name
    
    try:
        print(f"Downloading JDK (~180 MB, this may take a minute)...")
        urllib.request.urlretrieve(github_url, tmp_path)
        
        # Check file size to ensure download succeeded
        file_size = os.path.getsize(tmp_path)
        if file_size < 1000000:  # Less than 1 MB = likely an error page
            raise RuntimeError(f"Downloaded file too small ({file_size} bytes), likely 404 or error page")
        
        print(f"✓ Downloaded JDK ({file_size // 1024 // 1024} MB)")
        
        # Extract to target directory
        print(f"Extracting JDK...")
        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(path=jdk_platform_path)
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
            f"Architecture: {machine}\n"
            f"Archive: {jdk_archive_name}\n"
            f"URL: {github_url}\n"
            f"Error: {e}\n\n"
            f"INSTALLATION FAILED: JDK must be downloaded at installation time.\n"
            f"This is required for HPC and production environments.\n\n"
            f"Solutions:\n"
            f"1. Check your internet connection\n"
            f"2. Verify GitHub is accessible\n"
            f"3. Create the GitHub Release '{GITHUB_RELEASE_TAG}' with JDK archives\n"
            f"4. For offline installation: Manually download from GitHub Releases\n"
            f"   and extract to: {jdk_platform_path}\n"
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
