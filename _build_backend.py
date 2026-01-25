"""
Custom build backend that downloads JDK before building the wheel.

This ensures JDK is included in the wheel while keeping upload size under 100MB.
"""

import sys
import platform
import urllib.request
import tarfile
import tempfile
import shutil
import os
from pathlib import Path
from setuptools.build_meta import *  # noqa: F401, F403
from setuptools.build_meta import build_wheel as _build_wheel
from setuptools.build_meta import build_sdist as _build_sdist


# GitHub repository details for JDK downloads
GITHUB_REPO = "Euro-BioImaging/EuBI-Bridge"
GITHUB_RELEASE_TAG = "jdk-v11"
GITHUB_RELEASES_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{GITHUB_RELEASE_TAG}"
JDK_BASE_PATH = Path(__file__).parent / "eubi_bridge" / "bioformats" / "jdk"


def get_platform_identifier():
    """Determine platform and architecture for JDK download."""
    system = platform.system()
    machine = platform.machine().lower()
    
    if system == 'Darwin':
        if machine in ('arm64', 'aarch64'):
            return 'darwin', 'jdk_darwin_arm64.tar.gz', 'arm64'
        else:
            return 'darwin', 'jdk_darwin_x86_64.tar.gz', 'x86_64'
    elif system == 'Linux':
        return 'linux', 'jdk_linux.tar.gz', None
    elif system == 'Windows':
        return 'win32', 'jdk_win32.tar.gz', None
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


def download_and_extract_jdk():
    """Download JDK from GitHub Releases and extract it."""
    platform_id, jdk_archive_name, arch = get_platform_identifier()
    
    if arch:
        jdk_platform_path = JDK_BASE_PATH / platform_id / arch
    else:
        jdk_platform_path = JDK_BASE_PATH / platform_id
    
    # Check if JDK already exists
    if jdk_platform_path.exists() and list(jdk_platform_path.glob('**/*')):
        print(f"[OK] JDK for {platform_id} already present")
        return
    
    print(f"\n{'='*70}")
    print(f"DOWNLOADING JDK FOR {platform_id.upper()}")
    print(f"{'='*70}")
    
    jdk_platform_path.mkdir(parents=True, exist_ok=True)
    github_url = f"{GITHUB_RELEASES_URL}/{jdk_archive_name}"
    
    print(f"URL: {github_url}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tar.gz') as tmp:
        tmp_path = tmp.name
    
    try:
        print(f"Downloading JDK (~180 MB, this may take a minute)...")
        urllib.request.urlretrieve(github_url, tmp_path)
        
        file_size = os.path.getsize(tmp_path)
        if file_size < 1000000:
            raise RuntimeError(f"Downloaded file too small ({file_size} bytes), likely 404")
        
        print(f"[OK] Downloaded JDK ({file_size // 1024 // 1024} MB)")
        print(f"Extracting JDK...")
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(tmp_path, "r:gz") as tar:
                tar.extractall(path=tmpdir)
            
            # Handle tar files with top-level 'jdk' directory
            extracted_contents = list(Path(tmpdir).iterdir())
            
            if len(extracted_contents) == 1 and extracted_contents[0].name == 'jdk':
                # Tar has 'jdk/' as top-level, extract its contents directly
                jdk_root = extracted_contents[0]
                for item in jdk_root.iterdir():
                    dest = jdk_platform_path / item.name
                    if dest.exists():
                        shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
                    shutil.move(str(item), str(dest))
            else:
                # Tar contents go directly to jdk_platform_path
                for item in extracted_contents:
                    dest = jdk_platform_path / item.name
                    if dest.exists():
                        shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
                    shutil.move(str(item), str(dest))
        
        print(f"[OK] Extracted JDK")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR: JDK DOWNLOAD FAILED")
        print(f"{'='*70}")
        print(f"URL: {github_url}")
        print(f"Error: {e}")
        print(f"\nBuild cannot continue without JDK.")
        print(f"{'='*70}\n")
        raise
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build wheel after downloading JDK."""
    print("\n" + "="*70)
    print("BUILD BACKEND: Preparing JDK")
    print("="*70)
    download_and_extract_jdk()
    
    print("BUILD BACKEND: Building wheel with setuptools")
    return _build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    """Build source distribution WITHOUT JDK (will be downloaded by backend when sdist is installed)."""
    print("\n" + "="*70)
    print("BUILD BACKEND: Building sdist (JDK NOT included - will be downloaded on install)")
    print("="*70)
    # DO NOT download JDK for sdist - let the user's build process download it
    # when they install from this sdist
    
    print("BUILD BACKEND: Building sdist with setuptools")
    return _build_sdist(sdist_directory, config_settings)
