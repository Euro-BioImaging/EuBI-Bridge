"""JVM initialization and management."""

import os
import pathlib
import subprocess
import sys
from pathlib import Path
from typing import Optional

from eubi_bridge.utils.logging_config import get_logger

logger = get_logger(__name__)


def find_libjvm() -> str:
    """
    Find a bundled libjvm for the current platform.

    Searches in multiple locations:
    1. Root-level bioformats/jdk/<platform>/<arch> (new location after refactoring)
    2. Package-level eubi_bridge/bioformats/jdk/<platform>/<arch> (legacy)
    3. Package-level eubi_bridge/bioformats/libjvm/<platform>/<arch> (fallback)

    Returns:
        Path to libjvm library

    Raises:
        RuntimeError: If no libjvm found
    """
    pkg_dir = pathlib.Path(__file__).resolve().parent.parent  # eubi_bridge/
    repo_root = pkg_dir.parent  # Repository root
    
    libjvm_base = pkg_dir / "bioformats" / "libjvm"
    jdk_pkg_base = pkg_dir / "bioformats" / "jdk"
    jdk_root_base = repo_root / "bioformats" / "jdk"

    platform_map = {
        "darwin": ("darwin", "libjvm.dylib"),
        "win32": ("win32", "jvm.dll"),
        "linux": ("linux", "libjvm.so"),
    }

    if sys.platform not in platform_map:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")

    platform_dir, libjvm_name = platform_map[sys.platform]

    # arch resolution (used for darwin; harmless elsewhere)
    arch_dir = ""
    if sys.platform == "darwin":
        import platform as _platform
        arch = _platform.machine().lower()
        if arch in ("arm64", "aarch64"):
            arch_dir = "arm64"
        elif arch in ("x86_64", "amd64"):
            arch_dir = "x86_64"
        else:
            arch_dir = arch

    candidates = []

    # 1) Prefer bundled full JDK from root-level bioformats (new location)
    jdk_arch_paths = []
    if arch_dir:
        jdk_arch_paths.append(jdk_root_base / platform_dir / arch_dir)
    jdk_arch_paths.append(jdk_root_base / platform_dir)
    for jdk_path in jdk_arch_paths:
        candidates.append(jdk_path / "lib" / "server" / libjvm_name)
        # macOS JDKs often live under Contents/Home
        candidates.append(jdk_path / "Contents" / "Home" / "lib" / "server" / libjvm_name)

    # 2) Fallback to package-level bundled JDK (legacy)
    jdk_arch_paths = []
    if arch_dir:
        jdk_arch_paths.append(jdk_pkg_base / platform_dir / arch_dir)
    jdk_arch_paths.append(jdk_pkg_base / platform_dir)
    for jdk_path in jdk_arch_paths:
        candidates.append(jdk_path / "lib" / "server" / libjvm_name)
        # macOS JDKs often live under Contents/Home
        candidates.append(jdk_path / "Contents" / "Home" / "lib" / "server" / libjvm_name)

    # 3) Fallback to legacy bundled libjvm-only layout
    if sys.platform == "darwin" and arch_dir:
        candidates.append(libjvm_base / "darwin" / arch_dir / libjvm_name)
    candidates.append(libjvm_base / platform_dir / libjvm_name)

    for p in candidates:
        if p.exists():
            return str(p)

    checked = "\n".join(str(p) for p in candidates)
    raise RuntimeError(
        f"Bundled libjvm not found. Checked:\n{checked}\n"
        f"Please bundle a full JDK under {jdk_pkg_base} or {jdk_root_base}, "
        f"or provide libjvm under {libjvm_base}."
    )


def soft_start_jvm() -> None:
    """Start JVM with bundled JARs only, bypassing Maven/JGO entirely."""
    import pathlib
    import traceback

    # Critical: Import scyjava AFTER environment setup
    import scyjava

    if scyjava.jvm_started():
        return

    # Get bundled JARs directory
    pkg_dir = pathlib.Path(__file__).resolve().parent.parent
    jars_dir = pkg_dir / "bioformats"
    jars = sorted(str(p) for p in jars_dir.glob("*.jar"))

    if not jars:
        raise RuntimeError(f"No bundled jars found in {jars_dir}")

    # Method 1: Use jpype directly (bypasses scyjava's Maven logic)
    try:
        import jpype
        import jpype.imports

        # Find jpype's support JAR - critical for Windows
        jpype_dir = pathlib.Path(jpype.__file__).parent.parent
        jpype_jar = jpype_dir / "org.jpype.jar"

        if not jpype_jar.exists():
            # Try alternative locations
            for possible_jar in jpype_dir.rglob("org.jpype*.jar"):
                jpype_jar = possible_jar
                break

        if not jpype_jar.exists():
            raise RuntimeError(
                f"Cannot find org.jpype.jar in {jpype_dir}. "
                "JPype may not be installed correctly. "
                "Try: pip uninstall jpype1 && pip install jpype1 --force-reinstall"
            )

        # Build classpath: jpype JAR first, then bioformats JARs
        classpath = str(jpype_jar) + os.pathsep + os.pathsep.join(jars)

        logger.info(f"Starting JVM with {len(jars)} bioformats JARs + jpype JAR")

        # Prepare JVM arguments
        jvm_args = ['-Djava.awt.headless=true']  # Disable GUI for HPC/headless environments

        # Prepare JVM keyword arguments
        jvm_kwargs = {
            'classpath': classpath,
            'convertStrings': False
        }

        # Use bundled libjvm (native library)
        try:
            jvm_path = find_libjvm()
            jvm_kwargs['jvmpath'] = jvm_path
            logger.info(f"Using bundled libjvm: {jvm_path}")
        except RuntimeError as e:
            logger.warning(f"Could not find bundled libjvm - {e}")
            logger.warning("Attempting to start JVM without explicit jvmpath")

        # Start JVM with explicit classpath and headless mode
        jpype.startJVM(*jvm_args, **jvm_kwargs)
        scyjava._jpype_jvm = jpype
        logger.info("JVM started successfully with jpype")
        return

    except ImportError:
        logger.warning("jpype not available, falling back to scyjava")
    except Exception as e:
        logger.warning(f"JPype startup failed, falling back to scyjava: {e}")
        traceback.print_exc()

    # Method 2: Force scyjava to use local JARs only (fallback)
    # Clear any Maven configuration
    scyjava.config.endpoints.clear()
    scyjava.config.maven_offline = True

    # Set bundled libjvm path if available
    try:
        jvm_path = find_libjvm()
        scyjava.config.jvm_path = str(jvm_path)
        logger.info(f"Using bundled libjvm with scyjava: {jvm_path}")
    except RuntimeError:
        logger.warning("No bundled libjvm found, will rely on JAVA_HOME")

    # Disable JGO by monkeypatching
    try:
        import jgo.jgo
        jgo.jgo.resolve_dependencies = lambda *args, **kwargs: []
    except ImportError:
        pass

    # Add all JARs to classpath
    for jar in jars:
        scyjava.config.add_classpath(jar)

    # Start JVM
    scyjava.start_jvm()
    logger.info("JVM started successfully with scyjava")


def download_and_extract_jdk():
    """
    Download the platform-specific JDK from GitHub Releases and extract it.
    
    This function is called on first CLI invocation to prepare the JDK.
    The JDK is downloaded into the installed package location.
    
    JDK archives are hosted on GitHub Releases:
    - jdk_darwin_arm64.tar.gz (~178 MB) - macOS Apple Silicon
    - jdk_darwin_x86_64.tar.gz (~172 MB) - macOS Intel
    - jdk_linux.tar.gz (~184 MB) - Linux x86_64
    - jdk_win32.tar.gz - Windows (if available)
    """
    import platform
    import tarfile
    import tempfile
    import urllib.request
    
    GITHUB_REPO = "Euro-BioImaging/EuBI-Bridge"
    GITHUB_RELEASE_TAG = "jdk-v11"
    GITHUB_RELEASES_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{GITHUB_RELEASE_TAG}"
    
    # Get platform info
    system = platform.system()
    machine = platform.machine().lower()
    
    if system == 'Darwin':
        platform_id = 'darwin'
        if machine in ('arm64', 'aarch64'):
            jdk_archive_name = 'jdk_darwin_arm64.tar.gz'
            arch_subdir = 'arm64'
        else:
            jdk_archive_name = 'jdk_darwin_x86_64.tar.gz'
            arch_subdir = 'x86_64'
    elif system == 'Linux':
        platform_id = 'linux'
        jdk_archive_name = 'jdk_linux.tar.gz'
        arch_subdir = None
    elif system == 'Windows':
        platform_id = 'win32'
        jdk_archive_name = 'jdk_win32.tar.gz'
        arch_subdir = None
    else:
        logger.debug(f"JDK download not supported for platform: {system}")
        return
    
    # Determine JDK install path
    pkg_dir = Path(__file__).resolve().parent.parent  # eubi_bridge/
    jdk_base = pkg_dir / "bioformats" / "jdk"
    
    if arch_subdir:
        jdk_platform_path = jdk_base / platform_id / arch_subdir
    else:
        jdk_platform_path = jdk_base / platform_id
    
    # Check if JDK already exists with binaries
    if jdk_platform_path.exists():
        java_bin = jdk_platform_path / "Contents" / "Home" / "bin" / "java" if system == 'Darwin' else jdk_platform_path / "bin" / "java"
        if java_bin.exists():
            logger.debug(f"JDK for {platform_id} already present at {jdk_platform_path}")
            return
    
    logger.info(f"Downloading JDK for {platform_id}...")
    
    # Create directory structure
    jdk_platform_path.mkdir(parents=True, exist_ok=True)
    
    # Download from GitHub
    github_url = f"{GITHUB_RELEASES_URL}/{jdk_archive_name}"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tar.gz') as tmp:
        tmp_path = tmp.name
    
    try:
        urllib.request.urlretrieve(github_url, tmp_path)
        
        # Verify download
        file_size = os.path.getsize(tmp_path)
        if file_size < 1000000:
            raise RuntimeError(f"Downloaded file too small ({file_size} bytes)")
        
        logger.info(f"Extracting JDK ({file_size // 1024 // 1024} MB)...")
        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(path=jdk_platform_path)
        
        logger.info(f"JDK successfully downloaded and installed")
        
    except Exception as e:
        logger.debug(f"JDK download failed: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
