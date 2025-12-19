"""JVM initialization and management."""

import os
import pathlib
import subprocess
import sys
from pathlib import Path
from typing import Optional

from eubi_bridge.utils.logging_config import get_logger

logger = get_logger(__name__)


def get_or_install_jdk() -> Optional[str]:
    """
    Ensure a modern JDK is available, installing via install-jdk if needed.

    Returns:
        Path to java executable, or None if unable to find/install JDK
    """
    # First, check if JAVA_HOME is set and points to a good JDK
    java_home = os.environ.get('JAVA_HOME')
    if java_home:
        java_path = pathlib.Path(java_home) / 'bin' / 'java'
        if not java_path.exists():
            java_path = pathlib.Path(java_home) / 'bin' / 'java.exe'

        if java_path.exists():
            # Check version
            try:
                result = subprocess.run(
                    [str(java_path), '-version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                version_output = result.stderr + result.stdout

                # Check if it's Java 11+ (Bio-Formats requirement)
                if 'version "1.' in version_output:  # Java 8 or older
                    major_version = int(version_output.split('"')[1].split('.')[1])
                    if major_version >= 8:
                        logger.warning(f"Found Java {major_version} at {java_home}, but need 11+")
                elif 'version "' in version_output:
                    version_str = version_output.split('"')[1]
                    major_version = int(version_str.split('.')[0])
                    if major_version >= 11:
                        logger.info(f"Using Java {major_version} from JAVA_HOME: {java_home}")
                        return str(java_path)
            except Exception as e:
                logger.error(f"Error checking Java version: {e}")

    # Try to use install-jdk to get a proper JDK
    try:
        import jdk
        jdkpath = Path(jdk._JDK_DIR)
        jdkpath.mkdir(parents=True, exist_ok=True)

        installed_jdks = list(jdkpath.glob('*'))
        if len(installed_jdks) > 0:
            java_home = installed_jdks[0]
            logger.info(f"Using JDK: {java_home}")
        else:
            java_home = jdk.install('17')
            logger.info(f"JDK 17 installed at: {java_home}")

        # Set JAVA_HOME for this process and children
        os.environ['JAVA_HOME'] = str(java_home)

        # Return path to java executable
        java_path = pathlib.Path(java_home) / 'bin' / 'java'
        if not java_path.exists():
            java_path = pathlib.Path(java_home) / 'bin' / 'java.exe'

        return str(java_path)

    except ImportError:
        logger.warning("install-jdk not found. Install with: pip install install-jdk")
        logger.warning("Falling back to system Java (may be outdated)")
        return None
    except Exception as e:
        logger.error(f"Error installing JDK: {e}")
        return None


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

    # Ensure we have a modern JDK
    java_path = get_or_install_jdk()

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
        jvm_kwargs = {
            'classpath': classpath,
            'convertStrings': False
        }

        # Use bundled libjvm
        try:
            jvm_path = find_libjvm()
            jvm_kwargs['jvmpath'] = jvm_path
            logger.info(f"Using bundled libjvm: {jvm_path}")
        except RuntimeError as e:
            logger.warning(f"Could not find bundled libjvm - {e}")
            logger.warning("Attempting to start JVM without explicit jvmpath")
            jvm_path = None

        # Start JVM with explicit classpath
        try:
            jpype.startJVM(**jvm_kwargs)
        except Exception as e:
            logger.error(f"JPype start failed: {e}")
            traceback.print_exc()
            raise

        # Mark scyjava as started
        scyjava._jpype_jvm = jpype

        logger.info("JVM started successfully")
        return

    except ImportError:
        logger.warning("jpype not available, falling back to scyjava")
    except Exception as e:
        logger.warning(f"JPype path failed, falling back to scyjava: {e}")
        traceback.print_exc()

    # Method 2: Force scyjava to use local JARs only
    # Clear any Maven configuration
    scyjava.config.endpoints.clear()
    scyjava.config.maven_offline = True

    # Set JVM path if we have one
    jvm_path_override = None
    try:
        if 'jvm_kwargs' in locals():
            jvm_path_override = jvm_kwargs.get('jvmpath')
    except Exception:
        pass

    if jvm_path_override:
        scyjava.config.jvm_path = str(jvm_path_override)
    elif java_path:
        scyjava.config.jvm_path = str(java_path)

    # Disable JGO by monkeypatching
    try:
        import jgo.jgo
        original_resolve = jgo.jgo.resolve_dependencies
        jgo.jgo.resolve_dependencies = lambda *args, **kwargs: []
    except ImportError:
        pass

    # Add all JARs to classpath
    for jar in jars:
        scyjava.config.add_classpath(jar)

    # Start JVM
    scyjava.start_jvm()

    logger.info("JVM started successfully via scyjava")
