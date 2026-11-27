"""
Worker initialization module for multiprocessing with JVM support.
"""
import functools
import multiprocessing as mp
import os
import sys

from eubi_bridge.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

# Global flag to track if worker is initialized
_worker_initialized = False


#: glibc's malloc arenas a conversion worker uses (Linux). glibc gives each thread
#: that allocates at once an arena of its own, up to 8 per core, and memory freed in one
#: is not given back while others hold pieces of it: in the BARD reproduction (Workplan
#: E25, 2026-11-05) a worker with 2 arenas held 28 % less, at the same speed.
MALLOC_ARENAS = 2
_M_ARENA_MAX = -8                                  # mallopt's parameter (malloc.h)


def limit_malloc_arenas(arenas: int = MALLOC_ARENAS) -> bool:
    """Limit glibc's malloc arenas in this process to *arenas* (``mallopt``), before its
    threads start allocating; True when set. Not on other systems or C libraries (macOS,
    Windows, musl), and not when the user set ``MALLOC_ARENA_MAX`` (theirs stands)."""
    if not sys.platform.startswith("linux") or "MALLOC_ARENA_MAX" in os.environ:
        return False
    try:
        import ctypes
        libc = ctypes.CDLL(None)
        if not hasattr(libc, "gnu_get_libc_version"):      # not glibc
            return False
        return bool(libc.mallopt(_M_ARENA_MAX, int(arenas)))
    except Exception:                                      # noqa: BLE001 - leave it as it is
        return False


def _patch_xsdata_for_cython():
    """
    Patch xsdata to handle Cython types that don't have __subclasses__.

    This fixes: AttributeError: type object '_cython_3_2_1.cython_function_or_method'
    has no attribute '__subclasses__'
    """
    try:
        from xsdata.formats.dataclass.context import XmlContext

        original_get_subclasses = XmlContext.get_subclasses

        @classmethod
        def patched_get_subclasses(cls, clazz):
            """Patched version that handles types without __subclasses__."""
            try:
                # Try to get subclasses normally
                for subclass in clazz.__subclasses__():
                    yield subclass
                    # Recursively get subclasses of subclasses
                    if hasattr(subclass, '__subclasses__'):
                        yield from cls.get_subclasses(subclass)
            except (AttributeError, TypeError):
                # Skip types that don't support __subclasses__
                # (like Cython internal types)
                pass

        XmlContext.get_subclasses = patched_get_subclasses

    except ImportError:
        # xsdata not installed, no patching needed
        pass


def initialize_worker_process():
    """
    Initialize worker process with JVM and proper scyjava configuration.

    This is called once per worker process via ProcessPoolExecutor's initializer.
    It no longer takes tensorstore_data_copy_concurrency: the TensorStore context it
    built was never used, and its log line said the setting applied (2026-11-27).
    """
    global _worker_initialized

    if _worker_initialized:
        return

    # first, before any thread of this worker allocates: fewer malloc arenas (Linux)
    limit_malloc_arenas()

    # Suppress the JPype overload-resolution TypeError that fires when bfio's
    # BioReader.__del__ calls loci.formats.ImageReader.close() during GC.
    # The error is benign (the JVM cleans up regardless) but very noisy.
    _original_unraisablehook = sys.unraisablehook

    def _unraisablehook(unraisable):
        if (isinstance(unraisable.exc_value, TypeError)
                and 'loci.formats.ImageReader.close' in str(unraisable.exc_value)):
            return
        _original_unraisablehook(unraisable)

    sys.unraisablehook = _unraisablehook

    # Spawned worker processes inherit no logging handlers from the parent.
    # Configure a stderr handler here so every logger.info/warning/error call
    # from this process flows to the terminal.
    setup_logging()

    logger.info(f"[Worker {mp.current_process().name}] Starting initialization...")
    
    # === CRITICAL: Import tensorstore to register zarr2 driver ===
    # TensorStore's zarv2 driver is registered via C++ static initializers
    # that run when the C++ extension module loads. This must happen in
    # each spawned process separately (spawn context doesn't inherit registrations).
    import tensorstore as ts  # noqa: F401
    logger.debug(f"[Worker {mp.current_process().name}] TensorStore imported - zarv2 driver registered")

    # Patch xsdata BEFORE any ome_types imports
    _patch_xsdata_for_cython()

    # Set environment variables to prevent Maven access
    os.environ['JGO_CACHE_DIR'] = '/dev/null'
    os.environ['MAVEN_OFFLINE'] = 'true'

    # Configure scyjava BEFORE any imports that might use Java (the Java
    # bridge is in the full eubi-bridge only)
    from eubi_bridge.utils.optional_deps import is_installed
    if is_installed("scyjava"):
        import scyjava
        scyjava.config.endpoints.clear()
        scyjava.config.maven_offline = True

    # Disable JGO
    try:
        import jgo.jgo
        jgo.jgo.resolve_dependencies = lambda *args, **kwargs: []
        jgo.jgo.executable_path = lambda *args, **kwargs: None
    except ImportError:
        pass

    # Now start JVM with bundled JARs
    from eubi_bridge.utils.jvm_manager import soft_start_jvm

    try:
        status = soft_start_jvm()
        if status == "started":
            logger.info(f"[Worker {mp.current_process().name}] JVM initialized successfully")
        elif status == "on demand":
            logger.debug(f"[Worker {mp.current_process().name}] JVM not started: it starts "
                         f"only if a file needs Bio-Formats")
    except Exception as e:
        logger.error(f"[Worker {mp.current_process().name}] JVM init failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    _worker_initialized = True


def safe_worker_wrapper(func):
    """
    Decorator to wrap worker functions with exception handling.

    Converts unpicklable exceptions to picklable RuntimeError with full details.
    A MicroReaderVersionError (a wrong micro-reader installed) is raised again as
    itself, so a script can catch it by its type (its message names the fix).
    """
    import functools
    import traceback

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        version_error = None
        try:
            # Ensure worker is initialized (redundant safety check)
            if not _worker_initialized:
                initialize_worker_process()

            return func(*args, **kwargs)

        except Exception as e:
            # Capture full exception details before any str() call that might fail
            exc_type = type(e).__name__
            try:
                exc_msg = str(e)
            except Exception:
                exc_msg = repr(type(e))
            exc_tb = traceback.format_exc()

            # Create a simple, picklable RuntimeError
            first_line = (exc_msg.strip().splitlines() or [""])[0]
            error_msg = f"{exc_type}: {first_line}"
            cause = e.__cause__
            while cause is not None:
                error_msg += f" <- {type(cause).__name__}: {str(cause)[:200]}"
                cause = cause.__cause__
            if "os_error_code=112" in exc_msg or "No space left" in exc_msg:
                error_msg = f"the disk is full ({exc_type}: {first_line[:120]})"
            logger.debug(f"[Worker Error] {func.__name__}: {exc_tb}")
            # raised there, so already imported (no import that could fail here)
            micro_source = sys.modules.get("eubi_bridge.core.micro_source")
            if micro_source is not None and isinstance(e, micro_source.MicroReaderVersionError):
                version_error = micro_source.MicroReaderVersionError(exc_msg)

        # Raised only once the except block has closed.  Raising inside it
        # chains the original exception back on as __context__ -- clearing the
        # attribute beforehand does not survive the raise -- and a Java
        # exception there cannot be pickled, so the pool broke ("terminated
        # abruptly") instead of reporting the error, and the task was retried.
        if version_error is not None:
            raise version_error
        raise RuntimeError(error_msg)

    return wrapper