"""
Worker initialization module for multiprocessing with JVM support.
"""
import multiprocessing as mp
import os
import sys

from eubi_bridge.utils.logging_config import get_logger

logger = get_logger(__name__)

# Global flag to track if worker is initialized
_worker_initialized = False
# Global tensorstore context for data copy concurrency
_tensorstore_context = None

def build_tensorstore_context(data_copy_concurrency: int = 1):
    """
    Build a tensorstore Context with data_copy_concurrency limit.
    
    Parameters
    ----------
    data_copy_concurrency : int, optional
        Number of CPU cores to use for concurrent data copying/encoding/decoding
        in tensorstore operations. Default: 1 (serialized, safest for high-concurrency
        scenarios). Higher values allow more parallelism but increase thread contention.
    
    Returns
    -------
    tensorstore.Context or None
        Tensorstore Context object with data_copy_concurrency settings.
        Returns None if input is invalid (falls back to tensorstore default).
    """
    try:
        import tensorstore as ts
        
        # Validate input
        if data_copy_concurrency is None:
            data_copy_concurrency = 1
        
        # Convert to int if needed
        data_copy_concurrency = int(data_copy_concurrency)
        
        # Ensure positive value
        if data_copy_concurrency < 1:
            logger.warning(
                f"Invalid tensorstore_data_copy_concurrency value: {data_copy_concurrency}. "
                f"Must be >= 1. Using default: 1"
            )
            data_copy_concurrency = 1
        
        # Create tensorstore Context object with data_copy_concurrency limit
        context = ts.Context({
            "data_copy_concurrency": {"limit": data_copy_concurrency}
        })
        
        logger.debug(
            f"Built tensorstore context with data_copy_concurrency limit: {data_copy_concurrency}"
        )
        return context
        
    except (TypeError, ValueError, Exception) as e:
        logger.error(
            f"Error building tensorstore context: {e}. "
            f"Continuing without custom context (tensorstore will use defaults)."
        )
        return None


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


def initialize_worker_process(**kwargs):
    """
    Initialize worker process with JVM and proper scyjava configuration.

    This is called once per worker process via ProcessPoolExecutor's initializer.
    
    Parameters
    ----------
    **kwargs : dict, optional
        Additional initialization parameters:
        - tensorstore_data_copy_concurrency (int): CPU core limit for tensorstore data copying.
    """
    global _worker_initialized, _tensorstore_context

    if _worker_initialized:
        return

    logger.info(f"[Worker {mp.current_process().name}] Starting initialization...")
    
    # Build tensorstore context with data_copy_concurrency limit
    try:
        data_copy_concurrency = kwargs.get('tensorstore_data_copy_concurrency', 1)
        _tensorstore_context = build_tensorstore_context(data_copy_concurrency)
        logger.info(
            f"[Worker {mp.current_process().name}] Tensorstore context configured: "
            f"data_copy_concurrency={data_copy_concurrency}"
        )
    except Exception as e:
        logger.error(
            f"[Worker {mp.current_process().name}] Failed to build tensorstore context: {e}. "
            f"Continuing with default tensorstore settings."
        )
        _tensorstore_context = None

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

    # Ensure JDK is available (worker inherits JAVA_HOME from parent if already set)
    # But we may need to check/install if not present
    try:
        java_home = os.environ.get('JAVA_HOME')
        if not java_home:
            logger.info(f"[Worker {mp.current_process().name}] No JAVA_HOME, checking for JDK...")
            # Try to use install-jdk
            try:
                from pathlib import Path

                import jdk
                jdkpath = Path(jdk._JDK_DIR)
                jdkpath.mkdir(parents=True, exist_ok=True)

                installed_jdks = list(jdkpath.glob('*'))
                if len(installed_jdks) > 0:
                    os.environ['JAVA_HOME'] = installed_jdks[0]
                    logger.info(f"[Worker {mp.current_process().name}] Using JDK: {installed_jdks[0]}")
            except ImportError:
                logger.info(f"[Worker {mp.current_process().name}] install-jdk not available")
    except Exception as e:
        logger.warning(f"[Worker {mp.current_process().name}] JDK check warning: {e}")

    # Configure scyjava BEFORE any imports that might use Java
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
        soft_start_jvm()
        logger.info(f"[Worker {mp.current_process().name}] JVM initialized successfully")
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
    """
    import functools
    import traceback

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Ensure worker is initialized (redundant safety check)
            if not _worker_initialized:
                initialize_worker_process()

            return func(*args, **kwargs)

        except Exception as e:
            # Capture full exception details
            exc_type = type(e).__name__
            exc_msg = str(e)
            exc_tb = traceback.format_exc()

            # Create a simple, picklable RuntimeError
            error_msg = (
                f"Worker process failed\n"
                f"Function: {func.__name__}\n"
                f"Exception: {exc_type}: {exc_msg}\n"
                f"\nFull traceback:\n{exc_tb}"
            )

            logger.error(f"[Worker Error] {error_msg}")
            raise RuntimeError(error_msg) from None

    return wrapper