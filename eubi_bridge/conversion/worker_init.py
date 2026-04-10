"""
Worker initialization module for multiprocessing with JVM support.
"""
import functools
import multiprocessing as mp
import os
import sys

from eubi_bridge.utils.logging_config import get_logger

logger = get_logger(__name__)

# Global flag to track if worker is initialized
_worker_initialized = False
# Global tensorstore context for data copy concurrency
_tensorstore_context = None




def build_tensorstore_context(data_copy_concurrency=None):
    """
    Build a tensorstore Context with data_copy_concurrency limit.

    Parameters
    ----------
    data_copy_concurrency : int or None or 'default', optional
        Number of CPU cores to use for concurrent data copying/encoding/decoding
        in tensorstore operations.  Pass None or 'default' to let TensorStore
        manage its own thread pool (recommended for most workloads).
        Explicit integer values throttle the pool; useful when many worker
        processes would otherwise oversubscribe the CPU.

    Returns
    -------
    tensorstore.Context or None
        Tensorstore Context object with data_copy_concurrency settings, or
        None to use TensorStore defaults.
    """
    # Explicit sentinel values → let TensorStore manage its own pool
    if data_copy_concurrency is None or data_copy_concurrency == 'default':
        logger.debug("Tensorstore context: using TensorStore defaults (no limit set)")
        return None

    try:
        import tensorstore as ts

        # Guard against accidentally receiving a non-scalar (e.g. a Queue proxy
        # object passed in the wrong position via initargs).
        if not isinstance(data_copy_concurrency, (int, float, str, bytes)):
            logger.warning(
                f"build_tensorstore_context received unexpected type "
                f"{type(data_copy_concurrency).__name__} for data_copy_concurrency; "
                f"using TensorStore defaults."
            )
            return None

        limit = int(data_copy_concurrency)
        if limit < 1:
            logger.warning(
                f"Invalid tensorstore_data_copy_concurrency value: {limit}. "
                f"Must be >= 1; using TensorStore defaults."
            )
            return None

        context = ts.Context({"data_copy_concurrency": {"limit": limit}})
        logger.debug(f"Built tensorstore context with data_copy_concurrency limit: {limit}")
        return context

    except (TypeError, ValueError, Exception) as e:
        logger.error(
            f"Error building tensorstore context: {e}. "
            f"Using TensorStore defaults."
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


def initialize_worker_process(tensorstore_data_copy_concurrency='default'):
    """
    Initialize worker process with JVM and proper scyjava configuration.

    This is called once per worker process via ProcessPoolExecutor's initializer.
    
    Parameters
    ----------
    tensorstore_data_copy_concurrency : int, optional
        CPU core limit for tensorstore data copying. Default is 1.
    """
    global _worker_initialized, _tensorstore_context

    if _worker_initialized:
        return

    logger.info(f"[Worker {mp.current_process().name}] Starting initialization...")
    
    # Build tensorstore context with data_copy_concurrency limit
    try:
        # For ProcessPoolExecutor with spawn context, use initargs value
        # For ThreadPoolExecutor, also check environment variable set by parent
        data_copy_concurrency = tensorstore_data_copy_concurrency
        if (data_copy_concurrency in (1, 'default', None)
                and 'EUBI_TENSORSTORE_DATA_COPY_CONCURRENCY' in os.environ):
            # Env-var override (used by ThreadPoolExecutor workers where initargs
            # cannot easily carry the value).
            data_copy_concurrency = int(os.environ['EUBI_TENSORSTORE_DATA_COPY_CONCURRENCY'])
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
            # Capture full exception details before any str() call that might fail
            exc_type = type(e).__name__
            try:
                exc_msg = str(e)
            except Exception:
                exc_msg = repr(type(e))
            exc_tb = traceback.format_exc()

            # Create a simple, picklable RuntimeError
            error_msg = (
                f"Worker process failed\n"
                f"Function: {func.__name__}\n"
                f"Exception: {exc_type}: {exc_msg}\n"
                f"\nFull traceback:\n{exc_tb}"
            )

            logger.error(f"[Worker Error] {error_msg}")
            # Clear __context__ so multiprocessing doesn't try to pickle the
            # original (possibly un-picklable) Java exception when sending this
            # RuntimeError back to the main process.
            new_exc = RuntimeError(error_msg)
            new_exc.__context__ = None
            raise new_exc

    return wrapper