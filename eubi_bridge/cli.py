import os
import multiprocessing as mp

# === CRITICAL: Set multiprocessing method FIRST ===
# This MUST happen before any other imports that might create process pools
mp.set_start_method("spawn", force=True)

# === CRITICAL: Block Maven/network access BEFORE any imports ===
# Set environment variables to disable network access
os.environ['JGO_CACHE_DIR'] = '/dev/null'  # Prevents JGO cache creation
os.environ['MAVEN_OFFLINE'] = 'true'


# Block network at socket level (nuclear option)
def _block_network():
    import socket
    original_getaddrinfo = socket.getaddrinfo

    def _blocked_getaddrinfo(host, *args, **kwargs):
        # Allow localhost only
        if host in ('localhost', '127.0.0.1', '::1'):
            return original_getaddrinfo(host, *args, **kwargs)
        raise OSError(f"Network access blocked: {host}")

    socket.getaddrinfo = _blocked_getaddrinfo


# Uncomment if you want to completely prevent network access:
# _block_network()

# Now import scyjava and configure it
import scyjava

# Disable Maven completely
scyjava.config.endpoints.clear()
scyjava.config.maven_offline = True

# Monkey-patch JGO to prevent it from doing anything
try:
    import jgo.jgo

    jgo.jgo.resolve_dependencies = lambda *args, **kwargs: []
    jgo.jgo.executable_path = lambda *args, **kwargs: None
except ImportError:
    pass  # JGO not installed, even better

# === Now safe to import other modules ===
import fire
import warnings


# Patch xsdata for Cython compatibility BEFORE importing anything that uses ome_types
def _patch_xsdata_for_cython():
    """Patch xsdata to handle Cython types without __subclasses__."""
    try:
        from xsdata.formats.dataclass.context import XmlContext

        original_get_subclasses = XmlContext.get_subclasses

        @classmethod
        def patched_get_subclasses(cls, clazz):
            """Patched version that handles types without __subclasses__."""
            try:
                for subclass in clazz.__subclasses__():
                    yield subclass
                    if hasattr(subclass, '__subclasses__'):
                        yield from cls.get_subclasses(subclass)
            except (AttributeError, TypeError):
                pass

        XmlContext.get_subclasses = patched_get_subclasses
    except ImportError:
        pass


_patch_xsdata_for_cython()

from eubi_bridge.utils.convenience import soft_start_jvm
from eubi_bridge.ebridge import EuBIBridge
from eubi_bridge.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)
setup_logging()
warnings.filterwarnings("ignore", message="Casting invalid DichroicID*", category=UserWarning)


# --- Fire patch ---
def patch_fire_no_literal_eval_for(*arg_names):
    import fire.core
    if not hasattr(fire.core, "_original_ParseValue"):
        fire.core._original_ParseValue = fire.core._ParseValue

    def _parse_value_custom(value, index, arg, metadata):
        if any(name in arg for name in arg_names):
            return value
        return fire.core._original_ParseValue(value, index, arg, metadata)

    fire.core._ParseValue = _parse_value_custom


# --- Main ---
def main():
    patch_fire_no_literal_eval_for("includes", "excludes")

    # Start JVM with bundled JARs only
    soft_start_jvm()

    # Don't set spawn here - already set at module level
    fire.Fire(EuBIBridge)


if __name__ == "__main__":
    main()