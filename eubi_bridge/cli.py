import fire, fire.core
from eubi_bridge.ebridge import EuBIBridge
import multiprocessing as mp
import sys, logging
from eubi_bridge.utils.logging_config import get_logger, setup_logging

# Set up logger for this module
logger = get_logger(__name__)
setup_logging()

import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Casting invalid DichroicID*", category=UserWarning)

# Suppress noisy logs
# logging.getLogger('distributed.diskutils').setLevel(logging.CRITICAL)
# logging.getLogger('distributed.worker').setLevel(logging.WARNING)
# logging.getLogger('distributed.scheduler').setLevel(logging.WARNING)
# logging.basicConfig(level=logging.INFO)

def patch_fire_no_literal_eval_for(*arg_names):
    """
    Patch Fire so that specific argument names are never parsed
    with literal_eval (i.e., always treated as raw strings).

    Example:
        patch_fire_no_literal_eval_for("includes", "sample_id")
    """

    # Save original function
    if not hasattr(fire.core, "_original_ParseValue"):
        fire.core._original_ParseValue = fire.core._ParseValue

    def _parse_value_custom(value, index, arg, metadata):
        # arg is like '--includes' or '--sample_id'
        if any(name in arg for name in arg_names):
            return value  # return the raw string untouched

        # Otherwise, use Fire’s normal parsing
        return fire.core._original_ParseValue(value, index, arg, metadata)

    fire.core._ParseValue = _parse_value_custom

def main():
    """Main entry point for EuBIBridge CLI."""

    # Prevent Fire from misinterpreting specific args (like 17_03_18)
    patch_fire_no_literal_eval_for("includes", "excludes")

    if sys.platform == "win32":
        mp.set_start_method("spawn", force=True)

    fire.Fire(EuBIBridge)


if __name__ == "__main__":
    main()
