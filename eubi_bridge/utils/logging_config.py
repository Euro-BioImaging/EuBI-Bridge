import logging
import sys
import bfio  # must import first

def setup_logging(level=logging.INFO):
    """Configure root logger for eubi_bridge only, silence everything else."""
    # Root logger for eubi_bridge
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )

    # Silence all non-eubi_bridge loggers
    root_logger = logging.getLogger()
    for name, logger_obj in list(logging.root.manager.loggerDict.items()):
        if isinstance(logger_obj, logging.Logger) and not name.startswith("eubi_bridge"):
            logger_obj.handlers.clear()
            logger_obj.propagate = False
            logger_obj.setLevel(logging.CRITICAL)

    # Explicitly silence bfio after it has added its handler
    for name in ["bfio", "bfio.start"]:
        log = logging.getLogger(name)
        log.handlers.clear()        # remove bfio’s handler
        log.propagate = False       # prevent it from bubbling up
        log.setLevel(logging.CRITICAL)  # suppress everything

def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under eubi_bridge"""
    return logging.getLogger(f"eubi_bridge.{name}")
