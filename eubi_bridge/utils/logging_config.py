import logging
import sys

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(level=logging.INFO):
    """Configure rich-colored logging for eubi_bridge, silence everything else."""

    # Import bfio here (not at module level) to avoid slowdown in config-only commands
    try:
        import bfio
    except ImportError:
        pass  # bfio is optional

    # Respect the actual terminal state — don't force ANSI into non-terminal outputs
    # (e.g. subprocess workers write to a queue, not a real TTY).
    is_tty = getattr(sys.stdout, "isatty", lambda: False)()
    console = Console(
        force_terminal=is_tty,
        color_system="auto" if is_tty else None,
        highlight=is_tty,
    )

    # Set up RichHandler manually instead of via basicConfig
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        markup=True,
        show_path=True,
        log_time_format="[%X]",
    )

    # Manually configure root logger for eubi_bridge
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()  # ensure no duplicates
    root_logger.addHandler(rich_handler)

    # Ensure only eubi_bridge loggers propagate
    for name, logger_obj in list(logging.root.manager.loggerDict.items()):
        if isinstance(logger_obj, logging.Logger) and not name.startswith("eubi_bridge"):
            logger_obj.handlers.clear()
            logger_obj.propagate = False
            logger_obj.setLevel(logging.CRITICAL)

    # Explicitly silence bfio
    for name in ["bfio", "bfio.start"]:
        log = logging.getLogger(name)
        log.handlers.clear()
        log.propagate = False
        log.setLevel(logging.CRITICAL)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under eubi_bridge"""
    return logging.getLogger(f"eubi_bridge.{name}")
