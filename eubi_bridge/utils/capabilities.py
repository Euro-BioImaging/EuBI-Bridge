"""What this installation of eubi-bridge can do.

eubi-bridge-lite installs neither the Bio-Formats stack (bfio / bioio /
scyjava, the JVM) nor dask; the full eubi-bridge installs both.  Everything
reads micro-reader by default (``metadata_reader='micro'``,
``pixel_reader='micro'``); the rest of the stack is needed only for what
micro-reader does not cover: the Bio-Formats metadata readers
(``metadata_reader='bfio'`` / ``'bioio'``), the standard pixel readers
(``pixel_reader='standard'``), a file micro-reader does not read (it falls
back to them), and SLURM (``on_slurm``, through dask-jobqueue).

In lite, asking for those raises ``NeedsFullEubiBridge``, which says what to
install, before any work starts -- not an ImportError from deep inside.
"""
from __future__ import annotations

from eubi_bridge.utils.optional_deps import is_installed

#: the two packages share their files: lite must go before the full one comes
INSTALL_FULL = ("Needs the full eubi-bridge: pip uninstall eubi-bridge-lite, "
                "then pip install eubi-bridge.")


#: the modules the Bio-Formats path imports
BIOFORMATS_MODULES = ("bfio", "bioio_base", "scyjava")


class NeedsFullEubiBridge(ImportError):
    """A feature of the full eubi-bridge, asked for in eubi-bridge-lite."""


def has_bioformats() -> bool:
    """The Bio-Formats / bioio readers and the Java bridge are installed.
    (eubi-bridge depends on bioio's base and plugins, not on ``bioio`` itself.)"""
    return all(is_installed(m) for m in BIOFORMATS_MODULES)


def has_dask() -> bool:
    return is_installed("dask")


def has_slurm() -> bool:
    """SLURM through dask-jobqueue is possible (the full eubi-bridge)."""
    return all(is_installed(m) for m in ("dask", "distributed", "dask_jobqueue"))


def check_run(metadata_reader: str = "micro", pixel_reader: str = "micro",
              on_slurm: bool = False, force_bioformats: bool = False) -> None:
    """Raise ``NeedsFullEubiBridge`` if a conversion with these settings
    needs what this installation lacks; nothing to check in the full one."""
    missing = []
    if not has_bioformats():
        if metadata_reader not in ("micro", None):
            missing.append(f"metadata_reader={metadata_reader!r} (Bio-Formats); "
                           f"use metadata_reader='micro'")
        if pixel_reader == "standard" and metadata_reader != "micro":
            missing.append("pixel_reader='standard' (the standard readers); "
                           "use pixel_reader='micro'")
        if force_bioformats:
            missing.append("force_bioformats=True (Bio-Formats)")
    if on_slurm and not has_slurm():
        missing.append("on_slurm=True (SLURM)")
    if missing:
        raise NeedsFullEubiBridge(
            "This conversion uses what eubi-bridge-lite does not include:\n  - "
            + "\n  - ".join(missing) + f"\n{INSTALL_FULL}")


def micro_reader_declined(path: str) -> NeedsFullEubiBridge:
    """The error for a file micro-reader does not read, where the fallback
    (Bio-Formats) is not installed: worded for the lite user, who has one
    reader and need not know its name."""
    return NeedsFullEubiBridge(
        f"{path}: eubi-bridge-lite cannot read this file: it is not an image it "
        f"reads, nor a part of one (for a plate, choose its index file, e.g. "
        f"Index.idx.xml or index.xml). If it is an image in another format, the full "
        f"eubi-bridge may read it: pip uninstall eubi-bridge-lite, then pip install "
        f"eubi-bridge.")
