"""Output stores and the ome_zarr_pyramid write arguments eubi-bridge decides itself.

What was left of eubi-bridge's own writer (``core/writers.py``), its ozp adapter
(``core/ozp_writer.py``) and its NGFF layer (``ngff/``) once eubi-bridge 0.2.0 wrote
every output with ome_zarr_pyramid: the store checks, the S3 address translation, the chunk shape, the
OME-Zarr version and the ``eubi_bridge`` attrs block. The retired writers are kept for
reference in ``legacy_0_1/`` at the repository root.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from eubi_bridge.utils.array_utils import autocompute_chunk_shape
from eubi_bridge.utils.logging_config import get_logger
from eubi_bridge.utils.optional_deps import is_dask_array

logger = get_logger(__name__)


def _is_remote_path(path) -> bool:
    """True for an object-store URL rather than a local filesystem path."""
    return isinstance(path, str) and path.startswith(("https://", "s3://"))


def store_exists(path) -> bool:
    """True when a store already exists at *path*, local or remote.

    ``os.path.exists`` silently answers False for an ``https://`` URL -- it
    looks for a local directory literally named ``https:`` -- so a caller
    guarding against clobbering an existing dataset would sail straight past a
    populated bucket.  Remote paths are therefore asked of the object store.
    """
    if not _is_remote_path(path):
        return os.path.exists(path)
    try:
        import s3fs
    except ImportError:
        return False
    text = str(path)
    endpoint = 'https://' + text.replace('https://', '').split('/')[0]
    relpath = text[len(endpoint):].lstrip('/')
    try:
        fs = s3fs.S3FileSystem(
            anon=True,
            client_kwargs={'endpoint_url': endpoint},
            endpoint_url=endpoint,
        )
        fs.invalidate_cache()
        # A zarr store is a key prefix, not an object: exists() alone is False
        # for a prefix on some backends, so a non-empty listing counts too.
        return bool(fs.exists(relpath)) or bool(fs.ls(relpath, detail=False))
    except Exception:
        # Never let a probe failure masquerade as "already exists" and block a
        # legitimate conversion; the writer still raises if it truly collides.
        return False


def store_join(base, key: str) -> str:
    """Join a store path with a child key.

    ``os.path.join`` is wrong for a URL on Windows -- it inserts a backslash,
    which is not a key separator -- so remote paths are joined with '/'.
    """
    if _is_remote_path(base):
        return f"{str(base).rstrip('/')}/{key}"
    return os.path.join(base, key)


def _is_resource_backed(arr) -> bool:
    """True when *arr* is a resource-backed dask array.

    Such arrays read through one shared handle that is not thread-safe: Bio-Formats-
    backed arrays (one Java reader) and bioio-nd2 (the nd2 library's file handle /
    memory map, which returns wrong pixels or crashes the process when read from
    several threads at once). ``conversion/legacy_image.py`` reads them one region at
    a time.
    """
    if not is_dask_array(arr):
        return False
    try:
        from resource_backed_dask_array import ResourceBackedDaskArray
        return isinstance(arr, ResourceBackedDaskArray)
    except ImportError:
        return False


def _is_dask(arr) -> bool:
    """A dask array (subclasses included), without importing dask."""
    return type(arr).__module__.split(".")[0] == "dask" or any(
        c.__module__.split(".")[0] == "dask" for c in type(arr).__mro__)


def _memory_budget(arrays, kwargs) -> Optional[float]:
    """The memory budget for this write, or None where it does not apply.

    dyna-zarr bounds a write by its budget in its own region loop. A dask array is
    written by dask's own scheduler instead (its memory is max_threads x the output's
    write unit), and dyna refuses a budget for it rather than ignore it. eubi-bridge sets
    the budget for every conversion, so for a dask source (the standard readers) it is
    left out here. (eubi-bridge 0.1.4 passed it anyway, so every conversion read
    through dask failed.)"""
    budget = kwargs.get('memory_budget_mb')
    if budget is not None and any(_is_dask(a) for a in arrays):
        logger.info("memory_budget does not apply to a dask-read source: dask's scheduler "
                    "writes it (memory about max_threads x the output's write unit)")
        return None
    return budget


def _ozp_target_args(output_path):
    """``(path, storage_options)`` for ome_zarr_pyramid.

    eubi-bridge addresses object storage as ``https://<endpoint>/<bucket>/<key>`` and
    reads/writes it anonymously (s3fs ``anon=True``). ome_zarr_pyramid's dyna engine
    takes ``s3://<bucket>/<key>`` plus obstore options: the same endpoint, unsigned
    requests (anonymous) and path-style addressing (bucket in the path, as in the URL).
    Local paths pass through.
    """
    text = str(output_path)
    if text.startswith("https://"):
        host, _, key = text[len("https://"):].partition("/")
        return (f"s3://{key}", {"endpoint": f"https://{host}", "skip_signature": True,
                                "virtual_hosted_style_request": False})
    return text, None


def _ozp_version(zarr_format: int, ome_zarr_version: Optional[str]) -> str:
    """The OME-Zarr version: as given, else 0.5 for zarr v3 and 0.4 for zarr v2."""
    return ome_zarr_version or ('0.5' if zarr_format == 3 else '0.4')


def _chunks(arr, axes, auto_chunk, output_chunks, dtype, target_chunk_mb):
    """The level-0 chunk shape: the given one, or eubi-bridge's automatic one
    (``autocompute_chunk_shape``), never larger than the image."""
    if auto_chunk or output_chunks is None:
        chunks = autocompute_chunk_shape(arr.shape, axes=axes,
                                         target_chunk_mb=target_chunk_mb, dtype=dtype)
    else:
        chunks = output_chunks
    return tuple(int(c) for c in np.minimum(chunks, arr.shape).tolist())


def _eubi_attrs(acquisition_metadata: Optional[dict]) -> dict:
    """The namespaced ``eubi_bridge`` attrs block written into every OME-Zarr: the
    version stamp, plus the acquisition metadata NGFF has no field for (view angles,
    illumination directions, wavelengths, ...) when there is some. Namespaced rather
    than ``_creator`` so it cannot collide with a future standard key."""
    try:
        from eubi_bridge import __version__ as version
    except Exception:                                   # noqa: BLE001
        version = "unknown"
    block = {"version": version}
    if acquisition_metadata:
        block['acquisition'] = acquisition_metadata
    return block
