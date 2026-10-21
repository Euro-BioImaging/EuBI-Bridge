"""Metadata commands on the conversion path of eubi-bridge 0.2.0 (roadmap step 4).

- ``update_store``: ``update_pixel_meta`` / ``update_channel_meta`` on one OME-Zarr, as
  ONE ``ome_zarr_pyramid.IO().update_metadata`` call: the store is read by ozp, its
  scales, units, channels and windows edited, and only the metadata (and the OME-XML)
  written back. No Java VM, so the updates run on threads.
- ``read_metadata``: ``show_pixel_meta`` for a file, as the conversion would write it:
  the file opened like a conversion opens it (micro-reader, else the older readers;
  ``pyramid_path._open_input``) and the image turned into a ``Pyramid`` by
  ``from_image``, without reading pixels. Labels and colours are the ones a conversion
  writes.
- ``read_ome_zarr``: ``show_pixel_meta`` for an OME-Zarr, read by ozp.

The channel windows of an update (``channel_intensity_limits``): not given (``"keep"``)
leaves them (Bugra, 2026-10-16: also for ``update_channel_meta``, whose 0.1.x default
reset them), ``"from_dtype"`` the dtype's full range, ``"from_array"`` each
channel's minimum / maximum of level 0, ``"auto"`` the 0.1 / 99.9 percentiles of the
coarsest level, as for conversions (decision 'Windows for `auto`').
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Optional

from eubi_bridge.utils.logging_config import get_logger

logger = get_logger(__name__)

#: an update's channel_intensity_limits -> ozp's update_metadata(windows=...)
UPDATE_WINDOWS = {"keep": "keep", "from_dtype": "dtype", "from_array": "data", "auto": "auto"}
#: the scale / unit options of an update, per axis
_SCALES = {"t": "time_scale", "c": "channel_scale", "z": "z_scale", "y": "y_scale",
           "x": "x_scale"}
_UNITS = {"t": "time_unit", "z": "z_unit", "y": "y_unit", "x": "x_unit"}


# -- updates ---------------------------------------------------------------------------

def update_store(input_path: str, kwargs: dict):
    """Edit one OME-Zarr's metadata in place (see the module docstring); the pixels are
    never touched, so conversion settings that change them (squeeze, crop, dtype) do
    not apply. Returns the updated Pyramid."""
    from ome_zarr_pyramid import IO

    from eubi_bridge.conversion.pyramid_path import channel_overrides, ome_xml
    from eubi_bridge.core.config_models import MetadataUpdateConfig
    from eubi_bridge.core.output_store import _ozp_target_args
    from eubi_bridge.utils.path_utils import is_zarr_group

    if not is_zarr_group(str(input_path)):
        raise ValueError(f"Metadata update only works with OME-Zarr datasets: {input_path}")
    cfg = MetadataUpdateConfig(**kwargs)
    limits = kwargs.get("channel_intensity_limits") or "keep"
    if limits not in UPDATE_WINDOWS:
        raise ValueError(f"channel_intensity_limits={limits!r}: one of "
                         f"{sorted(set(UPDATE_WINDOWS) - {'keep'})}")
    target, storage_options = _ozp_target_args(str(input_path))
    io = IO()
    current = io.read_pyramid(target, include_labels=False, storage_options=storage_options)
    axes = current.meta.axis_order
    scales = {ax: getattr(cfg, name) for ax, name in _SCALES.items()
              if ax in axes and getattr(cfg, name) is not None}
    units = {ax: getattr(cfg, name) for ax, name in _UNITS.items()
             if ax in axes and getattr(cfg, name) is not None}
    count = current.base_array.shape[axes.index("c")] if "c" in axes else 1
    channels = channel_overrides(kwargs.get("channel_labels"), kwargs.get("channel_colors"),
                                 count, input_path)

    def xml(pyr):
        return ome_xml(pyr, pyr.base_array.dtype)

    return io.update_metadata(target, scales=scales or None, units=units or None,
                              channels=channels or None, windows=UPDATE_WINDOWS[limits],
                              ome_xml=xml if cfg.save_omexml else None,
                              storage_options=storage_options)


# -- reading metadata ------------------------------------------------------------------

def _summary(pyr, path: str, series) -> dict:
    """What show_pixel_meta prints, from a Pyramid."""
    import numpy as np
    meta = pyr.meta
    axes = meta.axis_order
    base = pyr.base_array
    channels = (meta.metadata.get("omero") or {}).get("channels") or []
    return {
        "status": "success",
        "input_path": str(path),
        "series": series,
        "axes": axes,
        "shape": dict(zip(axes, (int(n) for n in base.shape))),
        "scale": dict(zip(axes, (float(s) for s in meta.get_scale(meta.resolution_paths[0])))),
        "units": {ax: u for ax, u in zip(axes, meta.unit_list) if u},
        "dtype": str(np.dtype(getattr(base.dtype, "numpy_dtype", base.dtype))),
        "channels": [{"label": ch.get("label"), "color": ch.get("color")} for ch in channels],
        "ndim": len(axes),
    }


def read_ome_zarr(path: str) -> dict:
    """show_pixel_meta's summary of an OME-Zarr, read by ozp (no pixels)."""
    from ome_zarr_pyramid import IO

    from eubi_bridge.core.output_store import _ozp_target_args
    try:
        target, storage_options = _ozp_target_args(str(path))
        pyr = IO().read_pyramid(target, include_labels=False, storage_options=storage_options)
        return _summary(pyr, path, 0)
    except Exception as exc:                            # noqa: BLE001 - shown per file
        logger.error(f"Failed to read OME-Zarr metadata from {path}: {exc}")
        return {"status": "error", "input_path": str(path), "error": str(exc)}


def _scenes(series, count: int) -> list:
    if series in (None, "all"):
        return list(range(count)) if series == "all" else [0]
    if isinstance(series, str) and "," in series:
        return [int(v) for v in series.split(",")]
    if isinstance(series, (list, tuple)):
        return [int(v) for v in series]
    return [int(series)]


def read_metadata(input_path: str, kwargs: dict):
    """show_pixel_meta's summary of a file as a conversion would write it: one dict, or
    a list of dicts (one per series) when several series are asked for."""
    from ome_zarr_pyramid import PyramidCollection

    from eubi_bridge.conversion.pyramid_path import (NAME, SERIES, _open_input,
                                                     apply_rgb_channels)
    from eubi_bridge.core.config_models import ConversionJob
    from eubi_bridge.core.micro_source import MicroReaderVersionError
    from eubi_bridge.core.output_store import _ozp_version
    from eubi_bridge.utils.path_utils import is_zarr_group

    path = str(input_path)
    if is_zarr_group(path):
        return read_ome_zarr(path)
    series = kwargs.get("series", 0)
    try:
        job = ConversionJob.from_kwargs(input_path=path, output_path="", kwargs=kwargs)
        f, _ = _open_input(job)
        try:
            values = sorted({k[SERIES] for k in f.indices})
            wanted = [s for s in _scenes(series, len(values)) if 0 <= s < len(values)]
            if not wanted:
                raise ValueError(f"no series {series!r}: the file has {len(values)} "
                                 f"(0..{len(values) - 1})")
            version = _ozp_version(job.conversion.zarr_format, job.conversion.ome_zarr_version)
            out = []
            for scene in wanted:
                index = min((k for k in f.indices if k[SERIES] == values[scene]),
                            key=lambda k: tuple(k.values()))
                image = f.image(tuple(index.items()))
                pyr = PyramidCollection.from_file(
                    SimpleNamespace(images=[image], path=path), version=version, name=NAME,
                    resolutions="base")[tuple(index.items())]
                pyr = apply_rgb_channels(pyr, image)
                shown = path if len(wanted) == 1 else f"{path} [series {scene}]"
                out.append(_summary(pyr, shown, scene))
        finally:
            f.close()
        return out[0] if len(out) == 1 else out
    except MicroReaderVersionError:                     # the install, not this file: as itself
        raise
    except Exception as exc:                            # noqa: BLE001 - shown per file
        logger.error(f"[MetadataReader] Error reading {path}: {exc}")
        return {"status": "error", "input_path": path, "error": str(exc)}
