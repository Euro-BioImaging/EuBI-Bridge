"""
Spatial crop of an existing OME-Zarr to a new OME-Zarr.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import dask.array as da
import zarr

from eubi_bridge.core.writers import store_multiscale_async
from eubi_bridge.ngff.multiscales import Pyramid


async def crop_ome_zarr(
    source_path: str,
    output_path: str,
    crop_ranges: Dict[str, Tuple[int, int]],
    zarr_format: int = 2,
    scale_factors: Optional[Tuple[int, ...]] = None,
    n_layers: Optional[int] = None,
    overwrite: bool = False,
) -> None:
    """Crop an OME-Zarr and write the result as a new multi-level OME-Zarr.

    Parameters
    ----------
    source_path:
        Path to the source OME-Zarr group.
    output_path:
        Destination path for the cropped OME-Zarr.
    crop_ranges:
        Per-axis (start, stop) pairs, e.g. ``{'z': (5, 15), 'y': (100, 300), 'x': (200, 400)}``.
        Missing axes keep their full extent.
    zarr_format:
        Output zarr format (2 or 3).
    scale_factors:
        Downsample factors per spatial axis for pyramid generation.
        If None, inferred from the source pyramid (falls back to (2, 2) for YX).
    n_layers:
        Number of pyramid levels in the output. If None, matches the source.
    overwrite:
        Whether to overwrite an existing output.
    """
    pyr = Pyramid(source_path)
    axes: str = pyr.axes  # e.g. 'tczyx'

    # Build slice tuple ordered by axes string
    slices = []
    for ax in axes:
        if ax in crop_ranges:
            start, stop = crop_ranges[ax]
            slices.append(slice(start, stop))
        else:
            slices.append(slice(None))
    slices = tuple(slices)

    # Get base array as dask
    base = pyr.base_array
    if isinstance(base, zarr.Array):
        base = da.from_zarr(base)
    elif not isinstance(base, da.Array):
        import numpy as np
        base = da.from_array(base, chunks=base.shape)

    cropped = base[slices]

    # Physical metadata — pixel sizes stay the same after cropping
    base_path = pyr.meta.resolution_paths[0]
    base_scales = pyr.meta.get_scale(base_path)   # tuple of floats, one per axis
    unit_list = list(pyr.meta.unit_list or [])

    # Channel metadata — slice if 'c' was cropped
    channel_meta = None
    try:
        channels = pyr.meta.metadata.get('omero', {}).get('channels', [])
        if channels:
            if 'c' in crop_ranges and 'c' in axes:
                c_start, c_stop = crop_ranges['c']
                channels = channels[c_start:c_stop]
            channel_meta = channels
    except Exception:
        pass

    # Infer scale_factors from source pyramid if not provided
    if scale_factors is None:
        if len(pyr.meta.resolution_paths) > 1:
            p0 = pyr.meta.resolution_paths[0]
            p1 = pyr.meta.resolution_paths[1]
            s0 = pyr.meta.get_scale(p0)
            s1 = pyr.meta.get_scale(p1)
            scale_factors = tuple(
                max(1, round(b / a)) for a, b in zip(s0, s1)
            )
        else:
            # Single-level source: produce a 2-level output with YX 2× downscaling
            n_spatial = sum(1 for ax in axes if ax in 'zyx')
            scale_factors = tuple([1] * (len(axes) - n_spatial) + [2] * n_spatial)

    if n_layers is None:
        n_layers = max(1, len(pyr.meta.resolution_paths))

    await store_multiscale_async(
        arr=cropped,
        output_path=str(output_path),
        axes=list(axes),
        scales=[base_scales],
        units=unit_list,
        zarr_format=zarr_format,
        scale_factors=scale_factors,
        n_layers=n_layers,
        channel_meta=channel_meta,
        overwrite=overwrite,
    )
