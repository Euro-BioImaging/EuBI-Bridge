"""
Write annotation/classification masks as NGFF v0.4 label layers.

The output is written alongside the source OME-Zarr at:
    <zarr_path>/labels/<label_name>/0    (zarr array)
    <zarr_path>/labels/<label_name>/.zattrs  (multiscales + image-label metadata)
    <zarr_path>/labels/.zattrs           (list of label names)
    <zarr_path>/.zattrs                  (labels key added / updated)
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import zarr


async def write_ngff_labels(
    zarr_path: str,
    label_name: str,
    mask_array: np.ndarray,
    axes: str,
    scales: Sequence[float],
    units: Sequence[Optional[str]],
    class_colors: Sequence[Tuple[int, int, int]],
    zarr_format: int = 2,
) -> None:
    """Write a segmentation mask as an NGFF labels layer.

    Parameters
    ----------
    zarr_path:
        Root path of the source OME-Zarr (the label is written *inside* it).
    label_name:
        Name of the label dataset (e.g. ``"annotations"``).
    mask_array:
        nD uint8 array with shape matching the source image (no C axis).
        ``0`` = unannotated / background; ``1..N`` = class indices.
    axes:
        Axis order string without 'c', e.g. ``"tzyx"`` or ``"zyx"``.
    scales:
        Per-axis physical pixel sizes (same order as *axes*).
    units:
        Per-axis unit strings (same order as *axes*). May contain ``None``.
    class_colors:
        List of ``(r, g, b)`` tuples, one per class (index 0 → class label 1).
    zarr_format:
        Zarr format version (2 or 3).
    """
    # Open source group in append mode
    store = zarr.open_group(zarr_path, mode='a')

    # Create labels hierarchy
    labels_gr = store.require_group('labels')
    label_gr  = labels_gr.require_group(label_name)

    # Choose chunks: spatial dims get 256, non-spatial dims get 1
    spatial = set('zyx')
    chunks = tuple(
        256 if ax in spatial else 1
        for ax in axes.lower()
    )
    # Clamp to array shape
    chunks = tuple(min(c, s) for c, s in zip(chunks, mask_array.shape))

    # Write mask as zarr array at path '0'
    label_gr.require_dataset(
        '0',
        shape=mask_array.shape,
        dtype=np.uint8,
        chunks=chunks,
        overwrite=True,
    )
    label_gr['0'][:] = mask_array

    # ── image-label metadata ──────────────────────────────────────────────────
    colors = [
        {'label-value': int(i + 1), 'rgba': [int(r), int(g), int(b), 255]}
        for i, (r, g, b) in enumerate(class_colors)
    ]
    label_gr.attrs['image-label'] = {
        'colors': colors,
        'source': {'image': '../../'},
    }

    # ── multiscales metadata ──────────────────────────────────────────────────
    coord_transforms = [
        {
            'type': 'scale',
            'scale': [float(s) for s in scales],
        }
    ]
    axes_meta = []
    for ax, unit in zip(axes.lower(), units):
        entry: dict = {'name': ax}
        if ax == 't':
            entry['type'] = 'time'
            if unit:
                entry['unit'] = unit
        elif ax in 'zyx':
            entry['type'] = 'space'
            if unit:
                entry['unit'] = unit
        else:
            entry['type'] = 'space'
        axes_meta.append(entry)

    label_gr.attrs['multiscales'] = [{
        'version': '0.4',
        'axes': axes_meta,
        'datasets': [
            {
                'path': '0',
                'coordinateTransformations': coord_transforms,
            }
        ],
    }]

    # ── register label name in parent groups ──────────────────────────────────
    # labels/.zattrs
    existing = list(labels_gr.attrs.get('labels', []))
    if label_name not in existing:
        existing.append(label_name)
    labels_gr.attrs['labels'] = existing

    # root .zattrs — add 'labels' key if not present
    root_attrs = dict(store.attrs)
    root_labels = list(root_attrs.get('labels', []))
    if label_name not in root_labels:
        root_labels.append(label_name)
    root_attrs['labels'] = root_labels
    store.attrs.update(root_attrs)
