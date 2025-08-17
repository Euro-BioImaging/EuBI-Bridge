#!/usr/bin/env python3
"""
bigtiff_to_ngff.py

Convert a BigTIFF to an OME-NGFF (OME-Zarr) dataset with optional simple downscaling
(subsample by selecting every n-th pixel). Designed to stream data using tifffile.aszarr.

Usage:
    python bigtiff_to_ngff.py input.tif outdir \
        --x-scale 2 --y-scale 2 --z-scale 1 --n-layers 3 --compressor blosc
"""

import os
import json
import argparse
import math
from pathlib import Path

import numpy as np
import tifffile
import zarr
from numcodecs import Blosc
from eubi_bridge.base.data_manager import ArrayManager



# --- Helpers -----------------------------------------------------------------

# def parse_args():
#     p = argparse.ArgumentParser(description="Convert BigTIFF -> OME-NGFF (OME-Zarr) with streaming and simple subsample downscaling.")
#     p.add_argument("input_tiff", help="Path to input TIFF (BigTIFF)")
#     p.add_argument("output_zarr_dir", help="Path to output directory (will be treated as zarr store root)")
#     p.add_argument("--dimension-order", default="tczyx", help="Dimension order of output (default: tczyx). Will be used when building metadata.")
#     p.add_argument("--dtype", default=None, help="Optional target dtype (e.g. uint8). If omitted, source dtype used.")
#     p.add_argument("--compressor", default="blosc", choices=["blosc","none"], help="Compressor")
#     p.add_argument("--x-scale", type=int, default=2, help="Downscale factor in X (select every n-th pixel)")
#     p.add_argument("--y-scale", type=int, default=2, help="Downscale factor in Y")
#     p.add_argument("--z-scale", type=int, default=2, help="Downscale factor in Z")
#     p.add_argument("--time-scale", type=int, default=1, help="Downscale factor in Time")
#     p.add_argument("--channel-scale", type=int, default=1, help="Downscale factor in Channel")
#     p.add_argument("--min-dimension-size", type=int, default=64, help="Stop building pyramid when smallest dimension < this")
#     p.add_argument("--n-layers", type=int, default=None, help="Max number of pyramid levels (None = unlimited until min size)")
#     p.add_argument("--auto-chunk", action="store_true", help="Let zarr choose chunking (otherwise use source chunking where possible)")
#     p.add_argument("--overwrite", action="store_true", help="Overwrite output directory if exists")
#     p.add_argument("--save-omexml", action="store_true", help="Attempt to copy/save OME-XML metadata (if present)")
#     p.add_argument("--zarr_format", type=int, default=2, help="Zarr format version")
#     return p.parse_args()

# def make_compressor(name):
#     if name == "blosc":
#         return Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
#     return None

def make_compressor(name):
    if name == "blosc":
        # Using the updated Blosc configuration for newer Zarr versions
        return Blosc(
            cname='zstd',  # compression algorithm
            clevel=3,      # compression level (1-9)
            shuffle=Blosc.SHUFFLE,  # or Blosc.NOSHUFFLE, Blosc.BITSHUFFLE
            blocksize=0    # 0 means automatic block size
        )
    return None  # No compression

def compute_next_shape(shape, scales, dim_map):
    # shape: dict mapping axis->size, scales dict axis->factor
    next_shape = {}
    for ax, size in shape.items():
        factor = scales.get(ax, 1)
        next_shape[ax] = max(1, math.ceil(size / factor))
    return next_shape

def smallest_spatial(shape, spatial_axes):
    return min(shape[ax] for ax in spatial_axes)

# --- Core conversion ---------------------------------------------------------


# --- NGFF metadata helpers (drop into your script) ----------------------------

def _ngff_axis_entry(ax):
    """Return axis metadata entry for NGFF axes list."""
    # map short axis to semantic type and default unit for spatial axes
    if ax == "t":
        return {"name": "t", "type": "time"}
    if ax == "c":
        return {"name": "c", "type": "channel"}
    if ax == "z":
        return {"name": "z", "type": "space", "unit": "micrometer"}
    if ax == "y":
        return {"name": "y", "type": "space", "unit": "micrometer"}
    if ax == "x":
        return {"name": "x", "type": "space", "unit": "micrometer"}
    # fallback
    return {"name": ax, "type": "unknown"}

def _build_coordinate_transform_for_level(axes_map, cumulative_scales, pixel_sizes_level=None):
    """
    Build a list of coordinateTransformations for one level.
    Each transform is a scale transform with 'scale' array following axes_map ordering.
    If pixel_sizes_level is provided, multiply the dimensionless scale by pixel size for spatial axes.
    """
    scale_list = []
    for ax in axes_map:
        # cumulative_scales[ax] = how many input pixels per output pixel for that axis
        scale_val = float(cumulative_scales.get(ax, 1))
        # If pixel sizes are provided for level 0, we scale spacial axes by pixel_size * scale_val
        if pixel_sizes_level and ax in pixel_sizes_level and pixel_sizes_level[ax] is not None:
            # physical scale in micrometers (or given unit) per output pixel
            scale_phys = float(pixel_sizes_level[ax]) * scale_val
            scale_list.append(scale_phys)
        else:
            # dimensionless pixel scaling (units implicit)
            scale_list.append(scale_val)
    return [{"type": "scale", "scale": scale_list}]

def build_multiscales_metadata_full(
    name,
    axes_map,                # tuple/list of axis letters in order of array dims, e.g. ('t','c','z','y','x')
    axes_sizes,              # dict axis -> size at level 0 (source)
    base_scales,             # dict axis -> integer base scale (e.g. {'x':2,'y':2,'z':1,...})
    n_levels,                # integer number of pyramid levels produced
    pixel_sizes_level0=None, # dict axis -> physical pixel size (units implied by axis entries); optional
    channel_names=None,      # optional list of channel names (length matches axes_sizes['c'] if present)
    version="0.4",
):
    """
    Build NGFF multiscales metadata for a pyramid with n_levels levels.
    - axes_map: tuple like ('t','c','z','y','x') ordered according to array dimension order.
    - base_scales: scale factor per axis between successive levels (e.g., 2 for x and y)
    - n_levels: how many levels you produced (level 0..n_levels-1)
    - pixel_sizes_level0: physical size per pixel on level 0 for spatial axes (e.g., {'x':0.130, 'y':0.130, 'z':0.5})
    - channel_names: optional channel metadata
    Returns: dict suitable for root.attrs.put({...})
    """
    # Build axes[] list
    axes = [_ngff_axis_entry(ax) for ax in axes_map]

    # Precompute cumulative scales for each level:
    # cumulative_scales[level][ax] = base_scales[ax]**level
    cumulative_scales = []
    for level in range(n_levels):
        cs = {}
        for ax in axes_map:
            s = int(base_scales.get(ax, 1)) ** level
            if s < 1:
                s = 1
            cs[ax] = s
        cumulative_scales.append(cs)

    # Build datasets list: each dataset -> {path: str(level), coordinateTransformations: [...]}
    datasets = []
    for level in range(n_levels):
        # coordinate transforms for this level: scale per axis
        # For coordinate transforms we want the scale from level 0 pixel coordinates to this level pixel coordinates.
        coord_transforms = _build_coordinate_transform_for_level(
            axes_map,
            cumulative_scales[level],
            pixel_sizes_level0
        )
        datasets.append({
            "path": str(level),
            "coordinateTransformations": coord_transforms
        })

    multiscale_entry = {
        "version": version,
        "name": name,
        "datasets": datasets,
        "type": "image",
        "axes": axes,
    }

    meta = {
        "multiscales": [multiscale_entry]
    }

    # Optionally add channels metadata at the top-level of the image multiscale if we have channels
    # NGFF expects channel metadata to live under 'channels' inside the multiscales entry for each image,
    # but commonly viewers accept channel metadata placed in multiscales,[object Object],['metadata']['omero'] or channels list.
    if "c" in axes_map and channel_names:
        # Build channels entries in same order as channels axis
        channels_meta = []
        for ci, name_ch in enumerate(channel_names):
            ch = {"label": name_ch, "active": True}
            # allow adding contrast limits / window fields if desired later
            channels_meta.append(ch)
        # Put channels under multiscale entry as 'metadata':{'channels': [...]}
        # This is pragmatic and compatible with many viewers; for strict OME you can embed under .zattrs multiscales,[object Object],['metadata'].
        multiscale_entry.setdefault("metadata", {})["channels"] = channels_meta

    return meta

# Convenience wrapper: collect pixel sizes from tifffile if available
def extract_pixel_sizes_from_tif(tiff_file_obj, **kwargs):
    """
    Try to extract pixel sizes (physical sizes) from a tifffile.TiffFile instance (tf).
    Returns dict axis->size (units implied).
    Currently tries:
      - tf.series,[object Object],levels,[object Object],shape or .axes/.scales if present
      - tf.pages,[object Object],tags if resolution tags exist (not guaranteed for Bio-Formats OME)
      - tf.ome_metadata parsing would be more robust (not done here)
    If no info available, returns None for axes.
    """
    pixel_sizes = {}
    path = tiff_file_obj.filehandle.path
    axes = tiff_file_obj.series[0].axes
    # path = f"/home/oezdemir/PycharmProjects/eubizarr1/test_tiff1/Channel/channel0_image.tif"
    manager = ArrayManager(path)
    for ax in axes:
        if ax.lower() == 's':
            pixel_sizes['c'] = manager.scaledict['c']
        elif ax.lower() in manager.scaledict:
            pixel_sizes[ax.lower()] = manager.scaledict[ax.lower()]
    return pixel_sizes

# --- End of NGFF helpers -----------------------------------------------------




def open_tiff_as_zarr_store(tiff_path):
    # tifffile.TiffFile(...).aszarr() returns a MutableMapping that can be passed to zarr.open
    tf = tifffile.TiffFile(tiff_path)
    store = tf.aszarr()  # streaming store view of TIFF pages/tiles
    return tf, store

def determine_axes_and_shape(zarr_root):
    # zarr_root is a mapping representing the source zarr-like store from tifffile
    # We need to find the main array (likely '0' or '/')
    # Simpler heuristic: open root with zarr.open
    try:
        arr = zarr.open(zarr.mapping(zarr_root), mode='r')
    except Exception:
        # fallback: try opening the first array key
        keys = list(zarr_root.keys())
        if not keys:
            raise RuntimeError("No arrays found in aszarr store")
        arr = zarr.open(zarr.mapping(zarr_root, keys=keys), mode='r')
    shape = arr.shape  # shape tuple
    chunks = arr.chunks
    dtype = arr.dtype
    return arr, shape, chunks, dtype

def axis_order_from_shape(tf: tifffile.TiffFile,
                          # prefer="tczyx"
                          **kwargs
                          ):
    axes = tf.series[0].axes.lower()
    if 's' in axes:
        axes = axes.replace('s', 'c')
    order = ''.join(axes)
    return order

def subsample_slice(scales, axes_map):
    # return tuple of slice objects for numpy/zarr indexing
    # axes_map: sequence of axis labels corresponding to array axes: e.g. ('t','c','z','y','x')
    slices = []
    for ax in axes_map:
        factor = scales.get(ax, 1)
        if factor <= 1:
            slices.append(slice(None))
        else:
            slices.append(slice(0, None, factor))
    return tuple(slices)

def write_level(grp,
                level_index,
                arr_in,
                axes_map,
                scales_for_level,
                compressor,
                dtype_target=None,
                auto_chunk=False,
                ):
    """
    Copy arr_in (zarr array or array-like) into grp at array path str(level_index)
    Subsampling according to scales_for_level map.
    """

    # compute slices for subsampling
    sl = subsample_slice(scales_for_level, axes_map)
    print(f"sl: {sl}")
    # open view of arr_in with slicing (this returns a zarr.Array if arr_in is zarr)
    view = arr_in[sl]
    print(f"view.shape: {view.shape}")
    # possibly cast dtype
    if dtype_target is not None:
        # create target array and copy chunk-wise to avoid memory spikes
        target_dtype = np.dtype(dtype_target)
    else:
        target_dtype = view.dtype
    # Configure compressor and chunks
    compressor_obj = compressor
    # use view.chunks if available
    chunks = getattr(view, 'chunks', None)
    print(f"chunks: {chunks}")
    if chunks is None:
        # choose heuristic chunking (per-plane)
        # put small chunks on x/y axes when possible
        chunks = tuple(min(s, 64) for s in view.shape)
    # create zarr array
    print("Writing level", level_index)
    print("  shape:", view.shape)
    print("  chunks:", chunks)
    print("  dtype:", target_dtype)
    print("  compressor:", compressor_obj)
    z = grp.require_dataset(
        name=str(level_index),  # Convert level_index to string here
        shape=view.shape,
        chunks=chunks,
        dtype=target_dtype,
        compressor=compressor_obj,
        overwrite=True,
        chunk_key_encoding={"name": "v2", "separator": "/"}
    )
    # copy in chunks: iterate over chunks by creating index ranges
    # Use numpy slicing with steps = 1 (already subsampled view), so read view into memory per chunk
    # We'll iterate over the first 3 axes if many dims to reduce number of loops
    shape = view.shape
    ndim = len(shape)
    # Create ranges for each axis
    ranges = [range(0, shape[i], chunks[i]) for i in range(ndim)]
    # Nested loops
    idx_template = [slice(None)] * ndim
    for idx0 in ranges[0]:
        for idx1 in ranges[1] if ndim > 1 else [None]:
            for idx2 in ranges[2] if ndim > 2 else [None]:
                # build slices
                if ndim == 1:
                    s = (slice(idx0, min(idx0 + chunks[0], shape[0])),)
                elif ndim == 2:
                    s = (slice(idx0, min(idx0 + chunks[0], shape[0])),
                         slice(idx1, min(idx1 + chunks[1], shape[1])))
                elif ndim == 3:
                    s = (slice(idx0, min(idx0 + chunks[0], shape[0])),
                         slice(idx1, min(idx1 + chunks[1], shape[1])),
                         slice(idx2, min(idx2 + chunks[2], shape[2])))
                elif ndim == 4:
                    for idx3 in ranges[3]:
                        s = (slice(idx0, min(idx0 + chunks[0], shape[0])),
                             slice(idx1, min(idx1 + chunks[1], shape[1])),
                             slice(idx2, min(idx2 + chunks[2], shape[2])),
                             slice(idx3, min(idx3 + chunks[3], shape[3])))
                        data = view[s]
                        if target_dtype != data.dtype:
                            data = data.astype(target_dtype, copy=False)
                        z[s] = data
                    continue
                elif ndim == 5:
                    for idx3 in ranges[3]:
                        for idx4 in ranges[4]:
                            s = (slice(idx0, min(idx0 + chunks[0], shape[0])),
                                 slice(idx1, min(idx1 + chunks[1], shape[1])),
                                 slice(idx2, min(idx2 + chunks[2], shape[2])),
                                 slice(idx3, min(idx3 + chunks[3], shape[3])),
                                 slice(idx4, min(idx4 + chunks[4], shape[4])))
                            data = view[s]
                            if target_dtype != data.dtype:
                                data = data.astype(target_dtype, copy=False)
                            z[s] = data
                    continue
                # for ndim 1..3
                data = view[s]
                if target_dtype != data.dtype:
                    data = data.astype(target_dtype, copy=False)
                z[s] = data
    return grp

# --- Main --------------------------------------------------------------------

def convert_bigtiff_to_omezarr(
    input_tiff,
    output_zarr_dir,
    dimension_order="tczyx",
    dtype=None,
    compressor="blosc",
    x_scale=2,
    y_scale=2,
    z_scale=2,
    time_scale=1,
    channel_scale=1,
    min_dimension_size=64,
    n_layers=None,
    auto_chunk=False,
    overwrite=False,
    save_omexml=False,
    zarr_format=2,
    **kwargs
):
    """Convert a BigTIFF to OME-Zarr format with optional downscaling.
    
    Args:
        input_tiff: Path to input TIFF (BigTIFF)
        output_zarr_dir: Path to output directory (will be treated as zarr store root)
        dimension_order: Dimension order of output (default: "tczyx")
        dtype: Optional target dtype (e.g. "uint8"). If None, source dtype is used.
        compressor_name: Compression method ("blosc" or "none")
        x_scale: Downscale factor in X (select every n-th pixel)
        y_scale: Downscale factor in Y
        z_scale: Downscale factor in Z
        time_scale: Downscale factor in Time
        channel_scale: Downscale factor in Channel
        min_dimension_size: Stop building pyramid when smallest dimension < this
        n_layers: Max number of pyramid levels (None = unlimited until min size)
        auto_chunk: Let zarr choose chunking if True
        overwrite: Overwrite output directory if exists
        save_omexml: Attempt to copy/save OME-XML metadata if present
        zarr_format: Zarr format version (2 or 3)
    """
    inp = Path(input_tiff)
    out = Path(output_zarr_dir)
    if out.exists():
        if overwrite:
            import shutil
            shutil.rmtree(out)
        else:
            raise FileExistsError(f"Output {out} exists. Set overwrite=True to replace.")
    out.mkdir(parents=True, exist_ok=True)

    compressor = make_compressor(compressor)
    # open TIFF as zarr store (streaming)
    tf, store = open_tiff_as_zarr_store(str(inp))

    # open source array
    src = zarr.open(store, mode='r', zarr_format=zarr_format)
    src_shape = src.shape
    src_chunks = getattr(src, "chunks", None)
    src_dtype = src.dtype
    # decide axes mapping
    axes_map = axis_order_from_shape(tf, prefer=dimension_order)
    # axes_map is a string like 'tczyx' sliced to len
    axes_map = tuple(axes_map)
    # create a store for output zarr (directory store)
    out_store = zarr.storage.LocalStore(str(out))
    root = zarr.group(store=out_store, overwrite=True, zarr_format=zarr_format)

    # initial shape dict mapping axis->size
    axes_sizes = {axes_map[i]: src_shape[i] for i in range(len(axes_map))}
    scales_per_level = []
    # initial scale factors for level 0 are 1
    scales_per_level.append({ax: 1 for ax in axes_map})

    # prepare downscaling parameters
    base_scales = {
        "x": x_scale,
        "y": y_scale,
        "z": z_scale,
        "t": time_scale,
        "c": channel_scale
    }
    # compute pyramid levels
    cur_sizes = axes_sizes.copy()
    level = 0
    while True:
        # write current level
        print(f"Writing level {level}, shapes: {[ (ax, cur_sizes[ax]) for ax in axes_map ]}")
        # compute cumulative subsampling factors to reach this level from original
        cumulative = {}
        for ax in axes_map:
            # product of base scale for each level (level times)
            cumulative[ax] = base_scales.get(ax, 1) ** level
            if cumulative[ax] < 1:
                cumulative[ax] = 1
        # write level
        grp = write_level(root,
                          level,
                          src,
                          axes_map,
                          cumulative,
                          compressor,
                          dtype_target=dtype,
                          auto_chunk=auto_chunk)

        # compute next sizes
        next_sizes = {}
        for ax in axes_map:
            next_sizes[ax] = max(1, math.ceil(cur_sizes[ax] / base_scales.get(ax, 1)))
        level += 1
        # stop conditions
        if n_layers is not None and level >= n_layers:
            break
        # if smallest spatial dimension smaller than min_dimension_size, stop
        spatial_axes = [ax for ax in axes_map if ax in ("x", "y", "z")]
        if smallest_spatial(cur_sizes, spatial_axes) < min_dimension_size:
            break
        # also break if sizes don't change (divide by 1)
        if all(next_sizes[ax] == cur_sizes[ax] for ax in axes_map):
            break
        cur_sizes = next_sizes

    # After pyramid creation loop finishes, determine how many levels were written
    n_levels_written = level

    # base_scales is already built earlier in script (same dict used for pyramid generation)
    # Optional: try to get pixel sizes from TIFF (tf is TiffFile instance)
    pixel_sizes_level0 = extract_pixel_sizes_from_tif(tf)

    # Optional: channel names extraction (basic attempt)
    channel_names = None
    if "c" in axes_map:
        # Try to inspect OME-XML for channel names
        try:
            import xml.etree.ElementTree as ET
            if tf.ome_metadata:
                root_xml = ET.fromstring(tf.ome_metadata)
                # find Channel elements under Pixels
                ch_elems = root_xml.findall(".//{*}Channel")
                if ch_elems:
                    channel_names = [c.attrib.get("Name", f"Channel {i}") for i, c in enumerate(ch_elems)]
        except Exception:
            channel_names = None

    # Build full NGFF multiscales metadata
    ngff_meta = build_multiscales_metadata_full(
        name=inp.name,
        axes_map=axes_map,
        axes_sizes=axes_sizes,
        base_scales=base_scales,
        n_levels=n_levels_written,
        pixel_sizes_level0=pixel_sizes_level0,
        channel_names=channel_names,
        version="0.4",
    )

    # Save to .zattrs at root (OME-NGFF expects multiscales in .zattrs)
    root.attrs.put(ngff_meta)

    # optionally extract OME-XML and save
    if save_omexml:
        try:
            omexml = tf.ome_metadata
            if omexml:
                (out / "OMEXML.xml").write_text(omexml)
        except Exception:
            pass

    # Clean up
    store.close()
    tf.close()

# def main():
#     """Command-line entry point that parses arguments and calls convert_bigtiff_to_omezarr."""
#     p = argparse.ArgumentParser(description="Convert BigTIFF -> OME-NGFF (OME-Zarr) with streaming and simple subsample downscaling.")
#     p.add_argument("input_tiff", help="Path to input TIFF (BigTIFF)")
#     p.add_argument("output_zarr_dir", help="Path to output directory (will be treated as zarr store root)")
#     p.add_argument("--dimension-order", default="tczyx", help="Dimension order of output (default: tczyx). Will be used when building metadata.")
#     p.add_argument("--dtype", default=None, help="Optional target dtype (e.g. uint8). If omitted, source dtype used.")
#     p.add_argument("--compressor", default="blosc", choices=["blosc","none"], help="Compressor")
#     p.add_argument("--x-scale", type=int, default=2, help="Downscale factor in X (select every n-th pixel)")
#     p.add_argument("--y-scale", type=int, default=2, help="Downscale factor in Y")
#     p.add_argument("--z-scale", type=int, default=2, help="Downscale factor in Z")
#     p.add_argument("--time-scale", type=int, default=1, help="Downscale factor in Time")
#     p.add_argument("--channel-scale", type=int, default=1, help="Downscale factor in Channel")
#     p.add_argument("--min-dimension-size", type=int, default=64, help="Stop building pyramid when smallest dimension < this")
#     p.add_argument("--n-layers", type=int, default=None, help="Max number of pyramid levels (None = unlimited until min size)")
#     p.add_argument("--auto-chunk", action="store_true", help="Let zarr choose chunking (otherwise use source chunking where possible)")
#     p.add_argument("--overwrite", action="store_true", help="Overwrite output directory if exists")
#     p.add_argument("--save-omexml", action="store_true", help="Attempt to copy/save OME-XML metadata (if present)")
#     p.add_argument("--zarr_format", type=int, default=2, help="Zarr format version")
#     args = p.parse_args()
#     convert_bigtiff_to_omezarr(
#         input_tiff=args.input_tiff,
#         output_zarr_dir=args.output_zarr_dir,
#         dimension_order=args.dimension_order,
#         dtype=args.dtype,
#         compressor_name=args.compressor,
#         x_scale=args.x_scale,
#         y_scale=args.y_scale,
#         z_scale=args.z_scale,
#         time_scale=args.time_scale,
#         channel_scale=args.channel_scale,
#         min_dimension_size=args.min_dimension_size,
#         n_layers=args.n_layers,
#         auto_chunk=args.auto_chunk,
#         overwrite=args.overwrite,
#         save_omexml=args.save_omexml,
#         zarr_format=args.zarr_format
#     )
# 
# if __name__ == "__main__":
#     main()