"""The conversion path of eubi-bridge 0.2.0: micro-reader -> ome_zarr_pyramid, no ArrayManager.

A file is opened with micro-reader, every image becomes an ome_zarr_pyramid ``Pyramid``
(``PyramidCollection.from_file``), the job's settings are applied to each Pyramid by small
stateless steps, and each output is ONE ``IO().write_pyramid`` call. Nothing goes through
``ArrayManager``, ``MicroReader`` or the ``ome_types`` round trip of the old micro path.

The DEFAULT for the files micro-reader reads (roadmap step 1d), for the cases it
reproduces (``unsupported``, ``NotYet``); every other job keeps the old path, with one log
line saying why. Each case was compared with the reference set (eubi-bridge 0.1.5) before
it was added. ``EUBI_NEW_PATH=0`` is a private way back to the old path until that is
deleted (roadmap step 6).
"""
from __future__ import annotations

import asyncio
import os
import pathlib
import re
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd

from eubi_bridge.utils.logging_config import get_logger

logger = get_logger(__name__)

#: the private switch
ENV = "EUBI_NEW_PATH"
#: files the new path converts so far (grows case by case with the reference set)
FORMATS = (".ome.tif", ".ome.tiff",
           ".tif", ".tiff", ".lsm", ".btf",                # step 1c, group 1
           ".nd2", ".lif", ".oir",                         # group 2
           ".czi",                                          # groups 3-5
           ".ims")                                          # group 6
#: image indices the new path selects by so far
INDICES = {"scene", "tile", "view", "illumination"}
#: eubi-bridge's channel_intensity_limits -> ozp's write_pyramid(windows=...)
WINDOWS = {"from_dtype": "dtype", "from_array": "data", "auto": "auto"}
#: the multiscale name eubi-bridge writes
NAME = "Series 0"


class NotYet(Exception):
    """This job needs something the new path does not do yet: the old path converts it."""


def enabled() -> bool:
    """On unless EUBI_NEW_PATH=0 (the old path, private, until step 6 deletes it)."""
    return os.environ.get(ENV, "1") != "0"


_warned_check_pixels = False


def _retired_settings(job) -> None:
    """Warn once per process for settings that no longer do anything here (roadmap
    decision 'D5: micro_check_pixels': retired; saved configs still load)."""
    global _warned_check_pixels
    if getattr(job.readers, "micro_check_pixels", False) and not _warned_check_pixels:
        _warned_check_pixels = True
        logger.warning(
            "micro_check_pixels no longer has an effect: micro-reader's pixels are not "
            "compared with the standard reader's any more. To compare, convert the file "
            "once more with pixel_reader='standard' and compare the two outputs.")


def unsupported(job) -> str:
    """Why the new path does not take this job yet, or ''. Settings only: the file's own
    content (its image indices) is checked when it is opened (``NotYet``)."""
    path = str(job.input_path)
    if not path.lower().endswith(FORMATS):
        return f"format not on the new path yet ({', '.join(FORMATS)})"
    if job.metadata.metadata_reader != "micro" or job.readers.pixel_reader != "micro":
        return "the standard reader was asked for"
    if job.readers.force_bioformats:
        return "force_bioformats"
    if job.conversion.writer == "eubi":
        return "writer='eubi'"
    if job.downscale.keep_existing_resolutions:
        return "keep_existing_resolutions"
    return ""


# -- the input ------------------------------------------------------------------------

def _positions(value, count: int, what: str) -> list:
    """eubi-bridge's index option ('all', an int, '0,2' or a list), as positions counted
    from 0 among the `count` the file has; positions out of range are skipped with a
    warning, and none left is an error."""
    if value == "all":
        wanted = list(range(count))
    elif isinstance(value, (list, tuple)):
        wanted = [int(v) for v in value]
    elif isinstance(value, str) and "," in value:
        wanted = [int(v.strip()) for v in value.split(",")]
    else:
        wanted = [int(value)]
    valid = [p for p in wanted if 0 <= p < count]
    if len(valid) < len(wanted):
        logger.warning(f"Skipping out-of-range {what} indices: {set(wanted) - set(valid)}")
    if not valid:
        raise ValueError(f"No valid {what} index in {sorted(set(wanted))}: the file has "
                         f"{count} {what}(s), so valid indices are 0..{count - 1}.")
    return valid


def _tile_positions(value, count: int, scene: int) -> list:
    """The tiles of one scene that the mosaic_tile_index option selects, counted from 0
    among THIS scene's tiles (0.1.5 used scene 0's count for every scene, roadmap 'Bug:
    scenes with different tile counts'); positions it lacks are skipped with a warning."""
    try:
        return _positions(value, count, "mosaic tile")
    except ValueError:
        logger.warning(f"Scene {scene} has {count} tile(s): none of {value!r} is one of them")
        return []


def _multi(value):
    """eubi-bridge's view / illumination option: 'all', a list, '0,2' or one index."""
    if value == "all" or isinstance(value, (list, tuple)):
        return value
    if isinstance(value, str) and "," in value:
        return [int(v.strip()) for v in value.split(",")]
    return int(value)


def _chosen(option, count: int) -> list:
    """Positions an option picks among `count` (views / illuminations: out of range
    ones are dropped silently, as 0.1.5 did)."""
    wanted = list(range(count)) if option == "all" else (
        [int(option)] if np.isscalar(option) else [int(v) for v in option])
    return [v for v in wanted if 0 <= v < count]


def _values(keys, name) -> list:
    """The sorted values of index `name` among `keys` ([None] when it is not an index)."""
    found = sorted({dict(k)[name] for k in keys if name in dict(k)})
    return found or [None]


def _key(scene, tile, view, illumination) -> tuple:
    out = [("scene", scene)]
    for name, value in (("tile", tile), ("view", view), ("illumination", illumination)):
        if value is not None:
            out.append((name, value))
    return tuple(out)


def _joined(parts, path):
    """The parts joined along c (ozp's concatenate: they must agree in pixel size,
    position, shape and type), or a clear error saying why they cannot be."""
    from ome_zarr_pyramid import concatenate
    try:
        return concatenate(parts, axis="c")
    except ValueError as exc:
        raise ValueError(
            f"{os.path.basename(path)}: its views / illuminations cannot be joined into one "
            f"image: {exc}. An OME-Zarr image has one pixel size and one position per axis "
            f"for all its channels, so images that differ in either cannot share it. "
            f"Convert without --concat_views / --concat_illuminations: each view / "
            f"illumination then becomes its own output, with its own pixel size and "
            f"position.") from exc


def _join_labels(pyr, group, parts, images, readers):
    """0.1.5's labels for joined channels: ``View{v}_Illu{i}_<name>`` (only the joined
    indices), view outermost. <name> is the file's own channel name; an unnamed channel
    takes the name of the same channel in the view's first image, else ``Ch{i}``."""
    def own_names(image, count):
        named = [getattr(ch, "name", None) for ch in image.metadata.channels]
        return named if len(named) == count else [None] * count

    labels, n, first_of_view = {}, 0, {}
    for (view, illumination), part, image in zip(group, parts, images):
        prefix = "_".join(([f"View{view}"] if readers.concat_views else [])
                          + ([f"Illu{illumination}"] if readers.concat_illuminations
                             else [])) + "_"
        count = len(part.meta.metadata["omero"]["channels"])
        names = own_names(image, count)
        fallback = first_of_view.setdefault(view, names)
        for idx, name in enumerate(names):
            name = name or (fallback[idx] if idx < len(fallback) else None) or f"Ch{idx}"
            labels[n] = {"label": f"{prefix}{name}"}
            n += 1
    return pyr.set_channels(labels)


def _load_input_collection(job):
    """``(file, entries)``: the opened micro-reader file and one entry per output, each
    ``SimpleNamespace(pyramid, image, scene, tile, series_path, basename, name_scene,
    name_tile)`` in eubi's terms (scene / tile / view / illumination = positions among the
    file's own values), named as 0.1.5 names them."""
    import micro_reader
    from ome_zarr_pyramid import PyramidCollection

    from eubi_bridge.core.micro_source import RENAME, set_read_concurrency
    from eubi_bridge.core.ozp_writer import _ozp_version

    set_read_concurrency(job.cluster.decode_threads)
    path = str(job.input_path)
    readers = job.readers
    tiles = "stitched" if readers.as_mosaic else "separate"
    try:
        f = micro_reader.open(path, tiles=tiles, rename=RENAME, axes="tczyx",
                              samples="channels")
    except micro_reader.UnsupportedFormatError as exc:
        raise NotYet(f"micro-reader does not read it ({exc})") from exc
    try:
        names = set().union(*(set(k) for k in f.indices)) - {"resolution"}
        if names - INDICES:
            raise NotYet(f"image indices {sorted(names - INDICES)}")
        if names - {"scene"} and not path.lower().endswith(".czi"):
            # 0.1.5 reads tiles / views / illuminations outside a CZI with the standard
            # reader
            raise NotYet(f"{sorted(names - {'scene'})} outside a CZI")
        version = _ozp_version(job.conversion.zarr_format, job.conversion.ome_zarr_version)
        coll = PyramidCollection.from_file(f, version=version, name=NAME,
                                           resolutions="base")
        images = {tuple((k, v) for k, v in dict(img.index).items() if k != "resolution"): img
                  for img in f.images if int(dict(img.index).get("resolution", 0)) == 0}
        stem = job.resolved_basename or os.path.basename(path).split(".")[0]
        file = pathlib.Path(path)

        scene_values = _values(coll, "scene")
        scenes = _positions(readers.scene_index, len(scene_values), "scene")
        n_scenes = len(scene_values)
        concat_views, concat_illus = readers.concat_views, readers.concat_illuminations
        view_option = _multi(readers.view_index)
        illu_option = _multi(readers.illumination_index)
        # 0.1.5 iterates views / illuminations only when asked to (unary_worker)
        by_view = (view_option != 0 or concat_views or isinstance(view_option, list))
        by_illu = (illu_option != 0 or concat_illus or isinstance(illu_option, list))
        n_views_file = n_illus_file = 1
        if by_view or by_illu:
            view_option = view_option if by_view else 0
            illu_option = illu_option if by_illu else 0
            first = [k for k in coll if dict(k)["scene"] == scene_values[scenes[0]]]
            n_views_file = len([v for v in _values(first, "view") if v is not None]) or 1
            n_illus_file = len([v for v in _values(first, "illumination")
                                if v is not None]) or 1
            if n_views_file <= 1 and n_illus_file <= 1:
                by_view = by_illu = False             # nothing to iterate

        option = readers.mosaic_tile_index
        entries, tiled = [], False
        for scene in scenes:
            here = [k for k in coll if dict(k)["scene"] == scene_values[scene]]
            tile_values = _values(here, "tile")
            views, illus = _values(here, "view"), _values(here, "illumination")
            tiled = tiled or len(tile_values) > 1
            if option is None:
                # no tile asked for: the scene's first tile (a file without tiles: the
                # whole image), and no tile in the name or the acquisition block
                picked = [None]
            else:
                picked = _tile_positions(option, len(tile_values), scene)
            for tile in picked:
                tile_value = tile_values[tile or 0]
                if not (by_view or by_illu):
                    key = _key(scene_values[scene], tile_value, views[0], illus[0])
                    entries.append(SimpleNamespace(
                        pyramid=coll[key], image=images[key], scene=scene, tile=tile,
                        series_path=f"{path}_{scene}" + (f"_tile{tile}" if tile else ""),
                        basename=job.resolved_basename,
                        name_scene=scene if n_scenes > 1 else None, name_tile=tile))
                    continue
                # views / illuminations, as 0.1.5's load_views_illuminations
                tile_named = tile if tile is not None and len(tile_values) > 1 else None
                combos = [(v, i) for v in _chosen(view_option, len(views))
                          for i in _chosen(illu_option, len(illus))]
                if concat_views and concat_illus:
                    groups = [combos] if combos else []
                elif concat_views:
                    groups = [[c for c in combos if c[1] == i]
                              for i in dict.fromkeys(c[1] for c in combos)]
                elif concat_illus:
                    groups = [[c for c in combos if c[0] == v]
                              for v in dict.fromkeys(c[0] for c in combos)]
                else:
                    groups = [[c] for c in combos]
                for group in groups:
                    view, illumination = group[-1]
                    suffix = ""
                    if n_views_file > 1:
                        suffix += "_view_concat" if concat_views else f"_view{view}"
                    if n_illus_file > 1:
                        suffix += "_illu_concat" if concat_illus else f"_illu{illumination}"
                    if tile_named is not None:
                        suffix += f"_tile{tile_named}"
                    keys = [_key(scene_values[scene], tile_value, views[v], illus[i])
                            for v, i in group]
                    if concat_views or concat_illus:
                        if any("s" in str(getattr(images[k].source, "axes", "")) for k in keys):
                            raise NotYet("RGB views / illuminations joined")
                        parts = [coll[k] for k in keys]
                        pyr = _join_labels(_joined(parts, path), group, parts,
                                           [images[k] for k in keys], readers)
                        # 0.1.5 kept the LAST joined image's wavelengths and position
                        pyr.acquisition = getattr(parts[-1], "acquisition", None)
                    else:
                        pyr = coll[keys[0]]
                    entries.append(SimpleNamespace(
                        pyramid=pyr, image=images[keys[-1]], scene=scene, tile=tile_named,
                        series_path=str(file.parent / (file.stem + suffix + file.suffix)),
                        basename=stem + suffix,
                        name_scene=scene if n_scenes > 1 else None, name_tile=None))
        if not tiled:
            for entry in entries:
                entry.name_tile = None
        if not entries:
            raise ValueError(f"No valid mosaic tile index in {option!r} for "
                             f"{os.path.basename(path)}: tiles only exist when reading an "
                             f"unstitched mosaic (as_mosaic=False).")
        return f, entries
    except BaseException:
        f.close()
        raise


# -- the job's settings, one step each (Pyramid in, Pyramid out) -------------------------

def _shim(pyr) -> SimpleNamespace:
    """What conversion_worker's parse_* helpers read from an ArrayManager, from a
    Pyramid, so both paths parse the job's settings with the same code."""
    meta = pyr.meta
    axes = meta.axis_order
    scales = meta.get_scale(meta.resolution_paths[0])
    units = dict(zip(axes, meta.unit_list))
    return SimpleNamespace(axes=axes, scaledict=dict(zip(axes, scales)),
                           unitdict={ax: u for ax, u in units.items() if u})


def _origin(image, units: dict) -> dict:
    """The image's stage position in its scale units, as eubi-bridge's ``origindict``
    (micro_source.micro_extras): every axis the file positions, or {} when a unit does
    not convert."""
    origin = {}
    for axis, q in image.metadata.position.items():
        try:
            origin[axis] = float(q.to(units.get(axis) or "micrometer").value)
        except KeyError:
            return {}
    return origin


def apply_rgb_channels(pyr, image):
    """eubi-bridge's names for RGB samples read as channels (sample_channels): one channel
    of S samples -> S plain channels (``Channel {i}``, white, as eubi always wrote); C > 1
    channels -> ``"<name> R"`` / ``G`` / ``B`` (``A``), tinted red / green / blue."""
    from eubi_bridge.core.sample_channels import SAMPLE_COLORS, SAMPLE_LABELS
    source = getattr(image, "source", image)
    axes = str(getattr(source, "axes", ""))
    if "s" not in axes:
        return pyr
    samples = int(source.shape[axes.index("s")])
    channels = int(source.shape[axes.index("c")]) if "c" in axes else 1
    if channels == 1:
        overrides = {i: {"label": f"Channel {i}", "color": "FFFFFF"} for i in range(samples)}
    else:
        labels = SAMPLE_LABELS.get(samples, tuple(str(i) for i in range(samples)))
        colors = SAMPLE_COLORS.get(samples)
        described = list(image.metadata.channels)
        overrides = {}
        for c in range(channels):
            name = described[c].name if len(described) == channels else None
            for i in range(samples):
                color = colors[i].lstrip("#").upper() if colors else "FFFFFF"
                overrides[c * samples + i] = {"label": f"{name or f'Channel {c}'} {labels[i]}",
                                              "color": color}
    return pyr.set_channels(overrides)


def apply_axis_metadata(pyr, job):
    """Scale and unit overrides: the config's values, a batch table row's (job.extra)
    winning. Unset ones keep what the file says."""
    from eubi_bridge.conversion.conversion_worker import AXIS_PARAM_MAP, _axis_metadata_kwargs
    given = _axis_metadata_kwargs(job)
    scales, units = {}, {}
    for ax in pyr.meta.axis_order:
        names = AXIS_PARAM_MAP.get(ax)
        if names is None:
            continue
        for name, out in ((names[2], scales), (names[4], units)):
            if name is not None and name in given and not pd.isna(given[name]):
                out[ax] = given[name]
    if scales:
        pyr.update_scales(**{ax: float(v) for ax, v in scales.items()})
    if units:
        pyr.update_units(**units)
    return pyr


def apply_channels(pyr, job):
    """Channel labels and colours from the job (``"0,DAPI;1,GFP"``; colour names or
    hex); a channel out of range is skipped with a warning, '' and 'auto' leave it."""
    from eubi_bridge.utils.metadata_utils import ChannelMap, ChannelParser
    meta = job.metadata
    parse = ChannelParser(None)._parse_indexed_string
    labels = parse(job.extra.get("channel_labels", meta.channel_labels))
    colors = parse(job.extra.get("channel_colors", meta.channel_colors))
    if not labels and not colors:
        return pyr
    axes = pyr.meta.axis_order
    count = pyr.base_array.shape[axes.index("c")] if "c" in axes else 1
    names = ChannelMap()
    overrides = {}
    for idx in sorted(set(labels) | set(colors)):
        if not 0 <= idx < count:
            logger.warning(f"Channel {idx} out of range [0:{count - 1}] for "
                           f"{job.input_path}, skipping")
            continue
        fields = {}
        if labels.get(idx, "") not in ("", "auto"):
            fields["label"] = labels[idx]
        if colors.get(idx, "") not in ("", "auto"):
            fields["color"] = names[colors[idx]] or colors[idx]
        if fields:
            overrides[idx] = fields
    return pyr.set_channels(overrides) if overrides else pyr


def apply_squeeze_and_crop(pyr, job):
    """``squeeze`` (size-1 axes but y and x), then the crop ranges (``"start,stop"``,
    stop excluded) on the axes that remain."""
    from eubi_bridge.conversion.conversion_worker import _extract_cropping_slices
    conv = job.conversion
    if conv.squeeze:
        pyr = pyr.squeeze()
    ranges = _extract_cropping_slices({
        'time_range': conv.time_range, 'channel_range': conv.channel_range,
        'z_range': conv.z_range, 'y_range': conv.y_range, 'x_range': conv.x_range})
    axes = pyr.meta.axis_order
    crop = {key[0]: slice(*value) for key, value in ranges.items() if key[0] in axes}
    return pyr.isel(**crop) if crop else pyr


def apply_downscale(pyr, job):
    """The downscale plan (computed when written): factors per axis, level count or
    smallest size, method, and the smart (isotropic) first-level factors."""
    from eubi_bridge.conversion.conversion_worker import (parse_scale_factors,
                                                          parse_smart_scale_factors)
    ds = job.downscale
    shim = _shim(pyr)
    return pyr.downscale(scale_factor=tuple(parse_scale_factors(shim, job)),
                         n_layers=ds.n_layers, min_dimension_size=ds.min_dimension_size,
                         smart_scale_factor=parse_smart_scale_factors(shim, job),
                         downscale_method=ds.downscale_method)


def acquisition_metadata(entry, pyr, job, origin: dict) -> Optional[dict]:
    """conversion_worker.build_acquisition_metadata on the new path: what NGFF has no
    field for (wavelengths, other index values), plus, when one acquisition is split
    into several outputs, where this output came from."""
    flag = getattr(job.conversion, "export_acquisition_metadata", None)
    extras = dict(getattr(pyr, "acquisition", None) or {})
    base = os.path.basename(entry.series_path)
    is_split = (entry.tile is not None or bool(origin)
                or any(m in base for m in ("_view", "_illu", "_tile", "_scene")))
    if flag is False:
        return None
    if flag is None and not is_split:
        return extras or None
    meta = dict(extras)
    meta["scene_index"] = entry.scene
    if entry.tile is not None:
        meta["tile_index"] = entry.tile
    for key, pattern in (("view_index", r"_view(\d+)"), ("illumination_index", r"_illu(\d+)")):
        found = re.search(pattern, base)
        if found:
            meta[key] = int(found.group(1))
    if origin:
        unit = entry.units.get("x", "micrometer")
        meta["origin_um" if unit in ("micrometer", "µm", "um") else f"origin_{unit}"] = dict(origin)
    meta["source_path"] = str(job.input_path or "")
    return meta or None


def ome_xml(pyr, dtype) -> str:
    """The small OME-XML eubi-bridge saves beside an output (ArrayManager.save_omexml):
    shape, axes, pixel sizes and units, dtype and channel names of what is written."""
    from eubi_bridge.core.data_manager import PFFImageMeta, create_ome_xml
    meta = pyr.meta
    axes = meta.axis_order
    scales = dict(zip(axes, meta.get_scale(meta.resolution_paths[0])))
    units = {ax: u for ax, u in zip(axes, meta.unit_list) if u}
    channels = (meta.metadata.get("omero") or {}).get("channels") or []
    ome = create_ome_xml(
        image_shape=tuple(pyr.base_array.shape), axis_order=axes,
        **{f"pixel_size_{ax}": scales.get(ax, 1) for ax in "tzyx"},
        **{f"unit_{ax}": units.get(ax, "second" if ax == "t" else "micrometer")
           for ax in "tzyx"},
        dtype=str(np.dtype(dtype)), channel_names=[ch.get("label") for ch in channels])
    pixels = ome.images[0].pixels
    pixels.model_fields_set.update(PFFImageMeta.essential_omexml_fields - pixels.model_fields_set)
    ome.images[0].pixels = pixels
    return ome.to_xml()


# -- one output -----------------------------------------------------------------------

def _write(pyr, output_path: str, job, *, scene_budget_mb: float, region_size_mb: float,
           xml: Optional[str]) -> None:
    """ONE write_pyramid call with the job's storage settings (eubi-bridge decides the
    chunk shape and shards; ozp writes)."""
    from ome_zarr_pyramid import IO

    from eubi_bridge.conversion.conversion_worker import (_threads_per_scene, parse_chunks,
                                                          parse_shard_coefs)
    from eubi_bridge.core.ozp_writer import _chunks, _ozp_target_args
    conv, clus = job.conversion, job.cluster
    arr = pyr.base_array
    dtype = np.dtype(conv.dtype or arr.dtype)
    shim = _shim(pyr)
    chunks = _chunks(arr, pyr.meta.axis_order, conv.auto_chunk, parse_chunks(shim, job),
                     dtype, conv.target_chunk_mb)
    shards = parse_shard_coefs(shim, job)
    target, storage_options = _ozp_target_args(output_path)
    logger.info(f"{ENV}: {output_path} (chunks {chunks})")
    IO().write_pyramid(
        pyr, target, storage_options=storage_options, overwrite=conv.overwrite,
        max_threads=_threads_per_scene(clus), region_size_mb=float(region_size_mb),
        memory_budget_mb=scene_budget_mb, io_backend=conv.io_backend,
        chunk_shape=chunks,
        shard_coefficients=tuple(shards) if int(pyr.meta.zarr_format) == 3 else None,
        compressor=conv.compressor, compressor_params=conv.compressor_params or {},
        dtype=dtype if dtype != np.dtype(arr.dtype) else None,
        windows=WINDOWS[job.metadata.channel_intensity_limits],
        ome_xml=xml)


async def _convert_entry(entry, output_path: str, job, sem: asyncio.Semaphore) -> None:
    """One output: the job's settings applied to the entry's Pyramid, then written."""
    from eubi_bridge.conversion.conversion_worker import (_memory_plan_for,
                                                          _warn_unused_writer_settings)
    from eubi_bridge.core.ozp_writer import _eubi_attrs
    from eubi_bridge.core.writers import store_exists
    async with sem:
        conv = job.conversion
        if not conv.overwrite and store_exists(output_path):
            raise FileExistsError(
                f"Output already exists: '{output_path}'. Refusing to convert "
                f"with overwrite=False — the existing dataset was left untouched. "
                f"Enable overwrite to replace it, or choose a different output.")
        pyr = entry.pyramid
        meta = pyr.meta
        entry.units = {ax: u for ax, u in zip(meta.axis_order, meta.unit_list) if u}
        origin = _origin(entry.image, entry.units)
        acquisition = acquisition_metadata(entry, pyr, job, origin)

        pyr = apply_rgb_channels(pyr, entry.image)
        pyr = apply_axis_metadata(pyr, job)
        pyr = apply_channels(pyr, job)
        pyr = apply_squeeze_and_crop(pyr, job)
        axes = pyr.meta.axis_order
        if origin and not any(any(t) for t in _translations(pyr)):
            # an all-zero position is written too: the tile at (0, 0) belongs to the
            # same coordinate system as its neighbours (parse_translation)
            for path in pyr.meta.resolution_paths:
                pyr.meta.update_translation(path, [origin.get(ax, 0.0) for ax in axes])
        dtype = np.dtype(conv.dtype or pyr.base_array.dtype)
        xml = ome_xml(pyr, dtype) if conv.save_omexml else None
        pyr = apply_downscale(pyr, job)
        pyr.meta.set_custom("eubi_bridge", _eubi_attrs(acquisition))

        scene_budget_mb, region_size_mb = _memory_plan_for(pyr.base_array, job)
        _warn_unused_writer_settings(job.cluster)
        await asyncio.to_thread(_write, pyr, output_path, job,
                                scene_budget_mb=scene_budget_mb,
                                region_size_mb=region_size_mb, xml=xml)


def _translations(pyr) -> list:
    meta = pyr.meta
    out = []
    for path in meta.resolution_paths:
        try:
            out.append(list(meta.get_translation(path) or []))
        except Exception:                                   # noqa: BLE001 - none set
            out.append([])
    return out


async def convert(job) -> bool:
    """Convert the job on the new path; False (nothing written) when it is not a case
    the new path takes yet, so the caller uses the old one."""
    from eubi_bridge.conversion.conversion_worker import (_convert_safely,
                                                          _generate_output_path,
                                                          _report_failures)
    reason = unsupported(job)
    if not reason:
        try:
            f, entries = await asyncio.to_thread(_load_input_collection, job)
        except NotYet as exc:
            reason = str(exc)
    if reason:
        logger.info(f"{ENV}: {job.input_path}: old path ({reason})")
        return False
    logger.info(f"{ENV}: {job.input_path}: new path, {len(entries)} output(s)")
    _retired_settings(job)
    try:
        sem = asyncio.Semaphore(job.cluster.max_concurrent_scenes)
        tasks = []
        for entry in entries:
            out = _generate_output_path(job.output_path, entry.series_path,
                                        entry.name_scene, entry.name_tile,
                                        resolved_basename=entry.basename)
            tasks.append(asyncio.create_task(
                _convert_safely(_convert_entry(entry, out, job, sem), out)))
        results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        f.close()
    _report_failures(results, job)               # the count: the outputs asked for
    return True
