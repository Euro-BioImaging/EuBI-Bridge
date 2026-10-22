"""The conversion path of eubi-bridge 0.2.0: micro-reader -> ome_zarr_pyramid.

A file is opened with micro-reader (else with the older readers, ``legacy_image.py``),
every image becomes an ome_zarr_pyramid ``Pyramid`` (``PyramidCollection.from_file``), the
job's settings are applied to each Pyramid by small stateless steps, and each output is
ONE ``IO().write_pyramid`` call.

Every case was compared with eubi-bridge 0.1.x's output (the reference set, and the golden
files of tests/_golden.py) before the 0.1.x path was retired (roadmap step 6; kept for
reference in ``legacy_0_1/``).
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

#: files micro-reader reads here (the others go to the older readers, as in 0.1.5)
FORMATS = (".ome.tif", ".ome.tiff",
           ".tif", ".tiff", ".lsm", ".btf",                # step 1c, group 1
           ".nd2", ".lif", ".oir",                         # group 2
           ".czi",                                          # groups 3-5
           ".ims",                                          # group 6
           # plate index files (0.3.2): Harmony Index(.idx).xml, CellVoyager .wpi / .mlf,
           # MetaXpress .HTD, IN Cell .xdce; a file micro-reader does not read goes to
           # the older readers as before
           ".xml", ".wpi", ".mlf", ".htd", ".xdce")
#: micro-reader's name of the series index (micro-reader 0.1.0; it was "scene"). The
#: older readers (legacy_image) and the aggregative path name their index the same, so
#: one name serves every file; eubi's own option for it is series_index
SERIES = "series"
#: image indices outputs are selected by
INDICES = {SERIES, "tile", "view", "illumination"}
#: the image indices of a plate file (micro-reader's plates=True)
PLATE_INDICES = {"plate", "acquisition", "well", "field"}
#: plate index files whose name says nothing about the plate: outputs take the folder's
GENERIC_INDEX_NAMES = {"index", "measurementdata"}
#: eubi-bridge's channel_intensity_limits -> ozp's write_pyramid(windows=...)
WINDOWS = {"from_dtype": "dtype", "from_array": "data", "auto": "auto"}
#: the multiscale name eubi-bridge writes
NAME = "Series 0"


class NotYet(Exception):
    """micro-reader does not take this file here: the older readers read it."""


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


def _micro_reason(job) -> str:
    """Why micro-reader does not read this job's file here (the older readers do), or ''."""
    if not str(job.input_path).lower().endswith(FORMATS):
        return f"format not on micro-reader's route yet ({', '.join(FORMATS)})"
    if job.metadata.metadata_reader != "micro" or job.readers.pixel_reader != "micro":
        return "the standard reader was asked for"
    if job.readers.force_bioformats:
        return "force_bioformats"
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
        counted = what if what.endswith("s") else f"{what}(s)"       # "series" stays
        raise ValueError(f"No valid {what} index in {sorted(set(wanted))}: the file has "
                         f"{count} {counted}, so valid indices are 0..{count - 1}.")
    return valid


def _tile_positions(value, count: int, scene: int) -> list:
    """The tiles of one series that the mosaic_tile_index option selects, counted from 0
    among THIS series' tiles (0.1.5 used series 0's count for every series, roadmap 'Bug:
    scenes with different tile counts'); positions it lacks are skipped with a warning."""
    try:
        return _positions(value, count, "mosaic tile")
    except ValueError:
        logger.warning(f"Series {scene} has {count} tile(s): none of {value!r} is one of them")
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
    out = [(SERIES, scene)]
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
    indices), view outermost. <name> is the file's own channel name (RGB samples of a
    named channel: ``"<name> R"`` / ..., as ``rgb_sample_channels`` names them); an
    unnamed channel takes the name of the same channel in the view's first image, else
    ``Ch{i}``."""
    def own_names(image, count):
        named = [getattr(ch, "name", None) for ch in image.metadata.channels]
        if len(named) == count:
            return named
        rgb = rgb_sample_channels(image)
        if named and rgb is not None and len(rgb) == count and count % len(named) == 0:
            samples = count // len(named)
            return [label if named[k // samples] else None
                    for k, (label, _) in enumerate(rgb)]
        return [None] * count

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


class PlateFieldTilesError(ValueError):
    """A plate file whose fields hold several images (tiles), asked to be written as an
    NGFF plate, which holds one image per field."""


class _MicroFile:
    """A micro-reader File behind the loader's interface: ``indices``, ``image(key)``,
    ``close()``; for a plate file also ``plates`` and ``coordinate_systems``."""

    def __init__(self, f, plate: bool = False):
        self._f, self.path = f, f.path
        #: written as one NGFF plate (_convert_plate)
        self.is_plate = plate
        self.plates = getattr(f, "plates", None) or []
        self.coordinate_systems = getattr(f, "coordinate_systems", None)
        self._images = {tuple((k, v) for k, v in dict(img.index).items() if k != "resolution"):
                        img for img in f.images
                        if int(dict(img.index).get("resolution", 0)) == 0}
        self.indices = [dict(k) for k in self._images]

    def image(self, key):
        return self._images[tuple(key)]

    def close(self):
        self._f.close()


def _open_micro(job) -> _MicroFile:
    """The file opened with micro-reader, or ``NotYet`` when it is not micro-reader's."""
    import micro_reader

    from eubi_bridge.core.micro_source import (RENAME, require_micro_reader,
                                               set_read_concurrency)
    reason = _micro_reason(job)
    if reason:
        raise NotYet(reason)
    require_micro_reader()      # MicroReaderVersionError: a wrong micro-reader is no fallback
    set_read_concurrency(job.cluster.decode_threads)
    path = str(job.input_path)
    tiles = "mosaic" if job.readers.as_mosaic else "separate"
    try:
        # plates=True: a plate file is read as one (File.plates); other files are not
        # changed by it
        f = micro_reader.open(path, tiles=tiles, rename=RENAME, axes="tczyx",
                              samples="channels", plates=True)
    except micro_reader.UnsupportedFormatError as exc:
        raise NotYet(f"micro-reader does not read it ({exc})") from exc
    names = set().union(*(set(k) for k in f.indices)) - {"resolution"}
    if getattr(f, "plates", None):
        plate = _plate_or_not(f, job, names)
        if plate is not None:
            return plate
        f.close()                                  # as_plate=False: read without plates
        f = micro_reader.open(path, tiles=tiles, rename=RENAME, axes="tczyx",
                              samples="channels")
        names = set().union(*(set(k) for k in f.indices)) - {"resolution"}
    problem = ""
    if names - INDICES:
        problem = f"image indices {sorted(names - INDICES)}"
    elif names - {SERIES} and not path.lower().endswith(".czi"):
        # 0.1.5 reads tiles / views / illuminations outside a CZI with the standard reader
        problem = f"{sorted(names - {SERIES})} outside a CZI"
    if problem:
        f.close()
        raise NotYet(problem)
    return _MicroFile(f)


def _plate_or_not(f, job, names: set):
    """For a file micro-reader read as a plate: the plate file to write as one NGFF plate,
    or None to read it without plates (as_plate=False, with a warning). A plate whose
    fields hold several images (tiles, views) raises PlateFieldTilesError unless
    as_mosaic stitched them: an NGFF 0.4 / 0.5 / 0.6 plate holds one image per field (RFC 8
    keeps the tiles of a field as a sub-collection)."""
    path = os.path.basename(str(job.input_path))
    n_wells = sum(len(p.wells) for p in f.plates)
    if not job.readers.as_plate:
        logger.warning(
            f"{path} is a plate ({len(f.plates)} plate(s), {n_wells} wells) and is "
            f"converted with as_plate=False: each image becomes a store of its own, with "
            f"no plate, wells or fields. Leave as_plate at True to write one NGFF plate.")
        return None
    field = sorted(PLATE_INDICES & names)
    per_field = {}
    for index in f.indices:
        if int(index.get("resolution", 0)) != 0:
            continue
        place = tuple(index.get(k) for k in field)
        per_field[place] = per_field.get(place, 0) + 1
    most = max(per_field.values(), default=1)
    # RFC 8 holds a field of several images (a sub-collection per field); NGFF 0.4 / 0.5
    # / 0.6 plates hold one image per field
    if most > 1 and job.conversion.ome_zarr_version != "rfc8":
        f.close()
        extra = sorted(names - PLATE_INDICES - {SERIES})
        raise PlateFieldTilesError(
            f"{path} is a plate whose fields hold up to {most} images each (they differ in "
            f"{extra or ['series']}); an NGFF {job.conversion.ome_zarr_version} plate holds "
            f"one image per field. Convert it with as_mosaic=True to stitch each field's "
            f"tiles into one image, with ome_zarr_version='rfc8' to keep the tiles (RFC 8 is "
            f"a draft), or with as_plate=False to write each image as a store of its own.")
    return _MicroFile(f, plate=True)


def _open_input(job):
    """``(file, reader)``: an OME-Zarr input read by ozp itself; else micro-reader's when
    it takes the file, else the older readers' (``LegacyFile``, roadmap step 2), with the
    reason micro-reader did not."""
    from eubi_bridge.utils.path_utils import is_zarr_group
    if is_zarr_group(str(job.input_path)):
        from eubi_bridge.conversion.legacy_image import NgffFile
        return NgffFile(job), "ome_zarr_pyramid (an OME-Zarr input)"
    try:
        return _open_micro(job), "micro-reader"
    except NotYet as exc:
        from eubi_bridge.conversion.legacy_image import LegacyFile
        return LegacyFile(job), f"older readers ({exc})"


def _load_input_collection(job):
    """``(file, entries, reader)``: the opened file and one entry per output, each
    ``SimpleNamespace(pyramid, image, scene, tile, series_path, basename, name_scene,
    name_tile)`` in eubi's terms (scene = the series; scene / tile / view / illumination
    = positions among the file's own values), named as 0.1.5 names them (``_scene<N>``,
    kept: plan 8.32). The outputs are chosen from the file's index first; only their
    images are read."""
    from ome_zarr_pyramid import PyramidCollection

    from eubi_bridge.core.output_store import _ozp_version

    f, reader = _open_input(job)
    if getattr(f, "is_plate", False):
        # one NGFF plate for the whole file (_convert_plate)
        return f, [SimpleNamespace(plate=True)], reader + ", as a plate"
    try:
        path = str(job.input_path)
        readers = job.readers
        keys = [tuple((k, v) for k, v in index.items()) for index in f.indices]
        stem = job.resolved_basename or os.path.basename(path).split(".")[0]
        file = pathlib.Path(path)

        scene_values = _values(keys, SERIES)
        scenes = _positions(readers.series_index, len(scene_values), "series")
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
            first = [k for k in keys if dict(k)[SERIES] == scene_values[scenes[0]]]
            n_views_file = len([v for v in _values(first, "view") if v is not None]) or 1
            n_illus_file = len([v for v in _values(first, "illumination")
                                if v is not None]) or 1
            if n_views_file <= 1 and n_illus_file <= 1:
                by_view = by_illu = False             # nothing to iterate

        option = readers.mosaic_tile_index
        plans, tiled = [], False                  # (entry, keys, group or None)
        for scene in scenes:
            here = [k for k in keys if dict(k)[SERIES] == scene_values[scene]]
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
                    plans.append((SimpleNamespace(
                        scene=scene, tile=tile,
                        series_path=f"{path}_{scene}" + (f"_tile{tile}" if tile else ""),
                        basename=job.resolved_basename,
                        name_scene=scene if n_scenes > 1 else None, name_tile=tile),
                        [key], None))
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
                    group_keys = [_key(scene_values[scene], tile_value, views[v], illus[i])
                                  for v, i in group]
                    plans.append((SimpleNamespace(
                        scene=scene, tile=tile_named,
                        series_path=str(file.parent / (file.stem + suffix + file.suffix)),
                        basename=stem + suffix,
                        name_scene=scene if n_scenes > 1 else None, name_tile=None),
                        group_keys,
                        group if (concat_views or concat_illus) else None))
        if not plans:
            raise ValueError(f"No valid mosaic tile index in {option!r} for "
                             f"{os.path.basename(path)}: tiles only exist when reading an "
                             f"unstitched mosaic (as_mosaic=False).")

        # only the images the outputs need, read one at a time
        needed = list(dict.fromkeys(k for _, plan_keys, _ in plans for k in plan_keys))
        images = {k: f.image(k) for k in needed}
        version = _ozp_version(job.conversion.zarr_format, job.conversion.ome_zarr_version)
        # the file's coordinate systems (micro-reader; an OME-Zarr input's own), so each
        # output keeps the system its position is in (spatial plan 9.18); readers that
        # name none leave the images unplaced
        coll = PyramidCollection.from_file(
            SimpleNamespace(images=[images[k] for k in needed], path=path,
                            coordinate_systems=getattr(f, "coordinate_systems", None)),
            version=version, name=NAME,
            # keep_existing_resolutions: the file's stored levels too (folded into the
            # image's Pyramid by from_file)
            resolutions="all" if job.downscale.keep_existing_resolutions else "base")
        source = getattr(f, "source_pyramid", None)
        entries = []
        for entry, plan_keys, group in plans:
            if group is None and source is not None and job.downscale.keep_existing_resolutions:
                # an OME-Zarr input keeping its levels: the input Pyramid itself (its
                # own scales per level, as 0.1.x wrote them), named as eubi names
                # every output
                entry.pyramid = source.rename(NAME)
            elif group is None:
                # coll.pyramid(key): the image stamped with its system (3.26)
                entry.pyramid = coll.pyramid(plan_keys[0])
            else:
                # RGB samples named per part first (plain white channels), then the
                # joined labels View{v}_Illu{i}_<name> over them, as 0.1.5 wrote them
                parts = [apply_rgb_channels(coll.pyramid(k), images[k]) for k in plan_keys]
                entry.pyramid = _join_labels(_joined(parts, path), group, parts,
                                             [images[k] for k in plan_keys], readers)
                entry.rgb_named = True
                # 0.1.5 kept the LAST joined image's wavelengths and position
                entry.pyramid.acquisition = getattr(parts[-1], "acquisition", None)
            entry.image = images[plan_keys[-1]]
            if not tiled:
                entry.name_tile = None
            entries.append(entry)
        return f, entries, reader
    except BaseException:
        f.close()
        raise


# -- plates ------------------------------------------------------------------------------

def _plate_outputs(f, job) -> list:
    """``[(plate number, output path)]``: ``<file>.zarr``, or ``<file>_plate<n>.zarr`` for
    each plate of a file holding several."""
    from eubi_bridge.conversion.conversion_worker import _generate_output_path
    path = str(job.input_path)
    stem = job.resolved_basename or os.path.basename(path).split(".")[0]
    if stem.lower() in GENERIC_INDEX_NAMES:
        # an index file with a fixed name (Harmony Index.idx.xml, CellVoyager
        # MeasurementData.mlf): the plate is named after its folder
        stem = os.path.basename(os.path.dirname(os.path.abspath(path))) or stem
    many = len(f.plates) > 1
    return [(n, _generate_output_path(job.output_path, path,
                                      resolved_basename=f"{stem}_plate{n}" if many else stem))
            for n in range(len(f.plates))]


async def _convert_plate(f, job, plate: int, output_path: str) -> None:
    """Plate `plate` of the file as ONE NGFF plate store, in the OME-Zarr version asked
    for (0.4, 0.5, 0.6 or RFC 8): every field gets the job's settings (``_prepared``),
    then ozp writes the plate with eubi's own writer for each field (chunks, shards,
    codecs, memory budget, OME-XML). The fields keep their coordinate systems: the
    systems' axes are filled from the fields as written (spatial plan 3.12)."""
    import dataclasses

    from ome_zarr_pyramid import PyramidCollection
    from ome_zarr_pyramid.core.plates import file_stem, tree_from_file

    from eubi_bridge.conversion.conversion_worker import (_memory_plan_for,
                                                          _warn_unused_writer_settings)
    from eubi_bridge.core.output_store import _ozp_target_args, _ozp_version, store_exists
    conv = job.conversion
    if not conv.overwrite and store_exists(output_path):
        raise FileExistsError(
            f"Output already exists: '{output_path}'. Refusing to convert with "
            f"overwrite=False — the existing dataset was left untouched. Enable overwrite "
            f"to replace it, or choose a different output.")
    keys = [tuple(index.items()) for index in f.indices]
    outside = [k for k in keys if "plate" not in dict(k)]
    if outside and plate == 0:
        # a CZI's slide preview, a label picture: in the file, not in any plate
        logger.warning(
            f"{os.path.basename(str(job.input_path))}: {len(outside)} image(s) of the file "
            f"are not part of a plate and are not written (e.g. {dict(outside[0])}). "
            f"Convert with as_plate=False to write every image as a store of its own.")
    images = {k: f.image(k) for k in keys}
    source = SimpleNamespace(images=[images[k] for k in keys], path=f.path,
                             plates=f.plates, coordinate_systems=f.coordinate_systems)
    version = _ozp_version(conv.zarr_format, conv.ome_zarr_version)
    coll = PyramidCollection.from_file(
        source, version=version, name=NAME,
        resolutions="all" if job.downscale.keep_existing_resolutions else "base")
    # the job's settings on every field, before the plate is built: squeeze or crop can
    # change a field's axes, and the plate's systems and scenes follow the fields as
    # written
    processed = {}
    for key in coll:
        # a field has no series or tile index of eubi's: its place is plate, well, field
        entry = SimpleNamespace(pyramid=coll.pyramid(key), image=images[key],
                                series_path=str(f.path), scene=None, tile=None)
        processed[key], _ = _prepared(entry, job, output_path)
    systems = [dataclasses.replace(cs, axes=None)
               for cs in coll.coordinate_systems.values()]
    written = PyramidCollection(processed, name=f.path,
                                tree=tree_from_file(source, processed,
                                                    name=file_stem(f.path)),
                                coordinate_systems=systems, system_of=coll.system_of)
    _warn_unused_writer_settings(job.cluster)

    def write_field(pyr, target, **_):
        # one field, written as eubi writes an image (the plate's own version: ozp hands
        # each field over in it; RFC 8: the collection document is ozp's)
        budget, region = _memory_plan_for(pyr.base_array, job)
        dtype = np.dtype(conv.dtype or pyr.base_array.dtype)
        xml = ome_xml(pyr, dtype) if conv.save_omexml else None
        _write(pyr, str(target), job, scene_budget_mb=budget, region_size_mb=region,
               xml=xml, image_only=True)

    target, storage_options = _ozp_target_args(output_path)
    await asyncio.to_thread(written.write_plate, target, plate=plate,
                            version=conv.ome_zarr_version, overwrite=conv.overwrite,
                            storage_options=storage_options, writer=write_field)


# -- the job's settings, one step each (Pyramid in, Pyramid out) -------------------------

def _shim(pyr) -> SimpleNamespace:
    """What conversion_worker's parse_* helpers read (``axes``, ``scaledict``,
    ``unitdict``), from a Pyramid: the job's settings are parsed as in 0.1.x."""
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


def rgb_sample_channels(image) -> Optional[list]:
    """``[(label, colour), ...]`` for the channels the RGB samples of *image* become
    (sample_channels, the same rule as the older readers' ``expand_channel_metadata``),
    or None when its pixels have no samples: each channel of S samples -> ``"<name> R"``
    / ``G`` / ``B`` (``A``), tinted red / green / blue; one unnamed channel -> no label
    (``Channel 0/1/2`` where it is written), tinted too. Colours as six hex digits;
    white where none applies."""
    from eubi_bridge.core.sample_channels import sample_channel_labels
    source = getattr(image, "source", image)
    axes = str(getattr(source, "axes", ""))
    if "s" not in axes:
        return None
    samples = int(source.shape[axes.index("s")])
    channels = int(source.shape[axes.index("c")]) if "c" in axes else 1
    described = list(image.metadata.channels)
    names = ([getattr(ch, "name", None) for ch in described] if len(described) == channels
             else [None] * channels)
    return [(label, color.lstrip("#").upper() if color else "FFFFFF")
            for label, color in sample_channel_labels(names, samples)]


def apply_rgb_channels(pyr, image):
    """eubi-bridge's names and colours for RGB samples read as channels
    (``rgb_sample_channels``). The job's channel labels and colours are applied after
    (``apply_channels``) and win."""
    channels = rgb_sample_channels(image)
    if channels is None:
        return pyr
    return pyr.set_channels({i: {"label": label or f"Channel {i}", "color": color}
                             for i, (label, color) in enumerate(channels)})


def apply_axis_metadata(pyr, job, scales_too: bool = True):
    """Scale and unit overrides: the config's values, a batch table row's (job.extra)
    winning. Unset ones keep what the file says. Kept resolution levels take the unit
    overrides only, as in 0.1.x (its store_existing_pyramid had no scales)."""
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
    if scales and scales_too:
        pyr.update_scales(**{ax: float(v) for ax, v in scales.items()})
    if units:
        pyr.update_units(**units)
    return pyr


def channel_overrides(labels, colors, count: int, where) -> dict:
    """``set_channels`` overrides from eubi-bridge's channel options (``"0,DAPI;1,GFP"``;
    colour names or hex): a channel out of range is skipped with a warning, '' and
    'auto' leave it. Conversions and metadata updates parse them alike."""
    from eubi_bridge.utils.metadata_utils import ChannelMap, ChannelParser
    parse = ChannelParser(None)._parse_indexed_string
    labels, colors = parse(labels or ""), parse(colors or "")
    names = ChannelMap()
    overrides = {}
    for idx in sorted(set(labels) | set(colors)):
        if not 0 <= idx < count:
            logger.warning(f"Channel {idx} out of range [0:{count - 1}] for {where}, skipping")
            continue
        fields = {}
        if labels.get(idx, "") not in ("", "auto"):
            fields["label"] = labels[idx]
        if colors.get(idx, "") not in ("", "auto"):
            fields["color"] = names[colors[idx]] or colors[idx]
        if fields:
            overrides[idx] = fields
    return overrides


def apply_channels(pyr, job):
    """Channel labels and colours from the job (``channel_overrides``)."""
    meta = job.metadata
    axes = pyr.meta.axis_order
    count = pyr.base_array.shape[axes.index("c")] if "c" in axes else 1
    overrides = channel_overrides(job.extra.get("channel_labels", meta.channel_labels),
                                  job.extra.get("channel_colors", meta.channel_colors),
                                  count, job.input_path)
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
    """0.1.x's build_acquisition_metadata (conversion_worker): what NGFF has no
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
    # both names, for several releases (plan 8.32): series_index is the name from 0.3.0
    meta["series_index"] = entry.scene
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
    """The small OME-XML eubi-bridge saves beside an output (0.1.x's ArrayManager.save_omexml):
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
    # one image per document: a fixed id (ome_types counts ids up per process, so the
    # n-th output of a run said "Pixels:n")
    pixels.id = "Pixels:0"
    pixels.model_fields_set.update(PFFImageMeta.essential_omexml_fields - pixels.model_fields_set)
    ome.images[0].pixels = pixels
    return ome.to_xml()


# -- one output -----------------------------------------------------------------------

def _write(pyr, output_path: str, job, *, scene_budget_mb: float, region_size_mb: float,
           xml: Optional[str], image_only: bool = False) -> None:
    """ONE write_pyramid call with the job's storage settings (eubi-bridge decides the
    chunk shape and shards; ozp writes)."""
    from eubi_bridge.conversion.conversion_worker import parse_chunks, parse_shard_coefs
    from eubi_bridge.core.output_store import _chunks, _memory_budget, _ozp_target_args
    from eubi_bridge.utils.optional_deps import dask_config
    conv, clus = job.conversion, job.cluster
    arr = pyr.base_array
    dtype = np.dtype(conv.dtype or arr.dtype)
    shim = _shim(pyr)
    axes = pyr.meta.axis_order
    if job.downscale.keep_existing_resolutions and len(pyr.meta.resolution_paths) > 1:
        # kept levels: each level's own chunk shape (store_existing_pyramid_ozp)
        layers = pyr.layers
        chunks = {i: _chunks(layers[path], axes, conv.auto_chunk, parse_chunks(shim, job),
                             dtype, conv.target_chunk_mb)
                  for i, path in enumerate(pyr.meta.resolution_paths)}
    else:
        chunks = _chunks(arr, axes, conv.auto_chunk, parse_chunks(shim, job),
                         dtype, conv.target_chunk_mb)
    shards = parse_shard_coefs(shim, job)
    target, storage_options = _ozp_target_args(output_path)
    logger.info(f"{output_path}: writing, chunks {chunks}")
    # a dask level (the older readers' bioio / IMS / H5 arrays) is written by dask's own
    # scheduler: no memory budget (dyna refuses one for it), and bioformats_threads caps
    # its threads, as in 0.1.x
    budget = _memory_budget([arr], {"memory_budget_mb": scene_budget_mb})
    dask_kw = ({"num_workers": clus.bioformats_threads}
               if clus.bioformats_threads is not None else {})
    with dask_config(**dask_kw):
        _write_pyramid(pyr, target, storage_options, conv, clus, chunks, shards, dtype, arr,
                       region_size_mb, budget, job, xml, image_only=image_only)


def _system_for_output(pyr):
    """`pyr` with its coordinate system's axes made the output's own (spatial plan 3.12,
    9.18). The system was filled from the image as the file gave it; squeeze and crop
    can leave the output with fewer axes, and the definition a store of one image
    carries must fit that image. Id, name and estimated stay; an unplaced image is
    returned as it is."""
    import dataclasses

    from ome_zarr_pyramid.core.spatial import (SYSTEMS_KEY, minimal_system, pyramid_system,
                                               system_to_json)
    sid, cs = pyramid_system(pyr)
    if cs is None:
        return pyr
    own = minimal_system(sid, pyr).axes
    if cs.axes is not None and [a.name for a in cs.axes] == [a.name for a in own]:
        return pyr
    return pyr.set_custom_metadata(SYSTEMS_KEY,
                                   [system_to_json(dataclasses.replace(cs, axes=own))])


def _write_pyramid(pyr, target, storage_options, conv, clus, chunks, shards, dtype, arr,
                   region_size_mb, budget, job, xml, image_only=False):
    from ome_zarr_pyramid import IO

    from eubi_bridge.conversion.conversion_worker import _threads_per_scene
    pyr = _system_for_output(pyr)
    write = IO().write_pyramid
    if conv.ome_zarr_version == "rfc8" and not image_only:
        # NGFF RFC 8: a collection holding the image (an NGFF 0.5 multiscale) under the
        # output's root; ozp writes the collection document and the image with the same
        # settings (experimental: ozp warns that RFC 8 is a draft)
        from ome_zarr_pyramid import PyramidCollection

        def write(pyr, target, **kwargs):
            PyramidCollection({(("series", 0),): pyr}, name=pyr.meta.tag).write_collection(
                target, **kwargs)
    write(
        pyr, target, storage_options=storage_options, overwrite=conv.overwrite,
        max_threads=_threads_per_scene(clus), region_size_mb=float(region_size_mb),
        memory_budget_mb=budget, io_backend=conv.io_backend,
        chunk_shape=chunks,
        shard_coefficients=tuple(shards) if int(pyr.meta.zarr_format) == 3 else None,
        compressor=conv.compressor, compressor_params=conv.compressor_params or {},
        dtype=dtype if dtype != np.dtype(arr.dtype) else None,
        windows=WINDOWS[job.metadata.channel_intensity_limits],
        ome_xml=xml)


def _prepared(entry, job, output_path: str):
    """``(Pyramid, OME-XML or None)``: the job's settings applied to the entry's Pyramid
    (channels, scales and units, squeeze and crop, downscaling, eubi's own block), ready
    to be written. One image; a plate applies it to every field."""
    from eubi_bridge.core.output_store import _eubi_attrs
    conv = job.conversion
    pyr = entry.pyramid
    meta = pyr.meta
    entry.units = {ax: u for ax, u in zip(meta.axis_order, meta.unit_list) if u}
    origin = _origin(entry.image, entry.units)
    # keep_existing_resolutions with stored levels: written as they are, as the old
    # path's store_existing_pyramid did: no squeeze / crop, no scale overrides, no
    # downscaling, no acquisition split info, chunks computed per level
    keep = job.downscale.keep_existing_resolutions and len(meta.resolution_paths) > 1
    if job.downscale.keep_existing_resolutions and not keep:
        logger.info(f"keep_existing_resolutions is set but no existing multiscale "
                    f"pyramid was found for {output_path}; rebuilding pyramid normally")
    acquisition = None if keep else acquisition_metadata(entry, pyr, job, origin)

    if not getattr(entry, "rgb_named", False):     # joined parts are named already
        pyr = apply_rgb_channels(pyr, entry.image)
    pyr = apply_axis_metadata(pyr, job, scales_too=not keep)
    pyr = apply_channels(pyr, job)
    if not keep:
        pyr = apply_squeeze_and_crop(pyr, job)
    dtype = np.dtype(conv.dtype or pyr.base_array.dtype)
    xml = ome_xml(pyr, dtype) if conv.save_omexml else None
    if not keep:
        pyr = apply_downscale(pyr, job)
    pyr.meta.set_custom("eubi_bridge", _eubi_attrs(acquisition))
    return pyr, xml


async def _convert_entry(entry, output_path: str, job, sem: asyncio.Semaphore) -> None:
    """One output: the job's settings applied to the entry's Pyramid, then written."""
    from eubi_bridge.conversion.conversion_worker import (_memory_plan_for,
                                                          _warn_unused_writer_settings)
    from eubi_bridge.core.output_store import store_exists
    async with sem:
        conv = job.conversion
        if not conv.overwrite and store_exists(output_path):
            raise FileExistsError(
                f"Output already exists: '{output_path}'. Refusing to convert "
                f"with overwrite=False — the existing dataset was left untouched. "
                f"Enable overwrite to replace it, or choose a different output.")
        pyr, xml = _prepared(entry, job, output_path)
        scene_budget_mb, region_size_mb = _memory_plan_for(pyr.base_array, job)
        _warn_unused_writer_settings(job.cluster)
        await asyncio.to_thread(_write, pyr, output_path, job,
                                scene_budget_mb=scene_budget_mb,
                                region_size_mb=region_size_mb, xml=xml)


async def convert(job) -> None:
    """Convert the job: every output it asks for, one ``write_pyramid`` call each."""
    from eubi_bridge.conversion.conversion_worker import (_convert_safely,
                                                          _generate_output_path,
                                                          _report_failures)
    f, entries, reader = await asyncio.to_thread(_load_input_collection, job)
    logger.info(f"{job.input_path}: {len(entries)} output(s), read by {reader}")
    _retired_settings(job)
    if entries and getattr(entries[0], "plate", False):
        try:
            outputs = _plate_outputs(f, job)
            results = await asyncio.gather(
                *[_convert_safely(_convert_plate(f, job, plate, out), out)
                  for plate, out in outputs], return_exceptions=True)
        finally:
            f.close()
        _report_failures(results, job)
        return
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
