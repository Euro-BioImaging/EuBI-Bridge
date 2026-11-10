"""Aggregative conversion on the conversion path of eubi-bridge 0.2.0 (roadmap step 3).

Many files become one OME-Zarr image, joined along t / c / z by the tags in their names.
The parent still plans the outputs (``dispatcher._build_aggregative_plan``); each worker
gets one output's files and converts them here (0.1.x's ``AggregativeConverter`` /
``BatchFile`` / ``ArrayManager`` route is retired, ``legacy_0_1/``):

1. Every file's pixels, lazily: micro-reader files through ``micro_array`` (the files are
   opened through micro_source's bounded cache, so thousands of them never stay open at
   once), other files through the older readers (``LegacyFile``). Files are opened on a
   thread pool.
2. Each file's pixel type and shape are compared with the first file's (the shape on
   the axes that are not joined); a difference is a clear error naming the file.
3. The files are joined exactly as 0.1.5 joins them (``FileSet``: the same grouping and
   order along each axis).
4. The metadata comes from the first file; the channels from the first file of each
   channel group, in tag order. Unnamed channels are numbered across the output from 0,
   as on unary conversions; repeated colours are replaced by eubi-bridge's defaults.
5. The result is one image for ozp, converted by the unary path's steps
   (``pyramid_path._convert_entry``): scales, units, channel labels and colours, squeeze,
   crop, dtype, downscale, windows, chunks, one ``write_pyramid``.

``screen_group`` runs in the parent, on file names only: a ``.companion.ome`` is left
out, and files of mixed formats are refused.
"""
from __future__ import annotations

import asyncio
import os
import re
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from eubi_bridge.conversion.pyramid_path import SERIES
from eubi_bridge.utils.errors import InputError
from eubi_bridge.utils.logging_config import get_logger

logger = get_logger(__name__)

AXES = "tczyx"
#: OME companion files: metadata only; their planes are the other files of the set
COMPANION_SUFFIXES = (".companion.ome",)
#: suffixes that name one format family (an OME-TIFF is a TIFF)
_FAMILIES = {".ome.tif": "TIFF", ".ome.tiff": "TIFF", ".ome.btf": "TIFF", ".tif": "TIFF",
             ".tiff": "TIFF", ".btf": "TIFF", ".ome.zarr": "OME-Zarr", ".zarr": "OME-Zarr"}
#: files opened at once while the inputs are read
OPEN_THREADS = 16
#: image indices a joined output takes one value of, by --<index>_index
PICKED_INDICES = ("view", "illumination", "rotation", "phase")


# -- the parent: file names only ---------------------------------------------------------

def _family(path: str) -> str:
    name = os.path.basename(str(path).rstrip("/\\")).lower()
    for suffix, family in _FAMILIES.items():
        if name.endswith(suffix):
            return family
    return os.path.splitext(name)[1] or name


def screen_group(paths: list, output: str = "") -> list:
    """One output's files, as the conversion will join them: a ``.companion.ome`` among
    them is left out (its metadata still reaches the files that point to it), and files
    of more than one format are refused with the odd ones named."""
    paths = [str(p) for p in paths]
    companions = [p for p in paths if p.lower().endswith(COMPANION_SUFFIXES)]
    if companions:
        logger.info(f"Aggregative conversion{f' of {output}' if output else ''}: left out "
                    f"{len(companions)} OME companion file(s) "
                    f"({', '.join(os.path.basename(p) for p in companions[:3])}"
                    f"{', ...' if len(companions) > 3 else ''}): a companion holds metadata "
                    f"only; the files that point to it are read with it.")
        paths = [p for p in paths if p not in companions]
    families: dict = {}
    for p in paths:
        families.setdefault(_family(p), []).append(p)
    if len(families) > 1:
        ranked = sorted(families.items(), key=lambda kv: -len(kv[1]))
        main = ranked[0][0]
        odd = [p for _, group in ranked[1:] for p in group]
        listed = ", ".join(os.path.basename(p) for p in odd[:10])
        raise InputError(
            f"Aggregative conversion{f' of {output}' if output else ''}: the files to join "
            f"are of {len(families)} formats ({', '.join(f'{len(g)} {f}' for f, g in ranked)}). "
            f"Files joined into one image must be of one format. Not {main}: {listed}"
            f"{', ...' if len(odd) > 10 else ''}. Leave them out with --excludes (or choose "
            f"the files with --includes); in the GUI, the Exclude / Include fields.")
    return paths


# -- the worker ------------------------------------------------------------------------

def _axis_list(concatenation_axes) -> list:
    """The joined axes as positions in t c z y x, in the order given (0.1.5's digest)."""
    if concatenation_axes is None:
        return []
    return [AXES.index(x) for x in str(concatenation_axes).lower() if x in AXES]


def _tags(job_kwargs: dict) -> dict:
    return {**{f"axis_tag{i}": job_kwargs.get(name) for i, name in
               enumerate(("time_tag", "channel_tag", "z_tag", "y_tag", "x_tag"))},
            "tag_folders": job_kwargs.get("tag_folders")}


def _is_dask(array) -> bool:
    from eubi_bridge.utils.optional_deps import is_dask_array
    return is_dask_array(array)


def _as_tczyx(array, axes: str):
    """`array` with axes `axes` as t c z y x (missing axes of size 1 added)."""
    from eubi_bridge.utils.optional_deps import is_dask_array
    axes = str(axes).lower()
    if axes == AXES:
        return array
    dask = is_dask_array(array)
    if dask:
        import dask.array as da
    else:
        from dyna_zarr import operations as ops
    for ax in AXES:
        if ax not in axes:
            array = da.expand_dims(array, 0) if dask else ops.expand_dims(array, 0)
            axes = ax + axes
    order = [axes.index(ax) for ax in AXES]
    if order != list(range(len(AXES))):
        array = array.transpose(order) if dask else ops.transpose(array, order)
    return array


def _file_job(job, path: str):
    return job.model_copy(update={"input_path": path})


class _Inputs:
    """The files of one output: each one's lazy t c z y x array, and the metadata of
    the files it is taken from. Files the older readers opened stay open until
    ``close``."""

    def __init__(self, paths: list, job, scene):
        self.paths, self.job = paths, job
        #: series_index='all' joins each file's only series (one per file, checked)
        self.all_scenes = scene == ALL_SCENES
        self.scene = 0 if self.all_scenes else scene
        self._legacy: dict = {}
        #: {file: its _picks}: the image the metadata comes from is the image joined
        self._picked: dict = {}

    def _one_scene(self, path: str, count: int) -> None:
        if self.all_scenes and count != 1:
            # 0.1.5 folded the series into the joined axis (positions became time
            # points); an aggregative output joins ONE series of each file
            raise InputError(
                f"Aggregative conversion joins one series of each file, and "
                f"{os.path.basename(path)} has {count} series. Choose the series with "
                f"--series_index (0 to {count - 1}), with --as_plate False if it is a plate; "
                f"for every series, run once per series.")

    # -- pixels --------------------------------------------------------------
    def array(self, path: str):
        from eubi_bridge.conversion.pyramid_path import _micro_reason
        from eubi_bridge.core.micro_source import _using, micro_array
        job = _file_job(self.job, path)
        if not _micro_reason(job):
            # each file alone, also a member of a multi-file OME set: this path joins
            # the files itself (micro-reader's multifile=True would open the set)
            # tiles stitched by the job's rule, as the metadata is read (image): tiles that
            # cannot be placed stay separate there and here
            tile = self._chosen_tile()
            why = ""
            for tiles in ("mosaic", "separate") if job.readers.as_mosaic else ("separate",):
                self._picked[path] = picks = self._picks(path, tiles)
                arr, why = micro_array(path, self.scene, tile, as_mosaic=tiles == "mosaic",
                                       multifile=False, **picks)
                if arr is not None:
                    if self.all_scenes:
                        with _using(path, tiles, multifile=False) as entry:
                            self._one_scene(path, len({k[SERIES] for k in entry.file.indices}))
                    return arr
            logger.info(f"{os.path.basename(path)}: older readers ({why})")
        image = self._legacy_image(path)
        array = image.array
        # a zarr or numpy array of the older readers: the join takes lazy arrays
        if not hasattr(array, "_with_transform") and not _is_dask(array):
            from dyna_zarr import from_array
            array = from_array(array)
        return _as_tczyx(array, image.axes)

    def _chosen_tile(self):
        """The tile joined from each file, None without a choice (a joined output holds one
        image of each file)."""
        value = self.job.readers.mosaic_tile_index
        if value in (None, "all"):
            return None
        if isinstance(value, (list, tuple)) or "," in str(value):
            raise InputError(f"A joined output holds one tile of each file: --mosaic_tile_index "
                             f"{value} names several. Run once per tile.")
        return int(value)

    def _picks(self, path: str, tiles: str) -> dict:
        """{view, illumination, rotation, phase: position among the file's}. A joined output
        holds one of each, as it holds one series: a file with several needs a choice, and
        the others would otherwise be left out in silence. {} for a file not read here."""
        from eubi_bridge.core.micro_source import _using
        readers, name = self.job.readers, os.path.basename(path)
        try:
            with _using(path, tiles, multifile=False) as entry:
                indices = [dict(k) for k in entry.file.indices]
        except Exception as exc:                        # noqa: BLE001 - micro_array says why
            logger.debug(f"{name}: {exc}")
            return {}
        picks = {}
        for index in PICKED_INDICES:
            count = len({k[index] for k in indices if index in k})
            value = getattr(readers, f"{index}_index", 0)
            if value in (None, "all"):
                if count > 1:
                    raise InputError(f"{name} has {count} {index}s, and a joined output holds "
                                     f"one. Choose it with --{index}_index (0 to {count - 1}); "
                                     f"for every {index}, run once each.")
                value = 0
            if isinstance(value, (list, tuple)) or "," in str(value):
                raise InputError(f"A joined output holds one {index} of each file: "
                                 f"--{index}_index {value} names several. Run once each.")
            if not 0 <= int(value) < max(count, 1):
                raise InputError(f"{name}: no {index} {value} (it has {max(count, 1)}).")
            picks[index] = int(value)
        tile, count = self._chosen_tile(), len({k["tile"] for k in indices if "tile" in k})
        if tile is not None and tiles == "separate" and not 0 <= tile < max(count, 1):
            raise InputError(f"{name}: no tile {tile} (it has {max(count, 1)}).")
        return picks

    def _legacy_file(self, path: str):
        if path not in self._legacy:
            from eubi_bridge.conversion.legacy_image import LegacyFile, NgffFile
            from eubi_bridge.utils.path_utils import is_zarr_group
            job = _file_job(self.job, path)
            self._legacy[path] = NgffFile(job) if is_zarr_group(path) else LegacyFile(job)
        return self._legacy[path]

    def _refuse_picks(self, path: str) -> None:
        """The older readers join each file's first view, illumination and tile: one asked
        for is refused, not swapped for the first."""
        readers = self.job.readers
        for option, value in [(f"{index}_index", getattr(readers, f"{index}_index", 0))
                              for index in PICKED_INDICES] + [
                ("mosaic_tile_index", self._chosen_tile())]:
            if value not in (None, "all", 0, "0"):
                raise InputError(f"{os.path.basename(path)}: --{option} {value} needs "
                                 f"micro-reader: the older readers join each file's first.")

    def _legacy_image(self, path: str):
        self._refuse_picks(path)
        f = self._legacy_file(path)
        keys = sorted({int(k[SERIES]) for k in f.indices})
        self._one_scene(path, len(keys))
        if not 0 <= self.scene < len(keys):
            raise InputError(f"{os.path.basename(path)}: no series {self.scene} (it has "
                             f"{len(keys)})")
        first = next(k for k in f.indices if int(k[SERIES]) == keys[self.scene])
        return f.image(first)

    # -- metadata ------------------------------------------------------------
    def image(self, path: str):
        """The image `path` gives the metadata of (micro-reader's when it reads the
        file, read and closed again; else the older readers')."""
        from eubi_bridge.conversion.pyramid_path import NotYet, _open_micro
        job = _file_job(self.job, path)
        try:
            f = _open_micro(job)
        except NotYet:
            return self._legacy_image(path)
        try:
            if any(SERIES not in k for k in f.indices):
                raise InputError(f"{os.path.basename(path)} is a plate: its images have no "
                                 f"series to join. Convert the plates one by one, or add "
                                 f"--as_plate False to join them by series.")
            scenes = sorted({k[SERIES] for k in f.indices})
            if not 0 <= self.scene < len(scenes):
                raise InputError(f"{os.path.basename(path)}: no series {self.scene} (it "
                                 f"has {len(scenes)})")
            # the series' first image of those chosen: with its tiles separate, its tile 0
            here = [k for k in f.indices if k[SERIES] == scenes[self.scene]]
            for index, at in self._picked.get(path, {}).items():
                values = sorted({k[index] for k in here if index in k})
                if values:
                    here = [k for k in here if k[index] == values[at]]
            tile = self._chosen_tile()
            here = [k for k in here if k.get("tile") == tile] or here if tile is not None else here
            if tile is None and len({k.get("tile") for k in here}) > 1:
                logger.warning(f"{os.path.basename(path)}: its tiles are not stitched: only "
                               f"its first tile is joined.")
            index = min(here, key=lambda k: tuple(k.values()))
            image = f.image(tuple(index.items()))
            meta = image.metadata
            source = getattr(image, "source", image)
            # the system the image's position is in, with its definition (9.18)
            sid = getattr(meta, "coordinate_system", None)
            systems = [cs for cs in getattr(f, "coordinate_systems", None) or ()
                       if sid is not None and str(getattr(cs, "id", "")) == str(sid)]
            return SimpleNamespace(
                name=getattr(image, "name", None),
                coordinate_systems=systems or None,
                axes=str(getattr(source, "axes", "")),
                # the unfolded axes and shape: its RGB samples (rgb_sample_channels)
                source=SimpleNamespace(axes=str(getattr(source, "axes", "")),
                                       shape=tuple(getattr(source, "shape", ()))),
                metadata=SimpleNamespace(
                    pixel_sizes=dict(meta.pixel_sizes or {}),
                    channels=list(meta.channels or []),
                    position=dict(meta.position or {}),
                    index_values=dict(getattr(meta, "index_values", None) or {}),
                    coordinate_system=sid))
        finally:
            f.close()

    def close(self) -> None:
        for f in self._legacy.values():
            f.close()
        self._legacy.clear()


def _check_agreement(paths: list, arrays: list, joined: list) -> None:
    """Every file's pixel type, and its size on the axes that are not joined, must be
    the first file's: a clear error otherwise (a different type was widened silently or
    cut, a different size failed inside the join without naming the file)."""
    import numpy as np
    first, ref = os.path.basename(paths[0]), arrays[0]
    along = "/".join(AXES[a] for a in joined) or "no axis"
    for path, arr in zip(paths[1:], arrays[1:]):
        name = os.path.basename(path)
        if np.dtype(arr.dtype) != np.dtype(ref.dtype):
            raise InputError(
                f"Aggregative conversion: {name} has pixel type {np.dtype(arr.dtype)}, "
                f"{first} has {np.dtype(ref.dtype)}. Files joined into one image must have "
                f"one pixel type.")
        for axis, letter in enumerate(AXES):
            if axis not in joined and arr.shape[axis] != ref.shape[axis]:
                raise InputError(
                    f"Aggregative conversion: {name} has {arr.shape[axis]} along {letter}, "
                    f"{first} has {ref.shape[axis]}. Files joined along {along} must have "
                    f"the same size on the other axes (shapes {tuple(arr.shape)} and "
                    f"{tuple(ref.shape)}, t c z y x).")


_OME_TIFF = (".ome.tif", ".ome.tiff", ".ome.btf")


def _refuse_multifile_ome(paths: list) -> None:
    """A multi-file OME-TIFF among the inputs: the first file's OME-XML lists planes in
    other input files, or points to one of them for its metadata. Such a file opens as
    the whole set, so joining the files would repeat the set's planes; refused until
    micro-reader can open each file with its own planes only (micro-reader tracker,
    'Open an OME-TIFF with only its own planes')."""
    first = paths[0]
    if not first.lower().endswith(_OME_TIFF):
        return
    try:
        import tifffile
        with tifffile.TiffFile(first) as tif:
            xml = tif.pages[0].description or ""
    except Exception:                                    # noqa: BLE001 - not readable here
        return
    if "<OME" not in xml:
        return
    inputs = {os.path.basename(p) for p in paths}
    own = os.path.basename(first)
    named = set(re.findall(r'FileName="([^"]+)"', xml))
    named |= set(re.findall(r'MetadataFile="([^"]+)"', xml))
    others = sorted({os.path.basename(n.replace("\\", "/")) for n in named} & inputs - {own})
    if others:
        raise InputError(
            f"Aggregative conversion: {own} is part of a multi-file OME-TIFF: its OME-XML "
            f"refers to {len(others)} of the other input files ({', '.join(others[:3])}"
            f"{', ...' if len(others) > 3 else ''}). Such a file opens as the whole set, so "
            f"joining the files would repeat its planes. Convert the set as one image "
            f"instead: convert {own} on its own (no --concatenation_axes).")


def _channel_groups(paths: list, job_kwargs: dict) -> list:
    """The first file of each channel group, in tag order (0.1.5's channel samples)."""
    from eubi_bridge.conversion.fileset_io import FileSet, channel_tag_format_error
    fs = FileSet(paths, shapes=[(1,) * 5] * len(paths), **_tags(job_kwargs))
    tag = fs.axis_tags[1]
    if tag is None:
        return [paths[0]]
    groups = fs._split_by(tag)
    if not groups:
        raise InputError(channel_tag_format_error(tag))
    return [group[0] for group in groups.values()]


def _tag_names(samples: list, paths: list, job_kwargs: dict) -> dict:
    """{sample file: the name its channel tag gives it}, for a list of channel tags (named
    channels); empty for a numbered channel tag or none. Read as the run read them."""
    from eubi_bridge.conversion.fileset_io import FileSet
    fs = FileSet(paths, shapes=[(1,) * 5] * len(paths), **_tags(job_kwargs))
    tag = fs.axis_tags[1]
    if not isinstance(tag, tuple):
        return {}
    return {path: fs._places[tag][path] for path in samples if path in fs._places[tag]}


def _channels(samples: list, images: dict, sizes: dict, total: int, job_kwargs: dict,
              names: dict = None) -> list:
    """The output's channels: each sample file's own (RGB samples named and tinted as a
    single file's: ``rgb_sample_channels``; a file listing another count than it has:
    that many unnamed ones), joined; with override_channel_names and a list of channel
    tags, each file's channels take its own tag's name (*names*, ``_tag_names``): the
    name, or name_0, name_1, ... for a file of several channels. That holds whether the
    channels are joined into one output or each is an output of its own."""
    from eubi_bridge.conversion.pyramid_path import rgb_sample_channels
    blank = SimpleNamespace(name=None, color=None, emission_wavelength=None,
                            excitation_wavelength=None)
    override = bool(job_kwargs.get("override_channel_names")) and bool(names)
    out = []
    for path in samples:
        own = list(images[path].metadata.channels)
        rgb = rgb_sample_channels(images[path]) if len(own) != sizes[path] else None
        if rgb is not None and len(rgb) == sizes[path]:
            # an unnamed single RGB channel has no label: numbered in the output
            own = [SimpleNamespace(name=label, color=color, emission_wavelength=None,
                                   excitation_wavelength=None) for label, color in rgb]
        own = own if len(own) == sizes[path] else [blank] * sizes[path]
        if override and names.get(path):
            name = str(names[path])
            labels = [name] if len(own) == 1 else [f"{name}_{k}" for k in range(len(own))]
            own = [SimpleNamespace(name=label, color=getattr(ch, "color", None),
                                   emission_wavelength=getattr(ch, "emission_wavelength", None),
                                   excitation_wavelength=getattr(ch, "excitation_wavelength", None))
                   for label, ch in zip(labels, own)]
        out.extend(own)
    if len(out) != total:
        return []                                  # say nothing: ozp generates them
    # an unnamed channel without a colour takes eubi-bridge's default colour for its
    # place in the output (red, green, blue, ...), as 0.1.5's aggregative conversion did
    from eubi_bridge.core.data_manager import ChannelIterator
    defaults = ChannelIterator(num_channels=len(out))._channels
    out = [SimpleNamespace(name=None, color=defaults[idx]["color"],
                           emission_wavelength=getattr(ch, "emission_wavelength", None),
                           excitation_wavelength=getattr(ch, "excitation_wavelength", None))
           if not getattr(ch, "name", None) and not getattr(ch, "color", None) else ch
           for idx, ch in enumerate(out)]
    return out


def _distinct_colors(pyr):
    """eubi-bridge's default colours for every channel when two share a colour (0.1.5's
    _update_nonunique_channel_colors), so joined channels can be told apart."""
    from eubi_bridge.core.data_manager import ChannelIterator
    channels = (pyr.meta.metadata.get("omero") or {}).get("channels") or []
    colors = [str(ch.get("color", "")).upper() for ch in channels]
    if len(set(colors)) == len(colors):
        return pyr
    defaults = ChannelIterator(num_channels=len(colors))._channels
    return pyr.set_channels({i: {"color": d["color"]} for i, d in enumerate(defaults)})


#: series_index='all' (the GUI's default): files of one series each are joined; files
#: of several are refused (``_Inputs``: 0.1.5 folded their series into the joined axis)
ALL_SCENES = "all"


def _scene(job_kwargs: dict):
    """The series joined from each file (an int), or ALL_SCENES (series_index; the old
    name still works: config_models.rename_settings)."""
    from eubi_bridge.core.config_models import rename_settings
    value = rename_settings(job_kwargs).get("series_index", 0)
    value = 0 if value is None else value
    if value == ALL_SCENES:
        return ALL_SCENES
    try:
        return int(value)
    except (TypeError, ValueError):
        # 0.1.5 crashed on a list (TypeError in set_scene)
        raise InputError(
            f"Aggregative conversion joins one series of each file: --series_index "
            f"{value!r} names several. Give one series index; for several series, run "
            f"once per series.") from None


def _build_entry(paths: list, job, job_kwargs: dict):
    """``(entry, inputs)``: the joined image as one entry for ``_convert_entry``."""
    from natsort import natsorted
    from ome_zarr_pyramid import PyramidCollection

    from eubi_bridge.conversion.fileset_io import FileSet
    from eubi_bridge.conversion.legacy_image import LegacyImage
    from eubi_bridge.conversion.pyramid_path import NAME
    from eubi_bridge.core.output_store import _ozp_version

    paths = natsorted(str(p) for p in paths)
    # the output's first file, whose metadata it takes, is the first by its tags
    paths = FileSet(paths, shapes=[(1,) * 5] * len(paths), **_tags(job_kwargs)).in_order()
    scene = _scene(job_kwargs)
    joined_axes = _axis_list(job_kwargs.get("concatenation_axes"))
    _refuse_multifile_ome(paths)
    inputs = _Inputs(paths, job, scene)
    scene = inputs.scene                                   # 'all': each file's only series
    try:
        with ThreadPoolExecutor(max_workers=min(OPEN_THREADS, len(paths))) as pool:
            arrays = list(pool.map(inputs.array, paths))
        _check_agreement(paths, arrays, joined_axes)
        sizes = {p: int(a.shape[1]) for p, a in zip(paths, arrays)}

        fs = FileSet(paths, arrays=arrays, **_tags(job_kwargs))
        for axis in joined_axes:
            fs.concatenate_along(axis)
        outputs = fs.get_concatenated_arrays()
        if len(outputs) != 1:
            raise RuntimeError(f"the files of one output form {len(outputs)} images: "
                               f"{[os.path.basename(p) for p in outputs]}")
        (_, array), = outputs.values()

        samples = _channel_groups(paths, job_kwargs) if 1 in joined_axes else [paths[0]]
        # RGB files: their samples are channels already (micro-reader / the older readers);
        # a file describing fewer channels than it has gets unnamed ones, numbered and
        # coloured red / green / blue as 0.1.5 did (_channels)
        images = {p: inputs.image(p) for p in dict.fromkeys([paths[0]] + samples)}
        first = images[paths[0]].metadata
        # the output takes the first file's position and system (spatial plan 3.26, 9.18)
        image = LegacyImage(
            array=array, axes=AXES, index={SERIES: scene}, name=images[paths[0]].name,
            pixel_sizes=first.pixel_sizes,
            channels=_channels(samples, images, sizes, int(array.shape[1]), job_kwargs,
                               _tag_names(samples, paths, job_kwargs)),
            position=first.position, index_values=first.index_values,
            coordinate_system=getattr(first, "coordinate_system", None))
        version = _ozp_version(job.conversion.zarr_format, job.conversion.ome_zarr_version)
        coll = PyramidCollection.from_file(
            SimpleNamespace(images=[image], path=paths[0],
                            coordinate_systems=getattr(images[paths[0]], "coordinate_systems",
                                                       None)),
            version=version, name=NAME, resolutions="base")
        pyr = _distinct_colors(coll.pyramid(((SERIES, scene),)))
        entry = SimpleNamespace(pyramid=pyr, image=image, scene=scene, tile=None,
                                series_path=f"{paths[0]}_{scene}", basename=None,
                                name_scene=None, name_tile=None)
        return entry, inputs
    except BaseException:
        inputs.close()
        raise


async def convert_group(file_paths: list, output_path: str, job_kwargs: dict) -> None:
    """Convert one output's files: joined into one Pyramid, one ``write_pyramid`` call."""
    from natsort import natsorted

    from eubi_bridge.conversion.conversion_worker import _convert_safely
    from eubi_bridge.conversion.fileset_io import FileSet
    from eubi_bridge.conversion.pyramid_path import (_convert_entry, _retired_settings,
                                                     _warn_dtype, _warn_uncalibrated)
    from eubi_bridge.core.config_models import ConversionJob

    if job_kwargs.get("metadata_reader", "micro") == "micro":
        # micro-reader's metadata comes with its pixels, as in 0.1.x
        job_kwargs = {**job_kwargs, "pixel_reader": "micro"}
    if job_kwargs.get("dtype", "auto") == "auto":
        job_kwargs = {**job_kwargs, "dtype": None}         # the source's (as 0.1.x's ozp writer)
    from eubi_bridge.core.output_store import with_output_suffix
    out = with_output_suffix(output_path, job_kwargs.get("zip_output") or "off")
    # the first file by its tags, as in _build_entry: the warnings name the file whose
    # metadata the output takes
    paths = FileSet(natsorted(str(p) for p in file_paths), shapes=[(1,) * 5] * len(file_paths),
                    **_tags(job_kwargs)).in_order()
    job = ConversionJob.from_kwargs(input_path=paths[0], output_path=out, kwargs=job_kwargs)
    from eubi_bridge.core import memory
    from eubi_bridge.core.micro_source import (MAX_OPEN_FILES, open_file_cost_mb,
                                               open_files_for_join)
    # a join keeps more files open than one input (JOIN_OPEN_FILES), as many as an eighth of
    # this process's memory plan holds at what one of its files holds: with the 128 of one
    # input every region reopened the files of the last (Workplan E25, BARD report)
    cost = (await asyncio.to_thread(open_file_cost_mb, paths)
            if len(paths) > MAX_OPEN_FILES and job_kwargs.get("pixel_reader") == "micro"
            else None)
    plan = job.cluster.process_memory_mb or memory.process_mb(job.cluster)
    with open_files_for_join(len(paths), cost, plan):
        entry, inputs = await asyncio.to_thread(_build_entry, paths, job, job_kwargs)
        logger.info(f"{out}: {len(paths)} file(s) joined along "
                    f"{job_kwargs.get('concatenation_axes')}")
        _retired_settings(job)
        _warn_uncalibrated(job, [entry])
        _warn_dtype(job, [entry])
        try:
            await _convert_safely(_convert_entry(entry, out, job, asyncio.Semaphore(1)), out)
        finally:
            inputs.close()
