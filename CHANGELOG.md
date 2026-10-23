# Changelog

## 0.3.2 (2026-10-22): plates, coordinate systems, NGFF 0.6 and RFC 8 output

Requires `micro-reader[codecs]>=0.2.1,<0.3` (was 0.1.x; `[codecs]` brings `imagecodecs`, which
micro-reader needs for JPEG XR, the usual compression of Zeiss slide scans: without it every tile
of such a CZI failed in eubi-bridge-lite) and `ome_zarr_pyramid[remote]>=0.4.0,<0.5`
(was 0.3.x), in `pyproject.toml` (full and lite) and `requirements.txt`: the releases that
declare and write coordinate systems, NGFF 0.6 and RFC 8. A micro-reader outside that
range stops a conversion with `MicroReaderVersionError`, naming why (0.1.x declares no
coordinate systems) and the fix.

### New

- **Plates.** A plate file is written as one NGFF plate store: plate, wells and fields,
  in the OME-Zarr version asked for (0.4, 0.5, 0.6 or RFC 8). By default that is every
  file of a plate format (Harmony, CellVoyager, MetaXpress, IN Cell; also a subset of one
  well) and every other file whose images sit in two or more wells (a CZI of wells
  B2 + B3). A file whose images all sit in one well (a CZI acquired on a plate holder,
  even a single image) is converted per image as before, with an info line. Every field gets the
  job's settings (channels, scales, squeeze, crop, dtype, downscaling, chunks and codecs)
  and keeps its coordinate system. The store is named after the file
  (`<file>_plate<n>.zarr` for a file holding several plates); a plate opened from an
  index file with a fixed name (`Index.idx.xml`, `MeasurementData.mlf`) is named after its
  folder. Before, eubi-bridge-lite refused these files and the full eubi-bridge wrote each
  image through Bio-Formats.
  - A Harmony plate (index in `Images/`, or `images/` in a Harmony 5.2 / 6 "hs" archive)
    is named after its measurement folder; two plates of one name in a batch take the
    folder above as a prefix.
  - Any file of a plate may be chosen: the inputs (a file list, a folder, a batch
    table) are grouped with micro-reader's `group_files`, so a plate's index, side files
    (a Harmony measurement's `image.index.txt`, `.kw.txt`, `Assaylayout` files) and
    planes all stand for the plate, converted once (a batch: with the first row's
    settings). A text file that belongs to no image, chosen on its own (a Harmony "hs"
    archive's `_objmeta.txt`), stands for the one plate in its folder; with several
    there, it is skipped with a warning naming them. A text file chosen beside other
    files of its folder (a readme) is skipped.
  - Every file in a plate's own folder belongs to the plate, also those micro-reader
    names no part of it (an Opera Phenix export's `Evaluation1/` analysis results): an
    export converted by its folder is one plate, not one plate and 50 failed inputs.
- Any multi-file image set micro-reader recognises (a Leica XLEF, an Olympus OIF folder,
  a MetaMorph `.nd`, an OME companion) is one input, its main file, also when eubi reads
  that file with the older readers: the full eubi-bridge reads the set once through
  Bio-Formats; eubi-bridge-lite gives one error per set instead of one per part (an OIF
  folder of 1600 planes was 1600 inputs, and ran out of open files).
- Of several inputs (a folder, a file list), files no reader recognises (instrument and
  analysis side files: `.xml`, `.aas`, `.xgapp`, `.env`, ...) are skipped, named in one
  warning, instead of each failing the conversion. A file given on its own still fails
  loudly when it is not an image.
  - A plate whose fields hold several tiles needs `as_mosaic=True` for NGFF 0.4 / 0.5 /
    0.6 (a plate holds one image per field); the conversion stops with an error naming
    the ways out. RFC 8 keeps the tiles of a field as a sub-collection.
  - Images of the file that are not part of a plate (a CZI's slide preview) are not
    written, with a warning.
  - `as_plate` (reader setting): unset is the rule above; `True` writes every file read
    as a plate as one (a single well too); `False` writes each image as a store of its
    own, as before, with a warning that the file is a plate. The GUI shows it as "Write as
    NGFF plate" (auto / on / off) on the Reader tab and as a batch-table column.
- `ome_zarr_version` takes `"0.6"` (NGFF 0.6: the image's coordinate systems in its
  multiscale, one transformation per level) and `"rfc8"` (NGFF RFC 8, a draft: a
  collection holding the image as an NGFF 0.5 multiscale; ozp warns that RFC 8 is
  experimental), beside `"0.4"` and `"0.5"`. Spellings such as `v0.6` or `RFC8` are
  accepted. The GUI offers both in the OME-Zarr version list (Convert page and batch
  table); the shard settings apply to 0.5, 0.6 and RFC 8.
- More formats are read by micro-reader: Leica XLEF sets, Olympus OIB / OIF, MetaMorph
  `.nd` sets, DeltaVision (`.dv`, `.r3d`), Micro-Manager image-per-file acquisitions
  (their `metadata.txt`), and the tiles, views and illuminations of every format
  (until 0.3.1 those outside a CZI went to the older readers). Each was checked
  against the older readers on the samples held: the pixels are identical; micro-reader
  adds what the older readers dropped (time steps, the files' own channel colours and
  names, coordinate systems). An output of a Micro-Manager acquisition is named after
  the acquisition folder, not `metadata`.

- Each output keeps the coordinate system its position is in: the custom metadata
  `eubi:coordinate_system` (the system's id, e.g. `stage`) and `eubi:coordinate_systems`
  (its definition: name, axes, whether it is estimated), beside the image's translation.
  Outputs converted from one file name the same system, so other tools can place them
  together. The system's axes are the output's own, after squeeze and crop. An image the
  reader places in no system (the older readers; a file without positions) is written as
  before, without these keys.
- An aggregated output takes the first file's position and system.
- An OME-Zarr input keeps the system it names. An NGFF 0.6 image placed by a scene is read
  with its position (`apply_scene=True`).
- A group inside an OME-Zarr store that is itself an image (a well's field, one series of
  a layout) can be given as the input; before, only a path ending in `.zarr` was read as
  one OME-Zarr input, and a group inside a store was taken as a folder of files.
- An OME-Zarr 0.6 input's other coordinate systems and the transformations between them (an
  affine to a registered frame, a scale to an array system, ...) are written to the output: they
  were read but dropped. They are carried while the output has the input's axes, pixel size and
  position; after a squeeze, crop or rescale they would point at the wrong pixels, and are left
  out with a warning. An output written as 0.4 or 0.5 keeps them under `eubi:ngff06`.
- eubi-bridge-lite no longer skips a damaged image among several inputs: any file micro-reader did not
  recognise was taken for a side file, a corrupt `.tif` too, so the batch finished without it. A file with an
  image suffix is kept, and the conversion fails on it, as with the full install.
- Half-float (float16) images, such as Leica FLIM results, convert: the OME-XML writer had no
  entry for them and skipped every such scene. The XML says float (OME has no half float); the
  array stays float16.
- The notice for a partly converted file no longer calls it a damaged Zeiss file: it names how
  many scenes were skipped and the reasons they gave.
- On a plate, the "channels have no colour in the file" line is logged once, not once per well.

### Changed

- A CZI whose images sit in two or more wells of a plate is written as one NGFF plate
  store by default (it was one store per scene); `as_plate=False` gives the old outputs.
  A CZI of several wells whose fields are tiled now stops with the error above in 0.4 /
  0.5 / 0.6 unless `as_mosaic=True`, `ome_zarr_version="rfc8"` or `as_plate=False`.
- Channels without a colour of their own get one. Before, they were written white, so a
  multichannel image the file gives no colours for (DeltaVision, read by the older
  readers, which give every channel white) showed all grey. Now: the colour of the
  channel's emission wavelength (the file's; else its dye's, from the channel name:
  DAPI, FITC, Cy3, Texas Red / RD-TR-PE, Cy5, Alexa 488 / 594 / 647, GFP, mCherry, ...),
  else distinct defaults (red, green, blue, magenta, cyan, yellow, then generated),
  skipping colours close to one already in use. A transmitted-light channel (Brightfield,
  DIC, phase, ...) is white. A colour two or more channels share counts as no colour;
  colours the file gives are kept; a single channel stays as it is; `channel_colors`
  still wins. The golden outputs of the tests changed in these colours only.
- Two inputs of one folder that share a name but not a suffix (an Airyscan `x.czi` and
  its `x.ims`) are written as `x_czi.zarr` and `x_ims.zarr`. Before, the clash was
  resolved with parent folder names walked up to the drive (`C:_x.zarr`), a name Windows
  refuses, and both conversions failed.
- A codec the install lacks stops a file at its first output, with one error naming the
  fix (`ReaderCapabilityError`: micro-reader's "JPEG XR data needs ... the [codecs] extra").
  Before, every output of the file was tried, written and removed again: 1500 warnings
  for the 1500 tiles of a slide scan, and the last tile's store left behind.
- One unit per kind in every output: lengths in x's unit (an OIB states its z step in
  nanometres and its pixel size in micrometres), time in seconds (Micro-Manager's
  `Interval_ms`); scales and translations of every level are converted.
- A Micro-Manager `metadata.txt` given as the input is converted; before, every `.txt`
  was read as a conversion table.
- eubi-bridge-lite's error for a file it cannot read no longer says the full eubi-bridge
  reads it: it says the file is not an image nor a part of one, points to a plate's index
  file, and says the full eubi-bridge may read other image formats.
- An `ome_zarr_version` eubi does not write is an error naming the versions it writes.
  Before, it fell back to `zarr_format` (OME-Zarr 0.4) with only a log line.

## 0.3.1 (2026-10-22)

### Fixes

- Outputs written with their OME-XML (`save_omexml`, the default) still hold it at
  `OME/METADATA.ome.xml`. What changes: the `OME` folder is no longer a zarr group (no
  `OME/zarr.json`), just a folder with the XML file. ozp 0.3.0 made it a group with
  `series: ["0"]`, which NGFF allows only in a bioformats2raw layout and which validators
  that walk the whole hierarchy reject. Pixels and the image metadata were never
  affected. Requires `ome_zarr_pyramid>=0.3.1` (was 0.3.0), which makes this fix, and
  writes each multiscale's downscaling `type`.
- Repair outputs of 0.3.0 in place, without rewriting pixels:
  `ome_zarr_pyramid.io.update_metadata(path)` removes only `OME/zarr.json` and keeps the
  XML.

## 0.3.0 (2026-10-21)

eubi-bridge and eubi-bridge-lite 0.3.0 are the first releases that read micro-reader 0.1.0, which names the image index `series` (it was `scene`). They require `micro-reader>=0.1.0,<0.2`.

### Breaking changes

- **Requires micro-reader 0.1.0** (`micro-reader>=0.1.0,<0.2`). eubi reads micro-reader's `series` index; micro-reader 0.0.x is no longer supported. With another micro-reader forced in (`pip install --no-deps`, which `pip check` reports), a conversion that would read with micro-reader stops with `MicroReaderVersionError` (an `ImportError`) naming the fix, `pip install -U "micro-reader>=0.1,<0.2"`, instead of going to the older readers. Files take the same reader as before: micro-reader where it read them, the older readers (Bio-Formats, bioio, CZI, IMS, H5) elsewhere.
- **Requires ome_zarr_pyramid 0.3.0** (`ome_zarr_pyramid[remote]>=0.3.0,<0.4`). ozp 0.3.0 writes a translation on every level for each image that has a stage position, `(0, 0)` included, and takes the downscale method into account. eubi's own workaround for the image at `(0, 0)` is gone.
- **dyna-zarr is bounded**: `dyna-zarr>=0.1.0,<0.2`. eubi-bridge-lite now names it too (before, it came only through ome_zarr_pyramid).
- The same three bounds are in `pyproject.toml` (`[project]` and `[tool.eubi-bridge.lite]`) and `requirements.txt`.
- **A development install must be reinstalled.** `eubi_bridge.__version__` comes from the installed package's metadata, so an editable install made before this release still says 0.2.0, and micro-reader 0.1.0 refuses to load beside it (see "Guards"). Run `pip install -e .` again (or `python tools/build_lite.py` and install the lite wheel).

### Renamed: scene_index is now series_index

- The option that picks the series to convert is now `series_index` (same meaning, same values: an integer, `all`, or a list such as `0,2,4`). It is called series because micro-reader and OME-Zarr call it so; some formats call a series a scene.
- **The old name still works**, with a `FutureWarning` ("scene_index is deprecated; it is now called series_index (same meaning)"). It works everywhere a setting can be given: `eubi to_zarr --scene_index 1`, `eubi configure readers --scene_index 1`, scripts (`to_zarr(..., scene_index=1)`), saved configs, conversion tables with a `scene_index` column, and saved GUI batches. When both names are given with different values, `series_index` wins, with a warning naming both (below).
  - Removal: the old name (and the old keys and columns in saved files) is kept for several releases. It is removed in a planned minor release, announced in these notes, after the alias has shipped in at least three releases.
- **Files eubi saves hold both names**, with the same value, for several releases, so eubi 0.2.x and 0.3.0 can share them and read the same series:
  - the saved config (`.eubi_config.json`, and named configs): `readers` holds `series_index` and `scene_index`. A config saved by 0.2.x (only `scene_index`) loads with its value kept and is saved again with both.
  - a GUI batch: `batch.csv` has a `series_index` and a `scene_index` column, and `batch_config.json` has `seriesIndices` / `readAllSeries` and `sceneIndices` / `readAllScenes`. A batch saved by 0.2.x (only the old names) loads with one warning for each file.
  - Where both names are present and agree, they load without a word. Where they differ, eubi 0.2.x changed its own name: eubi uses `series_index` and warns (`SeriesIndexConflictWarning`, also in the log), naming both values, the file and, for a batch, the row. To settle it, set the series you want as `series_index` (in 0.3.0; it writes both).
  - **The next save by 0.3.0 settles a differing pair to `series_index`.** 0.3.0 saves the series under both names, so the next time it saves such a config or `batch.csv`, for any reason (a setting of another section, a GUI settings save, a load that updates the file, saving the batch), it writes the `series_index` value as `scene_index` too: the value 0.2.x set is replaced, and 0.2.x then reads the 0.3.0 series. The warning says so, naming both values and which one wins. A batch's `batch_config.json` is saved back as it was loaded, so a differing pair stays as it is; its warning says that.
  - A conversion table you write yourself may have either column, or both.
- **The output metadata holds both names.** The `eubi_bridge` block of an output that is one of several (a series, tile, view or illumination of a file) now holds `series_index` and, for several releases, also `scene_index`, with the same value. Scripts that read `scene_index` keep working.
- **Output names do not change.** The outputs of a multi-series file are still named `<file>_scene<N>.zarr`: scripts depend on file names. A change would be announced in a planned minor release first.
- In the GUI, the Reader tab's "Scenes" group and the batch column "Scene index" are now "Series" and "Series index".
- Messages say series: "No valid series index in [5]: the file has 1 series, so valid indices are 0..0.", and an aggregative conversion of files with several series names `--series_index`.
- `eubi show_pixel_meta` labels the summaries of a multi-series file `[series N]` (was `[scene N]`).

### Tiles words

- eubi asks micro-reader for `tiles="mosaic"` (micro-reader 0.1.0's word; `"stitched"`, its old name, still works there with a warning). eubi's own option is unchanged: `--as_mosaic` places a mosaic's tiles as one image.

### RGB channel names

- **A visible change for every one-channel RGB file** (RGB TIFF, JPEG/PNG, a Bgr24 CZI, an RGB ND2, ...): its samples, written as 3 channels, are now named and tinted as several RGB channels' samples already were. A named channel gives `"<name> R"`, `"<name> G"`, `"<name> B"`, tinted red / green / blue (an ND2 `Brightfield` gives `Brightfield R/G/B`, a CZI `Bright` gives `Bright R/G/B`); before, the name was lost and the channels were `Channel 0/1/2`, white. An unnamed one keeps `Channel 0/1/2`, now tinted red / green / blue too. RGBA adds `A`, white; other sample counts are numbered (`"<name> 0"`, ...) and stay white. Every route gives the same names: micro-reader, the older readers, `show_pixel_meta`, files joined by an aggregative conversion and views joined along c (`View0_Bright R`, ...). Your `channel_labels` / `channel_colors` (CLI, GUI, batch table) still win, and a file whose reader already lists the 3 channels (Bio-Formats for some ND2) is written as before.

### Fixes

- A conversion table whose column has the old name of a renamed setting (`scene_index`, `max_workers`, ...) uses the table's values. Before, the run-wide value of the new name filled in a column of that name and was used instead.

### Guards

- **micro-reader 0.1.0 refuses to load beside an older eubi.** `import micro_reader` raises an `ImportError` when eubi-bridge 0.1.3 to 0.2.x or eubi-bridge-lite below 0.3.0 is installed, because those releases read `scene`. A user of eubi-bridge 0.1.x or 0.2.x who upgrades micro-reader alone sees: "eubi-bridge 0.2.0 does not support micro-reader 0.1.0: this micro-reader names the image index 'series' (it was 'scene'), and eubi-bridge 0.2.0 reads 'scene'. Either upgrade: pip install -U 'eubi-bridge>=0.3.0', or keep eubi-bridge 0.2.0 with the micro-reader it was built for: pip install 'micro-reader<0.1'." Working on an eubi checkout, `MICRO_READER_SKIP_EUBI_CHECK=1` skips the check, with a warning.
- **The pin releases 0.1.7 and 0.2.1** of eubi-bridge and eubi-bridge-lite require `micro-reader>=0.0.3,<0.1`, `ome_zarr_pyramid[remote]` below 0.3 and `dyna-zarr>=0.1.0,<0.2`, so pip never installs them with micro-reader 0.1.0. Users who stay on the 0.1 or 0.2 line get those.
- eubi-bridge 0.3.0 checks the installed micro-reader once per process, the first time it reads with it: outside `>=0.1.0,<0.2` it raises `MicroReaderVersionError`, see "Breaking changes".
  - **It reaches a script as itself.** `EuBIBridge().to_zarr(...)` (and `show_pixel_meta`) raise `MicroReaderVersionError`, also when a worker process raised it, so `except MicroReaderVersionError` (or `except ImportError`) catches it. Before, the dispatcher wrapped it in its `RuntimeError("1/1 tasks failed: ...")`, with the name only in the text. `eubi to_zarr` still exits 1 with the `Fix:` line and writes nothing.
- The `scene_index` alias and the `scene_index` key in the output metadata and in the saved files, see above, with their removal condition.

### Supported consumers

- eubi-bridge and eubi-bridge-lite are consumers of micro-reader, ome_zarr_pyramid and dyna-zarr; this release supports micro-reader 0.1.x, ome_zarr_pyramid 0.3.x and dyna-zarr 0.1.x.
- The supported eubi lines are the newest two minor lines: 0.3.x and 0.2.x (0.2.1 is the 0.2 line's pin release). The 0.1.x line (0.1.7 its pin release) stays usable with micro-reader below 0.1.
