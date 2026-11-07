# Changelog

## 0.3.4 (2026-11-07)

Requires micro-reader 0.4 (`>=0.4.0,<0.5`) and ome_zarr_pyramid 0.6 (`>=0.6.0,<0.7`): with
the older ones (an upgrade that kept them) units were invented again, with no `NOT
CALIBRATED` line. With another one forced in (`--no-deps`), a conversion stops with
`MicroReaderVersionError` or `OzpVersionError` naming the fix, on every route; upgrade all
three: `pip install --upgrade eubi-bridge micro-reader ome-zarr-pyramid`. micro-reader 0.4.0
names images by one rule (the OME `Image Name` eubi
writes), opens in `t c z y x` by default and reads plates as plates by default;
eubi-bridge asks for the axes and plates it reads itself, and its output paths do not
depend on image names. ome_zarr_pyramid 0.6.0's `select` refuses
a name no plate has; eubi-bridge selects by position only.

### Changed
- Memory and cores that finished conversions held go to the ones still running. A run's
  budget was split once over `max_processes`, and each conversion kept its part to the end: of
  five wells on 8 GB, the two large ones wrote on with 2 GB each after the three small ones had
  ended. Now, once fewer conversions are left than run at once, each busy one may take the run's
  memory over the conversions left (never past what the others hold), with its auto threads.
  A conversion that starts late plans with it; one that is writing takes it before each
  pass and, every few seconds, during one. One line says so (`<output>: conversions of this run
  have ended; this one now has ... MB`). Needs dyna-zarr 0.1.2; with an older one, a line
  says that freed memory is not handed on.
- Tiles are stitched by default (`as_mosaic=True`, CLI and GUI alike): a tiled slide gives one
  image, not hundreds of stores. Tiles sit at their stored positions, nothing is registered or
  blended, and where tiles overlap one tile gives the pixel: use `--as_mosaic False` to keep the
  tiles for stitching elsewhere. A chosen `mosaic_tile_index` keeps tiles too. A file whose tiles
  cannot be placed keeps them separate, with a warning. Joined files are stitched as well: each
  file's canvas is joined (they were read as their first tile, in silence). `--force_bioformats`
  leaves tiles as Bio-Formats gives them. A saved `as_mosaic: false` is adopted once, with a line
  saying so.
- A joined output holds one image of each file, and now the one asked for: `--view_index`,
  `--illumination_index`, `--rotation_index`, `--phase_index` and `--mosaic_tile_index` were
  ignored. A file with several views or illuminations and no choice is refused, as a file of
  several series is; before, the first was written and the others left out in silence.
- `--overwrite` replaces a local store only when the new one is complete: it is written beside the
  old one (`<name>.writing-<id>.zarr`), which needs room for both; a line says so when the drive
  has less room than the old store takes. The old store goes aside by a rename and is deleted
  last, so a store in use stays whole, with an error saying so. A scene that failed left an
  all-zero store that looked finished, where a good one had been.
- Tags have one reader. What joined runs do with them changed, on purpose:
  - Files join in the order of the tag's number, whatever else varies in their names: `b_T1`,
    `a_T2`, `c_T3` were joined as 2, 1, 3 (by their paths). A line says so when numbers ahead of
    the tag (a date, a run) would order them otherwise: a tag that starts again each day joins the
    days out of order. Named tags (`Blue,Red`) join in the order given. The output takes the
    first file's metadata and name by the same order: its name with only the tag replaced
    (`aaa_Red`, `zzz_Blue` with `--channel_tag Blue,Red`: `zzz_BlueRed_cset`), where 0.3.3 glued
    every file's varying parts into one long name.
  - A numbered tag's value is the digits right after the tag, and nothing else of the name: in
    `s_C001Z001`, `C` is 1 and `Z` is 1, each given as its own tag. Numbers spread over several
    fields (`T00h15m30s`, `z0.5`) are not supported: `T` reads 0, `z` reads 0. A dimension
    without a tag is not read either: in `T0001F001`, two fields of one time point are two
    files at one place, refused (0.3.3 could join them as time points, in silence). Of named
    tags, one holding another is searched first (`GM2_GD2` before `GD2`), and its files are not
    searched for the names it holds; the join keeps the order given, and the output is named as
    for any named tags (`x_GD2GM2_GD2_cset`). A file's extension holds no tag.
  - A tag is read as a word of its own first: `ch` in `batch3_ch1` is `ch1`, not `ch3`, which
    turned channels into time points or wrote planes twice. A file with the tag only inside a word
    (`Test1.tif`) is left out when others have it as a word. A single tag needs its number:
    `Thumbnail.tif` no longer joins as a time point of `--time_tag T`.
  - The tags are checked against `--concatenation_axes` before any file is grouped: an axis
    joined without its tag, an axis named twice, a letter that is no axis, and one tag given
    for two axes (`frames/` and `frames` are one tag; `--z_tag ch` is read in `--channel_tag
    ch1,ch2`), a name given twice, and one name only for a joined axis (`--channel_tag Blue, Red`
    unquoted) are refused in one line (a raw error, or one output in silence, before). A tag
    of an axis that is not joined splits the outputs, one per value, and a line says so. Joining
    along several axes joins each in its own lines: along t, the files of one channel.
  - A file needs every tag given; those without are named in a warning and not converted.
  - Two files at one place of the joined tags (`a_T1.tif`, `a_T1_copy.tif`, or `pos1_T1`, `pos2_T1`
    with no tag for the position) are refused: they became extra planes in silence. A grid with a
    place empty is refused in a line. Files of one channel beyond the first, with named channels,
    became stores of their own; they are refused the same way.
  - Tags are read in the same place by the plan and by the worker, which sees one output's files:
    `gfp_mcherry/{gfp,mcherry}/a_T1..3` gave 9 time points for 3 (36 in 0.3.3).
- An output carries its image's name in `multiscales` and in its OME-XML (`Series 0` and
  `Default Image` before); file names are unchanged. Joined sets keep `Series 0`.
- KLB files are read by micro-reader 0.4.0, with and without Java, in eubi-bridge-lite too:
  every time point (Bio-Formats gave the first of `test.klb`'s 10, in silence). A 1 in the
  header is its default, no measurement.
- A partial conversion, where some of a file's outputs fail, ends with an error and exit code 1,
  so a batch script does not read it as a success; the outputs written stay.
- Every image of a file is converted by default, on the command line as in the GUI:
  `series_index`, `mosaic_tile_index`, `view_index` and `illumination_index` default to
  `all`. Before, the CLI wrote the first only and said nothing: a VSI gave its label
  photo, a CQ1 plate its title thumbnail. A config saved by an earlier version holds the
  old 0, saved on the first run and not chosen: it becomes `all` once, with a line saying
  so. Set them to 0 to convert the first only. Every installation default now agrees
  between CLI and GUI, and a test holds them together (`jvm_memory` too).
- Unknown calibration is no longer invented (with ome_zarr_pyramid 0.6.0): an axis the
  file gives no size for has scale 1 and no unit, instead of 1 micrometer or 1 second, and
  z is never copied from x. A loud `NOT CALIBRATED` line per file names the axes and the
  options that give the sizes. NGFF does not require a unit, and Neuroglancer reads it. The
  OME-XML beside an output leaves such an axis out (it gave 1 µm and 1 s), and an OME-Zarr input
  with no unit on an axis keeps it that way (it gained `micrometer`), its scale and translation
  with it. A scale given with no unit is a relative size, left out of the OME-XML too.
- A conversion table whose aggregative groups are partly refused (rows disagreeing on a
  concatenation setting, an unusable tag, two files at one place) converts the rest: each
  refused group is set aside, and the run ends with exit code 1 and a list of what was not
  converted, each table row with its index (0 = the first row under the header), its input
  path and the cells the table gives it. Before, one such group stopped the whole table.
- Outputs of same-named files in different folders are named after the folders below the
  inputs' common folder, every folder climbed kept: `coll/a/img.tif`, `coll/b/img.tif` and
  `coll/c/a/img.tif` give `a_img`, `b_img`, `c_a_img` (before `coll_img`, ... and `c_img`);
  `r1/s/img.tif` and `r2/s/img.tif` give `r1_s_img` and `r2_s_img` (before `r1_img`).
- A slide's associated images carry their role in the output's name when a file gives
  several images: `Image_V4.1_BF_scene0_label`, `_scene1_overview`, `_scene4_macro`; the
  scans keep their names.
- An index option on a file read as a plate (a well-plate CZI) is refused in one line naming
  `--as_plate False`: `--series_index`, `--view_index`, `--illumination_index`,
  `--mosaic_tile_index`, `--phase_index`, `--rotation_index`. Before, they were ignored and
  the whole plate written (a tile index gave a wrong hint about `as_mosaic`). The defaults
  convert the plate.
- An index option the file cannot meet is refused in one line, the same way for every index:
  `--view_index 3` or `--illumination_index 1` on a file without views or illuminations
  (ignored before, exit 0), a tile index past a series' tiles ("series 0 has 4 tiles
  (0..3)"; before, a wrong reason about unstitched mosaics), a series index past the file's,
  a tile index on a file without tiles ("has no tiles"), `--phase_index` or
  `--rotation_index` on a file without phases or rotations (ignored before).
- A gap in the numbers of a joined tag (`T1`, `T2`, `T4`) is said in a warning: the values
  found are joined one after the other. A tag is read below the folder the files share, and
  from that folder's own name only when no file has it below (`frames/000/a.tif`); before,
  it was looked for in every folder up to the drive (`t11` from a folder named `hunt11`). When
  no file carries it, the line names a tag that differs only in case
  ("did you mean --time_tag T?").
- Float images get a display window from their data (through ome-zarr-pyramid 0.6.0): the
  window of the type's whole range showed them as flat grey.

### New

- Each run's log is also written to a file, line by line, so it survives a crash that ends
  the terminal or the remote desktop (a BARD report): `<config folder>/logs/eubi-<start
  time>-<process id>.log` (the config folder: `EUBI_CONFIG_DIR`, else `~/.eubi_bridge`), the
  worker processes' lines and the error that stopped a run included; the newest 20 are kept.
  The run's first line names the file. `--log_file <path>` writes there instead,
  `--log_file False` writes none. The GUI's Convert page has an "Open Log Folder" button.
- Every image index converts (decision E-1): a file with `block`, `excitation`, `lifetime`
  or a BigDataViewer file's `sequence` gives one output per value, its name ending in the
  index (`scan_block1.zarr`). Before, such a file went to the older readers, and
  BigDataViewer `sequence` files did not convert at all. `phase_index` and `rotation_index`
  still pick one value, for one output named as before; a value the file does not have is
  an error.

### Fixed
- `memory_budget='auto'` inside a SLURM job or a container is half of what the job's memory
  limit leaves, not half of the node's free RAM (a BARD report: on a large node the plan was
  far bigger than the job's limit, and the kernel ended the whole job, remote desktop
  included). The limit is read from the process's cgroup (v2 `memory.max`, v1
  `memory.limit_in_bytes`, file pages the kernel can drop not counted as used) and from
  `SLURM_MEM_PER_NODE` / `SLURM_MEM_PER_CPU`; the least of these and the free RAM counts. One
  line names the budget and where it came from (`memory_budget='auto': 8.0 GB, half of the
  16.0 GB the SLURM job may still use (its limit: 32.0 GB)`), and an explicit budget above it
  warns. Windows and macOS are unchanged.
- eubi runs started side by side share one memory budget instead of each planning with all
  of it (a BARD report: four batches of `memory_budget=120GB` on a 128 GB job planned 480 GB,
  and the job was killed). The runs on one machine, or in one SLURM job, keep a small ledger
  in the temp folder: they share the smallest of their budgets (a run with `auto` that
  starts beside others takes the budget already in use), and each conversion claims its
  memory when it starts and gives it back when it ends. A conversion gets its run's fair
  part (its planned memory over the runs going on), never more than the other conversions
  leave; one that would get less than half its part waits, with a line, until others end.
  Lines say how many other runs share and what each conversion got (`f0.tif: 479 MB of the
  shared 3.0 GB (2 other eubi run(s) here)`). Runs of other users, of other SLURM jobs and
  SLURM-submitted conversions (`on_slurm`) are not counted.
- A long join of many one-plane files no longer grows past its memory plan (a BARD report: a
  join of 10288 OME-TIFFs per well wrote 500 GB over a night, then took the machine down;
  reproduced on Linux, where a worker planned at 2.5 GB held 5.4 GB). The cause was not the
  data: a join kept 128 files open, while a region spans 64 planes and several regions are
  read at once, so every region reopened the files of the last; each closed file left
  tifffile's objects in reference cycles that Python frees only in a rare full collection,
  and glibc kept the memory they took. Now, for every format, a join keeps up to 1024 of its
  files open: no more than an eighth of the process's memory plan holds at what one of its
  files holds (measured on the join's own files when it starts: 0.01-0.1 MB for CZI, ND2
  and LSM samples, 0.8 MB for a LIF, 0.03 MB for a stub OME-TIFF, about 6 MB for an
  OME-TIFF carrying a 3000-plane set's XML), and as many as the process's file handles
  allow (the soft limit is raised towards the hard one); never fewer than 128, what a join
  kept before. The garbage closed files leave (TIFF, LSM and CZI files leave some) is
  collected after 256 closes, at most every 10 s; and conversion workers on Linux limit
  glibc to 2 malloc arenas (a `MALLOC_ARENA_MAX` you set stands). On the reported shape
  (4 x 2572 planes): 125 s instead of 235 s, peak 2.6 GB instead of 3.9 GB for a 2.5 GB
  plan. One line names the files kept open and what bounded them (`Joining 10288 files: up
  to 1024 kept open`). Tests follow the memory over a whole join
  (`tests/test_join_memory_flat.py`). Tested on the BARD cluster as 0.3.4b1 / 0.3.4b2
  (2026-11-06): level 0 stayed flat in all four workers.
- Writing the coarser levels of a large image no longer climbs until the machine runs out of
  memory: tensorstore `>=0.1.86` and dyna-zarr `>=0.1.1` are required. tensorstore 0.1.85
  keeps memory while the levels below full resolution are built from the stored
  full-resolution level: one Viventis well (959 planes of 14040 x 24551) went from 2.4 GB to
  115 GB in 13 minutes. At that plane size the climb measured 2.5 -> 7.9 GB in 50 s with
  0.1.85 and stayed at 1.1-1.9 GB with 0.1.86, every other package the same. pip kept 0.1.85
  in an environment that had it, since 0.3.3 accepted any tensorstore.
- Two identical rows of a table or input list are one conversion, with a line (they raced and
  could leave no store); a blank cell and `all` are one choice; rows that differ in output, series,
  tile, view, `as_plate` or `as_mosaic` stay. The files of one plate (a folder's, or listed)
  stand for its index without that line: it was given for every plate folder.
- An image of its own beside a plate's planes (a CV7000 folder's
  `unreferenced_auxiliary_image.tif`, a TIFF dropped into a Harmony `Images/` folder) is
  converted as itself, whatever the order of the inputs and with a folder as input, where it
  was lost in silence (also in 0.3.3). Other files there (Harmony's `Evaluation1/` results)
  still belong to the plate, and an ImageXpress `.mxprotocol` is no image (the run exited 1).
- `--override_channel_names` names each channel by its own file's tag: when the channels are
  split into stores (joined along t only), every store was named by the first tag (the
  documented "Concatenate Along Time Only" example: all `moving`). A file of several channels
  gets `name_0`, `name_1`, ...; before, the tags were given out by position, past the files.
- An output path with `..` in it is taken as the folder it names (every task failed with
  TensorStore's "Invalid file path"). A run whose conversions partly fail ends with the count
  and each failure in full, exit code 1, not a `RuntimeError` traceback (`TasksFailedError`,
  a `RuntimeError`, for scripts); a table row with axes but no `aggregative_group`, and
  `--concat_views` on views that cannot share one image, are refused in one line.
- The GUI keeps its recent folders in the config folder (`EUBI_CONFIG_DIR`), not always in
  `~/.eubi_bridge`.
- `--view_index`, `--illumination_index` and `--mosaic_tile_index` on a joined run through the older
  readers, which join each file's first, are refused; an out-of-range tile is refused.
- With `--metadata_reader bfio`, a file of several views or illuminations, or a mosaic that is
  not a CZI, is refused in one line naming the default reader as the way out, or `--view_index
  0` (and the like) for the first only: the older readers read the first of each, and "every
  image converts by default" gave a light-sheet CZI's view 0 / illumination 0 and a LIF
  mosaic's tile 0, with exit code 0 (also in 0.3.3). Bio-Formats itself sees them (views as
  its series, illuminations folded into its channels); reading them on this route is planned
  for the next minor release.
- An output folder inside the input folder, also with a pattern as input, a store beside its
  source when both folders are one, and a store left by a killed overwrite, are no inputs.
- An OME-Zarr input's units come out by their NGFF names (`um`, `µm`, `micron`: `micrometer`), on
  images, plates and kept levels alike, and every NGFF unit is known to the OME-XML (`inch`, `day`,
  ...). Where a time origin on such an input is not kept, a warning says so.
- A joined run planned with `validate_aggregative` no longer gives a store named `..zarr`, says
  which files it leaves out, and refuses to run with other settings than it was planned with:
  another output, input, tags, axes or filters, or no axes (it was used for them in silence).
- Joining OME-Zarr inputs works (an AttributeError in every case), as does joining CZIs through
  the older readers. A file named twice in a joined run joins once, also spelled two ways
  (`a/b.tif`, `a\b.tif`), and names given as a list lose their spaces (`(' Blue', 'Red')`).
- A joined run's warnings name the file its metadata comes from, the first by its tags. Folders
  named by a tag (`Blue/`, `Red/`) no longer give the "do not belong together" warning, the
  line on a tag not joined is printed once, and the full file list of each output is logged at
  DEBUG only (15 MB of log for 20 000 files).
- The `--force_bioformats` lines appear only where they apply: for files of several series, and
  KLB's says what may happen.
- A store holds its source's file name, not its full path with the user's folders.
- Files no tag takes are named in a warning on every joined run (the warning never fired), and
  a joined output whose files come from several folders says so.
- `--dtype` narrower than a plate's data warns that values are clipped, as for images; so do
  joined runs and OME-Zarr plates.
- `--force_bioformats` on a tiled file works again: it mixed micro-reader's stitched size with
  Bio-Formats' pixels (LIF: an error; CZI: one scene of two).
- Filters on a plate say what happens: `--includes` picking scanR planes converts the plate once
  (it wrote the whole plate once per file), part of a NIS-Elements run warns that the whole plate
  is converted, and `--excludes` warns about files of a set that is converted whole.
- Tags are read below the folder the files share, in choosing the files and in grouping them: a
  folder named like its tags (`C1_T5run`, `gfp_mcherry`, `Tdata`) scrambled a joined image, wrote
  a channel twice or joined a stray file, with exit 0. A tag that is a folder itself
  (`frames/000/`) still counts. A numbered tag's groups survive a named tag's (joining z with `T`
  and `gfp,rfp` lost the time points' grouping). One file matching two named tags is refused.
- Joining a plate file (naming `--as_plate False`), updating an RFC 8 store, and what a joined run
  refuses are one line (a `KeyError` and a traceback before).
- Each table or batch row is what it writes: one input to two outputs stays two rows; the same
  input to the same output twice is converted once, with a warning.
- A unit spelled `um` or `microns` becomes `micrometer`; one OME has no name for leaves that axis
  out of the OME-XML with a warning (a raw validation error stopped the conversion).
- The OME-XML passes the OME schema (`MetadataOnly`).
- A joined run matching one file writes a store of its name, not a hidden `..zarr`.
- The false "missing planes are written as zeros" on mixed-type CZIs and lambda scans is gone.
- `--includes` matching files of a plate says that the whole set is converted.
- Parts of a set that are missing go to the log as a warning. An Imaris set with an `.h5` copy
  beside its `.ims` converts once.
- `update_pixel_meta` and `update_channel_meta` work on plates, every field.
- A missing input, a text file that is no table and an unknown option give one line, no
  traceback. `.bin`, `.mat`, `.npz`, `.npy` and pickles among the inputs are skipped.

- Converting one file also converted every file in its folder whose name ended in that
  name (`rois.nd2` took `with_rois.nd2`).
- `--as_plate False` crashed with `KeyError: 'series'` on plate formats; it writes one
  store per image, named by what varies (`_well2_field0`).
- The documented time-only concatenation failed with an `AssertionError`.
- `show_config` crashed when its output was redirected on Windows.
- The `zarr_format` deprecation warning printed on every run, for a setting nobody chose.
- A failed input prints one line, and the traceback only at debug level (a folder of
  unreadable files gave 3000 lines); a full disk says so. macOS stub files (`._name.tif`)
  are no inputs.
- A ragged image (a channel with fewer planes than the others) says that its missing planes
  are written as zeros, plates too.
- `NOT CALIBRATED` covers plates and joined sets as well, and Bio-Formats' default size is no
  longer taken for a recorded one.
- `--includes` matching plane files converts the plate once, not once per file.
- An option eubi-bridge does not know is an error naming the nearest one (`--z_scal` gave a
  wrong result with exit code 0).
- `show_pixel_meta` reads plates, the input and the store written, and lists an image set once.
- What micro-reader says about a file it refuses stays in the error when the older readers then
  fail too. A zip input says so. Imaris 3 `.ims` files (TIFF) go to Bio-Formats.
- `--dtype` narrower than the data warns that values are clipped; a file no tag takes is named
  in a warning, instead of being left out in silence; a joined name of many tags fits a path.
- The labels warning no longer fires for plates, whose labels are written.
- Two channels of one name with RGB samples keep separate labels, as micro-reader names them.
- A lambda scan converts whole. With micro-reader 0.4.0 it is one image with its bands as
  channels, each with its emission wavelength. Before, a Leica lambda scan went to the older
  readers and wrote one plane of its 29 bands, with no error.
- A folder given as an input (a GUI batch row) stays the input only when micro-reader
  opens it as its own dataset: the file it reads is the folder, in it or beside it (a
  Cellomics, NDTiff, Luxendo or NIS-Elements run folder, an OIF storage folder).
  micro-reader 0.4.0 also opens a folder whose one dataset lies in a sub-folder (a
  sample folder over its export); such a folder converts that dataset's main file,
  named after the dataset as before, not after the folder above it. Without this,
  eubi-bridge 0.3.3 with micro-reader 0.4.0 would name those outputs after the folder.
- Channel wavelengths reach the OME-XML (`EmissionWavelength`, `ExcitationWavelength`, with
  their units) for images and plates; they were only in the channel names.
- A file listing (a GUI batch, a CSV) drops macOS `._` stub files as a folder walk does: a
  joined group failed with "2 files are at one place: x.tiff, ._x.tiff". The GUI browser does
  not tick them.
- `folder/**/*.tif` takes the folder's own files and every depth below (it skipped the
  folder's own), without the chunks of `.zarr` stores in it.
- A table row's `output_path` with `..` (`sub/../out`) is the folder it names; every such
  row failed.
- A TIFF named like a Harmony plane that the plate's index does not list converts as an image
  of its own (it was folded into the plate and lost); plane files given without their index
  still stand for their plate.
- A Yokogawa CQ1 measurement folder converts with exit code 0: its calibration images and its
  projection companions, which the measurement does not list, are skipped with a line among
  several inputs (they counted as failed tasks); a file given alone still fails.
- A size given without its unit beside an axis that has one (`--y_scale 0.1 --x_scale 0.1
  --x_unit micrometer`) is said in a warning naming the option to add; before, y was written
  as a relative size in silence.
- `--force_bioformats` on a tiled LIF writes each tile at its stage position, as the default
  reader does; the tiles had no positions.
- Version guards read a pre-release's release (`0.5.0rc1`): it was let through.
- One line, no traceback, for: a folder with no image, `--io_backend zarrista` without
  zarrista installed, what eubi-bridge-lite does not include, and the refusals of a joined set
  (two formats, pixel types, sizes or series in one output).
- One `.nd2` of a NIS-Elements well-plate run converts the whole run (it is the plate); the
  line saying so names `--as_plate False`, which converts just that file.
- The command reference (`docs/cli_reference`) gives each option's real default (`Default:
  default` 92 times before; `None` for `mosaic_tile_index` and `queue_size`) and its whole
  description (wrapped lines were cut).
- A wrong option value (`--compressor lzma`, `--downscale_method gaussian`, `--x_scale -1`,
  `--ome_zarr_version 0.7`) and a forgotten output folder give one line, `ERROR: --option:
  message (given: value)`, instead of a 25-line traceback.
- The Python API takes `includes=` / `excludes=` for a list of files too (every listed file
  was converted). A file you name that is not an image is said in a warning (an info line,
  hidden by default, before).
- eubi-bridge-lite converts a Yokogawa CQ1 measurement folder with exit code 0, as the full
  package does: the calibration images its measurement does not list go to no other reader
  (lite failed the run; the full package started Java to skip them).
- The GUI's batch log shows a run's report (refused groups, failed files) as text, not a
  Python traceback; a refused table row is named as "row 7 (table index 6)", the row the
  GUI numbers 7.
- `NOT CALIBRATED` for a file of several images where only some lack sizes (a slide's macro
  photo) names those images and no longer suggests the size options, which would set every
  image of the file. The line names the input by its whole path.
- A joined run with an empty place in its grid names it: "no file for T4 Channel2".
- `update_pixel_meta --x_scale ...` keeps the coarser levels on level 0 (through
  ome-zarr-pyramid 0.6.0's `update_metadata`).
- A plate folder holding a stray plane-named TIFF no longer warns that its files convert
  the whole plate (the stray converts as an image of its own, as it should).

## 0.3.3 (2026-10-28)

Needs ome_zarr_pyramid 0.5 (`>=0.5.0,<0.6`) and reads micro-reader 0.2.1 to 0.3.x
(`>=0.2.1,<0.4`). The plate formats below need micro-reader 0.3.0; with 0.2.1 they go
to the older readers as before.

### New

- **Plates from more instruments**, read by micro-reader 0.3.0 and written as one NGFF
  plate each: PerkinElmer Opera Flex (`.flex`; a folder of several barcodes gives one
  plate each), Columbus exports (`ImageIndex.ColumbusIDX.xml`), Thermo Fisher Cellomics
  (`.C01`: any file or the folder opens the plate), Molecular Devices ImageXpress HCS.ai
  (`.jdce`), Olympus / Evident scanR (`experiment_descriptor.xml`), OME companion files
  that describe a plate, ND2 files whose points are named after wells, and NIS-Elements
  well-plate runs (one ND2 per well: the folder or any of its files gives one plate, not
  one copy of the plate per file; with `as_plate=False` each file is converted on its
  own). Yokogawa CQ1 (`MeasurementResult.ome.tif`) is a plate when its images sit in two
  or more wells. Each checked end to end on the samples held, from the command line and
  from a GUI batch. A plate is named after its folder when its file's name says nothing
  about it: scanR `experiment_descriptor.xml`, Columbus `ImageIndex.ColumbusIDX.xml`,
  Cellomics `.C01` files and NIS-Elements run files (as Harmony's `Index.idx.xml`
  already was), so choosing the folder or any of its files gives the same name; with
  `as_plate=False` each file keeps its own.

- **OME-Zarr plates as input.** An NGFF 0.4, 0.5 or 0.6 plate, or an RFC 8 collection
  holding one, is written as one plate again, in the version asked for: the same rows,
  columns, acquisitions, wells and fields, every field with the job's settings, at its
  position in its coordinate system. A 0.6 field's other systems and transformations
  are carried as for a single 0.6 image. Before, a plate store failed with ozp's
  "No valid metadata found in zarr group". `as_plate=False` is refused for an OME-Zarr
  plate (convert a field's own group as an image), and so are an RFC 8 collection that
  holds no plate and a bioformats2raw layout without one (convert a series' own group),
  each with an error that says so.
- **Label images of an OME-Zarr input are named in a warning**: eubi-bridge writes the
  images only, and before it left the labels out without a word (a Fractal plate's
  `org` segmentation, for one).
- **Olympus / Evident VSI is read by micro-reader** (with micro-reader 0.3.0 or newer;
  with 0.2.x the older readers read it, and the log says why). micro-reader gives a
  slide's stored pyramid levels as levels of one image; Bio-Formats lists each level as a
  series of its own (28 series for an Evident slide of 5 images). Checked against the
  older readers on the 9 samples held: the same pixels, except lossy JPEG 2000 tiles,
  where the two decoders round up to 1 apart; reading is 10-20 times faster.
- **Slide TIFFs are read by micro-reader**: Aperio SVS, Hamamatsu NDPI, Leica SCN,
  Akoya / PerkinElmer QPTIFF and Ventana / Roche BIF. Checked against the older readers
  on the 20 samples held: NDPI, SCN, QPTIFF and SVS give the same pixels on every level
  (reading is about 14 times faster), except a JPEG 2000 SVS, where Bio-Formats' decoder
  is up to 12 off (micro-reader equals OpenJPEG and is within 1 of OpenSlide). A
  NanoZoomer's tissue map, which Bio-Formats cannot read, is converted too. BIF tiles are
  placed where the scanner's joints say, not spread evenly as Bio-Formats does, so a BIF
  output's size and seams differ from before; a BIF micro-reader does not read (another
  scanner, several focal planes) still goes to the older readers. A slide's lengths keep
  the file's unit (Leica: nanometres). eubi-bridge-lite converts these formats too.
- **ICS is read by micro-reader** (`.ics` with its `.ids`; an `.ids` given as the input
  converts its `.ics`, once also when both are in the input folder). Checked against the
  older readers on the 4 samples both read (libics, a 1.2 GB stitched stack, two Huygens
  results): the same pixels and pixel sizes; reading is about 10 times faster. Pixels in a
  Unix-compressed `.ids.Z` are read by neither.
- **HDF5 is read by micro-reader** (`.h5`, `.hdf5`): ilastik files, BigDataViewer (an
  `.h5` given converts its `.xml`), Luxendo `.lux.h5`, Fiji / Vibez files and the `.h5`
  parts of an Imaris multi-file set (they convert the set's `.ims`). Before, eubi-bridge's
  own H5 reader took every `.h5`: it converted only plain datasets, wrote them in the
  file's axis order (`tyxc`, `xyc`, which OME-Zarr does not allow) with pixel sizes of
  0.0, and failed on BigDataViewer, Luxendo, Fiji and Imaris layouts. An ilastik output
  now has the same values in t, c, z, y, x order with positive pixel sizes. A file of no
  layout micro-reader knows goes to that reader as before. eubi-bridge-lite converts
  HDF5 too. Not yet: a BigDataViewer set whose setups have no attributes (micro-reader
  calls them a "sequence"; decision E-1).
- **More formats are read by micro-reader**, checked against Bio-Formats on the samples
  held:
  - MRC (`.mrc`, `.map`, `.mrcs`, `.st`, `.ali`, `.rec`): the same values. A map whose
    header stores its axes in another order (`EMD-3001.map`) is now written in z, y, x by
    that order, with each pixel size on its own axis; Bio-Formats kept the stored order.
    A 6.8 GB `.mrc` that Bio-Formats cannot read (one plane over 2 GB) converts.
  - JPEG, PNG and BMP files: identical.
  - Leica `.lof`: identical. An `.xlif` or `.xlcf` given alone (Bio-Formats reads them
    only through their `.xlef`) converts the same images as the `.xlef`.
  - Hamamatsu `.ndpis` sets: identical, except one 8 x 8 block of the macro image, where
    Bio-Formats' JPEG decoder differs from libjpeg and PIL (micro-reader equals both).
  - Olympus FluoView tiling sets (`.omp2info`): each tile, as its `.oir` reads; with
    `as_mosaic` the tiles are placed at their nominal grid positions, so the mosaic's
    size differs from Bio-Formats' stitching.
  - OME companion files (`.companion.ome`), including a Bruker set Bio-Formats cannot
    read.

### Changed

- **`decode_threads` is `"auto"` by default** (was 4): the cores this job may use
  (`SLURM_CPUS_PER_TASK`, else the process's CPU affinity, else the machine's cores),
  shared between the conversions that run at once, at most 16, and fewer when the
  reader's share of the memory budget holds fewer blocks. A number still sets it as
  given (`--decode_threads 8`); `0` means auto. The Memory line of the log names the
  thread count. On the GUI's Conversion page the field reads "auto" at 0. A saved
  configuration that stores a number keeps it: `eubi configure cluster --decode_threads auto`
  (or 0 on the GUI) switches it to auto.
- **`max_threads` is `"auto"` by default too** (was 4): the cores over the conversions
  running at once, at most 16; scenes converted at once still share them. A number sets
  it as given, `0` means auto, and the GUI's Max Threads field reads "auto" at 0. A saved
  configuration that stores a number keeps it.
- **The `Memory:` line is logged once per plan in each process**: a plate logged it again
  for every field. Repeats go to the debug log.
- **Loading a batch over edits not saved yet asks first** (GUI, Batch tab): it names what
  the load replaces, the queue table and, when the batch brings a config snapshot, the
  parameter tabs. No leaves everything as it was.
- **An explicit `decode_memory` is capped at half the process's memory budget**, with one
  warning: before, `decode_memory 1GB` in a 1 GB budget left nothing for the writer. A
  value within half is used as given; the cap never leaves less than one decode block.
- **The biggest inputs start first.** A batch submits its jobs from the most bytes on disk
  to the fewest (a plate or file-set index counts its whole folder; a folder input counts
  what it holds; remote inputs go last), so one large file no longer runs alone at the
  end. Results, failures and the job table keep the batch's own order and numbers.
- **An output keeps the dots in its input's name**: only the suffix is removed, a two-part
  one (`.ome.tif`, `.ome.zarr`, ...) whole. `Image_V4.1_BF.vsi` gives `Image_V4.1_BF.zarr`;
  before, everything after the first dot was dropped (`Image_V4.zarr`), so
  `Image_V4.1_BF.vsi` and `Image_V4.1_FL_3D.vsi` converted one at a time wrote one output
  (the second was refused, or replaced the first with `overwrite`). Names without a dot
  before the suffix are unchanged; a file with one gets a new output name.

### Fixed

- **Each saved batch keeps its own settings file** (GUI): Save Batch writes
  `<batch name>.config.json` beside the CSV (`my_batch.csv` → `my_batch.config.json`).
  Before, every batch of a folder shared one `batch_config.json` and the last save won:
  loading an older batch silently applied another batch's settings (a batch of
  2026-10-20 loaded the zarrista / zarr v2 settings another batch saved on 2026-10-23,
  and every row failed). A batch saved before 0.3.3 uses the folder's
  `batch_config.json` only when it was saved together with its CSV; otherwise its rows
  use the settings the parameter tabs show, and the status line and the log say why.
  Saving it again gives it its own file.
- **The Convert log names the OME-Zarr version the outputs get**: `Output format:
  OME-Zarr 0.5 (zarr v3)`, and for a batch the versions its rows ask for with their
  count. It said the base settings' `Zarr Format: v2` even when every row wrote
  OME-Zarr 0.5 (zarr v3).
- **A folder chosen as a batch row** (the GUI's input browser ticks folders too) is
  grouped like the files in it: a folder whose files form one image or plate that
  micro-reader does not open as a folder (an NDTiff dataset with downsampled levels, a
  NIS-Elements well-plate run) converts that image; a folder whose files belong to the
  other rows' images ('Select All' ticks the subfolders, e.g. a Leica export's
  `Metadata`) is skipped. Before, such rows went to Bio-Formats, which cannot open a
  folder ("Access is denied"). A folder micro-reader opens itself (a Cellomics plate
  folder, an NDTiff dataset folder, an OIF storage folder) stays the input, as before.
- **An NDTiff dataset with downsampled levels converts once** (Micro-Manager Magellan
  explorations, stitched acquisitions): the image in `Full resolution/` is converted, and
  the `Downsampled_x<N>/` folders beside it, previews Micro-Manager wrote, stand for it.
  Before, each level was an output of its own (up to 11 for one dataset), named after its
  folder. Choosing the dataset's folder, `Full resolution/` or a level's folder or file
  gives the same one output; a level whose `Full resolution/` is missing converts as it
  is. NDTiff datasets without levels are unchanged. Checked on the 7 multi-level samples
  of NDTiffStorage: every tile's pixels equal micro-reader's.
- **`io_backend='zarrista'` without zarrista installed** stops the run once, before any
  file is read, with `IOBackendUnavailableError` (an `ImportError`) naming the two ways out:
  `pip install "dyna-zarr[zarrista]"`, or `eubi configure conversion --io_backend tensorstore`
  (Byte Backend on the GUI's Conversion page). A saved configuration may name it; before,
  every file failed in its worker with a traceback of its own. A batch table row naming
  zarrista is checked the same way, before any job starts.

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
