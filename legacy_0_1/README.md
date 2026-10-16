# legacy_0_1: the conversion code of eubi-bridge 0.1.x

eubi-bridge 0.2.0 converts every file through `eubi_bridge/conversion/pyramid_path.py`
(single files) and `aggregative_path.py` (files joined into one image), and writes every
output with ome_zarr_pyramid. The 0.1.x route was retired in 0.2.0 (roadmap step 6).
This folder keeps it for reference. Nothing here is imported, packaged or tested.

Drop this folder once a few releases have shipped without needing it.

## What was retired

| What | Where it lived | What replaced it |
|---|---|---|
| eubi-bridge's own writer (`writer='eubi'`) | `core/writers.py` | ome_zarr_pyramid `IO().write_pyramid` |
| The adapter to ome_zarr_pyramid (`writer='ozp'`/`'auto'`) | `core/ozp_writer.py` | `conversion/pyramid_path.py`; its helpers moved to `core/output_store.py` |
| `ArrayManager`, `ArrayState`, `BatchManager` | `core/data_manager.py` | ozp `Pyramid`s, built in `pyramid_path.py` |
| `SceneLoader.load` / `_snapshot` | `core/data_manager.py` | `conversion/legacy_image.py` (`LegacyFile`) |
| The 0.1.x conversion steps (`_process_single_scene`, `_load_input_manager`, `aggregative_worker`, ...) | `conversion/conversion_worker.py` | `pyramid_path.convert`, `aggregative_path.convert_group` |
| `AggregativeConverter` | `conversion/aggregative_conversion_base.py` | `conversion/aggregative_path.py` |
| `BatchFile` | `conversion/fileset_io.py` | `aggregative_path.py` |
| The 0.1.x metadata commands (`update_pixel_meta`, `update_channel_meta`, `show_pixel_meta`) | `conversion/metadata_update_worker.py`, `utils/metadata_utils.py` | `conversion/metadata_path.py` (ozp `IO().update_metadata`) |
| The GUI on eubi-bridge's own Pyramid | `qt_gui/server/zarr_plane_server.py`, `qt_gui/pages/inspect_page.py` | ozp `Pyramid` and `IO().update_metadata` |
| The pixel check against the standard reader (`micro_check_pixels`: `swap_in`, `verify`) | `core/micro_source.py` | none (to compare, convert once more with `pixel_reader='standard'`) |
| eubi-bridge's own NGFF layer (`Pyramid`, `NGFFMetadataHandler`, defaults) | `ngff/` | ome_zarr_pyramid's `Pyramid`; the version stamp moved to `core/output_store.py`, the defaults into `core/data_manager.py` |
| eubi-bridge's own downscalers (`DownscaleManager`, `Downscaler`, `ts_downscale`) | `core/scale.py` (its smart scale factors stay) | ome_zarr_pyramid's `downscale` |
| The older readers' OME-Zarr reader (`NGFFImageMeta`, `read_pyramid`) | `core/data_manager.py`, `core/pyramid_reader.py`, `core/readers.py` | `conversion/legacy_image.py` (`NgffFile`: ome_zarr_pyramid reads OME-Zarr inputs) |
| The `EUBI_NEW_PATH=0` switch back to all of the above | all of the above | none |

These settings still load from configs, batches and scripts. They warn once and do
nothing:

- `writer` (`'eubi'` is written by ome_zarr_pyramid);
- `queue_size`;
- `max_concurrent_downscale_layers`;
- `tensorstore_data_copy_concurrency`;
- `micro_check_pixels`.

## The files here

- `eubi_bridge/`: each file as it was at commit `90ab07f` ("ozp-integration 3/4"), the
  last commit before step 6, where the 0.1.x route still ran. Files that 0.2.0 only
  trimmed (`ebridge.py`, `dispatcher.py`, `readers.py`, ...) are copied whole, so every
  removed branch can be read in its context.
- `tests/`: the tests that tested only the 0.1.x route, from the same commit.
- `tests/golden_recording/`: the versions of the comparison tests that recorded the
  golden files in `tests/golden_0_1/`, together with `_golden.py`. Each test ran
  every case on the 0.1.x route and on the new one, and checked the new output against
  what it recorded. They ran on `90ab07f` plus the step 6 changes, with
  `EUBI_RECORD_GOLDEN=1`. The tests in `tests/` now compare with those recorded outputs.

To run the 0.1.x route itself, check out `90ab07f` and set `EUBI_NEW_PATH=0`.
