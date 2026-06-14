# CLI API Audit — Findings Tracker

Working document for the `eubi` entrypoint audit (see plan: CLI API audit + documentation).
One section per command group. Each command: **Status**, **Doc fixes**, **Behavior fixes**, and a
group-level **Breaking changes** subsection (the audit trail for approved breaking changes).

Status legend: ⬜ not started · 🔄 in progress · ✅ audited & fixed · ⏳ blocked on real data

---

## Harness

- `tests/test_cli_commands.py` — drives commands in-process (`EuBIBridge(configpath=tmp)`) and a
  few real `eubi` subprocess calls. Includes an automated **signature↔doc parity** test.

---

## Display & Info + Reset

| Command | Status | Notes |
|---|---|---|
| `version` | ✅ | Smoke test passes (`test_version`). |
| `show_config` | ✅ | Default + named display tested. |
| `show_root_defaults` | ✅ | Smoke test passes. |
| `reset_config` | ✅ | Mutate→reset roundtrip restores defaults (`test_reset_config_roundtrip`). |

### Breaking changes
_(none)_

---

## Configuration

| Command | Status | Notes |
|---|---|---|
| `configure cluster` | ✅ | Persist + omit-keeps-value + invalid-value-rejected all pass. |
| `configure conversion` | 🔄 | Docstring uses grouped `a / b / c:` lines for `*_chunk`, `*_shard_coef`, `*_range`; the Args parser only captures the first name, so `channel_chunk`, `z_chunk`, … fall back to `_FIELD_HINTS`. Acceptable for generated docs but docstring is terse. `override_channel_names` not in Args block. |
| `configure downscale` | ✅ | **Fixed doc bug:** docstring said `n_layers` "default 5"; corrected to "None = auto" (ebridge.py). |
| `configure readers` | ✅ | Smoke-tested via persistence/omit tests. |
| `configure concatenation` | ✅ | Smoke-tested. |

### Breaking changes (applied)
- **Restructured five top-level `configure_*` commands into a single `configure` group**
  with subcommands `cluster` / `conversion` / `downscale` / `readers` / `concatenation`.
  - Was: `eubi configure_cluster --max_workers 8`  →  Now: `eubi configure cluster --max_workers 8`.
  - Rationale (user): better discoverability (`eubi configure` lists the sections), fewer top-level commands.
  - Implementation: new `ConfigureGroup` class in `ebridge.py`; `EuBIBridge.configure` property returns
    `ConfigureGroup(self._cfg)` bound to the active config (composes with `with_config`).
  - Verified: `eubi configure` lists sections; `configure cluster --max_workers 8` persists;
    `with_config hpc configure conversion --zarr_format 2` writes to the named config.
  - Doc generator updated: `COMMAND_GROUPS` uses dotted names (`configure.cluster`), `_resolve_method`
    resolves them, cards render as `configure cluster` with `eubi configure cluster …` usage.
  - No internal/GUI callers affected (`ConfigManager.configure_*` impl unchanged; GUI doesn't call these;
    `eubi-flow`'s `configure_*` are a separate namespace).

### Candidate breaking change (still open)
- `'default'`-string sentinel on `int`/`bool` params (now in `ConfigureGroup`): type-dishonest, surfaces
  oddly in `--help`. Superseded discussion led to the group refactor above; the sentinel itself is
  unchanged. Revisit after data-dependent commands.

---

## Named Configs

| Command | Status | Notes |
|---|---|---|
| `with_config` | ✅ | **Fixed:** added `Args: name`. Override + missing-name-raises tested. |
| `save_as` | ✅ | **Fixed:** added `Args: name`. Save/list/delete roundtrip tested. |
| `update_config` | ✅ | **Fixed:** added `Args: name_or_path, create`. create + missing-without-create tested. |
| `list_configs` | ✅ | Exercised in roundtrip test. |
| `show_configs` | ✅ | Renders table. |
| `delete_config` | ✅ | Exercised in roundtrip test. |

### Breaking changes
_(none)_

---

## Metadata

| Command | Status | Notes |
|---|---|---|
| `show_pixel_meta` | ✅ | Fast JVM-free zarr path renders shape/scale/units/channels correctly. Real-CZI scene readout still TODO. |
| `update_pixel_meta` | ✅ | `--z_scale 2.0 --z_unit micrometer` applied and reflected by `show_pixel_meta`. |
| `update_channel_meta` | ✅ | `--channel_labels "0,DAPI;1,GFP" --channel_colors "0,0000FF;1,00FF00"` applied correctly. |

Notes: `update_*` use a spawn ProcessPool, so they must be exercised via the
`eubi` subprocess (or a real module), not an in-process stdin/heredoc script.
No behavior bugs found; parity passes for all three.

### Breaking changes
_(none)_

---

## Conversion

| Command | Status | Notes |
|---|---|---|
| `to_zarr` | ✅ | Full real-data matrix 43/43 green across all 5 categories; fixed a single-tile multi-view/illumination V/I-collapse regression (see below). |
| `validate_aggregative` | ✅ | Dry-run returns an `AggregativePlan` for a channel-concat scenario; parity OK. |

### Breaking changes
_(none)_

### Real-data audit (`tests/test_realdata_conversions.py`, marker `realdata`)
Exhaustive per-parameter matrix across **all five categories** against real files
referenced in place from `EUBI_TEST_DATA` (default `C:/Users/oezdemir/Desktop/ome/input`);
skip-if-absent. 43 cases. Selected files: workhorse CZI `T=3_Z=5_CH=2` (conversion/
downscale/cluster), `S=2_…` (scenes), `S=2_2x2_…` (tiles/mosaic), `synthetic_rgb`,
`MouseBrain_…_2Illuminations_2Angles` (views/illuminations), plus LSM/LIF/ND2 format
coverage.

Reader category tests **both modes**:
- **extraction** (one OME-Zarr per index): `scene_index all`, `mosaic_tile_index all`,
  `view_index all`, `illumination_index all`.
- **combination** (indices grouped into one OME-Zarr): `as_mosaic`, `concat_views`,
  `concat_illuminations`.

Results: **41 / 43 passed** on the full matrix (12 min). Green across ALL conversion
params (zarr v2/v3, 5 crop ranges, dtype casts, 4 compressors, save_omexml, sharding),
ALL downscale (n_layers + 6 methods), ALL 9 cluster params, scene + tile extraction/
combination, RGB, and LSM/LIF/ND2 format coverage.

**Product bug found & FIXED (regression):** single-tile multi-view/illumination CZIs
collapsed V/I to a single output — `view_index all` / `illumination_index all` produced
1 zarr, not N. Root cause: `read_czi` routed single-tile files through the **pylibczirw**
reader, which exposes only X/Y/C/Z/T (drops V/I); **aicspylibczi** exposes the full
`BVITCZYX`. Fix (`czi_reader.py`): added a `needs_view_illu` branch so any request that
addresses an individual view/illumination uses aicspylibczi regardless of tile count
(with the large-file pylibczirw fallback preserved). Verified on
`MouseBrain_…_2Illuminations_2Angles.czi`: `view_index all` → 2 views exposed,
`illumination_index all` → 2 illuminations; default (view 0/illu 0) unchanged. Relates
to [[show-pixel-meta-dimensionality]] and [[large-czi-offset-truncation]].

Test-harness bug (fixed): `_output_zarrs` counted resolution-pyramid levels as separate
outputs; now counts only leaf `*.zarr` groups.

Post-fix verification: view/illumination **extraction → 2 zarrs each** and
**combination → 1 zarr each** all pass on the MouseBrain file. **Full matrix now 43/43.**
`concat_views` / `concat_illuminations` assertions strengthened to verify (a) the channel
axis deepens (C=3 × 2 → **6 channels**) and (b) the **OMERO channel labels carry
per-index provenance** — confirmed: `View0_Cam1-T1 … View1_Cam2-T2` (and `Illu0_*/Illu1_*`),
original names + colors preserved. So combination genuinely concatenates labeled data.

**Both-combined case** (`concat_views True` + `concat_illuminations True`,
`view_index all` + `illumination_index all`): 1 output, **12 channels** = full
2×2×3 cartesian product, labeled `View{v}_Illu{i}_<name>`. Structure correct.

**Finding (color metadata — needs a judgement call):** in the both-combined output the
`Illu1_*` channels are white `FFFFFF` while `Illu0_*` keep green/red/magenta.
Mechanism (traced): the OME metadata reports channels for *all* illuminations (6 = 2×3),
and `load_views_illuminations` slices a per-illumination window
`mgr.state._channels[offset:offset+c_size]` with `offset = i_idx*c_size`
([data_manager.py:1914-1918](eubi_bridge/core/data_manager.py#L1914)). So `Illu1` gets
`channels[3:6]`, whose **source colors are already white** as reported by the reader —
i.e. eubi is faithfully copying the source OME, not miscomputing. Same slice feeds both
extraction and combination. Open question for the user: for light-sheet data the two
illuminations are usually the *same* physical channels, so inheriting illu-0 colors when
the source leaves illu-1 colorless may be the desired behavior. Labels + channel count are
correct. **Not changed pending a decision.**

**Minor finding (naming, not yet fixed):** a concatenated output is named
`..._view1_illu0.zarr` — it carries the *last* view index as a `_view1` suffix even though
it contains all views. Channel labels already encode provenance, so the per-view suffix is
misleading for concat outputs. Candidate fix: drop the `_viewN`/`_illuN` suffix when the
corresponding `concat_*` is on. Logged as [[scene-tile-output-naming]]-adjacent.
