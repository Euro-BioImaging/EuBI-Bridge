# EuBI-Bridge AI Agent Instructions

## Project Overview
EuBI-Bridge is a distributed image format converter that transforms microscopic image collections (TIFF, CZI, ND2, etc.) into OME-Zarr format (v2/v3). Key feature: **aggregative conversion** concatenates multiple images along specified dimensions for large datasets.

**Tech Stack:** Python 3.11/3.12, Dask/distributed, zarr, bioio, Bio-Formats (Java), scyjava, FastAPI

---

## Critical Architecture & Data Flow

### 1. **JVM & Scyjava Initialization (Priority: CRITICAL)**
The JVM must be initialized precisely or Bio-Formats (Java-based image readers) will fail.

**Key Files:** [eubi_bridge/cli.py](../eubi_bridge/cli.py#L1-L40), [eubi_bridge/ebridge.py](../eubi_bridge/ebridge.py#L1-L25)

**Non-negotiable initialization order:**
```python
# MUST happen FIRST in cli.py:
mp.set_start_method("spawn", force=True)  # Multiprocessing must use spawn
os.environ['MAVEN_OFFLINE'] = 'true'      # Block Maven network access
import scyjava
scyjava.config.endpoints.clear()          # Disable Maven endpoints
scyjava.config.maven_offline = True       # OFFLINE MODE REQUIRED
scyjava.config.jgo_disabled = True        # Disable JGO
```

**Why?** - Bio-Formats needs JVM but network access breaks in offline/HPC environments. spawn method prevents forking issues with Java objects.

### 2. **Reader Dispatcher Pattern**
[eubi_bridge/core/readers.py](../eubi_bridge/core/readers.py#L20-L80) implements format-specific routing:
- `.zarr` → `pyramid_reader` (NGFF format)
- `.h5` → `h5_reader` (HDF5)
- `.ome.tif[f]` → `pff_reader` (OME-TIFF with bioformats)
- `.tif[f]` → `tiff_reader` (raw TIFF, bioio-based)
- `.czi`, `.lif`, `.nd2`, `.lsm` → `bioio_reader` (via bioio plugins)

**Key Insight:** Reader selection cascades format-specific optimizations (direct zarr backend for TIFF, specialized CZI reader, etc.). Always check readers.py when adding new formats.

### 3. **Dask Lazy Evaluation for Large Arrays**
[eubi_bridge/core/data_manager.py](../eubi_bridge/core/data_manager.py#L100-L200) and [eubi_bridge/ngff/multiscales.py](../eubi_bridge/ngff/multiscales.py) use dask.array heavily:
- Images loaded as `dask.array` (delayed computation)
- Chunks defined by `autocompute_chunk_shape()` (balances memory/parallelism)
- `Pyramid` class generates downscaled versions for multi-resolution storage

**Convention:** Always use `dask.delayed` for I/O and `dask.array` for numerical ops. Call `dask.compute()` only at write boundaries.

### 4. **Aggregative Conversion Workflow**
[eubi_bridge/conversion/aggregative_conversion_base.py](../eubi_bridge/conversion/aggregative_conversion_base.py#L30-L150):
1. Read dataset metadata from filepaths (glob patterns with include/exclude filters)
2. Load images as dask arrays via `read_dataset()` 
3. Stack arrays along specified dimension (Z, T, C) via `AggregativeConverter`
4. Write to single zarr output with `Pyramid.write()`

**Data Classes:** `BatchManager` in data_manager.py handles fileset grouping and metadata aggregation.

### 5. **Parallel Execution Models**
[eubi_bridge/conversion/converter.py](../eubi_bridge/conversion/converter.py#L20-L100):
- **ProcessPoolExecutor** (default): Each worker spawns isolated JVM via `initialize_worker_process()`
- **Dask LocalCluster** (distributed): Single shared scheduler, fallback for complex dependencies
- **CLI entry point** in [eubi_bridge/cli.py](../eubi_bridge/cli.py) routes to appropriate executor

**Important:** Worker initialization MUST call `soft_start_jvm()` from [eubi_bridge/utils/jvm_manager.py](../eubi_bridge/utils/jvm_manager.py#L70-L100) to set up JVM in spawned process.

---

## Developer Workflows

### Running CLI Commands
```bash
# Unary conversion (each file → separate zarr)
eubi to_zarr input_dir/ output_dir/

# With zarr format version (v3 support)
eubi to_zarr input_dir/ output_dir/ --zarr_format 3

# Aggregative (stack multiple files into one zarr)
eubi aggregate input_dir/ output.zarr --series_dimension T
```

**Implementation:** CLI uses Fire framework ([eubi_bridge/cli.py](../eubi_bridge/cli.py#L60+)) to auto-expose methods as commands.

### Local Development Setup
```bash
# Install with dependencies (Python 3.11 or 3.12 only)
mamba create -n eubizarr openjdk=11.* maven python=3.12
pip install -e .  # Editable install

# Reset config if upgrading
eubi reset_config
```

### Key Dependencies (watch these for breaking changes)
- **zarr ≥3.0**: NGFF spec compliance, format versioning
- **dask ≥2024.12.1**: Scheduler API, array chunking
- **bioio-***: Format-specific readers (separate packages, version pinned)
- **bioformats_jar**: Java backend, auto-downloaded at setup.py

---

## Project-Specific Patterns & Conventions

### 1. **Logging Throughout**
Every module imports `get_logger(__name__)` from [eubi_bridge/utils/logging_config.py](../eubi_bridge/utils/logging_config.py):
```python
from eubi_bridge.utils.logging_config import get_logger
logger = get_logger(__name__)
logger.info("Message")  # Always use module-level logger
```

### 2. **Type Hints with Union**
Use `Union[Type1, Type2]` for flexible inputs (not type unions with `|` operator):
```python
# ✅ Correct
def read_file(path: Union[str, Path]) -> ImageReader:

# ❌ Wrong (syntax errors in Python 3.11/3.12)
def read_file(path: str | Path) -> ImageReader:
```

### 3. **Zarr Format Versioning**
Two versions coexist: zarr v2 (legacy) and v3 (NGFF v0.5 compatible). Code branches on `self._zarr_format`:
```python
if zarr_format == 3:
    # Use zarr v3 API (zarr.open_group(store, mode='r+'))
else:
    # Use zarr v2 API (zarr.open_group(store, overwrite=True))
```

### 4. **Metadata Extraction Factory Pattern**
[eubi_bridge/core/metadata_extractors.py](../eubi_bridge/core/metadata_extractors.py) uses factory to select backend:
- **bioformats** (via scyjava): Most formats, handles complex OME metadata
- **bioio** (via bioio plugins): Modern async-friendly readers
- **extension-based fallback**: For unsupported formats

### 5. **Path Handling with Glob Patterns**
[eubi_bridge/utils/path_utils.py](../eubi_bridge/utils/path_utils.py) provides:
- `sensitive_glob()`: Pattern matching respecting case on macOS
- `take_filepaths()`: CSV/XLSX parsing with validation
- `is_zarr_group()`, `is_zarr_array()`: Fast negative checks (catch-all exceptions)

### 6. **Exception Handling (Code Review Priority)**
All bare `except:` statements have been replaced with specific types. Example pattern:
```python
try:
    zarr.open_group(path, mode='r')
except (ValueError, KeyError, OSError):  # Specific exceptions only
    return False
```

---

## Integration Points & Cross-Component Patterns

### Data Flow Through Core Modules
```
CLI (fire.Fire) 
  → ebridge.AggregativeConverter / converter.run_conversions()
    → data_manager.BatchManager (metadata collection)
      → readers.read_single_image() (format dispatch)
        → reader_interface.ImageReader (abstract)
          → {pyramid_reader, tiff_reader, czi_reader, ...}
      → ngff.multiscales.Pyramid (downscaling, NGFF metadata)
    → writers.create_zarr_array() (zarr output)
```

### Remote Storage (S3)
S3FS integration in [eubi_bridge/conversion/converter.py](../eubi_bridge/conversion/converter.py): Import `s3fs` then pass S3 paths (e.g., `s3://bucket/prefix`) directly to readers/writers. zarr auto-detects S3 via fsspec.

### Metadata Updating After Conversion
[eubi_bridge/conversion/updater.py](../eubi_bridge/conversion/updater.py) + [eubi_bridge/conversion/metadata_update_worker.py](../eubi_bridge/conversion/metadata_update_worker.py): Async post-processing to inject channel colors, scale metadata into zarr `.zattrs` files.

---

## Known Gotchas & Debugging Tips

1. **JVM hangs on startup**: Verify `scyjava.config.maven_offline = True` is set BEFORE bioio/bioformats import. Check MAVEN_OFFLINE env var.
2. **Zarr format mismatch**: Reader determines zarr format auto; only override `--zarr_format` if output format differs from input.
3. **Dask serialization errors**: ProcessPoolExecutor spawn context required; ensure all dask graph objects are pickle-serializable (avoid lambda functions).
4. **Memory explosions with chunking**: Call `autocompute_chunk_shape(array_shape, dtype, target_mb=128)` before writing; defaults assume 128MB chunks.
5. **OME metadata loss**: Use `bioformats` backend (not bioio direct) for complex OME-TIFF; bioio plugins may strip metadata.

---

## Documentation & Testing
- **Docs:** See [docs/](../docs/) folder (mkdocs configuration in mkdocs.yml)
- **Code Review:** [CODE_REVIEW.md](../CODE_REVIEW.md) tracks known issues (exception handling, type hints—mostly completed)
- **Tests:** No test suite yet (TODO in CODE_REVIEW.md); use manual CLI testing or integration workflows in `.github/workflows/`

---

## When Adding Features

1. **New Image Format?** Add reader in [eubi_bridge/core/readers.py](../eubi_bridge/core/readers.py#L20), implement `ImageReader` interface from [eubi_bridge/core/reader_interface.py](../eubi_bridge/core/reader_interface.py)
2. **New Conversion Type?** Subclass `AggregativeConverter`, override `read_dataset()` + `convert_dataset()`
3. **New CLI Command?** Add method to main converter class in [eubi_bridge/ebridge.py](../eubi_bridge/ebridge.py); Fire auto-exposes it
4. **Metadata Changes?** Update [eubi_bridge/ngff/defaults.py](../eubi_bridge/ngff/defaults.py) (axes, units, scales)

