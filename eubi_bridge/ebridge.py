"""EuBI-Bridge: convert bioimage datasets to OME-Zarr format.

Public entry point is :class:`EuBIBridge`, which composes three focused
delegator classes:

* :class:`ConfigManager`    — configuration persistence and Pydantic validation
* :class:`ConversionManager` — conversion pipeline (to_zarr)
* :class:`MetadataManager`  — metadata inspection and update
"""
from __future__ import annotations

import copy
import os
import shutil
import tempfile
import time
import warnings
from pathlib import Path
from typing import Literal, Optional, Union

from rich.console import Console
from rich.table import Table

from eubi_bridge.utils.logging_config import get_logger
from eubi_bridge.core.config_models import (
    AggregativeConversionJob,
    AggregativePlan,
    ClusterConfig,
    ConcatenationConfig,
    ConversionConfig,
    DownscaleConfig,
    ReaderConfig,
)

logger = get_logger(__name__)
_console = Console()

# Table/spreadsheet extensions accepted as batch-input by to_zarr
TABLE_FORMATS = [".csv", ".tsv", ".txt", ".xls", ".xlsx"]


# Heavy imports are deferred — config commands don't need zarr/dask/JVM.
def _ensure_heavy_imports():
    """Lazy-load heavy modules only when needed for actual conversions."""
    global dask, np, psutil, s3fs, zarr, da, AggregativeConverter, run_updates
    if 'dask' in globals():
        return
    import scyjava
    scyjava.config.endpoints.clear()
    scyjava.config.maven_offline = True
    scyjava.config.jgo_disabled = True
    import dask
    import numpy as np
    import psutil
    import s3fs
    import zarr
    from dask import array as da
    warnings.filterwarnings(
        "ignore",
        message="Dask configuration key 'distributed.p2p.disk' has been deprecated",
        category=FutureWarning, module="dask.config",
    )
    warnings.filterwarnings(
        "ignore", message="Could not parse tiff pixel size",
        category=UserWarning, module="bioio_tifffile.reader",
    )
    from eubi_bridge.conversion.aggregative_conversion_base import AggregativeConverter
    from eubi_bridge.conversion.updater import run_updates


# ---------------------------------------------------------------------------
# Module-level rendering helpers (used by MetadataManager)
# ---------------------------------------------------------------------------

def _format_scale_value(value) -> str:
    """Format a physical scale value to 3 significant figures."""
    if isinstance(value, (int, float)):
        if value == 0:
            return "0"
        v = float(value)
        if 0.001 <= abs(v) < 1000:
            return f"{v:.3g}" if v >= 1 else f"{v:.2g}"
        return f"{v:.2e}"
    return str(value)


def _wrap_text(text: str, width: int) -> str:
    """Wrap *text* to *width* characters, preserving existing line breaks."""
    if len(text) <= width:
        return text
    lines = []
    for line in text.split('\n'):
        if len(line) <= width:
            lines.append(line)
        else:
            words, current = line.split(), []
            for word in words:
                if len(' '.join(current + [word])) <= width:
                    current.append(word)
                else:
                    if current:
                        lines.append(' '.join(current))
                    current = [word]
            if current:
                lines.append(' '.join(current))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# ConfigManager
# ---------------------------------------------------------------------------

class ConfigManager:
    """Configuration persistence, access, and Pydantic validation.

    Single source of truth for all cluster, reader, conversion, and downscale
    parameters.  JSON on disk is the canonical store; in-memory dict is a
    lazy-loaded cache.
    """

    _ROOT_DEFAULTS: dict = dict(
        cluster=dict(
            on_local_cluster=False,
            on_slurm=False,
            use_threading=False,
            max_workers=4,
            queue_size=4,
            region_size_mb=256,
            max_concurrency=4,
            max_concurrent_scenes=1,
            memory_per_worker='1GB',
            tensorstore_data_copy_concurrency=4,
            max_retries=10,
            bf_read_concurrency=4,
            bf_tile_size_mb=512.0,
            jvm_memory='2g',
        ),
        readers=dict(
            as_mosaic=False,
            view_index=0,
            phase_index=0,
            illumination_index=0,
            scene_index=0,
            rotation_index=0,
            mosaic_tile_index=0,
            sample_index=0,
            force_bioformats=False,
            concat_views=False,
            concat_illuminations=False,
        ),
        conversion=dict(
            verbose=False,
            zarr_format=2,
            skip_dask=False,
            auto_chunk=True,
            target_chunk_mb=1,
            time_chunk=1,
            channel_chunk=1,
            z_chunk=96,
            y_chunk=96,
            x_chunk=96,
            time_shard_coef=1,
            channel_shard_coef=1,
            z_shard_coef=3,
            y_shard_coef=3,
            x_shard_coef=3,
            time_range=None,
            channel_range=None,
            z_range=None,
            y_range=None,
            x_range=None,
            dimension_order='tczyx',
            compressor='blosc',
            compressor_params={},
            overwrite=False,
            override_channel_names=False,
            channel_intensity_limits='from_dtype',
            metadata_reader='bfio',
            save_omexml=True,
            squeeze=True,
            dtype='auto',
        ),
        downscale=dict(
            time_scale_factor=1,
            channel_scale_factor=1,
            z_scale_factor=2,
            y_scale_factor=2,
            x_scale_factor=2,
            n_layers=None,
            min_dimension_size=64,
            downscale_method='simple',
        ),
        concatenation=dict(
            concatenation_axes=None,
            time_tag=None,
            channel_tag=None,
            z_tag=None,
            y_tag=None,
            x_tag=None,
        ),
    )

    def __init__(self, configpath: str):
        self._configpath = configpath
        self._config: dict | None = None  # lazy-loaded

    # ── path helpers ──────────────────────────────────────────────────────

    def _get_config_dir(self) -> Path:
        """Return the directory that holds all config files."""
        p = Path(self._configpath)
        return p.parent if p.suffix.lower() == '.json' else p

    # ── JSON persistence ──────────────────────────────────────────────────

    def _get_json_path(self) -> Path:
        p = Path(self._configpath)
        return p if p.suffix.lower() == '.json' else p / '.eubi_config.json'

    def _load_config_from_json(self) -> dict | None:
        import json
        path = self._get_json_path()
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def _save_config_to_json(self, config: dict) -> None:
        import json
        path = self._get_json_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
        except IOError as e:
            logger.warning(f"Failed to save config JSON: {e}")

    def _ensure_config_loaded(self) -> None:
        if self._config is not None:
            return
        config = self._load_config_from_json()
        if config is None:
            config = {k: dict(v) for k, v in self._ROOT_DEFAULTS.items()}
            self._save_config_to_json(config)
        else:
            dirty = False
            # Migrate legacy string shuffle values to int
            _shuffle_map = {'noshuffle': 0, 'shuffle': 1, 'bitshuffle': 2, 'autoshuffle': -1}
            cp = config.get('conversion', {}).get('compressor_params', {})
            if isinstance(cp.get('shuffle'), str):
                cp['shuffle'] = _shuffle_map.get(cp['shuffle'].lower(), 1)
                dirty = True
            # Upgrade: add concatenation section if absent (pre-existing config file)
            if 'concatenation' not in config:
                config['concatenation'] = dict(self._ROOT_DEFAULTS['concatenation'])
                dirty = True
            if dirty:
                self._save_config_to_json(config)
        self._config = config

    # ── config property ───────────────────────────────────────────────────

    @property
    def config(self) -> dict:
        self._ensure_config_loaded()
        assert self._config is not None  # guaranteed by _ensure_config_loaded
        return self._config

    @config.setter
    def config(self, value: dict) -> None:
        self._config = value
        self._save_config_to_json(value)

    # ── parameter collection ──────────────────────────────────────────────

    def _collect_params(self, param_type: str, **overrides) -> dict:
        """Return *param_type* config section merged with *overrides*.

        Converts ``dtype='auto'`` → ``None`` at runtime.
        """
        params = {}
        for key in self.config[param_type]:
            params[key] = overrides[key] if key in overrides else self.config[param_type][key]
            if key == 'dtype' and params[key] == 'auto':
                params[key] = None
        return params

    # ── configure_* ───────────────────────────────────────────────────────

    def configure_cluster(self,
                          max_workers: int = 'default',
                          queue_size: int = 'default',
                          region_size_mb: int = 'default',
                          memory_per_worker: str = 'default',
                          max_concurrency: int = 'default',
                          max_concurrent_scenes: int = 'default',
                          on_local_cluster: bool = 'default',
                          on_slurm: bool = 'default',
                          use_threading: bool = 'default',
                          tensorstore_data_copy_concurrency: int = 'default',
                          max_retries: int = 'default',
                          bf_read_concurrency: int = 'default',
                          bf_tile_size_mb: float = 'default',
                          jvm_memory: str = 'default') -> None:
        """Update cluster parameters. Omitted arguments keep their current values."""
        params = {k: v for k, v in locals().items() if k != 'self'}
        for key, val in params.items():
            if key in self.config['cluster'] and val != 'default':
                self.config['cluster'][key] = val
        ClusterConfig(**self.config['cluster'])
        self._save_config_to_json(self.config)

    def configure_readers(self,
                          as_mosaic: bool = 'default',
                          view_index: int = 'default',
                          phase_index: int = 'default',
                          illumination_index: int = 'default',
                          scene_index: int = 'default',
                          rotation_index: int = 'default',
                          mosaic_tile_index: int = 'default',
                          sample_index: int = 'default',
                          force_bioformats: bool = 'default',
                          concat_views: bool = 'default',
                          concat_illuminations: bool = 'default') -> None:
        """Update reader parameters. Omitted arguments keep their current values."""
        params = {k: v for k, v in locals().items() if k != 'self'}
        for key, val in params.items():
            if key in self.config['readers'] and val != 'default':
                self.config['readers'][key] = val
        ReaderConfig(**self.config['readers'])
        self._save_config_to_json(self.config)

    def configure_conversion(self,
                             zarr_format: int = 'default',
                             skip_dask: bool = 'default',
                             auto_chunk: bool = 'default',
                             target_chunk_mb: float = 'default',
                             time_chunk: int = 'default',
                             channel_chunk: int = 'default',
                             z_chunk: int = 'default',
                             y_chunk: int = 'default',
                             x_chunk: int = 'default',
                             time_shard_coef: int = 'default',
                             channel_shard_coef: int = 'default',
                             z_shard_coef: int = 'default',
                             y_shard_coef: int = 'default',
                             x_shard_coef: int = 'default',
                             time_range: int = 'default',
                             channel_range: int = 'default',
                             z_range: int = 'default',
                             y_range: int = 'default',
                             x_range: int = 'default',
                             compressor: str = 'default',
                             compressor_params: dict = 'default',
                             overwrite: bool = 'default',
                             override_channel_names: bool = 'default',
                             channel_intensity_limits: Literal["from_dtype", "from_array", "auto"] = 'default',
                             metadata_reader: str = 'default',
                             save_omexml: bool = 'default',
                             squeeze: bool = 'default',
                             dtype: str = 'default',
                             verbose: bool = 'default') -> None:
        """Update conversion parameters. Omitted arguments keep their current values."""
        params = {k: v for k, v in locals().items() if k != 'self'}
        for key, val in params.items():
            if key in self.config['conversion'] and val != 'default':
                self.config['conversion'][key] = val
        ConversionConfig(**self.config['conversion'])
        self._save_config_to_json(self.config)

    def configure_downscale(self,
                            n_layers: int = 'default',
                            min_dimension_size: int = 'default',
                            time_scale_factor: int = 'default',
                            channel_scale_factor: int = 'default',
                            z_scale_factor: int = 'default',
                            y_scale_factor: int = 'default',
                            x_scale_factor: int = 'default') -> None:
        """Update downscale parameters. Omitted arguments keep their current values."""
        params = {k: v for k, v in locals().items() if k != 'self'}
        for key, val in params.items():
            if key in self.config['downscale'] and val != 'default':
                self.config['downscale'][key] = val
        DownscaleConfig(**self.config['downscale'])
        self._save_config_to_json(self.config)

    def configure_concatenation(self,
                                concatenation_axes: Union[str, int, None] = 'default',
                                time_tag: Union[str, tuple, None] = 'default',
                                channel_tag: Union[str, tuple, None] = 'default',
                                z_tag: Union[str, tuple, None] = 'default',
                                y_tag: Union[str, tuple, None] = 'default',
                                x_tag: Union[str, tuple, None] = 'default') -> None:
        """Update aggregative (concatenation) parameters.

        Set ``concatenation_axes`` to a string of axis letters (e.g. ``'c'``,
        ``'tc'``) to enable aggregative conversion by default.  Each axis
        listed requires a corresponding tag (``time_tag``, ``channel_tag``,
        etc.) to be set as well — either here or at ``to_zarr()`` call time.
        Set ``concatenation_axes=None`` to revert to unary (non-concatenated)
        conversion.
        """
        params = {k: v for k, v in locals().items() if k != 'self'}
        for key, val in params.items():
            if key in self.config['concatenation'] and val != 'default':
                self.config['concatenation'][key] = val
        ConcatenationConfig(**self.config['concatenation'])
        self._save_config_to_json(self.config)

    # ── named configs ─────────────────────────────────────────────────────

    def list_configs(self) -> dict:
        """Return ``{name: path}`` for every named config in the config directory.

        Named configs are any ``*.json`` files in the config directory that are
        *not* the default ``.eubi_config.json``.  The name is the filename stem.
        """
        return {
            p.stem: str(p)
            for p in sorted(self._get_config_dir().glob('*.json'))
            if p.name != '.eubi_config.json'
        }

    def save_as(self, name: str) -> None:
        """Save the current config as a named config file.

        Creates ``<config_dir>/<name>.json``.  Overwrites an existing file with
        the same name.  ``name`` must contain only letters, digits, hyphens, and
        underscores.
        """
        import json
        import re
        if not re.match(r'^[A-Za-z0-9_\-]+$', name):
            raise ValueError(
                f"Config name '{name}' must contain only letters, digits, "
                "hyphens, and underscores."
            )
        dirpath = self._get_config_dir()
        dirpath.mkdir(parents=True, exist_ok=True)
        path = dirpath / f'{name}.json'
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Config saved as '{name}' → {path}")

    def with_config(self, name: str) -> 'ConfigManager':
        """Return a ``ConfigManager`` backed by the named config file.

        The named config provides the defaults for all parameters; any
        call-time overrides still take priority.  Raises ``KeyError`` if
        the name does not exist.
        """
        registry = self.list_configs()
        if name not in registry:
            available = sorted(registry) or ['(none)']
            raise KeyError(
                f"Named config '{name}' not found in {self._get_config_dir()}. "
                f"Available: {available}"
            )
        return ConfigManager(registry[name])

    def update_config(self, name_or_path: str, create: bool = False) -> None:
        """Write the current in-memory config into *name_or_path*.

        *name_or_path* is resolved as follows:

        * **Name** (no path separators, no ``.json`` suffix) — looked up in
          the named-config registry.  If not found and ``create=True`` the
          file ``<config_dir>/<name>.json`` is created.
        * **Path** (contains a separator or ends with ``.json``) — used as a
          literal filesystem path.  Relative paths are resolved from the
          current working directory.

        ``create=False`` (default) raises ``FileNotFoundError`` when the
        target does not exist.  ``create=True`` creates the parent directories
        as well.
        """
        import json as _json
        import re

        raw = name_or_path
        is_path = (os.sep in raw or '/' in raw or raw.endswith('.json'))

        if is_path:
            target = Path(raw) if Path(raw).is_absolute() else Path.cwd() / raw
            if not target.exists() and not create:
                raise FileNotFoundError(
                    f"Config file '{target}' does not exist. "
                    "Pass --create to create it."
                )
        else:
            name = raw
            registry = self.list_configs()
            if name in registry:
                target = Path(registry[name])
            elif create:
                if not re.match(r'^[A-Za-z0-9_\-]+$', name):
                    raise ValueError(
                        f"Config name '{name}' must contain only letters, "
                        "digits, hyphens, and underscores."
                    )
                target = self._get_config_dir() / f'{name}.json'
            else:
                available = sorted(registry) or ['(none)']
                raise FileNotFoundError(
                    f"Named config '{name}' not found. "
                    f"Pass --create to create it.  Available: {available}"
                )

        # Materialize config before opening the target for writing.
        # open(target, 'w') truncates the file immediately; accessing self.config
        # inside the with-block would trigger a lazy JSON read on an empty file,
        # causing JSONDecodeError → silent fallback to _ROOT_DEFAULTS.
        config_snapshot = self.config
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, 'w') as f:
            _json.dump(config_snapshot, f, indent=2)
        logger.info(f"Config written → {target}")

    def delete_config(self, name: str) -> None:
        """Delete the named config file.  Raises ``KeyError`` if not found."""
        registry = self.list_configs()
        if name not in registry:
            raise KeyError(f"Named config '{name}' not found.")
        Path(registry[name]).unlink()
        logger.info(f"Named config '{name}' deleted.")

    # ── display / reset ───────────────────────────────────────────────────

    def reset_config(self) -> None:
        """Reset all parameters to installation defaults."""
        self.config = copy.deepcopy(self._ROOT_DEFAULTS)

    def show_config(self, name: str = None) -> None:
        """Display configuration as a rich table.

        If *name* is given, display the named config file.
        If omitted, display the default (``.eubi_config.json``) config.
        """
        import json as _json
        if name is not None:
            registry = self.list_configs()
            if name not in registry:
                available = sorted(registry) or ['(none)']
                raise KeyError(
                    f"Named config '{name}' not found. Available: {available}"
                )
            with open(registry[name]) as f:
                config = _json.load(f)
            display_name = name
        else:
            config = self._load_config_from_json() or self.config
            # Auto-detect when called via `with_config(name).show_config()`:
            # if this manager is backed by a named file (not .eubi_config.json),
            # surface the name so the title is accurate.
            json_path = self._get_json_path()
            display_name = (None if json_path.name == '.eubi_config.json'
                            else json_path.stem)

        if display_name is not None:
            title       = f"Config: {display_name}"
            header_text = f"[bold cyan]Config: {display_name}[/bold cyan]"
        else:
            title       = "Current Configuration"
            header_text = "[bold cyan]Current Configuration[/bold cyan]"

        _console.print("\n[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
        _console.print(header_text)
        _console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]\n")
        sections = {k: v for k, v in config.items()
                    if k in ('cluster', 'readers', 'conversion', 'downscale', 'concatenation')}
        self._print_config_unified(title, sections)
        _console.print()

    def show_root_defaults(self) -> None:
        """Display installation defaults as a rich table."""
        _console.print("\n[bold yellow]═══════════════════════════════════════════════════════[/bold yellow]")
        _console.print("[bold yellow]Installation Defaults[/bold yellow]")
        _console.print("[bold yellow]═══════════════════════════════════════════════════════[/bold yellow]\n")
        sections = {k: v for k, v in self._ROOT_DEFAULTS.items()
                    if k in ('cluster', 'readers', 'conversion', 'downscale', 'concatenation')}
        self._print_config_unified("Defaults", sections)
        _console.print()

    def show_configs(self) -> None:
        """Display all named configs found in the config directory."""
        configs = self.list_configs()
        _console.print("\n[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
        _console.print(f"[bold cyan]Named Configs  ({self._get_config_dir()})[/bold cyan]")
        _console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]\n")
        if not configs:
            _console.print("[dim italic]  No named configs found.  "
                           "Use save_as('<name>') to create one.[/dim italic]\n")
            return
        table = Table(
            show_header=True, header_style="bold white on blue",
            padding=(0, 1), border_style="blue",
        )
        table.add_column("Name", style="cyan", width=20)
        table.add_column("Path", style="green")
        for name, path in configs.items():
            table.add_row(name, path)
        _console.print(table)
        _console.print()

    def _print_config_unified(self, title: str, config_sections: dict) -> None:
        table = Table(
            title=title, title_style="bold cyan", show_header=True,
            header_style="bold white on blue", padding=(0, 1), border_style="blue",
        )
        table.add_column("Section",   style="magenta", width=15)
        table.add_column("Parameter", style="cyan",    width=25)
        table.add_column("Value",     style="green")
        section_list = list(config_sections.items())
        for idx, (section_name, section_dict) in enumerate(section_list):
            is_first = True
            for key, value in sorted(section_dict.items()):
                if isinstance(value, bool):
                    val_str = "[bold green]True[/]" if value else "[bold red]False[/]"
                elif isinstance(value, dict):
                    val_str = "[dim]" + str(value) + "[/dim]"
                elif value is None:
                    val_str = "[dim italic]None[/]"
                else:
                    val_str = str(value)
                section_display = f"[bold magenta]{section_name.upper()}[/]" if is_first else ""
                table.add_row(section_display, key, val_str)
                is_first = False
            if idx < len(section_list) - 1:
                table.add_section()
        _console.print(table)


# ---------------------------------------------------------------------------
# ConversionManager
# ---------------------------------------------------------------------------

class ConversionManager:
    """Owns the conversion pipeline."""

    def __init__(self, config: ConfigManager):
        self._config = config

    def to_zarr(self,
                input_path,
                output_path=None,
                includes=None,
                excludes=None,
                time_tag: Union[str, tuple] = None,
                channel_tag: Union[str, tuple] = None,
                z_tag: Union[str, tuple] = None,
                y_tag: Union[str, tuple] = None,
                x_tag: Union[str, tuple] = None,
                concatenation_axes: Union[int, tuple, str] = None,
                max_workers: int = None,
                max_retries: int = None,
                tensorstore_data_copy_concurrency: int = None,
                use_threading: bool = None,
                on_slurm: bool = None,
                on_local_cluster: bool = None,
                plan: 'AggregativePlan | None' = None,
                **kwargs) -> None:
        """Convert image data to OME-Zarr format."""
        _ensure_heavy_imports()
        from eubi_bridge.utils.jvm_manager import soft_start_jvm
        soft_start_jvm()

        t0 = time.time()
        logger.info("Conversion starting.")
        if output_path is None:
            assert input_path.endswith(('.csv', '.tsv', '.txt', '.xlsx'))

        # Collect CLI overrides (explicit non-None args take priority)
        cli_kwargs = {k: v for k, v in dict(
            max_workers=max_workers,
            max_retries=max_retries,
            tensorstore_data_copy_concurrency=tensorstore_data_copy_concurrency,
            use_threading=use_threading,
            on_slurm=on_slurm,
            on_local_cluster=on_local_cluster,
        ).items() if v is not None}
        cli_kwargs.update(kwargs)

        # Build effective params (config defaults merged with CLI overrides).
        # Pydantic validates each section — catches bad CLI values early (Path 1).
        cluster_p    = self._config._collect_params('cluster',    **cli_kwargs)
        readers_p    = self._config._collect_params('readers',    **cli_kwargs)
        conversion_p = self._config._collect_params('conversion', **cli_kwargs)
        downscale_p  = self._config._collect_params('downscale',  **cli_kwargs)
        ClusterConfig(**cluster_p)
        ReaderConfig(**readers_p)
        ConversionConfig(**conversion_p)
        DownscaleConfig(**downscale_p)

        # Resolve concatenation params: call-time args take priority over config.
        cli_concat = {k: v for k, v in dict(
            concatenation_axes=concatenation_axes,
            time_tag=time_tag, channel_tag=channel_tag,
            z_tag=z_tag, y_tag=y_tag, x_tag=x_tag,
        ).items() if v is not None}
        concat_p = self._config._collect_params('concatenation', **cli_concat)
        ConcatenationConfig(**concat_p)

        effective_concatenation_axes = concat_p['concatenation_axes']
        effective_time_tag    = concat_p['time_tag']
        effective_channel_tag = concat_p['channel_tag']
        effective_z_tag       = concat_p['z_tag']
        effective_y_tag       = concat_p['y_tag']
        effective_x_tag       = concat_p['x_tag']

        merged = {**cluster_p, **readers_p, **conversion_p, **downscale_p}
        extra  = {k: v for k, v in kwargs.items() if k not in merged}
        if isinstance(input_path, tuple):
            _input: Union[str, list] = list(input_path)
        elif isinstance(input_path, list):
            _input = input_path
        else:
            _input = os.path.abspath(input_path)

        if effective_concatenation_axes is not None:
            from eubi_bridge.conversion.dispatcher import dispatch_aggregative_job
            job = AggregativeConversionJob(
                input_path=_input,
                output_path=str(output_path) if output_path is not None else "",
                cluster=ClusterConfig(**cluster_p),
                readers=ReaderConfig(**readers_p),
                conversion=ConversionConfig(**conversion_p),
                downscale=DownscaleConfig(**downscale_p),
                concatenation_axes=effective_concatenation_axes,
                time_tag=effective_time_tag,
                channel_tag=effective_channel_tag,
                z_tag=effective_z_tag,
                y_tag=effective_y_tag,
                x_tag=effective_x_tag,
                includes=includes,
                excludes=excludes,
                extra=extra,
            )
            dispatch_aggregative_job(job, plan=plan)
        else:
            # Unary path — build one validated ConversionJob per file here in
            # ebridge.py so the dispatcher receives already-validated objects.
            # CSV row params (Stage 2 triage) are validated here too (Path 2).
            from eubi_bridge.utils.path_utils import take_filepaths
            from eubi_bridge.core.config_models import ConversionJob

            df = take_filepaths(
                _input, output_path=output_path,
                includes=includes, excludes=excludes,
                **merged, **extra,
            )
            jobs = []
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                ip  = row_dict.pop('input_path')
                op  = row_dict.pop('output_path', None) or output_path
                # Drop NaN/None CSV cells so they don't shadow global values
                row_overrides = {k: v for k, v in row_dict.items()
                                 if v is not None and v == v}  # v==v filters NaN
                per_job = {**merged, **extra, **row_overrides}
                jobs.append(ConversionJob.from_kwargs(ip, op, per_job))

            from eubi_bridge.conversion.dispatcher import dispatch_unary_jobs
            dispatch_unary_jobs(jobs)

        logger.info("Conversion complete for all datasets.")
        logger.info(f"Elapsed for conversion + downscaling: {(time.time() - t0) / 60:.2f} min.")

    def validate_aggregative(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path, None] = None,
        includes: Optional[str] = None,
        excludes: Optional[str] = None,
        time_tag: Union[str, tuple, None] = None,
        channel_tag: Union[str, tuple, None] = None,
        z_tag: Union[str, tuple, None] = None,
        y_tag: Union[str, tuple, None] = None,
        x_tag: Union[str, tuple, None] = None,
        concatenation_axes: Union[str, int, None] = None,
        **kwargs,
    ) -> AggregativePlan:
        """Dry-run an aggregative conversion and return a resource plan.

        Determines output groups via pure filename matching (no data loaded),
        then reads one file header per group to estimate shapes and sizes.
        The returned :class:`AggregativePlan` can be passed to
        :meth:`to_zarr` to skip re-planning and apply the pre-computed
        worker allocation.

        Args:
            (same parameters as :meth:`to_zarr` with ``concatenation_axes``)

        Returns:
            :class:`AggregativePlan` with per-output metadata and
            recommended ``file_workers`` / ``region_workers``.
        """
        from eubi_bridge.utils.path_utils import take_filepaths
        from eubi_bridge.conversion.dispatcher import (
            _parse_filepaths_with_tags, _build_aggregative_plan,
        )

        cli_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        cluster_p    = self._config._collect_params('cluster',    **cli_kwargs)
        readers_p    = self._config._collect_params('readers',    **cli_kwargs)
        conversion_p = self._config._collect_params('conversion', **cli_kwargs)
        downscale_p  = self._config._collect_params('downscale',  **cli_kwargs)
        merged = {**cluster_p, **readers_p, **conversion_p, **downscale_p}
        extra  = {k: v for k, v in kwargs.items() if k not in merged}

        cli_concat = {k: v for k, v in dict(
            concatenation_axes=concatenation_axes,
            time_tag=time_tag, channel_tag=channel_tag,
            z_tag=z_tag, y_tag=y_tag, x_tag=x_tag,
        ).items() if v is not None}
        concat_p = self._config._collect_params('concatenation', **cli_concat)

        _input = (input_path if isinstance(input_path, (list, tuple))
                  else os.path.abspath(input_path))

        df = take_filepaths(
            _input, output_path=output_path,
            includes=includes, excludes=excludes,
            **merged, **extra,
        )
        filepaths = df.input_path.to_numpy().tolist()

        eff_tags = [concat_p['time_tag'], concat_p['channel_tag'],
                    concat_p['z_tag'], concat_p['y_tag'], concat_p['x_tag']]
        filepaths_accepted = _parse_filepaths_with_tags(filepaths, eff_tags)

        common_dir  = os.path.commonpath(filepaths_accepted)
        output_base = output_path or common_dir
        max_workers = cluster_p.get('max_workers', 4)

        return _build_aggregative_plan(
            filepaths_accepted,
            concatenation_axes=concat_p['concatenation_axes'],
            common_dir=common_dir,
            output_path=output_base,
            max_workers=max_workers,
            time_tag=concat_p['time_tag'],
            channel_tag=concat_p['channel_tag'],
            z_tag=concat_p['z_tag'],
            y_tag=concat_p['y_tag'],
            x_tag=concat_p['x_tag'],
        )


# ---------------------------------------------------------------------------
# MetadataManager
# ---------------------------------------------------------------------------

class MetadataManager:
    """Metadata inspection and update for existing OME-Zarr files."""

    def __init__(self, config: ConfigManager):
        self._config = config

    # ── shared rendering helper ───────────────────────────────────────────

    @staticmethod
    def _render_metadata(metadata_list: list, console: Console) -> None:
        """Render *metadata_list* to *console* (any Rich Console instance)."""
        _ax = {'t': 'time', 'c': 'channels', 'z': 'z', 'y': 'y', 'x': 'x'}
        console.print("\n[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
        console.print("[bold cyan]Image Metadata Summary[/bold cyan]")
        console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]\n")
        for idx, metadata in enumerate(metadata_list):
            if idx > 0:
                console.print("[cyan]" + "─" * 100 + "[/cyan]")
            if metadata.get('status') == 'error':
                console.print(f"[red bold]File:[/red bold] {Path(metadata.get('input_path', 'Unknown')).name}")
                console.print(f"[red]ERROR:[/red] {metadata.get('error', 'Unknown error')}\n")
                continue
            axes     = metadata.get('axes', 'Unknown')
            shape    = metadata.get('shape', {})
            scale    = metadata.get('scale', {})
            units    = metadata.get('units', {})
            dtype    = metadata.get('dtype', 'Unknown')
            channels = metadata.get('channels', [])

            console.print(f"[magenta bold]File:[/magenta bold] {_wrap_text(metadata['input_path'], 90)}")
            console.print(
                "[yellow bold]Shape:[/yellow bold]\n" +
                '\n'.join(f"  {_ax.get(ax, ax)}: {shape.get(ax, '?')}" for ax in axes)
            )
            scale_lines = []
            for ax in axes:
                if ax == 'c':
                    continue
                sv, uv = scale.get(ax, '?'), units.get(ax, '')
                label = _ax.get(ax, ax)
                scale_lines.append(
                    f"  {label}: {_format_scale_value(sv)} {uv}".rstrip() if sv != '?' else f"  {label}: ?"
                )
            console.print("[green bold]Scale & Units:[/green bold]\n" + '\n'.join(scale_lines))
            console.print(f"[white bold]Data Type:[/white bold] {dtype}")
            if channels:
                console.print("[bright_magenta bold]Channels:[/bright_magenta bold]")
                for i, ch in enumerate(channels):
                    lbl   = ch.get('label', f"Channel {i}")
                    color = ch.get('color') or "None"
                    console.print(f"  [{i}] Label: {_wrap_text(lbl, 80)}, Color: {color}")
            else:
                console.print("[bright_magenta bold]Channels:[/bright_magenta bold] None")
            console.print()
        console.print("[cyan]" + "═" * 100 + "[/cyan]")
        console.print()

    # ── public methods ────────────────────────────────────────────────────

    def show_pixel_meta(self,
                        input_path: Union[Path, str],
                        includes=None,
                        excludes=None,
                        series: int = None,
                        output_file: str = None,
                        **kwargs) -> None:
        """Display pixel-level and channel metadata for all datasets in *input_path*.

        Args:
            input_path: Path to input file or directory.
            includes: Filename patterns to include (comma-separated).
            excludes: Filename patterns to exclude (comma-separated).
            series: Series index. Defaults to configured scene_index.
            output_file: Save output as plain text (any ext) or HTML (.html).
        """
        import asyncio
        from eubi_bridge.utils.path_utils import take_filepaths

        df = None
        try:
            df = take_filepaths(input_path, **kwargs)
            all_zarr   = all(p.endswith('.zarr') for p in df['input_path'])
            zarr_paths = df['input_path'].tolist() if all_zarr else None
        except (ValueError, KeyError):
            all_zarr, zarr_paths = False, None

        if all_zarr:
            logger.info(f"Fast path: reading metadata from {len(zarr_paths)} OME-Zarr files.")
            from eubi_bridge.utils.metadata_utils import read_ome_zarr_metadata_from_collection
            metadata_list = asyncio.run(read_ome_zarr_metadata_from_collection(zarr_paths))
        else:
            logger.info("Non-Zarr files detected — initializing JVM.")
            _ensure_heavy_imports()
            from eubi_bridge.utils.jvm_manager import soft_start_jvm
            soft_start_jvm()
            cluster_p    = self._config._collect_params('cluster',    **kwargs)
            readers_p    = self._config._collect_params('readers',    **kwargs)
            conversion_p = self._config._collect_params('conversion', **kwargs)
            if series is None:
                series = readers_p['scene_index']
            combined = {
                **cluster_p, **readers_p, **conversion_p,
                'series': series,
                'use_threading': True,
                'max_workers': min(16, max(8, len(df['input_path']) if df is not None else 8)),
            }
            extra = {k: v for k, v in kwargs.items() if k not in combined}
            from eubi_bridge.conversion.dispatcher import run_metadata_collection_from_filepaths
            metadata_list = asyncio.run(run_metadata_collection_from_filepaths(
                input_path, includes=includes, excludes=excludes, **combined, **extra,
            ))

        # Render to stdout always
        self._render_metadata(metadata_list, _console)

        if output_file:
            if output_file.lower().endswith('.html'):
                html_con = Console(record=True, width=100, force_terminal=True, legacy_windows=False)
                self._render_metadata(metadata_list, html_con)
                with open(output_file, 'w') as f:
                    f.write(html_con.export_html())
                logger.info(f"Metadata saved to {output_file} (HTML)")
            else:
                with open(output_file, 'w') as f:
                    self._render_metadata(
                        metadata_list,
                        Console(file=f, width=100, force_terminal=True, legacy_windows=False),
                    )
                logger.info(f"Metadata saved to {output_file} (text)")

    def update_pixel_meta(self,
                          input_path: Union[Path, str],
                          includes=None,
                          excludes=None,
                          time_scale: Union[int, float, None] = None,
                          z_scale: Union[int, float, None] = None,
                          y_scale: Union[int, float, None] = None,
                          x_scale: Union[int, float, None] = None,
                          time_unit: Union[str, None] = None,
                          z_unit: Union[str, None] = None,
                          y_unit: Union[str, None] = None,
                          x_unit: Union[str, None] = None,
                          **kwargs) -> None:
        """Update pixel-scale and unit metadata on existing OME-Zarr files."""
        from eubi_bridge.conversion.updater import run_updates
        combined = {
            **self._config._collect_params('cluster',    **kwargs),
            **self._config._collect_params('readers',    **kwargs),
            **self._config._collect_params('conversion', **kwargs),
        }
        combined['channel_intensity_limits'] = 'auto'
        extra = {k: v for k, v in kwargs.items() if k not in combined}
        pixel_meta = {k: v for k, v in dict(
            time_scale=time_scale, z_scale=z_scale, y_scale=y_scale, x_scale=x_scale,
            time_unit=time_unit,   z_unit=z_unit,   y_unit=y_unit,   x_unit=x_unit,
        ).items() if v is not None}
        run_updates(input_path, includes=includes, excludes=excludes,
                    **combined, **pixel_meta, **extra)

    def update_channel_meta(self,
                            input_path: Union[Path, str],
                            channel_labels: str = '',
                            channel_colors: str = '',
                            channel_intensity_limits='from_dtype',
                            includes=None,
                            excludes=None,
                            **kwargs) -> None:
        """Update channel label, color, and intensity metadata on existing OME-Zarr files."""
        from eubi_bridge.conversion.updater import run_updates
        combined = {
            **self._config._collect_params('cluster',    **kwargs),
            **self._config._collect_params('readers',    **kwargs),
            **self._config._collect_params(
                'conversion', channel_intensity_limits=channel_intensity_limits, **kwargs),
        }
        extra = {k: v for k, v in kwargs.items() if k not in combined}
        channel_meta = {k: v for k, v in dict(
            channel_labels=channel_labels, channel_colors=channel_colors,
        ).items() if v}
        run_updates(os.path.abspath(input_path),
                    includes=includes, excludes=excludes,
                    **combined, **channel_meta, **extra)


# ---------------------------------------------------------------------------
# EuBIBridge — thin shell
# ---------------------------------------------------------------------------

class EuBIBridge:
    """Entry point for EuBI-Bridge.

    Composes :class:`ConfigManager`, :class:`ConversionManager`, and
    :class:`MetadataManager`.  All public methods delegate to the appropriate
    manager; signatures are preserved so the Fire CLI and existing call sites
    continue to work unchanged.
    """

    def __init__(self, configpath: str = f"{os.path.expanduser('~')}/.eubi_bridge"):
        self._cfg      = ConfigManager(configpath)
        self._conv     = ConversionManager(self._cfg)
        self._meta     = MetadataManager(self._cfg)
        self._dask_temp_dir = None

    # ── backward-compat attribute ─────────────────────────────────────────

    @property
    def root_defaults(self) -> dict:
        return self._cfg._ROOT_DEFAULTS

    # ── config pass-throughs ──────────────────────────────────────────────

    @property
    def config(self) -> dict:
        return self._cfg.config

    @config.setter
    def config(self, value: dict) -> None:
        self._cfg.config = value

    def configure_cluster(self,
                          max_workers: int = 'default',
                          queue_size: int = 'default',
                          region_size_mb: int = 'default',
                          memory_per_worker: str = 'default',
                          max_concurrency: int = 'default',
                          max_concurrent_scenes: int = 'default',
                          on_local_cluster: bool = 'default',
                          on_slurm: bool = 'default',
                          use_threading: bool = 'default',
                          tensorstore_data_copy_concurrency: int = 'default',
                          max_retries: int = 'default',
                          bf_read_concurrency: int = 'default',
                          bf_tile_size_mb: float = 'default',
                          jvm_memory: str = 'default') -> None:
        """Update cluster / concurrency parameters. Omitted arguments keep their current values.

        Args:
            max_workers: Number of parallel file-level worker processes (default 4).
            queue_size: Internal write-queue depth per worker (default 4).
            region_size_mb: Region size in MB for spatial partitioning (default 256).
            memory_per_worker: Memory limit accepted by SLURM / LocalCluster,
                e.g. ``'8GB'`` (default ``'1GB'``).
            max_concurrency: TensorStore write concurrency per worker (default 4).
            max_concurrent_scenes: Parallel scenes within one file (default 1).
                Increase only when writing a multi-scene file to a large store.
            on_local_cluster: Use a Dask LocalCluster backend (default False).
            on_slurm: Submit jobs to a SLURM cluster (default False).
            use_threading: Use ThreadPool instead of ProcessPool (default False).
            tensorstore_data_copy_concurrency: TensorStore internal copy threads
                (default 4).
            max_retries: Retries on broken worker process (default 10).
            bf_read_concurrency: Dask thread count for parallel bfio tile reads
                (default 4).  ``None`` lets dask choose (cpu_count).
            bf_tile_size_mb: Tile size budget in MB for bfio tiled reading
                (default 512).
            jvm_memory: Maximum JVM heap for Bio-Formats, e.g. ``'8GB'``, ``'4GB'``.
                Accepts ``'NGB'`` / ``'NMB'`` (like memory_per_worker) and normalises
                internally to JVM format (``'Ng'`` / ``'Nm'``).  Default ``'2g'``.
        """
        return self._cfg.configure_cluster(**{k: v for k, v in locals().items() if k != 'self'})

    def configure_readers(self,
                          as_mosaic: bool = 'default',
                          view_index: int = 'default',
                          phase_index: int = 'default',
                          illumination_index: int = 'default',
                          scene_index: int = 'default',
                          rotation_index: int = 'default',
                          mosaic_tile_index: int = 'default',
                          sample_index: int = 'default',
                          force_bioformats: bool = 'default',
                          concat_views: bool = 'default',
                          concat_illuminations: bool = 'default') -> None:
        """Update file-reader parameters. Omitted arguments keep their current values.

        Args:
            as_mosaic: Treat tiled acquisitions as a stitched mosaic (default False).
            view_index: View index for multi-view formats (default 0). Pass ``'all'``
                or a comma-separated list (e.g. ``'0,2'``) to expose multiple views.
            phase_index: Phase index (default 0).
            illumination_index: Illumination index (default 0). Pass ``'all'`` or a
                comma-separated list to expose multiple illuminations.
            scene_index: Scene / series index to read.  Pass an integer, ``'all'``,
                or a comma-separated list such as ``'0,2,4'`` (default 0).
            rotation_index: Rotation index (default 0).
            mosaic_tile_index: Mosaic tile index.  Pass an integer, ``'all'``,
                or a comma-separated list (default None = all tiles).
            sample_index: Sample index (default 0).
            force_bioformats: Force bfio tiled path even for natively-supported formats.
            concat_views: Stack multiple views along the channel axis instead of
                writing separate OME-Zarr outputs (default False).
            concat_illuminations: Stack multiple illuminations along the channel axis
                instead of writing separate OME-Zarr outputs (default False).
        """
        return self._cfg.configure_readers(**{k: v for k, v in locals().items() if k != 'self'})

    def configure_conversion(self,
                             zarr_format: int = 'default',
                             skip_dask: bool = 'default',
                             auto_chunk: bool = 'default',
                             target_chunk_mb: float = 'default',
                             time_chunk: int = 'default',
                             channel_chunk: int = 'default',
                             z_chunk: int = 'default',
                             y_chunk: int = 'default',
                             x_chunk: int = 'default',
                             time_shard_coef: int = 'default',
                             channel_shard_coef: int = 'default',
                             z_shard_coef: int = 'default',
                             y_shard_coef: int = 'default',
                             x_shard_coef: int = 'default',
                             time_range: int = 'default',
                             channel_range: int = 'default',
                             z_range: int = 'default',
                             y_range: int = 'default',
                             x_range: int = 'default',
                             compressor: str = 'default',
                             compressor_params: dict = 'default',
                             overwrite: bool = 'default',
                             override_channel_names: bool = 'default',
                             channel_intensity_limits: Literal["from_dtype", "from_array", "auto"] = 'default',
                             metadata_reader: str = 'default',
                             save_omexml: bool = 'default',
                             squeeze: bool = 'default',
                             dtype: str = 'default',
                             verbose: bool = 'default') -> None:
        """Update Zarr conversion parameters. Omitted arguments keep their current values.

        Args:
            zarr_format: Zarr version — ``2`` (default) or ``3``.
            auto_chunk: Auto-compute chunk shape from array size (default True).
            target_chunk_mb: Target chunk size in MB when auto_chunk is True.
            time_chunk / channel_chunk / z_chunk / y_chunk / x_chunk:
                Manual per-axis chunk sizes (used when auto_chunk is False).
            time_shard_coef / channel_shard_coef / z_shard_coef / y_shard_coef / x_shard_coef:
                Shard-to-chunk multipliers for Zarr v3 sharding (default 3 on spatial axes).
            time_range / channel_range / z_range / y_range / x_range:
                Crop ranges as ``"start,stop"`` strings applied before writing.
            compressor: Compression codec — ``'blosc'`` (default), ``'gzip'``,
                ``'zstd'``, ``'bz2'``, or ``'none'``.
            compressor_params: Dict of codec parameters, e.g.
                ``{'cname': 'lz4', 'clevel': 5}``.
            overwrite: Overwrite existing output zarr (default False).
            dtype: Output dtype — ``'auto'`` keeps the source dtype, or any
                NumPy dtype string such as ``'uint16'``.
            channel_intensity_limits: How to set OMERO window limits —
                ``'from_dtype'`` (default) uses dtype min/max,
                ``'from_array'`` computes per-channel min/max from pixel data,
                ``'auto'`` lets the viewer decide.
            squeeze: Remove singleton dimensions before writing (default True).
            metadata_reader: Metadata backend — ``'bfio'`` (default) or
                ``'bioformats'``.
            save_omexml: Write a companion OME-XML file alongside the zarr
                (default True).
            verbose: Log verbose progress output (default False).
        """
        return self._cfg.configure_conversion(**{k: v for k, v in locals().items() if k != 'self'})

    def configure_downscale(self,
                            n_layers: int = 'default',
                            min_dimension_size: int = 'default',
                            time_scale_factor: int = 'default',
                            channel_scale_factor: int = 'default',
                            z_scale_factor: int = 'default',
                            y_scale_factor: int = 'default',
                            x_scale_factor: int = 'default') -> None:
        """Update downscale parameters. Omitted arguments keep their current values.

        Args:
            n_layers: Number of downscale pyramid levels to generate (default 5).
            min_dimension_size: Stop adding levels when the smallest spatial
                dimension falls below this pixel count (default 64).
            time_scale_factor: Downscale factor along the time axis (default 1,
                i.e. no downscaling).
            channel_scale_factor: Downscale factor along the channel axis (default 1).
            z_scale_factor: Downscale factor along the z axis (default 2).
            y_scale_factor: Downscale factor along the y axis (default 2).
            x_scale_factor: Downscale factor along the x axis (default 2).
        """
        return self._cfg.configure_downscale(**{k: v for k, v in locals().items() if k != 'self'})

    def configure_concatenation(self,
                                concatenation_axes: Union[str, int, None] = 'default',
                                time_tag: Union[str, tuple, None] = 'default',
                                channel_tag: Union[str, tuple, None] = 'default',
                                z_tag: Union[str, tuple, None] = 'default',
                                y_tag: Union[str, tuple, None] = 'default',
                                x_tag: Union[str, tuple, None] = 'default') -> None:
        """Update aggregative (concatenation) parameters. Omitted arguments keep their current values.

        Args:
            concatenation_axes: Axes along which to concatenate files.  Pass a
                string of axis letters such as ``'tc'`` or ``'z'``, or an integer
                axis index.  ``None`` disables aggregative mode entirely.
            time_tag: Filename substring (or tuple of substrings) that identifies
                a file as contributing to the time axis.
            channel_tag: Filename substring (or tuple of substrings) identifying
                the channel axis.
            z_tag: Filename substring (or tuple of substrings) identifying the z
                axis.
            y_tag: Filename substring (or tuple of substrings) identifying the y
                axis.
            x_tag: Filename substring (or tuple of substrings) identifying the x
                axis.
        """
        return self._cfg.configure_concatenation(**{k: v for k, v in locals().items() if k != 'self'})

    # ── named-config pass-throughs ────────────────────────────────────────

    def with_config(self, name: str) -> 'EuBIBridge':
        """Return a new EuBIBridge backed by the named config file.

        All parameters in the named config are used as defaults; any
        arguments passed directly to ``to_zarr()`` (or other commands) still
        take priority.  Designed for Fire chaining::

            eubi with_config hpc to_zarr input/ output/
        """
        new_cfg = self._cfg.with_config(name)
        new_bridge = EuBIBridge.__new__(EuBIBridge)
        new_bridge._cfg  = new_cfg
        new_bridge._conv = ConversionManager(new_cfg)
        new_bridge._meta = MetadataManager(new_cfg)
        new_bridge._dask_temp_dir = None
        return new_bridge

    def list_configs(self) -> dict:
        """Return ``{name: path}`` for all named configs in the config directory."""
        return self._cfg.list_configs()

    def save_as(self, name: str) -> None:
        """Save the current config as a named config file."""
        return self._cfg.save_as(name)

    def update_config(self, name_or_path: str, create: bool = False) -> None:
        """Copy the current config into a named config or an explicit file path.

        Typical use: propagate default-config edits to a named config::

            eubi configure_cluster --max_workers 64   # edits .eubi_config.json
            eubi update_config hpc                     # copies it to hpc.json

        Note: ``with_config hpc configure_cluster ...`` already writes directly
        to hpc.json, so ``update_config`` is not needed in that flow.

        Pass ``--create`` to allow creating the target if it does not exist.
        """
        return self._cfg.update_config(name_or_path, create=create)

    def delete_config(self, name: str) -> None:
        """Permanently delete a named config file from the config directory.

        The active (default) config is not affected.  Only named configs
        created with ``save_as`` or ``update_config --create`` can be deleted
        this way.

        Args:
            name: Short name of the config to delete (without the ``.json``
                extension), e.g. ``hpc``.
        """
        return self._cfg.delete_config(name)

    def show_configs(self) -> None:
        """List all named config files found in the config directory.

        Prints a table of ``{name: path}`` entries.  Use ``with_config NAME``
        to activate one of them for a single command, or ``save_as`` /
        ``update_config`` to manage them.
        """
        return self._cfg.show_configs()

    def _get_json_path(self) -> Path:
        """Return the resolved path to the JSON config file."""
        return self._cfg._get_json_path()

    def reset_config(self) -> None:
        """Reset all parameters in the active config to installation defaults.

        Overwrites the active ``.eubi_config.json`` with the bundled root
        defaults.  Named configs (``hpc.json``, etc.) are not affected.
        Use ``show_root_defaults`` to inspect the values that will be written.
        """
        return self._cfg.reset_config()

    def show_config(self, name: str = None) -> None:
        """Display the active configuration, or a specific named config.

        With no arguments, prints the current ``.eubi_config.json``.
        Pass a name to inspect a saved config without activating it.

        Args:
            name: Named config to display (e.g. ``hpc``).  If omitted,
                the active (default) config is shown.
        """
        return self._cfg.show_config(name=name)

    def show_root_defaults(self) -> None:
        """Display the installation defaults for all configuration parameters.

        These are the values that ``reset_config`` will restore.  Useful as a
        reference when tuning cluster or conversion settings.
        """
        return self._cfg.show_root_defaults()

    # ── conversion pass-throughs ─────────────────────────────────────────

    def to_zarr(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path, None] = None,
        includes: Optional[str] = None,
        excludes: Optional[str] = None,
        time_tag: Union[str, tuple, None] = None,
        channel_tag: Union[str, tuple] = None,
        z_tag: Union[str, tuple] = None,
        y_tag: Union[str, tuple] = None,
        x_tag: Union[str, tuple] = None,
        concatenation_axes: Union[int, tuple, str] = None,
        max_workers: int = None,
        max_retries: int = None,
        tensorstore_data_copy_concurrency: int = None,
        use_threading: bool = None,
        on_slurm: bool = None,
        on_local_cluster: bool = None,
        plan: 'AggregativePlan | None' = None,
        **kwargs,
    ) -> None:
        """Convert image data to OME-Zarr format.

        Args:
            input_path: File path, directory, or table (.csv/.tsv/.xlsx) of paths.
            output_path: Destination directory.  Required unless input_path is a
                table that already contains an output_path column.
            includes: Comma-separated filename patterns to include (glob-style).
            excludes: Comma-separated filename patterns to exclude (glob-style).
            time_tag: Filename tag identifying the time axis (aggregative only).
            channel_tag: Filename tag identifying the channel axis.
            z_tag: Filename tag identifying the z axis.
            y_tag: Filename tag identifying the y axis.
            x_tag: Filename tag identifying the x axis.
            concatenation_axes: Axes to concatenate across files, e.g. ``'tc'``.
                If omitted the config value is used; ``None`` means unary conversion.
            max_workers: Override the configured max_workers for this run.
            max_retries: Override the configured max_retries.
            tensorstore_data_copy_concurrency: TensorStore internal copy threads.
            use_threading: Use a ThreadPool instead of a ProcessPool.
            on_slurm: Submit jobs to a SLURM cluster.
            on_local_cluster: Use a Dask LocalCluster.
            plan: Pre-computed AggregativePlan from validate_aggregative().
            **kwargs: Any ReaderConfig / ConversionConfig / DownscaleConfig field
                overrides (e.g. ``zarr_format=3``, ``z_chunk=64``,
                ``bf_tile_size_mb=1024``, ``jvm_memory='8GB'``,
                ``force_bioformats=True``).
        """
        return self._conv.to_zarr(
            input_path, output_path, includes, excludes,
            time_tag, channel_tag, z_tag, y_tag, x_tag,
            concatenation_axes, max_workers, max_retries,
            tensorstore_data_copy_concurrency, use_threading,
            on_slurm, on_local_cluster, plan, **kwargs,
        )

    def validate_aggregative(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path, None] = None,
        includes: Optional[str] = None,
        excludes: Optional[str] = None,
        time_tag: Union[str, tuple, None] = None,
        channel_tag: Union[str, tuple, None] = None,
        z_tag: Union[str, tuple, None] = None,
        y_tag: Union[str, tuple, None] = None,
        x_tag: Union[str, tuple, None] = None,
        concatenation_axes: Union[str, int, None] = None,
        **kwargs,
    ) -> AggregativePlan:
        """Dry-run an aggregative conversion and print the resource plan.

        Determines output groups and worker allocation without reading any pixel
        data.  Prints a human-readable summary and returns the plan object so it
        can be passed to :meth:`to_zarr` to skip re-planning.

        Args:
            input_path: File path, directory, or table of paths.
            output_path: Destination directory.
            includes: Comma-separated filename patterns to include.
            excludes: Comma-separated filename patterns to exclude.
            time_tag: Filename substring (or tuple of substrings) identifying the
                time axis.
            channel_tag: Filename substring (or tuple of substrings) identifying
                the channel axis.
            z_tag: Filename substring (or tuple of substrings) identifying the z
                axis.
            y_tag: Filename substring (or tuple of substrings) identifying the y
                axis.
            x_tag: Filename substring (or tuple of substrings) identifying the x
                axis.
            concatenation_axes: Axes to concatenate, e.g. ``'c'``.
            **kwargs: Additional conversion parameters.

        Returns:
            AggregativePlan with output groups and recommended worker counts.
        """
        plan = self._conv.validate_aggregative(
            input_path, output_path, includes, excludes,
            time_tag, channel_tag, z_tag, y_tag, x_tag,
            concatenation_axes, **kwargs,
        )
        logger.info("\n" + str(plan))
        return plan

    # ── metadata pass-throughs ────────────────────────────────────────────

    def show_pixel_meta(
        self,
        input_path: Union[Path, str],
        includes: Optional[str] = None,
        excludes: Optional[str] = None,
        series: int = None,
        output_file: str = None,
    ) -> None:
        """Display pixel-level and channel metadata for all files in input_path.

        Args:
            input_path: File path or directory.
            includes: Comma-separated filename patterns to include.
            excludes: Comma-separated filename patterns to exclude.
            series: Series / scene index (default: configured scene_index).
            output_file: Write output to this path; use ``.html`` for HTML output.
        """
        return self._meta.show_pixel_meta(
            input_path, includes, excludes, series, output_file)

    def update_pixel_meta(
        self,
        input_path: Union[Path, str],
        includes: Optional[str] = None,
        excludes: Optional[str] = None,
        time_scale: float = None,
        z_scale: float = None,
        y_scale: float = None,
        x_scale: float = None,
        time_unit: str = None,
        z_unit: str = None,
        y_unit: str = None,
        x_unit: str = None,
        max_workers: Optional[int] = None,
    ) -> None:
        """Update pixel-scale and unit metadata on existing OME-Zarr files.

        Args:
            input_path: OME-Zarr path or directory of OME-Zarr files.
            includes: Comma-separated filename patterns to include.
            excludes: Comma-separated filename patterns to exclude.
            time_scale: Physical pixel size along the time axis.
            z_scale: Physical pixel size along the z axis (e.g. ``0.5`` for
                0.5 µm z-steps).
            y_scale: Physical pixel size along the y axis.
            x_scale: Physical pixel size along the x axis.
            time_unit: Physical unit for the time axis (e.g. ``second``,
                ``millisecond``).
            z_unit: Physical unit for the z axis (e.g. ``micrometer``,
                ``nanometer``).
            y_unit: Physical unit for the y axis.
            x_unit: Physical unit for the x axis.
            max_workers: Number of files to process in parallel.
        """
        kwargs = {k: v for k, v in dict(max_workers=max_workers).items() if v is not None}
        return self._meta.update_pixel_meta(
            input_path, includes, excludes,
            time_scale, z_scale, y_scale, x_scale,
            time_unit, z_unit, y_unit, x_unit, **kwargs)

    def update_channel_meta(
        self,
        input_path: Union[Path, str],
        channel_labels: str = '',
        channel_colors: str = '',
        channel_intensity_limits: Literal["from_dtype", "from_array", "auto"] = "from_dtype",
        includes: Optional[str] = None,
        excludes: Optional[str] = None,
        max_workers: Optional[int] = None,
    ) -> None:
        """Update channel label, color, and intensity metadata.

        Args:
            input_path: OME-Zarr path or directory of OME-Zarr files.
            channel_labels: Index/name pairs, e.g. ``"0,DAPI;1,GFP"``.
            channel_colors: Index/hex pairs, e.g. ``"0,0000FF;1,00FF00"``.
            channel_intensity_limits: ``'from_dtype'`` (default),
                ``'from_array'``, or ``'auto'``.
            includes: Comma-separated filename patterns to include.
            excludes: Comma-separated filename patterns to exclude.
            max_workers: Number of files to process in parallel.
        """
        kwargs = {k: v for k, v in dict(max_workers=max_workers).items() if v is not None}
        return self._meta.update_channel_meta(
            input_path, channel_labels, channel_colors,
            channel_intensity_limits, includes, excludes, **kwargs)

    def version(self) -> None:
        """Display the installed EuBI-Bridge version."""
        from importlib.metadata import version as _v
        _console.print(f"EuBI-Bridge [bold cyan]{_v('eubi_bridge')}[/bold cyan]")

    # ── workflow sub-namespace ────────────────────────────────────────────

    @property
    def flow(self) -> "EuBIFlow":
        """Workflow operations — ``eubi flow create / run / select / list_waves``."""
        try:
            from eubi_flow.eubiflow import EuBIFlow  # type: ignore[import]
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "The 'eubi-flow' package is required for workflow operations but is not installed.\n"
                "Install it with:  pip install eubi-flow\n"
                "Or via the core extra:  pip install \"eubi-bridge[flow]\""
            ) from None
        return EuBIFlow()

    # ── resource management ───────────────────────────────────────────────

    def _cleanup_temp_dir(self) -> None:
        if self._dask_temp_dir is None:
            return
        try:
            if isinstance(self._dask_temp_dir, tempfile.TemporaryDirectory):
                self._dask_temp_dir.cleanup()
            elif isinstance(self._dask_temp_dir, (str, Path)):
                path = Path(self._dask_temp_dir)
                if path.exists():
                    shutil.rmtree(path)
        except Exception as e:
            logger.warning(f"Error cleaning up temporary directory: {e}")
        finally:
            self._dask_temp_dir = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_temp_dir()
        return False

    def __del__(self):
        try:
            self._cleanup_temp_dir()
        except Exception:
            pass
