"""
CLI command audit harness for the `eubi` entrypoint.

Two complementary layers:

1. **In-process** — call ``EuBIBridge`` methods directly with ``configpath``
   pointed at a tmp dir, so config/named-config/display commands are exercised
   fast and assertable without touching the real ``~/.eubi_bridge``.

2. **Subprocess** — a few real ``eubi ...`` invocations to catch Fire
   arg-parsing issues (comma-list args, the ``'default'`` sentinel, chaining).

Also includes an automated **signature ↔ documentation parity** test so a
parameter can never silently ship undocumented.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# Make the doc generator's Args-parser importable for the parity test.
_DOCS_SCRIPTS = _ROOT / "docs" / "scripts"
if str(_DOCS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_DOCS_SCRIPTS))

from eubi_bridge.ebridge import EuBIBridge
from gen_cli_ref import (
    _parse_args_section, _FIELD_HINTS, _resolve_method, COMMAND_GROUPS,
)

ALL_COMMANDS = [cmd for group in COMMAND_GROUPS.values() for cmd in group]


@pytest.fixture
def bridge(tmp_path):
    """An EuBIBridge whose config lives in an isolated tmp directory."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    return EuBIBridge(configpath=str(cfg_dir))


# ---------------------------------------------------------------------------
# Signature ↔ documentation parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_COMMANDS)
def test_signature_doc_parity(name):
    """Every CLI parameter must be documented in the method's Args: block or
    covered by the generator's _FIELD_HINTS fallback. Catches params that would
    render as a bare "—" in the generated reference.
    """
    method = _resolve_method(name)
    assert method is not None, f"Command {name!r} could not be resolved"

    sig = inspect.signature(method)
    params = [p for p in sig.parameters if p not in ("self", "args", "kwargs")]

    doc = inspect.getdoc(method) or ""
    documented = set(_parse_args_section(doc))

    missing = [
        p for p in params
        if p not in documented and p not in _FIELD_HINTS
    ]
    assert not missing, (
        f"{name}: parameters undocumented and not in _FIELD_HINTS: {missing}"
    )


@pytest.mark.parametrize("name", ALL_COMMANDS)
def test_has_summary_docstring(name):
    """Every command needs at least a one-line summary for its reference card."""
    method = _resolve_method(name)
    doc = inspect.getdoc(method) or ""
    assert doc.strip(), f"{name}: missing docstring summary"


# ---------------------------------------------------------------------------
# Display & Info + Reset (no image data)
# ---------------------------------------------------------------------------

def test_version(bridge, capsys):
    bridge.version()
    out = capsys.readouterr().out
    assert "EuBI-Bridge" in out


def test_show_root_defaults(bridge, capsys):
    bridge.show_root_defaults()
    out = capsys.readouterr().out
    assert "Defaults" in out or "Installation Defaults" in out


def test_show_config_default(bridge, capsys):
    bridge.show_config()
    out = capsys.readouterr().out
    assert "Configuration" in out


def test_reset_config_roundtrip(bridge):
    # Mutate, then reset, and confirm defaults are restored.
    default_workers = bridge.root_defaults["cluster"]["max_workers"]
    bridge.configure.cluster(max_workers=99)
    assert bridge.config["cluster"]["max_workers"] == 99
    bridge.reset_config()
    assert bridge.config["cluster"]["max_workers"] == default_workers


# ---------------------------------------------------------------------------
# Configuration (no image data)
# ---------------------------------------------------------------------------

def test_configure_cluster_persists(bridge):
    bridge.configure.cluster(max_workers=8, queue_size=8)
    cfg = json.loads(bridge._get_json_path().read_text())
    assert cfg["cluster"]["max_workers"] == 8
    assert cfg["cluster"]["queue_size"] == 8


def test_configure_omitted_args_unchanged(bridge):
    """Omitted args (the 'default' sentinel) must not overwrite existing values."""
    bridge.configure.cluster(max_workers=8)
    bridge.configure.cluster(queue_size=16)  # max_workers omitted
    cfg = bridge.config["cluster"]
    assert cfg["max_workers"] == 8   # preserved
    assert cfg["queue_size"] == 16


def test_configure_conversion_zarr_format(bridge):
    bridge.configure.conversion(zarr_format=3)
    assert bridge.config["conversion"]["zarr_format"] == 3


def test_configure_invalid_value_rejected(bridge):
    """Out-of-range / wrong-type values should fail fast via Pydantic."""
    with pytest.raises(Exception):
        bridge.configure.conversion(zarr_format=99)


# ---------------------------------------------------------------------------
# Named Configs (filesystem only)
# ---------------------------------------------------------------------------

def test_named_config_save_list_delete(bridge):
    bridge.configure.cluster(max_workers=32)
    bridge.save_as("hpc")
    assert "hpc" in bridge.list_configs()
    bridge.delete_config("hpc")
    assert "hpc" not in bridge.list_configs()


def test_with_config_overrides_defaults(bridge):
    bridge.configure.cluster(max_workers=64)
    bridge.save_as("hpc")
    # mutate the active config away from hpc
    bridge.configure.cluster(max_workers=4)
    hpc_bridge = bridge.with_config("hpc")
    assert hpc_bridge.config["cluster"]["max_workers"] == 64


def test_with_config_missing_raises(bridge):
    with pytest.raises(KeyError):
        bridge.with_config("does_not_exist")


def test_update_config_create(bridge):
    bridge.configure.cluster(max_workers=16)
    bridge.update_config("profileA", create=True)
    assert "profileA" in bridge.list_configs()


def test_update_config_missing_without_create_raises(bridge):
    with pytest.raises(FileNotFoundError):
        bridge.update_config("nope")


# ---------------------------------------------------------------------------
# Metadata round-trip (subprocess — update_* use a spawn ProcessPool)
# ---------------------------------------------------------------------------

def _run_eubi(*args):
    """Invoke the real `eubi` CLI; return CompletedProcess (raises on failure)."""
    import os
    import shutil
    import subprocess

    exe = shutil.which("eubi")
    if exe is None:
        scripts = Path(sys.executable).parent / "Scripts"
        for cand in (scripts / "eubi.exe", scripts / "eubi",
                     Path(sys.executable).parent / "eubi"):
            if cand.exists():
                exe = str(cand)
                break
    if exe is None:
        pytest.skip("eubi executable not found on PATH")

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    res = subprocess.run(
        [exe, *args], capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env,
    )
    assert res.returncode == 0, f"eubi {args} failed:\n{res.stderr}\n{res.stdout}"
    return res


def _read_zattrs_labels(zarr_path: Path):
    """Return the OMERO channel labels from a zarr's group attributes (v2 or v3)."""
    attrs_file = zarr_path / ".zattrs"
    if attrs_file.exists():
        attrs = json.loads(attrs_file.read_text())
    else:  # zarr v3
        attrs = json.loads((zarr_path / "zarr.json").read_text()).get("attributes", {})
    channels = attrs.get("omero", {}).get("channels", [])
    return [ch.get("label") for ch in channels]


def test_metadata_roundtrip(ome_tiff_3ch, tmp_path):
    """Convert a fixture, then update + read back channel metadata via the CLI."""
    src, _, _ = ome_tiff_3ch
    out = tmp_path / "out"
    _run_eubi("to_zarr", str(src), str(out))

    zarrs = list(out.rglob("*.zarr"))
    assert zarrs, f"no zarr produced under {out}"
    zarr_path = zarrs[0]

    _run_eubi("update_channel_meta", str(zarr_path),
              "--channel_labels", "0,DAPI;1,GFP;2,RFP")
    labels = _read_zattrs_labels(zarr_path)
    assert labels[:3] == ["DAPI", "GFP", "RFP"], labels

    # show_pixel_meta must run clean on the resulting store.
    res = _run_eubi("show_pixel_meta", str(zarr_path))
    assert "DAPI" in res.stdout
