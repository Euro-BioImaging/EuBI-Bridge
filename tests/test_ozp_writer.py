"""writer='ozp' (ome_zarr_pyramid on dyna-zarr) must write what writer='eubi' writes.

The same synthetic image is converted with both writers; every array (shape, dtype,
chunks, bytes) and every metadata file must match. zarr v3 array metadata is compared
with its defaults spelled out, since zarr-python writes them and TensorStore omits them.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("ome_zarr_pyramid")


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    import tifffile
    rng = np.random.default_rng(0)
    data = rng.integers(0, 4000, (13, 2, 70, 91), dtype=np.uint16)   # odd edges
    path = tmp_path_factory.mktemp("src") / "img.tif"
    tifffile.imwrite(path, data, imagej=True, metadata={"axes": "ZCYX"})
    return path


def _convert(src, out, **kw):
    from eubi_bridge.ebridge import EuBIBridge
    EuBIBridge().to_zarr(str(src), str(out), verbose=False, **kw)
    return next(Path(out).glob("*.zarr"))


def _v3_defaults(d):
    d = copy.deepcopy(d)
    if d.get("attributes") == {}:
        d.pop("attributes")
    if d.get("storage_transformers") == []:
        d.pop("storage_transformers")
    cke = d.get("chunk_key_encoding") or {}
    if cke.get("name") == "default" and cke.get("configuration") in ({"separator": "/"}, None):
        d["chunk_key_encoding"] = {"name": "default"}
    for codec in d.get("codecs", []):
        conf = codec.get("configuration") or {}
        if codec.get("name") == "sharding_indexed" and conf.get("index_location") == "end":
            conf.pop("index_location")
    return d


def _snapshot(root: Path):
    import zarr
    arrays, meta = {}, {}
    for f in sorted(root.rglob("*")):
        if f.name not in ("zarr.json", ".zattrs", ".zgroup", ".zarray"):
            continue
        rel = f.relative_to(root).as_posix()
        d = json.loads(f.read_text())
        if f.name == ".zattrs" and d == {}:
            continue
        if f.name == "zarr.json" and d.get("node_type") == "array":
            d = _v3_defaults(d)
        if f.name in (".zarray", "zarr.json") and (f.name == ".zarray"
                                                    or d.get("node_type") == "array"):
            a = zarr.open_array(str(f.parent), mode="r")
            arrays[f.parent.relative_to(root).as_posix()] = (
                tuple(a.shape), str(a.dtype),
                hashlib.sha1(np.ascontiguousarray(a[...]).tobytes()).hexdigest())
        meta[rel] = d
    return arrays, meta


@pytest.mark.parametrize("options", [
    {},
    {"downscale_method": "mean"},
    {"downscale_method": "mode"},
    {"dtype": "uint8"},
    {"compressor": "zstd", "compressor_params": {"level": 7}},
    {"auto_chunk": False, "z_chunk": 4, "y_chunk": 32, "x_chunk": 32},
    {"n_layers": 2},
    {"channel_intensity_limits": "from_array"},
], ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()) or "defaults")
def test_ozp_writes_what_eubi_writes(tmp_path, source, options):
    a = _snapshot(_convert(source, tmp_path / "eubi", writer="eubi", **options))
    b = _snapshot(_convert(source, tmp_path / "ozp", writer="ozp", **options))
    assert a[0] == b[0]                       # every array: shape, dtype, bytes
    assert a[1] == b[1]                       # every metadata file


@pytest.mark.parametrize("options", [{"zarr_format": 3},
                                     {"zarr_format": 3, "downscale_method": "mean"}],
                         ids=["v3", "v3-mean"])
def test_ozp_v3_sharded_matches(tmp_path, source, options):
    """zarr v3 / OME-Zarr 0.5 with shards: both writers clip shards to the array on
    every level (eubi used to leave level 0 unclipped), so everything matches."""
    a = _snapshot(_convert(source, tmp_path / "eubi", writer="eubi", **options))
    b = _snapshot(_convert(source, tmp_path / "ozp", writer="ozp", **options))
    assert a[0] == b[0]
    assert a[1] == b[1]


def test_target_chunk_mb_reaches_both_writers(tmp_path, source):
    """It used to be dropped before the writer, so auto-chunking always used 1 MB."""
    small = _snapshot(_convert(source, tmp_path / "s", writer="eubi", target_chunk_mb=0.05))
    big = _snapshot(_convert(source, tmp_path / "b", writer="eubi", target_chunk_mb=4))
    level0 = [k for k in small[1] if k.endswith("0/.zarray")][0]
    assert small[1][level0]["chunks"] != big[1][level0]["chunks"]
    ozp = _snapshot(_convert(source, tmp_path / "o", writer="ozp", target_chunk_mb=0.05))
    assert ozp == small


def test_writer_is_validated_and_defaults_to_auto():
    from pydantic import ValidationError

    from eubi_bridge.core.config_models import ConversionConfig
    assert ConversionConfig().writer == "auto"
    assert ConversionConfig(writer="ozp").writer == "ozp"
    assert ConversionConfig(writer="eubi").writer == "eubi"
    with pytest.raises(ValidationError):
        ConversionConfig(writer="tensorstore")


def test_auto_is_ozp_except_where_ozp_cannot_write_the_levels():
    from types import SimpleNamespace

    from eubi_bridge.core.ozp_writer import resolve_writer
    numbered = SimpleNamespace(layers={"0": 0, "1": 0, "2": 0})
    named = SimpleNamespace(layers={"s0": 0, "s1": 0})
    assert resolve_writer("auto") == "ozp"
    assert resolve_writer("auto", numbered) == "ozp"
    assert resolve_writer("auto", named) == "eubi"
    assert resolve_writer("ozp", named) == "ozp"          # explicit: refused loudly later
    assert resolve_writer("eubi") == "eubi"


def test_default_conversion_is_written_by_ozp(tmp_path, source, monkeypatch):
    import eubi_bridge.core.ozp_writer as ow
    calls = []
    orig = ow._store_multiscale_sync

    def spy(*a):
        calls.append(a[1])
        return orig(*a)
    monkeypatch.setattr(ow, "_store_multiscale_sync", spy)
    out = _snapshot(_convert(source, tmp_path / "auto", writer="auto", use_threading=True))
    assert len(calls) == 1
    assert out == _snapshot(_convert(source, tmp_path / "eubi", writer="eubi"))


# --- 0.1.5: a dask-read source with the memory budget -------------------------------
# eubi-bridge 0.1.4 passed memory_budget_mb for every ozp write; dyna refuses it for a
# dask array (dask's scheduler writes those), so EVERY conversion read through the
# standard readers (dask) failed.

def test_a_dask_source_writes_with_a_memory_budget(tmp_path):
    import asyncio
    da = pytest.importorskip("dask.array")
    import zarr
    from eubi_bridge.core.ozp_writer import store_multiscale_ozp
    data = np.random.default_rng(1).integers(0, 999, (8, 64, 64), dtype=np.uint16)
    out = tmp_path / "dask.zarr"
    asyncio.run(store_multiscale_ozp(da.from_array(data, chunks=(4, 32, 32)), str(out),
                                     axes="zyx", scales=[1.0, 1.0, 1.0],
                                     units=["micrometer"] * 3, overwrite=True,
                                     memory_budget_mb=512))
    np.testing.assert_array_equal(zarr.open_array(str(out / "0"))[...], data)


def test_the_standard_reader_converts_with_the_default_writer(tmp_path, source):
    import tifffile
    import zarr
    # worker processes, as a CLI run: the memory budget is set there
    out = _convert(source, tmp_path / "standard", pixel_reader="standard")
    # the ImageJ source is stored ZCYX; eubi-bridge writes it CZYX
    np.testing.assert_array_equal(zarr.open_array(str(out / "0"))[...],
                                  np.moveaxis(tifffile.imread(source), 1, 0))
