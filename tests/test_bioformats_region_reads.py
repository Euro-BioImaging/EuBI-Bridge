"""Bio-Formats images are read one writer region at a time.

A large MRC (one 67320 x 50960 uint16 plane) exhausted RAM and then the Java
heap: the reader cut the plane into a fixed grid of 16384 x 16384 tiles while
the writer asked for 2172-row strips, and dask can only compute whole tiles, so
every strip read the four full tiles it crossed -- each tile about seven times
over, several at once.  The image is now a lazy source that reads exactly the
rectangle each region needs; these tests pin that down through the real writer.
"""
from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest

from eubi_bridge.core import data_manager
from eubi_bridge.core.data_manager import _BioFormatsSource, _build_bioformats_array
from eubi_bridge.core.writers import write_with_queue_async


class _FakeBioReader:
    """Stands in for bfio's BioReader: indexed [y, x, z, c, t], records reads."""

    def __init__(self, data):
        self.data = data                                   # (T, C, Z, Y, X)
        self.reads = []
        self.live = 0
        self.peak_live = 0
        self._lock = threading.Lock()

    def __getitem__(self, key):
        ys, xs, zs, cs, ts = key
        with self._lock:
            self.live += 1
            self.peak_live = max(self.peak_live, self.live)
        time.sleep(0.002)                  # let concurrent reads overlap
        block = self.data[ts, cs, zs, ys, xs]              # (t, c, z, y, x)
        with self._lock:
            self.reads.append(block.shape)
            self.live -= 1
        return np.ascontiguousarray(block.transpose(3, 4, 2, 1, 0))


@pytest.fixture
def fake_reader(monkeypatch):
    def make(data):
        reader = _FakeBioReader(data)
        monkeypatch.setattr(data_manager, "_get_cached_reader",
                            lambda path, series: reader)
        return reader
    return make


def _image(shape, seed=0):
    return np.random.default_rng(seed).integers(0, 60000, shape).astype(np.uint16)


class TestSource:
    def test_reads_exactly_the_requested_rectangle(self, fake_reader):
        data = _image((1, 2, 3, 40, 50))
        reader = fake_reader(data)
        source = _BioFormatsSource("x", 0, data.shape, data.dtype)

        out = source[0:1, 1:2, 0:3, 10:25, 5:45]

        np.testing.assert_array_equal(out, data[0:1, 1:2, 0:3, 10:25, 5:45])
        # One read per plane, each exactly the requested rows and columns.
        assert reader.reads == [(1, 1, 1, 15, 40)] * 3

    @pytest.mark.parametrize("key", [
        (0, 1), (0, slice(None), 2), (Ellipsis, slice(3, 9)),
        (slice(None), slice(None), slice(None), slice(0, 40, 3), slice(1, 50, 7)),
        (-1, -1, -1, 5),
    ])
    def test_numpy_indexing_semantics(self, fake_reader, key):
        data = _image((1, 2, 3, 40, 50))
        fake_reader(data)
        source = _BioFormatsSource("x", 0, data.shape, data.dtype)
        np.testing.assert_array_equal(source[key], data[key])

    def test_a_rectangle_too_big_for_the_heap_is_read_in_strips(
            self, fake_reader, monkeypatch):
        """No single Java read may exceed the per-read budget."""
        data = _image((1, 1, 1, 100, 64))
        reader = fake_reader(data)
        monkeypatch.setattr(data_manager, "_java_read_budget_bytes",
                            lambda: 64 * 2 * 7)            # 7 rows of 64 uint16
        source = _BioFormatsSource("x", 0, data.shape, data.dtype)

        out = source[...]

        np.testing.assert_array_equal(out, data)
        assert max(r[3] for r in reader.reads) == 7
        assert sum(r[3] for r in reader.reads) == 100      # every row once


class TestWriterReadsEachPixelOnce:
    """The regression: strip regions across square tiles re-read the tiles."""

    def _write(self, tmp_path, fake_reader, data, region_mb, **kw):
        reader = fake_reader(data)
        arr = _build_bioformats_array("x", 0, data.shape, data.dtype)
        asyncio.run(write_with_queue_async(
            arr=arr, output_path=str(tmp_path / "out.zarr"),
            output_chunks=kw.pop("chunks", (1, 1, 1, 64, 64)),
            zarr_format=2, dtype=data.dtype, region_size_mb=region_mb,
            max_concurrency=2, overwrite=True, **kw))
        return reader

    def test_strip_regions_read_each_pixel_exactly_once(self, tmp_path,
                                                        fake_reader):
        import zarr
        data = _image((1, 1, 1, 1000, 700))
        # Small regions: the writer reads many strips, as with the MRC.
        reader = self._write(tmp_path, fake_reader, data, region_mb=0.2)

        pixels_read = sum(int(np.prod(r)) for r in reader.reads)
        assert pixels_read == data.size, (
            f"read {pixels_read / data.size:.2f}x the image")
        np.testing.assert_array_equal(zarr.open_array(str(tmp_path / "out.zarr"))[:],
                                      data)

    def test_no_read_is_larger_than_a_region(self, tmp_path, fake_reader):
        data = _image((1, 1, 1, 1000, 700))
        region_mb = 0.2
        reader = self._write(tmp_path, fake_reader, data, region_mb=region_mb)
        largest = max(int(np.prod(r)) for r in reader.reads) * data.itemsize
        # A region is rounded to whole output chunks, so allow one chunk row.
        assert largest <= region_mb * 1024 ** 2 + 64 * 700 * data.itemsize


class TestRegionBudget:
    def _clus(self, **kw):
        from eubi_bridge.core.config_models import ClusterConfig
        return ClusterConfig(**{"region_size_mb": 256, "max_workers": 4,
                                "max_concurrency": 2, "queue_size": 4, **kw})

    def test_kept_when_everything_fits(self, monkeypatch):
        from eubi_bridge.conversion import conversion_worker as cw
        monkeypatch.setattr(cw.psutil, "virtual_memory",
                            lambda: type("M", (), {"available": 64 * 1024 ** 3})())
        assert cw._region_budget_mb(self._clus()) == 256

    def test_lowered_when_regions_in_flight_exceed_half_the_ram(self,
                                                               monkeypatch):
        from eubi_bridge.conversion import conversion_worker as cw
        monkeypatch.setattr(cw.psutil, "virtual_memory",
                            lambda: type("M", (), {"available": 8 * 1024 ** 3})())
        # 4 workers x (3 x 2 + 4) = 40 regions in flight; half of 8 GB / 40.
        assert cw._region_budget_mb(self._clus()) == pytest.approx(4096 / 40)

    def test_never_below_the_floor(self, monkeypatch):
        from eubi_bridge.conversion import conversion_worker as cw
        monkeypatch.setattr(cw.psutil, "virtual_memory",
                            lambda: type("M", (), {"available": 64 * 1024 ** 2})())
        assert cw._region_budget_mb(self._clus()) == cw._MIN_REGION_MB


class _Unpicklable(Exception):
    """Stands in for a JPype Java exception, which pickle cannot look up."""

    def __reduce__(self):
        raise TypeError("cannot pickle a java exception")


def _raise_unpicklable():
    try:
        raise _Unpicklable("java.lang.OutOfMemoryError: Java heap space")
    except Exception as exc:
        raise RuntimeError("scene failed") from exc


def _run_wrapped():
    from eubi_bridge.conversion import worker_init
    worker_init._worker_initialized = True               # skip JVM start-up
    return worker_init.safe_worker_wrapper(_raise_unpicklable)()


def test_a_java_error_is_reported_not_a_broken_pool():
    """It broke the pool ('terminated abruptly') and triggered retries."""
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(1) as pool:
        with pytest.raises(RuntimeError) as info:
            pool.submit(_run_wrapped).result()
    assert "Java heap space" in str(info.value)
    assert "terminated abruptly" not in str(info.value)


class TestBfTileSizeDeprecation:
    def test_still_accepted(self):
        from eubi_bridge.core.config_models import ClusterConfig
        assert ClusterConfig(bf_tile_size_mb=512).bf_tile_size_mb == 512

    def test_dropped_from_an_old_config_on_load(self, tmp_path, monkeypatch):
        import json
        from eubi_bridge.ebridge import ConfigManager, EuBIBridge
        config_dir = tmp_path / "cfg"
        config_dir.mkdir()
        path = config_dir / ".eubi_config.json"
        monkeypatch.setattr(ConfigManager, "_get_config_dir", lambda self: config_dir)
        monkeypatch.setattr(ConfigManager, "_get_json_path", lambda self: path)
        EuBIBridge().config                                  # writes defaults
        old = json.loads(path.read_text())
        old["cluster"]["bf_tile_size_mb"] = 512.0            # as 0.1.2 wrote it
        path.write_text(json.dumps(old))

        assert "bf_tile_size_mb" not in EuBIBridge().config["cluster"]
        assert "bf_tile_size_mb" not in json.loads(path.read_text())["cluster"]
