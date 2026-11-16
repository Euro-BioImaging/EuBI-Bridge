"""A join's region is read in one call (MicroSource.read_many): dyna's concatenation
hands all its parts to it, and it reads them with micro_reader.read_many in groups
that keep the open-file limit (micro-reader 0.5.0, Workplan MR23 joins)."""
import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
micro_reader = pytest.importorskip("micro_reader")

from dyna_zarr import from_array, operations as ops  # noqa: E402

from eubi_bridge.core import micro_source as ms  # noqa: E402

N, Y, X = 6, 130, 170


@pytest.fixture
def planes(tmp_path):
    rng = np.random.default_rng(3)
    paths, datas = [], []
    for z in range(N):
        data = rng.integers(0, 60000, size=(Y, X)).astype("u2")
        path = str(tmp_path / f"plane_z{z:03d}.tif")
        tifffile.imwrite(path, data, tile=(64, 64), compression="zlib")
        paths.append(path)
        datas.append(data)
    yield paths, np.stack(datas)
    ms.close_all()


def _join(paths):
    sources = [ms.MicroSource(p, "separate", 0, (1, 1, 1, Y, X), "u2", multifile=False)
               for p in paths]
    return ops.concatenate([from_array(s) for s in sources], axis=2)


@pytest.fixture
def micro_calls(monkeypatch):
    if not hasattr(micro_reader, "read_many"):
        pytest.skip("this micro-reader has no read_many (0.5.0)")
    calls = []
    real = micro_reader.read_many

    def counting(requests):
        requests = list(requests)
        calls.append(len(requests))
        return real(requests)
    monkeypatch.setattr(micro_reader, "read_many", counting)
    return calls


KEY = (0, 0, slice(1, 6), slice(10, 120), slice(5, 160))


def test_a_join_reads_as_its_files(planes):
    paths, want = planes
    got = np.asarray(_join(paths)[KEY])
    np.testing.assert_array_equal(got, want[1:6, 10:120, 5:160])


def test_one_micro_reader_call_per_region(planes, micro_calls):
    paths, want = planes
    got = np.asarray(_join(paths)[KEY])
    np.testing.assert_array_equal(got, want[1:6, 10:120, 5:160])
    assert micro_calls == [5]


def test_files_are_borrowed_in_groups_under_the_open_file_limit(planes, micro_calls,
                                                                monkeypatch):
    monkeypatch.setattr(ms, "open_files_limit", lambda: 8)     # groups of 2
    paths, want = planes
    got = np.asarray(_join(paths)[KEY])
    np.testing.assert_array_equal(got, want[1:6, 10:120, 5:160])
    assert micro_calls == [2, 2, 1]


def test_without_read_many_each_source_is_read(planes, monkeypatch):
    """micro-reader before 0.5.0: one read per source, same result."""
    monkeypatch.delattr(micro_reader, "read_many", raising=False)
    paths, want = planes
    got = np.asarray(_join(paths)[KEY])
    np.testing.assert_array_equal(got, want[1:6, 10:120, 5:160])
