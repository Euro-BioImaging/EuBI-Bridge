"""io_backend='zarrista' without zarrista installed: one clear error before any
file is converted, for a run's own setting and for a table row naming it."""
import os

import pandas as pd
import pytest

from eubi_bridge import ebridge
from eubi_bridge.ebridge import EuBIBridge, IOBackendUnavailableError, _check_io_backend

SMALL_CZI = os.path.join(os.environ.get("MICROSCOPY_DATA", os.path.expanduser("~/Desktop/data/microscopy")),
                         "formats", "czi", "assorted", "xyz__multiple_images.czi")


@pytest.fixture
def no_zarrista(monkeypatch):
    monkeypatch.setattr(ebridge, "_zarrista_installed", lambda: False)


@pytest.fixture
def no_dispatch(monkeypatch):
    """Fail the test if a job reaches the workers."""
    import eubi_bridge.conversion.dispatcher as dispatcher

    def refuse(*args, **kwargs):
        raise AssertionError("a job was dispatched")

    for name in ("dispatch_unary_jobs", "dispatch_mixed_jobs", "dispatch_aggregative_job"):
        monkeypatch.setattr(dispatcher, name, refuse)


def test_the_check_names_both_ways_out(no_zarrista):
    _check_io_backend("tensorstore", None)                  # nothing asked of zarrista
    with pytest.raises(IOBackendUnavailableError, match=r'dyna-zarr\[zarrista\].*--io_backend tensorstore'):
        _check_io_backend("tensorstore", "zarrista")
    assert issubclass(IOBackendUnavailableError, ImportError)


def test_an_installed_zarrista_passes(monkeypatch):
    monkeypatch.setattr(ebridge, "_zarrista_installed", lambda: True)
    _check_io_backend("zarrista")


def test_a_run_set_to_zarrista_stops_before_any_file(tmp_path, no_zarrista, no_dispatch):
    if not os.path.exists(SMALL_CZI):
        pytest.skip(f"sample not found: {SMALL_CZI}")
    out = tmp_path / "out"
    with pytest.raises(IOBackendUnavailableError):
        EuBIBridge().to_zarr(SMALL_CZI, str(out), io_backend="zarrista")
    assert not out.exists() or not any(out.iterdir())


def test_a_table_row_naming_zarrista_stops_the_run(tmp_path, no_zarrista, no_dispatch):
    if not os.path.exists(SMALL_CZI):
        pytest.skip(f"sample not found: {SMALL_CZI}")
    table = tmp_path / "batch.csv"
    pd.DataFrame({"input_path": [SMALL_CZI, SMALL_CZI],
                  "output_path": [str(tmp_path / "a"), str(tmp_path / "b")],
                  "io_backend": ["tensorstore", "zarrista"]}).to_csv(table, index=False)
    with pytest.raises(IOBackendUnavailableError):
        EuBIBridge().to_zarr(str(table), str(tmp_path / "out"))
