"""micro-reader 0.1.0 names the image index "series" (it was "scene"); eubi-bridge 0.3.0
reads it, and its own option follows: scene_index -> series_index (plan 7.7, 8.32).

- micro-reader takes the files it took before: a renamed index outside eubi's names would
  send them to the older readers with only a log line (plan 7.10), so this is checked on
  files of every route, not left to the conversions to show.
- the old option name works everywhere a user gives it (a script, the CLI, a saved
  config, a conversion table, a saved GUI batch), with a FutureWarning; the new name wins
  when both are given.
- the output's eubi_bridge block holds both series_index and scene_index (for several
  releases), and outputs keep their _scene<N> names.
- so do the files eubi saves (the config, a batch's batch.csv and batch_config.json):
  equal pairs load silently, pairs that differ (eubi 0.2.x changed its own) warn.
- a micro-reader outside >=0.2.1,<0.4 (since eubi-bridge 0.3.3; <0.3 in 0.3.2, >=0.1,<0.2
  before) stops
  a conversion with MicroReaderVersionError, not a silent fallback to the older readers.
"""
from __future__ import annotations

import csv
import json
import warnings

import numpy as np
import pytest

pytest.importorskip("micro_reader")
pytest.importorskip("ome_zarr_pyramid")
tifffile = pytest.importorskip("tifffile")
zarr = pytest.importorskip("zarr")

from eubi_bridge.core import config_models  # noqa: E402
from tests.conftest import requires_full  # noqa: E402
from eubi_bridge.core.config_models import (ConversionJob, ReaderConfig,  # noqa: E402
                                            SeriesIndexConflictWarning, rename_settings,
                                            series_ui)

MESSAGE = "scene_index is deprecated; it is now called series_index"


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch):
    monkeypatch.setattr(config_models, "_renamed_warned", set())


def _future(record) -> list:
    """The rename's FutureWarnings among `record` (not the other libraries' own)."""
    return [w for w in record if issubclass(w.category, FutureWarning)
            and ("scene" in str(w.message) or "Scenes" in str(w.message))]


def _two_series_tiff(path):
    """Two series with a stage position each (a positioned output is one of several,
    so its eubi_bridge block says which)."""
    with tifffile.TiffWriter(str(path), ome=True) as tw:
        for value in (5, 9):
            tw.write(np.full((2, 16, 20), value, "u2"), photometric="minisblack",
                     metadata={"axes": "CYX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5,
                               "Plane": {"PositionX": [10.0 * value] * 2,
                                         "PositionXUnit": ["µm"] * 2,
                                         "PositionY": [0.0] * 2,
                                         "PositionYUnit": ["µm"] * 2}})
    return path


def _czi(path, tiles=1):
    from tests._czi_writer import raw_payload, write_czi
    write_czi(str(path), [dict(scene=0, mosaic=m, plane={"C": 0}, x=16 * m, y=0, height=12,
                               width=16, dtype="u2", compression=0,
                               payload=raw_payload(np.full((12, 16), m, "u2")))
                          for m in range(tiles)])
    return path


# -- the reader choice -------------------------------------------------------------

def test_micro_reader_names_the_index_series():
    """The installed micro-reader is one this eubi-bridge reads (MICRO_READER_RANGE)."""
    import micro_reader

    from eubi_bridge.core.micro_source import MICRO_READER_RANGE
    parts = [int(p) for p in micro_reader.__version__.split(".")[:3]]
    got = tuple(parts + [0] * (3 - len(parts)))
    low, high = MICRO_READER_RANGE
    assert low <= got < high


@pytest.mark.parametrize("kind", ["ome_tiff", "czi", "czi_tiles"])
def test_micro_reader_takes_the_file(tmp_path, kind):
    """``_open_input`` must not send micro-reader's files to the older readers: with
    'series' outside eubi's index names it did, saying so only in the log."""
    from eubi_bridge.conversion.pyramid_path import SERIES, _open_input
    path = (_two_series_tiff(tmp_path / "two.ome.tif") if kind == "ome_tiff"
            else _czi(tmp_path / "one.czi", tiles=4 if kind == "czi_tiles" else 1))
    job = ConversionJob.from_kwargs(str(path), "out", {})
    f, reader = _open_input(job)
    try:
        assert reader == "micro-reader", reader
        assert all(SERIES in index for index in f.indices)
    finally:
        f.close()


def test_micro_reader_open_takes_a_czi(tmp_path):
    """``MicroReader.open`` returns None on ANY failure (the older reader then reads the
    file, silently): a renamed index it does not know would be such a failure."""
    from eubi_bridge.core.micro_source import MicroReader
    reader = MicroReader.open(str(_czi(tmp_path / "one.czi")))
    assert reader is not None
    assert reader.n_scenes == 1
    assert reader.get_image_dask_data().shape == (1, 1, 1, 12, 16)


def test_micro_array_finds_the_series(tmp_path):
    from eubi_bridge.core.micro_source import micro_array
    path = _two_series_tiff(tmp_path / "two.ome.tif")
    arr, why = micro_array(str(path), 1, None)
    assert why == "" and int(np.asarray(arr).max()) == 9


@pytest.mark.parametrize("version, why", [("0.0.4", "names the series index 'scene'"),
                                          ("0.1.0", "declares no coordinate systems"),
                                          ("0.4.0", None)])
def test_a_wrong_micro_reader_is_no_fallback(tmp_path, monkeypatch, version, why):
    """A micro-reader outside >=0.2.1,<0.4 forced in (pip check reports it): every route
    that reads with micro-reader raises MicroReaderVersionError naming the fix, none
    goes to the older readers with only a log line."""
    import micro_reader

    from eubi_bridge.conversion.pyramid_path import _open_input
    from eubi_bridge.core import micro_source
    path = str(_czi(tmp_path / "one.czi"))
    monkeypatch.setattr(micro_reader, "__version__", version)
    monkeypatch.setattr(micro_source, "_micro_reader_checked", False)
    fix = r'pip install -U "micro-reader>=0\.2\.1,<0\.4"'
    for call in (lambda: _open_input(ConversionJob.from_kwargs(path, "out", {})),
                 lambda: micro_source.MicroReader.open(path),
                 lambda: micro_source.micro_array(path, 0, None)):
        with pytest.raises(micro_source.MicroReaderVersionError,
                           match=f"micro-reader {version} is installed.*{fix}") as err:
            call()
        for reason in ("names the series index 'scene'", "declares no coordinate systems"):
            assert (reason in str(err.value)) == (reason == why)
    monkeypatch.setattr(micro_reader, "__version__", "0.2.1")     # the right one: taken
    assert micro_source.MicroReader.open(path) is not None


FIX = r'pip install -U "micro-reader>=0\.2\.1,<0\.4"'


def test_a_wrong_micro_reader_reaches_a_script_as_itself(tmp_path, monkeypatch):
    """Through the Python API, the guard is MicroReaderVersionError, not the
    dispatcher's "1/1 tasks failed" RuntimeError holding its name (G2 observation 2);
    nothing is written. Threads here (a worker process imports micro-reader anew: the
    next test)."""
    import micro_reader

    from eubi_bridge.core import micro_source
    from eubi_bridge.ebridge import EuBIBridge
    monkeypatch.setattr(micro_reader, "__version__", "0.0.4")
    monkeypatch.setattr(micro_source, "_micro_reader_checked", False)
    out = tmp_path / "out"
    with pytest.raises(micro_source.MicroReaderVersionError,
                       match=f"micro-reader 0.0.4 is installed.*{FIX}"):
        EuBIBridge(configpath=str(tmp_path / "cfg")).to_zarr(
            str(_czi(tmp_path / "one.czi")), str(out), use_threading=True, verbose=False)
    assert not out.exists()
    from eubi_bridge.conversion import metadata_path            # show_pixel_meta's read
    with pytest.raises(micro_source.MicroReaderVersionError, match=FIX):
        metadata_path.read_metadata(str(tmp_path / "one.czi"), {})


def test_a_wrong_micro_reader_reaches_a_script_from_a_worker_process(tmp_path):
    """The same from the default process pool (spawned workers): a micro_reader that
    says 0.0.4 first on the path, as a forced install leaves it."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    fake = tmp_path / "fake" / "micro_reader"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text('__version__ = "0.0.4"\n')
    script = tmp_path / "convert.py"
    script.write_text(
        "import sys\n"
        "from eubi_bridge.core.micro_source import MicroReaderVersionError\n"
        "from eubi_bridge.ebridge import EuBIBridge\n"
        "try:\n"
        "    EuBIBridge().to_zarr(sys.argv[1], sys.argv[2], verbose=False)\n"
        "except MicroReaderVersionError as e:\n"
        "    print('TYPED', e)\n"
        "except Exception as e:\n"
        "    print('OTHER', type(e).__name__, e)\n")
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "EUBI_CONFIG_DIR": str(tmp_path / "cfg"),
           "PYTHONPATH": os.pathsep.join([str(tmp_path / "fake"), str(root)])}
    out = tmp_path / "out"
    run = subprocess.run([sys.executable, str(script), str(_czi(tmp_path / "one.czi")),
                          str(out)], env=env, cwd=str(tmp_path), capture_output=True,
                         text=True, timeout=300)
    last = [line for line in run.stdout.splitlines() if line.startswith(("TYPED", "OTHER"))]
    assert last and last[-1].startswith("TYPED"), (run.stdout[-2000:], run.stderr[-2000:])
    assert 'pip install -U "micro-reader>=0.2.1,<0.4"' in last[-1]
    assert not out.exists()


def _raise_version_error():
    from eubi_bridge.core.micro_source import MicroReaderVersionError
    raise MicroReaderVersionError("eubi-bridge needs micro-reader >=0.2.1,<0.4. Fix: x")


def test_the_worker_wrapper_and_the_dispatcher_keep_the_version_error():
    """safe_worker_wrapper turns other errors into a RuntimeError (picklable); this one
    stays itself (it pickles: a worker process sends it back), and the dispatcher raises
    it rather than its "tasks failed" summary, also from an aggregative group."""
    import pickle

    from eubi_bridge.conversion import dispatcher
    from eubi_bridge.conversion.worker_init import safe_worker_wrapper
    from eubi_bridge.core.micro_source import MicroReaderVersionError
    with pytest.raises(MicroReaderVersionError, match="Fix: x") as err:
        safe_worker_wrapper(_raise_version_error)()
    again = pickle.loads(pickle.dumps(err.value))
    assert type(again) is MicroReaderVersionError and str(again) == str(err.value)

    def other():
        raise ValueError("bad file")
    with pytest.raises(RuntimeError, match="ValueError: bad file"):
        safe_worker_wrapper(other)()
    with pytest.raises(MicroReaderVersionError, match="Fix: x"):
        dispatcher._report_failures([{"status": "success"}, RuntimeError("other"),
                                     err.value])
    with pytest.raises(RuntimeError, match="1/2 tasks failed"):     # no version error
        dispatcher._report_failures([{"status": "success"}, RuntimeError("other")])

    def failing_group(job, plan=None):
        raise err.value
    from unittest import mock
    with mock.patch.object(dispatcher, "dispatch_aggregative_job", failing_group):
        with pytest.raises(MicroReaderVersionError):
            dispatcher.dispatch_aggregative_jobs([object(), object()])


def test_as_mosaic_asks_micro_reader_for_mosaic(tmp_path, monkeypatch):
    """eubi passes micro-reader 0.1.0's word: no DeprecationWarning for 'stitched'."""
    import micro_reader

    from eubi_bridge.conversion.pyramid_path import _open_micro
    asked = []
    real = micro_reader.open

    def spy(path, *args, **kwargs):
        asked.append(kwargs.get("tiles"))
        return real(path, *args, **kwargs)
    monkeypatch.setattr(micro_reader, "open", spy)
    job = ConversionJob.from_kwargs(str(_czi(tmp_path / "m.czi", tiles=2)), "out",
                                    {"as_mosaic": True})
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _open_micro(job).close()
    assert asked == ["mosaic"]


@requires_full                                  # the older readers: not in eubi-bridge-lite
def test_the_older_readers_name_their_index_series_too(tmp_path):
    """One index name for every file: the aggregative path keys its joined image
    ``(("series", s),)`` whichever reader read the files."""
    from eubi_bridge.conversion.legacy_image import LegacyFile
    from eubi_bridge.conversion.pyramid_path import SERIES
    path = tmp_path / "plain.tif"
    tifffile.imwrite(str(path), np.zeros((3, 8, 8), "u1"), imagej=True,
                     metadata={"axes": "ZYX"})
    job = ConversionJob.from_kwargs(str(path), "out", {"pixel_reader": "standard",
                                                       "metadata_reader": "bfio"})
    f = LegacyFile(job)
    try:
        assert f.indices == [{SERIES: 0}]
        assert f.image({SERIES: 0}).index == {SERIES: 0}
    finally:
        f.close()


# -- the option's old name --------------------------------------------------------

def test_the_old_name_works_with_a_future_warning():
    with pytest.warns(FutureWarning, match=MESSAGE):
        config = ReaderConfig(scene_index="0,2")
    assert config.series_index == "0,2"
    assert "scene_index" not in config.model_dump()


def test_the_new_name_gives_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ReaderConfig(series_index=1).series_index == 1


def test_given_both_the_new_name_wins():
    """Both names, different values: series_index, with a warning naming both."""
    with pytest.warns(SeriesIndexConflictWarning, match=r"scene_index=2 and series_index=3"):
        assert rename_settings({"scene_index": 2, "series_index": 3}) == {"series_index": 3}


@pytest.mark.parametrize("scene, series", [(1, 1), ("1", 1), (1.0, "1"), ("0,2", [0, 2]),
                                           ("all", "all")])
def test_given_both_and_equal_is_silent(scene, series):
    """What a file eubi-bridge 0.3.0 saved holds (plan 8.32): no warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert rename_settings({"scene_index": scene, "series_index": series}) == {
            "series_index": series}


def test_a_job_never_carries_the_old_name():
    with pytest.warns(FutureWarning):
        job = ConversionJob.from_kwargs("in.tif", "out", {"scene_index": "all"})
    assert job.readers.series_index == "all"
    assert "scene_index" not in job.extra and "scene_index" not in job.to_worker_kwargs()


def test_a_saved_config_loads_its_scene_index_as_series_index(tmp_path):
    from eubi_bridge.ebridge import EuBIBridge
    EuBIBridge(configpath=str(tmp_path)).config                  # writes the defaults
    path = tmp_path / ".eubi_config.json"
    saved = json.loads(path.read_text())
    saved["readers"].pop("series_index")                          # as 0.2.x wrote it
    saved["readers"]["scene_index"] = "all"
    path.write_text(json.dumps(saved))
    readers = EuBIBridge(configpath=str(tmp_path)).config["readers"]
    assert readers["series_index"] == "all" and "scene_index" not in readers
    on_disk = json.loads(path.read_text())["readers"]              # saved again: both
    assert on_disk["series_index"] == on_disk["scene_index"] == "all"


def _saved_readers(folder) -> dict:
    return json.loads((folder / ".eubi_config.json").read_text())["readers"]


def test_the_saved_config_holds_both_names(tmp_path):
    """Plan 8.32: an eubi-bridge 0.2.x sharing the config folder reads the same series."""
    from eubi_bridge.ebridge import EuBIBridge
    bridge = EuBIBridge(configpath=str(tmp_path))
    bridge.config                                                  # writes the defaults
    on_disk = _saved_readers(tmp_path)
    assert on_disk["series_index"] == on_disk["scene_index"] == 0
    bridge.configure.readers(series_index=3)
    on_disk = _saved_readers(tmp_path)
    assert on_disk["series_index"] == on_disk["scene_index"] == 3
    assert "scene_index" not in bridge.config["readers"]           # in memory: one name
    with warnings.catch_warnings():
        warnings.simplefilter("error")                              # equal: no warning
        assert EuBIBridge(configpath=str(tmp_path)).config["readers"]["series_index"] == 3
    bridge.save_as("named")
    named = json.loads((tmp_path / "named.json").read_text())["readers"]
    assert named["series_index"] == named["scene_index"] == 3


def test_a_saved_config_whose_names_differ(tmp_path):
    """eubi-bridge 0.2.x changed scene_index in a shared config: series_index is used,
    with a warning naming both values and the file; the file is left as it is until a
    0.3.0 setting is saved (then both are equal)."""
    from eubi_bridge.ebridge import EuBIBridge
    EuBIBridge(configpath=str(tmp_path)).config
    path = tmp_path / ".eubi_config.json"
    saved = json.loads(path.read_text())
    saved["readers"]["series_index"], saved["readers"]["scene_index"] = 2, 3
    path.write_text(json.dumps(saved))
    with pytest.warns(SeriesIndexConflictWarning,
                      match=r"\.eubi_config\.json: scene_index=3 and series_index=2 differ"):
        bridge = EuBIBridge(configpath=str(tmp_path))
        assert bridge.config["readers"]["series_index"] == 2
    assert _saved_readers(tmp_path)["scene_index"] == 3              # not settled silently
    bridge.configure.readers(series_index=3)
    on_disk = _saved_readers(tmp_path)
    assert on_disk["series_index"] == on_disk["scene_index"] == 3


REPLACED = (r"The next time eubi-bridge 0\.3\.0 saves this file .* it writes "
            r"series_index=2 under both names: scene_index=3 is then replaced")


@pytest.mark.parametrize("cause", ["another setting saved", "a load that updates it"])
def test_a_differing_scene_index_is_replaced_by_the_next_save_and_says_so(tmp_path, cause):
    """B4 r2 minor 1: 0.3.0 saves the series under both names, so whatever saves a
    config whose names differ (a setting of another section; a load that drops a
    retired key) replaces scene_index. The load's warning says so, naming both values
    and which one wins."""
    from eubi_bridge.ebridge import EuBIBridge
    EuBIBridge(configpath=str(tmp_path)).config
    path = tmp_path / ".eubi_config.json"
    saved = json.loads(path.read_text())
    saved["readers"]["series_index"], saved["readers"]["scene_index"] = 2, 3
    if cause == "a load that updates it":
        saved["conversion"]["skip_dask"] = True                  # a retired key
    path.write_text(json.dumps(saved))
    with pytest.warns(SeriesIndexConflictWarning,
                      match=r"scene_index=3 and series_index=2 differ; series_index=2 "
                            r"is used.*" + REPLACED):
        bridge = EuBIBridge(configpath=str(tmp_path))
        bridge.config
    if cause == "another setting saved":
        assert _saved_readers(tmp_path)["scene_index"] == 3        # the load alone keeps it
        bridge.configure.cluster(max_retries=5)
    on_disk = _saved_readers(tmp_path)
    assert on_disk["series_index"] == on_disk["scene_index"] == 2


def test_only_saved_files_speak_of_the_next_save():
    """Both names in a call or a conversion table the user wrote: nothing eubi saves."""
    with pytest.warns(SeriesIndexConflictWarning) as record:
        rename_settings({"scene_index": 3, "series_index": 2})
    assert "next time" not in str(record[0].message)


def test_configure_readers_takes_the_old_name(tmp_path):
    from eubi_bridge.ebridge import EuBIBridge
    bridge = EuBIBridge(configpath=str(tmp_path))
    with pytest.warns(FutureWarning, match=MESSAGE):
        bridge.configure.readers(scene_index=2)
    assert bridge.config["readers"]["series_index"] == 2
    with pytest.raises(TypeError, match="unknown setting"):
        bridge.configure.readers(scene_indx=2)


def test_a_conversion_table_column_named_scene_index(tmp_path):
    """The table's own column wins over the run-wide series_index (which fills in only
    the columns a table lacks)."""
    from eubi_bridge.utils.path_utils import take_filepaths
    table = tmp_path / "table.csv"
    table.write_text("input_path,scene_index\nimg.tif,1\n")
    with pytest.warns(FutureWarning, match=MESSAGE):
        df = take_filepaths(str(table), series_index=0)
    assert "scene_index" not in df.columns
    assert df["series_index"].tolist() == [1]


def test_a_conversion_table_with_both_columns(tmp_path):
    """Per row: equal or a blank new cell is silent; different values warn, naming the
    row and both values, and series_index is used."""
    from eubi_bridge.utils.path_utils import take_filepaths
    table = tmp_path / "table.csv"
    table.write_text("input_path,series_index,scene_index\n"
                     "a.tif,1,1\nb.tif,,2\nc.tif,all,all\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = take_filepaths(str(table), series_index=0)
    assert "scene_index" not in df.columns
    assert [str(v) for v in df["series_index"]] == ["1", "2", "all"]
    table.write_text("input_path,series_index,scene_index\na.tif,1,3\n")
    with pytest.warns(SeriesIndexConflictWarning,
                      match=r"row 0 \(.*a\.tif\): scene_index=3 and series_index=1"):
        df = take_filepaths(str(table), series_index=0)
    assert df["series_index"].tolist() == [1]


def _outputs(root) -> dict:
    """{store name: (eubi_bridge block, pixels)}."""
    out = {}
    for store in sorted(root.glob("*.zarr")):
        group = zarr.open_group(str(store), mode="r")
        attrs = group.attrs.asdict()
        block = (attrs.get("ome") or attrs).get("eubi_bridge") or attrs.get("eubi_bridge")
        block = {k: v for k, v in (block or {}).items() if k != "version"}
        out[store.name] = (block, np.asarray(group["0"]).tolist())
    return out


def test_scene_index_converts_as_series_index(tmp_path):
    """The old name in a script: a FutureWarning, and the very same output."""
    from eubi_bridge.ebridge import EuBIBridge
    src = _two_series_tiff(tmp_path / "two.ome.tif")
    results = {}
    for name in ("series_index", "scene_index"):
        out = tmp_path / name
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            EuBIBridge(configpath=str(tmp_path / f"cfg_{name}")).to_zarr(
                str(src), str(out), use_threading=True, verbose=False, **{name: 1})
        assert bool(_future(record)) == (name == "scene_index"), [str(w.message)
                                                                  for w in record]
        results[name] = _outputs(out)
    assert results["series_index"] == results["scene_index"]
    (name, (block, pixels)), = results["series_index"].items()
    assert name.endswith("_scene1.zarr")                         # the suffix stays (8.32)
    acquisition = block["acquisition"]
    assert acquisition["series_index"] == acquisition["scene_index"] == 1
    assert np.unique(pixels).tolist() == [9]


# -- the GUI's saved batches ------------------------------------------------------

def test_series_ui_reads_the_scene_keys_of_a_0_2_config():
    with pytest.warns(FutureWarning, match="sceneIndices"):
        reader = series_ui({"readAllScenes": False, "sceneIndices": "1"})
    assert (reader["readAllSeries"], reader["seriesIndices"]) == (False, "1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert series_ui(reader) is reader                        # both pairs: no warning


def test_gui_kwargs_from_a_0_2_snapshot():
    from eubi_bridge.qt_gui.workers.conversion_worker import _build_kwargs
    with pytest.warns(FutureWarning):
        kwargs = _build_kwargs({"reader": {"readAllScenes": False, "sceneIndices": "2"}})
    assert kwargs["series_index"] == 2 and "scene_index" not in kwargs


def test_a_0_2_batch_loads_with_one_warning_each(tmp_path):
    """batch.csv with a scene_index column and batch_config.json with sceneIndices /
    readAllScenes, as eubi-bridge 0.2.x saved them."""
    from eubi_bridge.qt_gui.core.batch import (CONFIG_SNAPSHOT_NAME, BatchModel,
                                               config_snapshot_path)
    folder = tmp_path / "batch"
    folder.mkdir()
    (folder / "batch.csv").write_text("input_path,output_path,scene_index\n"
                                      "a.tif,out,1\nb.tif,out,all\n")
    (folder / CONFIG_SNAPSHOT_NAME).write_text(json.dumps(
        {"reader": {"readAllScenes": False, "sceneIndices": "0", "readAllTiles": True}}))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model = BatchModel.load(str(folder / "batch.csv"))
    messages = [str(w.message) for w in _future(record)]
    assert len(messages) == 2, messages
    assert any(MESSAGE in m for m in messages) and any("sceneIndices" in m for m in messages)
    assert [row["series_index"] for row in model._rows] == ["1", "all"]
    assert all("scene_index" not in row for row in model._rows)
    assert model._base_config["reader"]["seriesIndices"] == "0"

    model.save(str(folder / "batch.csv"))                         # written back: both pairs
    with open(folder / "batch.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [(r["series_index"], r["scene_index"]) for r in rows] == [("1", "1"),
                                                                    ("all", "all")]
    # saved as the batch's own settings file (eubi-bridge 0.3.3)
    reader = json.loads(open(config_snapshot_path(str(folder / "batch.csv"))).read())["reader"]
    assert (reader["seriesIndices"], reader["sceneIndices"]) == ("0", "0")
    assert (reader["readAllSeries"], reader["readAllScenes"]) == (False, False)


def test_a_saved_batch_holds_both_columns(tmp_path):
    """Plan 8.32: a batch.csv eubi-bridge 0.3.0 saved has series_index and scene_index,
    equal, so eubi-bridge 0.2.x converts the same series; it loads back silently, and a
    row whose two cells differ (0.2.x changed its own) warns, naming the row."""
    from eubi_bridge.qt_gui.core.batch import BatchModel
    folder = tmp_path / "batch"
    folder.mkdir()
    (folder / "batch.csv").write_text("input_path,output_path,series_index\n"
                                      "a.tif,out,1\nb.tif,out,\n")
    model = BatchModel.load(str(folder / "batch.csv"))
    path = model.save(str(folder / "batch.csv"))
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        header, rows = reader.fieldnames, list(reader)
    assert header.index("scene_index") == header.index("series_index") + 1
    assert [(r["series_index"], r["scene_index"]) for r in rows] == [("1", "1"), ("", "")]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        again = BatchModel.load(path)
    assert [row.get("series_index") for row in again._rows] == ["1", None]
    assert all("scene_index" not in row for row in again._rows)

    with open(path, encoding="utf-8", newline="") as fh:
        text = fh.read()
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text.replace(",1,1", ",1,3", 1))
    with pytest.warns(SeriesIndexConflictWarning,
                      match=r"row 1: scene_index='3' and series_index='1' differ"
                            r".*it writes series_index='1' under both names: "
                            r"scene_index='3' is then replaced"):
        again = BatchModel.load(path)
    assert again._rows[0]["series_index"] == "1"
    again.save(path)                                              # as the warning said
    with open(path, newline="") as fh:
        first = next(csv.DictReader(fh))
    assert (first["series_index"], first["scene_index"]) == ("1", "1")


def test_a_batch_config_whose_pairs_differ_is_saved_as_it_is(tmp_path):
    """batch_config.json is written back as loaded: pairs that differ stay so, and the
    warning says that (not that a save settles them)."""
    from eubi_bridge.qt_gui.core.batch import (CONFIG_SNAPSHOT_NAME, BatchModel,
                                               config_snapshot_path)
    folder = tmp_path / "batch"
    folder.mkdir()
    (folder / "batch.csv").write_text("input_path,output_path,series_index,scene_index\n"
                                      "a.tif,out,1,1\n")
    reader = {"readAllSeries": False, "seriesIndices": "1",
              "readAllScenes": False, "sceneIndices": "2"}
    (folder / CONFIG_SNAPSHOT_NAME).write_text(json.dumps({"reader": reader}))
    with pytest.warns(SeriesIndexConflictWarning,
                      match=r"scene_index='2' and series_index='1' differ; series_index='1'"
                            r" is used.*Saving it again in eubi-bridge 0\.3\.0 keeps both "
                            r"values as they are"):
        model = BatchModel.load(str(folder / "batch.csv"))
    with warnings.catch_warnings():
        # the save reads the rows' settings from the snapshot again (_build_kwargs),
        # which warns once more, without the save sentence
        warnings.simplefilter("ignore", SeriesIndexConflictWarning)
        model.save(str(folder / "batch.csv"))
    assert json.loads(open(config_snapshot_path(str(folder / "batch.csv"))).read())[
        "reader"] == reader


def test_series_ui_pairs_that_differ_warn():
    reader = {"readAllScenes": True, "sceneIndices": "", "readAllSeries": False,
              "seriesIndices": "3"}
    with pytest.warns(SeriesIndexConflictWarning,
                      match="scene_index='all' and series_index='3'"):
        assert series_ui(reader) is reader
    from eubi_bridge.qt_gui.workers.conversion_worker import _build_kwargs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SeriesIndexConflictWarning)
        assert _build_kwargs({"reader": reader})["series_index"] == 3


def test_the_gui_config_writes_both_pairs():
    from eubi_bridge.qt_gui.server.config_manager import _config_to_react, _react_to_config
    react = _config_to_react({"readers": {"series_index": 2}})
    assert (react["reader"]["seriesIndices"], react["reader"]["sceneIndices"]) == ("2", "2")
    react["reader"]["readAllSeries"] = react["reader"]["readAllScenes"] = False
    assert _react_to_config(react)["readers"]["series_index"] == 2
    with pytest.warns(FutureWarning):
        old = _react_to_config({"reader": {"readAllScenes": False, "sceneIndices": "3"}})
    assert old["readers"]["series_index"] == 3
