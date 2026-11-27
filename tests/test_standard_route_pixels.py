"""Wrong pixels on the standard reader route (pixel_reader='standard', or the fallback
when micro-reader declines a file), Workplan E31 and E9.

E31: bioio-lif 1.1.0 (readlif 0.6.5) mixed up tiles and time points in LIFs with
both (XYCST.lif: 18 of 24 planes wrong); bioio-lif 2 and later read through
liffile and match micro-reader on every image.

E9: libCZI (pylibCZIrw 6.1.0, aicspylibczi) decodes 16-bit RGB (Bgr48) zstd
subblocks stored hi-lo packed wrongly; Gray16 ones correctly.  Such files are read
by micro-reader even with pixel_reader='standard', or refused when micro-reader
cannot take them.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from tests.conftest import requires_full  # noqa: E402

DATA = Path(os.environ.get("MICROSCOPY_DATA") or os.environ.get("EUBI_TEST_DATA")
            or "C:/Users/oezdemir/Desktop/data/microscopy")
LIF = DATA / "formats" / "lif"
HILO_CZI = DATA / "formats" / "czi" / "os-zeiss" / "Zeiss-5-SlidePreview-Zstd1-HiLo.czi"

#: LIFs bioio-lif 1.1.0 read wrongly (all their scenes now match micro-reader), and
#: two it read right (unchanged)
LIF_FILES = [
    "ome-leica-xlef-lif/dimension tests LIFs/XYCST.lif",        # tiles over time: was 18/24 wrong
    "ome-leica-xlef-lif/dimension tests LIFs/XYCZST.lif",       # was 90/120 wrong
    "ome-samples/PR2729_frameOrderCombinedScanTypes.lif",       # was wrong
    "ome-samples/Project007.lif",                               # 2 of 4 scenes were wrong
    "ome-leica-xlef-lif/dimension tests LIFs/XYS.lif",          # tiles, no time: always right
    "ome-leica-xlef-lif/dimension tests LIFs/XYZCS.lif",
]


# -- E31: LIF through bioio-lif 2+ --------------------------------------------------------

@requires_full
@pytest.mark.realdata
@pytest.mark.parametrize("name", LIF_FILES)
def test_standard_lif_route_matches_micro_reader(name):
    path = LIF / name
    if not path.exists():
        pytest.skip(f"{path} not available")
    micro_reader = pytest.importorskip("micro_reader")
    from eubi_bridge.core.readers import read_image_sync
    reader = read_image_sync(str(path), pixel_reader="standard")
    # the standard LIF route reads a tile scan's first tile only (bioio's M = 0;
    # n_tiles is 1): compared with micro-reader's tile 0 of each series
    reference = [im for im in micro_reader.open(str(path), tiles="separate").images
                 if im.index.get("tile", 0) == 0]
    assert reader.n_scenes == len(reference)
    for scene, image in enumerate(reference):
        reader.set_scene(scene)
        got = np.asarray(reader.get_image_dask_data())
        want = np.asarray(image)
        assert got.shape == want.shape, f"scene {scene}"
        wrong = sum(not np.array_equal(got[t, c, z], want[t, c, z])
                    for t in range(got.shape[0]) for c in range(got.shape[1])
                    for z in range(got.shape[2]))
        assert wrong == 0, f"scene {scene}: {wrong} planes differ from micro-reader"


def test_full_install_pins_bioio_lif_2_and_drops_readlif():
    import tomllib
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("no pyproject.toml next to the tests (an installed copy)")
    deps = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["dependencies"]
    lif = [d for d in deps if d.replace("_", "-").startswith("bioio-lif")]
    assert lif and ">=2.0.1" in lif[0], lif
    assert not [d for d in deps if d.startswith("readlif")]


# -- E9: 16-bit RGB hi-lo CZIs -------------------------------------------------------------

HILO = "zstd1:ExplicitLevel=1;PreProcess=HiLoByteUnpack"


def _write_czi(path, data, options):
    pyczi = pytest.importorskip("pylibCZIrw.czi")
    with pyczi.create_czi(str(path), exist_ok=True, compression_options=options) as w:
        w.write(data, plane={"C": 0, "T": 0, "Z": 0})
    return str(path)


@pytest.fixture
def rgb48():
    return np.random.default_rng(0).integers(0, 65535, (48, 72, 3), dtype=np.uint16)


@pytest.fixture
def gray16():
    return np.random.default_rng(1).integers(0, 65535, (48, 72), dtype=np.uint16)


@requires_full
@pytest.mark.parametrize("kind, options, expected", [
    ("rgb48", HILO, True),
    ("rgb48", "zstd1:ExplicitLevel=1", False),        # zstd1 without the packing
    ("rgb48", "zstd0:ExplicitLevel=1", False),
    ("rgb48", None, False),                           # uncompressed
    ("gray16", HILO, False),                          # libCZI reads Gray16 hi-lo right
])
def test_misread_by_libczi_detects_only_rgb48_hilo(tmp_path, request, kind, options, expected):
    from eubi_bridge.core.czi_hilo import misread_by_libczi
    path = _write_czi(tmp_path / "a.czi", request.getfixturevalue(kind), options)
    assert misread_by_libczi(path) is expected


def test_misread_by_libczi_is_false_for_what_is_no_czi(tmp_path):
    from eubi_bridge.core.czi_hilo import misread_by_libczi
    (tmp_path / "empty.czi").write_bytes(b"")
    (tmp_path / "text.czi").write_bytes(b"not a CZI at all" * 20)
    (tmp_path / "cut.czi").write_bytes(b"ZISRAWFILE" + b"\x00" * 50)
    assert not misread_by_libczi(str(tmp_path / "empty.czi"))
    assert not misread_by_libczi(str(tmp_path / "text.czi"))
    assert not misread_by_libczi(str(tmp_path / "cut.czi"))
    assert not misread_by_libczi(str(tmp_path / "missing.czi"))


@requires_full
def test_libczi_still_misreads_rgb48_hilo(tmp_path, rgb48):
    """Why E9's routing exists.  When this fails, libCZI reads these files right:
    check the released pylibCZIrw on the Zeiss-5 sample and drop the routing."""
    pyczi = pytest.importorskip("pylibCZIrw.czi")
    path = _write_czi(tmp_path / "a.czi", rgb48, HILO)
    with pyczi.open_czi(path) as r:
        back = r.read(plane={"C": 0, "T": 0, "Z": 0})
    assert not np.array_equal(back, rgb48)


@requires_full
def test_standard_route_reads_rgb48_hilo_with_micro_reader(tmp_path, rgb48, caplog):
    pytest.importorskip("micro_reader")
    from eubi_bridge.core.micro_source import MicroReader
    from eubi_bridge.core.readers import read_image_sync
    path = _write_czi(tmp_path / "a.czi", rgb48, HILO)
    reader = read_image_sync(path, pixel_reader="standard")
    assert isinstance(reader, MicroReader)
    got = np.asarray(reader.get_image_dask_data())            # T C Z Y X, C = R G B
    rgb = rgb48[..., ::-1]                                     # stored B G R
    np.testing.assert_array_equal(got[0, :, 0], np.moveaxis(rgb, -1, 0))


@requires_full
def test_standard_route_leaves_gray16_hilo_to_the_standard_reader(tmp_path, gray16):
    from eubi_bridge.core.micro_source import MicroReader
    from eubi_bridge.core.readers import read_image_sync
    path = _write_czi(tmp_path / "a.czi", gray16, HILO)
    reader = read_image_sync(path, pixel_reader="standard")
    assert not isinstance(reader, MicroReader)
    np.testing.assert_array_equal(np.asarray(reader.get_image_dask_data())[0, 0, 0], gray16)


@requires_full
@pytest.mark.parametrize("pixel_reader", ["standard", "micro"])
def test_rgb48_hilo_is_refused_when_micro_reader_declines(tmp_path, rgb48, monkeypatch,
                                                          pixel_reader):
    from eubi_bridge.core import micro_source
    from eubi_bridge.core.readers import read_image_sync
    from eubi_bridge.utils.errors import InputError
    path = _write_czi(tmp_path / "a.czi", rgb48, HILO)
    monkeypatch.setattr(micro_source.MicroReader, "open", classmethod(lambda cls, *a, **k: None))
    with pytest.raises(InputError, match="hi-lo"):
        read_image_sync(path, pixel_reader=pixel_reader)


@requires_full
@pytest.mark.realdata
@pytest.mark.skipif(not HILO_CZI.exists(), reason="Zeiss-5 hi-lo sample not available")
def test_real_rgb48_hilo_sample_reads_micro_readers_pixels():
    micro_reader = pytest.importorskip("micro_reader")
    from eubi_bridge.core.czi_hilo import misread_by_libczi
    from eubi_bridge.core.readers import read_image_sync
    assert misread_by_libczi(str(HILO_CZI))
    got = np.asarray(read_image_sync(str(HILO_CZI), pixel_reader="standard").get_image_dask_data())
    want = np.asarray(micro_reader.open(str(HILO_CZI)).images[0])
    np.testing.assert_array_equal(got, want)
