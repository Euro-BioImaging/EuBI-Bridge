"""Which input files are skipped as not images (``path_utils._not_an_image``).

With only micro-reader (eubi-bridge-lite) every file it does not recognise used to count,
a damaged ``.tif`` too: the batch skipped it with a warning and finished, where the full
install keeps it and the conversion fails on it. A file with an image suffix is a damaged
image, never a side file."""
from __future__ import annotations

import pytest

pytest.importorskip("micro_reader")

from eubi_bridge.utils import capabilities, path_utils  # noqa: E402


@pytest.fixture(params=[False, True], ids=["lite", "full"])
def install(request, monkeypatch):
    monkeypatch.setattr(capabilities, "has_bioformats", lambda: request.param)
    return request.param


def test_a_damaged_image_is_kept_with_either_install(tmp_path, install):
    for name in ("broken.tif", "broken.czi", "broken.nd2", "broken.lif"):
        path = tmp_path / name
        path.write_bytes(b"not an image")
        assert path_utils._not_an_image(str(path)) is False


def test_a_side_file_is_skipped_with_either_install(tmp_path, install):
    for name in ("notes.txt", "readme.md", "table.csv", "report.pdf"):
        path = tmp_path / name
        path.write_text("text")
        assert path_utils._not_an_image(str(path)) is True


def test_an_unknown_suffix_is_skipped_only_by_the_lite_install(tmp_path, install):
    path = tmp_path / "something.xyz"
    path.write_bytes(b"\x00\x01\x02")
    assert path_utils._not_an_image(str(path)) is (not install)
