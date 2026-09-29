"""Selecting a scene or tile that does not exist must fail loudly.

An out-of-range index used to be dropped with a warning.  With nothing left to
convert the run then wrote no output and still reported success, which is
harder to diagnose than an error -- especially in a batch, where a row can
quietly produce nothing.  On some earlier versions the same empty selection
surfaced instead as ``Call load_scenes() before load_tiles()``, blaming the
wrong function entirely.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

@pytest.fixture
def single_scene_tiff(tmp_path):
    """A plain single-scene stack: valid scene index is 0 and nothing else."""
    path = tmp_path / "img.tif"
    tifffile.imwrite(
        path, np.random.randint(0, 255, (4, 16, 16)).astype("uint8"),
        imagej=True, metadata={"axes": "ZYX"})
    return path


_SCRIPT = """
import sys
sys.path.insert(0, r"{root}")
from eubi_bridge.ebridge import EuBIBridge
def main():
    EuBIBridge().to_zarr(r"{src}", r"{out}", verbose=False, {kwargs})
if __name__ == "__main__":
    main()
"""


def _convert(path, out, **kwargs):
    """Run a conversion in its own process and return (returncode, output).

    Conversion spawns worker processes, so it is driven through a real script
    rather than in-process: importing it under pytest on Windows leaves the
    workers unable to re-import ``__main__``.
    """
    script = path.parent / "run_conv.py"
    script.write_text(_SCRIPT.format(
        root=_ROOT, src=path, out=out,
        kwargs=", ".join(f"{k}={v!r}" for k, v in kwargs.items())))
    proc = subprocess.run([sys.executable, str(script)],
                          capture_output=True, text=True, timeout=600)
    return proc.returncode, proc.stdout + proc.stderr


class TestSceneIndexValidation:
    def test_out_of_range_scene_fails(self, single_scene_tiff, tmp_path):
        code, output = _convert(single_scene_tiff, tmp_path / "out", scene_index=5)
        assert code != 0
        assert "No valid scene index" in output

    def test_it_does_not_silently_write_nothing(self, single_scene_tiff, tmp_path):
        """The regression: this used to exit 0 having converted nothing."""
        out = tmp_path / "out"
        code, _ = _convert(single_scene_tiff, out, scene_index=5)
        assert code != 0, "conversion reported success without writing output"
        assert not out.exists() or not list(out.iterdir())

    def test_the_message_says_what_is_valid(self, single_scene_tiff, tmp_path):
        """The user needs the real count to correct their input."""
        _, output = _convert(single_scene_tiff, tmp_path / "out", scene_index=5)
        assert "img.tif" in output
        assert "1 scene(s)" in output
        assert "0..0" in output

    def test_negative_scene_index_is_rejected(self, single_scene_tiff, tmp_path):
        """Negative indices would otherwise index from the end of the list."""
        code, output = _convert(single_scene_tiff, tmp_path / "out", scene_index=-1)
        assert code != 0
        assert "No valid scene index" in output

    def test_valid_scene_still_converts(self, single_scene_tiff, tmp_path):
        out = tmp_path / "out"
        code, output = _convert(single_scene_tiff, out, scene_index=0)
        assert code == 0, output
        assert list(out.iterdir())


class TestTileIndexValidation:
    def test_out_of_range_tile_fails(self, single_scene_tiff, tmp_path):
        code, output = _convert(single_scene_tiff, tmp_path / "out",
                                scene_index=0, mosaic_tile_index=9)
        assert code != 0
        assert "No valid mosaic tile index" in output

    def test_the_message_mentions_the_mosaic_condition(self, single_scene_tiff, tmp_path):
        """Tiles exist only for an unstitched mosaic, the usual reason for this."""
        _, output = _convert(single_scene_tiff, tmp_path / "out",
                             scene_index=0, mosaic_tile_index=9)
        assert "as_mosaic" in output
