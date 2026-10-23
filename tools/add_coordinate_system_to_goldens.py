"""Add the coordinate-system keys to the golden files whose outputs now carry them
(spatial plan 9.18).

From the eubi release with coordinate systems, an image micro-reader places in a system is
written with the custom metadata ``eubi:coordinate_system`` (the system's id) and
``eubi:coordinate_systems`` (its definition). The goldens of tests/golden_0_1/ were
recorded on eubi-bridge 0.1.x and cannot be recorded again (tests/_golden.py), so the keys
are added to them by this script instead:

    python tools/add_coordinate_system_to_goldens.py          # rewrite the files
    python tools/add_coordinate_system_to_goldens.py --check  # only say what would change

Which stores are placed depends on the reader (an image without a stage position is not),
so each case is converted again with the tests' own inputs and settings. A golden is
rewritten only when every store of the new output equals the golden's plus exactly these
two keys, with the same levels (shapes, dtypes, chunks, pixel hashes); anything else
stops the script. Files are written in their own format (sorted keys, indent 1, CRLF),
checked by writing the old content back first and comparing bytes. Run it with the
micro-reader and ome_zarr_pyramid releases eubi requires, and an empty EUBI_CONFIG_DIR.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KEYS = ("eubi:coordinate_system", "eubi:coordinate_systems")


def _cases():
    """{golden name: function(tmp) -> output folder}, the conversions of the goldens that
    gain the keys (tests/test_pyramid_path.py)."""
    import numpy as np

    from tests import test_pyramid_path as T

    def run(make, filename, args):
        def go(tmp: Path) -> Path:
            T.RNG = np.random.default_rng(7)               # as the tests' autouse fixture
            T._convert(make(tmp / filename), tmp / "new", args)
            return tmp / "new"
        return go

    out = {f"pyramid_path-ome_tiff-{c}": run(T._ome_tiff, "img.ome.tif", T.CASES[c])
           for c in ("defaults", "settings", "auto_windows")}
    out["pyramid_path-czi_mosaic-tiles"] = run(T._mosaic_czi, "mosaic.czi", T.ALL)
    out["pyramid_path-czi_mosaic-stitched"] = run(T._mosaic_czi, "mosaic.czi",
                                                  ["--series_index", "all", "--as_mosaic"])
    out["pyramid_path-czi_views-separate"] = run(T._views_czi, "views.czi",
                                                 ["--view_index", "all"])
    out["pyramid_path-czi_views-joined"] = run(T._views_czi, "views.czi",
                                               ["--view_index", "all", "--concat_views"])
    return out


def _dump(value) -> bytes:
    return (json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False) + "\n").replace(
        "\n", "\r\n").encode("utf-8")


def main(argv=None) -> int:
    from tests import _golden

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="only say what would change")
    a = ap.parse_args(argv)
    for name, convert in _cases().items():
        path = _golden.GOLDEN / f"{name}.json"
        raw = path.read_bytes()
        old = json.loads(raw)
        if _dump(old) != raw and _dump(old).rstrip(b"\r\n") != raw.rstrip(b"\r\n"):
            raise SystemExit(f"{path.name}: not in the format this script writes; stopped")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            actual = _golden.describe(convert(tmp), tmp)
        # compared as the tests compare (paths under <tmp> with '/', _golden.load);
        # written as the file holds it
        expected = _golden.load(name)
        if sorted(actual) != sorted(expected):
            raise SystemExit(f"{name}: stores {sorted(actual)} != {sorted(expected)}")
        new, added = json.loads(json.dumps(old)), 0
        for store, entry in actual.items():
            attrs = dict(entry["attrs"])
            extra = {k: attrs.pop(k) for k in KEYS if k in attrs}
            if attrs != expected[store]["attrs"] or entry["levels"] != expected[store]["levels"]:
                raise SystemExit(f"{name}/{store}: differs from the golden in more than "
                                 f"{list(KEYS)}; stopped")
            new[store]["attrs"].update(extra)
            added += bool(extra)
        print(f"{name}: {added} of {len(new)} store(s) gain {list(KEYS)}")
        if not a.check and added:
            data = _dump(new) if raw.endswith(b"\n") else _dump(new).rstrip(b"\r\n")
            path.write_bytes(data)
            if json.loads(path.read_bytes()) != new:
                raise SystemExit(f"{path.name}: written file does not read back")
    return 0


if __name__ == "__main__":
    sys.exit(main())
