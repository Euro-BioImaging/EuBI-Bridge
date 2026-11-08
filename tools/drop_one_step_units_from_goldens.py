"""Drop the unit of a one-step t or z from the golden files (eubi-bridge 0.3.4, ozp 0.6.0).

From ome_zarr_pyramid 0.6.0 an axis the file gives no size for has no unit, a t or z of one
step included (tenth review N-08: with ``--squeeze False`` they kept ``second`` and
``micrometer``). The goldens of tests/golden_0_1/ were recorded on eubi-bridge 0.1.x and cannot
be recorded again (tests/_golden.py), so the unit is taken out of them by this script:

    python tools/drop_one_step_units_from_goldens.py          # rewrite the files
    python tools/drop_one_step_units_from_goldens.py --check  # only say what would change

Only the axes listed in ONE_STEP lose their unit, each checked first: it is t or z, its unit
is ozp's default (second, micrometer), and where the golden holds the level shapes, its size
is 1. Every rewritten file is parsed again and must equal the old one less those keys. The
files keep their form (indent 1, sorted keys, no newline at the end).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

GOLDEN = Path(__file__).resolve().parents[1] / "tests" / "golden_0_1"
DEFAULT = {"t": "second", "z": "micrometer"}
#: golden -> [(store, or None for a golden of one store's attributes, axis)], the units the
#: comparison tests found that 0.3.4 no longer writes (and nothing else)
ONE_STEP = {
    "metadata_path-channel_meta-v2": [(None, "t")],
    "metadata_path-channel_meta-v3": [(None, "t")],
    "metadata_path-from_array_windows-v2": [(None, "t")],
    "metadata_path-from_array_windows-v3": [(None, "t")],
    "pyramid_path-ome_tiff-v05_no_squeeze": [("img_scene1.zarr", "t"), ("img_scene1.zarr", "z")],
}


def _axes(attrs: dict) -> list:
    return attrs.get("ome", attrs)["multiscales"][0]["axes"]


def dropped(name: str, golden: dict) -> dict:
    new = copy.deepcopy(golden)
    for store, axis in ONE_STEP[name]:
        holder = new if store is None else new[store]
        axes = _axes(holder["attrs"])
        names = [a["name"] for a in axes]
        entry = axes[names.index(axis)]
        assert entry.get("unit") == DEFAULT[axis], (name, store, axis, entry)
        if store is not None:
            size = holder["levels"][0][0][names.index(axis)]
            assert size == 1, (name, store, axis, size)
        del entry["unit"]
    return new


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="only say what would change")
    args = ap.parse_args()
    for name in ONE_STEP:
        path = GOLDEN / f"{name}.json"
        text = path.read_text(encoding="utf-8")
        old = json.loads(text)
        if all(axis_unit_gone(old, store, axis) for store, axis in ONE_STEP[name]):
            print(f"{name}: done already")
            continue
        new = dropped(name, old)
        out = json.dumps(new, indent=1, sort_keys=True)
        assert json.loads(out) == new
        print(f"{name}: {len(ONE_STEP[name])} unit(s) dropped")
        if not args.check:
            path.write_text(out, encoding="utf-8")
    return 0


def axis_unit_gone(golden: dict, store, axis) -> bool:
    holder = golden if store is None else golden[store]
    axes = _axes(holder["attrs"])
    return "unit" not in axes[[a["name"] for a in axes].index(axis)]


if __name__ == "__main__":
    sys.exit(main())
