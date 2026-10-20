"""Add "series_index" beside every "scene_index" in the golden files (eubi-bridge 0.3.0).

From 0.3.0 the eubi_bridge block of an output holds both names of the series index, with
the same value (plan 8.32: both written for several releases). The goldens of
tests/golden_0_1/ were recorded on eubi-bridge 0.1.x and cannot be recorded again
(tests/_golden.py), so the new key is added to them by this script instead:

    python tools/add_series_index_to_goldens.py          # rewrite the files
    python tools/add_series_index_to_goldens.py --check  # only say what would change

Each "scene_index": N line gets a "series_index": N line after it, with the same
indentation and line ending (the files are written with sorted keys, and series_index
sorts right after scene_index). Nothing else in a file changes. A file that has
series_index already is left as it is. Every rewritten file is parsed again and must
equal the old one plus the new keys.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

GOLDEN = Path(__file__).resolve().parents[1] / "tests" / "golden_0_1"

#: one "scene_index": <value> line, its comma and its line ending
LINE = re.compile(r'^(?P<indent>[ \t]*)"scene_index": (?P<value>[^,\r\n]+)(?P<comma>,?)'
                  r'(?P<eol>\r?\n|$)', re.MULTILINE)


def _with_series_index(text: str) -> str:
    newline = "\r\n" if "\r\n" in text else "\n"

    def add(m: re.Match) -> str:
        indent, value, comma, eol = m["indent"], m["value"], m["comma"], m["eol"]
        # the new key sorts after scene_index: when scene_index was last in its
        # object, it now takes a comma and series_index is last
        return (f'{indent}"scene_index": {value},{eol or newline}'
                f'{indent}"series_index": {value}{comma}{eol}')
    return LINE.sub(add, text)


def _added(old, new) -> bool:
    """True when `new` is `old` with a "series_index" equal to "scene_index" added to
    every dict that has scene_index, and nothing else changed."""
    if isinstance(old, dict):
        if not isinstance(new, dict):
            return False
        new = dict(new)
        if "scene_index" in old:
            if new.pop("series_index", object()) != old["scene_index"]:
                return False
        return new.keys() == old.keys() and all(_added(old[k], new[k]) for k in old)
    if isinstance(old, list):
        return (isinstance(new, list) and len(old) == len(new)
                and all(_added(o, n) for o, n in zip(old, new)))
    return old == new


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="change nothing")
    args = parser.parse_args(argv)
    changed = 0
    for path in sorted(GOLDEN.glob("*.json")):
        raw = path.read_bytes().decode("utf-8")
        if '"scene_index"' not in raw or '"series_index"' in raw:
            continue
        text = _with_series_index(raw)
        if not _added(json.loads(raw), json.loads(text)):
            print(f"{path.name}: the rewrite would change more than series_index",
                  file=sys.stderr)
            return 1
        changed += 1
        print(f"{path.name}: {raw.count(chr(34) + 'scene_index' + chr(34))} series_index added")
        if not args.check:
            path.write_bytes(text.encode("utf-8"))
    print(f"{changed} file(s) {'would change' if args.check else 'changed'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
