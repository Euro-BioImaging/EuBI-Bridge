"""Rename the stores of the golden outputs to the names eubi-bridge writes since decision E-3
(Workplan, 2026-11-09): ``X.ome.zarr``, where eubi-bridge 0.1.x - 0.3.x wrote ``X.zarr``.

Only the top-level keys of each ``tests/golden_0_1/<name>.json`` (the store names) change; a
store already named ``X.ome.zarr`` (a joined output named after .ome.tif files) keeps its
name, and nothing inside a store (attributes, levels) is touched. No re-recording. Run once:

    python tools/rename_golden_stores_ome_zarr.py
"""
import json
from pathlib import Path

GOLDEN = Path(__file__).resolve().parent.parent / "tests" / "golden_0_1"


def renamed(name: str) -> str:
    if name.endswith(".ome.zarr") or not name.endswith(".zarr"):
        return name
    return name[:-len(".zarr")] + ".ome.zarr"


def main() -> None:
    for path in sorted(GOLDEN.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        new = {renamed(k): v for k, v in data.items()}
        if list(new) == list(data):
            continue
        assert len(new) == len(data), f"{path.name}: two stores would share a name"
        path.write_text(json.dumps(new, indent=1, sort_keys=True), encoding="utf-8")
        print(f"{path.name}: {', '.join(f'{k} -> {renamed(k)}' for k in data if renamed(k) != k)}")


if __name__ == "__main__":
    main()
