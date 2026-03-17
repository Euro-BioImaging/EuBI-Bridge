"""
Convert a LabID-export CSV (one row per channel per file) to an EuBI-Bridge
batch-conversion CSV (one row per file).

Usage
-----
    python labid_to_eubi_csv.py source.csv output.csv /path/to/zarr/output

Arguments
---------
source_csv   : Path to the LabID export CSV (source format).
output_csv   : Path where the EuBI-Bridge CSV will be written.
output_dir   : Zarr output directory written into the output_path column.

Optional flags
--------------
--scene_index        : Value for the scene_index column (default: all)
--mosaic_tile_index  : Value for the mosaic_tile_index column (default: all)
"""

import argparse
import csv
import sys
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert LabID export CSV to EuBI-Bridge batch CSV."
    )
    parser.add_argument("source_csv", help="Input LabID CSV file.")
    parser.add_argument("output_csv", help="Output EuBI-Bridge CSV file.")
    parser.add_argument("output_dir", help="Zarr output directory for output_path column.")
    parser.add_argument("--scene_index", default="all", help="Value for scene_index (default: all).")
    parser.add_argument("--mosaic_tile_index", default="all", help="Value for mosaic_tile_index (default: all).")
    return parser.parse_args()


def convert(source_csv: str, output_csv: str, output_dir: str,
            scene_index: str = "all", mosaic_tile_index: str = "all"):
    # filepath -> list of (channel_number, channel_name)
    file_channels: dict = defaultdict(list)
    # Preserve insertion order of filepaths
    filepath_order: list = []

    with open(source_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            filepath = row["FilePath"].strip()
            try:
                ch_num = int(row["Channel Number"])
            except (KeyError, ValueError):
                print(f"WARNING: skipping row with invalid 'Channel Number': {row}", file=sys.stderr)
                continue
            ch_name = row.get("Channel Name", "").strip()

            if filepath not in file_channels:
                filepath_order.append(filepath)
            file_channels[filepath].append((ch_num, ch_name))

    rows = []
    for filepath in filepath_order:
        channels = file_channels[filepath]
        # Sort by channel number (source is 1-based), convert to 0-based index
        channels.sort(key=lambda x: x[0])
        labels = ";".join(
            f"{ch_num - 1},{ch_name}" for ch_num, ch_name in channels
        )
        rows.append({
            "input_path": filepath,
            "output_path": output_dir,
            "channel_labels": labels,
            "scene_index": scene_index,
            "mosaic_tile_index": mosaic_tile_index,
        })

    with open(output_csv, "w", newline="", encoding="utf-8") as fh:
        fieldnames = ["input_path", "output_path", "channel_labels", "scene_index", "mosaic_tile_index"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    args = parse_args()
    convert(
        source_csv=args.source_csv,
        output_csv=args.output_csv,
        output_dir=args.output_dir,
        scene_index=args.scene_index,
        mosaic_tile_index=args.mosaic_tile_index,
    )
