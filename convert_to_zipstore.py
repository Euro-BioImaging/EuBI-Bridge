"""
Convert OME-Zarr directories to ZipStore-compatible .zip archives.

Usage:
    python convert_to_zipstore.py INPUT [INPUT ...] [OPTIONS]

Positional arguments:
    INPUT           One or more paths OR glob patterns pointing to OME-Zarr
                    directories (or their parent directories).
                    Examples:
                      C:/data/pffzarr1                   # scan directory
                      "C:/data/pffzarr1/*.zarr"          # glob (quote on Unix)
                      C:/data/a.zarr C:/data/b.zarr      # explicit list

Options:
    -o / --output   Output directory for .zip files.
                    Default: same directory as each source zarr.
    -w / --workers  Number of zarr stores to zip in parallel [default: 4].
    -t / --threads  I/O reader threads per worker process [default: 8].
    --delete        Delete the original directory after successful zipping.
    --read-mb       Memory budget (MB) for in-flight read buffer per worker
                    [default: 512].
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Empty, Queue
from threading import Thread


# ── helpers ────────────────────────────────────────────────────────────────────

def is_ome_zarr(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (
        (path / ".zattrs").exists()
        or (path / ".zgroup").exists()
        or (path / "zarr.json").exists()
    )


def resolve_inputs(patterns: list[str]) -> list[Path]:
    """
    Expand glob patterns and plain paths into a deduplicated list of
    top-level OME-Zarr directories.
    """
    candidates: list[Path] = []
    for pat in patterns:
        # Try as-is first (literal path or glob)
        expanded = glob.glob(pat, recursive=True)
        if expanded:
            for p in expanded:
                candidates.append(Path(p))
        else:
            candidates.append(Path(pat))

    zarr_dirs: list[Path] = []
    seen: set[Path] = set()
    for p in candidates:
        if is_ome_zarr(p):
            if p not in seen:
                zarr_dirs.append(p)
                seen.add(p)
        elif p.is_dir():
            # Treat p as a parent — scan one level deep
            for child in sorted(p.iterdir()):
                if is_ome_zarr(child) and child not in seen:
                    zarr_dirs.append(child)
                    seen.add(child)
    return zarr_dirs


# ── per-file reader (runs in thread pool) ──────────────────────────────────────

def _read_file(args: tuple[Path, Path]) -> tuple[str, bytes]:
    """Read one file and return (arcname_str, bytes)."""
    file_path, src = args
    return (str(file_path.relative_to(src).as_posix()), file_path.read_bytes())


# ── main worker (runs in a subprocess via ProcessPoolExecutor) ─────────────────

def _zip_worker(
    src_str: str,
    dst_str: str,
    io_threads: int,
    read_mb: int,
    delete_original: bool,
) -> tuple[bool, str]:
    """
    Zip one zarr directory.  Runs in a worker process.

    Strategy:
    - A ThreadPoolExecutor reads chunk files concurrently (I/O parallelism).
    - A producer thread feeds a bounded queue so memory stays capped.
    - The main loop drains the queue and writes to ZipFile sequentially
      (zipfile is not thread-safe for writes).
    """
    src = Path(src_str)
    dst = Path(dst_str)

    try:
        all_files = sorted(p for p in src.rglob("*") if p.is_file())
        n = len(all_files)
        if n == 0:
            print(f"[{src.name}] No files found — skipping", flush=True)
            return (True, src_str)

        # Estimate average file size to set queue depth
        sample_size = all_files[0].stat().st_size if all_files else 4096
        read_bytes = read_mb * 1024 * 1024
        queue_depth = max(4, min(512, read_bytes // max(1, sample_size)))

        q: Queue = Queue(maxsize=int(queue_depth))
        errors: list[Exception] = []

        def _producer():
            with ThreadPoolExecutor(max_workers=io_threads) as pool:
                futures = {
                    pool.submit(_read_file, (fp, src)): fp for fp in all_files
                }
                for fut in as_completed(futures):
                    try:
                        q.put(fut.result())
                    except Exception as exc:
                        errors.append(exc)
            q.put(None)  # sentinel

        producer = Thread(target=_producer, daemon=True)
        producer.start()

        t0 = time.time()
        print(f"[{src.name}] Zipping {n} files → {dst.name}", flush=True)
        written = 0
        with zipfile.ZipFile(str(dst), mode="w",
                             compression=zipfile.ZIP_STORED,
                             allowZip64=True) as zf:
            while True:
                item = q.get()
                if item is None:
                    break
                arcname, data = item
                zf.writestr(arcname, data)
                written += 1
                if written % 500 == 0 or written == n:
                    pct = 100 * written / n
                    print(f"  [{src.name}] {written}/{n} ({pct:.0f}%)", flush=True)

        producer.join()

        if errors:
            raise errors[0]

        elapsed = time.time() - t0
        size_mb = dst.stat().st_size / 1_048_576
        print(
            f"[{src.name}] Done  {size_mb:.1f} MB  in {elapsed:.1f}s", flush=True
        )

        if delete_original:
            shutil.rmtree(src)
            print(f"[{src.name}] Original directory removed.", flush=True)

        return (True, src_str)

    except Exception as exc:
        print(f"[{src.name}] FAILED: {exc}", file=sys.stderr, flush=True)
        if dst.exists():
            dst.unlink(missing_ok=True)
        return (False, src_str)


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert OME-Zarr directories to ZipStore .zip archives.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help="Path(s) or glob pattern(s) to OME-Zarr directories (or parent dirs).",
    )
    p.add_argument(
        "-o", "--output",
        metavar="DIR",
        default=None,
        help="Output directory for .zip files (default: alongside each source).",
    )
    p.add_argument(
        "-w", "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Parallel worker processes (one per zarr). Default: 4.",
    )
    p.add_argument(
        "-t", "--threads",
        type=int,
        default=8,
        metavar="N",
        help="I/O reader threads per worker. Default: 8.",
    )
    p.add_argument(
        "--delete",
        action="store_true",
        help="Delete original zarr directory after successful zipping.",
    )
    p.add_argument(
        "--read-mb",
        type=int,
        default=512,
        metavar="MB",
        help="In-flight read-buffer size per worker in MB. Default: 512.",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    zarr_dirs = resolve_inputs(args.inputs)
    if not zarr_dirs:
        print("No OME-Zarr directories found for the given inputs.", file=sys.stderr)
        sys.exit(1)

    out_root = Path(args.output) if args.output else None
    if out_root:
        out_root.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(zarr_dirs)} OME-Zarr(s).  Workers: {args.workers}  Threads/worker: {args.threads}\n")

    jobs = []
    for zarr_dir in zarr_dirs:
        dst_dir = out_root if out_root else zarr_dir.parent
        zip_path = dst_dir / (zarr_dir.name + ".zip")
        jobs.append((str(zarr_dir), str(zip_path)))

    t_total = time.time()
    failed: list[str] = []

    n_workers = min(args.workers, len(jobs))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                _zip_worker,
                src,
                dst,
                args.threads,
                args.read_mb,
                args.delete,
            ): src
            for src, dst in jobs
        }
        for fut in as_completed(futures):
            ok, src_str = fut.result()
            if not ok:
                failed.append(src_str)

    elapsed = time.time() - t_total
    print(f"\nAll done in {elapsed:.1f}s.  "
          f"{len(jobs) - len(failed)}/{len(jobs)} succeeded.")
    if failed:
        print("Failed:", *failed, sep="\n  ", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
