"""
File system service for the Qt GUI.

Provides local directory listing and S3 listing with OME-Zarr detection,
mirroring the logic in routes.ts.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict


class FileEntry(TypedDict):
    name: str
    path: str
    isDirectory: bool
    isOmeZarr: bool


PAGE_SIZE = 50


def is_remote(path: str) -> bool:
    return path.startswith(("s3://", "gs://", "http://", "https://"))


def expand_path(path: str) -> str:
    """Expand ~ and resolve the path."""
    return str(Path(path.replace("\\", "/")).expanduser().resolve())


def _is_ome_zarr_local(folder_path: str) -> bool:
    try:
        return (
            os.path.exists(os.path.join(folder_path, ".zattrs"))
            or os.path.exists(os.path.join(folder_path, "zarr.json"))
        )
    except OSError:
        return False


def list_local(path: str) -> list[FileEntry]:
    """
    List contents of a local directory.
    Returns entries sorted: directories first, then files, both alphabetically.
    """
    resolved = expand_path(path)
    if not os.path.isdir(resolved):
        return []

    entries: list[FileEntry] = []
    try:
        with os.scandir(resolved) as it:
            for entry in it:
                try:
                    is_dir = entry.is_dir(follow_symlinks=False)
                    is_zarr = _is_ome_zarr_local(entry.path) if is_dir else False
                    entries.append(
                        FileEntry(
                            name=entry.name,
                            path=entry.path,
                            isDirectory=is_dir,
                            isOmeZarr=is_zarr,
                        )
                    )
                except OSError:
                    pass
    except PermissionError:
        return []

    # Directories first, then files; alphabetical within each group (case-insensitive)
    entries.sort(key=lambda e: (not e["isDirectory"], e["name"].lower()))
    return entries


def list_local_recursive(
    root: str,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> list[FileEntry]:
    """Recursively walk *root* and return file entries that pass the filters.

    Each entry's ``name`` is the path relative to *root* (using forward slashes),
    and ``path`` is the absolute path.  Directories are never returned — only files.
    """
    import fnmatch

    resolved = expand_path(root)
    if not os.path.isdir(resolved):
        return []

    entries: list[FileEntry] = []
    for dirpath, dirnames, filenames in os.walk(resolved):
        # Skip hidden directories
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fname in sorted(filenames):
            if fname.startswith("."):
                continue
            abs_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(abs_path, resolved).replace("\\", "/")
            # Match patterns against both the bare filename AND the relative path
            # so that e.g. `*subfolder*` or `parent/*.tif` work across all depths.
            if include_patterns and not any(
                fnmatch.fnmatch(fname, p) or fnmatch.fnmatch(rel_path, p)
                for p in include_patterns
            ):
                continue
            if exclude_patterns and any(
                fnmatch.fnmatch(fname, p) or fnmatch.fnmatch(rel_path, p)
                for p in exclude_patterns
            ):
                continue
            entries.append(FileEntry(
                name=rel_path,
                path=abs_path,
                isDirectory=False,
                isOmeZarr=False,
            ))
    return entries


def paginate(entries: list, page: int = 0, page_size: int = PAGE_SIZE) -> tuple[list, int]:
    """Return (page_entries, total_count)."""
    total = len(entries)
    start = page * page_size
    return entries[start : start + page_size], total


def list_s3(
    s3_path: str,
    page: int = 0,
    page_size: int = PAGE_SIZE,
) -> dict:
    """
    List an S3 (or GCS/HTTP) path, detecting OME-Zarr stores on the current page only.

    Returns:
        {
            "currentPath": str,
            "parentPath": str | None,
            "items": list[FileEntry],
            "total": int,
            "page": int,
            "pageSize": int,
        }
    """
    try:
        import s3fs  # type: ignore
    except ImportError:
        return {
            "currentPath": s3_path,
            "parentPath": None,
            "items": [],
            "total": 0,
            "page": page,
            "pageSize": page_size,
            "error": "s3fs not installed",
        }

    s3_path = s3_path.rstrip("/")
    # Strip scheme for s3fs
    if s3_path.startswith("s3://"):
        stripped = s3_path[5:]
    else:
        stripped = s3_path

    try:
        fs = s3fs.S3FileSystem(anon=False)
        raw = fs.ls(stripped, detail=True)
    except Exception as exc:
        return {
            "currentPath": s3_path,
            "parentPath": None,
            "items": [],
            "total": 0,
            "page": page,
            "pageSize": page_size,
            "error": str(exc),
        }

    all_candidates: list[dict] = []
    for entry in raw:
        name_key = entry.get("Key") or entry.get("name") or ""
        entry_name = name_key.rstrip("/").split("/")[-1]
        is_dir = entry.get("type") == "directory" or name_key.endswith("/")
        all_candidates.append(
            {
                "name": entry_name,
                "s3key": name_key,
                "isDirectory": is_dir,
            }
        )

    all_candidates.sort(key=lambda e: (not e["isDirectory"], e["name"].lower()))
    total = len(all_candidates)
    page_candidates = all_candidates[page * page_size : (page + 1) * page_size]

    items: list[FileEntry] = []
    for c in page_candidates:
        full_path = f"s3://{c['s3key'].rstrip('/')}"
        is_zarr = False
        if c["isDirectory"]:
            # Lightweight probe: check for .zattrs or zarr.json
            try:
                for fname in [".zattrs", "zarr.json"]:
                    probe = f"{c['s3key'].rstrip('/')}/{fname}"
                    if fs.exists(probe):
                        is_zarr = True
                        break
            except Exception:
                pass
        items.append(
            FileEntry(
                name=c["name"],
                path=full_path,
                isDirectory=c["isDirectory"],
                isOmeZarr=is_zarr,
            )
        )

    # Compute parent path
    parent_path: str | None = None
    if stripped and "/" in stripped:
        parent_stripped = "/".join(stripped.split("/")[:-1])
        parent_path = f"s3://{parent_stripped}" if parent_stripped else None

    return {
        "currentPath": s3_path,
        "parentPath": parent_path,
        "items": items,
        "total": total,
        "page": page,
        "pageSize": page_size,
    }


def get_parent(path: str) -> str | None:
    """Return parent directory of *path*, or None if already at root."""
    if is_remote(path):
        stripped = path.split("://", 1)[1].rstrip("/")
        if "/" not in stripped:
            return None
        scheme = path.split("://")[0]
        parent = "/".join(stripped.split("/")[:-1])
        return f"{scheme}://{parent}" if parent else None
    resolved = expand_path(path)
    parent = os.path.dirname(resolved)
    if parent == resolved:
        return None
    return parent
