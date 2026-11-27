"""Whether a CZI stores 16-bit RGB subblocks as zstd with hi-lo byte packing.

ZEN can compress 16-bit data as "zstd1" with hi-lo packing: each subblock's high
bytes are stored before its low bytes, which compresses better.  libCZI (inside
pylibCZIrw 6.1.0 and aicspylibczi, eubi-bridge's standard CZI readers) undoes the
packing correctly for 16-bit grey (Gray16) but not for 16-bit RGB (Bgr48): every
pixel of such a subblock comes back wrong (the Zeiss-5 slide preview; also files
pylibCZIrw writes itself).  micro-reader reads both correctly (Workplan E9).

Only the file header, the subblock directory and the few bytes in front of some
subblocks' data are read; no pixels.  Layout as in micro-reader's CZI backend.
"""
from __future__ import annotations

import struct

MAGIC = b"ZISRAWFILE"
#: subblock compression code of zstd with a header (where the hi-lo flag sits)
ZSTD1 = 6
#: pixel types whose hi-lo packed subblocks libCZI misreads: Bgr48 (Gray16 is read
#: correctly)
MISREAD_PIXEL_TYPES = (4,)
#: subblocks whose header is checked, spread over the directory: the packing is a
#: setting of the acquisition, so every subblock of one kind shares it
SAMPLES = 8


def _directory(fh) -> list:
    """(pixel type, file position, compression) of every subblock directory entry,
    or [] when the file is not a CZI the check understands (the readers then say why)."""
    head = fh.read(32 + 80)
    if head[:10] != MAGIC or len(head) < 32 + 80:
        return []
    directory_position, = struct.unpack_from("<q", head, 32 + 52)
    if directory_position <= 0:
        return []
    fh.seek(directory_position)
    seg = fh.read(32)
    if len(seg) < 32 or seg[:16].rstrip(b"\x00") != b"ZISRAWDIRECTORY":
        return []
    allocated, used = struct.unpack_from("<qq", seg, 16)
    data = fh.read(used or allocated)
    if len(data) < 128:
        return []
    count, = struct.unpack_from("<i", data)
    pos, entries = 128, []
    for _ in range(count):
        if pos + 32 > len(data):
            break                                   # truncated: what was read is checked
        schema, pixel_type, position, _part, compression = struct.unpack_from("<2siqii", data, pos)
        ndims, = struct.unpack_from("<i", data, pos + 28)
        if schema != b"DV" or not 0 < ndims < 64:
            break
        entries.append((pixel_type, position, compression))
        pos += 32 + 20 * ndims
    return entries


def _packed(fh, position: int) -> bool:
    """Whether the zstd1 subblock at *position* has the hi-lo flag in its data header."""
    fh.seek(position)
    head = fh.read(32 + 48)
    if len(head) < 80 or head[:16].rstrip(b"\x00") != b"ZISRAWSUBBLOCK":
        return False
    metadata_size, = struct.unpack_from("<i", head, 32)
    ndims, = struct.unpack_from("<i", head, 32 + 44)
    fh.seek(position + 32 + 16 + max(240, 32 + 20 * ndims) + metadata_size)
    data = fh.read(3)
    # header: its size, then (chunk type 1 = hi-lo packing, value bit 0 = on)
    return len(data) == 3 and data[0] >= 3 and data[1] == 1 and bool(data[2] & 1)


def misread_by_libczi(path: str) -> bool:
    """True when *path* is a CZI with 16-bit RGB zstd subblocks stored hi-lo packed,
    which the standard CZI readers (libCZI) decode wrongly."""
    try:
        with open(path, "rb") as fh:
            entries = [e for e in _directory(fh)
                       if e[2] == ZSTD1 and e[0] in MISREAD_PIXEL_TYPES]
            if not entries:
                return False
            step = max(1, len(entries) // SAMPLES)
            return any(_packed(fh, position) for _, position, _ in entries[::step][:SAMPLES + 1])
    except (OSError, struct.error):
        return False                                # unreadable here: the readers report it
