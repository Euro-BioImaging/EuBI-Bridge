"""A minimal CZI (ZISRAW) writer for tests: any payload under any compression
code, so every codec can be tested without an instrument that writes it.

Layout: ZISRAWFILE header, one ZISRAWSUBBLOCK segment per tile (directory
entry copy, no subblock metadata, then the payload), ZISRAWDIRECTORY,
ZISRAWMETADATA with a minimal XML document.  pylibCZIrw and czifile read the
files, which checks the writer.
"""
from __future__ import annotations

import struct
import uuid

import numpy as np

#: numpy dtype and samples -> CZI pixel type
PIXEL_TYPE = {("u1", 1): 0, ("u2", 1): 1, ("f4", 1): 2, ("u1", 3): 3, ("u2", 3): 4,
              ("f4", 3): 8, ("u1", 4): 9}


def _segment(sid: bytes, data: bytes) -> bytes:
    allocated = -(-len(data) // 32) * 32
    return sid.ljust(16, b"\x00") + struct.pack("<qq", allocated, len(data)) \
        + data.ljust(allocated, b"\x00")


def _entry(pixel_type, position, compression, dims, pyramid=0) -> bytes:
    """A DV directory entry; dims: [(name, start, size, stored)]."""
    out = b"DV" + struct.pack("<iqiiBB4si", pixel_type, position, 0, compression, pyramid, 0,
                              b"\x00" * 4, len(dims))
    for name, start, size, stored in dims:
        out += name.encode().ljust(4, b"\x00") + struct.pack("<iifi", start, size, 0.0, stored)
    return out


def write_czi(path, tiles, metadata: str = ""):
    """*tiles*: [dict(scene, plane={'C': c, ...}, x, y, height, width, dtype,
    samples, compression, payload)] -- payload is the encoded subblock data
    (raw bytes stored B, G, R for colour when compression is 0).  A pyramid
    subblock also has ``stored=(width, height)`` (the pixels stored, fewer
    than its width x height) and ``pyramid`` (its pyramid type, 1 or 2); any
    subblock may have ``tags`` ({tag: value}: its subblock metadata).
    *metadata*: XML put inside the document's ``Metadata`` element."""
    header_size = 32 + 512
    out = bytearray(header_size)                                 # file header, filled in last
    entries = []
    for t in tiles:
        pixel_type = PIXEL_TYPE[(np.dtype(t["dtype"]).str[1:], t.get("samples", 1))]
        sw, sh = t.get("stored", (t["width"], t["height"]))
        dims = [("X", t["x"], t["width"], sw), ("Y", t["y"], t["height"], sh)]
        dims += [(d, v, 1, 1) for d, v in t.get("plane", {}).items()]
        dims += [("S", t.get("scene", 0), 1, 1)]
        if "mosaic" in t:
            dims += [("M", t["mosaic"], 1, 1)]
        position = len(out)
        entry = _entry(pixel_type, position, t["compression"], dims, t.get("pyramid", 0))
        tags = "".join(f"<{k}>{v}</{k}>" for k, v in t.get("tags", {}).items())
        sub_xml = f"<METADATA><Tags>{tags}</Tags></METADATA>".encode() if tags else b""
        body = struct.pack("<iiq", len(sub_xml), 0, len(t["payload"])) \
            + entry.ljust(240, b"\x00") + sub_xml + bytes(t["payload"])
        out += _segment(b"ZISRAWSUBBLOCK", body)
        entries.append(entry)
    directory_position = len(out)
    out += _segment(b"ZISRAWDIRECTORY",
                    struct.pack("<i", len(entries)) + b"\x00" * 124 + b"".join(entries))
    xml = (b'<?xml version="1.0"?><ImageDocument><Metadata>'
           + (metadata.encode() or b"<Information><Image></Image></Information>")
           + b"</Metadata></ImageDocument>")
    metadata_position = len(out)
    out += _segment(b"ZISRAWMETADATA", struct.pack("<ii", len(xml), 0) + b"\x00" * 248 + xml)
    guid = uuid.uuid4().bytes
    head = struct.pack("<iiii16s16siqqiq", 1, 0, 0, 0, guid, guid, 0,
                       directory_position, metadata_position, 0, 0)
    out[:header_size] = _segment(b"ZISRAWFILE", head.ljust(512, b"\x00"))
    with open(path, "wb") as fh:
        fh.write(out)


def raw_payload(array: np.ndarray) -> bytes:
    """Uncompressed subblock data; colour as stored in CZI (B, G, R)."""
    a = np.ascontiguousarray(array)
    if a.ndim == 3:
        a = np.ascontiguousarray(a[..., [2, 1, 0, 3][:a.shape[-1]]])
    return a.astype(a.dtype.newbyteorder("<")).tobytes()
