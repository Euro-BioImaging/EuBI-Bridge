"""A minimal ND2 (version 3) writer for tests: Nikon's chunk layout and
CLX-lite metadata, enough for micro-reader and the nd2 library to read.

Frames run over *loops* (outermost first), each frame a (height, width,
channels, samples) block with rows optionally padded, stored raw or as one
zlib stream ("lossless").
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

MAGIC = 0x0ABECEDA
FILE_SIGNATURE = b"ND2 FILE SIGNATURE CHUNK NAME01!"
CHUNKMAP_SIGNATURE = b"ND2 CHUNK MAP SIGNATURE 0000001!"
FILEMAP_SIGNATURE = b"ND2 FILEMAP SIGNATURE NAME 0001!"
IMAGE_NAME_BYTES = 4072        # Nikon pads frame chunk names so pixels start 4096 bytes in


# -- CLX-lite encoding ---------------------------------------------------------------

def _entry(kind: int, name: str, payload: bytes) -> bytes:
    encoded = (name + "\x00").encode("utf-16-le")
    return bytes([kind, len(encoded) // 2]) + encoded + payload


def clx(value, name: str = "") -> bytes:
    """Encode a Python value as one CLX-lite entry."""
    if isinstance(value, bool):
        return _entry(1, name, struct.pack("<B", value))
    if isinstance(value, int):
        return _entry(3, name, struct.pack("<I", value)) if value >= 0 else \
            _entry(2, name, struct.pack("<i", value))
    if isinstance(value, float):
        return _entry(6, name, struct.pack("<d", value))
    if isinstance(value, str):
        return _entry(8, name, (value + "\x00").encode("utf-16-le"))
    if isinstance(value, bytes):
        return _entry(9, name, struct.pack("<Q", len(value)) + value)
    if isinstance(value, (dict, list)):
        items = [clx(v, k) for k, v in value.items()] if isinstance(value, dict) \
            else [clx(v, "") for v in value]
        inner = b"".join(items)
        encoded = (name + "\x00").encode("utf-16-le")
        head = bytes([11, len(encoded) // 2]) + encoded
        length = len(head) + 12 + len(inner)
        return head + struct.pack("<IQ", len(items), length) + inner + b"\x00" * (8 * len(items))
    raise TypeError(type(value))


def clx_doc(root_name: str, value: dict) -> bytes:
    return clx(value, root_name)


# -- file ------------------------------------------------------------------------------

def _chunk(name: bytes, data: bytes, name_bytes: int | None = None) -> bytes:
    name = name.ljust(name_bytes or len(name), b"\x00")
    return struct.pack("<IIQ", MAGIC, len(name), len(data)) + name + data


def _loop(kind: str, count: int, inner: dict | None, valid=None, ne_periods=None) -> dict:
    if kind == "T":
        exp = {"eType": 1, "uLoopPars": {"uiCount": count, "dStart": 0.0, "dPeriod": 100.0,
                                         "dDuration": 0.0}}
    elif kind == "NE":
        periods = ne_periods or [count]
        exp = {"eType": 8, "uLoopPars": {
            "uiPeriodCount": len(periods),
            "pPeriod": {f"i{i:010}": {"uiCount": n, "dStart": 0.0, "dPeriod": 100.0,
                                      "dDuration": 0.0} for i, n in enumerate(periods)},
            "pPeriodValid": bytes([1] * len(periods)),
        }}
    elif kind == "P":
        points = {f"i{i:010}": {"dPosX": float(i), "dPosY": 0.0, "dPosZ": 0.0,
                                "dPFSOffset": -1.0, "dPosName": f"p{i}"}
                  for i in range(count)}
        exp = {"eType": 2, "uLoopPars": {"bUseZ": False, "bRelativeXY": False,
                                         "Points": points}}
        if valid is not None:
            exp["pItemValid"] = bytes(int(v) for v in valid)
    elif kind == "Z":
        exp = {"eType": 4, "uLoopPars": {"uiCount": count, "dZLow": 0.0,
                                         "dZHigh": float(count - 1), "dZStep": 1.0,
                                         "dZHome": 0.0, "bZInverted": False, "iType": 0}}
    if inner is not None:
        exp["ppNextLevelEx"] = {"i0000000000": inner}
    return exp


def write_nd2(path, frames: np.ndarray, loops: list, *, compression: str | None = None,
              pad_bytes: int = 0, missing=(), valid_positions=None, ne_periods=None) -> None:
    """*frames*: (n_frames, height, width, channels, samples) in frame order.
    *loops*: [(kind, count)] outermost first, kind in T / NE / P / Z; the
    product of counts must be n_frames.  *missing*: frame numbers left out.
    *valid_positions*: flags for the P loop (count = all points, frames only
    for the valid ones)."""
    n, height, width, channels, samples = frames.shape
    dtype = frames.dtype
    bpc = dtype.itemsize * 8
    row_bytes = width * channels * samples * dtype.itemsize
    width_bytes = row_bytes + pad_bytes

    exp = None
    for kind, count in reversed(loops):
        if kind == "P" and valid_positions is not None:
            exp = _loop(kind, len(valid_positions), exp, valid=valid_positions)
        else:
            exp = _loop(kind, count, exp, ne_periods=ne_periods if kind == "NE" else None)
    attrs = {"uiWidth": width, "uiWidthBytes": width_bytes, "uiHeight": height,
             "uiComp": channels * samples, "uiBpcInMemory": bpc, "uiBpcSignificant": bpc,
             "uiSequenceCount": n, "uiTileWidth": width, "uiTileHeight": height,
             "eCompression": 0 if compression == "lossless" else 2,
             "dCompressionParam": 0.0, "ePixelType": 1 if dtype.kind == "f" else 0,
             "uiVirtualComponents": channels * samples}
    # pFluorescentProbe: the nd2 library's wavelength parsing needs it (real
    # files always carry one)
    planes = {f"i{c:010}": {"uiCompCount": samples, "uiSampleIndex": c,
                            "sDescription": f"Ch{c}", "uiColor": 0xFFFFFF,
                            "pFluorescentProbe": {"m_sName": f"probe{c}",
                                                  "m_ExcitationSpectrum": {},
                                                  "m_EmissionSpectrum": {}}}
              for c in range(channels)}
    # bCalibrated / dCalibration: the pixel size, which the nd2 library (and so
    # bioio-nd2) requires to load metadata
    meta = {"bCalibrated": True, "dCalibration": 0.5,
            "sPicturePlanes": {"uiCount": channels, "sPlaneNew": planes,
                               "uiSampleCount": channels}}

    out = bytearray(_chunk(FILE_SIGNATURE, b"Ver3.0".ljust(64, b"\x00")))
    where: dict = {}

    def add(name: bytes, data: bytes, name_bytes=None):
        where[name] = (len(out), len(data))
        out.extend(_chunk(name, data, name_bytes))

    add(b"ImageAttributesLV!", clx_doc("SLxImageAttributes", attrs))
    add(b"ImageMetadataSeqLV|0!", clx_doc("SLxPictureMetadata", meta))
    if exp is not None:
        add(b"ImageMetadataLV!", clx_doc("SLxExperiment", exp))
    # frame acquisition times (ms), which bioio-nd2 reads for its metadata
    add(b"CustomData|AcqTimesCache!", (np.arange(n, dtype="<f8") * 100.0).tobytes())
    for i, frame in enumerate(frames):
        if i in missing:
            continue
        raw = np.zeros((height, width_bytes), np.uint8)
        raw[:, :row_bytes] = np.ascontiguousarray(frame).view(np.uint8).reshape(height, row_bytes)
        payload = raw.tobytes()
        if compression == "lossless":
            payload = zlib.compress(payload)
        add(f"ImageDataSeq|{i}!".encode(), struct.pack("<d", float(i)) + payload,
            IMAGE_NAME_BYTES)
    table = b"".join(name + struct.pack("<QQ", off, size) for name, (off, size) in where.items())
    table += CHUNKMAP_SIGNATURE + struct.pack("<QQ", 0, 0)
    map_at = len(out)
    out.extend(_chunk(FILEMAP_SIGNATURE, table))
    out.extend(CHUNKMAP_SIGNATURE + struct.pack("<Q", map_at))
    with open(path, "wb") as fh:
        fh.write(out)
