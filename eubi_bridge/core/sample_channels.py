"""Multi-sample pixels (RGB, RGBA) become channels: one rule for every reader.

A pixel with ``S`` samples in each of ``C`` channels becomes ``C x S``
channels, channel-major (c0 R, c0 G, c0 B, c1 R, ...), samples in R, G, B
order.  Nothing is dropped.  Readers used to keep only the first sample
whenever a file had real channels *and* samples (e.g. a 2-channel RGB ND2
wrote 2 channels of red only); single-channel RGB already became 3 channels,
and still does, unchanged.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

#: labels and colours of the channels that samples become
SAMPLE_LABELS = {3: ("R", "G", "B"), 4: ("R", "G", "B", "A")}
SAMPLE_COLORS = {3: ("#FF0000", "#00FF00", "#0000FF"),
                 4: ("#FF0000", "#00FF00", "#0000FF", "#FFFFFF")}


def fold_dask(data, reverse: bool = False):
    """Dask (T, C, Z, Y, X, S) -> (T, C*S, Z, Y, X), lazily.

    *reverse* flips the samples first (CZI stores B, G, R).
    """
    import dask.array as da
    if reverse:
        data = data[..., ::-1]
    t, c, z, y, x, s = data.shape
    data = da.moveaxis(data, -1, 2)                   # T C S Z Y X
    data = data.rechunk({2: s})                       # merging C and S needs whole S chunks
    return data.reshape(t, c * s, z, y, x)


class SamplesAsChannels:
    """A (T, C, Z, Y, X, S) array-like -> (T, C*S, Z, Y, X), read by rectangle.

    For region-reading sources (``DynamicArray`` over a reader), where a
    reshape would read the whole array: each read fetches only the channels
    the requested sample-channels belong to (all their samples come from the
    same read) and the requested rectangle.
    """

    def __init__(self, source, reverse: bool = False):
        t, c, z, y, x, s = source.shape
        self.source = source
        self.samples = s
        self.reverse = reverse
        self.shape = (t, c * s, z, y, x)
        self.dtype = np.dtype(source.dtype)
        self.ndim = 5
        chunks = getattr(source, "chunks", None)
        self.chunks = ((1, 1) + tuple(chunks[2:5])) if chunks and len(chunks) == 6 \
            and all(isinstance(v, (int, np.integer)) for v in chunks) else None

    def __getitem__(self, key):
        from eubi_bridge.utils.array_utils import normalise_basic_index
        spans, squeeze, steps = normalise_basic_index(key, self.shape)
        (t0, t1), (k0, k1), (z0, z1), (y0, y1), (x0, x1) = spans
        s = self.samples
        c0, c1 = k0 // s, (k1 - 1) // s + 1 if k1 > k0 else k0 // s
        block = np.asarray(self.source[t0:t1, c0:c1, z0:z1, y0:y1, x0:x1, :])
        if self.reverse:
            block = block[..., ::-1]
        block = np.moveaxis(block, -1, 2).reshape(
            block.shape[0], block.shape[1] * s, *block.shape[2:5])
        out = block[:, k0 - c0 * s:k1 - c0 * s]
        if any(step != 1 for step in steps):
            out = out[tuple(slice(None, None, step) for step in steps)]
        if squeeze:
            out = out.squeeze(axis=squeeze)
        return np.ascontiguousarray(out)


def expand_channel_metadata(pixels, channels: int, samples: int):
    """Make OME *pixels* describe the ``channels x samples`` channels of the
    folded data, in place; returns *pixels*.

    Already right (e.g. Bio-Formats lists RGB as 3 channels): left as is.
    Single-channel RGB otherwise: ``samples`` plain channels, exactly what
    eubi-bridge wrote before.  Several real channels (the case that used to
    lose samples): each becomes ``samples`` channels named "<name> R/G/B"
    and tinted red, green, blue.
    """
    from ome_types.model import Channel

    total = channels * samples
    current = list(getattr(pixels, "channels", None) or [])
    if pixels.size_c == total and len(current) == total:
        return pixels
    expanded = []
    if channels == 1:
        expanded = [Channel(id=f"Channel:{i}", samples_per_pixel=1) for i in range(samples)]
    else:
        labels = SAMPLE_LABELS.get(samples, tuple(str(i) for i in range(samples)))
        colors = SAMPLE_COLORS.get(samples)
        originals = current if len(current) == channels else [None] * channels
        for c, original in enumerate(originals):
            base = getattr(original, "name", None) or f"Channel {c}"
            for i in range(samples):
                name: Optional[str] = f"{base} {labels[i]}"
                kwargs = {"color": colors[i]} if colors else {}
                expanded.append(Channel(id=f"Channel:{len(expanded)}", name=name,
                                        samples_per_pixel=1, **kwargs))
    pixels.size_c = total
    pixels.channels = expanded
    return pixels
