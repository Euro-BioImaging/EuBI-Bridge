"""Multi-sample pixels (RGB, RGBA) become channels: one rule for every reader.

A pixel with ``S`` samples in each of ``C`` channels becomes ``C x S``
channels, channel-major (c0 R, c0 G, c0 B, c1 R, ...), samples in R, G, B
order.  Nothing is dropped.  Readers used to keep only the first sample
whenever a file had real channels *and* samples (e.g. a 2-channel RGB ND2
wrote 2 channels of red only); single-channel RGB already became 3 channels.

Their names and colours (``sample_channel_labels``) are the same on every
path: "<name> R/G/B" tinted red / green / blue; an unnamed single channel
stays ``Channel 0/1/2``, tinted too (0.3.0: it used to be white, and a named
single channel lost its name).
"""
from __future__ import annotations

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


def sample_channel_labels(names, samples: int) -> list:
    """``(label, colour)`` for each channel the samples of the channels named
    *names* (one entry per real channel, None for unnamed) become, channel-major.

    "<name> R" / "G" / "B" ("A"), tinted from ``SAMPLE_COLORS``; an unnamed
    channel among several is "Channel {c} R" / ...  A single unnamed channel
    keeps the writer's own labels (label None: ``Channel 0/1/2``), tinted too.
    Sample counts without labels (e.g. 2) are numbered ("<name> 0") and get no
    colour (None: the writer's white).
    """
    names = list(names)
    labels = SAMPLE_LABELS.get(samples, tuple(str(i) for i in range(samples)))
    colors = SAMPLE_COLORS.get(samples) or (None,) * samples
    if len(names) == 1 and not names[0]:
        return [(None, colors[i]) for i in range(samples)]
    return [(f"{name or f'Channel {c}'} {labels[i]}", colors[i])
            for c, name in enumerate(names) for i in range(samples)]


def expand_channel_metadata(pixels, channels: int, samples: int):
    """Make OME *pixels* describe the ``channels x samples`` channels of the
    folded data, in place; returns *pixels*.

    Already right (e.g. Bio-Formats lists RGB as 3 channels): left as is.
    Otherwise each channel becomes ``samples`` channels, named and tinted by
    ``sample_channel_labels``: "<name> R/G/B" in red, green, blue (a single
    unnamed channel: no name, so the writer's ``Channel 0/1/2``, tinted).
    """
    from ome_types.model import Channel

    total = channels * samples
    current = list(getattr(pixels, "channels", None) or [])
    if pixels.size_c == total and len(current) == total:
        return pixels
    originals = current if len(current) == channels else [None] * channels
    names = [getattr(original, "name", None) for original in originals]
    expanded = []
    for name, color in sample_channel_labels(names, samples):
        kwargs: dict = {"name": name} if name else {}
        if color:
            kwargs["color"] = color
        expanded.append(Channel(id=f"Channel:{len(expanded)}", samples_per_pixel=1, **kwargs))
    pixels.size_c = total
    pixels.channels = expanded
    return pixels
