"""Spatial, Fourier, and morphological filter wave processors (REGION type).

All processors accept numpy or dask arrays and return a **lazy ``dask.Array``**.
No materialisation happens inside ``process()`` — the executor handles writing.

Chunk boundary effects are managed with ``dask.array.map_overlap``: each chunk
is extended by the filter's reach before scipy is called, then trimmed back.

**Fourier filters** operate per-chunk after a local FFT — effectively a
spatially-windowed frequency-domain filter, equivalent to their spatial
counterparts for sigma / size << chunk size.

**Binary fill holes** and **distance_transform_edt** operate fully per-chunk
(no overlap). Results near chunk boundaries are approximate; use large chunks
or rechunk to mitigate.
"""
from __future__ import annotations

import math

import numpy as np
import dask.array as da
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from eubi_flow.processing.base import BaseWaveProcessor, axis_param, AXIS_INDEX
from eubi_flow.registry import register_wave


def _as_dask(data) -> da.Array:
    if isinstance(data, da.Array):
        return data
    return da.from_array(np.asarray(data), chunks=np.asarray(data).shape)


def _ball_struct(ndim: int, radius: int) -> np.ndarray:
    """Hyper-spherical binary structuring element of the given radius."""
    coords = np.mgrid[tuple(slice(-radius, radius + 1) for _ in range(ndim))]
    return np.sqrt(sum(c ** 2 for c in coords)) <= radius


def _clip_depth(depth: tuple[int, ...], arr: da.Array) -> tuple[int, ...]:
    """Clamp overlap depth so no axis exceeds half its array dimension.

    Prevents ``map_overlap`` from failing when a computed depth is larger than
    a singleton axis (e.g. t=1 or c=1 in tczyx data).
    """
    return tuple(int(min(d, max(s // 2, 0))) for d, s in zip(depth, arr.shape))


# ---------------------------------------------------------------------------
# Smoothing filters
# ---------------------------------------------------------------------------

@register_wave
class GaussianFilterWave(BaseWaveProcessor):
    """Gaussian smoothing via ``scipy.ndimage.gaussian_filter``.

    Parameters
    ----------
    sigma : float, default 0.0
        Uniform sigma (pixels) for all axes.
    {t,c,z,y,x}_sigma : float
        Per-axis sigma; overrides the uniform value.
    """

    name = "gaussian_filter"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        sigma: float = Field(0.0, ge=0.0)

    params_model = Params

    def overlap(self, params: dict) -> tuple[int, ...]:
        return tuple(math.ceil(3.0 * s) for s in axis_param(params, "sigma", 0.0))

    def process(self, data, params: dict) -> da.Array:
        from scipy.ndimage import gaussian_filter
        arr = _as_dask(data)
        sigma = axis_param(params, "sigma", 0.0)
        depth = _clip_depth(self.overlap(params), arr)
        dtype = arr.dtype

        def _apply(chunk: np.ndarray) -> np.ndarray:
            return gaussian_filter(chunk.astype(np.float64), sigma=sigma).astype(dtype)

        return da.map_overlap(_apply, arr, depth=depth, boundary="reflect", trim=True, dtype=dtype)


@register_wave
class MedianFilterWave(BaseWaveProcessor):
    """Median filter via ``scipy.ndimage.median_filter``.

    Parameters
    ----------
    size : int, default 1
        Uniform kernel size (must be odd). ``size=1`` is identity.
    {t,c,z,y,x}_size : int
        Per-axis kernel size.
    """

    name = "median_filter"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        size: int = Field(1, ge=1)

    params_model = Params

    def overlap(self, params: dict) -> tuple[int, ...]:
        return tuple(int(s) // 2 for s in axis_param(params, "size", 1))

    def process(self, data, params: dict) -> da.Array:
        from scipy.ndimage import median_filter
        arr = _as_dask(data)
        size = tuple(int(s) for s in axis_param(params, "size", 1))
        depth = _clip_depth(self.overlap(params), arr)
        dtype = arr.dtype

        def _apply(chunk: np.ndarray) -> np.ndarray:
            return median_filter(chunk, size=size)

        return da.map_overlap(_apply, arr, depth=depth, boundary="reflect", trim=True, dtype=dtype)


@register_wave
class UniformFilterWave(BaseWaveProcessor):
    """Uniform (box) filter via ``scipy.ndimage.uniform_filter``.

    Parameters
    ----------
    size : int, default 1
        Uniform kernel size. ``size=1`` is identity.
    {t,c,z,y,x}_size : int
        Per-axis kernel size.
    """

    name = "uniform_filter"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        size: int = Field(1, ge=1)

    params_model = Params

    def overlap(self, params: dict) -> tuple[int, ...]:
        return tuple(int(s) // 2 for s in axis_param(params, "size", 1))

    def process(self, data, params: dict) -> da.Array:
        from scipy.ndimage import uniform_filter
        arr = _as_dask(data)
        size = tuple(int(s) for s in axis_param(params, "size", 1))
        depth = _clip_depth(self.overlap(params), arr)
        dtype = arr.dtype

        def _apply(chunk: np.ndarray) -> np.ndarray:
            return uniform_filter(chunk, size=size)

        return da.map_overlap(_apply, arr, depth=depth, boundary="reflect", trim=True, dtype=dtype)


@register_wave
class MaximumFilterWave(BaseWaveProcessor):
    """Local maximum filter via ``scipy.ndimage.maximum_filter``.

    Parameters
    ----------
    size : int, default 3
        Uniform kernel size.
    {t,c,z,y,x}_size : int
        Per-axis kernel size.
    """

    name = "maximum_filter"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        size: int = Field(3, ge=1)

    params_model = Params

    def overlap(self, params: dict) -> tuple[int, ...]:
        return tuple(int(s) // 2 for s in axis_param(params, "size", 3))

    def process(self, data, params: dict) -> da.Array:
        from scipy.ndimage import maximum_filter
        arr = _as_dask(data)
        size = tuple(int(s) for s in axis_param(params, "size", 3))
        depth = _clip_depth(self.overlap(params), arr)
        dtype = arr.dtype

        def _apply(chunk: np.ndarray) -> np.ndarray:
            return maximum_filter(chunk, size=size)

        return da.map_overlap(_apply, arr, depth=depth, boundary="reflect", trim=True, dtype=dtype)


@register_wave
class MinimumFilterWave(BaseWaveProcessor):
    """Local minimum filter via ``scipy.ndimage.minimum_filter``.

    Parameters
    ----------
    size : int, default 3
        Uniform kernel size.
    {t,c,z,y,x}_size : int
        Per-axis kernel size.
    """

    name = "minimum_filter"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        size: int = Field(3, ge=1)

    params_model = Params

    def overlap(self, params: dict) -> tuple[int, ...]:
        return tuple(int(s) // 2 for s in axis_param(params, "size", 3))

    def process(self, data, params: dict) -> da.Array:
        from scipy.ndimage import minimum_filter
        arr = _as_dask(data)
        size = tuple(int(s) for s in axis_param(params, "size", 3))
        depth = _clip_depth(self.overlap(params), arr)
        dtype = arr.dtype

        def _apply(chunk: np.ndarray) -> np.ndarray:
            return minimum_filter(chunk, size=size)

        return da.map_overlap(_apply, arr, depth=depth, boundary="reflect", trim=True, dtype=dtype)


@register_wave
class PercentileFilterWave(BaseWaveProcessor):
    """Percentile filter via ``scipy.ndimage.percentile_filter``.

    Parameters
    ----------
    size : int, default 3
        Kernel size.
    percentile : float, default 50.0
        Percentile value (0–100). 50 = median.
    {t,c,z,y,x}_size : int
        Per-axis kernel size.
    """

    name = "percentile_filter"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        size: int = Field(3, ge=1)
        percentile: float = Field(50.0, ge=0.0, le=100.0)

    params_model = Params

    def overlap(self, params: dict) -> tuple[int, ...]:
        return tuple(int(s) // 2 for s in axis_param(params, "size", 3))

    def process(self, data, params: dict) -> da.Array:
        from scipy.ndimage import percentile_filter
        arr = _as_dask(data)
        size = tuple(int(s) for s in axis_param(params, "size", 3))
        pct = float(params.get("percentile", 50.0))
        depth = _clip_depth(self.overlap(params), arr)
        dtype = arr.dtype

        def _apply(chunk: np.ndarray) -> np.ndarray:
            return percentile_filter(chunk, percentile=pct, size=size)

        return da.map_overlap(_apply, arr, depth=depth, boundary="reflect", trim=True, dtype=dtype)


@register_wave
class RankFilterWave(BaseWaveProcessor):
    """Rank filter via ``scipy.ndimage.rank_filter``.

    Returns the n-th ranked value in each local neighbourhood.
    ``rank=0`` → local minimum; ``rank=-1`` → local maximum.

    Parameters
    ----------
    size : int, default 3
        Kernel size.
    rank : int, default 0
        Which rank to select. Negative values count from the top.
    {t,c,z,y,x}_size : int
        Per-axis kernel size.
    """

    name = "rank_filter"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        size: int = Field(3, ge=1)
        rank: int = 0

    params_model = Params

    def overlap(self, params: dict) -> tuple[int, ...]:
        return tuple(int(s) // 2 for s in axis_param(params, "size", 3))

    def process(self, data, params: dict) -> da.Array:
        from scipy.ndimage import rank_filter
        arr = _as_dask(data)
        size = tuple(int(s) for s in axis_param(params, "size", 3))
        rank = int(params.get("rank", 0))
        depth = _clip_depth(self.overlap(params), arr)
        dtype = arr.dtype

        def _apply(chunk: np.ndarray) -> np.ndarray:
            return rank_filter(chunk, rank=rank, size=size)

        return da.map_overlap(_apply, arr, depth=depth, boundary="reflect", trim=True, dtype=dtype)


# ---------------------------------------------------------------------------
# Gradient / edge filters
# ---------------------------------------------------------------------------

@register_wave
class LaplaceWave(BaseWaveProcessor):
    """Isotropic Laplace filter via ``scipy.ndimage.laplace``.

    Computes the sum of second derivatives along all axes.
    Useful for sharpening and edge enhancement. Output is float64.
    """

    name = "laplace"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")

    params_model = Params

    def overlap(self, params: dict) -> tuple[int, ...]:  # noqa: ARG002
        return (1, 1, 1, 1, 1)

    def process(self, data, params: dict) -> da.Array:  # noqa: ARG002
        from scipy.ndimage import laplace
        arr = _as_dask(data)
        depth = _clip_depth((1, 1, 1, 1, 1), arr)

        def _apply(chunk: np.ndarray) -> np.ndarray:
            return laplace(chunk.astype(np.float64))

        return da.map_overlap(_apply, arr, depth=depth,
                              boundary="reflect", trim=True, dtype=np.float64)


@register_wave
class GaussianLaplaceWave(BaseWaveProcessor):
    """Laplacian of Gaussian (LoG) via ``scipy.ndimage.gaussian_laplace``.

    Combines Gaussian smoothing with the Laplacian operator, giving
    scale-selective blob/edge detection. Output is float64.

    Parameters
    ----------
    sigma : float, default 1.0
        Gaussian smoothing sigma before the Laplacian.
    {t,c,z,y,x}_sigma : float
        Per-axis sigma.
    """

    name = "gaussian_laplace"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        sigma: float = Field(1.0, ge=0.0)

    params_model = Params

    def overlap(self, params: dict) -> tuple[int, ...]:
        return tuple(math.ceil(3.0 * s) + 1 for s in axis_param(params, "sigma", 1.0))

    def process(self, data, params: dict) -> da.Array:
        from scipy.ndimage import gaussian_laplace
        arr = _as_dask(data)
        sigma = axis_param(params, "sigma", 1.0)
        depth = _clip_depth(self.overlap(params), arr)

        def _apply(chunk: np.ndarray) -> np.ndarray:
            return gaussian_laplace(chunk.astype(np.float64), sigma=sigma)

        return da.map_overlap(_apply, arr, depth=depth, boundary="reflect",
                              trim=True, dtype=np.float64)


@register_wave
class GaussianGradientMagnitudeWave(BaseWaveProcessor):
    """Gradient magnitude of Gaussian-smoothed image.

    Computes sqrt(Σ (∂I/∂xᵢ)²) via
    ``scipy.ndimage.gaussian_gradient_magnitude``.
    Useful for edge detection and boundary enhancement. Output is float64.

    Parameters
    ----------
    sigma : float, default 1.0
        Gaussian smoothing sigma.
    {t,c,z,y,x}_sigma : float
        Per-axis sigma.
    """

    name = "gaussian_gradient_magnitude"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        sigma: float = Field(1.0, ge=0.0)

    params_model = Params

    def overlap(self, params: dict) -> tuple[int, ...]:
        return tuple(math.ceil(3.0 * s) + 1 for s in axis_param(params, "sigma", 1.0))

    def process(self, data, params: dict) -> da.Array:
        from scipy.ndimage import gaussian_gradient_magnitude
        arr = _as_dask(data)
        sigma = axis_param(params, "sigma", 1.0)
        depth = _clip_depth(self.overlap(params), arr)

        def _apply(chunk: np.ndarray) -> np.ndarray:
            return gaussian_gradient_magnitude(chunk.astype(np.float64), sigma=sigma)

        return da.map_overlap(_apply, arr, depth=depth, boundary="reflect",
                              trim=True, dtype=np.float64)


@register_wave
class SobelWave(BaseWaveProcessor):
    """Sobel gradient filter along one axis via ``scipy.ndimage.sobel``.

    Returns the Sobel approximation of ∂I/∂axis. Output is float64.

    Parameters
    ----------
    axis : str, default "z"
        Named axis to differentiate along (one of t, c, z, y, x).
    """

    name = "sobel"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        axis: Literal["t", "c", "z", "y", "x"] = "z"

    params_model = Params

    def overlap(self, params: dict) -> tuple[int, ...]:
        return (1, 1, 1, 1, 1)

    def process(self, data, params: dict) -> da.Array:
        from scipy.ndimage import sobel
        arr = _as_dask(data)
        ax = AXIS_INDEX[params.get("axis", "z")]
        depth = _clip_depth((1, 1, 1, 1, 1), arr)

        def _apply(chunk: np.ndarray) -> np.ndarray:
            return sobel(chunk.astype(np.float64), axis=ax)

        return da.map_overlap(_apply, arr, depth=depth,
                              boundary="reflect", trim=True, dtype=np.float64)


@register_wave
class PrewittWave(BaseWaveProcessor):
    """Prewitt gradient filter along one axis via ``scipy.ndimage.prewitt``.

    Returns the Prewitt approximation of ∂I/∂axis. Output is float64.

    Parameters
    ----------
    axis : str, default "z"
        Named axis to differentiate along (one of t, c, z, y, x).
    """

    name = "prewitt"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        axis: Literal["t", "c", "z", "y", "x"] = "z"

    params_model = Params

    def overlap(self, params: dict) -> tuple[int, ...]:
        return (1, 1, 1, 1, 1)

    def process(self, data, params: dict) -> da.Array:
        from scipy.ndimage import prewitt
        arr = _as_dask(data)
        ax = AXIS_INDEX[params.get("axis", "z")]
        depth = _clip_depth((1, 1, 1, 1, 1), arr)

        def _apply(chunk: np.ndarray) -> np.ndarray:
            return prewitt(chunk.astype(np.float64), axis=ax)

        return da.map_overlap(_apply, arr, depth=depth,
                              boundary="reflect", trim=True, dtype=np.float64)


# ---------------------------------------------------------------------------
# Fourier filters
# ---------------------------------------------------------------------------

@register_wave
class FourierGaussianWave(BaseWaveProcessor):
    """Gaussian filter in the Fourier domain.

    Each chunk is FFT'd, multiplied by a Gaussian kernel in frequency space,
    then inverse-FFT'd (``scipy.ndimage.fourier_gaussian``).
    Equivalent to spatial Gaussian filtering; boundary effects are handled
    via overlap. Output dtype is preserved.

    Parameters
    ----------
    sigma : float, default 1.0
        Gaussian sigma in pixels (frequency domain).
    {t,c,z,y,x}_sigma : float
        Per-axis sigma.
    """

    name = "fourier_gaussian"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        sigma: float = Field(1.0, ge=0.0)

    params_model = Params

    def overlap(self, params: dict) -> tuple[int, ...]:
        return tuple(math.ceil(3.0 * s) for s in axis_param(params, "sigma", 1.0))

    def process(self, data, params: dict) -> da.Array:
        from scipy.ndimage import fourier_gaussian
        arr = _as_dask(data)
        sigma = axis_param(params, "sigma", 1.0)
        depth = _clip_depth(self.overlap(params), arr)
        dtype = arr.dtype

        def _apply(chunk: np.ndarray) -> np.ndarray:
            freq = np.fft.fftn(chunk.astype(np.float64))
            return np.real(np.fft.ifftn(fourier_gaussian(freq, sigma=sigma))).astype(dtype)

        return da.map_overlap(_apply, arr, depth=depth, boundary="reflect", trim=True, dtype=dtype)


@register_wave
class FourierUniformWave(BaseWaveProcessor):
    """Box (uniform) filter in the Fourier domain.

    Each chunk is FFT'd, multiplied by a box kernel in frequency space,
    then inverse-FFT'd (``scipy.ndimage.fourier_uniform``). Output dtype
    is preserved.

    Parameters
    ----------
    size : int, default 3
        Box size in pixels.
    {t,c,z,y,x}_size : int
        Per-axis size.
    """

    name = "fourier_uniform"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        size: int = Field(3, ge=1)

    params_model = Params

    def overlap(self, params: dict) -> tuple[int, ...]:
        return tuple(int(s) // 2 for s in axis_param(params, "size", 3))

    def process(self, data, params: dict) -> da.Array:
        from scipy.ndimage import fourier_uniform
        arr = _as_dask(data)
        size = tuple(int(s) for s in axis_param(params, "size", 3))
        depth = _clip_depth(self.overlap(params), arr)
        dtype = arr.dtype

        def _apply(chunk: np.ndarray) -> np.ndarray:
            freq = np.fft.fftn(chunk.astype(np.float64))
            return np.real(np.fft.ifftn(fourier_uniform(freq, size=size))).astype(dtype)

        return da.map_overlap(_apply, arr, depth=depth, boundary="reflect", trim=True, dtype=dtype)


@register_wave
class FourierEllipsoidWave(BaseWaveProcessor):
    """Ellipsoidal filter in the Fourier domain.

    Each chunk is FFT'd, multiplied by an ellipsoidal kernel in frequency
    space, then inverse-FFT'd (``scipy.ndimage.fourier_ellipsoid``).
    Output dtype is preserved.

    Parameters
    ----------
    size : int, default 3
        Ellipsoid radius in pixels (uniform across axes).
    """

    name = "fourier_ellipsoid"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        size: int = Field(3, ge=1)

    params_model = Params

    def overlap(self, params: dict) -> tuple[int, ...]:
        s = int(params.get("size", 3))
        return (s // 2,) * 5

    def process(self, data, params: dict) -> da.Array:
        from scipy.ndimage import fourier_ellipsoid
        arr = _as_dask(data)
        size = int(params.get("size", 3))
        depth = _clip_depth(self.overlap(params), arr)
        dtype = arr.dtype

        # scipy.ndimage.fourier_ellipsoid only supports up to 3-D.
        # Apply it to each (z, y, x) sub-volume, looping over t and c.
        def _apply(chunk: np.ndarray) -> np.ndarray:
            result = np.empty(chunk.shape, dtype=dtype)
            for ti in range(chunk.shape[0]):
                for ci in range(chunk.shape[1]):
                    sub = chunk[ti, ci].astype(np.float64)
                    freq = np.fft.fftn(sub)
                    result[ti, ci] = np.real(
                        np.fft.ifftn(fourier_ellipsoid(freq, size=size))
                    ).astype(dtype)
            return result

        return da.map_overlap(_apply, arr, depth=depth, boundary="reflect", trim=True, dtype=dtype)


# ---------------------------------------------------------------------------
# Morphological filters — binary
# ---------------------------------------------------------------------------

@register_wave
class BinaryErosionWave(BaseWaveProcessor):
    """Binary erosion via ``scipy.ndimage.binary_erosion``.

    Input: any array (thresholded to binary with > 0).
    Output: uint8 (0 / 255). Uses a spherical structuring element.

    Parameters
    ----------
    radius : int, default 1
        Radius of the spherical structuring element.
    iterations : int, default 1
        Number of erosion iterations.
    """

    name = "binary_erosion"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        radius: int = Field(1, ge=1)
        iterations: int = Field(1, ge=1)

    params_model = Params

    def overlap(self, params: dict) -> tuple[int, ...]:
        r = int(params.get("radius", 1)) * int(params.get("iterations", 1))
        return (r,) * 5

    def process(self, data, params: dict) -> da.Array:
        from scipy.ndimage import binary_erosion
        arr = _as_dask(data)
        radius = int(params.get("radius", 1))
        iterations = int(params.get("iterations", 1))
        depth = _clip_depth(self.overlap(params), arr)
        struct = _ball_struct(arr.ndim, radius)

        def _apply(chunk: np.ndarray) -> np.ndarray:
            return binary_erosion(chunk > 0, structure=struct,
                                  iterations=iterations).astype(np.uint8) * 255

        return da.map_overlap(_apply, arr, depth=depth, boundary=0,
                              trim=True, dtype=np.uint8)


@register_wave
class BinaryDilationWave(BaseWaveProcessor):
    """Binary dilation via ``scipy.ndimage.binary_dilation``.

    Input: any array (thresholded to binary with > 0).
    Output: uint8 (0 / 255). Uses a spherical structuring element.

    Parameters
    ----------
    radius : int, default 1
        Radius of the spherical structuring element.
    iterations : int, default 1
        Number of dilation iterations.
    """

    name = "binary_dilation"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        radius: int = Field(1, ge=1)
        iterations: int = Field(1, ge=1)

    params_model = Params

    def overlap(self, params: dict) -> tuple[int, ...]:
        r = int(params.get("radius", 1)) * int(params.get("iterations", 1))
        return (r,) * 5

    def process(self, data, params: dict) -> da.Array:
        from scipy.ndimage import binary_dilation
        arr = _as_dask(data)
        radius = int(params.get("radius", 1))
        iterations = int(params.get("iterations", 1))
        depth = _clip_depth(self.overlap(params), arr)
        struct = _ball_struct(arr.ndim, radius)

        def _apply(chunk: np.ndarray) -> np.ndarray:
            return binary_dilation(chunk > 0, structure=struct,
                                   iterations=iterations).astype(np.uint8) * 255

        return da.map_overlap(_apply, arr, depth=depth, boundary=0,
                              trim=True, dtype=np.uint8)


@register_wave
class BinaryOpeningWave(BaseWaveProcessor):
    """Binary opening (erosion→dilation) via ``scipy.ndimage.binary_opening``.

    Removes small bright features; preserves shape of larger ones.
    Output: uint8 (0 / 255). Uses a spherical structuring element.

    Parameters
    ----------
    radius : int, default 1
        Radius of the structuring element.
    iterations : int, default 1
        Number of opening iterations.
    """

    name = "binary_opening"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        radius: int = Field(1, ge=1)
        iterations: int = Field(1, ge=1)

    params_model = Params

    def overlap(self, params: dict) -> tuple[int, ...]:
        r = int(params.get("radius", 1)) * int(params.get("iterations", 1))
        return (r * 2,) * 5

    def process(self, data, params: dict) -> da.Array:
        from scipy.ndimage import binary_opening
        arr = _as_dask(data)
        radius = int(params.get("radius", 1))
        iterations = int(params.get("iterations", 1))
        depth = _clip_depth(self.overlap(params), arr)
        struct = _ball_struct(arr.ndim, radius)

        def _apply(chunk: np.ndarray) -> np.ndarray:
            return binary_opening(chunk > 0, structure=struct,
                                  iterations=iterations).astype(np.uint8) * 255

        return da.map_overlap(_apply, arr, depth=depth, boundary=0,
                              trim=True, dtype=np.uint8)


@register_wave
class BinaryClosingWave(BaseWaveProcessor):
    """Binary closing (dilation→erosion) via ``scipy.ndimage.binary_closing``.

    Fills small dark holes; preserves overall shape of bright regions.
    Output: uint8 (0 / 255). Uses a spherical structuring element.

    Parameters
    ----------
    radius : int, default 1
        Radius of the structuring element.
    iterations : int, default 1
        Number of closing iterations.
    """

    name = "binary_closing"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        radius: int = Field(1, ge=1)
        iterations: int = Field(1, ge=1)

    params_model = Params

    def overlap(self, params: dict) -> tuple[int, ...]:
        r = int(params.get("radius", 1)) * int(params.get("iterations", 1))
        return (r * 2,) * 5

    def process(self, data, params: dict) -> da.Array:
        from scipy.ndimage import binary_closing
        arr = _as_dask(data)
        radius = int(params.get("radius", 1))
        iterations = int(params.get("iterations", 1))
        depth = _clip_depth(self.overlap(params), arr)
        struct = _ball_struct(arr.ndim, radius)

        def _apply(chunk: np.ndarray) -> np.ndarray:
            return binary_closing(chunk > 0, structure=struct,
                                  iterations=iterations).astype(np.uint8) * 255

        return da.map_overlap(_apply, arr, depth=depth, boundary=0,
                              trim=True, dtype=np.uint8)


@register_wave
class BinaryFillHolesWave(BaseWaveProcessor):
    """Fill holes in a binary image via ``scipy.ndimage.binary_fill_holes``.

    **Note**: operates per chunk. Holes spanning chunk boundaries may not
    be filled. Use large regions to mitigate.
    Output: uint8 (0 / 255).
    """

    name = "binary_fill_holes"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")

    params_model = Params

    def overlap(self, params: dict) -> tuple[int, ...]:
        return (0,) * 5

    def process(self, data, params: dict) -> da.Array:
        from scipy.ndimage import binary_fill_holes
        arr = _as_dask(data)

        def _apply(chunk: np.ndarray) -> np.ndarray:
            return np.asarray(binary_fill_holes(chunk > 0)).astype(np.uint8) * 255

        return da.map_blocks(_apply, arr, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Morphological filters — greyscale
# ---------------------------------------------------------------------------

@register_wave
class GreyErosionWave(BaseWaveProcessor):
    """Greyscale erosion (local minimum) via ``scipy.ndimage.grey_erosion``.

    Parameters
    ----------
    size : int, default 3
        Cubic footprint size.
    {t,c,z,y,x}_size : int
        Per-axis footprint size.
    """

    name = "grey_erosion"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        size: int = Field(3, ge=1)

    params_model = Params

    def overlap(self, params: dict) -> tuple[int, ...]:
        return tuple(int(s) // 2 for s in axis_param(params, "size", 3))

    def process(self, data, params: dict) -> da.Array:
        from scipy.ndimage import grey_erosion
        arr = _as_dask(data)
        size = tuple(int(s) for s in axis_param(params, "size", 3))
        depth = _clip_depth(self.overlap(params), arr)
        dtype = arr.dtype

        def _apply(chunk: np.ndarray) -> np.ndarray:
            return grey_erosion(chunk, size=size)

        return da.map_overlap(_apply, arr, depth=depth, boundary="reflect", trim=True, dtype=dtype)


@register_wave
class GreyDilationWave(BaseWaveProcessor):
    """Greyscale dilation (local maximum) via ``scipy.ndimage.grey_dilation``.

    Parameters
    ----------
    size : int, default 3
        Cubic footprint size.
    {t,c,z,y,x}_size : int
        Per-axis footprint size.
    """

    name = "grey_dilation"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        size: int = Field(3, ge=1)

    params_model = Params

    def overlap(self, params: dict) -> tuple[int, ...]:
        return tuple(int(s) // 2 for s in axis_param(params, "size", 3))

    def process(self, data, params: dict) -> da.Array:
        from scipy.ndimage import grey_dilation
        arr = _as_dask(data)
        size = tuple(int(s) for s in axis_param(params, "size", 3))
        depth = _clip_depth(self.overlap(params), arr)
        dtype = arr.dtype

        def _apply(chunk: np.ndarray) -> np.ndarray:
            return grey_dilation(chunk, size=size)

        return da.map_overlap(_apply, arr, depth=depth, boundary="reflect", trim=True, dtype=dtype)


@register_wave
class GreyOpeningWave(BaseWaveProcessor):
    """Greyscale opening via ``scipy.ndimage.grey_opening``.

    Suppresses bright features smaller than the footprint.

    Parameters
    ----------
    size : int, default 3
        Cubic footprint size.
    {t,c,z,y,x}_size : int
        Per-axis footprint size.
    """

    name = "grey_opening"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        size: int = Field(3, ge=1)

    params_model = Params

    def overlap(self, params: dict) -> tuple[int, ...]:
        # opening = erosion + dilation, so depth = size - 1
        return tuple(int(s) - 1 for s in axis_param(params, "size", 3))

    def process(self, data, params: dict) -> da.Array:
        from scipy.ndimage import grey_opening
        arr = _as_dask(data)
        size = tuple(int(s) for s in axis_param(params, "size", 3))
        depth = _clip_depth(self.overlap(params), arr)
        dtype = arr.dtype

        def _apply(chunk: np.ndarray) -> np.ndarray:
            return grey_opening(chunk, size=size)

        return da.map_overlap(_apply, arr, depth=depth, boundary="reflect", trim=True, dtype=dtype)


@register_wave
class GreyClosingWave(BaseWaveProcessor):
    """Greyscale closing via ``scipy.ndimage.grey_closing``.

    Fills dark features smaller than the footprint.

    Parameters
    ----------
    size : int, default 3
        Cubic footprint size.
    {t,c,z,y,x}_size : int
        Per-axis footprint size.
    """

    name = "grey_closing"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        size: int = Field(3, ge=1)

    params_model = Params

    def overlap(self, params: dict) -> tuple[int, ...]:
        return tuple(int(s) - 1 for s in axis_param(params, "size", 3))

    def process(self, data, params: dict) -> da.Array:
        from scipy.ndimage import grey_closing
        arr = _as_dask(data)
        size = tuple(int(s) for s in axis_param(params, "size", 3))
        depth = _clip_depth(self.overlap(params), arr)
        dtype = arr.dtype

        def _apply(chunk: np.ndarray) -> np.ndarray:
            return grey_closing(chunk, size=size)

        return da.map_overlap(_apply, arr, depth=depth, boundary="reflect", trim=True, dtype=dtype)


# ---------------------------------------------------------------------------
# Distance transform
# ---------------------------------------------------------------------------

@register_wave
class DistanceTransformEdtWave(BaseWaveProcessor):
    """Euclidean distance transform via ``scipy.ndimage.distance_transform_edt``.

    Each voxel receives the distance to the nearest background voxel (zero).
    Commonly used to generate watershed markers.

    **Note**: operates per chunk. Distances near chunk boundaries are
    underestimated. Use large processing regions to mitigate.
    Output: float64.

    Parameters
    ----------
    sampling_z : float, default 1.0
        Physical voxel size along z (used for anisotropic data).
    sampling_y : float, default 1.0
        Physical voxel size along y.
    sampling_x : float, default 1.0
        Physical voxel size along x.
    """

    name = "distance_transform_edt"

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")
        sampling_z: float = Field(1.0, gt=0.0)
        sampling_y: float = Field(1.0, gt=0.0)
        sampling_x: float = Field(1.0, gt=0.0)

    params_model = Params

    def overlap(self, params: dict) -> tuple[int, ...]:
        return (0,) * 5

    def process(self, data, params: dict) -> da.Array:
        from scipy.ndimage import distance_transform_edt
        arr = _as_dask(data)
        # sampling for tczyx: t and c are not spatial
        sampling = (
            1.0, 1.0,
            float(params.get("sampling_z", 1.0)),
            float(params.get("sampling_y", 1.0)),
            float(params.get("sampling_x", 1.0)),
        )

        def _apply(chunk: np.ndarray) -> np.ndarray:
            return np.asarray(distance_transform_edt(chunk > 0, sampling=sampling))

        return da.map_blocks(_apply, arr, dtype=np.float64)
