"""Smart downscaling: integer per-axis scale factors that bring a pyramid's levels
towards isotropic voxels (``apply_smart_downscaling``).

eubi-bridge's own downscalers (``DownscaleManager`` / ``Downscaler`` / ``ts_downscale``)
went with its writer in 0.2.0 (``legacy_0_1/``): ome_zarr_pyramid downscales.
"""
from __future__ import annotations

import itertools
from fractions import Fraction
from math import gcd, lcm

from eubi_bridge.utils.logging_config import get_logger

logger = get_logger(__name__)

SPATIAL_AXES = {'z', 'y', 'x'}
_SMART_DEFAULT_MAX_FACTOR = 16
_SMART_DEFAULT_TOLERANCE = 0.10


def compute_isotropic_scale_factors(
    pixel_sizes: dict,
    axes: str,
    max_factor: int = _SMART_DEFAULT_MAX_FACTOR,
    tolerance: float = _SMART_DEFAULT_TOLERANCE,
) -> dict:
    """Compute integer scale factors that bring spatial axes to near-isotropy.

    Parameters
    ----------
    pixel_sizes : dict
        Physical pixel size per axis, e.g. {'t': 1.0, 'z': 0.5, 'y': 0.1, 'x': 0.1}.
    axes : str
        Axis order string, e.g. 'tczyx'.
    max_factor : int
        Maximum allowed scale factor per spatial axis (default 16).
    tolerance : float
        Acceptable residual anisotropy: max/min - 1 (default 0.10 = 10%).

    Returns
    -------
    dict
        Integer scale factor per axis, e.g. {'t': 1, 'c': 1, 'z': 1, 'y': 5, 'x': 5}.
        Non-spatial axes always get factor 1.
    """
    result = {ax: 1 for ax in axes}

    spatial = [ax for ax in axes if ax in SPATIAL_AXES and ax in pixel_sizes and pixel_sizes[ax] > 0]
    if len(spatial) < 2:
        return result  # 0 or 1 spatial axis: already trivially isotropic

    ps = [float(pixel_sizes[ax]) for ax in spatial]

    # Fast path: already isotropic within tolerance
    if max(ps) / min(ps) - 1 <= tolerance / 10:
        return result

    # ── Step 1: exact LCM approach via rational arithmetic ─────────────────
    fracs = [Fraction(p).limit_denominator(1000) for p in ps]
    num_lcm = fracs[0].numerator
    den_gcd = fracs[0].denominator
    for f in fracs[1:]:
        num_lcm = lcm(num_lcm, f.numerator)
        den_gcd = gcd(den_gcd, f.denominator)
    target = num_lcm / den_gcd
    exact_factors = [max(1, round(target / float(f))) for f in fracs]

    if all(1 <= s <= max_factor for s in exact_factors):
        result_sizes = [ps[i] * exact_factors[i] for i in range(len(ps))]
        if max(result_sizes) / min(result_sizes) - 1 <= tolerance:
            for ax, s in zip(spatial, exact_factors):
                result[ax] = int(s)
            logger.info(
                f"[smart_downscale] LCM approach: factors={dict(zip(spatial, exact_factors))}, "
                f"result_sizes={dict(zip(spatial, result_sizes))}"
            )
            return result

    # ── Step 2: brute-force search over [1..max_factor]^N ─────────────────
    best_factors = [1] * len(spatial)
    best_aniso = float('inf')

    for combo in itertools.product(range(1, max_factor + 1), repeat=len(spatial)):
        result_sizes = [ps[i] * combo[i] for i in range(len(ps))]
        aniso = max(result_sizes) / min(result_sizes) - 1
        if aniso < best_aniso:
            best_aniso = aniso
            best_factors = list(combo)
            if aniso <= tolerance:
                break  # good enough, stop early

    for ax, s in zip(spatial, best_factors):
        result[ax] = int(s)

    result_sizes = [ps[i] * best_factors[i] for i in range(len(ps))]
    logger.info(
        f"[smart_downscale] brute-force: factors={dict(zip(spatial, best_factors))}, "
        f"result_sizes={dict(zip(spatial, result_sizes))}, anisotropy={best_aniso:.3f}"
    )
    return result
