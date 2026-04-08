import asyncio
import dataclasses
import itertools
import os.path
from fractions import Fraction
from math import gcd, lcm
from typing import Union, Optional, Dict, Any, List

import dask.array as da
import numpy as np
import tensorstore as ts
import zarr

from eubi_bridge.utils.storage_utils import make_kvstore
from eubi_bridge.utils.logging_config import get_logger

logger = get_logger(__name__)


def simple_downscale(
                     darr,
                     scale_factor: Union[tuple, list, np.ndarray] = None,
                     backend = 'numpy' # placeholder
                     ):
    """Downscale a Dask array using simple stride slicing.
    
    Parameters
    ----------
    darr : dask.array.Array
        Input Dask array to downscale.
    scale_factor : Union[tuple, list, np.ndarray]
        Downsampling factors for each dimension.
    backend : str, optional
        Backend to use (placeholder for future use). Default is 'numpy'.
        
    Returns
    -------
    dask.array.Array
        Downscaled Dask array.
        
    Raises
    ------
    ValueError
        If scale_factor length doesn't match array dimensions.
    """
    if len(scale_factor) != darr.ndim:
        raise ValueError("scale_factors must have the same length as the array's number of dimensions")
    slices = tuple(slice(None, None, int(scale)) for scale in scale_factor)
    downscaled_arr = darr[slices]
    return downscaled_arr

def mean_downscale(arr: da.Array,
                   scale_factor: Union[tuple, list, np.ndarray] = None
                   ):
    """Downscale a Dask array using mean coarsening.
    
    Parameters
    ----------
    arr : dask.array.Array
        Input Dask array to downscale.
    scale_factor : Union[tuple, list, np.ndarray]
        Downsampling factors for each dimension.
        
    Returns
    -------
    dask.array.Array
        Downscaled Dask array with mean aggregation.
        
    Raises
    ------
    ValueError
        If scale_factor length doesn't match array dimensions.
    """
    if len(scale_factor) != arr.ndim:
        raise ValueError("scale_factors must have the same length as the array's number of dimensions")
    axes = dict({idx: factor for idx, factor in enumerate(scale_factor)})
    downscaled_arr = da.coarsen(da.mean, arr,
                                axes = axes, trim_excess = True).astype(arr.dtype)
    return downscaled_arr

def median_downscale(arr: da.Array,
                   scale_factor: Union[tuple, list, np.ndarray] = None
                   ):
    """Downscale a Dask array using median coarsening.
    
    Parameters
    ----------
    arr : dask.array.Array
        Input Dask array to downscale.
    scale_factor : Union[tuple, list, np.ndarray]
        Downsampling factors for each dimension.
        
    Returns
    -------
    dask.array.Array
        Downscaled Dask array with median aggregation.
        
    Raises
    ------
    ValueError
        If scale_factor length doesn't match array dimensions.
    """
    if len(scale_factor) != arr.ndim:
        raise ValueError("scale_factors must have the same length as the array's number of dimensions")
    axes = dict({idx: factor for idx, factor in enumerate(scale_factor)})
    downscaled_arr = da.coarsen(da.median, arr,
                                axes = axes, trim_excess = True).astype(arr.dtype)
    return downscaled_arr

async def ts_downscale(arr: Union[zarr.Array, str],
                          scale_factor: Union[tuple, list, np.ndarray] = None,
                          downscale_method: str = 'stride'
                          ):
    # 'simple' is the dask-path alias for stride-based subsampling
    ts_method = 'stride' if downscale_method == 'simple' else downscale_method
    return ts.downsample(arr,
                         [int(np.round(factor)) for factor in scale_factor],
                         method=ts_method)  # type: ignore[arg-type]

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


@dataclasses.dataclass
class DownscaleManager:
    base_shape: Union[list, tuple]
    scale_factor: Union[list, tuple]
    n_layers: Union[list, tuple]
    scale: Union[list, tuple] = None
    smart_scale_factor: Union[list, tuple] = None  # isotropic first-level factors

    def __post_init__(self):
        ndim = len(self.base_shape)
        assert len(self.scale_factor) == ndim

    @property
    def _scale_ids(self):
        return np.arange(self.n_layers).reshape(-1, 1)

    @property
    def _theoretical_scale_factors(self):
        if self.smart_scale_factor is None:
            return np.power(self.scale_factor, self._scale_ids)
        smart = np.array(self.smart_scale_factor, dtype=float)
        user = np.array(self.scale_factor, dtype=float)
        # If smart factors are all 1 (data already isotropic, e.g. 2D),
        # no anisotropy correction is needed → fall back to regular downscaling.
        if np.all(smart == 1.0):
            return np.power(self.scale_factor, self._scale_ids)
        # Two-phase: level 1 = smart_factor; levels 2+ = smart_factor * user_factor^(k-1)
        n = self.n_layers
        ndim = len(self.base_shape)
        result = np.zeros((n, ndim))
        result[0] = 1.0
        for k in range(1, n):
            result[k] = smart * np.power(user, k - 1)
        return result

    @property
    def output_shapes(self): # TODO: parameterize this for floor or ceil
        # shapes = np.floor_divide(self.base_shape, self._theoretical_scale_factors)
        shapes = np.ceil(np.divide(self.base_shape, self._theoretical_scale_factors))
        shapes[shapes == 0] = 1
        return shapes.astype(int)

    @property
    def scale_factors(self):
        return self._theoretical_scale_factors

    @property
    def scales(self):
        return np.multiply(self.scale, self._theoretical_scale_factors)




@dataclasses.dataclass
class Downscaler:
    array: Union[da.Array, zarr.Array, str]
    scale_factor: Union[list, tuple]
    n_layers: int
    scale: Union[list, tuple] = None
    output_chunks: Union[list, tuple] = None
    backend: str = 'numpy'
    downscale_method: str = 'simple'
    smart_scale_factor: Union[list, tuple] = None  # isotropic first-level factors
    
    def get_tensorstore_context(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve tensorstore context from worker initialization, if available.
        
        Returns the context set by initialize_worker_process() if called in a worker
        process. Falls back to None if not in a worker context.
        
        Returns
        -------
        dict or None
            Tensorstore context dict with data_copy_concurrency limits, or None
            if not available (will use tensorstore defaults).
        """
        try:
            from eubi_bridge.conversion import worker_init
            return getattr(worker_init, '_tensorstore_context', None)
        except (ImportError, AttributeError):
            return None

    def __post_init__(self):
        # if self.output_chunks is None:
        #     self.output_chunks = [self.array.chunksize] * self.n_layers
        if isinstance(self.array, str):
            store_path = self.array
            kvstore = make_kvstore(store_path)
            self.base_array_root = os.path.abspath(self.array)
            ts_context = self.get_tensorstore_context()
            self.array = ts.open(
                {
                    "driver": "zarr",
                    "kvstore": kvstore
                },
                context=ts_context,
                open=True,
            ).result()
        elif isinstance(self.array, zarr.Array):
            try:
                self.base_array_root = os.path.abspath(str(self.array.store.root))
            except AttributeError:
                # Fallback for stores without .root attribute
                self.base_array_root = self.array.store.path
            arraypath = self.array.path
            
            logger = __import__('eubi_bridge.utils.logging_config', fromlist=['get_logger']).get_logger(__name__)
            logger.info(f"[Downscaler] base_array_root={self.base_array_root}")
            logger.info(f"[Downscaler] arraypath={arraypath}")
            logger.info(f"[Downscaler] store type={type(self.array.store)}")
            logger.info(f"[Downscaler] store={self.array.store}")
            
            # Create kvstore pointing to the zarr store ROOT, not the array path
            # The array path is specified separately in the 'path' parameter
            kvstore = make_kvstore(self.base_array_root)
            logger.info(f"[Downscaler] kvstore={kvstore}")

            # Use appropriate driver based on zarr format
            # zarr v2 uses "zarr2" driver, zarr v3 uses "zarr3" driver
            zarr_format = self.array.metadata.zarr_format
            driver_name = "zarr3" if zarr_format == 3 else "zarr2"
            logger.info(f"[Downscaler] Opening with driver={driver_name}, path={arraypath}")
            ts_context = self.get_tensorstore_context()
            self.array = ts.open(
                {
                    "driver": driver_name,
                    "kvstore": kvstore,
                    "path": arraypath  # Specify array path separately
                },
                context=ts_context,
                open=True,
            ).result()
        else:
            self.base_array_root = None

        self.param_names = ['array', 'scale_factor', 'n_layers', 'scale', 'output_chunks', 'backend', 'downscale_method', 'smart_scale_factor']
        # self.update()

    def get_method(self):
        if self.base_array_root is None: # array is dask array
            if self.downscale_method == 'simple':
                method = simple_downscale
            elif self.downscale_method == "mean":
                method = mean_downscale
            elif self.downscale_method == "median":
                method = mean_downscale
            else:
                raise NotImplementedError(f"Currently, only 'simple', 'mean' and 'median' methods are implemented.")
        else:
            method = ts_downscale
        return method

    async def run(self):
        # self.method = self.get_method()
        self.method = ts_downscale
        # assert isinstance(self.array, da.Array)
        self.dm = DownscaleManager(self.array.shape,
                                   self.scale_factor,
                                   self.n_layers,
                                   self.scale,
                                   smart_scale_factor=self.smart_scale_factor,
                                   )

        downscaled = {}
        for (idx,
             scale_factor) in enumerate(self.dm.scale_factors):
            if idx == 0:
                pass
            else:
                factor = tuple(int(np.round(x)) for x in scale_factor)
                res1 = asyncio.create_task(
                    self.method(self.array, scale_factor=factor, downscale_method=self.downscale_method),
                    name = f"downscale_{idx}"
                )
                downscaled[idx] = res1
        results = await asyncio.gather(*downscaled.values(), return_exceptions=False)
        self.downscaled_arrays = {'0': self.array}
        for idx in range(len(results)):
            self.downscaled_arrays[str(idx + 1)] = results[idx]
        return self

    async def update(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.param_names:
                self.__setattr__(key, value)
            else:
                logger.warning(f"The given parameter name '{key}' is not valid, ignoring it..")
        await self.run()
        return self
