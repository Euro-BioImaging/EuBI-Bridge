import os.path

import zarr, dataclasses
from pathlib import Path
import numpy as np, zarr
import dask.array as da
from typing import Callable
import tensorstore as ts
import asyncio


def simple_downscale(
                     darr,
                     scale_factor: (tuple, list, np.ndarray) = None,
                     backend = 'numpy' # placeholder
                     ):
    """
    Downscale a Dask array along each dimension by given scale factors.

    Parameters:
    arr (dask.array): The input n-dimensional Dask array.
    scale_factors (tuple): The downsampling factors for each dimension.

    Returns:
    dask.array: The downscaled Dask array.
    """
    if len(scale_factor) != darr.ndim:
        raise ValueError("scale_factors must have the same length as the array's number of dimensions")
    slices = tuple(slice(None, None, int(scale)) for scale in scale_factor)
    downscaled_arr = darr[slices]
    return downscaled_arr

def mean_downscale(arr: da.Array,
                   scale_factor: (tuple, list, np.ndarray) = None
                   ):
    if len(scale_factor) != arr.ndim:
        raise ValueError("scale_factors must have the same length as the array's number of dimensions")
    axes = dict({idx: factor for idx, factor in enumerate(scale_factor)})
    downscaled_arr = da.coarsen(da.mean, arr,
                                axes = axes, trim_excess = True).astype(arr.dtype)
    return downscaled_arr

def median_downscale(arr: da.Array,
                   scale_factor: (tuple, list, np.ndarray) = None
                   ):
    if len(scale_factor) != arr.ndim:
        raise ValueError("scale_factors must have the same length as the array's number of dimensions")
    axes = dict({idx: factor for idx, factor in enumerate(scale_factor)})
    downscaled_arr = da.coarsen(da.median, arr,
                                axes = axes, trim_excess = True).astype(arr.dtype)
    return downscaled_arr

async def ts_downscale(arr: (zarr.Array, str),
                          scale_factor: (tuple, list, np.ndarray) = None
                          ):
    # Method 1: Try using tensorstore's virtual downsampling if available
    return ts.downsample(arr,
                         [int(np.round(factor)) for factor in scale_factor],
                         method='stride')

@dataclasses.dataclass
class DownscaleManager:
    base_shape: (list, tuple)
    scale_factor: (list, tuple)
    n_layers: (list, tuple)
    scale: (list, tuple) = None

    def __post_init__(self):
        ndim = len(self.base_shape)
        assert len(self.scale_factor) == ndim

    @property
    def _scale_ids(self):
        return np.arange(self.n_layers).reshape(-1, 1)

    @property
    def _theoretical_scale_factors(self):
        return np.power(self.scale_factor, self._scale_ids)

    @property
    def output_shapes(self): # TODO: parameterize this for floor or ceil
        # shapes = np.floor_divide(self.base_shape, self._theoretical_scale_factors)
        shapes = np.ceil(np.divide(self.base_shape, self._theoretical_scale_factors))
        shapes[shapes == 0] = 1
        return shapes.astype(int)

    @property
    def scale_factors(self):
        return np.true_divide(self.output_shapes[0], self.output_shapes)

    @property
    def scales(self):
        return np.multiply(self.scale, self.scale_factors)


@dataclasses.dataclass
class Downscaler:
    array: (da.Array, zarr.Array, str)
    scale_factor: (list, tuple)
    n_layers: int
    scale: (list, tuple) = None
    output_chunks: (list, tuple) = None
    backend: str = 'numpy'
    downscale_method: str = 'simple'

    def __post_init__(self):
        # if self.output_chunks is None:
        #     self.output_chunks = [self.array.chunksize] * self.n_layers
        if isinstance(self.array, str):
            self.base_array_root = os.path.abspath(self.array)
            self.downscale_method = 'ts'
            self.array = ts.open(
                {
                    "driver": "zarr",
                    "kvstore": {
                        "driver": "file",
                        "path": self.base_array_root
                    }
                },
                open=True,
            ).result()
        elif isinstance(self.array, zarr.Array):
            self.base_array_root = str(self.array.store_path)
            self.downscale_method = 'ts'
            self.array = ts.open(
                self.base_array_root,
                open=True,
            ).result()
        else:
            self.base_array_root = None

        self.param_names = ['array', 'scale_factor', 'n_layers', 'scale', 'output_chunks', 'backend', 'downscale_method']
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
                                   self.scale
                                   )

        downscaled = {}
        for (idx,
             scale_factor) in enumerate(self.dm.scale_factors):
            if idx == 0:
                pass
            else:
                factor = tuple(int(np.round(x)) for x in scale_factor)
                res1 = asyncio.create_task(
                    self.method(self.array, scale_factor = factor),
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
                warnings.warn(f"The given parameter name '{key}' is not valid, ignoring it..")
        await self.run()
        return self
