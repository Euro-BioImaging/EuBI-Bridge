import sys, copy
import zarr
from eubi_bridge.ngff import defaults
from eubi_bridge.utils import convenience as cnv
from pathlib import Path
from typing import (Union, Iterable)
import numpy as np
from eubi_bridge.base.scale import Downscaler
# from base.writers import store_arrays
import dask.array as da


class Multimeta: # Maybe add updateable "_shape"
    def __init__(self,
                 multimeta = None
                 ):
        if multimeta is None:
            self.multimeta = [{'axes': [],
                               'datasets': [],
                               'name': "default",
                               'version': "0.4"
                               }]
        else:
            self.multimeta = multimeta
        self.gr = None

    def __repr__(self):
        return f"Multiscales metadata for indices: {self.axis_order}"

    def from_ngff(self,
                  store: (zarr.Group, zarr.storage.Store, Path, str)
                  ):
        if isinstance(store, zarr.Group):
            self.gr = store
        else:
            self.gr = zarr.open_group(store, mode = 'a')
        if 'multiscales' in self.gr.attrs.keys():
            self.multimeta = self.gr.attrs['multiscales']
        # else:
        #     print(f"The given store does not contain multiscales metadata.")
        return self

    def to_ngff(self,
                  store: (zarr.Group, zarr.storage.Store, Path, str)
                  ):
        try:
            if isinstance(store, zarr.Group):
                self.gr = store
            else:
                self.gr = zarr.open_group(store, mode = 'a')
            self.gr.attrs['multiscales'] = self.multimeta
        except:
            raise Exception(f"The given store is invalid")
        return self

    @property
    def axis_order(self):
        try:
            ret = ''.join([item['name'] for item in self.multimeta[0]['axes']])
        except:
            ret = ''
        return ret

    @property
    def has_axes(self):
        return len(self.axis_order) > 0

    @property
    def is_imglabel(self):
        try:
            func = self.__getattribute__('set_image_label_metadata')
            return func()
        except:
            return False

    @property
    def ndim(self):
        return len(self.axis_order)

    @property
    def nlayers(self):
        return len(self.resolution_paths)

    @property
    def unit_list(self):
        return [self.unit_dict[ax] for ax in self.axis_order]
    
    @property
    def unit_dict(self):
        try:
            if self.has_axes:
                d = {}
                for item in self.multimeta[0]['axes']:
                    if 'unit' in item.keys():
                        ax = item['name']
                        d[ax] = item['unit'].lower()
                    else:
                        ax = item['name']
                        default_unit = defaults.unit_map[ax].lower()
                        d[ax] = default_unit
                return d
            else:
                return defaults.AxisNotFoundException
        except:
            return defaults.NotMultiscalesException

    @property
    def tag(self):
        return self.multimeta[0]['name']

    def retag(self,
              new_tag: str
              ):
        self.multimeta[0]['name'] = new_tag
        return self

    def _add_axis(self,
                  name: str,
                  unit: str = None,
                  index: int = -1,
                  overwrite: bool = False
                  ):
        if name in self.axis_order:
            if not overwrite:
                raise ValueError(f'{name} axis already exists.')

        def axmake(name, unit):
            if unit is None or name == 'c':
                axis = {'name': name, 'type': defaults.type_map[name]}
            else:
                axis = {'name': name, 'type': defaults.type_map[name], 'unit': unit}
            return axis

        if index == -1:
            index = len(self.multimeta[0]['axes'])
        if not self.has_axes:
            self.multimeta[0]['axes'] = []
            index = 0
        axis = axmake(name, unit)
        if overwrite:
            self.multimeta[0]['axes'][index] = axis
        else:
            self.multimeta[0]['axes'].insert(index, axis)

    def parse_axes(self,
                   axis_order,
                   unit_list: Union[list, tuple] = None,
                   overwrite: bool = None
                   ):
        if len(self.multimeta[0]['axes']) > 0:
            if not overwrite:
                raise ValueError('The current axis metadata is not empty. Cannot overwrite.')
        if unit_list is None:
            unit_list = [None] * len(axis_order)
        elif unit_list == 'default':
            unit_list = [defaults.unit_map[i] for i in axis_order]

        if (len(axis_order) - len(unit_list)) > 1:
            raise ValueError(f"unit_list does not contain all essential units")
        if (len(axis_order) - len(unit_list)) == 1:
            assert 'c' in axis_order, f"Unit list is missing some of the essential axes."
            c_idx = axis_order.index('c')
            unit_list.insert(c_idx, None)

        assert len(axis_order) == len(unit_list), 'Unit list and axis order must have the same length.'
        for i, n in enumerate(axis_order):
            self._add_axis(name = n,
                           unit = unit_list[i],
                           index = i,
                           overwrite = overwrite
                           )
        return self

    def rename_paths(self):
        for i, _ in enumerate(self.multimeta[0]['datasets']):
            newkey = str(i)
            oldkey = self.multimeta[0]['datasets'][i]['path']
            self._arrays[newkey] = self._arrays.pop(oldkey)
            self._array_meta_[newkey] = self._array_meta_.pop(oldkey)
            self.multimeta[0]['datasets'][i]['path'] = newkey
        return self

    @property
    def resolution_paths(self):
        try:
            paths = [item['path'] for item in self.multimeta[0]['datasets']]
            return sorted(paths)
        except:
            return []

    def del_dataset(self, path: Union[str, int], hard = False):
        for i, dataset in enumerate(self.multimeta[0]['datasets']):
            if dataset['path'] == cnv.asstr(path):
                del self.multimeta[0]['datasets'][i]
        # if hard:
        #     self.gr.attrs['multiscales'] = self.multimeta
        return

    def add_dataset(self,
                    path: Union[str, int],
                    scale: Iterable[Union[int, float]],
                    translation: Iterable[Union[int, float]] = None,
                    overwrite: bool = False
                    ):
        if not overwrite:
            assert path not in self.resolution_paths, 'Path already exists.'
        assert scale is not None, f"The parameter scale must not be None"
        assert isinstance(scale, (tuple, list))
        transforms = {'scale': scale, 'translation': translation}
        dataset = {'coordinateTransformations': [{f'{key}': list(value), 'type': f'{key}'}
                                                 for key, value in transforms.items()
                                                 if not value is None
                                                 ],
                   'path': str(path)
                   }
        if path in self.resolution_paths:
            idx = self.resolution_paths.index(path)
            self.multimeta[0]['datasets'][idx] = dataset
        else:
            self.multimeta[0]['datasets'].append(dataset)
        args = np.argsort([int(pth) for pth in self.resolution_paths])
        self.multimeta[0]['datasets'] = [self.multimeta[0]['datasets'][i] for i in args]

    @property
    def transformation_types(self):
        transformations = self.multimeta[0]['datasets'][0]['coordinateTransformations']
        return [list(dict.keys())[0] for dict in transformations]

    @property
    def has_translation(self):
        return 'translation' in self.transformation_types

    def get_scaledict(self,
                      pth: Union[str, int]
                      ):
        pth = cnv.asstr(pth)
        idx = self.resolution_paths.index(pth)
        scale = self.multimeta[0]['datasets'][idx]['coordinateTransformations'][0]['scale']         
        return dict(zip(self.axis_order, scale))

    def get_base_scaledict(self):
        basepath = self.resolution_paths[0]
        return self.get_scaledict(basepath)

    def get_scale(self,
                  pth: Union[str, int]
                  ):
        scaledict = self.get_scaledict(pth)
        return [scaledict[ax] for ax in self.axis_order]
    
    def get_base_scale(self):
        basepath = self.resolution_paths[0]
        return self.get_scale(basepath)

    def set_scale(self,
                  pth: Union[str, int] = 'auto',
                  scale: Union[tuple, list, dict] = 'auto',
                  hard = False
                  ):
        if isinstance(scale, tuple):
            scale = list(scale)
            ch_index = self.axes.index('c')
            scale[ch_index] = 1
        elif hasattr(scale, 'tolist'):
            scale = scale.tolist()
        elif isinstance(scale, dict): # TODO: test this block further
            assert all([ax in self.axis_order for ax in scale])
            fullscale = self.get_scale(pth)
            scaledict = dict(zip(self.axis_order, fullscale))
            scaledict.update(**scale)
            scale = [scaledict[ax] for ax in self.axis_order]

        if pth == 'auto':
            pth = self.refpath
        if scale == 'auto':
            pth = self.scales[pth]
        idx = self.resolution_paths.index(pth)
        self.multimeta[0]['datasets'][idx]['coordinateTransformations'][0]['scale'] = scale
        if hard:
            self.gr.attrs['multiscales'] = self.multimeta
        return

    def update_scales(self,
                      reference_scale: Union[tuple, list],#, dict],
                      scale_factors: dict,
                      hard = True
                      ):
        for pth, factor in scale_factors.items():
            new_scale = np.multiply(factor, reference_scale)
            self.set_scale(pth, new_scale, hard)
        return self

    def update_unitlist(self,
                 unitlist=None,
                 hard=False
                 ):
        if isinstance(unitlist, tuple):
            unitlist = list(unitlist)
        assert isinstance(unitlist, list)
        self.parse_axes(self.axis_order, unitlist, overwrite = True)
        if hard:
            self.gr.attrs['multiscales'] = self.multimeta
        return

    @property
    def scales(self):
        scales = {}
        for pth in self.resolution_paths:
            scl = self.get_scale(pth)
            scales[pth] = scl
        return scales

    @property
    def scaledict(self):
        scales = {}
        for pth in self.resolution_paths:
            scl = self.get_scaledict(pth)
            scales[pth] = scl
        return scales

    def get_translation(self,
                        pth: Union[str, int]
                        ):
        if not self.has_translation: return None
        pth = cnv.asstr(pth)
        idx = self.resolution_paths.index(pth)
        return self.multimeta[0]['datasets'][idx]['coordinateTransformations'][1]['translation']

    def set_translation(self,
                        pth: Union[str, int],
                        translation
                        ):
        if isinstance(translation, np.ndarray):
            translation = translation.tolist()
        idx = self.resolution_paths.index(pth)
        if len(self.multimeta[0]['datasets'][idx]['coordinateTransformations']) < 2:
            self.multimeta[0]['datasets'][idx]['coordinateTransformations'].append({'translation': None, 'type': 'translation'})
        self.multimeta[0]['datasets'][idx]['coordinateTransformations'][1]['translation'] = translation

    @property
    def translations(self):
        translations = {}
        for pth in self.resolution_paths:
            translation = self.get_translation(pth)
            translations[pth] = translation
        return translations

    def del_axis(self,
                 name: str
                 ):
        if name not in self.axis_order:
            raise ValueError(f'The axis "{name}" does not exist.')

        for pth in self.resolution_paths:
            scaledict = self.get_scaledict(pth)
            scale = [scaledict[ax] for ax in self.axis_order if ax != name]
            idx = self.resolution_paths.index(pth)
            self.multimeta[0]['datasets'][idx]['coordinateTransformations'][0]['scale'] = scale
            translation = self.get_translation(pth)
            if translation is not None:
                self.multimeta[0]['datasets'][idx]['coordinateTransformations'][0]['translation'] = scale

        idx = self.axis_order.index(name)
        self.multimeta[0]['axes'].pop(idx)

    @property
    def label_paths(self):
        try:
            return list(self.labels.keys())
        except:
            return []

    @property
    def has_label_paths(self):
        try:
            return self.labels is not None
        except:
            return False

    @property
    def label_meta(self):
        if self.has_label_paths:
            meta = {'labels': []}
            for name in self.label_paths:
                meta['labels'].append(name)
            return meta
        else:
            return None


class Pyramid:
    def __init__(self,
                 gr: (zarr.Group, zarr.storage.Store, Path, str) = None # An NGFF group. This contains the multiscales metadata in attrs and image layers as
                 ):
        if gr is not None:
            self.from_ngff(gr)

    def __repr__(self):
        return f"NGFF with {self.nlayers} layers."

    def from_ngff(self, gr):
        self.meta = Multimeta().from_ngff(gr)
        self.gr = self.meta.gr
        return self

    @property
    def axes(self):
        return self.meta.axis_order

    @property
    def nlayers(self):
        return self.meta.nlayers

    @property
    def layers(self):
        return {path: self.gr[path] for path in self.gr.array_keys()}

    def get_dask_data(self):
        return {str(path): da.from_zarr(self.layers[path]) for path in self.gr.array_keys()}

    @property
    def dask_arrays(self):
        return self.get_dask_data()

    @property
    def base_array(self):
        return self.dask_arrays['0']

    def shrink(self,
               hard = False
               ):
        """
        Delete all arrays except for the base array (zeroth array)
        :return:
        """
        for key in self.meta.resolution_paths:
            if key == '0':
                continue
            self.meta.del_dataset(key)
            if hard:
                del self.gr[key]
        if hard:
            self.meta.to_ngff(self.gr)
        return

    def update_scales(self,
                      **kwargs
                      ):
        """
        Automatically updates all pixel values for all layers based on
        provided pixel values for specific axes corresponding to
        the top resolution layer.
        :param kwargs:
        :return:
        """
        hard = kwargs.get('hard', False)
        
        new_scaledict = self.meta.get_base_scaledict()
        for ax in self.meta.axis_order:
            if ax in kwargs:
                new_scaledict[ax] = kwargs.get(ax)                
        new_scale = [new_scaledict[ax] for ax in self.meta.axis_order]
        ###
        shapes = [self.layers[key].shape for key in self.meta.resolution_paths]
        scale_factors = np.divide(shapes[0], shapes)
        scale_factordict = {pth: scale
                            for pth, scale in 
                            zip(self.meta.resolution_paths, scale_factors.tolist())}
        self.meta.update_scales(reference_scale = new_scale,
                                scale_factors = scale_factordict,
                                hard = hard
                                )

    def update_units(self,
                      **kwargs
                      ):
        """
        Automatically updates all pixel units based on provided unit strings for each axis.
        :param kwargs:
        :return:
        """
        hard = kwargs.get('hard', False)

        new_unitdict = self.meta.unit_dict
        for ax in self.meta.axis_order:
            if ax in kwargs:
                new_unitdict[ax] = kwargs.get(ax)
        new_unitlist = [new_unitdict[ax] for ax in self.meta.axis_order]
        ###
        self.meta.update_unitlist(unitlist=new_unitlist, hard = hard)

    def update_downscaler(self,
                          scale_factor = None,
                          n_layers = 1,
                          downscale_method='simple',
                          backend = 'numpy',
                          **kwargs
                          ):
        darr = self.base_array
        if scale_factor is None:
            scale_factor = tuple([defaults.scale_factor_map[key] for key in self.axes])
        scale = self.meta.scales['0']
        scale_factor = tuple(np.minimum(darr.shape, scale_factor))
        self.downscaler = Downscaler(array=darr,
                                     scale_factor=scale_factor,
                                     n_layers=n_layers,
                                     scale=scale,
                                     downscale_method = downscale_method,
                                     backend = backend
                                     )
        return self
