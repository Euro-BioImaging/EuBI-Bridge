
default_axes = 'tczyx'

unit_map = {
    't': 'second',
    'c': 'Channel',
    'z': 'micrometer',
    'y': 'micrometer',
    'x': 'micrometer'
}



scale_factor_map = {
    't': 1,
    'c': 1,
    'z': 1,
    'y': 2,
    'x': 2
}

scale_map = {
    't': 1,
    'c': 1,
    'z': 1,
    'y': 1,
    'x': 1
}

type_map = {
    't': 'time',
    'c': 'channel',
    'z': 'space',
    'y': 'space',
    'x': 'space'
}

datasets_archetype = {
                '0': {
                    'array': None,
                    'axis_order': None,
                    'scale': None,
                    'unit_list': None,
                    'chunks': None,
                    'dtype': None
                },
                '1': {
                    'array': None,
                    'axis_order': None,
                    'scale': None,
                    'unit_list': None,
                    'chunks': None,
                    'dtype': None
                }
        }

NotMultiscalesException = Exception('Not applicable as this group is not a single image or label.')
NotImageException = Exception('Not applicable as this group is not an image.')
AxisNotFoundException = Exception('Axis information not provided in the metadata.')

TempDir = '/tmp/OME-Zarr'

refpath = '0'

rechunker_maxmem = '1G'

array_meta_keys = ['chunks', 'shape', 'compressor', 'dtype', 'dimension_separator']

