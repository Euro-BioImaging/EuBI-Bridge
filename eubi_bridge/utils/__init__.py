"""EuBI-Bridge utilities module.

This module provides convenient access to utility functions across several submodules.
For backward compatibility, all previously available functions from convenience.py
are re-exported here.
"""

# Array utilities
from eubi_bridge.utils.array_utils import (as_dask_array, asdask,
                                           autocompute_chunk_shape,
                                           compute_chunk_batch, get_array_size,
                                           get_chunk_shape,
                                           get_chunksize_from_array,
                                           parse_memory, sizeof)
# JSON utilities
from eubi_bridge.utils.json_utils import (convert_np_types, is_valid_json,
                                          make_json_safe, turn2json)
# JVM management
from eubi_bridge.utils.jvm_manager import (find_libjvm, soft_start_jvm)
# Miscellaneous utilities
from eubi_bridge.utils.misc_utils import (ChannelMap, argsorter, as_store,
                                          asstr, get_collection_paths,
                                          index_nth_dimension,
                                          insert_at_indices,
                                          is_generic_collection,
                                          retry_decorator, transpose_dict)
# Path utilities
from eubi_bridge.utils.path_utils import (find_common_root,
                                          find_common_root_relative, includes,
                                          is_zarr_array, is_zarr_group,
                                          parse_as_list, path_has_pyramid,
                                          sensitive_glob, take_filepaths,
                                          take_filepaths_from_path)
# Storage utilities
from eubi_bridge.utils.storage_utils import make_kvstore
# Unit conversion
from eubi_bridge.utils.unit_converter import abbreviate_units, expand_units

__all__ = [
    # JVM
    'soft_start_jvm',
    'find_libjvm',
    # Arrays
    'asdask',
    'as_dask_array',
    'get_array_size',
    'sizeof',
    'get_chunk_shape',
    'get_chunksize_from_array',
    'parse_memory',
    'autocompute_chunk_shape',
    'compute_chunk_batch',
    # Paths
    'parse_as_list',
    'includes',
    'path_has_pyramid',
    'is_zarr_array',
    'is_zarr_group',
    'sensitive_glob',
    'take_filepaths_from_path',
    'take_filepaths',
    'find_common_root',
    'find_common_root_relative',
    # JSON
    'convert_np_types',
    'is_valid_json',
    'turn2json',
    'make_json_safe',
    # Units
    'abbreviate_units',
    'expand_units',
    # Misc
    'asstr',
    'transpose_dict',
    'argsorter',
    'insert_at_indices',
    'index_nth_dimension',
    'is_generic_collection',
    'get_collection_paths',
    'as_store',
    'retry_decorator',
    'ChannelMap',
    # Storage
    'make_kvstore',
]

