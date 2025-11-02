# # from concurrent.futures import ProcessPoolExecutor
# import copy
# from typing import Union
# import tensorstore as ts
# import zarr, os, asyncio
# import numpy as np
# from eubi_bridge.conversion.fileset_io import BatchFile, FileSet
# from eubi_bridge.utils.convenience import take_filepaths, ChannelMap
# from eubi_bridge.utils.metadata_utils import generate_channel_metadata
# from eubi_bridge.core.readers import read_single_image_asarray
# from eubi_bridge.core.data_manager import ArrayManager
# from eubi_bridge.core.writers import write_with_tensorstore_async, _get_or_create_multimeta, _create_zarr_v2_array, CompressorConfig
# from eubi_bridge.core.writers import store_multiscale_async
# from eubi_bridge.conversion.fileset_io import BatchFile
# from eubi_bridge.conversion.aggregative_conversion_base import AggregativeConverter
# import time
# import multiprocessing as mp
#
# from eubi_bridge.utils.logging_config import get_logger
# logger = get_logger(__name__)
#
# output_path = f"/home/oezdemir/PycharmProjects/TIM2025/data/example_images1/test9out/00001_01.zarr"
#
# output_path = f"/home/oezdemir/PycharmProjects/TIM2025/data/example_images1/test9out/imageJ_test.zarr"
#
# chman = ArrayManager(output_path)
# await chman.init()
# channels = parse_channels(chman,
#                           channel_intensity_limits='from_array',
#                           channel_indices='all')
# meta = chman.pyr.meta
# meta.metadata['omero']['channels'] = channels
# meta._pending_changes = True
# meta.save_changes()