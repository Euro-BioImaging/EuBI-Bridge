import fsspec
import numpy as np

from dask import delayed
import dask, zarr, asyncio
from eubi_bridge.utils.convenience import soft_start_jvm
# soft_start_jvm()

from eubi_bridge.ngff.multiscales import Pyramid

from bioio_bioformats import utils

from eubi_bridge.utils.logging_config import get_logger

logger = get_logger(__name__)

# The block below moved to the 'ebridge.py' module in the 'to_zarr' method.
# import scyjava
# import jpype
# # IMPORTANT: jvm must be started before importing bioio_bioformats readers
# if not scyjava.jvm_started():
#     scyjava.config.endpoints.append("ome:formats-gpl:6.7.0")
#     scyjava.start_jvm()


async def read_single_image(input_path,
                            aszarr=False,
                            **kwargs
                            ):
    """
    Reads a single image file.

    Parameters
    ----------
    input_path : str
        Path to the image file.
    **kwargs : dict
        Additional keyword arguments, such as `verbose` and `scene`.

    Returns
    -------
    arr : dask.array.Array
        The image array.
    """
    from eubi_bridge.utils.logging_config import get_logger
    logger = get_logger(__name__)
    reader_kwargs = {}
    dimensions = 'TCZYX'

    if input_path.endswith('.zarr'):
        from eubi_bridge.core.pyramid_reader import read_pyramid as reader
        kwargs['aszarr'] = aszarr
    elif input_path.endswith('.h5'):
        from eubi_bridge.core.h5_reader import read_h5 as reader
    elif input_path.endswith(('.ome.tiff', '.ome.tif')) and not aszarr:
        from eubi_bridge.core.pff_reader import read_pff as reader
    elif input_path.endswith(('.tif', '.tiff')):
        from eubi_bridge.core.tiff_reader import read_tiff_image as reader
        kwargs['aszarr'] = aszarr
    elif input_path.endswith('.lsm'):
        from eubi_bridge.core.tiff_reader import read_tiff_with_bioio as reader
        kwargs['aszarr'] = False
    else: ### is another kind of pff, will use bioformats for reading
        from eubi_bridge.core.pff_reader import read_pff as reader
    logger.info(f"Reading with '{reader.__qualname__}': {input_path}.")
    verbose = kwargs.get('verbose', False)
    if verbose:
        logger.info(f"The aszarr option is {aszarr}")

    img = reader(input_path, **kwargs)
    series = kwargs.get('scene_index', None)
    if series is not None:
        img.set_scene(series)
    if verbose:
        logger.info(f"Current series index is {series}")
    return img



def read_image_sync(path, **kwargs):
    return asyncio.run(read_single_image(path, **kwargs))

@delayed
def read_single_image_delayed(path, **kwargs):
    return read_image_sync(path, **kwargs)
    

def get_metadata_reader_by_path(input_path, **kwargs):
    if input_path.endswith(('ome.tiff', 'ome.tif')):
        from bioio_ome_tiff.reader import Reader as reader # pip install bioio-ome-tiff --no-deps
    elif input_path.endswith(('.tif', '.tiff', '.lsm')):
        from bioio_tifffile.reader import Reader as reader
    elif input_path.endswith('.czi'):
        from bioio_czi.reader import Reader as reader
    elif input_path.endswith('.lif'):
        from bioio_lif.reader import Reader as reader
    elif input_path.endswith('.nd2'):
        from bioio_nd2.reader import Reader as reader
    elif input_path.endswith(('.png','.jpg','.jpeg')):
        from bioio_imageio.reader import Reader as reader
    else:
        from bioio_bioformats.reader import Reader as reader
    return reader

def read_metadata_via_bioio_bioformats(input_path, **kwargs):
    from bioio_bioformats.reader import Reader
    series = kwargs.get('series', None)
    try:
        img = Reader(input_path)
        if series is not None:
            img.set_scene(series)
        omemeta = img.ome_metadata
    except FileNotFoundError as e:
        if ".jgo" in str(e):
            raise RuntimeError("JGO cache may be corrupted. Run `rm -rf ~/.jgo/` and retry.") from e
    return omemeta

def read_metadata_via_bfio(input_path, **kwargs):
    from bfio import BioReader
    try:
        omemeta = BioReader(input_path, backend='bioformats').metadata
    except FileNotFoundError as e:
        if ".jgo" in str(e):
            raise RuntimeError("JGO cache may be corrupted. Run `rm -rf ~/.jgo/` and retry.") from e
    return omemeta

def read_metadata_via_extension(input_path, **kwargs):
    Reader = get_metadata_reader_by_path(input_path)
    series = kwargs.get('series', None)
    try:
        img = Reader(input_path)
        if series is not None:
            img.set_scene(series)
        omemeta = img.ome_metadata
    except FileNotFoundError as e:
        if ".jgo" in str(e):
            raise RuntimeError("JGO cache may be corrupted. Run `rm -rf ~/.jgo/` and retry.") from e
    return omemeta
