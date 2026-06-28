import asyncio
from typing import Union

from eubi_bridge.core.config_models import MetadataUpdateConfig
from eubi_bridge.core.data_manager import ArrayManager
from eubi_bridge.utils.logging_config import get_logger
from eubi_bridge.utils.metadata_utils import parse_channels
from eubi_bridge.utils.path_utils import is_zarr_group

logger = get_logger(__name__)


def parse_scales(manager, **kwargs) -> dict:
    """Return per-axis scale dict, merging kwargs overrides with manager defaults."""
    return MetadataUpdateConfig(**kwargs).scales_for(manager)


def parse_units(manager, **kwargs) -> dict:
    """Return per-axis unit dict, merging kwargs overrides with manager defaults."""
    return MetadataUpdateConfig(**kwargs).units_for(manager)


async def update_worker(input_path: Union[str, ArrayManager], **kwargs):
    cfg = MetadataUpdateConfig(**kwargs)

    if not is_zarr_group(input_path):
        raise Exception("Metadata update only works with OME-Zarr datasets.")

    manager = ArrayManager(
        input_path,
        series=0,
        metadata_reader=kwargs.get("metadata_reader", "bfio"),
        skip_dask=kwargs.get("skip_dask", True),
    )
    await manager.init()

    manager.fill_default_meta()
    manager.fix_bad_channels()

    if cfg.squeeze:
        manager.squeeze()

    crop_slices = cfg.crop_slices()
    if any(s is not None for s in crop_slices):
        manager.crop(*crop_slices)

    manager.update_meta(
        new_scaledict=cfg.scales_for(manager),
        new_unitdict=cfg.units_for(manager),
    )

    await manager.sync_pyramid(save_changes=False)

    channels = parse_channels(manager, **kwargs)
    meta = manager.pyr.meta
    meta.metadata["omero"]["channels"] = channels

    if meta.zarr_group is not None:
        if "ome" not in meta.zarr_group.attrs:
            meta.zarr_group.attrs.update({"omero": []})

    meta._pending_changes = True
    meta.save_changes()

    if cfg.save_omexml:
        await manager.save_omexml(input_path, overwrite=True)


def update_worker_sync(input_path, kwargs: dict):
    return asyncio.run(update_worker(input_path, **kwargs))
