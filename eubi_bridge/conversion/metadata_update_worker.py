import asyncio


async def update_worker(input_path: str, **kwargs):
    """Edit the metadata of an existing OME-Zarr in place: one ome_zarr_pyramid
    ``update_metadata`` call (conversion/metadata_path.py, roadmap step 4).

    Metadata only: the arrays are never touched, so conversion settings that
    change pixels (squeeze, crop ranges, dtype) are ignored here.
    """
    from eubi_bridge.conversion import metadata_path
    await asyncio.to_thread(metadata_path.update_store, input_path, kwargs)


def update_worker_sync(input_path, kwargs: dict):
    return asyncio.run(update_worker(input_path, **kwargs))
