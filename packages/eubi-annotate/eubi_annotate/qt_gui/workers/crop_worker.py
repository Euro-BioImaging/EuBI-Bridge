"""Background worker that runs crop_ome_zarr in a dedicated event loop."""
from __future__ import annotations

import asyncio
from typing import Dict, Optional, Tuple

from PyQt6.QtCore import QThread, pyqtSignal

from eubi_annotate.core.crop import crop_ome_zarr


class CropWorker(QThread):
    """Runs crop_ome_zarr asynchronously in a background thread."""

    finished = pyqtSignal(bool, str)   # success, message

    def __init__(
        self,
        source_path: str,
        output_path: str,
        crop_ranges: Dict[str, Tuple[int, int]],
        zarr_format: int = 2,
        scale_factors: Optional[Tuple[int, ...]] = None,
        n_layers: Optional[int] = None,
        overwrite: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._source_path  = source_path
        self._output_path  = output_path
        self._crop_ranges  = crop_ranges
        self._zarr_format  = zarr_format
        self._scale_factors = scale_factors
        self._n_layers     = n_layers
        self._overwrite    = overwrite

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                crop_ome_zarr(
                    source_path=self._source_path,
                    output_path=self._output_path,
                    crop_ranges=self._crop_ranges,
                    zarr_format=self._zarr_format,
                    scale_factors=self._scale_factors,
                    n_layers=self._n_layers,
                    overwrite=self._overwrite,
                )
            )
            self.finished.emit(True, "Crop complete.")
        except Exception as exc:
            self.finished.emit(False, str(exc))
        finally:
            loop.close()
            asyncio.set_event_loop(None)
