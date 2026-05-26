"""Background worker that trains or applies the RF classifier on a region."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from eubi_annotate.core.annotation import (
    predict_region,
    train_classifier_region,
)


class ClassifierWorker(QThread):
    """Trains or applies a RandomForestClassifier within a rectangular region.

    If *existing_model* is supplied (annotations not dirty), prediction runs
    directly without retraining.  Otherwise the model is trained on labeled
    pixels in *label_region* before predicting.

    Emits ``finished(prediction, clf)`` on success, ``error(str)`` on failure.
    """

    finished = pyqtSignal(object, object)   # (prediction: np.ndarray, clf)
    error    = pyqtSignal(str)

    def __init__(
        self,
        label_region: np.ndarray,
        slice_region: np.ndarray,
        sigmas: Sequence[float] = (1, 1.5, 2, 3, 4),
        gaussian_types: Sequence[str] = ('smooth', 'gradient', 'laplacian'),
        window_sizes: Sequence[int] = (3, 5, 7),
        window_types: Sequence[str] = ('mean', 'median', 'min', 'max'),
        include_raw: bool = True,
        use_dinov2: bool = False,
        dinov2_model: str = 'vits14',
        dinov2_device: Optional[str] = None,
        n_estimators: int = 100,
        classifier: str = 'rf',
        existing_model=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._label_region    = label_region
        self._slice_region    = slice_region
        self._sigmas          = sigmas
        self._gaussian_types  = gaussian_types
        self._window_sizes    = window_sizes
        self._window_types    = window_types
        self._include_raw     = include_raw
        self._use_dinov2      = use_dinov2
        self._dinov2_model    = dinov2_model
        self._dinov2_device   = dinov2_device
        self._n_estimators    = n_estimators
        self._classifier      = classifier
        self._existing_model  = existing_model

    # ── feature config forwarding helper ──────────────────────────────────────

    @property
    def _feat_kwargs(self) -> dict:
        return dict(
            sigmas=self._sigmas,
            gaussian_types=self._gaussian_types,
            window_sizes=self._window_sizes,
            window_types=self._window_types,
            include_raw=self._include_raw,
            use_dinov2=self._use_dinov2,
            dinov2_model=self._dinov2_model,
            dinov2_device=self._dinov2_device,
        )

    def run(self) -> None:
        try:
            if self._existing_model is not None:
                clf = self._existing_model
            else:
                flat = self._label_region.ravel()
                labeled = flat > 0
                unique = np.unique(flat[labeled])
                if not labeled.any() or len(unique) < 2:
                    raise ValueError(
                        "No labels in the annotation box and no existing model. "
                        "Paint at least 2 classes inside the box first."
                    )
                clf = train_classifier_region(
                    label_region=self._label_region,
                    slice_region=self._slice_region,
                    n_estimators=self._n_estimators,
                    classifier=self._classifier,
                    **self._feat_kwargs,
                )

            prediction = predict_region(clf, self._slice_region, **self._feat_kwargs)
            self.finished.emit(prediction, clf)
        except Exception as exc:
            self.error.emit(str(exc))
