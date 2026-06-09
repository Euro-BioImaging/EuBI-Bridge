"""
In-memory annotation store and scikit-learn Random Forest pixel classifier.

AnnotationStore wraps a uint8 ndarray whose shape matches the source image
(axes: tczyx or any subset). Users paint class labels into the 2-D view plane;
the full nD array is written out as an NGFF label layer on save.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

# Orientation → (v_axis, h_axis, through_axis_key)
_ORI = {
    "XY": ("y", "x", "z"),
    "XZ": ("z", "x", "y"),
    "YZ": ("y", "z", "x"),
}


class AnnotationStore:
    """Holds per-pixel class labels for a single OME-Zarr dataset."""

    def __init__(self, shape: Tuple[int, ...], axes: str) -> None:
        self._array = np.zeros(shape, dtype=np.uint8)
        self._axes  = axes.lower()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_index(self, t: int, z: int, orientation: str) -> tuple:
        """Return a full index tuple that selects the 2-D (H×W) slice."""
        axes = self._axes
        v_ax, h_ax, through_ax = _ORI[orientation]
        idx = []
        for ax in axes:
            if ax == 't':
                idx.append(t)
            elif ax == 'c':
                idx.append(slice(None))  # collapse below
            elif ax == through_ax:
                idx.append(z)
            else:
                idx.append(slice(None))
        return tuple(idx)

    def _get_2d(self, t: int, z: int, orientation: str) -> np.ndarray:
        """Return the 2-D (H×W) annotation slice (view, not copy)."""
        idx = self._build_index(t, z, orientation)
        arr = self._array[idx]
        # If 'c' is in axes the slice still has a channel dim; collapse via max
        if 'c' in self._axes:
            arr = arr.max(axis=0)
        return arr

    # ── public API ────────────────────────────────────────────────────────────

    def paint(
        self,
        t: int,
        z: int,
        orientation: str,
        row: int,
        col: int,
        radius: int,
        class_idx: int,
    ) -> None:
        """Paint a filled circle of *class_idx* at (row, col) in the 2-D plane."""
        mask_2d = self._get_2d(t, z, orientation)
        h, w = mask_2d.shape
        r = max(0, radius)
        r0, r1 = max(0, row - r), min(h, row + r + 1)
        c0, c1 = max(0, col - r), min(w, col + r + 1)
        rr, cc = np.ogrid[r0 - row : r1 - row, c0 - col : c1 - col]
        circle = (rr ** 2 + cc ** 2) <= r ** 2
        mask_2d[r0:r1, c0:c1][circle] = class_idx

        # Write back (needed when 'c' was present and _get_2d collapsed it)
        if 'c' in self._axes:
            idx = self._build_index(t, z, orientation)
            # Set all channels to the same value in the painted region
            for ci in range(self._array.shape[self._axes.index('c')]):
                full_idx = list(idx)
                full_idx[self._axes.index('c')] = ci
                self._array[tuple(full_idx)][r0:r1, c0:c1][circle] = class_idx

    def get_slice_2d(self, t: int, z: int, orientation: str) -> np.ndarray:
        """Return a copy of the (H×W) uint8 annotation slice."""
        return self._get_2d(t, z, orientation).copy()

    def get_region_2d(
        self,
        t: int,
        z: int,
        orientation: str,
        v0: int,
        v1: int,
        h0: int,
        h1: int,
    ) -> np.ndarray:
        """Return a copy of the annotation sub-region [v0:v1, h0:h1].

        Avoids copying the full annotation array — only the requested window
        is materialised.  Coordinates are clipped to the array bounds.
        """
        full = self._get_2d(t, z, orientation)   # view, no copy
        H, W = full.shape
        r0 = max(0, min(v0, H - 1))
        r1 = max(r0 + 1, min(v1, H))
        c0 = max(0, min(h0, W - 1))
        c1 = max(c0 + 1, min(h1, W))
        return full[r0:r1, c0:c1].copy()

    def clear_slice(self, t: int, z: int, orientation: str) -> None:
        mask = self._get_2d(t, z, orientation)
        mask[:] = 0

    def set_slice_2d(self, t: int, z: int, orientation: str, mask_2d: np.ndarray) -> None:
        """Replace the entire 2D annotation slice with *mask_2d* values."""
        axes = self._axes
        if 'c' in axes:
            idx = self._build_index(t, z, orientation)
            n_c = self._array.shape[axes.index('c')]
            for ci in range(n_c):
                full_idx = list(idx)
                full_idx[axes.index('c')] = ci
                self._array[tuple(full_idx)][:] = mask_2d
        else:
            idx = self._build_index(t, z, orientation)
            self._array[idx][:] = mask_2d

    def set_region_2d(
        self,
        t: int,
        z: int,
        orientation: str,
        v_start: int,
        v_end: int,
        h_start: int,
        h_end: int,
        mask_region: np.ndarray,
    ) -> None:
        """Write *mask_region* into the sub-region [v_start:v_end, h_start:h_end]
        of the 2-D slice for (t, z, orientation).  Values of 0 are left unchanged
        so that the classifier doesn't erase existing manual annotations.
        """
        axes = self._axes
        if 'c' in axes:
            idx = self._build_index(t, z, orientation)
            n_c = self._array.shape[axes.index('c')]
            for ci in range(n_c):
                full_idx = list(idx)
                full_idx[axes.index('c')] = ci
                target = self._array[tuple(full_idx)][v_start:v_end, h_start:h_end]
                # Only write where the prediction is non-zero (preserve manual labels)
                nz = mask_region > 0
                target[nz] = mask_region[nz]
        else:
            idx = self._build_index(t, z, orientation)
            target = self._array[idx][v_start:v_end, h_start:h_end]
            nz = mask_region > 0
            target[nz] = mask_region[nz]

    def clear_all(self) -> None:
        self._array[:] = 0

    @property
    def array(self) -> np.ndarray:
        """Full nD uint8 annotation array (for saving)."""
        return self._array


# ── Label-offset wrapper ──────────────────────────────────────────────────────

class _OffsetClassifier:
    """Sklearn-compatible wrapper that shifts class labels to 0-indexed.

    XGBoost (and some other estimators) require labels in range 0..N-1.
    Our annotation store uses 1..N.  This wrapper records the minimum label
    seen during ``fit``, subtracts it before training, and adds it back after
    ``predict`` — so the surrounding ``Pipeline`` and all callers stay unaware
    of the shift.

    Inherits from ``BaseEstimator`` so scikit-learn ≥ 1.6 finds
    ``__sklearn_tags__`` without extra boilerplate.
    """

    def __init__(self, clf):
        # sklearn convention: attribute name must match __init__ parameter name
        # for BaseEstimator.get_params / set_params to work automatically.
        self.clf = clf
        self._offset = 0

    def fit(self, X, y):
        y = np.asarray(y)
        self._offset = int(y.min())
        self.clf.fit(X, y - self._offset)
        # classes_ is the standard sklearn "I am fitted" marker attribute.
        # __sklearn_is_fitted__ (below) also uses it.
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.asarray(self.clf.predict(X), dtype=np.int32) + self._offset

    def __sklearn_is_fitted__(self) -> bool:
        """Tell sklearn ≥ 1.6 check_is_fitted that we are ready."""
        return hasattr(self, 'classes_')

    # ── sklearn Pipeline compatibility ────────────────────────────────────────

    def get_params(self, deep=True):
        params = {'clf': self.clf}
        if deep:
            inner = self.clf.get_params(deep=True)
            params.update({f'clf__{k}': v for k, v in inner.items()})
        return params

    def set_params(self, **params):
        if 'clf' in params:
            self.clf = params.pop('clf')
        inner = {k[5:]: v for k, v in params.items() if k.startswith('clf__')}
        if inner:
            self.clf.set_params(**inner)
        return self

    def __sklearn_tags__(self):
        # Delegate to the wrapped estimator so sklearn's Pipeline validation
        # sees the correct tags (e.g. multiclass support).
        try:
            return self.clf.__sklearn_tags__()
        except AttributeError:
            from sklearn.utils._tags import get_tags
            return get_tags(self.clf)


# ── DINOv2 feature extractor ─────────────────────────────────────────────────

_dinov2_cache: dict = {}   # (model_name, device) → loaded model


def _get_dinov2_model(model_name: str, device: str):
    """Load and cache a DINOv2 model from torch.hub (downloaded once per session)."""
    key = (model_name, device)
    if key not in _dinov2_cache:
        import torch
        model = torch.hub.load(
            'facebookresearch/dinov2', f'dinov2_{model_name}', verbose=False
        )
        model.eval().to(device)
        _dinov2_cache[key] = model
    return _dinov2_cache[key]


def extract_dinov2_features(
    slice_region: np.ndarray,
    model_name: str = 'vits14',
    device: Optional[str] = None,
) -> np.ndarray:
    """Extract DINOv2 patch features and upsample to pixel resolution.

    DINOv2 operates at 14×14 px patch granularity.  The patch-level token
    embeddings are bilinearly interpolated back to the original ``(H, W)``
    resolution so they can be concatenated with per-pixel handcrafted features.

    The pre-trained ViT encoder is run with frozen weights — no fine-tuning.
    Its global self-attention means every token already encodes the full
    spatial context of the input patch (far beyond any Gaussian kernel).

    Parameters
    ----------
    slice_region : np.ndarray
        ``(H, W)`` or ``(H, W, C)`` float32 image data.
    model_name : str
        DINOv2 variant: ``'vits14'`` (384-d, ~86 MB),
        ``'vitb14'`` (768-d, ~330 MB), ``'vitl14'`` (1024-d), ``'vitg14'`` (1536-d).
    device : str or None
        ``'cpu'``, ``'cuda'``, or ``None`` (auto-detect).

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(H*W, D)`` where *D* is the model's embedding
        dimension.
    """
    import torch
    import torch.nn.functional as F

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    H, W = slice_region.shape[:2]
    img = slice_region.astype(np.float32)

    # DINOv2 expects 3-channel input — handle arbitrary channel counts.
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    else:
        C = img.shape[2]
        if C == 1:
            img = np.concatenate([img] * 3, axis=-1)
        elif C == 2:
            avg = ((img[..., 0] + img[..., 1]) / 2)[..., None]
            img = np.concatenate([img, avg], axis=-1)
        else:
            img = img[..., :3]   # use first 3 channels for C > 3

    # Normalise to [0, 1] then apply ImageNet mean/std.
    lo, hi = float(img.min()), float(img.max())
    img = (img - lo) / max(hi - lo, 1e-8)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img  = (img - mean) / std

    # Pad spatial dims to a multiple of the patch size (14).
    patch = 14
    H_pad = (-H) % patch
    W_pad = (-W) % patch
    if H_pad or W_pad:
        img = np.pad(img, ((0, H_pad), (0, W_pad), (0, 0)), mode='reflect')
    H_in, W_in = img.shape[:2]

    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    model = _get_dinov2_model(model_name, device)
    with torch.no_grad():
        out    = model.forward_features(tensor)
        tokens = out['x_norm_patchtokens']          # (1, n_patches, D)
        D      = tokens.shape[-1]
        h_p, w_p = H_in // patch, W_in // patch
        grid   = tokens.reshape(1, h_p, w_p, D).permute(0, 3, 1, 2).float()
        up     = F.interpolate(grid, size=(H, W), mode='bilinear', align_corners=False)

    feat = up.squeeze(0).permute(1, 2, 0).cpu().numpy()   # (H, W, D)
    return feat.reshape(H * W, D).astype(np.float32)


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(
    slice_data: np.ndarray,
    sigmas: Sequence[float] = (1, 1.5, 2, 3, 4),
    gaussian_types: Sequence[str] = ('smooth', 'gradient', 'laplacian'),
    window_sizes: Sequence[int] = (3, 5, 7),
    window_types: Sequence[str] = ('mean', 'median', 'min', 'max'),
    include_raw: bool = True,
    use_dinov2: bool = False,
    dinov2_model: str = 'vits14',
    dinov2_device: Optional[str] = None,
) -> np.ndarray:
    """Build a per-pixel feature matrix from a 2-D or 3-D image slice.

    Supported feature groups per channel:

    *Gaussian-based* (controlled by ``sigmas`` and ``gaussian_types``):
      - ``'smooth'``    — Gaussian-smoothed intensity
      - ``'gradient'``  — Gaussian gradient magnitude
      - ``'laplacian'`` — Laplacian of Gaussian

    *Rank / morphological* (controlled by ``window_sizes`` and ``window_types``):
      - ``'mean'``   — uniform (box) mean
      - ``'median'`` — median
      - ``'min'``    — minimum (erosion)
      - ``'max'``    — maximum (dilation)

    Raw intensity is included when ``include_raw=True``.

    Parameters
    ----------
    slice_data : np.ndarray
        Shape ``(H, W)`` or ``(H, W, C)``, float32.
    sigmas : sequence of float
        Gaussian scales for Gaussian-based features.
    gaussian_types : sequence of str
        Which Gaussian feature types to compute.
    window_sizes : sequence of int
        Window side lengths for rank/morphological filters.
    window_types : sequence of str
        Which rank/morphological filter types to compute.
    include_raw : bool
        Whether to include the unfiltered pixel values.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(H*W, n_features)``.
    """
    from scipy.ndimage import (
        gaussian_filter,
        gaussian_gradient_magnitude,
        gaussian_laplace,
        maximum_filter,
        median_filter,
        minimum_filter,
        uniform_filter,
    )

    if slice_data.ndim == 2:
        channels = [slice_data.astype(np.float32)]
    else:
        channels = [slice_data[..., c].astype(np.float32) for c in range(slice_data.shape[-1])]

    H, W = channels[0].shape
    feature_maps = []

    g_types = {t.lower() for t in gaussian_types}
    w_types = {t.lower() for t in window_types}

    for ch in channels:
        if include_raw:
            feature_maps.append(ch)

        # ── Gaussian-based features ───────────────────────────────────────────
        for sigma in sigmas:
            if 'smooth' in g_types:
                feature_maps.append(gaussian_filter(ch, sigma=sigma))
            if 'gradient' in g_types:
                feature_maps.append(gaussian_gradient_magnitude(ch, sigma=sigma))
            if 'laplacian' in g_types:
                feature_maps.append(gaussian_laplace(ch, sigma=sigma))

        # ── Rank / morphological features ─────────────────────────────────────
        for size in window_sizes:
            if 'mean' in w_types:
                feature_maps.append(uniform_filter(ch, size=size))
            if 'median' in w_types:
                feature_maps.append(median_filter(ch, size=size))
            if 'min' in w_types:
                feature_maps.append(minimum_filter(ch, size=size))
            if 'max' in w_types:
                feature_maps.append(maximum_filter(ch, size=size))

    if not feature_maps:
        # Fallback: always include raw to avoid empty feature matrix
        feature_maps = [ch for ch in channels]

    # Stack → (H, W, n_features) → (H*W, n_features)
    stacked = np.stack(feature_maps, axis=-1)
    result  = stacked.reshape(H * W, stacked.shape[-1]).astype(np.float32)

    # DINOv2 patch features — appended as additional columns
    if use_dinov2:
        dino = extract_dinov2_features(
            slice_data.astype(np.float32), dinov2_model, dinov2_device
        )
        result = np.concatenate([result, dino], axis=1)

    return result


# ── Classifier ────────────────────────────────────────────────────────────────

def train_classifier_region(
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
    n_augment: int = 3,
    classifier: str = 'rf',
):
    """Train a normalized Random Forest pipeline on labeled pixels in a region.

    Applies two improvements over a bare classifier:

    1. **Normalization** — a ``StandardScaler`` is fitted on the training
       features and baked into a ``sklearn.pipeline.Pipeline``.  The same
       scaling is applied automatically at prediction time so there is no
       train/predict mismatch.

    2. **Noise augmentation** — Gaussian noise (std = 5 % of image std) is
       added to the raw pixel values before feature extraction, producing
       ``n_augment`` additional training copies.  This encourages the forest
       to learn texture boundaries rather than absolute intensities.

    Parameters
    ----------
    label_region:
        ``(H, W)`` uint8 array — 0 = unlabeled, 1..N = class index.
    slice_region:
        ``(H, W)`` or ``(H, W, C)`` float32 image data for the same region.
    sigmas:
        Gaussian scales for feature extraction.
    n_estimators:
        Number of trees in the Random Forest.
    n_augment:
        Number of noisy copies added per annotated pixel (default 3).

    Returns
    -------
    Fitted ``sklearn.pipeline.Pipeline`` (StandardScaler → RandomForest).

    Raises
    ------
    ValueError
        If fewer than 2 distinct classes are labeled in the region.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    flat = label_region.ravel()
    labeled_mask = flat > 0
    unique_classes = np.unique(flat[labeled_mask])

    if len(unique_classes) < 2:
        raise ValueError(
            f"At least 2 classes must be labeled in the annotation box "
            f"(found {len(unique_classes)})."
        )

    img = slice_region.astype(np.float32)

    # DINOv2 features are expensive — extract once from the clean image and
    # reuse for all augmented copies (noise on raw pixels doesn't change the
    # global-context embedding meaningfully anyway).
    dino_feats: Optional[np.ndarray] = None
    if use_dinov2:
        dino_feats = extract_dinov2_features(img, dinov2_model, dinov2_device)

    # Noise std = 5 % of signal std; floor at a small absolute value
    noise_std = float(np.std(img)) * 0.05
    noise_std = max(noise_std, 1e-4)

    rng = np.random.default_rng(42)
    X_parts, y_parts = [], []

    for i in range(n_augment + 1):
        if i == 0:
            noisy = img
        else:
            noisy = img + rng.normal(0, noise_std, img.shape).astype(np.float32)
        features = extract_features(
            noisy, sigmas=sigmas, gaussian_types=gaussian_types,
            window_sizes=window_sizes, window_types=window_types,
            include_raw=include_raw,
            use_dinov2=False,   # handled above — avoid re-running per augmented copy
        )
        if dino_feats is not None:
            features = np.concatenate([features, dino_feats], axis=1)
        X_parts.append(features[labeled_mask])
        y_parts.append(flat[labeled_mask])

    X_train = np.vstack(X_parts)
    y_train = np.concatenate(y_parts)

    if classifier == 'xgb':
        from xgboost import XGBClassifier
        clf_obj = _OffsetClassifier(
            XGBClassifier(
                n_estimators=n_estimators,
                n_jobs=-1,
                random_state=42,
                tree_method='hist',
                eval_metric='mlogloss',
            )
        )
    else:  # 'rf' (default)
        clf_obj = RandomForestClassifier(
            n_estimators=n_estimators, n_jobs=-1, random_state=42
        )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', clf_obj),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def predict_region(
    clf,
    slice_region: np.ndarray,
    sigmas: Sequence[float] = (1, 1.5, 2, 3, 4),
    gaussian_types: Sequence[str] = ('smooth', 'gradient', 'laplacian'),
    window_sizes: Sequence[int] = (3, 5, 7),
    window_types: Sequence[str] = ('mean', 'median', 'min', 'max'),
    include_raw: bool = True,
    use_dinov2: bool = False,
    dinov2_model: str = 'vits14',
    dinov2_device: Optional[str] = None,
) -> np.ndarray:
    """Apply a fitted pipeline/classifier to all pixels in a region.

    Feature parameters must match those used during training exactly.
    The ``StandardScaler`` baked into the pipeline handles normalization.

    Returns ``(H, W)`` uint8 prediction mask.
    """
    H, W = slice_region.shape[:2]
    features = extract_features(
        slice_region.astype(np.float32),
        sigmas=sigmas, gaussian_types=gaussian_types,
        window_sizes=window_sizes, window_types=window_types,
        include_raw=include_raw,
        use_dinov2=use_dinov2, dinov2_model=dinov2_model, dinov2_device=dinov2_device,
    )
    return clf.predict(features).reshape(H, W).astype(np.uint8)


def train_classifier(
    store: AnnotationStore,
    slice_data: np.ndarray,
    t: int,
    z: int,
    orientation: str,
    sigmas: Sequence[float] = (1, 2, 4),
    n_estimators: int = 100,
):
    """Train a Random Forest on annotated pixels and return the fitted model.

    Raises ``ValueError`` if fewer than 2 distinct classes are annotated.
    Returns the fitted ``RandomForestClassifier``.
    """
    from sklearn.ensemble import RandomForestClassifier

    ann = store.get_slice_2d(t, z, orientation)   # (H, W) uint8
    labeled_mask = ann > 0
    unique_classes = np.unique(ann[labeled_mask])

    if len(unique_classes) < 2:
        raise ValueError(
            f"At least 2 classes must be annotated before running the classifier "
            f"(found {len(unique_classes)})."
        )

    features = extract_features(slice_data, sigmas=sigmas)   # (H*W, F)
    X_train = features[labeled_mask.ravel()]
    y_train = ann.ravel()[labeled_mask.ravel()]

    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    return clf


def predict_slice(clf, slice_data: np.ndarray, sigmas: Sequence[float] = (1, 2, 4)) -> np.ndarray:
    """Apply an existing fitted classifier to a new 2-D slice.

    Returns ``(H, W)`` uint8 prediction mask.
    """
    if slice_data.ndim == 2:
        H, W = slice_data.shape
    else:
        H, W = slice_data.shape[:2]
    features = extract_features(slice_data, sigmas=sigmas)
    return clf.predict(features).reshape(H, W).astype(np.uint8)


def train_and_classify(
    store: AnnotationStore,
    slice_data: np.ndarray,
    t: int,
    z: int,
    orientation: str,
    sigmas: Sequence[float] = (1, 2, 4),
    n_estimators: int = 100,
):
    """Train and immediately predict on the same slice. Returns ``(clf, mask)``."""
    clf = train_classifier(store, slice_data, t, z, orientation, sigmas, n_estimators)
    mask = predict_slice(clf, slice_data, sigmas)
    return clf, mask
