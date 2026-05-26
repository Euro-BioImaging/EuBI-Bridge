"""
DINOv2 feature extraction — placeholder module.

The DINOv2-based pixel feature extraction logic currently lives inside
``eubi_annotate.core.annotation`` (specifically ``extract_dinov2_features``
and the ``use_dinov2`` flag on ``extract_features`` /
``train_classifier_region``).

In a future refactor this module will house a standalone, reusable
implementation that can be imported by both ``eubi_annotate`` and any
other package that needs deep vision features without pulling in the full
annotation stack.

Planned public API (not yet implemented):

    from eubi_deep.features import extract_dinov2_features

    features = extract_dinov2_features(
        slice_region,          # (H, W) or (H, W, C) float32 array
        model_name='vits14',   # 'vits14' | 'vitb14' | 'vitl14' | 'vitg14'
        device=None,           # 'cpu' | 'cuda' | None (auto-detect)
    )
    # returns float32 array of shape (H*W, D)

See ``eubi_annotate.core.annotation.extract_dinov2_features`` for the
current implementation.
"""
