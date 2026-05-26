"""eubi-deep: Deep learning features (DINOv2 embeddings) for EuBI-Bridge annotation."""

try:
    from importlib.metadata import version as _metadata_version
    __version__ = _metadata_version("eubi-deep")
except Exception:
    __version__ = "0.0.0+dev"
