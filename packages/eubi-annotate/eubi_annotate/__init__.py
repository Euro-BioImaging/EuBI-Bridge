"""eubi-annotate: Pixel annotation and ML classifier (RF/XGB) for EuBI-Bridge."""

try:
    from importlib.metadata import version as _metadata_version
    __version__ = _metadata_version("eubi-annotate")
except Exception:
    __version__ = "0.0.0+dev"
