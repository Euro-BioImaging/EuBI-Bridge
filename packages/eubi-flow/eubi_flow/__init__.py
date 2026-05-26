"""eubi-flow: Workflow DAG engine and Flow GUI for EuBI-Bridge."""

try:
    from importlib.metadata import version as _metadata_version
    __version__ = _metadata_version("eubi-flow")
except Exception:
    __version__ = "0.0.0+dev"

from eubi_flow.eubiflow import EuBIFlow

__all__ = ["EuBIFlow"]
