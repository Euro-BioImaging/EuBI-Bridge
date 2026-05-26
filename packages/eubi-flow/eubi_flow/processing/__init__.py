# Import all processors so their @register_wave decorators fire on import.
from eubi_flow.processing import (
    filters,
    thresholding,
    reductions,
    reshape,
    numerical,
)

__all__ = ["filters", "thresholding", "reductions", "reshape", "numerical"]
