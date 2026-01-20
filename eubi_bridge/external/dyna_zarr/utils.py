"""
Utility functions and helpers for dyna-zarr.

Includes dtype parsing, storage utilities, and other shared functions.
"""

import numpy as np


def parse_dtype(dtype_input):
    """
    Parse and standardize dtype specifications for Zarr.
    
    Converts various dtype inputs to standardized formats:
    - Returns a numpy dtype object
    - Provides Zarr v2 format string (e.g., '<f4', '<u2')
    - Provides Zarr v3 format name (e.g., 'float32', 'uint16')
    
    Args:
        dtype_input: numpy dtype, dtype name string, or type object
        
    Returns:
        tuple: (np.dtype object, zarr_v2_str, zarr_v3_name)
            - np.dtype: The normalized numpy dtype
            - zarr_v2_str: String format for Zarr v2 (with endianness)
            - zarr_v3_name: Clean name for Zarr v3 (no endianness)
            
    Examples:
        >>> dtype_obj, v2_str, v3_name = parse_dtype('float32')
        >>> dtype_obj
        dtype('float32')
        >>> v2_str
        '<f4'
        >>> v3_name
        'float32'
        
        >>> dtype_obj, v2_str, v3_name = parse_dtype(np.uint16)
        >>> v2_str
        '<u2'
        >>> v3_name
        'uint16'
    """
    # Handle numpy dtype objects
    if isinstance(dtype_input, np.dtype):
        dtype_obj = dtype_input
    else:
        # Try to convert to numpy dtype
        try:
            dtype_obj = np.dtype(dtype_input)
        except TypeError:
            # If that fails, try to extract the actual type from string representation
            # This handles cases like 'dtype("float32")'
            dtype_str = str(dtype_input).strip()
            # Remove dtype(...) wrapper if present
            if dtype_str.startswith('dtype('):
                dtype_str = dtype_str[6:-1]  # Remove 'dtype(' and ')'
            dtype_str = dtype_str.strip('\'"')  # Remove quotes
            dtype_obj = np.dtype(dtype_str)
    
    # Zarr v2 format: endian prefix + type letter + itemsize (e.g., '<f4', '<u2')
    zarr_v2_str = dtype_obj.str
    
    # Zarr v3 format: clean dtype name (e.g., 'float32', 'uint16')
    # Remove endianness marker from the name
    zarr_v3_name = dtype_obj.name
    
    return dtype_obj, zarr_v2_str, zarr_v3_name
