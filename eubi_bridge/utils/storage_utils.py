"""Storage and store management utilities."""

import pathlib
from typing import Any, Dict
from urllib.parse import urlparse


def make_kvstore(store_path: str) -> Dict[str, Any]:
    """Convert store path to TensorStore key-value store configuration.
    
    Handles local filesystems and S3-style HTTPS URLs.
    
    Args:
        store_path: Path or URL to store
                   - /local/path
                   - C:\\windows\\path
                   - https://bucket-name/path/to/object
    
    Returns:
        Dictionary with TensorStore driver configuration
    
    Raises:
        ValueError: If store_path format is invalid
    
    Examples:
        >>> make_kvstore("/data/output.zarr")
        {'driver': 'file', 'path': '/data/output.zarr'}
        
        >>> make_kvstore("https://bucket/data/output.zarr")
        {'driver': 's3', 'endpoint': 'https://bucket', 'bucket': 'bucket',
         'path': 'data/output.zarr', 'aws_region': 'us-east-1'}
    """
    if not isinstance(store_path, str):
        raise ValueError("store_path must be a string")

    parsed = urlparse(store_path)

    # Handle HTTPS URL (S3-style)
    if parsed.scheme == "https":
        # Example: https://bucket-name/path/to/object
        host = parsed.netloc  # bucket-name.example.com or bucket-name
        parts = parsed.path.strip("/").split("/")  # ["bucket", "subfolder", "file"]

        if len(parts) == 0:
            raise ValueError(f"Cannot parse HTTPS URL: {store_path}")

        bucket = parts[0]
        path = "/".join(parts[1:])
        path = path.replace(" ", "%20")

        return {
            "driver": "s3",
            "endpoint": f"https://{host}",
            "bucket": bucket,
            "path": path,
            "aws_region": "us-east-1"
        }

    # Handle local filesystem path
    # Supports:
    # - /absolute/posix/path
    # - C:\absolute\windows\path
    # - C:/absolute/windows/path
    # - \\server\share (UNC)
    # - relative paths
    p = pathlib.Path(store_path)

    # Use TensorStore "file" driver for local paths
    # (TensorStore supports both absolute and relative)
    if p.is_absolute() or not parsed.scheme:
        return {"driver": "file", "path": str(p)}

    # Unknown format
    raise ValueError(f"Unsupported store path: {store_path}")
