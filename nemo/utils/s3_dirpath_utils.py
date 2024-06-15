from pathlib import Path
from typing import Optional

S3_PATH_PREFIX = 's3://'


def build_s3_url(bucket, key) -> str:
    """
    This function constructs an s3 address given a bucket and key.
    It has no reliance on any S3-related dependencies as the file pre-defines the S3 path prefix.
    """
    return f'{S3_PATH_PREFIX}{bucket}/{key}'


def is_s3_url(path: Optional[str]) -> bool:
    """
    This function checks if a path is an S3 url.
    It has no reliance on any S3-related dependencies as the file pre-defines the S3 path prefix.
    """
    if isinstance(path, Path):
        path = str(path)
    return path is not None and path.strip().startswith(S3_PATH_PREFIX)
