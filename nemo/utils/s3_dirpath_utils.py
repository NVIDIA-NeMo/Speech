# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
