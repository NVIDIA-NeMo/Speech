# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import shutil
from pathlib import Path
from typing import Union


def robust_copy(src: Union[Path, str], dst: Union[Path, str]) -> str:
    """
    Copy file from src to dst, falling back to shutil.copy if shutil.copy2 fails.
    shutil.copy2 preserves metadata, but can fail on some filesystems.
    """
    try:
        return shutil.copy2(src, dst)
    except PermissionError:
        # copy2 can fail on some filesystems due to metadata copy errors
        # (e.g., permission errors on setting timestamps).
        # In such cases, we fallback to a plain copy.
        return shutil.copy(src, dst)
