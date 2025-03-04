# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import functools
from typing import Optional

import torch

from nemo.utils.app_state import AppState

# pylint: disable=C0116


@functools.lru_cache(maxsize=None)
def _nvtx_enabled() -> bool:
    """Check if NVTX range profiling is enabled"""
    return AppState()._nvtx_ranges


# Messages associated with active NVTX ranges
_nvtx_range_messages: list[str] = []


def nvtx_range_push(msg: str) -> None:
    # Return immediately if NVTX range profiling is not enabled
    if not _nvtx_enabled():
        return

    # Push NVTX range to stack
    _nvtx_range_messages.append(msg)
    torch.cuda.nvtx.range_push(msg)


def nvtx_range_pop(msg: Optional[str] = None) -> None:
    # Return immediately if NVTX range profiling is not enabled
    if not _nvtx_enabled():
        return

    # Update list of NVTX range messages and check for consistency
    if not _nvtx_range_messages:
        raise RuntimeError("Attempted to pop NVTX range from empty stack")
    last_msg = _nvtx_range_messages.pop()
    if msg is not None and msg != last_msg:
        raise ValueError(
            f"Attempted to pop NVTX range from stack with msg={msg}, " f"but last range has msg={last_msg}"
        )

    # Pop NVTX range
    torch.cuda.nvtx.range_pop()


# pylint: enable=C0116
