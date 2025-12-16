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

import torch
from nemo.utils import logging

COMPUTE_DTYPE_MAP = {
    'bfloat16': torch.bfloat16,
    'float16': torch.float16,
    'float32': torch.float32,
}

DEVICE_TYPES = ["cuda", "mps", "cpu"]


def setup_device(device: str, device_id: int | None, compute_dtype: str) -> tuple[str, int, torch.dtype]:
    """
    Set up the compute device for the model.

    Args:
        device (str): Requested device type ('cuda', 'mps' or 'cpu').
        device_id (int | None): Requested CUDA device ID (None for CPU or MPS).
        compute_dtype (str): Requested compute dtype.

    Returns:
        tuple(str, int, torch.dtype): Tuple of (device_string, device_id, compute_dtype) for model initialization.
    """
    device = device.strip()
    if device not in DEVICE_TYPES:
        raise ValueError(f"Invalid device type: {device}. Must be one of {DEVICE_TYPES}")

    device_id = int(device_id) if device_id is not None else 0

    # Handle CUDA devices
    if torch.cuda.is_available() and device == "cuda":
        if device_id >= torch.cuda.device_count():
            logging.warning(f"Device ID {device_id} is not available. Using GPU 0 instead.")
            device_id = 0

        compute_dtype = COMPUTE_DTYPE_MAP.get(compute_dtype, None)
        if compute_dtype is None:
            raise ValueError(
                f"Invalid compute dtype: {compute_dtype}. Must be one of {list(COMPUTE_DTYPE_MAP.keys())}"
            )

        device_str = f"cuda:{device_id}"
        return device_str, device_id, compute_dtype

    # Handle MPS devices
    if torch.backends.mps.is_available() and device == "mps":
        return "mps", -1, torch.float32

    # Handle CPU devices
    if device == "cpu":
        return "cpu", -1, torch.float32

    raise ValueError(f"Device {device} is not available.")
