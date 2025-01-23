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

from pathlib import Path
from typing import Union

import torch


def is_nemo2_checkpoint(checkpoint_path: str) -> bool:
    """
    Checks if the checkpoint is in NeMo 2.0 format.
    Args:
        checkpoint_path (str): Path to a checkpoint.
    Returns:
        bool: True if the path points to a NeMo 2.0 checkpoint; otherwise false.
    """

    ckpt_path = Path(checkpoint_path)
    return (ckpt_path / 'context').is_dir()


# Copied from nemo.collections.nlp.parts.utils_funcs to avoid introducing extra NeMo dependencies:
def torch_dtype_from_precision(precision: Union[int, str], megatron_amp_O2: bool = True) -> torch.dtype:
    """
    Mapping from PyTorch Lighthing (PTL) precision types to corresponding PyTorch parameter data type.

    Args:
        precision (Union[int, str]): The PTL precision type used.
        megatron_amp_O2 (bool): A flag indicating if Megatron AMP O2 is enabled.

    Returns:
        torch.dtype: The corresponding PyTorch data type based on the provided precision.
    """
    if not megatron_amp_O2:
        return torch.float32

    if precision in ['bf16', 'bf16-mixed']:
        return torch.bfloat16
    elif precision in [16, '16', '16-mixed']:
        return torch.float16
    elif precision in [32, '32', '32-true']:
        return torch.float32
    else:
        raise ValueError(f"Could not parse the precision of '{precision}' to a valid torch.dtype")
