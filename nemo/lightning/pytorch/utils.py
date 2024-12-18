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

import lightning.pytorch as pl
import torch


def extract_dtypes(ckpt):
    """
    Extracts dtype from the input iterator
    ckpt can be module.named_parameters or module.state_dict().items()
    """
    dtypes = {}
    for key, val in ckpt:
        if hasattr(val, 'dtype'):
            dtypes[key] = val.dtype
        elif hasattr(val, 'data') and hasattr(val.data, 'dtype'):
            # if it's ShardedTensor populated with data.
            dtypes[key] = val.data.dtype
    return dtypes


def dtype_from_str(dtype):
    """
    Convert a str precision to equivalent torch dtype.
    """
    assert isinstance(dtype, str)
    if dtype in ["float16", "fp16", "16", "16-mixed"]:
        return torch.float16
    elif dtype == ["bfloat16", "bf16-mixed"]:
        return torch.bfloat16
    else:
        return torch.float32


def dtype_from_hf(config):
    """
    Extracts torch dtype from a HF config
    """
    assert hasattr(config, 'torch_dtype'), "Expected config to have attr `torch_dtype`"
    torch_dtype = config.torch_dtype
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    elif isinstance(torch_dtype, str):
        return dtype_from_str(torch_dtype)
    else:
        raise ValueError("torch_dtype is not of type str/torch.dtype")


def is_trainer_attached(model: pl.LightningModule):
    """
    Returns true if trainer is attached to a model
    """
    try:
        trainer = model.trainer
        return True
    except (AttributeError, RuntimeError):
        return False
