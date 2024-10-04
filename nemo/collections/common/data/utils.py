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

from dataclasses import fields, is_dataclass
from typing import Any, Union

import torch


def move_data_to_device(inputs: Any, device: Union[str, torch.device], non_blocking: bool = True) -> Any:
    """Recursively moves inputs to the specified device"""
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device, non_blocking=non_blocking)
    elif isinstance(inputs, (list, tuple, set)):
        return inputs.__class__([move_data_to_device(i, device, non_blocking) for i in inputs])
    elif isinstance(inputs, dict):
        return {k: move_data_to_device(v, device, non_blocking) for k, v in inputs.items()}
    elif is_dataclass(inputs):
        return type(inputs)(
            **{
                field.name: move_data_to_device(getattr(inputs, field.name), device, non_blocking)
                for field in fields(inputs)
            }
        )
    else:
        return inputs
