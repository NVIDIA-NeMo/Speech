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

from dataclasses import fields, is_dataclass
from typing import Any, Union

import torch

from nemo.utils.dependency import is_module_available


def move_data_to_device(inputs: Any, device: Union[str, torch.device], non_blocking: bool = True) -> Any:
    """Recursively moves inputs to the specified device"""
    if inputs is None:
        return None
    if isinstance(inputs, torch.Tensor):
        if inputs.dtype == torch.float64 and device.type == "mps":
            # weird dataloader behavior: in some cases, it returns empty float64 tensors
            # float64 is not supported by mps, need to force-cast to float32
            inputs = inputs.to(dtype=torch.float32)
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


def is_dataloader(obj: Any) -> bool:
    """Return True if ``obj`` is a prebuilt dataloader that can be iterated over directly.

    Besides ``torch.utils.data.DataLoader``, this recognises the stateful Lhotse loader wrapper
    returned by ``get_lhotse_dataloader_from_config(..., use_stateful_dataloader=True)``, which
    delegates to an inner ``StatefulDataLoader`` instead of subclassing it.
    """
    if isinstance(obj, torch.utils.data.DataLoader):
        return True
    # Lhotse is optional for this module: without it, the stateful Lhotse wrapper cannot exist.
    if not is_module_available("lhotse"):
        return False
    from nemo.collections.common.data.lhotse.dataloader import _PerRankStatefulDataLoader

    return isinstance(obj, _PerRankStatefulDataLoader)
