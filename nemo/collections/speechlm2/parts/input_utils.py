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
from __future__ import annotations

from typing import Optional

import torch


def _unpad_inputs(
    input_ids: torch.Tensor,
    embeds: torch.Tensor,
    target_ids: Optional[torch.Tensor],
    padding_id: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], Optional[list[torch.Tensor]]]:
    def first_index_not_value(tensor, value):
        mask = tensor != value
        indices = torch.nonzero(mask, as_tuple=False)
        if indices.numel() > 0:
            return indices[0].item()
        else:
            return -1

    input_ids_unpad, embeds_unpad = [], []
    target_ids_unpad = [] if target_ids is not None else None
    for i in range(input_ids.shape[0]):
        idx = first_index_not_value(input_ids[i], padding_id)
        input_ids_unpad.append(input_ids[i, idx:])
        embeds_unpad.append(embeds[i, idx:])
        if target_ids is not None:
            target_ids_unpad.append(target_ids[i, idx:])
    return input_ids_unpad, embeds_unpad, target_ids_unpad
