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

from dataclasses import dataclass

import pytest
import torch

from nemo.collections.common.data.utils import move_data_to_device


@dataclass
class _Batch:
    data: torch.Tensor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="This test requires GPUs.")
@pytest.mark.parametrize(
    "batch",
    [
        torch.tensor([0]),
        (torch.tensor([0]),),
        [torch.tensor([0])],
        {"data": torch.tensor([0])},
        _Batch(torch.tensor([0])),
        "not a tensor",
    ],
)
def test_move_data_to_device(batch):
    cuda_batch = move_data_to_device(batch, device="cuda")
    assert type(batch) == type(cuda_batch)
    if isinstance(batch, _Batch):
        assert cuda_batch.data.is_cuda
    elif isinstance(batch, dict):
        assert cuda_batch["data"].is_cuda
    elif isinstance(batch, (list, tuple)):
        assert cuda_batch[0].is_cuda
    elif isinstance(batch, torch.Tensor):
        assert cuda_batch.is_cuda
    else:
        assert cuda_batch == batch
