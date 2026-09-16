# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from typing import Any, Optional

from omegaconf import MISSING

__all__ = ['DataLoaderConfig']


@dataclass
class DataLoaderConfig:
    """
    Configuration of PyTorch DataLoader.

    It is not derived from Config as it is not a NeMo object (and in particular it doesn't need a name).

    ..note:
        For the details on the function/meanings of the arguments, please refer to:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """

    batch_size: int = MISSING
    shuffle: bool = False
    sampler: Optional[Any] = None
    batch_sampler: Optional[Any] = None
    num_workers: int = 0
    collate_fn: Optional[Any] = None
    pin_memory: bool = False
    drop_last: bool = False
    timeout: int = 0
    worker_init_fn: Optional[Any] = None
    multiprocessing_context: Optional[Any] = None
