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

from nemo.collections.vlm.neva.model import Llava15Config7B, Llava15Config13B, LlavaModel


def llava15_7b() -> pl.LightningModule:
    return LlavaModel(Llava15Config7B())


def llava15_13b() -> pl.LightningModule:
    return LlavaModel(Llava15Config13B())


__all__ = [
    "llava15_7b",
    "llava15_13b",
]
