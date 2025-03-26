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

from nemo.collections.vlm.qwen2vl.model import Qwen2VLConfig2B, Qwen2VLConfig7B, Qwen2VLConfig72B, Qwen2VLModel


def qwen2vl_2b() -> pl.LightningModule:
    # pylint: disable=C0115,C0116
    return Qwen2VLModel(Qwen2VLConfig2B())


def qwen2vl_7b() -> pl.LightningModule:
    # pylint: disable=C0115,C0116
    return Qwen2VLModel(Qwen2VLConfig7B())


def qwen2vl_72b() -> pl.LightningModule:
    # pylint: disable=C0115,C0116
    return Qwen2VLModel(Qwen2VLConfig72B())


__all__ = ["qwen2vl_2b", "qwen2vl_7b", "qwen2vl_72b"]
