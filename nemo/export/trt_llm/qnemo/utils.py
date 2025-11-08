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

import os
from pathlib import Path

from nemo.export.tarutils import TarPath

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "rank{}.safetensors"


def is_qnemo_checkpoint(path: str) -> bool:
    """Detect if a given path is a TensorRT-LLM a.k.a. "qnemo" checkpoint based on config & tensor data presence."""
    if os.path.isdir(path):
        path = Path(path)
    else:
        path = TarPath(path)
    config_path = path / CONFIG_NAME
    tensor_path = path / WEIGHTS_NAME.format(0)
    return config_path.exists() and tensor_path.exists()
