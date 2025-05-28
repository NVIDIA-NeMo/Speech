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

import json
import tempfile

import pytest

from nemo.collections.llm.modelopt.quantization.quant_cfg_choices import get_quant_cfg_choices
from nemo.collections.llm.modelopt.quantization.utils import load_quant_cfg

QUANT_CFG_CHOICES = get_quant_cfg_choices()


@pytest.mark.parametrize("cfg_name", ["nvfp4", "fp8"])
def test_load_quant_cfg(cfg_name):
    """Test loading a quantization config from a JSON file."""

    quant_cfg_org = QUANT_CFG_CHOICES[cfg_name]

    with tempfile.NamedTemporaryFile(mode="w") as temp_file:
        json.dump(quant_cfg_org, temp_file)
        temp_file.flush()
        quant_cfg_loaded = load_quant_cfg(temp_file.name)
        assert quant_cfg_loaded == quant_cfg_org
