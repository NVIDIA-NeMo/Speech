# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import pytest
from examples.tts.easy_magpietts import _select_model_class
from omegaconf import OmegaConf

from nemo.collections.tts.models import EasyMagpieTTSModel, NemotronTTSModelV2

pytestmark = pytest.mark.unit


def test_select_v2_model():
    model_cfg = OmegaConf.create({"model_type": "v2"})

    assert _select_model_class(model_cfg) is NemotronTTSModelV2


def test_select_v1_model():
    assert _select_model_class(OmegaConf.create({})) is EasyMagpieTTSModel
    assert _select_model_class(OmegaConf.create({"model_type": "v1"})) is EasyMagpieTTSModel


def test_reject_unknown_model_type():
    with pytest.raises(ValueError, match="Unsupported model_type"):
        _select_model_class(OmegaConf.create({"model_type": "v3"}))
