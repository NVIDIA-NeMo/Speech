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

import pytest

from nemo.collections.asr.models import ASRModel, EncDecHybridRNNTCTCBPEModel, EncDecRNNTBPEModel
from nemo.collections.asr.models.asr_eou_models import EncDecHybridRNNTCTCBPEEOUModel, EncDecRNNTBPEEOUModel


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_name, expected_cls",
    [
        ("stt_en_fastconformer_transducer_large", EncDecRNNTBPEModel),
        ("stt_en_fastconformer_hybrid_large_pc", EncDecHybridRNNTCTCBPEModel),
    ],
)
def test_eou_import_preserves_pretrained_model_discovery(model_name, expected_cls):
    # Importing the EOU subclasses must not redirect ordinary checkpoints to them.
    models = {info.pretrained_model_name: info.class_ for info in ASRModel.list_available_models()}
    assert models[model_name] is expected_cls


@pytest.mark.unit
@pytest.mark.parametrize("model_cls", [EncDecRNNTBPEEOUModel, EncDecHybridRNNTCTCBPEEOUModel])
def test_eou_models_do_not_advertise_parent_checkpoints(model_cls):
    assert model_cls.list_available_models() == []
