# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""
This file implemented unit tests for loading all pretrained UnivNet NGC checkpoints and converting Mel-spectrograms into
audios. In general, each test for a single model is ~2 seconds on an NVIDIA RTX A6000.
"""

import pytest

from nemo.collections.tts.models import UnivNetModel

available_models = [model.pretrained_model_name for model in UnivNetModel.list_available_models()]


@pytest.fixture(params=available_models, ids=available_models)
@pytest.mark.run_only_on('GPU')
def pretrained_model(request, get_language_id_from_pretrained_model_name):
    model_name = request.param
    language_id = get_language_id_from_pretrained_model_name(model_name)
    model = UnivNetModel.from_pretrained(model_name=model_name)
    return model, language_id


@pytest.mark.nightly
@pytest.mark.run_only_on('GPU')
def test_inference(pretrained_model, mel_spec_example):
    model, _ = pretrained_model
    _ = model.convert_spectrogram_to_audio(spec=mel_spec_example)
