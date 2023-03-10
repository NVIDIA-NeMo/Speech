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
This file implemented unit tests for loading all pretrained AlignerModel NGC checkpoints and generating Mel-spectrograms.
The test duration breakdowns are shown below. In general, each test for a single model is ~24 seconds on an NVIDIA RTX A6000.
"""
import pytest
import torch

from nemo.collections.tts.models import AlignerModel

available_models = [model.pretrained_model_name for model in AlignerModel.list_available_models()]


@pytest.fixture(params=available_models, ids=available_models)
@pytest.mark.run_only_on('GPU')
def pretrained_model(request, get_language_id_from_pretrained_model_name):
    model_name = request.param
    language_id = get_language_id_from_pretrained_model_name(model_name)
    model = AlignerModel.from_pretrained(model_name=model_name)
    return model, language_id


@pytest.mark.nightly
@pytest.mark.run_only_on('GPU')
def test_inference(pretrained_model, audio_text_pair_example_english):
    model, _ = pretrained_model
    audio, audio_len, text_raw = audio_text_pair_example_english

    # Generate mel-spectrogram
    spec, spec_len = model.preprocessor(input_signal=audio, length=audio_len)

    # Process text
    text_normalized = model.normalizer.normalize(text_raw, punct_post_process=True)
    text_tokens = model.tokenizer(text_normalized)
    text = torch.tensor(text_tokens, device=spec.device).unsqueeze(0).long()
    text_len = torch.tensor(len(text_tokens), device=spec.device).unsqueeze(0).long()

    # Run the Aligner
    _, _ = model(spec=spec, spec_len=spec_len, text=text, text_len=text_len)
