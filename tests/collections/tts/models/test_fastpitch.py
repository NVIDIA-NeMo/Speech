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
This file implemented unit tests for loading all pretrained FastPitch NGC checkpoints and generating Mel-spectrograms.
The test duration breakdowns are shown below. In general, each test for a single model is ~25 seconds on an NVIDIA RTX A6000.
"""
import random

import pytest
import torch

from nemo.collections.tts.models import FastPitchModel

available_models = [model.pretrained_model_name for model in FastPitchModel.list_available_models()]


@pytest.fixture(params=available_models, ids=available_models)
@pytest.mark.run_only_on('GPU')
def pretrained_model(request, get_language_id_from_pretrained_model_name):
    model_name = request.param
    language_id = get_language_id_from_pretrained_model_name(model_name)
    model = FastPitchModel.from_pretrained(model_name=model_name)
    return model, language_id


# This test can only pass when nemo_text_process<=0.1.8rc0. If >0.1.8rc0, the normalized outputs are unexpected for Chinese.
# Will remove the marker `pleasefixme` once next-text-processing new release fixes the bug.
# Tracking bugfix in https://github.com/NVIDIA/NeMo-text-processing/issues/109.
@pytest.mark.pleasefixme
@pytest.mark.nightly
@pytest.mark.run_only_on('GPU')
def test_inference(pretrained_model, language_specific_text_example):
    model, language_id = pretrained_model
    text = language_specific_text_example[language_id]
    parsed_text = model.parse(text)

    # Multi-Speaker
    speaker_id = None
    reference_spec = None
    reference_spec_lens = None

    if hasattr(model.fastpitch, 'speaker_emb'):
        speaker_id = 0

    if hasattr(model.fastpitch, 'speaker_encoder'):
        if hasattr(model.fastpitch.speaker_encoder, 'lookup_module'):
            speaker_id = 0
        if hasattr(model.fastpitch.speaker_encoder, 'gst_module'):
            bs, lens, t_spec = parsed_text.shape[0], random.randint(50, 100), model.cfg.n_mel_channels
            reference_spec = torch.rand(bs, lens, t_spec)
            reference_spec_lens = torch.tensor([lens]).long().expand(bs)

    _ = model.generate_spectrogram(
        tokens=parsed_text, speaker=speaker_id, reference_spec=reference_spec, reference_spec_lens=reference_spec_lens
    )
