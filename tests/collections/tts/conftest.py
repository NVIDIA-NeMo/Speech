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

import os

import pytest
import soundfile as sf
import torch

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest


@pytest.fixture()
def set_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture()
def language_specific_text_example():
    return {
        "en": "Caslon's type is clear and neat, and fairly well designed;",
        "de": "Ich trinke gerne Kräutertee mit Lavendel.",
        "es": "Los corazones de pollo son una delicia.",
        "zh": "双辽境内除东辽河、西辽河等5条河流",
    }


@pytest.fixture()
def supported_languages(language_specific_text_example):
    return sorted(language_specific_text_example.keys())


@pytest.fixture()
def get_language_id_from_pretrained_model_name(supported_languages):
    def _validate(pretrained_model_name):
        language_id = pretrained_model_name.split("_")[1]
        if language_id not in supported_languages:
            pytest.fail(
                f"`PretrainedModelInfo.pretrained_model_name={pretrained_model_name}` does not follow the naming "
                f"convention as `tts_languageID_model_*`, or `languageID` is not supported in {supported_languages}."
            )
        return language_id

    return _validate


@pytest.fixture()
def mel_spec_example(set_device):
    # specify a value range of mel-spectrogram close to ones generated in practice. But we can also mock the values
    # by `torch.randn` for testing purpose.
    min_val = -11.0
    max_val = 0.5
    batch_size = 1
    n_mel_channels = 80
    n_frames = 330
    spec = (min_val - max_val) * torch.rand(batch_size, n_mel_channels, n_frames, device=set_device) + max_val
    return spec


@pytest.fixture()
def audio_text_pair_example_english(test_data_dir, set_device):
    manifest_path = os.path.join(test_data_dir, 'tts/mini_ljspeech/manifest.json')
    data = read_manifest(manifest_path)
    audio_filepath = data[-1]["audio_filepath"]
    text_raw = data[-1]["text"]

    # Load audio
    audio_data, orig_sr = sf.read(audio_filepath)
    audio = torch.tensor(audio_data, dtype=torch.float, device=set_device).unsqueeze(0)
    audio_len = torch.tensor(audio_data.shape[0], device=set_device).long().unsqueeze(0)

    return audio, audio_len, text_raw
