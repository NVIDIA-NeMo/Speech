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

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import nemo.collections.tts.data.text_to_speech_dataset as text_to_speech_dataset

pytestmark = pytest.mark.unit

SAMPLE_RATE = 22050
CODEC_MODEL_SAMPLES_PER_FRAME = 256


def _write_wav(path: Path, num_samples: int) -> None:
    sf.write(str(path), np.zeros(num_samples, dtype=np.float32), SAMPLE_RATE, format="WAV")


def _make_dataset(audio_dir: Path, monkeypatch) -> text_to_speech_dataset.MagpieTTSDataset:
    # Bypass __init__ (and the heavyweight phoneme tokenizer it wires up) and set only the
    # attributes the raw-audio branch of __getitem__ actually reads.
    monkeypatch.setattr(text_to_speech_dataset, "tokenize_text_with_phoneme_spans", lambda **kwargs: [1, 2, 3])

    dataset = object.__new__(text_to_speech_dataset.MagpieTTSDataset)
    dataset.sample_rate = SAMPLE_RATE
    dataset.codec_model_samples_per_frame = CODEC_MODEL_SAMPLES_PER_FRAME
    dataset.load_cached_codes_if_available = False
    dataset.volume_norm = False
    dataset.load_16khz_audio = False
    dataset.use_text_conditioning_tokenizer = False
    dataset.include_align_prior = False
    dataset.bos_id = 0
    dataset.eos_id = 1
    dataset.text_tokenizer = None
    dataset.phoneme_tokenizer = None
    dataset.enable_phoneme_text_input = False
    dataset.text_phoneme_token_offset = None
    dataset.phoneme_text_bop_marker = "<bop>"
    dataset.phoneme_text_eop_marker = "<eop>"
    dataset.default_tokenizer_name = "english_phoneme"
    dataset.ignore_phoneme_languages = []
    dataset.dataset_type = "train"
    return dataset


def _get_audio_len(tmp_path: Path, monkeypatch, num_input_samples: int) -> int:
    _write_wav(tmp_path / "sample.wav", num_input_samples)
    dataset = _make_dataset(tmp_path, monkeypatch)
    dataset.data_samples = [
        text_to_speech_dataset.DatasetSample(
            dataset_name="unit",
            manifest_entry={"audio_filepath": "sample.wav"},
            audio_dir=tmp_path,
            feature_dir=None,
            text="hello world",
        )
    ]
    example = dataset[0]
    return int(example["audio_len"])


class TestMagpieTTSDatasetAudioPadding:
    def test_exact_multiple_length_audio_is_not_padded(self, tmp_path, monkeypatch):
        """Audio whose length already divides the codec frame size must not gain a spurious frame."""
        num_input_samples = CODEC_MODEL_SAMPLES_PER_FRAME * 5

        audio_len = _get_audio_len(tmp_path, monkeypatch, num_input_samples)

        assert audio_len == num_input_samples

    def test_non_multiple_length_audio_is_rounded_up(self, tmp_path, monkeypatch):
        """Audio that is not an exact multiple must still be padded up to the next frame boundary."""
        num_input_samples = CODEC_MODEL_SAMPLES_PER_FRAME * 5 + 37

        audio_len = _get_audio_len(tmp_path, monkeypatch, num_input_samples)

        assert audio_len == CODEC_MODEL_SAMPLES_PER_FRAME * 6
