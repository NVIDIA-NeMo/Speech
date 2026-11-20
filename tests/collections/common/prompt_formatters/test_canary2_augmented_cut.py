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

import copy
from io import BytesIO

import numpy as np
import pytest
import soundfile as sf
import torch
from lhotse import Recording, SupervisionSegment
from lhotse.cut import MixedCut, MixTrack, MultiCut

from nemo.collections.common.prompts.canary2 import Canary2PromptFormatter, canary2


@pytest.fixture
def speech_cut():
    samples = np.sin(2 * np.pi * 440 * np.arange(1600) / 16000).astype(np.float32)[None, :]
    cut = _recording(samples, 'speech').to_cut()
    cut.supervisions = [SupervisionSegment('s', 'speech', 0.0, cut.duration, text='TEST', language='en')]
    cut.custom = {'source_lang': 'en', 'target_lang': 'en', 'pnc': '<|nopnc|>', 'decodercontext': 'TEST'}
    return cut


@pytest.mark.parametrize('augmentation', ['clean', 'mixed', 'nested-2', 'nested-3', 'rir-padding'])
def test_canary2_augmented_cut_preserves_prompt(speech_cut, canary2_tokenizer, augmentation):
    formatter = Canary2PromptFormatter(canary2_tokenizer)
    expected = canary2(speech_cut, formatter)
    noise = _recording(np.random.default_rng(42).normal(size=(1, 1600)).astype(np.float32), 'noise').to_cut()
    cut = speech_cut
    if augmentation == 'mixed':
        cut = speech_cut.mix(noise, snr=20)
    elif augmentation.startswith('nested-'):
        for _ in range(int(augmentation.split('-')[1])):
            cut = MixedCut('augmented', tracks=[MixTrack(cut=cut), MixTrack(cut=noise, snr=20)])
    elif augmentation == 'rir-padding':
        cut = _reverberate_and_pad(speech_cut.mix(noise, snr=20))
        assert isinstance(cut, MixedCut)
        assert isinstance(cut._first_non_padding_cut, MixedCut)
    before = copy.deepcopy(cut.to_dict())

    actual = canary2(cut, formatter)

    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])
    assert cut.to_dict() == before


def test_canary2_formatting_preserves_augmented_audio(speech_cut, canary2_tokenizer):
    cut = _reverberate_and_pad(speech_cut)
    before = cut.load_audio()
    clean = speech_cut.pad(0.2).load_audio()
    assert not np.allclose(before, clean)

    canary2(cut, Canary2PromptFormatter(canary2_tokenizer))

    np.testing.assert_array_equal(cut.load_audio(), before)


@pytest.mark.parametrize('nesting', [0, 1, 2])
def test_canary2_multichannel_cut_is_rejected(nesting):
    cut = MultiCut('stereo', 0.0, 1.0, [0, 1])
    for _ in range(nesting):
        cut = MixedCut('mixed', tracks=[MixTrack(cut=cut)])
    with pytest.raises(TypeError, match='single channel'):
        canary2(cut, None)


def _reverberate_and_pad(cut):
    samples = np.zeros((1, 160), dtype=np.float32)
    samples[0, 0], samples[0, 80] = 1.0, 0.5
    rir = _recording(samples, 'rir')
    return cut.reverb_rir(rir_recording=rir).pad(0.2)


def _recording(samples, recording_id):
    stream = BytesIO()
    sf.write(stream, samples.T, 16000, format='WAV', subtype='FLOAT')
    return Recording.from_bytes(stream.getvalue(), recording_id=recording_id)
