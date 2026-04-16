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
import torch
from lhotse import CutSet, SupervisionSegment
from lhotse.testing.dummies import dummy_cut, dummy_recording

from nemo.collections.common.data.utils import move_data_to_device
from nemo.collections.speechlm2 import DuplexSTTDataset


@pytest.fixture(scope="session")
def model(tiny_voicechat_model):
    return tiny_voicechat_model


@pytest.fixture(scope="session")
def dataset(model):
    return DuplexSTTDataset(
        model.stt_model.tokenizer,
        frame_length=0.08,
        source_sample_rate=16000,
        input_roles=["user"],
        output_roles=["assistant"],
    )


@pytest.fixture(scope="session")
def training_cutset_batch():
    cut = dummy_cut(0, recording=dummy_recording(0, with_data=True, duration=1.0, sampling_rate=22050))
    cut.target_audio = dummy_recording(1, with_data=True, duration=1.0, sampling_rate=22050)
    cut.supervisions = [
        SupervisionSegment(
            id=cut.id,
            recording_id=cut.recording_id,
            start=0,
            duration=0.1,
            text='hi',
            speaker="user",
        ),
        SupervisionSegment(
            id=cut.id,
            recording_id=cut.recording_id,
            start=0.3,
            duration=0.1,
            text='hello',
            speaker="assistant",
        ),
        SupervisionSegment(
            id=cut.id,
            recording_id=cut.recording_id,
            start=0.5,
            duration=0.1,
            text='ok',
            speaker="user",
        ),
        SupervisionSegment(
            id=cut.id,
            recording_id=cut.recording_id,
            start=0.6,
            duration=0.1,
            text='okay',
            speaker="assistant",
        ),
    ]
    return CutSet([cut])


def test_e2e_validation_step(model, dataset, training_cutset_batch):
    model.eval()
    model.on_validation_epoch_start()
    batch = dataset[training_cutset_batch]
    batch = move_data_to_device(batch, device=model.device)
    results = model.validation_step(
        {"dummy_val_set": batch},
        batch_idx=0,
        speaker_audio=torch.randn(1, 22050, device=model.device),
        speaker_audio_lens=torch.tensor([22050], device=model.device),
    )
    assert results is None  # no return value


def test_e2s_offline_generation(model):
    model.eval()
    # 16000 samples == 1 second == 12.5 frames ~= 14 frames after encoder padding
    ans = model.offline_inference(
        input_signal=torch.randn(1, 16000, device=model.device),
        input_signal_lens=torch.tensor([16000], device=model.device),
        speaker_audio=torch.randn(1, 22050, device=model.device),
        speaker_audio_lens=torch.tensor([22050], device=model.device),
    )

    assert ans.keys() == {
        'text',
        'src_text',
        'tokens_text_src',
        'tokens_text',
        'tokens_len',
        'source_audio',
        'source_audio_len',
        "audio",
        "audio_len",
    }

    assert isinstance(ans["text"], list)
    assert isinstance(ans["text"][0], str)

    gen_text = ans["tokens_text"]
    assert gen_text.shape == (1, 14)
    assert gen_text.dtype == torch.long
    assert (gen_text >= 0).all()
    assert (gen_text < model.stt_model.text_vocab_size).all()
    # 14 tokens = 24696 audio frames
    gen_audio = ans["audio"]
    assert gen_audio.shape == (1, 24696)
