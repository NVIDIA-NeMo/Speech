# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import nemo.collections.speechlm2.data.salm_dataset as salm_dataset_module
from nemo.collections.common.data.lhotse import NeMoMultimodalConversation
from nemo.collections.common.data.lhotse.text_adapters import AudioTurn, TextTurn


class _Tokenizer:
    pad = 0
    unk_id = 1


@pytest.mark.unit
def test_salm_dataset_builds_aligned_sot_targets(monkeypatch):
    text = "<spk:0> hello world <spk:1> yes now"
    cut = dummy_cut(0, duration=0.04, recording=dummy_recording(0, duration=0.04, with_data=True))
    cut.custom = {}
    cut.supervisions = [
        SupervisionSegment(id=cut.id, recording_id=cut.recording_id, start=0.0, duration=0.04, text=text)
    ]
    conversation = NeMoMultimodalConversation(
        id="example-0",
        turns=[
            AudioTurn(role="user", cut=cut, audio_locator_tag="<|audio|>", text=text),
            TextTurn(role="assistant", value=text),
        ],
        token_equivalent_duration=0.01,
    )
    conversation.input_ids = torch.tensor([7, 8, 9], dtype=torch.long)
    conversation.mask = torch.tensor([False, True, True])
    conversations = CutSet([conversation])

    def fake_audio_collate(conversations, *args, **kwargs):
        return torch.zeros(1, 640), torch.tensor([640], dtype=torch.long), conversations

    def fake_speaker_activity_from_cut(cut, **kwargs):
        assert cut.supervisions[0].text == text
        return torch.tensor(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        )

    monkeypatch.setattr(salm_dataset_module, "collate_conversation_audio_fault_tolerant", fake_audio_collate)
    monkeypatch.setattr(salm_dataset_module, "speaker_activity_from_cut", fake_speaker_activity_from_cut)

    dataset = salm_dataset_module.SALMDataset(
        tokenizer=_Tokenizer(),
        multispeaker_cfg={
            "num_speakers": 2,
            "sample_rate": 16000,
            "window_stride": 0.01,
            "subsampling_factor": 1,
        },
    )
    assert dataset.multispeaker_cfg == salm_dataset_module.MultiSpeakerConfig(
        num_speakers=2,
        num_sample_per_mel_frame=160,
        num_mel_frame_per_target_frame=1,
    )
    assert dataset.multispeaker_cfg.num_sample_per_mel_frame == 160

    batch = dataset[conversations]

    assert torch.equal(
        batch["spk_targets"],
        torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                ]
            ]
        ),
    )
    assert torch.equal(batch["spk_target_length"], torch.tensor([4]))
