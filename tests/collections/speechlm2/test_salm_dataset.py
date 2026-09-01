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

import io

import numpy as np
import pytest
import soundfile as sf
import torch
from lhotse import CutSet, Recording, SupervisionSegment, fastcopy
from lhotse.testing.dummies import dummy_cut, dummy_recording

import nemo.collections.speechlm2.data.salm_dataset as salm_dataset_module
from nemo.collections.common.data.lhotse import NeMoMultimodalConversation
from nemo.collections.common.data.lhotse.text_adapters import AudioTurn, TextTurn
from nemo.collections.speechlm2.parts.encoder_chunking import _split_spk_targets_into_chunks


class _Tokenizer:
    pad = 0
    unk_id = 1


@pytest.fixture(autouse=True)
def stub_audio_samples(monkeypatch):
    monkeypatch.setattr(salm_dataset_module, "AudioSamples", lambda **kwargs: object())


@pytest.mark.unit
def test_multispeaker_config_requires_explicit_speaker_count():
    with pytest.raises(ValueError, match="no implicit 4-speaker cap"):
        salm_dataset_module.MultiSpeakerConfig.from_dict({})


@pytest.mark.unit
def test_multispeaker_targets_precede_in_memory_audio_drop(monkeypatch):
    audio = io.BytesIO()
    sf.write(audio, np.zeros((640, 2), dtype=np.float32), 16000, format="WAV")
    cut = Recording.from_bytes(audio.getvalue(), recording_id="native-wds").to_cut()
    cut = fastcopy(
        cut,
        custom={
            "_source_read_key": "/data/shard-0.tar#sample-0",
            "_source_range_bytes": len(audio.getvalue()),
        },
    )
    conversation = NeMoMultimodalConversation(
        id="example-0",
        turns=[
            AudioTurn(role="user", cut=cut, audio_locator_tag="<|audio|>"),
            TextTurn(role="assistant", value="hello"),
        ],
        token_equivalent_duration=0.01,
    )
    conversation.input_ids = torch.tensor([7, 8], dtype=torch.long)
    conversation.mask = torch.tensor([False, True])
    conversations = CutSet([conversation])

    def fake_audio_collate(conversations_arg, *args, **kwargs):
        return torch.zeros(1, 640), torch.tensor([640]), conversations_arg

    def fake_speaker_activity_from_cut(materialized_cut, **kwargs):
        assert materialized_cut.load_audio().shape == (1, 640)
        return torch.zeros(4, 2)

    monkeypatch.setattr(
        salm_dataset_module,
        "collate_conversation_audio_fault_tolerant",
        fake_audio_collate,
    )
    monkeypatch.setattr(
        salm_dataset_module,
        "speaker_activity_from_cut",
        fake_speaker_activity_from_cut,
    )
    dataset = salm_dataset_module.SALMDataset(
        tokenizer=_Tokenizer(),
        multispeaker_cfg={
            "num_speakers": 2,
            "sample_rate": 16000,
            "window_stride": 0.01,
            "subsampling_factor": 1,
        },
    )

    batch = dataset[conversations]

    assert torch.all(batch["spk_targets"] == -1.0)
    (returned_cut,) = next(iter(batch["conversations"])).list_cuts()
    assert returned_cut.recording.sources[0].type == "shar"
    assert returned_cut.custom["_source_read_key"] == "/data/shard-0.tar#sample-0"


@pytest.mark.unit
def test_salm_dataset_can_return_packed_audio_samples(monkeypatch):
    cut = dummy_cut(0, duration=0.03, recording=dummy_recording(0, duration=0.03, with_data=True))
    conversation = NeMoMultimodalConversation(
        id="example-0",
        turns=[
            AudioTurn(role="user", cut=cut, audio_locator_tag="<|audio|>"),
            TextTurn(role="assistant", value="hello"),
        ],
        token_equivalent_duration=0.01,
    )
    conversation.input_ids = torch.tensor([7, 8], dtype=torch.long)
    conversation.mask = torch.tensor([False, True])
    conversations = CutSet([conversation])
    expected_samples = torch.tensor([1.0, 2.0, 3.0])
    expected_cu_seqlens = torch.tensor([0, 3], dtype=torch.long)
    expected_lens = torch.tensor([3], dtype=torch.long)

    def fake_packed_audio_collate(conversations_arg, *args, **kwargs):
        assert conversations_arg is conversations
        return expected_samples, expected_cu_seqlens, expected_lens, conversations_arg

    monkeypatch.setattr(
        salm_dataset_module,
        "collate_conversation_audio_packed_fault_tolerant",
        fake_packed_audio_collate,
    )
    dataset = salm_dataset_module.SALMDataset(tokenizer=_Tokenizer(), pack_audio=True)

    batch = dataset[conversations]

    assert "audios" not in batch
    assert batch["packed_audio_samples"] is expected_samples
    assert batch["audio_cu_seqlens"] is expected_cu_seqlens
    assert batch["audio_lens"] is expected_lens
    assert batch["input_ids"].shape == (1, 2)


@pytest.mark.unit
def test_salm_dataset_packed_sequences_never_pads_variable_length_tensors(monkeypatch):
    conversations = []
    for idx, (input_ids, mask) in enumerate(
        (
            ([7, 8, 9], [False, True, True]),
            ([10, 11], [False, True]),
        )
    ):
        cut = dummy_cut(idx, duration=0.03, recording=dummy_recording(idx, duration=0.03, with_data=True))
        conversation = NeMoMultimodalConversation(
            id=f"example-{idx}",
            turns=[
                AudioTurn(role="user", cut=cut, audio_locator_tag="<|audio|>"),
                TextTurn(role="assistant", value="hello"),
            ],
            token_equivalent_duration=0.01,
        )
        conversation.input_ids = torch.tensor(input_ids, dtype=torch.long)
        conversation.mask = torch.tensor(mask)
        conversations.append(conversation)
    conversations = CutSet(conversations)

    def fake_packed_audio_collate(conversations_arg, *args, **kwargs):
        return (
            torch.tensor([1.0, 2.0, 10.0, 11.0, 12.0]),
            torch.tensor([0, 2, 5], dtype=torch.long),
            torch.tensor([2, 3], dtype=torch.long),
            conversations_arg,
        )

    monkeypatch.setattr(
        salm_dataset_module,
        "collate_conversation_audio_packed_fault_tolerant",
        fake_packed_audio_collate,
    )
    dataset = salm_dataset_module.SALMDataset(tokenizer=_Tokenizer(), pack_sequences=True)

    batch = dataset[conversations]

    assert dataset.pack_audio is True
    assert "audios" not in batch
    assert batch["packed_audio_samples"].shape == (5,)
    assert torch.equal(batch["audio_cu_seqlens"], torch.tensor([0, 2, 5]))
    assert torch.equal(batch["input_ids"], torch.tensor([7, 8, 9, 10, 11]))
    assert torch.equal(batch["loss_mask"], torch.tensor([False, True, True, False, True]))
    assert torch.equal(batch["text_cu_seqlens"], torch.tensor([0, 3, 5]))


@pytest.mark.unit
def test_salm_dataset_packs_multispeaker_targets_without_time_padding(monkeypatch):
    conversations = []
    for idx in range(2):
        cut = dummy_cut(idx, duration=0.03, recording=dummy_recording(idx, duration=0.03, with_data=True))
        conversation = NeMoMultimodalConversation(
            id=f"example-{idx}",
            turns=[
                AudioTurn(role="user", cut=cut, audio_locator_tag="<|audio|>"),
                TextTurn(role="assistant", value="hello"),
            ],
            token_equivalent_duration=0.01,
        )
        conversation.input_ids = torch.tensor([7 + idx, 9], dtype=torch.long)
        conversation.mask = torch.tensor([False, True])
        conversations.append(conversation)
    conversations = CutSet(conversations)

    def fake_packed_audio_collate(conversations_arg, *args, **kwargs):
        return (
            torch.arange(6, dtype=torch.float32),
            torch.tensor([0, 3, 6], dtype=torch.long),
            torch.tensor([3, 3], dtype=torch.long),
            conversations_arg,
        )

    activity_lengths = iter((3, 1))

    def fake_speaker_activity_from_cut(cut, **kwargs):
        return torch.ones(next(activity_lengths), 1)

    monkeypatch.setattr(
        salm_dataset_module,
        "collate_conversation_audio_packed_fault_tolerant",
        fake_packed_audio_collate,
    )
    monkeypatch.setattr(salm_dataset_module, "speaker_activity_from_cut", fake_speaker_activity_from_cut)
    dataset = salm_dataset_module.SALMDataset(
        tokenizer=_Tokenizer(),
        pack_sequences=True,
        multispeaker_cfg={"num_speakers": 2},
    )

    batch = dataset[conversations]

    assert batch["spk_targets"].shape == (4, 2)
    assert torch.all(batch["spk_targets"] == -1.0)
    assert torch.equal(batch["spk_target_length"], torch.tensor([3, 1]))
    assert torch.equal(batch["spk_target_cu_seqlens"], torch.tensor([0, 3, 4]))


@pytest.mark.unit
def test_salm_dataset_reports_exact_packing_efficiency(monkeypatch):
    conversations = []
    for idx, num_tokens in enumerate((3072, 7168)):
        cut = dummy_cut(idx, duration=0.03, recording=dummy_recording(idx, duration=0.03, with_data=True))
        conversation = NeMoMultimodalConversation(
            id=f"example-{idx}",
            turns=[
                AudioTurn(role="user", cut=cut, audio_locator_tag="<|audio|>"),
                TextTurn(role="assistant", value="hello"),
            ],
            token_equivalent_duration=0.01,
        )
        conversation.input_ids = torch.tensor([7, 8], dtype=torch.long)
        conversation.mask = torch.tensor([False, True])
        conversation.num_tokens = num_tokens
        conversations.append(conversation)
    conversations = CutSet(conversations)

    def fake_audio_collate(conversations_arg, *args, **kwargs):
        return torch.zeros(2, 480), torch.tensor([480, 480]), conversations_arg

    monkeypatch.setattr(salm_dataset_module, "collate_conversation_audio_fault_tolerant", fake_audio_collate)
    dataset = salm_dataset_module.SALMDataset(tokenizer=_Tokenizer(), batch_tokens=12288)

    batch = dataset[conversations]

    assert batch["packing_efficiency"].item() == pytest.approx(10240 / 12288)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rttm_filepath", "expected_targets"),
    [
        (
            "/fake/example.rttm",
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
        ),
        (None, [[-1.0, -1.0]] * 4),
    ],
)
def test_salm_dataset_routes_speaker_targets_by_rttm_presence(monkeypatch, rttm_filepath, expected_targets):
    text = "<spk:0> hello world <spk:1> yes now"
    cut = dummy_cut(0, duration=0.04, recording=dummy_recording(0, duration=0.04, with_data=True))
    cut.custom = {"rttm_filepath": rttm_filepath} if rttm_filepath is not None else {}
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
        activity = torch.tensor(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        )
        if kwargs.get("return_permutation_resolved"):
            return activity, False
        return activity

    monkeypatch.setattr(salm_dataset_module, "collate_conversation_audio_fault_tolerant", fake_audio_collate)
    monkeypatch.setattr(salm_dataset_module, "speaker_activity_from_cut", fake_speaker_activity_from_cut)
    alignment_calls = []
    original_fix_speaker_activity = salm_dataset_module.fix_speaker_activity

    def counted_fix_speaker_activity(*args, **kwargs):
        alignment_calls.append(True)
        return original_fix_speaker_activity(*args, **kwargs)

    monkeypatch.setattr(salm_dataset_module, "fix_speaker_activity", counted_fix_speaker_activity)

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
        torch.tensor([expected_targets]),
    )
    assert torch.equal(batch["spk_target_length"], torch.tensor([4]))
    assert len(alignment_calls) == (1 if rttm_filepath is not None else 0)


@pytest.mark.unit
def test_salm_dataset_skips_legacy_alignment_for_pr_rttm(monkeypatch):
    text = "<spk:0> hello world <spk:1> yes now"
    cut = dummy_cut(0, duration=0.04, recording=dummy_recording(0, duration=0.04, with_data=True))
    cut.custom = {"rttm_filepath": "/fake/pr.rttm"}
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
    resolved_activity = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )

    def fake_audio_collate(conversations, *args, **kwargs):
        return torch.zeros(1, 640), torch.tensor([640], dtype=torch.long), conversations

    def fake_speaker_activity_from_cut(cut, **kwargs):
        assert kwargs["text"] == text
        assert kwargs["return_permutation_resolved"]
        return resolved_activity, True

    def unexpected_fix_speaker_activity(*args, **kwargs):
        raise AssertionError("fix_speaker_activity must not run for permutation-resolved RTTMs")

    monkeypatch.setattr(salm_dataset_module, "collate_conversation_audio_fault_tolerant", fake_audio_collate)
    monkeypatch.setattr(salm_dataset_module, "speaker_activity_from_cut", fake_speaker_activity_from_cut)
    monkeypatch.setattr(salm_dataset_module, "fix_speaker_activity", unexpected_fix_speaker_activity)

    dataset = salm_dataset_module.SALMDataset(
        tokenizer=_Tokenizer(),
        multispeaker_cfg={
            "num_speakers": 2,
            "sample_rate": 16000,
            "window_stride": 0.01,
            "subsampling_factor": 1,
        },
    )
    batch = dataset[conversations]

    assert torch.equal(batch["spk_targets"], resolved_activity.unsqueeze(0))


@pytest.mark.unit
def test_salm_dataset_preserves_all_eight_pr_rttm_speakers(monkeypatch):
    text = " ".join(f"<spk:{speaker}> utterance-{speaker}" for speaker in range(8))
    cut = dummy_cut(0, duration=0.08, recording=dummy_recording(0, duration=0.08, with_data=True))
    cut.custom = {"rttm_filepath": "/fake/pr-8speaker.rttm"}
    cut.supervisions = [
        SupervisionSegment(id=cut.id, recording_id=cut.recording_id, start=0.0, duration=0.08, text=text)
    ]
    conversation = NeMoMultimodalConversation(
        id="eight-speaker-example",
        turns=[
            AudioTurn(role="user", cut=cut, audio_locator_tag="<|audio|>", text=text),
            TextTurn(role="assistant", value=text),
        ],
        token_equivalent_duration=0.01,
    )
    conversation.input_ids = torch.tensor([7, 8, 9], dtype=torch.long)
    conversation.mask = torch.tensor([False, True, True])
    conversations = CutSet([conversation])
    resolved_activity = torch.eye(8)

    def fake_audio_collate(conversations, *args, **kwargs):
        return torch.zeros(1, 1280), torch.tensor([1280], dtype=torch.long), conversations

    def fake_speaker_activity_from_cut(cut, **kwargs):
        assert kwargs["num_speakers"] == 8
        assert kwargs["text"] == text
        assert kwargs["return_permutation_resolved"]
        return resolved_activity, True

    def unexpected_fix_speaker_activity(*args, **kwargs):
        raise AssertionError("Permutation-resolved 8-speaker PR-RTTM targets must not be realigned")

    monkeypatch.setattr(salm_dataset_module, "collate_conversation_audio_fault_tolerant", fake_audio_collate)
    monkeypatch.setattr(salm_dataset_module, "speaker_activity_from_cut", fake_speaker_activity_from_cut)
    monkeypatch.setattr(salm_dataset_module, "fix_speaker_activity", unexpected_fix_speaker_activity)

    dataset = salm_dataset_module.SALMDataset(
        tokenizer=_Tokenizer(),
        multispeaker_cfg={
            "num_speakers": 8,
            "sample_rate": 16000,
            "window_stride": 0.01,
            "subsampling_factor": 1,
        },
    )
    batch = dataset[conversations]

    assert batch["spk_targets"].shape == (1, 8, 8)
    assert torch.equal(batch["spk_targets"], resolved_activity.unsqueeze(0))
    assert batch["spk_target_length"].tolist() == [8]


@pytest.mark.unit
def test_salm_dataset_mixed_rttm_batch_survives_chunking(monkeypatch):
    text = "<spk:0> hello world <spk:1> yes now"
    conversations = []
    for idx, rttm_filepath in enumerate(("/fake/example.rttm", None)):
        cut = dummy_cut(
            idx,
            duration=0.04,
            recording=dummy_recording(idx, duration=0.04, with_data=True),
        )
        cut.custom = {"rttm_filepath": rttm_filepath} if rttm_filepath is not None else {}
        cut.supervisions = [
            SupervisionSegment(id=cut.id, recording_id=cut.recording_id, start=0.0, duration=0.04, text=text)
        ]
        conversation = NeMoMultimodalConversation(
            id=f"example-{idx}",
            turns=[
                AudioTurn(role="user", cut=cut, audio_locator_tag="<|audio|>", text=text),
                TextTurn(role="assistant", value=text),
            ],
            token_equivalent_duration=0.01,
        )
        conversation.input_ids = torch.tensor([7, 8, 9], dtype=torch.long)
        conversation.mask = torch.tensor([False, True, True])
        conversations.append(conversation)
    conversations = CutSet(conversations)

    def fake_audio_collate(conversations, *args, **kwargs):
        return torch.zeros(2, 640), torch.tensor([640, 640], dtype=torch.long), conversations

    def fake_speaker_activity_from_cut(cut, **kwargs):
        activity = torch.tensor(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        )
        if kwargs.get("return_permutation_resolved"):
            return activity, False
        return activity

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
    batch = dataset[conversations]

    expected_targets = torch.tensor(
        [
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            [[-1.0, -1.0]] * 4,
        ]
    )
    assert torch.equal(batch["spk_targets"], expected_targets)
    assert torch.equal(batch["spk_target_length"], torch.tensor([4, 4]))

    chunked_targets = _split_spk_targets_into_chunks(
        batch["spk_targets"],
        input_signal_lengths=[640, 640],
        chunk_spans=[
            (0, 0, 320),
            (0, 320, 640),
            (1, 0, 320),
            (1, 320, 640),
        ],
        spk_target_lengths=batch["spk_target_length"],
        spk_target_stride=160,
    )
    assert torch.equal(
        chunked_targets,
        torch.stack(
            [
                expected_targets[0, :2],
                expected_targets[0, 2:],
                expected_targets[1, :2],
                expected_targets[1, 2:],
            ]
        ),
    )
