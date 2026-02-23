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
"""Test StreamingSALMDataset and extract_transcripts."""

import pytest
import torch
from lhotse import CutSet
from lhotse.testing.dummies import dummy_cut

from nemo.collections.common.data.lhotse import NeMoMultimodalConversation
from nemo.collections.common.data.lhotse.text_adapters import AudioTurn, TextTurn
from nemo.collections.speechlm2.data.streaming_salm_dataset import (
    StreamingSALMDataset,
    extract_transcripts,
)


AUDIO_LOCATOR_TAG = "<aud>"


_NEXT_CUT_ID = 0


def _make_conversation(conv_id: str, transcript: str, duration: float = 1.0) -> NeMoMultimodalConversation:
    """Create a simple NeMoMultimodalConversation with one user audio turn and one assistant text turn."""
    global _NEXT_CUT_ID
    cut = dummy_cut(_NEXT_CUT_ID, duration=duration, with_data=True)
    _NEXT_CUT_ID += 1
    return NeMoMultimodalConversation(
        id=conv_id,
        turns=[
            TextTurn(role="user", value="Transcribe:"),
            AudioTurn(role="user", cut=cut, audio_locator_tag=AUDIO_LOCATOR_TAG),
            TextTurn(role="assistant", value=transcript),
        ],
        token_equivalent_duration=0.08,
    )


class _FakeTokenizer:
    """Minimal tokenizer stub for StreamingSALMDataset tests."""

    @property
    def pad(self):
        return 0

    @property
    def unk_id(self):
        return 0


@pytest.fixture
def tokenizer():
    return _FakeTokenizer()


@pytest.fixture
def dataset(tokenizer):
    return StreamingSALMDataset(tokenizer=tokenizer)


@pytest.fixture
def training_cutset_batch():
    """A CutSet batch with two conversations."""
    return CutSet(
        [
            _make_conversation("conv-0", "hello world"),
            _make_conversation("conv-1", "foo bar baz"),
        ]
    )


class TestExtractTranscripts:
    def test_single_assistant_turn(self):
        conv = _make_conversation("test", "hello world")
        transcripts = extract_transcripts(CutSet([conv]))
        assert transcripts == ["hello world"]

    def test_multiple_conversations(self):
        convs = CutSet(
            [
                _make_conversation("c0", "first transcript"),
                _make_conversation("c1", "second transcript"),
                _make_conversation("c2", "third transcript"),
            ]
        )
        transcripts = extract_transcripts(convs)
        assert transcripts == ["first transcript", "second transcript", "third transcript"]

    def test_empty_assistant_turn(self):
        """Conversation with no assistant turn should produce empty string."""
        global _NEXT_CUT_ID
        cut = dummy_cut(_NEXT_CUT_ID, duration=1.0, with_data=True)
        _NEXT_CUT_ID += 1
        conv = NeMoMultimodalConversation(
            id="no-assistant",
            turns=[
                TextTurn(role="user", value="Transcribe:"),
                AudioTurn(role="user", cut=cut, audio_locator_tag=AUDIO_LOCATOR_TAG),
            ],
            token_equivalent_duration=0.08,
        )
        transcripts = extract_transcripts(CutSet([conv]))
        assert transcripts == [""]

    def test_multiple_assistant_turns_joined(self):
        """Multiple assistant text turns should be joined with a space."""
        global _NEXT_CUT_ID
        cut = dummy_cut(_NEXT_CUT_ID, duration=1.0, with_data=True)
        _NEXT_CUT_ID += 1
        conv = NeMoMultimodalConversation(
            id="multi-assistant",
            turns=[
                TextTurn(role="user", value="Transcribe:"),
                AudioTurn(role="user", cut=cut, audio_locator_tag=AUDIO_LOCATOR_TAG),
                TextTurn(role="assistant", value="hello"),
                TextTurn(role="assistant", value="world"),
            ],
            token_equivalent_duration=0.08,
        )
        transcripts = extract_transcripts(CutSet([conv]))
        assert transcripts == ["hello world"]

    def test_user_text_not_included(self):
        """Only assistant turns are extracted, not user turns."""
        global _NEXT_CUT_ID
        cut = dummy_cut(_NEXT_CUT_ID, duration=1.0, with_data=True)
        _NEXT_CUT_ID += 1
        conv = NeMoMultimodalConversation(
            id="user-only-text",
            turns=[
                TextTurn(role="user", value="Please transcribe this audio"),
                AudioTurn(role="user", cut=cut, audio_locator_tag=AUDIO_LOCATOR_TAG),
                TextTurn(role="assistant", value="target text"),
            ],
            token_equivalent_duration=0.08,
        )
        transcripts = extract_transcripts(CutSet([conv]))
        assert transcripts == ["target text"]
        assert "Please transcribe" not in transcripts[0]


class TestStreamingSALMDataset:
    def test_returns_required_keys(self, dataset, training_cutset_batch):
        batch = dataset[training_cutset_batch]
        assert batch is not None
        assert "audios" in batch
        assert "audio_lens" in batch
        assert "transcripts" in batch
        assert "conversations" in batch
        assert isinstance(batch["transcripts"], list)

    def test_audios_shape(self, dataset, training_cutset_batch):
        batch = dataset[training_cutset_batch]
        assert batch["audios"].dim() == 2  # (B_audio, T_samples)
        assert batch["audios"].shape[0] == 2  # 2 conversations, each with 1 audio turn

    def test_audio_lens_shape(self, dataset, training_cutset_batch):
        batch = dataset[training_cutset_batch]
        assert batch["audio_lens"].dim() == 1
        assert batch["audio_lens"].shape[0] == 2

    def test_transcripts_content(self, dataset, training_cutset_batch):
        batch = dataset[training_cutset_batch]
        assert batch["transcripts"] == ["hello world", "foo bar baz"]

    def test_returns_none_on_empty_cutset(self, dataset):
        """Empty CutSet should return None (handled by FallbackDataset)."""
        result = dataset[CutSet([])]
        assert result is None

    def test_conversations_preserved(self, dataset, training_cutset_batch):
        """Returned conversations should have the same IDs as input."""
        batch = dataset[training_cutset_batch]
        conv_ids = [c.id for c in batch["conversations"]]
        assert conv_ids == ["conv-0", "conv-1"]
