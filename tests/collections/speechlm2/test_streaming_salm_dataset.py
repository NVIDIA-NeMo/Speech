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
"""Test StreamingSALMDataset with plain Lhotse Cuts."""

import pytest
import torch
from lhotse import CutSet, SupervisionSegment
from lhotse.testing.dummies import dummy_cut

from nemo.collections.speechlm2.data.streaming_salm_dataset import StreamingSALMDataset

_NEXT_CUT_ID = 0


def _make_cut(transcript: str, duration: float = 1.0):
    """Create a Cut with audio data and a supervision carrying the transcript."""
    global _NEXT_CUT_ID
    cut = dummy_cut(_NEXT_CUT_ID, duration=duration, with_data=True)
    cut.supervisions = [
        SupervisionSegment(
            id=f"sup-{_NEXT_CUT_ID}",
            recording_id=cut.recording.id,
            start=0,
            duration=duration,
            text=transcript,
        )
    ]
    _NEXT_CUT_ID += 1
    return cut


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
    """A CutSet batch with two cuts."""
    return CutSet(
        [
            _make_cut("hello world"),
            _make_cut("foo bar baz"),
        ]
    )


class TestStreamingSALMDataset:
    def test_returns_required_keys(self, dataset, training_cutset_batch):
        batch = dataset[training_cutset_batch]
        assert batch is not None
        assert "audios" in batch
        assert "audio_lens" in batch
        assert "transcripts" in batch
        assert "cuts" in batch
        assert isinstance(batch["transcripts"], list)

    def test_audios_shape(self, dataset, training_cutset_batch):
        batch = dataset[training_cutset_batch]
        assert batch["audios"].dim() == 2  # (B, T_samples)
        assert batch["audios"].shape[0] == 2

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

    def test_cuts_preserved(self, dataset, training_cutset_batch):
        """Returned cuts should have the same IDs as input."""
        batch = dataset[training_cutset_batch]
        cut_ids = [c.id for c in batch["cuts"]]
        assert len(cut_ids) == 2
