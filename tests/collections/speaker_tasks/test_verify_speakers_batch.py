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

import numpy as np
import pytest

from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel


class _FakeSpeakerModel:
    """Stands in for an EncDecSpeakerLabelModel: only implements what verify_speakers_batch calls."""

    def __init__(self, embeddings1, embeddings2):
        self._embeddings1 = embeddings1
        self._embeddings2 = embeddings2
        self._batch_inference_calls = 0

    def path2audio_files_to_manifest(self, audio_files, manifest_filepath):
        # verify_speakers_batch only needs this to not raise; batch_inference below is stubbed
        # to ignore the manifest file entirely, so no manifest content is required.
        with open(manifest_filepath, 'w'):
            pass

    def batch_inference(self, manifest_filepath, batch_size, sample_rate, device):
        self._batch_inference_calls += 1
        embs = self._embeddings1 if self._batch_inference_calls == 1 else self._embeddings2
        return embs, None, None, None


@pytest.mark.unit
def test_verify_speakers_batch_single_pair_returns_indexable_array():
    """A single-pair batch must return a length-1 array, matching verify_speakers_batch's own
    documented per-pair 'True/False' contract -- not a 0-d scalar that cannot be indexed or iterated."""
    same_speaker_embedding = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    fake_model = _FakeSpeakerModel(same_speaker_embedding, same_speaker_embedding)

    decision = EncDecSpeakerLabelModel.verify_speakers_batch(
        fake_model,
        [("audio_a.wav", "audio_b.wav")],
        threshold=0.5,
        batch_size=1,
        sample_rate=16000,
        device='cpu',
    )

    assert decision.shape == (1,), f"expected a length-1 array, got shape {decision.shape}"
    # A 0-d array raises on both of these -- this is the user-visible symptom.
    assert decision[0] == True  # noqa: E712
    assert list(decision) == [True]


@pytest.mark.unit
def test_verify_speakers_batch_multi_pair_unaffected():
    """The fix must not change results for batches of more than one pair."""
    same = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    different = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    embs1 = np.stack([same, same, same])
    embs2 = np.stack([same, different, same])
    fake_model = _FakeSpeakerModel(embs1, embs2)

    decision = EncDecSpeakerLabelModel.verify_speakers_batch(
        fake_model,
        [("a0", "b0"), ("a1", "b1"), ("a2", "b2")],
        threshold=0.7,
        batch_size=3,
        sample_rate=16000,
        device='cpu',
    )

    assert decision.shape == (3,)
    assert list(decision) == [True, False, True]
