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
"""Test QwenForcedAligner wrapper with mocked qwen-asr backend."""

import pytest
import torch

from nemo.collections.speechlm2.parts.interleaving import WordAlignment


@pytest.fixture
def mock_qfa(monkeypatch):
    """Mock the qwen_asr.Qwen3ForcedAligner."""

    class FakeAlignmentResult:
        def __init__(self, text, start, end):
            self.text = text
            self.start_time = start
            self.end_time = end

    class FakeAligner:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def align(self, audio, text, language):
            results = []
            for t in text:
                words = t.split()
                time = 0.0
                word_results = []
                for w in words:
                    duration = len(w) * 0.1  # fake: 100ms per character
                    word_results.append(FakeAlignmentResult(w, time, time + duration))
                    time += duration + 0.05
                results.append(word_results)
            return results

    import qwen_asr

    monkeypatch.setattr(qwen_asr, "Qwen3ForcedAligner", FakeAligner)


class TestQwenForcedAligner:
    def test_align_returns_word_alignments(self, mock_qfa):
        from nemo.collections.speechlm2.modules.qwen_forced_aligner import QwenForcedAligner

        aligner = QwenForcedAligner(pretrained_model="fake")
        audio = torch.randn(1, 16000)
        audio_lens = torch.tensor([16000])
        results = aligner.align(audio, audio_lens, ["hello world"])
        assert len(results) == 1
        assert len(results[0]) == 2  # two words
        assert results[0][0].text == "hello"
        assert results[0][1].text == "world"

    def test_align_batch(self, mock_qfa):
        from nemo.collections.speechlm2.modules.qwen_forced_aligner import QwenForcedAligner

        aligner = QwenForcedAligner(pretrained_model="fake")
        audio = torch.randn(3, 16000)
        audio_lens = torch.tensor([16000, 16000, 16000])
        results = aligner.align(audio, audio_lens, ["a b", "c d e", "f"])
        assert len(results) == 3
        assert len(results[0]) == 2
        assert len(results[1]) == 3
        assert len(results[2]) == 1

    def test_alignment_times_are_monotonic(self, mock_qfa):
        from nemo.collections.speechlm2.modules.qwen_forced_aligner import QwenForcedAligner

        aligner = QwenForcedAligner(pretrained_model="fake")
        audio = torch.randn(1, 32000)
        audio_lens = torch.tensor([32000])
        results = aligner.align(audio, audio_lens, ["the quick brown fox"])
        for i in range(1, len(results[0])):
            assert results[0][i].start_time >= results[0][i - 1].start_time

    def test_word_alignment_dataclass(self):
        wa = WordAlignment(text="hello", start_time=0.5, end_time=1.0)
        assert wa.text == "hello"
        assert wa.start_time == 0.5
        assert wa.end_time == 1.0

    # ---- New tests below: verify real wrapper logic ----

    def test_result_types_are_word_alignment(self, mock_qfa):
        """Verify the wrapper converts raw aligner results to WordAlignment dataclass."""
        from nemo.collections.speechlm2.modules.qwen_forced_aligner import QwenForcedAligner

        aligner = QwenForcedAligner(pretrained_model="fake")
        audio = torch.randn(1, 16000)
        audio_lens = torch.tensor([16000])
        results = aligner.align(audio, audio_lens, ["hello world"])
        for word in results[0]:
            assert isinstance(word, WordAlignment)
            assert isinstance(word.start_time, float)
            assert isinstance(word.end_time, float)

    def test_audio_sliced_to_audio_lens(self, mock_qfa):
        """Verify each sample is sliced to audio_lens[i] before passing to the backend."""
        from nemo.collections.speechlm2.modules.qwen_forced_aligner import QwenForcedAligner

        received_audio = []

        class InspectingAligner:
            @classmethod
            def from_pretrained(cls, *a, **kw):
                return cls()

            def align(self, audio, text, language):
                received_audio.extend(audio)
                results = []
                for t in text:
                    results.append([])
                return results

        import qwen_asr
        original = qwen_asr.Qwen3ForcedAligner
        qwen_asr.Qwen3ForcedAligner = InspectingAligner

        try:
            aligner = QwenForcedAligner(pretrained_model="fake")
            audio = torch.randn(2, 32000)
            audio_lens = torch.tensor([16000, 24000])
            aligner.align(audio, audio_lens, ["a", "b"], source_sample_rate=16000)

            # First sample should have exactly 16000 samples
            assert received_audio[0][0].shape[0] == 16000
            # Second sample should have exactly 24000 samples
            assert received_audio[1][0].shape[0] == 24000
        finally:
            qwen_asr.Qwen3ForcedAligner = original

    def test_audio_passed_as_numpy_tuples(self, mock_qfa):
        """Verify audio is converted to (ndarray, sample_rate) tuples for qwen_asr."""
        from nemo.collections.speechlm2.modules.qwen_forced_aligner import QwenForcedAligner
        import numpy as np

        received_audio = []

        class InspectingAligner:
            @classmethod
            def from_pretrained(cls, *a, **kw):
                return cls()

            def align(self, audio, text, language):
                received_audio.extend(audio)
                results = []
                for t in text:
                    results.append([])
                return results

        import qwen_asr
        original = qwen_asr.Qwen3ForcedAligner
        qwen_asr.Qwen3ForcedAligner = InspectingAligner

        try:
            aligner = QwenForcedAligner(pretrained_model="fake")
            audio = torch.randn(1, 16000)
            audio_lens = torch.tensor([16000])
            aligner.align(audio, audio_lens, ["hello"], source_sample_rate=16000)

            # Should be a (ndarray, int) tuple
            assert isinstance(received_audio[0], tuple)
            assert isinstance(received_audio[0][0], np.ndarray)
            assert received_audio[0][1] == 16000
        finally:
            qwen_asr.Qwen3ForcedAligner = original

    def test_resampling_adjusts_audio_lens(self, mock_qfa):
        """When source_sample_rate != 16kHz, audio_lens should be scaled accordingly."""
        from nemo.collections.speechlm2.modules.qwen_forced_aligner import QwenForcedAligner

        received_audio = []

        class InspectingAligner:
            @classmethod
            def from_pretrained(cls, *a, **kw):
                return cls()

            def align(self, audio, text, language):
                received_audio.extend(audio)
                results = []
                for t in text:
                    results.append([])
                return results

        import qwen_asr
        original = qwen_asr.Qwen3ForcedAligner
        qwen_asr.Qwen3ForcedAligner = InspectingAligner

        try:
            aligner = QwenForcedAligner(pretrained_model="fake")
            # 24kHz audio with 24000 samples = 1 second
            # After resampling to 16kHz: 16000 samples
            audio = torch.randn(1, 24000)
            audio_lens = torch.tensor([24000])
            aligner.align(audio, audio_lens, ["hello"], source_sample_rate=24000)

            # Expected: resampled to 16kHz, so sample count = 24000 * (16000/24000) = 16000
            assert received_audio[0][0].shape[0] == 16000
            assert received_audio[0][1] == 16000
        finally:
            qwen_asr.Qwen3ForcedAligner = original
