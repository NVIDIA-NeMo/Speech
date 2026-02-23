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
"""Qwen Forced Aligner wrapper for word-level alignment in StreamingSALM."""

from __future__ import annotations

import torch
from torch import Tensor

from nemo.collections.speechlm2.parts.interleaving import WordAlignment


class QwenForcedAligner:
    """
    Wraps qwen_asr.Qwen3ForcedAligner to provide word-level alignment
    from audio tensors and text.

    Used during training to obtain word timestamps for interleaving.
    Internally resamples to 16 kHz if the input is at a different rate.
    """

    SAMPLE_RATE = 16000  # QFA expects 16kHz input

    def __init__(
        self,
        pretrained_model: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        language: str = "English",
    ):
        import qwen_asr

        self.aligner = qwen_asr.Qwen3ForcedAligner.from_pretrained(pretrained_model)
        self.language = language

    @torch.no_grad()
    def align(
        self,
        audio: Tensor,
        audio_lens: Tensor,
        texts: list[str],
        source_sample_rate: int = 16000,
    ) -> list[list[WordAlignment]]:
        """
        Compute word-level forced alignment.

        Args:
            audio: (B, T_samples) raw audio waveform.
            audio_lens: (B,) sample counts.
            texts: list of B transcription strings.
            source_sample_rate: sample rate of ``audio``.  The audio will
                be resampled to 16 kHz internally if needed.

        Returns:
            List of B lists of WordAlignment (one per word per utterance).
        """
        if source_sample_rate != self.SAMPLE_RATE:
            from nemo.collections.audio.parts.utils.transforms import resample

            audio = resample(audio, source_sample_rate, self.SAMPLE_RATE)
            ratio = self.SAMPLE_RATE / source_sample_rate
            audio_lens = (audio_lens.float() * ratio).long()

        # Convert tensor to list of (ndarray, sample_rate) tuples for qwen_asr
        audio_list = []
        for i in range(audio.shape[0]):
            samples = audio[i, : audio_lens[i]].cpu().numpy()
            audio_list.append((samples, self.SAMPLE_RATE))

        results = self.aligner.align(audio_list, texts, self.language)

        # Convert ForcedAlignResult/ForcedAlignItem → List[List[WordAlignment]]
        word_alignments = []
        for result in results:
            words = []
            for item in result:
                words.append(
                    WordAlignment(
                        text=item.text,
                        start_time=item.start_time,
                        end_time=item.end_time,
                    )
                )
            word_alignments.append(words)
        return word_alignments
