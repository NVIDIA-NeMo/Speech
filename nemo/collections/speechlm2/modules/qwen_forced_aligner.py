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

import logging

import numpy as np
import torch
from torch import Tensor

from nemo.collections.speechlm2.parts.alignments import ForcedAligner, WordAlignment
from nemo.collections.speechlm2.parts.utils import freeze_module

log = logging.getLogger(__name__)


class QwenForcedAligner(torch.nn.Module, ForcedAligner):
    """
    Wraps qwen_asr.Qwen3ForcedAligner to provide word-level alignment
    from audio tensors and text.

    Used during training to obtain word timestamps for interleaving.
    Supports two input modes:

    1. **GPU tensor** (legacy): ``align(audio_tensor, audio_lens, texts)``
       Resamples on GPU, transfers to CPU, then runs alignment.

    2. **Pre-resampled numpy** (fast): ``align_numpy(audio_arrays, texts)``
       Accepts 16 kHz numpy arrays directly (produced by dataloader workers),
       avoiding GPU resampling and GPU→CPU transfer overhead.
    """

    SAMPLE_RATE = 16000  # QFA expects 16kHz input

    def __init__(
        self,
        pretrained_model: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        language: str = "English",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        import qwen_asr

        self.aligner = qwen_asr.Qwen3ForcedAligner.from_pretrained(
            pretrained_model,
            torch_dtype=dtype,
            device_map=device,
        )
        self.language = language
        freeze_module(self)

    @torch.no_grad()
    def align_numpy(
        self,
        audio_arrays: list[np.ndarray],
        texts: list[str],
    ) -> list[list[WordAlignment]]:
        """
        Fast alignment from pre-resampled 16 kHz numpy arrays.

        Skips GPU resampling and GPU→CPU transfer — the audio stays on CPU
        and goes directly into the HF processor for mel extraction.

        Args:
            audio_arrays: list of B numpy float32 arrays at 16 kHz.
            texts: list of B transcription strings.

        Returns:
            List of B lists of WordAlignment (one per word per utterance).
        """
        # Prepare text input using the aligner's text processor
        word_lists = []
        aligner_input_texts = []
        for t in texts:
            wl, ait = self.aligner.aligner_processor.encode_timestamp(t, self.language)
            word_lists.append(wl)
            aligner_input_texts.append(ait)

        # Prepare audio: qwen_asr processor expects list[ndarray] at 16 kHz.
        # normalize_audio_input would resample+normalize, but our audio is already
        # 16 kHz float32 mono from the dataloader. We just need float range [-1, 1].
        normed = []
        for arr in audio_arrays:
            if isinstance(arr, torch.Tensor):
                arr = arr.cpu().float().numpy()
            a = np.asarray(arr, dtype=np.float32)
            peak = float(np.max(np.abs(a))) if a.size > 0 else 0.0
            if peak > 1.0:
                a = a / peak
            normed.append(a)

        # Temporarily restore float32 as default dtype so that qwen_asr's internal
        # tensor-from-scalar operations don't produce bf16 (which numpy can't handle).
        prev_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)

            inputs = self.aligner.processor(
                text=aligner_input_texts,
                audio=normed,
                return_tensors="pt",
                padding=True,
            )
            inputs = inputs.to(self.aligner.model.device).to(self.aligner.model.dtype)

            logits = self.aligner.model.thinker(**inputs).logits
        finally:
            torch.set_default_dtype(prev_dtype)

        output_ids = logits.argmax(dim=-1)

        # Post-process: extract timestamps
        word_alignments = []
        for input_id, output_id, word_list in zip(inputs["input_ids"], output_ids, word_lists):
            masked_output_id = output_id[input_id == self.aligner.timestamp_token_id]
            timestamp_ms = (masked_output_id * self.aligner.timestamp_segment_time).cpu().float().numpy()
            ts_out = self.aligner.aligner_processor.parse_timestamp(word_list, timestamp_ms)
            words = []
            for it in ts_out:
                words.append(
                    WordAlignment(
                        text=it["text"],
                        start_time=round(it["start_time"] / 1000.0, 3),
                        end_time=round(it["end_time"] / 1000.0, 3),
                    )
                )
            word_alignments.append(words)
        return word_alignments

    @torch.no_grad()
    def align(
        self,
        audio: Tensor,
        audio_lens: Tensor,
        texts: list[str],
        source_sample_rate: int = 16000,
    ) -> list[list[WordAlignment]]:
        """
        Legacy alignment from GPU tensors. Prefer ``align_numpy`` for training.
        """
        if source_sample_rate != self.SAMPLE_RATE:
            from nemo.collections.audio.parts.utils.transforms import resample

            audio = resample(audio, source_sample_rate, self.SAMPLE_RATE)
            ratio = self.SAMPLE_RATE / source_sample_rate
            audio_lens = (audio_lens.float() * ratio).long()

        # Batch GPU→CPU transfer (single call instead of per-utterance loop)
        audio_cpu = audio.cpu().float()
        audio_list = [(audio_cpu[i, : audio_lens[i]].numpy(), self.SAMPLE_RATE) for i in range(audio_cpu.shape[0])]

        prev_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            results = self.aligner.align(audio_list, texts, self.language)
        finally:
            torch.set_default_dtype(prev_dtype)

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
