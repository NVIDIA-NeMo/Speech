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

"""Streaming state and step data for EasyMagpieTTS teacher rollouts."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from nemo.collections.tts.models.easy_magpietts_inference import StreamingConfig


@dataclass
class AudioCodesStep:
    """Predicted audio-code data for one rollout step."""

    codes_t: Tensor
    codes_t_argmax: Tensor
    logits_t: Tensor

    codes_t_lt: Optional[Tensor]
    codes_t_argmax_lt: Optional[Tensor]
    logits_t_lt: Optional[Tensor]

    def unstack(
        self,
        batch_size: int,
        num_codebooks: int,
        fs_factor: int,
    ) -> None:
        """Reshape flattened audio-code predictions into codec layout.

        Args:
            batch_size: Number of batch items represented by the flattened tensors.
            num_codebooks: Number of codec codebooks.
            fs_factor: Number of audio frames stacked into each prediction step.
        """
        self.codes_t = self.codes_t.view(batch_size, num_codebooks, fs_factor)
        self.codes_t_argmax = self.codes_t_argmax.view(batch_size, num_codebooks, fs_factor)

        if self.codes_t_lt is not None and self.codes_t_argmax_lt is not None and self.logits_t_lt is not None:
            self.codes_t_lt = self.codes_t_lt.view(batch_size, num_codebooks, fs_factor)
            self.codes_t_argmax_lt = self.codes_t_argmax_lt.view(batch_size, num_codebooks, fs_factor)


@dataclass
class PhonemeTokensStep:
    """Predicted phoneme data for one rollout step."""

    tokens_t: Tensor
    logits_t: Tensor


@dataclass
class StreamingChannelStatus:
    """Track status masks for each model input channel."""

    context: Tensor
    text: Tensor
    phoneme: Tensor
    audio: Tensor


@dataclass
class TeacherStreamingState:
    """Maintain state across streaming teacher-rollout steps."""

    config: StreamingConfig
    past_key_values: Optional[tuple]
    cache_seq_len: int

    pred_codes: list[Tensor]
    pred_codes_logits: list[Tensor]
    pred_codes_lt: list[Tensor]
    pred_codes_logits_lt: list[Tensor]
    pred_phoneme_tokens: list[Tensor]
    pred_phoneme_logits: list[Tensor]

    context_audio_codes: Tensor
    context_audio_codes_lens: Tensor
    context_lens: Tensor
    full_context_embedding: Tensor
    full_context_lens: Tensor

    context_position: Tensor
    text_tokens_seen: Tensor
    phoneme_steps: Tensor
    audio_steps: Tensor
    phoneme_stream_ended: Tensor
    phoneme_eos_detected: Tensor
    finished: Tensor
    last_hidden: Tensor
    text_finished: Tensor

    last_audio_codes: Optional[Tensor]
    last_audio_codes_lt: Optional[Tensor]
    last_phoneme_tokens: Optional[Tensor]

    audio_prediction_start_idx: Tensor
    audio_prediction_end_idx: Tensor
    phoneme_prediction_start_idx: Tensor
    phoneme_prediction_end_idx: Tensor

    gt_phoneme_embeddings: Optional[Tensor] = None
    gt_phoneme_lens: Optional[Tensor] = None

    rejected: Optional[Tensor] = None
    sample_weights: Optional[Tensor] = None

    @classmethod
    def create(
        cls,
        config: StreamingConfig,
        last_hidden: Tensor,
        past_kv: list[Tensor],
        min_context_len: Tensor,
        context_audio_codes: Tensor,
        context_audio_codes_lens: Tensor,
        context_lens: Tensor,
        full_context_embedding: Tensor,
        full_context_lens: Tensor,
        gt_phoneme_embeddings: Tensor,
        gt_phoneme_lens: Tensor,
        use_rollout_rejection: bool,
        rejection_weight: Optional[float],
    ) -> "TeacherStreamingState":
        """Create an initialized streaming state for a teacher rollout.

        Initializes cache metadata, per-item counters, prediction buffers, and optional
        rollout-rejection state.

        Args:
            config: Static configuration for the streaming rollout.
            last_hidden: Hidden states from the prepared context forward pass.
            past_kv: Initial transformer key-value cache.
            min_context_len: Shared cached-context-prefix length, equal to the minimum
                prepared-context length across the batch.
            context_audio_codes: Prepared context audio codec tokens.
            context_audio_codes_lens: Lengths of the context audio-code sequences.
            context_lens: Lengths of the prepared context sequences.
            full_context_embedding: Embeddings for the complete prepared context.
            full_context_lens: Lengths of the complete prepared-context embeddings.
            gt_phoneme_embeddings: Ground-truth phoneme embeddings used for guidance.
            gt_phoneme_lens: Lengths of the ground-truth phoneme sequences.
            use_rollout_rejection: Whether to initialize per-item rejection tracking.
            rejection_weight: Optional loss weight to assign to rejected rollouts.

        Returns:
            An initialized streaming state.
        """
        bs = config.batch_size
        device = config.device
        rejected = None
        sample_weights = None

        if use_rollout_rejection:
            rejected = torch.zeros(bs, dtype=torch.bool, device=device)

        if rejection_weight is not None:
            sample_weights = torch.ones(bs, dtype=torch.float, device=device)

        obj = cls(
            config=config,
            past_key_values=past_kv,
            cache_seq_len=min_context_len,
            pred_codes=[],
            pred_codes_logits=[],
            pred_codes_lt=[],
            pred_codes_logits_lt=[],
            pred_phoneme_tokens=[],
            pred_phoneme_logits=[],
            context_audio_codes=context_audio_codes,
            context_audio_codes_lens=context_audio_codes_lens,
            context_lens=context_lens,
            full_context_embedding=full_context_embedding,
            full_context_lens=full_context_lens,
            context_position=torch.full((bs,), min_context_len, dtype=torch.long, device=device),
            text_tokens_seen=torch.zeros(bs, dtype=torch.long, device=device),
            phoneme_steps=torch.zeros(bs, dtype=torch.long, device=device),
            audio_steps=torch.zeros(bs, dtype=torch.long, device=device),
            phoneme_stream_ended=torch.zeros(bs, dtype=torch.bool, device=device),
            phoneme_eos_detected=torch.zeros(bs, dtype=torch.bool, device=device),
            finished=torch.zeros(bs, dtype=torch.bool, device=device),
            last_hidden=last_hidden,
            text_finished=torch.zeros(bs, dtype=torch.bool, device=device),
            last_audio_codes=None,
            last_audio_codes_lt=None,
            last_phoneme_tokens=None,
            audio_prediction_start_idx=torch.full((bs,), -1, dtype=torch.long, device=device),
            audio_prediction_end_idx=torch.full((bs,), -1, dtype=torch.long, device=device),
            phoneme_prediction_start_idx=torch.full((bs,), -1, dtype=torch.long, device=device),
            phoneme_prediction_end_idx=torch.full((bs,), -1, dtype=torch.long, device=device),
            gt_phoneme_embeddings=gt_phoneme_embeddings,
            gt_phoneme_lens=gt_phoneme_lens,
            rejected=rejected,
            sample_weights=sample_weights,
        )
        return obj

    def create_channel_status(self) -> StreamingChannelStatus:
        """Create per-channel status masks for the current rollout step.

        Returns:
            A container with status masks indicating which context, text,
            phoneme and audio channels are active for each batch item.
        """
        context = self.context_position < self.full_context_lens
        text = (~context) & (~self.text_finished)
        phoneme = (~context) & (self.text_tokens_seen >= self.streaming_phonemes_delay) & (~self.phoneme_stream_ended)
        audio = (~context) & (self.text_tokens_seen >= self.streaming_speech_delay) & (~self.finished)

        return StreamingChannelStatus(
            context=context,
            text=text,
            phoneme=phoneme,
            audio=audio,
        )

    def update_counters(self, status: StreamingChannelStatus) -> None:
        """Update per-item context, text, phoneme and audio rollout counters.

        Args:
            status: Status masks indicating which channels progress for each batch item.
        """
        self.context_position = self.context_position + status.context.long()
        self.text_tokens_seen = self.text_tokens_seen + (~status.context).long()
        self.phoneme_steps = self.phoneme_steps + status.phoneme.long()
        self.audio_steps = self.audio_steps + status.audio.long()

    def update_phoneme_start_idx(self, needs_phoneme: Tensor) -> None:
        """Record the first rollout step that predicts phonemes for each item.

        Args:
            needs_phoneme: Mask identifying items that require phoneme prediction.
        """
        first_phoneme_step = needs_phoneme & (self.phoneme_prediction_start_idx == -1)

        if not first_phoneme_step.any():
            return

        self.phoneme_prediction_start_idx = torch.where(
            condition=first_phoneme_step,
            input=torch.full_like(self.phoneme_prediction_start_idx, self.current_phoneme_step_idx),
            other=self.phoneme_prediction_start_idx,
        )

    def update_phoneme_end_status(
        self,
        tokens: PhonemeTokensStep,
        needs_phoneme: Tensor,
        eos_id: int,
    ) -> None:
        """Track phoneme EOS predictions and record phoneme end steps.

        Args:
            tokens: Predicted phoneme tokens for the current rollout step.
            needs_phoneme: Mask identifying items that require phoneme prediction.
            eos_id: Phoneme end-of-sequence token ID.
        """
        eos_detected = needs_phoneme & (tokens.tokens_t == eos_id).any(dim=1)
        self.phoneme_eos_detected = self.phoneme_eos_detected | eos_detected

        newly_ended = eos_detected & (self.phoneme_prediction_end_idx == -1)

        if newly_ended.any():
            self.phoneme_prediction_end_idx = torch.where(
                condition=newly_ended,
                input=torch.full_like(self.phoneme_prediction_end_idx, self.current_phoneme_step_idx),
                other=self.phoneme_prediction_end_idx,
            )

    def update_phoneme_step(
        self,
        tokens: PhonemeTokensStep,
        fs_factor: int,
        vocab_size: int,
    ) -> None:
        """Store phoneme predictions and logits from the current rollout step.

        Args:
            tokens: Predicted phoneme tokens and their logits.
            fs_factor: Number of phoneme frames stacked per rollout step.
            vocab_size: Size of the phoneme vocabulary.
        """
        self.last_phoneme_tokens = tokens.tokens_t
        self.pred_phoneme_tokens.append(tokens.tokens_t.unsqueeze(1))  # (B, 1, FS)
        logits = tokens.logits_t.view(self.batch_size, 1, fs_factor * vocab_size)
        self.pred_phoneme_logits.append(logits)

    def update_audio_start_idx(self, needs_audio: Tensor) -> None:
        """Record the first rollout frame that predicts audio for each item.

        Args:
            needs_audio: Mask identifying items that require audio prediction.
        """
        first_audio_step = needs_audio & (self.audio_prediction_start_idx == -1)

        if not first_audio_step.any():
            return

        self.audio_prediction_start_idx = torch.where(
            condition=first_audio_step,
            input=torch.full_like(self.audio_prediction_start_idx, self.current_frame_idx),
            other=self.audio_prediction_start_idx,
        )

    def update_last_audio_codes(
        self,
        audio_codes: AudioCodesStep,
        needs_audio: Tensor,
    ) -> None:
        """Update the most recently predicted audio codes for active items.

        Args:
            audio_codes: Backbone and optional local-transformer audio-code predictions.
            needs_audio: Mask identifying items that require audio prediction.
        """
        codes_t = audio_codes.codes_t.reshape(self.batch_size, -1)

        if self.last_audio_codes is None:
            self.last_audio_codes = codes_t
        else:
            update_mask = needs_audio.view(self.batch_size, 1).expand_as(codes_t)
            self.last_audio_codes = torch.where(update_mask, codes_t, self.last_audio_codes)

        if self.use_lt:
            codes_t_lt = audio_codes.codes_t_lt.reshape(self.batch_size, -1)

            if self.last_audio_codes_lt is None:
                self.last_audio_codes_lt = codes_t_lt
            else:
                update_mask = needs_audio.view(self.batch_size, 1).expand_as(codes_t_lt)
                self.last_audio_codes_lt = torch.where(update_mask, codes_t_lt, self.last_audio_codes_lt)

    def update_audio_end_status(
        self,
        audio_codes: AudioCodesStep,
        needs_audio: Tensor,
        eos_id: int,
        fs_factor: int,
    ) -> list[int]:
        """Track audio EOS predictions and record completed rollout end frames.

        Args:
            audio_codes: Sampled and greedy audio-code predictions for the current step.
            needs_audio: Mask identifying items that require audio prediction.
            eos_id: Audio end-of-sequence token ID.
            fs_factor: Number of audio frames stacked per rollout step.

        Returns:
            Batch indices of items whose audio rollout ended at the current step.
        """
        # Use backbone codes to preserve EOS prediction policy.
        codes_t = audio_codes.codes_t
        codes_t_argmax = audio_codes.codes_t_argmax

        eos_in_sampled = codes_t == eos_id
        eos_in_argmax = codes_t_argmax == eos_id
        eos_any_codebook = eos_in_sampled.any(dim=1) | eos_in_argmax.any(dim=1)
        eos_detected = eos_any_codebook.any(dim=1) & needs_audio
        self.finished = self.finished | eos_detected

        newly_ended = eos_detected & (self.audio_prediction_end_idx == -1)

        if not newly_ended.any():
            return []

        # Intentionally retain the full stacked frame containing EOS.
        current_frame_count = len(self.pred_codes) * fs_factor
        end_frame_idx = torch.full_like(self.audio_prediction_end_idx, current_frame_count + fs_factor)
        self.audio_prediction_end_idx = torch.where(newly_ended, end_frame_idx, self.audio_prediction_end_idx)

        return torch.nonzero(newly_ended, as_tuple=False).flatten().tolist()

    def add_audio_codes(
        self,
        audio_codes: AudioCodesStep,
    ) -> None:
        """Append audio-code predictions from the current rollout step.

        Args:
            audio_codes: Backbone and optional local-transformer audio-code predictions.
        """
        self.pred_codes.append(audio_codes.codes_t)
        self.pred_codes_logits.append(audio_codes.logits_t.unsqueeze(1))

        if self.use_lt:
            self.pred_codes_lt.append(audio_codes.codes_t_lt)
            self.pred_codes_logits_lt.append(audio_codes.logits_t_lt.unsqueeze(1))

    @property
    def current_phoneme_step_idx(self) -> int:
        """Get the current phoneme rollout step index."""
        return len(self.pred_phoneme_tokens)

    @property
    def current_frame_idx(self) -> int:
        """Get the current audio frame index."""
        return sum(p.size(-1) for p in self.pred_codes)

    @property
    def batch_size(self) -> int:
        """Get the rollout batch size."""
        return self.config.batch_size

    @property
    def temperature(self) -> float:
        """Get the audio sampling temperature."""
        return self.config.temperature

    @property
    def topk(self) -> int:
        """Get the top-k audio sampling limit."""
        return self.config.topk

    @property
    def cfg_scale(self) -> float:
        """Get the classifier-free guidance scale."""
        return self.config.cfg_scale

    @property
    def device(self) -> torch.device:
        """Get the rollout device."""
        return self.config.device

    @property
    def use_lt(self) -> bool:
        """Get whether local-transformer rollout is enabled."""
        return self.config.use_local_transformer

    @property
    def phoneme_sampling_method(self) -> str:
        """Get the phoneme sampling method."""
        return self.config.phoneme_sampling_method

    @property
    def streaming_speech_delay(self) -> int:
        """Get the text-token delay before audio generation begins."""
        return self.config.training_mode.streaming_speech_delay

    @property
    def streaming_phonemes_delay(self) -> int:
        """Get the text-token delay before phoneme generation begins."""
        return self.config.training_mode.streaming_phonemes_delay
