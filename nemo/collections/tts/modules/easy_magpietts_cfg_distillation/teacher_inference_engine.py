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

"""Teacher inference and streaming rollout helpers for EasyMagpieTTS CFG distillation."""

import time
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from nemo.collections.tts.models.easy_magpietts import EasyMagpieTTSModel
from nemo.collections.tts.models.easy_magpietts_inference import StreamingConfig, TrainingMode
from nemo.collections.tts.modules.easy_magpietts_cfg_distillation.distillation_types import TeacherOutput
from nemo.collections.tts.modules.easy_magpietts_cfg_distillation.rollout_rejection import (
    apply_rollout_lower_bound_rejection,
    apply_rollout_max_steps_rejection,
    apply_rollout_upper_bound_rejection,
)
from nemo.collections.tts.modules.easy_magpietts_cfg_distillation.teacher_streaming_state import (
    AudioCodesStep,
    PhonemeTokensStep,
    StreamingChannelStatus,
    TeacherStreamingState,
)
from nemo.collections.tts.modules.magpietts_modules import LocalTransformerType, clear_forbidden_logits
from nemo.utils import logging


def _lt_sample_autoregressive(
    model: EasyMagpieTTSModel,
    dec_output: Tensor,
    temperature: float = 0.7,
    topk: int = 80,
    cfg_scale: float = 1.0,
    use_kv_cache: bool = True,
    forbid_audio_eos: bool = False,
    sanitize_logits: bool = False,
) -> tuple[Tensor, Tensor]:
    """Sample stacked audio codes autoregressively with the local transformer.

    Args:
        model: EasyMagpie model providing the local transformer and audio-token
            projections.
        dec_output: Decoder output for the current audio frame, including
            conditional and unconditional CFG batches.
        temperature: Sampling temperature. Values less than or equal to zero use
            greedy decoding.
        topk: Number of highest-logit tokens retained for sampling per codebook.
        cfg_scale: Classifier-free guidance scale applied to conditional and
            unconditional logits.
        use_kv_cache: Whether to use the local transformer's key-value cache.
        forbid_audio_eos: Whether to prevent sampling audio EOS tokens.
        sanitize_logits: Whether to replace non-finite logits and clamp their
            range before sampling.

    Returns:
        A tuple containing the sampled codec codes with shape
        `(batch_size, num_audio_codebooks, frame_stacking_factor)` and the
        corresponding CFG logits for the conditional batch.
    """
    model.local_transformer.reset_cache(use_cache=use_kv_cache)
    dec_output = dec_output.unsqueeze(1)
    local_transformer_input = model.local_transformer_in_projection(dec_output)
    predicted_codes = []
    predicted_logits = []

    for codebook_num in range(model.num_audio_codebooks * model.frame_stacking_factor):
        size = (local_transformer_input.size(0), local_transformer_input.size(1))
        _mask = torch.ones(*size, device=local_transformer_input.device)
        local_transformer_output = model.local_transformer(local_transformer_input, _mask)["output"]

        lt_out_for_proj = model.local_transformer_audio_out_projection(local_transformer_output[:, -1, :])
        codebook_logits = model.local_transformer_out_projections[codebook_num](lt_out_for_proj)

        bs = codebook_logits.size(0) // 2
        conditional_logits = codebook_logits[:bs]
        unconditional_logits = codebook_logits[bs:]
        cfg_logits = cfg_scale * conditional_logits + (1.0 - cfg_scale) * unconditional_logits
        codebook_logits[:bs] = cfg_logits
        predicted_logits.append(codebook_logits.clone())

        if sanitize_logits:
            codebook_logits = torch.nan_to_num(codebook_logits, nan=0.0, posinf=100.0, neginf=-100.0)
            codebook_logits = codebook_logits.clamp(min=-100.0, max=100.0)

        codebook_logits = clear_forbidden_logits(
            logits=codebook_logits.unsqueeze(1),
            codebook_size=model.codebook_size,
            forbid_audio_eos=forbid_audio_eos,
        )
        codebook_logits = codebook_logits.squeeze(1)

        codebook_logits_topk = torch.topk(codebook_logits, topk, dim=-1)[0]
        indices_to_remove = codebook_logits < codebook_logits_topk[:, -1].unsqueeze(-1)
        codebook_logits_rescored = codebook_logits.clone()
        codebook_logits_rescored[indices_to_remove] = float("-inf")

        if temperature <= 0.0:
            codebook_preds = codebook_logits_rescored.argmax(dim=-1, keepdim=True)
        else:
            codebook_probs = torch.softmax(codebook_logits_rescored / temperature, dim=-1)
            codebook_preds = torch.multinomial(codebook_probs, 1)

        codebook_preds[bs:] = codebook_preds[:bs]
        predicted_codes.append(codebook_preds)

        next_local_transformer_input = model.audio_embeddings[codebook_num](codebook_preds.squeeze(-1)).unsqueeze(1)
        next_local_transformer_input = model.audio_in_projection(next_local_transformer_input)
        next_local_transformer_input = model.local_transformer_in_projection(next_local_transformer_input)
        local_transformer_input = torch.cat([local_transformer_input, next_local_transformer_input], dim=1)

    predicted_codes = torch.cat(predicted_codes, dim=1)[:bs]
    predicted_codes = predicted_codes.view(bs, model.num_audio_codebooks, model.frame_stacking_factor)
    predicted_logits = torch.cat(predicted_logits, dim=1)[:bs]

    return predicted_codes, predicted_logits


@dataclass
class _InitContext:
    """Store prepared context inputs used to initialize a teacher rollout."""

    embeds: Tensor
    lens: Tensor
    min_len: Tensor
    full_embeds: Tensor
    full_lens: Tensor
    dummy_embeds_uncond: Tensor
    cache_position: Tensor
    audio_codes: Tensor
    audio_codes_lens: Tensor


@dataclass
class _TeacherPredictedCodes:
    """Store audio-code predictions generated during a teacher rollout."""

    codes: Tensor
    logits: Tensor
    codes_len: Tensor
    codes_lt: Optional[Tensor]
    logits_lt: Optional[Tensor]


def _validate_predicted_codes(
    codes: Tensor,
    pred_codes_lens: Tensor,
    start_indices: Tensor,
    end_indices: Tensor,
    fs_factor: int,
) -> None:
    """Validate teacher rollout audio-code boundaries and lengths.

    Args:
        codes: Predicted audio codes for the rollout.
        pred_codes_lens: Finalized predicted-code lengths for each batch item.
        start_indices: Audio prediction start indices for each batch item.
        end_indices: Audio prediction end indices for each batch item.
        fs_factor: Number of audio frames stacked per decoder step.

    Raises:
        RuntimeError: If rollout boundaries or lengths are invalid, unaligned with
            stacked decoder steps or contain no audio predictions.
    """
    # If an item never entered audio generation, give it zero length by
    # using its end position as its start position. Clamping -1 to zero
    # would incorrectly assign predictions generated for other batch items to this item.
    invalid_bounds = (start_indices < 0) | (end_indices < start_indices) | (end_indices > codes.size(-1))

    if invalid_bounds.any():
        invalid_items = torch.nonzero(invalid_bounds, as_tuple=False).flatten()
        raise RuntimeError(
            "Invalid teacher rollout boundaries: "
            f"items={invalid_items.tolist()}, "
            f"starts={start_indices[invalid_items].tolist()}, "
            f"ends={end_indices[invalid_items].tolist()}, "
            f"available_frames={codes.size(-1)}."
        )

    # All boundaries must align with complete stacked decoder steps because
    # each stored logit position corresponds to one block of fs_factor unstacked codec frames.
    unaligned_bounds = (
        (start_indices % fs_factor != 0) | (end_indices % fs_factor != 0) | (pred_codes_lens % fs_factor != 0)
    )
    if unaligned_bounds.any():
        invalid_items = torch.nonzero(unaligned_bounds, as_tuple=False).flatten()
        raise RuntimeError(
            "Teacher rollout boundaries are not aligned with the audio "
            f"items={invalid_items.tolist()}, "
            f"starts={start_indices[invalid_items].tolist()}, "
            f"ends={end_indices[invalid_items].tolist()}, "
            f"lengths={pred_codes_lens[invalid_items].tolist()}."
        )

    if pred_codes_lens.max().item() == 0:
        raise RuntimeError("Teacher rollout produced no audio predictions.")


class TeacherInferenceEngine:
    """Teacher-model inference engine for generating distillation targets."""

    def __init__(
        self,
        model: EasyMagpieTTSModel,
    ) -> None:
        self.model = model

    @property
    def phoneme_stacking_factor(self) -> int:
        """Get the number of phoneme frames stacked per prediction step."""
        return self.model.phoneme_stacking_factor

    @property
    def audio_stacking_factor(self) -> int:
        """Get the number of audio frames stacked per prediction step."""
        return self.model.frame_stacking_factor

    @property
    def phoneme_vocab_size(self) -> int:
        """Get the phoneme vocabulary size."""
        return self.model.phoneme_vocab_size

    @property
    def num_audio_codebooks(self) -> int:
        """Get the number of audio codebooks."""
        return self.model.num_audio_codebooks

    @property
    def num_audio_tokens_per_codebook(self) -> int:
        """Get the number of tokens in each audio codebook."""
        return self.model.num_all_tokens_per_codebook

    @property
    def phoneme_bos_id(self) -> int:
        """Get the phoneme BOS token ID."""
        return self.model.phoneme_tokenizer.bos_token_id

    @property
    def phoneme_eos_id(self) -> int:
        """Get the phoneme EOS token ID."""
        return self.model.phoneme_tokenizer.eos_token_id

    @property
    def audio_bos_id(self) -> int:
        """Get the audio BOS token ID."""
        return self.model.audio_bos_id

    @property
    def audio_eos_id(self) -> int:
        """Get the audio EOS token ID."""
        return self.model.audio_eos_id

    @property
    def eos_id(self) -> int:
        """Get the model EOS token ID."""
        return self.model.eos_id

    @property
    def embedding_dim(self) -> int:
        """Get the model embedding dimension."""
        return self.model.cfg.embedding_dim

    @property
    def use_multiturn_dataset(self) -> bool:
        """Get whether the model uses the multiturn dataset format."""
        return self.model.cfg.get("use_multiturn_dataset", False)

    def _prepare_init_context(
        self,
        context_audio_codes: Tensor,
        context_audio_codes_lens: Tensor,
        context_text_tokens: Tensor,
        context_text_tokens_lens: Tensor,
        training_mode: TrainingMode,
    ) -> _InitContext:
        """Prepare context tensors for initializing a CFG teacher rollout.

        Args:
            context_audio_codes: Context audio codec tokens.
            context_audio_codes_lens: Lengths of the context audio-code sequences.
            context_text_tokens: Context text token IDs.
            context_text_tokens_lens: Lengths of the context text-token sequences.
            training_mode: Training-mode settings used to prepare context inputs.

        Returns:
            A container with prepared context tensors, cache positions and embeddings
            for the unconditional CFG branch.
        """
        bs = context_audio_codes.size(0)
        device = context_audio_codes.device

        embeds, lens, audio_codes, audio_codes_lens = self.model.prepare_context_tensors(
            context_text_tokens=context_text_tokens,
            context_text_tokens_lens=context_text_tokens_lens,
            context_audio_codes=context_audio_codes,
            context_audio_codes_lens=context_audio_codes_lens,
            training_mode=training_mode,
            dropout_conditional_input=False,
        )
        full_embeds, full_lens = embeds.clone(), lens.clone()
        min_len = lens.min().item()
        cache_position = torch.arange(min_len, device=device)

        dummy_embeds_uncond = self.model.embed_text_tokens(
            torch.full((1, 1), self.model.cfg_unk_token_id, device=device),
            text_lens=torch.ones(1, dtype=torch.long, device=device),
            disable_cas_embedding=self.model.disable_cas_for_context_text,
        )
        dummy_embeds_expanded = dummy_embeds_uncond.expand(bs, embeds.size(1), -1)
        embeds = torch.cat([embeds, dummy_embeds_expanded], dim=0)

        return _InitContext(
            embeds=embeds,
            lens=lens,
            min_len=min_len,
            full_embeds=full_embeds,
            full_lens=full_lens,
            dummy_embeds_uncond=dummy_embeds_uncond,
            cache_position=cache_position,
            audio_codes=audio_codes,
            audio_codes_lens=audio_codes_lens,
        )

    def _prepare_phoneme_embeds(
        self,
        gt_phoneme_tokens: Optional[Tensor],
        gt_phoneme_tokens_lens: Optional[Tensor],
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        """Prepare stacked embeddings for optional ground-truth phoneme tokens.

        Args:
            gt_phoneme_tokens: Optional ground-truth phoneme token IDs.
            gt_phoneme_tokens_lens: Optional lengths of the phoneme-token sequences.

        Returns:
            A tuple with ground-truth phoneme embeddings and their stacked sequence lengths.
        """
        embeds, lens = None, None

        if gt_phoneme_tokens is not None and gt_phoneme_tokens_lens is not None:
            gt_phoneme_expanded = gt_phoneme_tokens.unsqueeze(1)
            gt_phoneme_stacked, lens = self.model.stack_codes(
                codes=gt_phoneme_expanded,
                codes_lens=gt_phoneme_tokens_lens,
                bos_id=self.phoneme_bos_id,
                eos_id=self.phoneme_eos_id,
                stacking_factor=self.phoneme_stacking_factor,
                num_codebooks=1,
            )
            embeds = self.model.embed_phoneme_tokens(gt_phoneme_stacked)

            if self.use_multiturn_dataset:
                phoneme_pad_id = getattr(self.model.phoneme_tokenizer, "pad", -1)
                phoneme_mask = gt_phoneme_stacked[:, 0, :] != phoneme_pad_id
                embeds = embeds * phoneme_mask.unsqueeze(2)

        return embeds, lens

    def _streaming_init(
        self,
        context_audio_codes: Tensor,
        context_audio_codes_lens: Tensor,
        context_text_tokens: Tensor,
        context_text_tokens_lens: Tensor,
        cfg_scale: float = 1.0,
        use_lt: bool = False,
        temperature: float = 0.7,
        topk: int = 80,
        phoneme_input_type: str = "predicted",
        phoneme_sampling_method: str = "argmax",
        gt_phoneme_tokens: Optional[Tensor] = None,
        gt_phoneme_tokens_lens: Optional[Tensor] = None,
        use_rollout_rejection: bool = False,
        rejection_weight: Optional[float] = None,
    ) -> TeacherStreamingState:
        """Initialize the streaming state and transformer cache for a teacher rollout.

        Args:
            context_audio_codes: Context audio codec tokens.
            context_audio_codes_lens: Lengths of the context audio-code sequences.
            context_text_tokens: Context text token IDs.
            context_text_tokens_lens: Lengths of the context text-token sequences.
            cfg_scale: Classifier-free guidance scale for the rollout.
            use_lt: Whether to enable local-transformer predictions.
            temperature: Audio sampling temperature.
            topk: Number of highest-logit audio tokens retained for sampling.
            phoneme_input_type: Source of phoneme inputs during rollout.
            phoneme_sampling_method: Method used to sample predicted phoneme tokens.
            gt_phoneme_tokens: Optional ground-truth phoneme token IDs.
            gt_phoneme_tokens_lens: Optional lengths of the ground-truth phoneme sequences.
            use_rollout_rejection: Whether to track rejected rollouts.
            rejection_weight: Optional loss weight for rejected rollouts.

        Returns:
            An initialized streaming-state container.
        """
        with torch.no_grad():
            training_mode = self.model.mode_name_to_mode[self.model.default_inference_mode]

            init_context = self._prepare_init_context(
                context_audio_codes=context_audio_codes,
                context_audio_codes_lens=context_audio_codes_lens,
                context_text_tokens=context_text_tokens,
                context_text_tokens_lens=context_text_tokens_lens,
                training_mode=training_mode,
            )
            transformer_out = self.model.forward(
                inputs_embeds=init_context.embeds[:, : init_context.min_len, :],
                attention_mask=None,
                use_cache=True,
                past_key_values=None,
                cache_position=init_context.cache_position,
            )
            gt_phoneme_embeds, gt_phoneme_lens = self._prepare_phoneme_embeds(
                gt_phoneme_tokens, gt_phoneme_tokens_lens
            )
            config = StreamingConfig(
                batch_size=context_audio_codes.size(0),
                device=context_audio_codes.device,
                training_mode=training_mode,
                use_cfg=True,
                cfg_scale=cfg_scale,
                use_local_transformer=use_lt,
                temperature=temperature,
                topk=topk,
                phoneme_input_type=phoneme_input_type,
                phoneme_sampling_method=phoneme_sampling_method,
                dummy_context_embedding_unconditional=init_context.dummy_embeds_uncond,
            )
            state = TeacherStreamingState.create(
                config=config,
                last_hidden=transformer_out.last_hidden_state,
                past_kv=transformer_out.past_key_values,
                min_context_len=init_context.min_len,
                context_audio_codes=init_context.audio_codes,
                context_audio_codes_lens=init_context.audio_codes_lens,
                context_lens=init_context.lens,
                full_context_embedding=init_context.full_embeds,
                full_context_lens=init_context.full_lens,
                gt_phoneme_embeddings=gt_phoneme_embeds,
                gt_phoneme_lens=gt_phoneme_lens,
                use_rollout_rejection=use_rollout_rejection,
                rejection_weight=rejection_weight,
            )
            return state

    def _prepare_input_context_phase(
        self,
        next_input: Tensor,
        state: TeacherStreamingState,
        status: StreamingChannelStatus,
    ) -> Tensor:
        """Add active context embeddings to the next streaming-model input.

        Args:
            next_input: Input embeddings for the next rollout step.
            state: Current streaming rollout state.
            status: Status masks for the current rollout step.

        Returns:
            Next-step input embeddings with active context embeddings added.
        """
        if status.context.any():
            ctx_positions = state.context_position.clone()
            ctx_positions = ctx_positions.clamp(max=state.full_context_embedding.size(1) - 1)
            indexes = torch.arange(state.batch_size, device=state.device)
            ctx_emb = state.full_context_embedding[indexes, ctx_positions, :].unsqueeze(1)
            context_mask = status.context.view(state.batch_size, 1, 1).float()
            next_input = next_input + ctx_emb * context_mask
        return next_input

    def _prepare_input_text_phase(
        self,
        next_input: Tensor,
        state: TeacherStreamingState,
        status: StreamingChannelStatus,
        text_tokens: Optional[Tensor],
    ) -> tuple[Tensor, TeacherStreamingState]:
        """Add active text embeddings to the next streaming-model input.

        Args:
            next_input: Input embeddings for the next rollout step.
            state: Current streaming rollout state.
            status: Status masks for the current rollout step.
            text_tokens: Optional text token IDs for the current step.

        Returns:
            A tuple containing the updated input embeddings and streaming state.
        """
        if text_tokens is not None and status.text.any():
            text_tokens_2d = text_tokens.unsqueeze(1)
            text_lens = torch.ones(state.batch_size, dtype=torch.long, device=state.device)

            text_embedded = self.model.embed_text_tokens(
                text_tokens=text_tokens_2d,
                text_lens=text_lens,
                is_multiturn=self.use_multiturn_dataset,
            )
            if self.use_multiturn_dataset:
                text_embedded[text_tokens_2d == self.model.pad_id] = 0.0

            is_eos_token = (text_tokens == self.eos_id) & status.text
            text_add_mask = status.text.view(state.batch_size, 1, 1).float()
            next_input = next_input + text_embedded * text_add_mask
            state.text_finished = state.text_finished | is_eos_token

        elif text_tokens is None:
            state.text_finished = state.text_finished | ~status.context

        return next_input, state

    def _prepare_input_phoneme_phase(
        self,
        next_input: Tensor,
        state: TeacherStreamingState,
        status: StreamingChannelStatus,
    ) -> tuple[Tensor, TeacherStreamingState]:
        """Add active phoneme embeddings to the next streaming-model input.

        Args:
            next_input: Input embeddings for the next rollout step.
            state: Current streaming rollout state.
            status: Status masks for the current rollout step.

        Returns:
            A tuple containing the updated input embeddings and streaming state.
        """
        if self.model.phoneme_tokenizer is None:
            return next_input, state

        bs = state.batch_size
        device = state.device

        if status.phoneme.any():
            phoneme_emb = torch.zeros(bs, 1, self.embedding_dim, device=device)

            if state.config.phoneme_input_type == "gt" and state.gt_phoneme_embeddings is not None:
                within_gt_len = state.phoneme_steps < state.gt_phoneme_lens
                positions = state.phoneme_steps.clamp(max=state.gt_phoneme_embeddings.size(1) - 1)
                gt_emb = state.gt_phoneme_embeddings[torch.arange(bs, device=device), positions, :].unsqueeze(1)
                phoneme_mask = (status.phoneme & within_gt_len).view(bs, 1, 1).float()
                phoneme_emb = phoneme_emb + gt_emb * phoneme_mask
            else:
                first_phoneme_step = status.phoneme & (state.phoneme_steps == 0)
                has_last_phoneme = status.phoneme & (~first_phoneme_step) & (state.last_phoneme_tokens is not None)

                if first_phoneme_step.any():
                    size = (bs, self.phoneme_stacking_factor, 1)
                    phoneme_bos = torch.full(size, self.phoneme_bos_id, device=device).long()
                    phoneme_bos_emb = self.model.embed_phoneme_tokens(phoneme_bos)
                    first_mask = first_phoneme_step.view(bs, 1, 1).float()
                    phoneme_emb = phoneme_emb + phoneme_bos_emb * first_mask

                if has_last_phoneme.any() and state.last_phoneme_tokens is not None:
                    last_phoneme_emb = self.model.embed_phoneme_tokens(state.last_phoneme_tokens.unsqueeze(2))
                    last_mask = has_last_phoneme.view(bs, 1, 1).float()
                    phoneme_emb = phoneme_emb + last_phoneme_emb * last_mask

                state.phoneme_stream_ended = state.phoneme_stream_ended | state.phoneme_eos_detected

            next_input = next_input + phoneme_emb

        return next_input, state

    def _prepare_input_audio_phase(
        self,
        next_input: Tensor,
        state: TeacherStreamingState,
        status: StreamingChannelStatus,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Add active audio embeddings to the next streaming-model input.

        Args:
            next_input: Input embeddings for the next rollout step.
            state: Current streaming rollout state.
            status: Status masks for the current rollout step.

        Returns:
            A tuple containing the updated input embeddings and the optional audio
            embeddings used for the current step.
        """
        bs = state.batch_size
        device = state.device
        audio_emb = None

        if status.audio.any():
            audio_emb = torch.zeros(bs, 1, self.embedding_dim, device=device)
            # Use backbone codes to preserve rollout policy.
            last_audio_codes = state.last_audio_codes

            first_audio_step = status.audio & (state.audio_steps == 0)
            has_last_audio = status.audio & ~first_audio_step & (last_audio_codes is not None)

            if first_audio_step.any():
                size = (bs, self.num_audio_codebooks * self.audio_stacking_factor, 1)
                audio_bos = torch.full(size, self.audio_bos_id, device=device).long()
                audio_bos_emb = self.model.embed_audio_tokens(audio_bos)
                first_mask = first_audio_step.view(bs, 1, 1).float()
                audio_emb = audio_emb + audio_bos_emb * first_mask

            if has_last_audio.any() and last_audio_codes is not None:
                last_audio_emb = self.model.embed_audio_tokens(last_audio_codes.unsqueeze(2))
                last_mask = has_last_audio.view(bs, 1, 1).float()
                audio_emb = audio_emb + last_audio_emb * last_mask

            next_input = next_input + audio_emb

        return next_input, audio_emb

    def _prepare_input_cfg_phase(
        self,
        next_input: Tensor,
        state: TeacherStreamingState,
        status: StreamingChannelStatus,
        audio_emb: Optional[Tensor],
    ) -> Tensor:
        """Append unconditional CFG inputs to the next streaming-model input.

        Args:
            next_input: Conditional input embeddings for the next rollout step.
            state: Current streaming rollout state.
            status: Status masks for the current rollout step.
            audio_emb: Optional audio embeddings for the current step.

        Returns:
            Input embeddings containing conditional and unconditional CFG branches.
        """
        uncond_context = state.config.dummy_context_embedding_unconditional.expand(state.batch_size, 1, -1)
        uncond_zeros = torch.zeros_like(uncond_context)
        context_mask = status.context.view(state.batch_size, 1, 1).float()
        uncond_next_input = context_mask * uncond_context + (1 - context_mask) * uncond_zeros

        if status.audio.any():
            audio_mask = status.audio.view(state.batch_size, 1, 1).float()
            uncond_next_input = uncond_next_input * (1 - audio_mask) + audio_emb * audio_mask

        next_input = torch.cat([next_input, uncond_next_input], dim=0)

        return next_input

    def _prepare_streaming_input(
        self,
        state: TeacherStreamingState,
        text_tokens: Optional[Tensor],
    ) -> tuple[Tensor, StreamingChannelStatus]:
        """Prepare the combined CFG input for a streaming rollout step.

        Args:
            state: Current streaming rollout state.
            text_tokens: Optional text token IDs at the current rollout position.

        Returns:
            A tuple containing the combined next-step input embeddings and the
            masks that indicate active channels for each batch item.
        """
        bs = state.batch_size
        device = state.device

        status = state.create_channel_status()
        next_input = torch.zeros(bs, 1, self.embedding_dim, device=device)

        next_input = self._prepare_input_context_phase(next_input, state, status)
        next_input, state = self._prepare_input_text_phase(next_input, state, status, text_tokens)
        next_input, state = self._prepare_input_phoneme_phase(next_input, state, status)
        next_input, audio_emb = self._prepare_input_audio_phase(next_input, state, status)
        next_input = self._prepare_input_cfg_phase(next_input, state, status, audio_emb)

        return next_input, status

    def _predict_phoneme_tokens(
        self,
        state: TeacherStreamingState,
    ) -> PhonemeTokensStep:
        """Predict phoneme tokens for the current streaming rollout step.

        Args:
            state: Current streaming rollout state.

        Returns:
            A container with the sampled phoneme tokens and their prediction logits.
        """
        bs = state.batch_size
        last_hidden = state.last_hidden
        temp = state.temperature
        topk = state.topk

        logits_t = self.model.phoneme_final_proj(last_hidden[:, -1, :])
        logits_t = logits_t[:bs]

        if state.phoneme_sampling_method == "argmax":
            tokens_t = self.model.sample_codes_from_logits_phoneme(logits_t, temperature=0.0)
        else:
            tokens_t = self.model.sample_codes_from_logits_phoneme(logits_t, temperature=temp, topk=topk)

        logits_t = logits_t.view(bs, self.phoneme_stacking_factor, self.phoneme_vocab_size)

        return PhonemeTokensStep(tokens_t=tokens_t, logits_t=logits_t)

    def _predict_audio_codes(
        self,
        state: TeacherStreamingState,
        use_lt: bool,
    ) -> AudioCodesStep:
        """Predict backbone and optional local-transformer audio codes for one rollout step.

        Args:
            state: Current streaming rollout state.
            use_lt: Whether to generate local-transformer audio-code predictions.

        Returns:
            A container with sampled and greedy backbone codes, backbone logits and
            optional local-transformer predictions.
        """
        bs = state.batch_size
        last_hidden = state.last_hidden
        temp = state.temperature
        topk = state.topk
        cfg_scale = state.cfg_scale

        last_hidden_audio = self.model.audio_out_projection(last_hidden[:, -1, :])
        code_logits_t = self.model.final_proj(last_hidden_audio)

        cond_logits = code_logits_t[:bs]
        uncond_logits = code_logits_t[bs:]
        logits_t = cfg_scale * cond_logits + (1.0 - cfg_scale) * uncond_logits

        codes_t = self.model.sample_codes_from_logits(logits_t, temperature=temp, topk=topk)
        codes_t_argmax = codes_t if temp <= 0.0 else self.model.sample_codes_from_logits(logits_t, temperature=0.01)
        codes_t_lt, codes_t_argmax_lt, logits_t_lt = None, None, None

        if use_lt:
            codes_t_lt, logits_t_lt = _lt_sample_autoregressive(
                model=self.model,
                dec_output=last_hidden[:, -1, :],
                temperature=temp,
                topk=topk,
                cfg_scale=cfg_scale,
                use_kv_cache=True,
                sanitize_logits=True,
            )
            codes_t_lt = codes_t_lt.reshape(bs, -1)
            # As in the pre-training stage, we do not use greedy sampling for LT.
            codes_t_argmax_lt = codes_t_lt

        return AudioCodesStep(
            codes_t=codes_t,
            codes_t_argmax=codes_t_argmax,
            logits_t=logits_t,
            codes_t_lt=codes_t_lt,
            codes_t_argmax_lt=codes_t_argmax_lt,
            logits_t_lt=logits_t_lt,
        )

    def _process_predictions(
        self,
        state: TeacherStreamingState,
        status: StreamingChannelStatus,
    ) -> TeacherStreamingState:
        """Update rollout state with predictions from the current streaming step.

        The method updates per-channel counters, generates active phoneme and audio
        predictions, records them in the state and tracks detected sequence endings.

        Args:
            state: Current streaming rollout state.
            status: Status masks for the current rollout step.

        Returns:
            Updated streaming rollout state.
        """
        state.update_counters(status)

        if status.phoneme.any() and self.model.phoneme_tokenizer is not None:
            state.update_phoneme_start_idx(status.phoneme)
            phoneme_tokens = self._predict_phoneme_tokens(state)

            # Retain predicted phoneme tokens/logits for rollout debugging.
            # They are not used by the regular ground-truth phoneme training loss.
            state.update_phoneme_step(
                tokens=phoneme_tokens,
                fs_factor=self.phoneme_stacking_factor,
                vocab_size=self.phoneme_vocab_size,
            )
            state.update_phoneme_end_status(
                tokens=phoneme_tokens,
                needs_phoneme=status.phoneme,
                eos_id=self.phoneme_eos_id,
            )

        if status.audio.any():
            state.update_audio_start_idx(status.audio)
            audio_codes = self._predict_audio_codes(state, use_lt=state.use_lt)

            audio_codes.unstack(
                batch_size=state.batch_size,
                num_codebooks=self.num_audio_codebooks,
                fs_factor=self.audio_stacking_factor,
            )
            state.update_last_audio_codes(
                audio_codes=audio_codes,
                needs_audio=status.audio,
            )
            newly_ended_items = state.update_audio_end_status(
                audio_codes=audio_codes,
                needs_audio=status.audio,
                eos_id=self.audio_eos_id,
                fs_factor=self.audio_stacking_factor,
            )
            for idx in newly_ended_items:
                logging.info(f"End detected for item {idx} at streaming generation step {len(state.pred_codes) + 1}.")

            state.add_audio_codes(audio_codes)

        return state

    def _streaming_step(
        self,
        state: TeacherStreamingState,
        text_tokens: Optional[Tensor] = None,
    ) -> TeacherStreamingState:
        """Run one teacher-model streaming inference step.

        The method prepares the next CFG input, updates the transformer cache and
        hidden state and processes any phoneme or audio predictions.

        Args:
            state: Current streaming rollout state.
            text_tokens: Optional text token IDs at the current rollout position.

        Returns:
            Updated streaming rollout state.
        """
        if state.finished.all():
            return state

        with torch.no_grad():
            device = state.config.device

            next_input, status = self._prepare_streaming_input(state, text_tokens)
            cache_position = torch.tensor([state.cache_seq_len], device=device)

            transformer_out = self.model.forward(
                inputs_embeds=next_input,
                attention_mask=None,
                use_cache=True,
                past_key_values=state.past_key_values,
                cache_position=cache_position,
            )
            state.last_hidden = transformer_out.last_hidden_state
            state.past_key_values = transformer_out.past_key_values
            state.cache_seq_len += 1
            state = self._process_predictions(state, status)

        return state

    def _finalize_predicted_codes(
        self,
        state: TeacherStreamingState,
    ) -> _TeacherPredictedCodes:
        """Finalize teacher-rollout predictions into padded batch tensors.

        The method extracts each item's predicted audio span, pads codes and logits to
        the batch maximum length and includes local-transformer predictions when enabled.

        Args:
            state: Streaming state containing buffered rollout predictions and their
                start and end indices.

        Returns:
            A container with finalized teacher audio codes, logits, sequence lengths
            and optional local-transformer predictions.

        Raises:
            RuntimeError: If the rollout contains no audio predictions or invalid prediction boundaries.
        """
        bs = state.batch_size
        device = state.device
        use_lt = state.use_lt
        fs_factor = self.audio_stacking_factor

        if len(state.pred_codes) == 0:
            raise RuntimeError("Teacher rollout produced no audio predictions.")

        pred_codes_lt, pred_logits_lt = None, None

        with torch.no_grad():
            codes = torch.cat(state.pred_codes, dim=-1)
            logits = torch.cat(state.pred_codes_logits, dim=1)

            if use_lt:
                codes_lt = torch.cat(state.pred_codes_lt, dim=-1)
                logits_lt = torch.cat(state.pred_codes_logits_lt, dim=1)

            raw_start_indices = state.audio_prediction_start_idx
            end_indices = torch.where(
                condition=state.audio_prediction_end_idx >= 0,
                input=state.audio_prediction_end_idx,
                other=torch.full_like(state.audio_prediction_end_idx, codes.size(-1)),
            )

            never_started = raw_start_indices < 0
            start_indices = torch.where(
                condition=never_started,
                input=end_indices,
                other=raw_start_indices,
            )
            pred_codes_lens = end_indices - start_indices

            _validate_predicted_codes(codes, pred_codes_lens, start_indices, end_indices, fs_factor)

            max_len = pred_codes_lens.max().item()
            max_stacked_len = max_len // self.audio_stacking_factor
            codes_size = (bs, self.num_audio_codebooks, max_len)
            logits_dim = self.num_audio_codebooks * self.num_audio_tokens_per_codebook * self.audio_stacking_factor
            logits_size = (bs, max_stacked_len, logits_dim)
            pred_codes = torch.zeros(*codes_size, dtype=codes.dtype, device=device)
            pred_logits = torch.zeros(*logits_size, dtype=logits.dtype, device=device)

            if use_lt:
                pred_codes_lt = torch.zeros(*codes_size, dtype=codes.dtype, device=device)
                pred_logits_lt = torch.zeros(*logits_size, dtype=logits.dtype, device=device)

            for i in range(bs):
                start, end = start_indices[i].item(), end_indices[i].item()
                start_, end_ = start // self.audio_stacking_factor, end // self.audio_stacking_factor
                length = end - start

                if length == 0:
                    continue

                pred_codes[i, :, :length] = codes[i, :, start:end]
                pred_logits[i, : end_ - start_, :] = logits[i, start_:end_, :]

                if use_lt:
                    pred_codes_lt[i, :, :length] = codes_lt[i, :, start:end]
                    pred_logits_lt[i, : end_ - start_, :] = logits_lt[i, start_:end_, :]

        return _TeacherPredictedCodes(
            codes=pred_codes,
            logits=pred_logits,
            codes_len=pred_codes_lens,
            codes_lt=pred_codes_lt,
            logits_lt=pred_logits_lt,
        )

    def infer_batch(
        self,
        batch: dict[str, Tensor],
        use_lt: bool,
        max_decoder_steps: int = 500,
        temperature: float = 0.7,
        topk: int = 80,
        cfg_scale: float = 1.0,
        phoneme_input_type: str = "pred",
        phoneme_sampling_method: str = "argmax",
        lower_rejection_threshold: Optional[float] = None,
        upper_rejection_threshold: Optional[float] = None,
        rejection_weight: Optional[float] = None,
    ) -> TeacherOutput:
        """Generate finalized teacher-rollout distillation targets for a batch.

        Performs streaming autoregressive generation from batch context and text
        inputs, applies optional rollout-length rejection, and finalizes generated
        audio-code targets and logits.

        Args:
            batch: Input batch.
            use_lt: Whether to generate local-transformer predictions.
            max_decoder_steps: Maximum number of streaming audio decoding steps.
            temperature: Audio sampling temperature.
            topk: Number of highest-logit audio tokens retained for sampling.
            cfg_scale: Classifier-free guidance scale.
            phoneme_input_type: Source used for phoneme inputs during rollout.
            phoneme_sampling_method: Method used to sample predicted phoneme tokens.
            lower_rejection_threshold: Fraction of ground-truth audio-code length used
                for final inclusive lower-bound rejection, or `None` to disable it.
            upper_rejection_threshold: Fraction of ground-truth audio-code length used
                for upper-bound rejection during rollout, or `None` to disable it.
            rejection_weight: Optional loss weight assigned to rejected rollouts.

        Returns:
            Teacher-rollout output containing finalized backbone codes, logits, and
            lengths; optional local-transformer predictions; rejection metadata; and
            CPU-observed generation-loop and total rollout wall times in seconds.

        Raises:
            ValueError: If local-transformer predictions are requested for a non-AR
                local transformer, or `batch` lacks `"context_audio_codes"`.
            KeyError: If `batch` lacks another required input field, or lacks
                `"audio_codes_lens"` while length-based rejection is enabled.
            RuntimeError: If finalized backbone and local-transformer outputs have
                inconsistent lengths.
        """
        if use_lt and self.model.local_transformer_type != LocalTransformerType.AR:
            raise ValueError(
                f"Only `LocalTransformerType.AR` is supported for local-transformer distillation, "
                f"but got `{self.model.local_transformer_type}`."
            )
        if "context_audio_codes" not in batch:
            raise ValueError("Missing required batch field `context_audio_codes` for teacher rollout inference.")

        text = batch["text"]
        bs = text.size(0)
        device = text.device
        text_lens = batch["text_lens"]
        use_rollout_rejection = lower_rejection_threshold is not None or upper_rejection_threshold is not None

        with torch.no_grad():
            rollout_start_time = time.perf_counter()
            state = self._streaming_init(
                context_audio_codes=batch["context_audio_codes"],
                context_audio_codes_lens=batch["context_audio_codes_lens"],
                context_text_tokens=batch["context_text_tokens"],
                context_text_tokens_lens=batch["context_text_tokens_lens"],
                cfg_scale=cfg_scale,
                use_lt=use_lt,
                temperature=temperature,
                topk=topk,
                phoneme_input_type=phoneme_input_type,
                phoneme_sampling_method=phoneme_sampling_method,
                gt_phoneme_tokens=batch.get("phoneme_tokens"),
                gt_phoneme_tokens_lens=batch.get("phoneme_tokens_lens"),
                use_rollout_rejection=use_rollout_rejection,
                rejection_weight=rejection_weight,
            )
            logging.info("Generation started")
            generation_start_time = time.perf_counter()

            while not state.finished.all() and len(state.pred_codes) < max_decoder_steps:
                positions = state.text_tokens_seen.clamp(max=text.size(1) - 1)
                current_tokens = text[torch.arange(bs, device=device), positions]

                current_tokens = torch.where(
                    condition=state.text_tokens_seen >= text_lens,
                    input=torch.full_like(current_tokens, self.eos_id),
                    other=current_tokens,
                )
                state = self._streaming_step(state, text_tokens=current_tokens)

                if upper_rejection_threshold is not None:
                    state = apply_rollout_upper_bound_rejection(
                        state, batch, upper_rejection_threshold, rejection_weight
                    )

            rollout_generation_time = time.perf_counter() - generation_start_time

            if use_rollout_rejection and not state.finished.all():
                state = apply_rollout_max_steps_rejection(
                    state=state,
                    batch=batch,
                    rejection_weight=rejection_weight,
                )

            teacher_codes = self._finalize_predicted_codes(state)

            if teacher_codes.codes_lt is not None and teacher_codes.codes.size(-1) != teacher_codes.codes_lt.size(-1):
                raise RuntimeError(
                    "Mismatch between backbone and local-transformer generated code lengths: "
                    f"`teacher_codes.codes` has length {teacher_codes.codes.size(-1)}, "
                    f"while `teacher_codes.codes_lt` has length {teacher_codes.codes_lt.size(-1)}."
                )
            teacher_output = TeacherOutput(
                codes=teacher_codes.codes,
                logits=teacher_codes.logits,
                lens=teacher_codes.codes_len,
                sample_weights=state.sample_weights,
                codes_lt=teacher_codes.codes_lt,
                logits_lt=teacher_codes.logits_lt,
                rejected=state.rejected.clone() if use_rollout_rejection else None,
            )
            if lower_rejection_threshold is not None:
                teacher_output = apply_rollout_lower_bound_rejection(
                    teacher_output, batch, lower_rejection_threshold, rejection_weight
                )

        if use_rollout_rejection:
            rejected_count = int(teacher_output.rejected.sum().item())
            logging.info(f"Total rejected samples in batch: {rejected_count}/{bs}.")

        teacher_output.timing_metrics = {
            "rollout_generation_time": rollout_generation_time,
            "rollout_total_time": time.perf_counter() - rollout_start_time,
        }
        return teacher_output
