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
"""Inference-only EasyMagpieTTS model for vLLM-Omni.

The Nemotron-H backbone consumes additive text, phoneme, and previous-audio
embeddings. A local transformer predicts the stacked audio codebooks for each
frame. Request metadata supplies target text, optional context text and task
mode, audio sampling parameters, and either precomputed or raw reference audio
conditioning.

``prompt_token_ids`` must have the same length as the assembled speaker
conditioning plus any causal target-text rows moved into prefill. Raw reference
and multiturn user waveforms use ``audio_input_token_id`` placeholders plus
matching ``multi_modal_data["audio"]`` items. Streaming text requests provide
one or more ``text_token`` ids per decode chunk and terminate the stream with ``text_eos_id``.
"""
from __future__ import annotations

import bisect
import os
from collections.abc import Callable, Iterable
from typing import Any, Optional

import torch
from easymagpie_vllm_omni.backbone_patches import (
    patch_mamba_streaming_decode,
    patch_moe_routed_scale,
    patch_shared_expert_activation,
)
from easymagpie_vllm_omni.codec.encoder import EasyMagpieCodecEncoder
from easymagpie_vllm_omni.codec.encoder_config import EasyMagpieCodecEncoderConfig
from easymagpie_vllm_omni.config import EasyMagpieOmniArch
from easymagpie_vllm_omni.local_transformer import EasyMagpieCodePredictor
from easymagpie_vllm_omni.multimodal import (
    EasyMagpieDummyInputsBuilder,
    EasyMagpieMultiModalProcessor,
    EasyMagpieProcessingInfo,
)
from easymagpie_vllm_omni.tokenizer import EasyMagpieTextTokenizer
from torch import nn
from vllm.compilation.backends import set_model_tag
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    SupportsMambaPrefixCaching,
    SupportsMultiModal,
)
from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM, NemotronHModel
from vllm.model_executor.models.utils import _merge_multimodal_embeddings, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

# Placeholder token id stuffed into the per-step ``input_ids`` returned by
# ``preprocess`` — the model never consumes ``input_ids`` (decode behaviour is
# driven by the per-token buffers), and ``compute_logits`` returns
# argmax-at-0 dummy logits, so this only needs to be a valid id.
_DUMMY_TOKEN_ID = 0
_AUDIO_OUTPUT_USER = 0
_AUDIO_OUTPUT_REFERENCE = 1


def _merge_streaming_text_chunk(
    text_tokens: list[int], incoming: list[int], text_token_start: Any
) -> tuple[list[int], bool]:
    """Merge one absolute-position text chunk into a request's token buffer.

    vLLM async scheduling can expose the next segment's metadata one decode
    step early. Absolute positions make that lookahead harmless: an already
    merged chunk is a no-op, while the next contiguous chunk is appended once.
    Gaps and conflicting overlaps indicate a malformed streaming request and
    are rejected instead of silently dropping text conditioning.
    """
    if not incoming:
        return text_tokens, False
    if text_token_start is None:
        raise ValueError("Streaming text_token updates require text_token_start")

    start = int(text_token_start)
    if start < 0 or start > len(text_tokens):
        raise ValueError(f"Invalid text_token_start={start} for accumulated text length {len(text_tokens)}")

    chunk = [int(token) for token in incoming]
    overlap = min(len(chunk), len(text_tokens) - start)
    if text_tokens[start : start + overlap] != chunk[:overlap]:
        raise ValueError(f"Conflicting streaming text chunk at absolute position {start}")
    if overlap == len(chunk):
        return text_tokens, False

    return text_tokens + chunk[overlap:], True


# Context text used when the request omits ``context_text``
_DEFAULT_CONTEXT_TEXT = "[EN]"


# This class is not wrapped in ``@support_torch_compile``: the Nemotron-H
# backbone and :class:`EasyMagpieCodePredictor` each manage their own
# ``torch.compile`` / CUDA-graph capture internally, so the outer ``forward``
# runs eagerly and dispatches into the two self-compiled subgraphs.
@MULTIMODAL_REGISTRY.register_processor(
    EasyMagpieMultiModalProcessor,
    info=EasyMagpieProcessingInfo,
    dummy_inputs=EasyMagpieDummyInputsBuilder,
)
class EasyMagpieTTSForConditionalGeneration(
    nn.Module,
    HasInnerState,
    IsHybrid,
    SupportsMambaPrefixCaching,
    SupportsMultiModal,
):
    """EasyMagpie LM stage for vLLM-Omni.

    See the module docstring for the per-step flow and the per-request I/O
    contract. The class exposes the omni hooks (``has_preprocess`` /
    ``has_postprocess`` / ``have_multimodal_outputs``) consumed by the
    ``OmniGPUModelRunner``.
    """

    # Hybrid-Mamba bookkeeping (delegated to vLLM's NemotronH causal-LM). vLLM
    # expects these as class attributes.
    get_mamba_state_dtype_from_config = NemotronHForCausalLM.get_mamba_state_dtype_from_config
    get_mamba_state_shape_from_config = NemotronHForCausalLM.get_mamba_state_shape_from_config
    get_mamba_state_copy_func = NemotronHForCausalLM.get_mamba_state_copy_func

    # Omni runner hooks.
    has_preprocess: bool = True
    has_postprocess: bool = True
    # Keep token-id views available to the custom Omni preprocess hook.
    requires_raw_input_tokens: bool = True
    have_multimodal_outputs: bool = True

    # Stage 1 (Code2Wav) consumes only the sampled codes (multimodal outputs),
    # never the backbone hidden states. Opt out of attaching ``hidden`` to the
    # inter-stage pooler payload so the runner skips the per-step D2H copy +
    # transport of hidden states (the default is True and would pass them).
    omni_pooler_payload_include_hidden: bool = False

    # Keep small per-step tensors GPU-resident across steps (no D2H/H2D).
    gpu_resident_buffer_keys: set[str] = {"last_audio_codes", "last_phoneme_token"}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        hf_config = vllm_config.model_config.hf_config
        self.hf_config = hf_config
        self.vllm_config = vllm_config
        self.arch = EasyMagpieOmniArch.from_hf_config(hf_config)
        self.model_path = vllm_config.model_config.model

        arch = self.arch
        self.hidden_dim = arch.hidden_dim
        self.embedding_dim = arch.embedding_dim
        self.num_codebooks = arch.num_stacked_codebooks

        # How to surface sampled acoustic codes from ``make_omni_output``, driven
        # by the stage's pipeline-config ``engine_output_type``:
        #
        # * "audio" (single-stage EasyMagpie LM that is also the *final*, client-facing
        #   stage): emit under the ``model_outputs`` key. vLLM-Omni's output
        #   processor remaps ``model_outputs`` -> the drainable ``audio`` modality
        #   key, so DELTA streaming DRAINS the codes every step (the client gets
        #   per-step deltas) instead of re-accumulating and re-sending the whole
        #   cumulative code tensor on every step.
        # * otherwise (two-stage EasyMagpie LM; ``engine_output_type="latent"``): emit
        #   the inter-stage keys (``audio_codes`` + nested ``codes.audio``) that
        #   the Code2Wav connector / async-chunk streamer consume.
        engine_output_type = getattr(vllm_config.model_config, "engine_output_type", None)
        self._single_stage_audio = str(engine_output_type or "").lower() == "audio"

        # ── Backbone (reused vLLM Nemotron-H LM; fed via inputs_embeds) ──
        self.backbone = NemotronHModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "backbone"),
        )
        # vLLM 0.24's NemotronHMLP hard-codes ReLU² in shared_experts,
        # ignoring the checkpoint's mlp_hidden_act. Restore the configured
        # activation (no-op when the backbone has no MoE layers).
        patch_shared_expert_activation(self.backbone)
        # vLLM's FusedMoE defers routed_scaling_factor to the decoder layer in
        # FP16, but NemotronH's decoder layer never compensates, so the MoE
        # output is under-scaled by routed_scaling_factor. Restore it (no-op in
        # fp32/bf16 and when there are no MoE layers).
        patch_moe_routed_scale(self.backbone)
        # The streaming-input path keeps extending the prompt, so vLLM's Mamba2
        # metadata builder would classify every single-token decode step as a
        # prefill — breaking the FULL decode cudagraph (stale
        # state_indices_tensor_d). Force single-token extends to classify as
        # decodes so FULL/FULL_DECODE_ONLY cudagraphs read the right Mamba slot.
        patch_mamba_streaming_decode()

        # ── Local transformer (its own compile group / CUDA graph) ──────
        with set_model_tag("local_transformer"):
            self.code_predictor = EasyMagpieCodePredictor(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "code_predictor"),
            )

        # Conversion optionally packages the codec and reference-speaker encoders
        # as one raw-audio tower. Multi-turn capability is checked separately.
        self.codec_encoder: EasyMagpieCodecEncoder | None = None
        if arch.codec_encoder_bundled:
            arch.ensure_reference_audio_available()
            self._validate_audio_encoder_artifacts()
            encoder_config_path = os.path.join(self.model_path, "codec_encoder.json")
            encoder_config = EasyMagpieCodecEncoderConfig.from_json_file(encoder_config_path)
            with self._mark_tower_model(vllm_config, "audio"):
                self.codec_encoder = EasyMagpieCodecEncoder(encoder_config).float()

        # ── Text + phoneme embedding heads ──────────────────────────────
        # Precomputed per-subword text embedding (one row per subword id), baked
        # at conversion time and fed additively on every decode step.
        text_vocab_size = int(getattr(hf_config, "text_vocab_size", getattr(hf_config, "vocab_size", 0)))
        self.text_embedding = nn.Embedding(text_vocab_size, self.embedding_dim)

        # Text-stream EOS id. Legacy checkpoints place it at vocab_size - 2;
        # multiturn checkpoints append an interruption token after the existing
        # specials, so their converter pins the unchanged EOS id explicitly.
        self.text_eos_id = arch.resolved_text_eos_id(text_vocab_size)

        # Task ("service token") embedding — a single learned per-mode row
        # prepended to the prefill context for multi-mode checkpoints. Built only
        # when the checkpoint carries one; otherwise ``None``.
        self.num_task_embeddings = int(arch.num_task_embeddings)
        if self.num_task_embeddings > 0:
            self.task_embedding = nn.Embedding(self.num_task_embeddings, self.embedding_dim)
        else:
            self.task_embedding = None

        # Context-text tokenizer, loaded lazily from the model directory. It
        # turns the per-request ``context_text`` string (e.g. ``"[EN]"``) into the
        # subword ids that the baked ``text_embedding`` table consumes — so the
        # caller passes plain text, never pre-tokenized ids.
        self._text_tokenizer: Any = None

        # ── Streaming delays (text leads phoneme by ``phonemes_delay`` and audio
        # by ``speech_delay`` decode steps; 0/0 == lock-step). ──
        self.phonemes_delay = int(getattr(arch, "streaming_phonemes_delay", 0) or 0)
        self.speech_delay = int(getattr(arch, "streaming_speech_delay", 0) or 0)

        # Phoneme channel (optional — only built when the checkpoint has one).
        self.has_phoneme = arch.phoneme_vocab_size > 0 and arch.phoneme_stacking_factor > 0
        if self.has_phoneme:
            self.phoneme_embeddings = nn.ModuleList(
                [
                    nn.Embedding(arch.phoneme_vocab_size, self.embedding_dim)
                    for _ in range(arch.phoneme_stacking_factor)
                ]
            )
            self.phoneme_final_proj = nn.Linear(
                self.hidden_dim, arch.phoneme_vocab_size * arch.phoneme_stacking_factor
            )
            # Phoneme special-token ids + confidence→UNK replacement threshold.
            self.phoneme_bos_id = int(arch.resolved_phoneme_bos_id)
            self.phoneme_eos_id = int(arch.resolved_phoneme_eos_id)
            self.phoneme_unk_id = int(arch.resolved_phoneme_unk_id)
            self.phoneme_confidence_unk_threshold = float(arch.phoneme_confidence_unk_threshold)

        # ── Persistent, address-stable scratch buffers ─────────────────
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        dtype = vllm_config.model_config.dtype
        # Combined per-token input embedding fed into the backbone.
        self._combined_embeddings = torch.zeros(max_num_tokens, self.embedding_dim, dtype=dtype)
        # Per-token decode inputs assembled by ``preprocess``.
        self._dec_text_tokens = torch.zeros(max_num_tokens, dtype=torch.long)
        self._dec_text_mask = torch.zeros(max_num_tokens, dtype=torch.long)
        self._dec_audio_codes = torch.zeros(max_num_tokens, self.num_codebooks, dtype=torch.long)
        self._dec_audio_valid = torch.zeros(max_num_tokens, dtype=torch.long)
        if self.has_phoneme:
            self._dec_phoneme_tokens = torch.zeros(max_num_tokens, arch.phoneme_stacking_factor, dtype=torch.long)
            self._dec_phoneme_valid = torch.zeros(max_num_tokens, dtype=torch.long)

        self._out_codes = torch.zeros(max_num_tokens, self.num_codebooks, dtype=torch.long)

        # ── Audio-EOS → engine stop ─────────────────────────────────────
        # The model signals end-of-speech inside the audio codebooks.
        # To make vLLM terminate the request at the EOS frame,
        # we flags decode positions with ``audio_eos_id`` emit designated ``stop_token_id``
        # in ``compute_logits``.
        # Callers must pass ``SamplingParams(stop_token_ids=[stop_id])`` with
        # ``stop_id = audio_eos_stop_token_id(hf_config)``.
        self.audio_eos_id = int(arch.audio_eos_id)
        self._stop_token_id = self.audio_eos_stop_token_id(hf_config)
        # flags frames in which ``_out_codes`` contain ``audio_eos_id``
        self._token_stop = torch.zeros(max_num_tokens, dtype=torch.bool)
        # slice of ``token_stop`` based on ``logit_idx`` that can be used in
        # ``compute_logits``
        self._sample_stop = torch.zeros(max_num_tokens, dtype=torch.bool)

        # ── Assembled prefill context embeddings (the only context cache) ──
        # ``preprocess`` runs on the host, once per request, serially on the
        # runner's critical path, so per-request speaker-tensor transfer + the
        # tokenize/embed/cat dominate TTFT under concurrency. Cache the *whole*
        # assembled context ``[task | speaker | context_text]`` per
        # ``(task_mode_id, speaker_id, context_text, device)`` (see
        # :meth:`_build_prefill_embeds`): for a known speaker it is identical on
        # every request, so the cache subsumes a separate speaker-embedding table
        # — the speaker ``.pt`` is read from disk only on the (first) cache miss
        # for that combo (see :meth:`_load_known_speaker_embedding`), then never
        # again. Raw-reference embeddings are owned by vLLM's multimodal cache.
        self._prefill_cache: dict[tuple, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def audio_eos_stop_token_id(hf_config: Any) -> int:
        """Backbone token id this model emits when audio EOS is reached.

        Audio end-of-speech lives in the codebooks, not the backbone token
        stream, so the dummy backbone vocab is repurposed as a 2-way stop
        signal: index ``0`` == "continue", the last index == "stop". Callers
        must pass ``SamplingParams(stop_token_ids=[this])``
        """
        return max(1, int(getattr(hf_config, "vocab_size", 2)) - 1)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compatibility shim — unused at runtime (everything goes via inputs_embeds)."""
        return self.text_embedding(input_ids)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
        *,
        is_multimodal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embeddings = self.get_input_embeddings(input_ids)
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return embeddings
        if is_multimodal is None:
            raise ValueError("is_multimodal is required when multimodal embeddings are present")
        return _merge_multimodal_embeddings(
            inputs_embeds=embeddings,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def embed_multimodal(self, **kwargs: object) -> list[torch.Tensor]:
        """Encode batched speaker references and user turns through one codec tower."""

        audio_values = kwargs.get("audio_values")
        audio_lens = kwargs.get("audio_lens")
        if audio_values is None or not isinstance(audio_lens, torch.Tensor):
            return []

        lengths = audio_lens.reshape(-1).long()
        if lengths.numel() == 0:
            return []
        if torch.any(lengths <= 0):
            raise ValueError("audio_lens entries must be positive")

        tensor_batch: Optional[torch.Tensor] = None
        waveform_items: Optional[list[torch.Tensor]] = None
        if isinstance(audio_values, torch.Tensor):
            if audio_values.ndim == 1:
                tensor_batch = audio_values.unsqueeze(0)
            elif audio_values.ndim == 2:
                tensor_batch = audio_values
            else:
                raise ValueError(f"audio_values must be [T], [B, T], or a list of [T], got {audio_values.shape}")
            if tensor_batch.shape[0] != lengths.numel():
                raise ValueError(f"Got {tensor_batch.shape[0]} waveforms but {lengths.numel()} audio lengths")
            if torch.any(lengths > tensor_batch.shape[1]):
                raise ValueError("audio_lens contains a length larger than the corresponding waveform")
            audio_device = tensor_batch.device
        elif isinstance(audio_values, list):
            if len(audio_values) != lengths.numel():
                raise ValueError(f"Got {len(audio_values)} waveforms but {lengths.numel()} audio lengths")
            if not all(isinstance(value, torch.Tensor) and value.ndim == 1 for value in audio_values):
                raise TypeError("audio_values must be a tensor or list of 1-D tensors")
            waveform_items = audio_values
            audio_device = waveform_items[0].device
            for waveform, item_len in zip(audio_values, lengths.tolist(), strict=True):
                if item_len > waveform.numel():
                    raise ValueError("audio_lens contains a length larger than the corresponding waveform")
                if waveform.device != audio_device:
                    raise ValueError("All audio_values tensors must be on the same device")
        else:
            raise TypeError("audio_values must be a tensor or list of tensors")

        lengths = lengths.to(audio_device)

        raw_output_kinds = kwargs.get("audio_output_kinds")
        if raw_output_kinds is None:
            output_kinds = torch.full_like(lengths, _AUDIO_OUTPUT_REFERENCE)
        elif isinstance(raw_output_kinds, torch.Tensor):
            output_kinds = raw_output_kinds.reshape(-1).to(device=lengths.device, dtype=torch.long)
        elif isinstance(raw_output_kinds, list):
            output_kinds = torch.tensor(
                [int(value.item()) if isinstance(value, torch.Tensor) else int(value) for value in raw_output_kinds],
                device=lengths.device,
            )
        else:
            raise TypeError("audio_output_kinds must be a tensor or list")
        if output_kinds.numel() != lengths.numel():
            raise ValueError(f"Got {output_kinds.numel()} audio outputs but {lengths.numel()} audio lengths")
        if torch.any((output_kinds != _AUDIO_OUTPUT_USER) & (output_kinds != _AUDIO_OUTPUT_REFERENCE)):
            raise ValueError("audio_output_kinds contains an unsupported internal value")
        if torch.any(output_kinds == _AUDIO_OUTPUT_REFERENCE):
            self.arch.ensure_reference_audio_available()
        if torch.any(output_kinds == _AUDIO_OUTPUT_USER):
            self.arch.require_user_audio_prefill()
        if self.codec_encoder is None:
            raise RuntimeError("The converted EasyMagpie artifact does not contain the configured audio encoders")

        min_samples = self.speech_delay * self.arch.codec_samples_per_row
        short_user_items = (output_kinds == _AUDIO_OUTPUT_USER) & (lengths < min_samples)
        batch_lengths = torch.where(short_user_items, torch.full_like(lengths, min_samples), lengths)
        target_samples = int(batch_lengths.max().item())
        has_short_user = bool(torch.any(short_user_items).item())
        if tensor_batch is not None and not has_short_user:
            # vLLM stacks only equal-shaped items. Cropping to the longest valid
            # item is a view; the codec tower masks every shorter item's tail.
            audio_batch = tensor_batch[:, :target_samples].float()
        else:
            # Variable-shaped vLLM items, and short user turns that need left
            # warmup padding, are materialized exactly once.
            audio_batch = torch.zeros(
                (lengths.numel(), target_samples),
                device=audio_device,
                dtype=torch.float32,
            )
            length_values = lengths.tolist()
            short_user_flags = short_user_items.tolist()
            for item_index, (original_len, is_short_user) in enumerate(zip(length_values, short_user_flags)):
                if tensor_batch is not None:
                    source = tensor_batch[item_index]
                else:
                    assert waveform_items is not None
                    source = waveform_items[item_index]
                start = int(batch_lengths[item_index].item()) - original_len if is_short_user else 0
                audio_batch[item_index, start : start + original_len].copy_(source[:original_len])

        encoded = self.codec_encoder(
            audio_batch,
            batch_lengths,
            audio_frame_embedder=self.code_predictor.embed_audio_frame,
        )
        profile_codes = torch.full(
            (1, self.num_codebooks),
            self.arch.audio_user_speaking_id,
            device=encoded.acoustic_codes.device,
            dtype=torch.long,
        )
        profile_embedding = self.code_predictor.embed_audio_frame(profile_codes)

        if encoded.reference_speaker_embeddings is None or encoded.reference_speaker_embedding_lens is None:
            raise RuntimeError("Codec tower did not return reference-speaker embeddings for every audio item")
        outputs: list[torch.Tensor] = []
        for item_index, output_kind in enumerate(output_kinds.tolist()):
            if output_kind == _AUDIO_OUTPUT_REFERENCE:
                num_rows = int(encoded.reference_speaker_embedding_lens[item_index].item())
                output = encoded.reference_speaker_embeddings[item_index, :num_rows].to(
                    self._combined_embeddings.dtype
                )
            else:
                num_frames = int(encoded.acoustic_lens[item_index].item())
                item_codes = encoded.acoustic_codes[item_index, :, :num_frames].T.long()
                user_embedding = self.code_predictor.embed_audio_frame(item_codes)
                user_embedding = torch.cat((torch.zeros_like(user_embedding[:1]), user_embedding), dim=0)
                output = (user_embedding + profile_embedding).to(self._combined_embeddings.dtype)
            outputs.append(output)
        return outputs

    def _embed_phoneme(self, phoneme_tokens: torch.Tensor) -> torch.Tensor:
        """Average the per-stack phoneme embeddings (``[num_tokens, S] -> [num_tokens, dim]``)."""
        acc = self.phoneme_embeddings[0](phoneme_tokens[:, 0])
        for s in range(1, len(self.phoneme_embeddings)):
            acc = acc + self.phoneme_embeddings[s](phoneme_tokens[:, s])
        return acc / len(self.phoneme_embeddings)

    # ------------------------------------------------------------------
    # Decode-token dispatch (which positions need the local transformer)
    # ------------------------------------------------------------------

    @staticmethod
    def _select_query_layout(attn_metadata):
        """Return ``(max_query_len, query_start_loc)`` from heterogeneous metadata.

        The Nemotron-H backbone is hybrid, so ``attn_metadata`` is a per-layer
        dict mixing two metadata types:

        * **attention** layers expose the batch-level ``max_query_len`` and
          ``query_start_loc``;
        * **Mamba2** layers carry ``Mamba2AttentionMetadata``, which has *no*
          ``max_query_len`` and splits the query layout into ``query_start_loc_p``
          / ``query_start_loc_d`` instead.

        Both are built from the same batch query layout, so we prefer any
        attention-layer metadata. As a fallback for a (hypothetical) attention-free
        backbone, we infer a decode-only batch from the Mamba2 ``num_prefills``
        counter. Returns ``(None, None)`` when the layout can't be determined.
        """
        metas = list(attn_metadata.values()) if isinstance(attn_metadata, dict) else [attn_metadata]

        # Preferred: an attention layer exposes the unified query layout.
        for m in metas:
            mql = getattr(m, "max_query_len", None)
            qsl = getattr(m, "query_start_loc", None)
            if mql is not None and qsl is not None:
                return int(mql), qsl

        # Fallback: Mamba2-only backbone. We can at least detect a decode-only
        # batch (every request contributes a single token) from the counters.
        for m in metas:
            if hasattr(m, "num_prefills") and hasattr(m, "num_decodes"):
                if int(getattr(m, "num_prefills", 0)) == 0:
                    return 1, None  # decode-only -> caller runs the LT everywhere
                break
        return None, None

    def _get_query_dispatch(self):
        """Return decode rows and the final row of each prefill query.

        * ``(None, 0, None)`` → run the local transformer on every token (a warm-up
          run with no ``attn_metadata``, or a decode-only batch where
          ``max_query_len == 1``), so the captured CUDA graph covers every
          ``cudagraph_capture_sizes`` value.
        * ``(indices, num_requests, prefill_last_indices)`` → run the local
          transformer only on listed decode rows, and the phoneme projection on
          each final prefill row. ``indices`` is CUDA-graph padded;
          ``num_requests`` is its unpadded count.
        """
        ctx = get_forward_context()
        attn_metadata = ctx.attn_metadata
        if attn_metadata is None:
            return None, 0, None

        max_query_len, start_loc = self._select_query_layout(attn_metadata)

        # Decode-only batch (or layout unavailable) -> run the LT on every token.
        if max_query_len is None or max_query_len == 1 or start_loc is None:
            return None, 0, None

        tokens_per_req = start_loc[1:] - start_loc[:-1]
        is_decode = tokens_per_req == 1
        decode_token_indices = start_loc[:-1][is_decode]
        prefill_last_indices = start_loc[1:][~is_decode] - 1

        num_requests = decode_token_indices.shape[0]
        padded_num_requests = num_requests
        if self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
            idx = bisect.bisect_left(sizes, num_requests)
            if idx < len(sizes):
                padded_num_requests = sizes[idx]
        if padded_num_requests != num_requests:
            decode_token_indices = torch.nn.functional.pad(
                decode_token_indices, (0, padded_num_requests - num_requests)
            )
        return decode_token_indices, num_requests, prefill_last_indices

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Assemble the per-token embedding, run the backbone, then the codes.

        ``inputs_embeds`` carries the prefill embedding span produced by
        :meth:`preprocess` (zeros at decode positions). For decode positions we
        assemble ``text_emb + phoneme_emb + audio_emb`` in-place from the
        per-token buffers, run the backbone, then sample the codebooks with the
        local transformer (skipping prefill positions).
        """
        num_tokens = input_ids.shape[0]
        combined = self._combined_embeddings[:num_tokens]
        if inputs_embeds is not None:
            combined.copy_(inputs_embeds)
        else:
            combined.zero_()

        # Reset per-token stop flags for this step (so prefill / warm-up rows stay
        # "continue"); decode positions get set below by :meth:`_flag_audio_eos`.
        self._token_stop[:num_tokens].zero_()
        logits_index = kwargs.get("logits_index")

        decode_idx, num_req, prefill_last_idx = self._get_query_dispatch()

        if decode_idx is not None:
            # Acoustic prediction is skipped on prefill rows. Clear their shared
            # scratch slots so a prior request cannot leak a fake codec frame.
            self._out_codes[:num_tokens].zero_()
        if decode_idx is None:
            # Warm-up or decode-only batches exercise the full decode path.
            self._assemble_decode_embeddings(combined, slice(0, num_tokens))
        elif num_req > 0:
            valid = decode_idx[:num_req]
            self._assemble_decode_embeddings(combined, valid)

        hidden_states = self.backbone(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=combined,
        )

        # Sample codes (local transformer) only where needed.
        if decode_idx is None:
            codes = self.code_predictor.generate_codes(hidden_states)
            self._out_codes[:num_tokens].copy_(codes)
            self._flag_audio_eos(codes, slice(0, num_tokens))
            if self.has_phoneme:
                self._predict_phonemes(hidden_states, slice(0, num_tokens))
        elif num_req > 0:
            ctx = get_forward_context()
            orig_bd = ctx.batch_descriptor
            ctx.batch_descriptor = BatchDescriptor(num_tokens=decode_idx.shape[0])
            codes = self.code_predictor.generate_codes(hidden_states[decode_idx])
            ctx.batch_descriptor = orig_bd
            valid = decode_idx[:num_req]
            self._out_codes[valid] = codes[:num_req]
            self._flag_audio_eos(codes[:num_req], valid)
            if self.has_phoneme:
                self._predict_phonemes(hidden_states, valid)

        # The final prefetched text position predicts the phoneme consumed by the
        # first decode step. It does not need an acoustic-code prediction.
        if self.has_phoneme and prefill_last_idx is not None and prefill_last_idx.numel() > 0:
            self._predict_phonemes(hidden_states, prefill_last_idx)

        # Re-index _token_stop into _sample_stop.
        # this only happens for mixed/prefill, since for capture logits_index is None,
        # so during decode-only the branch for logits_index is None will be executed.
        if logits_index is not None:
            self._sample_stop[: logits_index.shape[0]] = self._token_stop[logits_index]
        else:
            self._sample_stop[:num_tokens].copy_(self._token_stop[:num_tokens])

        return hidden_states

    def _flag_audio_eos(self, codes: torch.Tensor, idx) -> None:
        """Flag decode positions whose newly sampled frame ends speech.
        Checks codes for eos and assigns token_stop[idx]

        Note: this uses the *sampled* codes. NeMo also checks armax(logits) == eos_idx,
        i.e. checks if EOS is emited without sampling. Skip for now.
        """
        eos = (codes == self.audio_eos_id).any(dim=1) & (self._dec_audio_valid[idx] == 1)
        self._token_stop[idx] = eos

    def _assemble_decode_embeddings(self, combined: torch.Tensor, idx) -> None:
        """Add ``text + phoneme + audio`` embeddings into ``combined`` at ``idx``."""
        # Audio: previous-frame codes (gated by validity).
        audio_codes = self._dec_audio_codes[idx]
        audio_emb = self.code_predictor.embed_audio_frame(audio_codes)
        audio_emb = audio_emb * self._dec_audio_valid[idx].unsqueeze(-1).to(audio_emb.dtype)
        combined[idx] += audio_emb

        # Text: current subword token (gated by validity).
        text_emb = self.text_embedding(self._dec_text_tokens[idx])
        text_emb = text_emb * self._dec_text_mask[idx].unsqueeze(-1).to(text_emb.dtype)
        combined[idx] += text_emb

        # Phoneme: previous predicted phoneme (gated by validity).
        if self.has_phoneme:
            phon_emb = self._embed_phoneme(self._dec_phoneme_tokens[idx])
            phon_emb = phon_emb * self._dec_phoneme_valid[idx].unsqueeze(-1).to(phon_emb.dtype)
            combined[idx] += phon_emb

    @torch.no_grad()
    def _predict_phonemes(self, hidden_states: torch.Tensor, idx) -> None:
        """Argmax the phoneme head (with confidence→UNK replacement) and stash it.

        When any stacked channel falls below
        ``phoneme_confidence_unk_threshold``,
        the whole step is replaced with the UNK id to curb error propagation.

        This is done here — not in ``preprocess``/``postprocess`` — because this
        is the only place the phoneme logits exist (preprocess has no logits, and
        postprocess only sees the argmax id). It uses only elementwise ops +
        ``torch.where`` (no ``.item()`` / host sync), so it stays CUDA-graph safe.
        """
        # Run in the model dtype (don't force fp32): ``phoneme_final_proj`` weights
        # follow ``model_config.dtype`` (e.g. bf16), and argmax is dtype-insensitive,
        # so an fp32 upcast here would mismatch the weight dtype in ``F.linear``.
        logits = self.phoneme_final_proj(hidden_states[idx])
        s = self.arch.phoneme_stacking_factor
        logits = logits.view(-1, s, self.arch.phoneme_vocab_size)
        preds = logits.argmax(dim=-1).long()  # (n, S)

        if self.phoneme_confidence_unk_threshold > 0.0:
            max_probs = torch.softmax(logits.float(), dim=-1).amax(dim=-1)  # (n, S)
            underconfident = (max_probs < self.phoneme_confidence_unk_threshold).any(dim=1, keepdim=True)
            eos_step = (preds == self.phoneme_eos_id).any(dim=1, keepdim=True)
            replace = underconfident & (~eos_step)
            preds = torch.where(replace, torch.full_like(preds, self.phoneme_unk_id), preds)

        self._dec_phoneme_tokens[idx] = preds
        self._dec_phoneme_valid[idx] = 1

    # ------------------------------------------------------------------
    # compute_logits — dummy (real output is the codes tensor)
    # ------------------------------------------------------------------

    def compute_logits(self, hidden_states, sampling_metadata: Any = None) -> Optional[torch.Tensor]:
        f"""Dummy backbone logits, repurposed as a 2-way continue/stop signal.
        ``_sample_stop`` indicates which frames contain EOS. We set logits,
        based on that: logits[sample_stop == True, stop_token_id] = 30 or -30 otherwise.
        SamplingParams should set stop_token_id as EOS token though.
        """
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        batch_size = hidden_states.shape[0]
        logits = hidden_states.new_zeros(batch_size, int(self.hf_config.vocab_size))
        if self._stop_token_id < logits.shape[1]:
            stop_rows = self._sample_stop[:batch_size]
            logits[:, self._stop_token_id] = torch.where(
                stop_rows,
                logits.new_full((), 30.0),
                logits.new_full((), -30.0),
            )
        return logits

    # ------------------------------------------------------------------
    # multimodal output plumbing
    # ------------------------------------------------------------------

    def make_omni_output(self, model_outputs, **_: Any) -> OmniOutput:
        """Surface the sampled codes (``BT x num_codebooks``).

        The codes are exposed under **two** keys so the same model serves both
        deployment shapes:

        * ``audio_codes`` — the flat single-stage key read by :meth:`postprocess`.
        * ``codes.audio`` — the nested :class:`~vllm_omni.data_entry_keys.OmniPayload`
          layout consumed by the in-engine two-stage pipeline (Code2Wav). The
          AR runner's ``flatten_payload`` turns this into the ``codes.audio``
          dotted key, which CONCATenates across decode steps into the full
          acoustic sequence for the Stage-1 producer / async-chunk streamer
          (see :mod:`easymagpie_vllm_omni.stage_processors`).
        """
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        hidden = model_outputs
        num_tokens = int(hidden.shape[0])
        audio_codes = self._out_codes[:num_tokens].clone()
        if self._single_stage_audio:
            # Drainable client key (see ``self._single_stage_audio`` in __init__):
            # ``model_outputs`` is remapped to the ``audio`` modality and drained
            # per step, so the client streams true deltas rather than a growing
            # cumulative payload. ``postprocess`` also reads this key.
            return OmniOutput(
                text_hidden_states=hidden,
                multimodal_outputs={"model_outputs": audio_codes},
            )
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={"audio_codes": audio_codes, "codes": {"audio": audio_codes}},
        )

    # ------------------------------------------------------------------
    # preprocess / postprocess
    # ------------------------------------------------------------------

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: Optional[torch.Tensor],
        *,
        start: int = 0,
        end: int = 0,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build per-request ``(input_ids, inputs_embeds)`` for this step.

        Prefill (as marked by vLLM-Omni): assemble the full context embedding
        (``[task_embedding | speaker_embedding | context_text_embedded]`` from
        the per-request inputs; see :meth:`_build_prefill_embeds`), slice this
        chunk out of it, and return it;
        ``input_ids`` are placeholders. Decode (``span_len == 1``): write the per-token decode
        inputs (previous codes, current text token, previous phoneme) into the
        model buffers at ``start`` and return a zero embedding that
        :meth:`forward` accumulates into.
        """
        nested = info_dict.get("additional_information")
        if isinstance(nested, dict):
            merged = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for k, v in nested.items():
                merged.setdefault(k, v)
            info_dict = merged

        device = input_ids.device
        span_len = int(input_ids.shape[0])
        if span_len <= 0:
            base = input_embeds if input_embeds is not None else self.embed_input_ids(input_ids)
            return input_ids, base, {}

        if bool(info_dict["_omni_is_prefill"]):
            return self._preprocess_prefill(input_ids, input_embeds, span_len, device, info_dict)

        start = self._batch_slot_offset(input_ids, start)
        return self._preprocess_decode(input_ids, start, device, info_dict)

    @staticmethod
    def _batch_slot_offset(input_ids_view: torch.Tensor, fallback: int) -> int:
        """Recover a request's batch-row offset from its 1-D ``input_ids`` view.
        The runner passes ``input_ids = input_ids_buffer[s:e]``
        """
        if input_ids_view.dim() == 1 and input_ids_view.is_contiguous():
            return int(input_ids_view.storage_offset())
        return int(fallback)

    def _preprocess_prefill(
        self,
        input_ids: torch.Tensor,
        input_embeds: Optional[torch.Tensor],
        span_len: int,
        device: torch.device,
        info_dict: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        self._maybe_set_lt_sampling_params(info_dict)
        # Absolute position within this request's retained Stage-0 state.
        offset = int(info_dict["_omni_num_computed_tokens"])
        text_tokens = list(info_dict.get("text_tokens") or [])
        text_tokens_changed = False
        text = info_dict.get("text")
        incoming = info_dict.get("text_token") or []
        if text is not None:
            # Replace the previous turn's target buffer without resetting the
            # backbone/Mamba state.
            text_tokens = self._encode_text_stream(text)
            text_tokens_changed = True
        elif incoming:
            text_token_start = info_dict.get("text_token_start")
            base_tokens = [] if int(text_token_start or 0) == 0 else text_tokens
            text_tokens, text_tokens_changed = _merge_streaming_text_chunk(base_tokens, incoming, text_token_start)

        prompt_len = int(info_dict["_omni_prompt_len"])
        reference_prefilled = bool(info_dict.get("speaker_reference_prefilled", False))
        reference_prompt_len = int(info_dict.get("reference_prefill_prompt_len", 0) or 0)
        continuing_reference_prefill = (
            reference_prefilled and reference_prompt_len > 0 and offset < reference_prompt_len
        )
        uses_raw_reference = not info_dict.get("speaker_id") and (
            not reference_prefilled or continuing_reference_prefill
        )
        reference_rows = 0

        # Select the Stage-0 prefill path for this request. Streaming turns in
        # one generate() call share the request's retained reference state.
        if uses_raw_reference:
            # Process the initial raw-reference prompt, including subsequent
            # scheduler chunks that still belong to that prompt.
            take, conditioning_len, has_user_audio, reference_rows = self._build_reference_audio_prefill_chunk(
                input_embeds=input_embeds,
                info_dict=info_dict,
                text_tokens=text_tokens,
                offset=offset,
                span_len=span_len,
                prompt_len=prompt_len,
                input_ids=input_ids,
            )
            decode_offset = (
                0
                if conditioning_len == 0
                else (self.speech_delay if has_user_audio else int(info_dict.get("text_prefill_num", 0) or 0))
            )
        elif reference_prefilled:
            # Process a later user-audio turn in the same request. Reference conditioning
            # is already retained, so the stored prefix length is used to align the
            # appended user-audio rows.
            conditioning_len = int(info_dict.get("reference_conditioning_len", 0) or 0)
            if conditioning_len <= 0:
                raise ValueError("speaker_reference_prefilled requires reference_conditioning_len")
            dtype = input_embeds.dtype if input_embeds is not None else self._combined_embeddings.dtype
            conditioning = torch.zeros(
                (conditioning_len, self.embedding_dim),
                device=device,
                dtype=dtype,
            )
            take = self._build_user_audio_prefill_chunk(
                conditioning=conditioning,
                input_embeds=input_embeds,
                text_tokens=text_tokens,
                offset=offset,
                span_len=span_len,
                prompt_len=prompt_len,
            )
            decode_offset = self.speech_delay
            has_user_audio = True
        else:
            # Load the precomputed speaker embedding selected by speaker_id.
            conditioning = self._build_prefill_embeds(device, info_dict, include_text_prefill=False)
            conditioning_len = int(conditioning.shape[0])
            legacy_prompt_len = conditioning_len + int(info_dict.get("text_prefill_num", 0) or 0)
            has_user_audio = self._is_user_audio_prefill_chunk(
                input_ids=input_ids,
                offset=offset,
                conditioning_len=conditioning_len,
                prompt_len=prompt_len,
                legacy_prompt_len=legacy_prompt_len,
            )

            if has_user_audio:
                take = self._build_user_audio_prefill_chunk(
                    conditioning=conditioning,
                    input_embeds=input_embeds,
                    text_tokens=text_tokens,
                    offset=offset,
                    span_len=span_len,
                    prompt_len=prompt_len,
                )
                decode_offset = self.speech_delay
            else:
                target_prefill = self._build_text_prefill_embeds(device, conditioning.dtype, info_dict)
                prefill_embeds = (
                    conditioning if target_prefill is None else torch.cat((conditioning, target_prefill), dim=0)
                )
                take = prefill_embeds[offset : offset + span_len]
                total = int(prefill_embeds.shape[0])
                assert int(take.shape[0]) == span_len, (
                    f"EasyMagpieTTS prefill chunk [{offset}:{offset + span_len}] is not fully covered by the "
                    f"assembled context embedding (length {total}). The caller must size prompt_token_ids to "
                    "[task?] + speaker frames + context-text tokens + text_prefill_num."
                )
                decode_offset = int(info_dict.get("text_prefill_num", 0) or 0)

        info_update: dict[str, Any] = {"decode_offset": decode_offset}
        if uses_raw_reference and conditioning_len > 0:
            info_update.update(
                speaker_reference_prefilled=True,
                reference_conditioning_len=conditioning_len,
                reference_audio_num_rows=reference_rows,
                reference_prefill_prompt_len=prompt_len,
            )
        if has_user_audio:
            info_update.update(
                user_audio_prefilled=True,
                phoneme_ended=False,
                # The final prefill row predicts a fresh phoneme for this turn.
                last_phoneme_token=None,
            )
        if text_tokens_changed:
            info_update["text_tokens"] = text_tokens
        input_ids_out = torch.full_like(input_ids, _DUMMY_TOKEN_ID)
        return input_ids_out, take, info_update

    def _build_reference_audio_prefill_chunk(
        self,
        *,
        input_embeds: Optional[torch.Tensor],
        info_dict: dict[str, Any],
        text_tokens: list[int],
        offset: int,
        span_len: int,
        prompt_len: int,
        input_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, int, bool, int]:
        """Overlay deterministic rows around codec-produced speaker conditioning.

        The expanded prompt layout is ``[task? | reference | context text |
        text prefill or user audio]``. Reference and user rows already contain
        codec-tower embeddings, so only the surrounding deterministic streams
        are replaced or added here.
        """
        if input_embeds is None:
            raise RuntimeError("Reference-audio prefill requires vLLM multimodal embeddings")
        if offset + span_len > prompt_len:
            raise ValueError(
                f"Reference-audio chunk [{offset}:{offset + span_len}] exceeds prompt length {prompt_len}"
            )

        device = input_embeds.device
        dtype = input_embeds.dtype
        task_mode_id = self._require_default_task_mode(info_dict)
        task_rows: Optional[torch.Tensor] = None
        if self.task_embedding is not None:
            task_rows = self.task_embedding(torch.tensor([task_mode_id], device=device, dtype=torch.long)).to(dtype)
        task_len = 0 if task_rows is None else int(task_rows.shape[0])

        context_text = info_dict.get("context_text") or _DEFAULT_CONTEXT_TEXT
        context_ids = self._encode_context_text(context_text, device)
        context_rows = self.text_embedding(context_ids).to(dtype)
        take = input_embeds.clone()

        def overlay(rows: Optional[torch.Tensor], absolute_start: int, *, add: bool = False) -> None:
            if rows is None or rows.shape[0] == 0:
                return
            absolute_end = absolute_start + int(rows.shape[0])
            overlap_start = max(offset, absolute_start)
            overlap_end = min(offset + span_len, absolute_end)
            if overlap_end <= overlap_start:
                return
            dst = slice(overlap_start - offset, overlap_end - offset)
            src = slice(overlap_start - absolute_start, overlap_end - absolute_start)
            if add:
                take[dst] += rows[src]
            else:
                take[dst].copy_(rows[src])

        # The task row is deterministic even when a chunk does not yet expose
        # the boundary between variable-length reference and user audio.
        overlay(task_rows, 0)
        reference_start = task_len
        text_prefill_num = int(info_dict.get("text_prefill_num", 0) or 0)
        reference_rows = int(info_dict.get("reference_audio_num_rows", 0) or 0)
        if reference_rows <= 0:
            if input_ids is None:
                raise ValueError("input_ids are required to locate variable-length reference audio")
            scan_start = max(reference_start - offset, 0)
            if offset + span_len > reference_start:
                reference_chunk = input_ids[scan_start:]
                boundary = torch.nonzero(reference_chunk != self.arch.audio_input_token_id, as_tuple=False).flatten()
                if boundary.numel() > 0:
                    reference_rows = offset + scan_start + int(boundary[0].item()) - reference_start
        if reference_rows <= 0:
            if offset + span_len == prompt_len:
                raise ValueError(
                    "A fresh request without speaker_id must place reference audio before context rows; "
                    "a sole final audio marker is valid only for known-speaker or resumed raw-reference turns"
                )
            return take, 0, False, 0
        context_start = reference_start + reference_rows
        conditioning_len = context_start + int(context_rows.shape[0])
        trailing_rows = prompt_len - conditioning_len
        if trailing_rows == text_prefill_num:
            has_user_audio = False
        elif trailing_rows >= self.speech_delay:
            self.arch.require_user_audio_prefill()
            has_user_audio = True
        else:
            raise ValueError(
                f"Raw-reference prompt has {trailing_rows} trailing rows; expected {text_prefill_num} text-prefill "
                f"rows or at least {self.speech_delay} user-audio rows"
            )

        overlay(context_rows, context_start)
        if has_user_audio:
            warmup = self._build_user_audio_warmup_embeds(device, dtype, text_tokens)
            overlay(warmup, prompt_len - self.speech_delay, add=True)
        else:
            target_prefill = self._build_text_prefill_embeds(device, dtype, info_dict)
            overlay(target_prefill, conditioning_len)
        return take, conditioning_len, has_user_audio, reference_rows

    def _is_user_audio_prefill_chunk(
        self,
        *,
        input_ids: torch.Tensor,
        offset: int,
        conditioning_len: int,
        prompt_len: int,
        legacy_prompt_len: int,
    ) -> bool:
        """Return whether this request chunk contains appended user-audio rows."""
        if not self.arch.condition_on_user_speech or prompt_len <= legacy_prompt_len:
            return False
        # The first scheduling chunk can contain only the ordinary context
        # prefix. Every later audio chunk contains the expanded placeholder.
        # Checking it prevents a later text-only streaming update from being
        # mistaken for another user utterance merely because prompt_len is
        # cumulative across the preserved Stage-0 session.
        return offset < conditioning_len or bool(torch.any(input_ids == self.arch.audio_input_token_id).item())

    def _build_user_audio_prefill_chunk(
        self,
        *,
        conditioning: torch.Tensor,
        input_embeds: Optional[torch.Tensor],
        text_tokens: list[int],
        offset: int,
        span_len: int,
        prompt_len: int,
    ) -> torch.Tensor:
        """Compose one chunk of ``[base conditioning | encoded user speech]``."""
        if input_embeds is None:
            raise RuntimeError("Raw user-audio prefill requires vLLM multimodal embeddings")
        conditioning_len = int(conditioning.shape[0])
        if prompt_len - conditioning_len < self.speech_delay:
            raise ValueError("User-audio prefill is shorter than streaming_speech_delay")
        if offset + span_len > prompt_len:
            raise ValueError(
                f"User-audio prefill chunk [{offset}:{offset + span_len}] exceeds prompt length {prompt_len}"
            )

        take = input_embeds.clone()
        prefix_end = min(offset + span_len, conditioning_len)
        if prefix_end > offset:
            count = prefix_end - offset
            take[:count].copy_(conditioning[offset:prefix_end])

        # Native inference executes the final ``speech_delay`` user rows as
        # prefill-like warmup steps with the first target-text tokens. vLLM can
        # causally prefill these rows together. The only approximation is that
        # the final warmup row cannot consume the phoneme predicted by its
        # predecessor until vLLM supports an interleaved prefill callback.
        warmup_start = prompt_len - self.speech_delay
        overlap_start = max(offset, warmup_start)
        overlap_end = min(offset + span_len, prompt_len)
        if overlap_end > overlap_start:
            warmup = self._build_user_audio_warmup_embeds(take.device, take.dtype, text_tokens)
            dst_start = overlap_start - offset
            dst_end = overlap_end - offset
            src_start = overlap_start - warmup_start
            src_end = overlap_end - warmup_start
            take[dst_start:dst_end] += warmup[src_start:src_end]
        return take

    def _build_user_audio_warmup_embeds(
        self,
        device: torch.device,
        dtype: torch.dtype,
        text_tokens: list[int],
    ) -> torch.Tensor:
        """Build text and phoneme rows aligned with the user-audio warmup."""
        rows = torch.zeros((self.speech_delay, self.embedding_dim), device=device, dtype=dtype)
        prefix_ids = text_tokens[: self.speech_delay]
        if prefix_ids:
            ids = torch.tensor(prefix_ids, device=device, dtype=torch.long)
            rows[: len(prefix_ids)] += self.text_embedding(ids).to(dtype)
        if self.has_phoneme and self.phonemes_delay < self.speech_delay:
            bos = torch.full(
                (1, self.arch.phoneme_stacking_factor),
                self.phoneme_bos_id,
                device=device,
                dtype=torch.long,
            )
            rows[self.phonemes_delay : self.phonemes_delay + 1] += self._embed_phoneme(bos).to(dtype)
        return rows

    def _build_prefill_embeds(
        self,
        device: torch.device,
        info_dict: dict[str, Any],
        *,
        include_text_prefill: bool = True,
    ) -> torch.Tensor:
        """Assemble the full ``(T_ctx, embedding_dim)`` prefill context embedding::

            [task_embedding | speaker_embedding | context_text_embedded | target_text_prefill]

        from the per-request inputs:

        * ``speaker_id`` — identifies a known speaker whose embedding is stored
          in the converted artifact.
        * ``context_text`` — a plain string (e.g. ``"[EN]"``); tokenized in-model
          and embedded through the baked per-subword ``text_embedding`` table.
        * ``task_mode_id`` — must be row 0; the default task ("service token")
          embedding row is prepended when the checkpoint has a task table.

        Returns the full conditioning plus request-specific causal text prefix;
        per-chunk slicing is done by :meth:`_preprocess_prefill`.

        For a known ``speaker_id`` the result is a pure function of
        ``(task_mode_id, speaker_id, context_text)`` and is cached in
        ``self._prefill_cache``, so the tokenize + embed + cat below run once per
        distinct combo instead of on every request's prefill. The returned tensor
        is only ever read (sliced) downstream, never mutated, so sharing the
        cached instance is safe.
        """
        speaker_id = info_dict.get("speaker_id")
        if not speaker_id:
            raise ValueError("Known-speaker prefill requires additional_information.speaker_id")
        context_text = info_dict.get("context_text") or _DEFAULT_CONTEXT_TEXT
        task_mode_id = self._require_default_task_mode(info_dict)

        cache_key = (task_mode_id, speaker_id, context_text, str(device))
        cached = self._prefill_cache.get(cache_key)
        if cached is not None:
            if not include_text_prefill:
                return cached
            target_prefill = self._build_text_prefill_embeds(device, self._combined_embeddings.dtype, info_dict)
            return cached if target_prefill is None else torch.cat((cached, target_prefill), dim=0)

        dtype = self._combined_embeddings.dtype
        parts: list[torch.Tensor] = []

        # Task / "service token" embedding (prepended), when present.
        if self.task_embedding is not None:
            task_row = self.task_embedding(torch.tensor([task_mode_id], device=device, dtype=torch.long))
            parts.append(task_row.to(dtype))

        # Speaker-encoded context audio stored in the converted checkpoint.
        parts.append(self._load_known_speaker_embedding(speaker_id, device, dtype))

        # Context text: tokenized in-model and embedded through the baked table.
        ctx_ids = self._encode_context_text(context_text, device)
        if ctx_ids.numel() > 0:
            parts.append(self.text_embedding(ctx_ids).to(dtype))

        embeds = torch.cat(parts, dim=0)
        self._prefill_cache[cache_key] = embeds
        if not include_text_prefill:
            return embeds
        target_prefill = self._build_text_prefill_embeds(device, dtype, info_dict)
        return embeds if target_prefill is None else torch.cat((embeds, target_prefill), dim=0)

    def _build_text_prefill_embeds(
        self,
        device: torch.device,
        dtype: torch.dtype,
        info_dict: dict[str, Any],
    ) -> Optional[torch.Tensor]:
        """Build the causal text-led rows moved from decode into prefill."""
        text_prefill_num = int(info_dict.get("text_prefill_num", 0) or 0)
        if text_prefill_num == 0:
            return None
        assert text_prefill_num == self.arch.text_prefill_num, (
            f"EasyMagpieTTS expected text_prefill_num={self.arch.text_prefill_num}, " f"got {text_prefill_num}"
        )

        prefix_ids = list(info_dict.get("prefill_text_tokens") or [])
        assert len(prefix_ids) <= text_prefill_num, (
            f"EasyMagpieTTS got {len(prefix_ids)} prefill text tokens for " f"text_prefill_num={text_prefill_num}"
        )
        rows = torch.zeros((text_prefill_num, self.embedding_dim), device=device, dtype=dtype)
        if prefix_ids:
            ids = torch.tensor(prefix_ids, device=device, dtype=torch.long)
            rows[: len(prefix_ids)] = self.text_embedding(ids).to(dtype)

        # At position phonemes_delay the phoneme input is known: it is BOS. The
        # phoneme projected from this row is fed back at the first decode step.
        if self.has_phoneme:
            bos_row = self.phonemes_delay
            assert bos_row < text_prefill_num
            bos = torch.full(
                (1, self.arch.phoneme_stacking_factor),
                self.phoneme_bos_id,
                device=device,
                dtype=torch.long,
            )
            rows[bos_row : bos_row + 1] += self._embed_phoneme(bos).to(dtype)
        return rows

    @staticmethod
    def _require_default_task_mode(info_dict: dict[str, Any]) -> int:
        """Reject task rows that the serving implementation does not support."""
        task_mode_id = int(info_dict.get("task_mode_id", 0) or 0)
        if task_mode_id != 0:
            raise ValueError(f"Only task_mode_id=0 is supported; got {task_mode_id}")
        return task_mode_id

    def _load_known_speaker_embedding(self, speaker_id: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Read one known speaker's embedding from ``<model_path>/speaker_embeddings/<id>.pt``.

        The file holds either a bare ``(T_audio, embedding_dim)`` tensor or a dict
        with a ``speaker_encoding`` key (the converter/caller layout); it is moved
        to ``device`` in model dtype. Called only on a prefill-cache miss, so each
        known speaker is read at most once per ``(context_text, task)`` combo and
        the result is then baked into ``self._prefill_cache``. Read from disk (not
        via :meth:`load_weights`) so known speakers work even under
        ``--load-format dummy``, which skips weight loading.
        """
        import glob
        import os

        spk_dir = os.path.join(self.model_path, "speaker_embeddings")
        path = os.path.join(spk_dir, f"{speaker_id}.pt")
        if not os.path.exists(path):
            known = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(spk_dir, "*.pt")))
            raise AssertionError(
                f"EasyMagpieTTS preprocess got unknown speaker_id {speaker_id!r}; known speakers: {known}. "
                "Register it under the checkpoint's speaker_embeddings/ dir, or supply raw reference audio."
            )
        loaded = torch.load(path, map_location="cpu")
        embedding = loaded["speaker_encoding"] if isinstance(loaded, dict) else loaded
        assert isinstance(embedding, torch.Tensor) and embedding.ndim == 2, (
            f"EasyMagpieTTS: speaker embedding {path} must be a 2-D (T_audio, embedding_dim) tensor; "
            f"got {type(embedding).__name__}"
            + (f" with ndim={embedding.ndim}" if isinstance(embedding, torch.Tensor) else "")
        )
        return embedding.to(device=device, dtype=dtype)

    def _maybe_set_lt_sampling_params(self, info_dict: dict[str, Any]) -> None:
        """Apply per-request audio sampling params to the local transformer.

        Reads ``temperature`` / ``top_k`` (alias ``topk``) from the request's
        ``additional_information`` and stores them on the code predictor. Absent
        keys leave the existing defaults untouched.
        """
        temperature = info_dict.get("temperature")
        if temperature is not None:
            self.code_predictor.temperature = float(temperature)
        top_k = info_dict.get("top_k", info_dict.get("topk"))
        if top_k is not None:
            self.code_predictor.top_k = int(top_k)

    def _get_text_tokenizer(self):
        """Lazily load target/context tokenization from the model directory."""
        if self._text_tokenizer is None:
            self._text_tokenizer = EasyMagpieTextTokenizer.from_pretrained(self.model_path)
        return self._text_tokenizer

    def _encode_context_text(self, context_text: str, device: torch.device) -> torch.Tensor:
        """Tokenize ``context_text`` to subword ids.

        The text-conditioning tokenizer sits at offset 0 in the model's
        tokenizer aggregate, so its raw ids index the baked ``text_embedding``
        table directly.
        """
        tok = self._get_text_tokenizer()
        ids = tok.encode_context(context_text)
        return torch.tensor(ids, device=device, dtype=torch.long)

    def _encode_text_stream(self, text: str) -> list[int]:
        """Tokenize the target ``text`` into the streaming subword-id list.

        HF special tokens are disabled so the raw ids index the baked
        ``text_embedding`` table directly, and the trailing text-EOS id closes
        the stream. One id is consumed per decode step (see
        :meth:`_preprocess_decode`); once exhausted the text channel is masked
        off.
        """
        tok = self._get_text_tokenizer()
        ids = tok.encode(text, add_special_tokens=False)
        return list(ids) + [self.text_eos_id]

    @staticmethod
    def estimate_prompt_len(
        speaker_embedding: torch.Tensor,
        *,
        tokenize: Callable[[str], Iterable[int]],
        context_text: str = _DEFAULT_CONTEXT_TEXT,
        has_task_embedding: bool = False,
    ) -> int:
        """Compute the speaker-conditioning prefill length for a custom voice.

        The engine assembles the prefill context as
        ``[task_embedding? | speaker_embedding | context_text_embedded]``, so the
        this base length plus ``text_prefill_num`` target rows. The caller must
        pass that total as the placeholder length so it matches the assembled
        embedding (otherwise vLLM pads / truncates and quality drops). The base
        length is a pure function of
        lengths, so it stays static — callable in the request-building process
        without an engine instance.

        For a **known speaker** the caller holds only a ``speaker_id`` (not the
        tensor); use :meth:`get_prompt_len`, which loads the embedding from the
        checkpoint dir and calls this method.

        Args:
            speaker_embedding: ``(T_audio, embedding_dim)`` speaker-encoded
                context-audio embedding (only its length is used).
            tokenize: callable turning ``context_text`` into its subword ids
                (e.g. ``lambda t: tokenizer.encode(t)``) — must match the
                tokenizer the engine loads from ``model_path``.
            context_text: conditioning string (default ``"[EN]"``).
            has_task_embedding: whether the checkpoint prepends a task /
                "service token" embedding (``num_task_embeddings > 0``).
        """
        t_audio = int(speaker_embedding.shape[0])
        ctx_len = len(list(tokenize(context_text or _DEFAULT_CONTEXT_TEXT)))
        task_len = 1 if has_task_embedding else 0
        return task_len + t_audio + ctx_len

    @classmethod
    def get_prompt_len(cls, speaker_id: str, model_path: str, *, tokenize: Callable[[str], Iterable[int]]) -> int:
        """Known-speaker convenience wrapper around :meth:`estimate_prompt_len`.

        Resolves everything from the checkpoint dir so it cannot disagree with
        what the engine actually uses: loads the speaker embedding from
        ``speaker_embeddings/<speaker_id>.pt`` (the same file the engine reads in
        :meth:`_load_known_speaker_embedding`), reads ``has_task_embedding`` from
        ``config.json`` (``num_task_embeddings``),
        and conditions on the fixed :data:`_DEFAULT_CONTEXT_TEXT`. Lets a caller
        holding only a ``speaker_id`` size ``prompt_token_ids`` without an engine
        instance (``context_text`` / ``has_task_embedding`` are intentionally not
        params — they must match the precomputed checkpoint, not be overridden).
        """
        import json
        import os

        path = os.path.join(model_path, "speaker_embeddings", f"{speaker_id}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"EasyMagpieTTS: no speaker embedding {path} for speaker_id {speaker_id!r}")
        loaded = torch.load(path, map_location="cpu")
        speaker_embedding = loaded["speaker_encoding"] if isinstance(loaded, dict) else loaded

        with open(os.path.join(model_path, "config.json")) as f:
            num_task_embeddings = int(json.load(f).get("num_task_embeddings", 0))

        return cls.estimate_prompt_len(
            speaker_embedding,
            tokenize=tokenize,
            context_text=_DEFAULT_CONTEXT_TEXT,
            has_task_embedding=num_task_embeddings > 0,
        )

    def _preprocess_decode(
        self,
        input_ids: torch.Tensor,
        start: int,
        device: torch.device,
        info_dict: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        decode_offset = int(info_dict.get("decode_offset", 0) or 0)
        info_update: dict[str, Any] = {"decode_offset": decode_offset + 1}

        # ── Text channel ── (delay 0: one subword per step from step 0). The text
        # stream leads the phoneme/audio streams by their respective delays. The
        # model always consumes exactly one buffered subword id per decode step,
        # indexed by ``decode_offset`` from a persistent ``text_tokens`` list. That
        # list is populated by one of two mutually exclusive input modes:
        #
        # * **Whole-text (non-streaming)** — the caller passed ``text`` whole at
        #   prefill; it was tokenized in-model and stashed as the ``text_tokens``
        #   list (see :meth:`_preprocess_prefill`). No per-step ``text_token``
        #   arrives, so the buffer never grows here.
        # * **Streamed** — the caller did *not* pass ``text`` at prefill and instead
        #   pushes subword ids during decode via ``additional_information`` under
        #   ``text_token`` (always a ``list[int]``). Each chunk may carry a single id
        #   (``[id]`` with ``max_tokens == 1``, one frame per chunk) or several ids at
        #   once (``max_tokens == N``, so the engine free-runs N frames off one chunk —
        #   fewer round-trips). Those ids are appended to ``text_tokens`` and consumed
        #   one per step.
        #
        # In both modes, once the buffer is exhausted the channel is masked off
        # (adds nothing) rather than repeating the last token, so the caller can keep
        # pumping decode steps (passing an empty ``text_token`` list) while the audio
        # tail finishes.
        #
        # A streamed chunk's ``text_token`` payload stays identical across every
        # decode step of its segment. ``text_token_start`` is its absolute position
        # in the accumulated buffer, which makes repeated metadata and async
        # one-segment lookahead safe.
        text_tokens = info_dict.get("text_tokens") or []
        incoming = info_dict.get("text_token") or []
        text_token_start = info_dict.get("text_token_start")
        if incoming:
            text_tokens, appended = _merge_streaming_text_chunk(text_tokens, incoming, text_token_start)
            if appended:
                info_update["text_tokens"] = text_tokens
        if decode_offset < len(text_tokens):
            self._dec_text_tokens[start] = int(text_tokens[decode_offset])
            self._dec_text_mask[start] = 1
        else:
            self._dec_text_mask[start] = 0

        # ── Phoneme channel ── opens at decode step == ``phonemes_delay`` (seeded
        # with phoneme BOS), then feeds back the previous step's prediction, and
        # closes one step after the model emits the phoneme EOS (sticky flag).
        if self.has_phoneme:
            phoneme_ended = bool(info_dict.get("phoneme_ended", False))
            feed_eos = False
            if phoneme_ended or decode_offset < self.phonemes_delay:
                self._dec_phoneme_valid[start] = 0
            elif decode_offset == self.phonemes_delay:
                self._dec_phoneme_tokens[start].fill_(self.phoneme_bos_id)
                self._dec_phoneme_valid[start] = 1
            else:
                last_phon = info_dict.get("last_phoneme_token")
                if isinstance(last_phon, torch.Tensor) and last_phon.numel() > 0:
                    p = last_phon.to(device=device, dtype=torch.long).reshape(-1)[: self.arch.phoneme_stacking_factor]
                    self._dec_phoneme_tokens[start, : p.shape[0]].copy_(p)
                    self._dec_phoneme_valid[start] = 1
                    feed_eos = bool((p == self.phoneme_eos_id).any())
                else:
                    self._dec_phoneme_valid[start] = 0
            if phoneme_ended or feed_eos:
                info_update["phoneme_ended"] = True

        # ── Audio channel ── opens at decode step == ``speech_delay`` (seeded with
        # audio BOS), then feeds back the previous frame's codes. For the leading
        # ``speech_delay`` steps the channel is masked off (only text/phoneme
        # condition the backbone); the local transformer still runs for CUDA-graph
        # stability but its codes for those frames are discarded by the caller and
        # never fed back here.
        if decode_offset < self.speech_delay:
            self._dec_audio_valid[start] = 0
        elif decode_offset == self.speech_delay:
            if info_dict.get("user_audio_prefilled") and self.arch.use_user_speaking_end_token:
                # Native multiturn inference places USER_SPEAKING_END in the
                # input slot that predicts the first agent acoustic frame.
                self._dec_audio_codes[start].fill_(self.arch.audio_user_speaking_end_id)
            else:
                self._dec_audio_codes[start].fill_(self.arch.audio_bos_id)
            self._dec_audio_valid[start] = 1
        else:
            last_codes = info_dict.get("last_audio_codes")
            if isinstance(last_codes, torch.Tensor) and last_codes.numel() > 0:
                c = last_codes.to(device=device, dtype=torch.long).reshape(-1)[: self.num_codebooks]
                self._dec_audio_codes[start, : c.shape[0]].copy_(c)
                self._dec_audio_valid[start] = 1
            else:
                # Fallback (should not happen once audio has started): seed BOS.
                self._dec_audio_codes[start].fill_(self.arch.audio_bos_id)
                self._dec_audio_valid[start] = 1

        inputs_embeds_out = torch.zeros((1, self.embedding_dim), device=device, dtype=self._combined_embeddings.dtype)
        return input_ids, inputs_embeds_out, info_update

    def postprocess(self, hidden_states: torch.Tensor, multimodal_outputs: Optional[dict[str, Any]] = None, **_: Any):
        """Stash the last frame's codes (and phoneme) for the next decode step."""
        if hidden_states.numel() == 0:
            return {}
        stride0 = hidden_states.stride(0) or 1
        req_start = hidden_states.storage_offset() // stride0
        last = req_start + hidden_states.shape[0] - 1

        out: dict[str, Any] = {}
        mm = multimodal_outputs or {}
        # The codes key depends on the emission mode (see make_omni_output):
        # single-stage uses "model_outputs", two-stage uses "audio_codes" /
        # nested "codes.audio". Read whichever is present.
        audio_codes = mm.get("audio_codes")
        if audio_codes is None:
            audio_codes = mm.get("model_outputs")
        if audio_codes is None:
            codes = mm.get("codes")
            if isinstance(codes, dict):
                audio_codes = codes.get("audio")
            elif "codes.audio" in mm:
                audio_codes = mm.get("codes.audio")
        if isinstance(audio_codes, torch.Tensor) and audio_codes.numel() > 0:
            out["last_audio_codes"] = audio_codes[last : last + 1].detach()
        if self.has_phoneme:
            out["last_phoneme_token"] = self._dec_phoneme_tokens[last : last + 1].detach().clone()
        return out

    # ------------------------------------------------------------------
    # weight loading
    # ------------------------------------------------------------------

    # Checkpoint prefixes (EasyMagpieTTS state dict) → in-model paths.
    # ``decoder.*`` is fed to the vLLM backbone loader separately (it understands
    # HF Nemotron-H naming + Mamba/MoE packing). The TTS submodules are copied
    # manually.
    _TTS_PREFIX_MAP = {
        "audio_encoder.": "codec_encoder.audio_encoder.",
        "reference_speaker_encoder.": "codec_encoder.reference_speaker_encoder.",
        "local_transformer.": "code_predictor.local_transformer.",
        "local_transformer_in_projection.": "code_predictor.local_transformer_in_projection.",
        "local_transformer_audio_out_projection.": "code_predictor.local_transformer_audio_out_projection.",
        "local_transformer_out_projections.": "code_predictor.local_transformer_out_projections.",
        "audio_embeddings.": "code_predictor.audio_embeddings.",
        "audio_in_projection.": "code_predictor.audio_in_projection.",
        "phoneme_embeddings.": "phoneme_embeddings.",
        "phoneme_final_proj.": "phoneme_final_proj.",
        "text_embedding.": "text_embedding.",
        "task_embedding.": "task_embedding.",
    }

    def _validate_audio_encoder_artifacts(self) -> None:
        """Validate optional raw-audio files once, while constructing the model."""
        required = ("codec_encoder.json", "codec_encoder.safetensors")
        missing = [name for name in required if not os.path.isfile(os.path.join(self.model_path, name))]
        if missing:
            raise FileNotFoundError(
                "This EasyMagpie artifact declares raw-audio support, but "
                f"{', '.join(missing)} is missing from {self.model_path}. "
                "Re-run convert_to_vllm.py with --bundle-audio-encoders."
            )

    def _remap_tts_key(self, name: str) -> Optional[str]:
        """Map a raw checkpoint key to its in-model parameter path (or ``None``)."""
        for src, dst in self._TTS_PREFIX_MAP.items():
            if name.startswith(src):
                return dst + name[len(src) :]
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load backbone (Nemotron-H) + TTS submodule weights from a converted checkpoint.

        The converted checkpoint carries the backbone under ``decoder.*`` (HF
        Nemotron-H names) and the TTS submodules at top level
        (``audio_embeddings.*``, ``local_transformer.*``, ``phoneme_*``,
        ``text_embedding.*``, projection heads). Backbone weights are routed to
        :meth:`NemotronHModel.load_weights` (which handles HF naming + Mamba/MoE
        packing); TTS weights are copied directly by name.
        """
        # The codec's fixed STFT windows are persistent buffers in the
        # standalone safetensors shard and must be restored alongside weights.
        own_params = {**dict(self.named_parameters()), **dict(self.named_buffers())}
        loaded: set[str] = set()
        backbone_weights: list[tuple[str, torch.Tensor]] = []

        for name, tensor in weights:
            if name.startswith("decoder."):
                backbone_weights.append((name[len("decoder.") :], tensor))
                continue
            mapped = self._remap_tts_key(name)
            if mapped is None:
                # Unrelated checkpoint section (codec, speaker encoder, CAS, etc.).
                continue
            if mapped.startswith("codec_encoder.") and self.codec_encoder is None:
                continue
            if mapped.startswith("task_embedding.") and self.task_embedding is None:
                # Single-mode model: checkpoint may still ship an (unused) table.
                continue
            target = own_params.get(mapped)
            if target is None:
                logger.warning("EasyMagpieTTS: no parameter for checkpoint key %s -> %s", name, mapped)
                continue
            # The local-transformer FFN ships as kernel-1 ``Conv1d`` weights
            # (``[out, in, 1]``) but now lives as ``nn.Linear`` (``[out, in]``).
            # Squeeze the trailing singleton conv dim so the dense layer loads 1:1.
            if tensor.ndim == target.ndim + 1 and tensor.shape[-1] == 1:
                tensor = tensor.squeeze(-1)
            if target.shape != tensor.shape:
                raise RuntimeError(
                    f"EasyMagpieTTS weight shape mismatch at {mapped!r}: "
                    f"ckpt {tuple(tensor.shape)} vs model {tuple(target.shape)}"
                )
            with torch.no_grad():
                target.data.copy_(tensor.to(target.dtype))
            loaded.add(mapped)

        # ``NemotronHModel.load_weights`` (the inner model) does *not* apply the
        # HF->vLLM renaming that lives on the ``NemotronHForCausalLM`` wrapper, so
        # raw HF names such as ``embeddings.weight`` / ``...mixer.A_log`` would not
        # match the inner param names (``embed_tokens.weight`` / ``...mixer.A``).
        # Apply that mapper here so the converted checkpoint can keep stock HF
        # Nemotron-H names. The wrapper's ``backbone -> model`` prefix rule is a
        # no-op here because we already stripped the ``decoder.`` prefix.
        backbone_weights = list(NemotronHForCausalLM.hf_to_vllm_mapper.apply(backbone_weights))
        backbone_loaded = self.backbone.load_weights(backbone_weights)
        loaded |= {f"backbone.{n}" for n in backbone_loaded}

        # Derived runtime state.
        self.code_predictor.init_forbidden_mask()

        logger.info("Loaded %d weights for EasyMagpieTTSForConditionalGeneration", len(loaded))
        return loaded
