# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

EasyMagpieTTS is a decoder-only streaming TTS model. A Nemotron-H hybrid
(Mamba2 + attention + MoE) text-LM backbone consumes a per-frame additive input
embedding (text + phoneme + audio) and emits a per-frame hidden state. A small
autoregressive *local transformer* then samples all ``C * S`` stacked audio
codebooks for that frame (see :mod:`easymagpie_vllm_omni.local_transformer`).

This module wires that architecture into vLLM-Omni's
``preprocess`` / ``forward`` / ``compute_logits`` / ``make_omni_output`` /
``postprocess`` contract:

* **Backbone** — vLLM's
  :class:`~vllm.model_executor.models.nemotron_h.NemotronHModel` is reused
  wholesale (hybrid Mamba2 state + KV cache + paged attention). Every step feeds
  the backbone via ``inputs_embeds``; its own ``embed_tokens`` table is never
  consumed. Because the backbone is a hybrid-Mamba model, the class implements
  vLLM's :class:`HasInnerState` / :class:`IsHybrid` /
  :class:`SupportsMambaPrefixCaching` contracts (mamba-state helpers are
  delegated to :class:`NemotronHForCausalLM`), and a SiLU shared-experts fix is
  applied at construction (see :mod:`easymagpie_vllm_omni.backbone_patches`).
* **Local transformer** — :class:`EasyMagpieCodePredictor`, a
  CUDA-graph-capturable implementation that runs as a single compiled graph.
* **compute_logits** — returns trivial logits so vLLM's sampler always picks
  index 0; the real audio output is the codes tensor surfaced through
  :meth:`make_omni_output` under the ``"audio_codes"`` key.

Text is embedded via a precomputed per-subword lookup table baked at
checkpoint-conversion time, so the char-aware subword encoder is never run
inside the engine.

Per-request I/O (via ``additional_information``):

* ``speaker_embedding`` (prefill only) — ``(T_audio, embedding_dim)``
  speaker-encoded context-audio embedding. ``preprocess`` assembles the full
  prefill context embedding itself as
  ``[task_embedding | speaker_embedding | context_text_embedded]``, so the
  caller only does the speaker-encoder math and passes plain context text (the
  model tokenizes + embeds it and prepends the per-mode service token).
* ``context_text`` (prefill only, optional) — plain conditioning string (e.g.
  ``"[EN]"``); tokenized in-model with the checkpoint's text tokenizer and
  embedded through the baked per-subword ``text_embedding`` table.
* ``task_mode_id`` (prefill only, optional) — int selecting the per-mode task
  ("service token") embedding row; defaults to ``0``. Ignored for single-mode
  checkpoints (no ``task_embedding`` table).

  The caller passes ``prompt_token_ids = [0] * T_ctx``, where ``T_ctx`` is the
  assembled context length (``[task?] + T_audio + len(tokenize(context_text))``).
* ``text`` (prefill only) — the plain target sentence to synthesize. This is the
  caller's text input: the model tokenizes it in-model at prefill with the
  checkpoint's text tokenizer (HF special tokens disabled, trailing text-EOS id
  appended), so callers never tokenize themselves. The resulting subword ids are
  consumed one per decode step (step ``k`` consumes id ``k``, embedded through
  the precomputed per-subword ``text_embedding`` table); once exhausted the text
  channel is masked off. (Internal: the tokenized ids are stashed as
  ``text_tokens`` in the per-request info dict between prefill and decode.)
* ``text_token`` (decode only, **streaming-text mode**) — when the caller omits
  ``text`` at prefill, the request runs in streaming-text mode: the caller pushes
  one subword id per decode step via ``additional_information`` under
  ``text_token`` (a single int / 1-element tensor), embedded through the same
  baked ``text_embedding`` table. This is the per-step counterpart to the whole
  ``text`` string and is driven by vLLM-Omni's streaming-input API (an async
  generator of ``StreamingInput`` chunks passed as the prompt, with
  ``async_chunk=True``). Push the text-EOS id as the last real token; on any step
  with no id (``text_token`` absent or ``< 0``, e.g. the sentinel ``-1``) the text
  channel is masked off so the caller can keep pumping decode steps while the
  audio tail finishes. Caller tokenization mirrors :meth:`_encode_text_stream`
  (``tokenizer.encode(text, add_special_tokens=False) + [text_eos_id]``).
* ``temperature`` / ``top_k`` (prefill only, optional) — audio sampling params
  for the local transformer. vLLM's ``SamplingParams.temperature`` drives only
  the dummy backbone token sampler, so the *audio* temperature/top-k are passed
  here and applied to the code predictor (defaults: ``0.7`` / ``80``).

Streaming delays: the text, phoneme and audio streams are temporally offset by
the checkpoint's ``streaming_phonemes_delay`` / ``streaming_speech_delay`` (baked
into ``config.json`` by the converter from the default inference mode). The text
stream runs from decode step 0; the phoneme stream opens at step
``phonemes_delay`` (seeded with phoneme BOS) and the audio stream at step
``speech_delay`` (seeded with audio BOS). The leading ``speech_delay`` decoded
frames are warm-up only and must be dropped by the caller. Delays of 0/0
reproduce a lock-step / non-delayed model.
"""
from __future__ import annotations

import bisect
from collections.abc import Callable, Iterable
from typing import Any, Optional

import torch
from torch import nn
from vllm.compilation.backends import set_model_tag
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    SupportsMambaPrefixCaching,
)
from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM, NemotronHModel
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from easymagpie_vllm_omni.backbone_patches import (
    patch_mamba_streaming_decode,
    patch_moe_routed_scale,
    patch_silu_shared_experts,
)
from easymagpie_vllm_omni.config import EasyMagpieOmniArch
from easymagpie_vllm_omni.local_transformer import EasyMagpieCodePredictor

logger = init_logger(__name__)

# Placeholder token id stuffed into the per-step ``input_ids`` returned by
# ``preprocess`` — the model never consumes ``input_ids`` (decode behaviour is
# driven by the per-token buffers), and ``compute_logits`` returns
# argmax-at-0 dummy logits, so this only needs to be a valid id.
_DUMMY_TOKEN_ID = 0

# Context text used when the request omits ``context_text``
_DEFAULT_CONTEXT_TEXT = "[EN]"

# This class is not wrapped in ``@support_torch_compile``: the Nemotron-H
# backbone and :class:`EasyMagpieCodePredictor` each manage their own
# ``torch.compile`` / CUDA-graph capture internally, so the outer ``forward``
# runs eagerly and dispatches into the two self-compiled subgraphs.
class EasyMagpieTTSForConditionalGeneration(
    nn.Module,
    HasInnerState,
    IsHybrid,
    SupportsMambaPrefixCaching,
):
    """EasyMagpieTTS talker for vLLM-Omni.

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
    have_multimodal_outputs: bool = True

    # Keep small per-step tensors GPU-resident across steps (no D2H/H2D).
    gpu_resident_buffer_keys: set[str] = {
        "last_audio_codes",
        "last_phoneme_token",
        "last_hidden",
    }

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

        # ── Backbone (reused vLLM Nemotron-H LM; fed via inputs_embeds) ──
        self.backbone = NemotronHModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "backbone"),
        )
        # The checkpoint was trained with mlp_hidden_act=silu but vLLM's
        # NemotronHMLP hard-codes ReLU² in shared_experts. Restore SiLU (no-op
        # when the backbone has no MoE layers).
        patch_silu_shared_experts(self.backbone)
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

        # ── Text + phoneme embedding heads ──────────────────────────────
        # Precomputed per-subword text embedding (one row per subword id), baked
        # at conversion time and fed additively on every decode step.
        text_vocab_size = int(getattr(hf_config, "text_vocab_size", getattr(hf_config, "vocab_size", 0)))
        self.text_embedding = nn.Embedding(text_vocab_size, self.embedding_dim)

        # Text-stream EOS id — the last-but-one row of the text vocab, matching
        # the reference ``EasyMagpieTTSInferenceModel.eos_id = num_tokens - 2``.
        # Appended to the in-model-tokenized target text stream (see
        # :meth:`_encode_text_stream`).
        self.text_eos_id = text_vocab_size - 2

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
                [nn.Embedding(arch.phoneme_vocab_size, self.embedding_dim) for _ in range(arch.phoneme_stacking_factor)]
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
            self._dec_phoneme_tokens = torch.zeros(
                max_num_tokens, arch.phoneme_stacking_factor, dtype=torch.long
            )
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

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.get_input_embeddings(input_ids)

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

        * **attention** layers carry standard metadata that exposes the
          batch-level ``max_query_len`` + ``query_start_loc`` (e.g.
          ``TritonAttentionMetadata``);
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

    def _get_decode_idxs(self):
        """Return ``(decode_token_indices, num_requests)`` for code-predictor dispatch.

        * ``(None, 0)`` → run the local transformer on every token (profile /
          dummy run with no ``attn_metadata``, or a decode-only batch where
          ``max_query_len == 1``), so the captured CUDA graph covers every
          ``cudagraph_capture_sizes`` value.
        * ``(indices, num_requests)`` → run only on the listed decode positions
          (mixed prefill+decode batch). ``indices`` is padded to the next
          captured graph size; ``num_requests`` is the unpadded count.
        """
        ctx = get_forward_context()
        attn_metadata = ctx.attn_metadata
        if attn_metadata is None:
            return None, 0

        max_query_len, start_loc = self._select_query_layout(attn_metadata)

        # Decode-only batch (or layout unavailable) -> run the LT on every token.
        if max_query_len is None or max_query_len == 1 or start_loc is None:
            return None, 0

        tokens_per_req = start_loc[1:] - start_loc[:-1]
        is_decode = tokens_per_req == 1
        decode_token_indices = start_loc[:-1][is_decode]

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
        return decode_token_indices, num_requests

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

        decode_idx, num_req = self._get_decode_idxs()

        if decode_idx is None:
            # Profile / dummy run or decode-only batch: assemble decode
            # embeddings everywhere so the captured graph sees the full path.
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

        # Re-index _token_stop into _sample_stop.
        # this only happens for mixed/prefill, since for capture logits_index is None,
        # so during decode-only the branch for logits_index is None will be executed.        
        if logits_index is not None:
            self._sample_stop[:logits_index.shape[0]] = self._token_stop[logits_index]
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

        The UNK replacement mirrors the reference: when the max phoneme
        probability of any stacked channel falls below
        ``phoneme_confidence_unk_threshold`` (and the step is not an EOS step),
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
        """Surface the sampled codes (``BT x num_codebooks``) under ``audio_codes``."""
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        hidden = model_outputs
        num_tokens = int(hidden.shape[0])
        audio_codes = self._out_codes[:num_tokens].clone()
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={"audio_codes": audio_codes},
        )

    # ------------------------------------------------------------------
    # preprocess / postprocess
    # ------------------------------------------------------------------

    @staticmethod
    def _first_str(value: Any) -> str:
        """Return the first element of a list-wrapped scalar, or the scalar itself, as a string."""
        if isinstance(value, list):
            return str(value[0]) if value else ""
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _coerce_opt_int(value: Any) -> Optional[int]:
        """Best-effort extract a single int from a scalar / list / tensor / str.

        Used to read a per-step streamed ``text_token`` out of the request's
        ``additional_information`` (which may wrap the id as a list, a 1-element
        tensor, or a string depending on how the caller / transport packed it).
        Returns ``None`` when no usable integer is present.
        """
        if value is None:
            return None
        if isinstance(value, bool):  # bool is an int subclass — handle explicitly.
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, torch.Tensor):
            return int(value.reshape(-1)[0].item()) if value.numel() > 0 else None
        if isinstance(value, (list, tuple)):
            return EasyMagpieTTSForConditionalGeneration._coerce_opt_int(value[0]) if value else None
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                return None
        return None

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

        Prefill (``span_len > 1``): assemble the full context embedding
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

        if span_len > 1:
            return self._preprocess_prefill(input_ids, span_len, device, info_dict)

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
        span_len: int,
        device: torch.device,
        info_dict: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        # Forward the audio (local-transformer) sampling params from the request.
        # vLLM's ``SamplingParams.temperature`` drives only the dummy backbone
        # token sampler, so the real audio temperature/top-k are passed via
        # ``additional_information`` and applied to the code predictor here (once,
        # at prefill — they are scalars that persist across decode steps).
        self._maybe_set_lt_sampling_params(info_dict)

        prefill_embeds = self._build_prefill_embeds(device, info_dict)

        offset = int(info_dict.get("prefill_offset", 0) or 0)
        total = int(prefill_embeds.shape[0])
        s = max(0, min(offset, total))
        e = max(0, min(offset + span_len, total))
        take = prefill_embeds[s:e]
        if int(take.shape[0]) < span_len:
            pad_n = span_len - int(take.shape[0])
            pad_rows = (
                take[-1:].expand(pad_n, -1)
                if take.shape[0] > 0
                else prefill_embeds.new_zeros(pad_n, prefill_embeds.shape[-1])
            )
            take = torch.cat([take, pad_rows], dim=0)

        info_update = {
            "prefill_offset": offset + span_len,
            "decode_offset": 0,
        }
        # Tokenize the caller's ``text`` in-model and stash the subword ids in the
        # per-request info dict (alongside the offsets) so each decode step
        # consumes one id from it without the caller ever running the tokenizer
        # (see :meth:`_preprocess_decode`). When the caller passes ``text`` whole
        # at prefill we bake the ``text_tokens`` list here; an already-present
        # ``text_tokens`` list is left untouched. When *neither* ``text`` nor
        # ``text_tokens`` is provided the request runs in **streaming-text mode**:
        # no list is baked, and :meth:`_preprocess_decode` instead reads one
        # subword id per step from the streamed ``additional_information.text_token``.
        if not info_dict.get("text_tokens"):
            text = self._first_str(info_dict.get("text"))
            if text:
                info_update["text_tokens"] = self._encode_text_stream(text)
        input_ids_out = torch.full_like(input_ids, _DUMMY_TOKEN_ID)
        return input_ids_out, take, info_update

    def _build_prefill_embeds(
        self,
        device: torch.device,
        info_dict: dict[str, Any],
    ) -> torch.Tensor:
        """Assemble the full ``(T_ctx, embedding_dim)`` prefill context embedding::

            [task_embedding | speaker_embedding | context_text_embedded]

        from the per-request inputs:

        * ``speaker_embedding`` — the speaker-encoded context-audio embedding,
          required as a 2-D ``(T_audio, embedding_dim)`` tensor.
        * ``context_text`` — a plain string (e.g. ``"[EN]"``); tokenized in-model
          (see :meth:`_encode_context_text`) and embedded through the baked
          per-subword ``text_embedding`` table.
        * ``task_mode_id`` — selects the per-mode task ("service token")
          embedding row; prepended only when the checkpoint has a task table.

        Returns the full context embedding; the per-chunk slicing/padding is done
        by :meth:`_preprocess_prefill`.
        """
        dtype = self._combined_embeddings.dtype

        speaker_embedding = info_dict.get("speaker_embedding")
        assert isinstance(speaker_embedding, torch.Tensor) and speaker_embedding.ndim == 2, (
            "EasyMagpieTTS preprocess expects additional_information.speaker_embedding to be a 2-D "
            "(T_audio, embedding_dim) tensor (the speaker-encoded context audio); "
            f"got {type(speaker_embedding).__name__}"
            + (f" with ndim={speaker_embedding.ndim}" if isinstance(speaker_embedding, torch.Tensor) else "")
        )

        parts: list[torch.Tensor] = []

        # Task / "service token" embedding (prepended), when present.
        if self.task_embedding is not None:
            task_mode_id = int(info_dict.get("task_mode_id", 0) or 0)
            task_mode_id = max(0, min(task_mode_id, self.num_task_embeddings - 1))
            task_row = self.task_embedding(torch.tensor([task_mode_id], device=device, dtype=torch.long))
            parts.append(task_row.to(dtype))

        # Speaker-encoded context audio.
        parts.append(speaker_embedding.to(device=device, dtype=dtype))

        # Context text: tokenized in-model and embedded through the baked table.
        context_text = self._first_str(info_dict.get("context_text")) or _DEFAULT_CONTEXT_TEXT
        ctx_ids = self._encode_context_text(context_text, device)
        if ctx_ids.numel() > 0:
            parts.append(self.text_embedding(ctx_ids).to(dtype))

        return torch.cat(parts, dim=0)

    def _maybe_set_lt_sampling_params(self, info_dict: dict[str, Any]) -> None:
        """Apply per-request audio sampling params to the local transformer.

        Reads ``temperature`` / ``top_k`` (alias ``topk``) from the request's
        ``additional_information`` and stores them on the code predictor. Absent
        keys leave the existing defaults untouched.
        """
        temperature = info_dict.get("temperature")
        if temperature is not None:
            self.code_predictor.temperature = float(self._first_str(temperature) or 0.0)
        top_k = info_dict.get("top_k", info_dict.get("topk"))
        if top_k is not None:
            self.code_predictor.top_k = int(float(self._first_str(top_k) or 0))

    def _get_text_tokenizer(self):
        """Lazily load the context-text tokenizer from the model directory.

        The converted checkpoint ships a HuggingFace ``AutoTokenizer`` (the
        model's text-conditioning tokenizer) alongside its weights, so we load it
        on first use from ``model_path``.
        """
        if self._text_tokenizer is None:
            from transformers import AutoTokenizer

            self._text_tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        return self._text_tokenizer

    def _encode_context_text(self, context_text: str, device: torch.device) -> torch.Tensor:
        """Tokenize ``context_text`` to subword ids.

        The text-conditioning tokenizer sits at offset 0 in the model's
        tokenizer aggregate, so its raw ids index the baked ``text_embedding``
        table directly.
        """
        tok = self._get_text_tokenizer()
        ids = tok.encode(context_text)
        return torch.tensor(ids, device=device, dtype=torch.long)

    def _encode_text_stream(self, text: str) -> list[int]:
        """Tokenize the target ``text`` into the streaming subword-id list.

        Mirrors the reference ``tokenizer.encode(transcript) + [eos_id]``: HF
        special tokens are disabled so the raw ids index the baked
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
        """Length-only mirror of :meth:`_build_prefill_embeds`.

        The engine assembles the prefill context as
        ``[task_embedding? | speaker_embedding | context_text_embedded]``, so the
        caller must pass ``prompt_token_ids = [0] * estimate_prompt_len(...)`` for
        the placeholder length to match the assembled embedding length (otherwise
        vLLM pads / truncates and quality drops).

        Args:
            speaker_embedding: ``(T_audio, embedding_dim)`` speaker-encoded
                context-audio embedding (only its length is used).
            tokenize: callable turning ``context_text`` into its subword ids
                (e.g. ``lambda t: tokenizer.encode(t)``) — must match the
                tokenizer the engine loads from ``model_path``.
            context_text: conditioning string (default ``"[NO TEXT CONTEXT]"``).
            has_task_embedding: whether the checkpoint prepends a task /
                "service token" embedding (``num_task_embeddings > 0``).
        """
        t_audio = int(speaker_embedding.shape[0])
        ctx_len = len(list(tokenize(context_text or _DEFAULT_CONTEXT_TEXT)))
        task_len = 1 if has_task_embedding else 0
        return task_len + t_audio + ctx_len

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
        # stream leads the phoneme/audio streams by their respective delays. Two
        # mutually exclusive input modes are supported:
        #
        # * **Whole-text (non-streaming)** — the caller passed ``text`` whole at
        #   prefill; it was tokenized in-model and stashed as the ``text_tokens``
        #   list (see :meth:`_preprocess_prefill`). Step k consumes
        #   ``text_tokens[k]`` (the list ends with the text-EOS id); once the
        #   stream is exhausted the channel is masked off (adds nothing) rather
        #   than repeating the last token.
        # * **Streamed** — the caller did *not* pass ``text`` at prefill and
        #   instead pushes one subword id per decode step via
        #   ``additional_information`` under ``text_token`` (a single int / 1-elem
        #   tensor; close the stream by pushing the text-EOS id as the last real
        #   token). The model embeds that step's id and masks the channel off on
        #   any step that carries no id (``text_token`` absent or ``< 0``), so the
        #   caller can keep pumping decode steps after the text ends while the
        #   audio tail finishes. Because each streamed chunk overwrites the
        #   previous ``text_token`` in the per-request buffer, every step gets a
        #   fresh value (or the caller's sentinel ``-1`` to mask).
        text_tokens = info_dict.get("text_tokens")
        if isinstance(text_tokens, list):
            if decode_offset < len(text_tokens):
                self._dec_text_tokens[start] = int(text_tokens[decode_offset])
                self._dec_text_mask[start] = 1
            else:
                self._dec_text_mask[start] = 0
        else:
            streamed_id = self._coerce_opt_int(info_dict.get("text_token"))
            if streamed_id is not None and streamed_id >= 0:
                self._dec_text_tokens[start] = streamed_id
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
        audio_codes = (multimodal_outputs or {}).get("audio_codes")
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
        own_params = dict(self.named_parameters())
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
