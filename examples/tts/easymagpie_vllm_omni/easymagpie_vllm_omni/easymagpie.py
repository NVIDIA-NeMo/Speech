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

EasyMagpieTTS is a decoder-only streaming TTS model: a text-LM backbone (the
reference checkpoint uses Qwen2.5-1.5B) consumes a per-frame additive input
embedding (text + phoneme + audio) and emits a per-frame hidden state, from
which a small autoregressive *local transformer* samples all ``C * S`` stacked
audio codebooks for that frame (see :mod:`easymagpie_vllm_omni.local_transformer`).

This module wires that architecture into vLLM-Omni's
``preprocess`` / ``forward`` / ``compute_logits`` / ``make_omni_output`` /
``postprocess`` contract, following the same conventions as the upstream
qwen3-tts and eartts vLLM-Omni model definitions:

* **Backbone** — vLLM's :class:`~vllm.model_executor.models.qwen2.Qwen2Model`,
  reused wholesale (KV cache + paged attention) the same way the EasyMagpie
  vLLM *sidecar* reuses ``NemotronHModel``. Every step feeds the backbone via
  ``inputs_embeds``; its own ``embed_tokens`` table is never consumed.
* **Local transformer** — :class:`EasyMagpieCodePredictor`, a from-scratch,
  CUDA-graph-capturable re-implementation that runs as a single compiled graph.
* **compute_logits** — returns trivial logits (à la eartts) so vLLM's sampler
  always picks index 0; the real audio output is the codes tensor surfaced
  through :meth:`make_omni_output` under the ``"audio_codes"`` key.

Text is embedded via a precomputed per-subword lookup table baked at
checkpoint-conversion time (the reference char-aware subword encoder is
deterministic per subword id, so it is never run inside the engine).

Per-request I/O (via ``additional_information``):

* ``prompt_embeds`` (prefill only) — ``(T_ctx, embedding_dim)`` precomputed
  context/prompt embedding (speaker-encoded context audio + context text)
  produced by the caller, exactly like qwen3-tts ``talker_prompt_embeds`` /
  eartts ``speaker_latent``. The user passes ``prompt_token_ids = [0] * T_ctx``.
* ``text_tokens`` — Python ``list[int]`` of subword ids that grows by one per
  decode step; step ``k`` consumes ``text_tokens[k]`` (embedded through the
  precomputed per-subword table).
* ``phoneme_tokens`` (optional) — same streaming-list contract for the phoneme
  channel; if omitted the phoneme branch is skipped.
"""
from __future__ import annotations

import bisect
from collections.abc import Iterable
from typing import Any, Optional

import torch
from torch import nn
from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import ignore_torch_compile, support_torch_compile
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from easymagpie_vllm_omni.config import EasyMagpieOmniArch
from easymagpie_vllm_omni.local_transformer import EasyMagpieCodePredictor

logger = init_logger(__name__)

# Placeholder token id stuffed into the per-step ``input_ids`` returned by
# ``preprocess`` — the model never consumes ``input_ids`` (decode behaviour is
# driven by the per-token buffers), and ``compute_logits`` returns
# argmax-at-0 dummy logits, so this only needs to be a valid id.
_DUMMY_TOKEN_ID = 0


# ``dynamic_arg_dims`` is passed explicitly: this file uses
# ``from __future__ import annotations`` (PEP 563), so ``forward``'s annotations
# are strings and vLLM's annotation-based inference would fail with
# "No dynamic dimensions found...". These mirror vLLM's default inference
# (dim 0 for every tensor / IntermediateTensors argument).
@ignore_torch_compile
@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class EasyMagpieTTSForConditionalGeneration(nn.Module):
    """EasyMagpieTTS talker for vLLM-Omni.

    See the module docstring for the per-step flow and the per-request I/O
    contract. The class exposes the omni hooks (``has_preprocess`` /
    ``has_postprocess`` / ``have_multimodal_outputs``) consumed by the
    ``OmniGPUModelRunner``.
    """

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

        # ── Backbone (reused vLLM text LM; fed via inputs_embeds) ───────
        self.backbone = Qwen2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "backbone"),
        )

        # ── Local transformer (its own compile group / CUDA graph) ──────
        with set_model_tag("local_transformer"):
            self.code_predictor = EasyMagpieCodePredictor(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "code_predictor"),
            )

        # ── Text + phoneme embedding heads ──────────────────────────────
        # Precomputed per-subword text embedding. The reference model embeds
        # text with a char-aware subword (CAS) encoder + the decoder's subword
        # table; both are deterministic per subword id, so the checkpoint
        # converter bakes their combined result into this single lookup table
        # (one row per subword id). It is fed additively on every decode step;
        # the CAS encoder is never run inside the engine.
        text_vocab_size = int(getattr(hf_config, "text_vocab_size", getattr(hf_config, "vocab_size", 0)))
        self.text_embedding = nn.Embedding(text_vocab_size, self.embedding_dim)

        # Phoneme channel (optional — only built when the checkpoint has one).
        self.has_phoneme = arch.phoneme_vocab_size > 0 and arch.phoneme_stacking_factor > 0
        if self.has_phoneme:
            self.phoneme_embeddings = nn.ModuleList(
                [nn.Embedding(arch.phoneme_vocab_size, self.embedding_dim) for _ in range(arch.phoneme_stacking_factor)]
            )
            self.phoneme_final_proj = nn.Linear(
                self.hidden_dim, arch.phoneme_vocab_size * arch.phoneme_stacking_factor
            )

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

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

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

    def _get_decode_idxs(self):
        """Return ``(decode_token_indices, num_requests)`` for code-predictor dispatch.

        Mirrors the qwen3-tts / eartts pattern:

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

        if isinstance(attn_metadata, dict):
            any_layer_meta = next(iter(attn_metadata.values()))
        else:
            any_layer_meta = attn_metadata

        if any_layer_meta.max_query_len == 1:
            return None, 0

        start_loc = any_layer_meta.query_start_loc
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
        **_: Any,
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

        decode_idx, num_req = self._get_decode_idxs()

        if decode_idx is None:
            # Profile / dummy run or decode-only batch: assemble decode
            # embeddings everywhere so the captured graph sees the full path.
            self._assemble_decode_embeddings(combined, slice(0, num_tokens))
        elif num_req > 0:
            valid = decode_idx[:num_req]
            self._assemble_decode_embeddings(combined, valid)

        hidden_states = self.backbone(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=combined,
        )

        # Sample codes (local transformer) only where needed.
        if decode_idx is None:
            codes = self.code_predictor.generate_codes(hidden_states)
            self._out_codes[:num_tokens].copy_(codes)
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
            if self.has_phoneme:
                self._predict_phonemes(hidden_states, valid)

        return hidden_states

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
        """Argmax the phoneme head and stash the prediction for the next step."""
        logits = self.phoneme_final_proj(hidden_states[idx].float())
        s = self.arch.phoneme_stacking_factor
        logits = logits.view(-1, s, self.arch.phoneme_vocab_size)
        self._dec_phoneme_tokens[idx] = logits.argmax(dim=-1).long()
        self._dec_phoneme_valid[idx] = 1

    # ------------------------------------------------------------------
    # compute_logits — dummy (real output is the codes tensor)
    # ------------------------------------------------------------------

    def compute_logits(self, hidden_states, sampling_metadata: Any = None) -> Optional[torch.Tensor]:
        """Return zero logits so vLLM's sampler always picks index 0.

        The width is taken from ``hf_config.vocab_size`` so the sampler's
        working buffers match. The sampled id is irrelevant — audio is surfaced
        via :meth:`make_omni_output`.
        """
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        batch_size = hidden_states.shape[0]
        return hidden_states.new_zeros(batch_size, int(self.hf_config.vocab_size))

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
    def _unwrap(value: Any) -> Any:
        if isinstance(value, list):
            return value[0] if value else None
        return value

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

        Prefill (``span_len > 1``): slice the precomputed ``prompt_embeds``
        context embedding into this chunk and return it; ``input_ids`` are
        placeholders. Decode (``span_len == 1``): write the per-token decode
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
        return self._preprocess_decode(input_ids, start, device, info_dict)

    def _preprocess_prefill(
        self,
        input_ids: torch.Tensor,
        span_len: int,
        device: torch.device,
        info_dict: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        prompt_embeds = self._unwrap(info_dict.get("prompt_embeds"))
        if not isinstance(prompt_embeds, torch.Tensor) or prompt_embeds.ndim != 2:
            raise ValueError(
                "EasyMagpieTTS preprocess requires additional_information.prompt_embeds "
                "of shape (T_ctx, embedding_dim) for prefill."
            )
        prompt_embeds = prompt_embeds.to(device=device, dtype=self._combined_embeddings.dtype)

        offset = int(info_dict.get("ear_prefill_offset", 0) or 0)
        total = int(prompt_embeds.shape[0])
        s = max(0, min(offset, total))
        e = max(0, min(offset + span_len, total))
        take = prompt_embeds[s:e]
        if int(take.shape[0]) < span_len:
            pad_n = span_len - int(take.shape[0])
            pad_rows = (
                take[-1:].expand(pad_n, -1)
                if take.shape[0] > 0
                else prompt_embeds.new_zeros(pad_n, prompt_embeds.shape[-1])
            )
            take = torch.cat([take, pad_rows], dim=0)

        info_update = {
            "ear_prefill_offset": offset + span_len,
            "ear_decode_offset": 0,
        }
        input_ids_out = torch.full_like(input_ids, _DUMMY_TOKEN_ID)
        return input_ids_out, take, info_update

    def _preprocess_decode(
        self,
        input_ids: torch.Tensor,
        start: int,
        device: torch.device,
        info_dict: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        decode_offset = int(info_dict.get("ear_decode_offset", 0) or 0)

        # Text channel (streaming list that grows by one per step).
        text_tokens = info_dict.get("text_tokens")
        if isinstance(text_tokens, list) and text_tokens:
            idx = min(decode_offset, len(text_tokens) - 1)
            self._dec_text_tokens[start] = int(text_tokens[idx])
            self._dec_text_mask[start] = 1
        else:
            self._dec_text_mask[start] = 0

        # Phoneme channel: previous-step prediction stashed by postprocess.
        if self.has_phoneme:
            last_phon = info_dict.get("last_phoneme_token")
            if isinstance(last_phon, torch.Tensor) and last_phon.numel() > 0:
                p = last_phon.to(device=device, dtype=torch.long).reshape(-1)
                self._dec_phoneme_tokens[start, : p.shape[0]].copy_(p[: self.arch.phoneme_stacking_factor])
                self._dec_phoneme_valid[start] = 1
            else:
                self._dec_phoneme_valid[start] = 0

        # Audio channel: previous-frame codes (BOS seed on the first step).
        last_codes = info_dict.get("last_audio_codes")
        if isinstance(last_codes, torch.Tensor) and last_codes.numel() > 0:
            c = last_codes.to(device=device, dtype=torch.long).reshape(-1)[: self.num_codebooks]
            self._dec_audio_codes[start, : c.shape[0]].copy_(c)
            self._dec_audio_valid[start] = 1
        else:
            # First decode step after prefill: seed with audio BOS.
            self._dec_audio_codes[start].fill_(self.arch.audio_bos_id)
            self._dec_audio_valid[start] = 1

        inputs_embeds_out = torch.zeros((1, self.embedding_dim), device=device, dtype=self._combined_embeddings.dtype)
        info_update = {"ear_decode_offset": decode_offset + 1}
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

    # Checkpoint prefixes (reference EasyMagpieTTS state dict) → in-model paths.
    # ``decoder.*`` is fed to the vLLM backbone loader separately (it understands
    # HF Qwen2 naming + qkv packing). The TTS submodules are copied manually.
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
    }

    def _remap_tts_key(self, name: str) -> Optional[str]:
        """Map a raw checkpoint key to its in-model parameter path (or ``None``)."""
        for src, dst in self._TTS_PREFIX_MAP.items():
            if name.startswith(src):
                return dst + name[len(src) :]
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load backbone (Qwen2) + TTS submodule weights from a converted checkpoint.

        The converted checkpoint is expected to use the reference EasyMagpieTTS
        key layout: the backbone under ``decoder.*`` (HF Qwen2 names) and the
        TTS submodules at top level (``audio_embeddings.*``, ``local_transformer.*``,
        ``phoneme_*``, ``text_embedding.*``, projection heads). Backbone weights
        are routed to :meth:`Qwen2Model.load_weights` (which packs qkv / gate-up
        and handles HF naming); TTS weights are copied directly by name.
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
            target = own_params.get(mapped)
            if target is None:
                logger.warning("EasyMagpieTTS: no parameter for checkpoint key %s -> %s", name, mapped)
                continue
            if target.shape != tensor.shape:
                raise RuntimeError(
                    f"EasyMagpieTTS weight shape mismatch at {mapped!r}: "
                    f"ckpt {tuple(tensor.shape)} vs model {tuple(target.shape)}"
                )
            with torch.no_grad():
                target.data.copy_(tensor.to(target.dtype))
            loaded.add(mapped)

        backbone_loaded = self.backbone.load_weights(backbone_weights)
        loaded |= {f"backbone.{n}" for n in backbone_loaded}

        # Derived runtime state.
        self.code_predictor.init_forbidden_mask()

        # The backbone's vestigial embed_tokens table is never consumed
        # (everything goes through inputs_embeds); don't flag it as missing.
        loaded.add("backbone.embed_tokens.weight")

        logger.info("Loaded %d weights for EasyMagpieTTSForConditionalGeneration", len(loaded))
        return loaded
