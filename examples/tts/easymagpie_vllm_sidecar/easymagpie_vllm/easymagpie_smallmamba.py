"""EasyMagpie SmallMamba TTS model for vLLM.

Composition over inheritance:

* ``self.backbone`` — vLLM's ``NemotronHModel`` (hybrid Mamba2+Attention+MoE),
  same weights as production.
* ``self.streaming_head`` — :class:`EasyMagpieStreamingHead` holding the TTS
  submodules and per-step streaming math.
* ``self.lm_head`` — real ``ParallelLMHead``; ``compute_logits`` returns its
  output with a +1e9 boost at ``VIRTUAL_EOS_TOKEN_ID`` when ``audio_finished``
  flips, so vLLM's standard sampler terminates the request cleanly.
* ``self._request_state`` — current request's :class:`StreamingHeadState`
  (or a dummy state during warmup capture).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    SupportsMambaPrefixCaching,
)
from vllm.model_executor.models.nemotron_h import (
    NemotronHForCausalLM,
    NemotronHModel,
)
from vllm.model_executor.models.utils import maybe_prefix

from .backbone_patches import patch_silu_shared_experts
from .streaming_head import (
    SMALLMAMBA,
    EasyMagpieStreamingHead,
    StreamingHeadState,
)


logger = logging.getLogger(__name__)


class EasyMagpieSmallMamba(
    nn.Module,
    HasInnerState,
    IsHybrid,
    SupportsMambaPrefixCaching,
):
    """vLLM-resident SmallMamba TTS model.

    Loaded by the plugin under both ``EasyMagpieSmallMamba`` and the legacy
    ``EasyMagpieSmallMambaV2`` arch names so existing checkpoints resolve.
    """

    # vLLM expects these as class attributes for hybrid Mamba bookkeeping.
    get_mamba_state_dtype_from_config = NemotronHForCausalLM.get_mamba_state_dtype_from_config
    get_mamba_state_shape_from_config = NemotronHForCausalLM.get_mamba_state_shape_from_config
    get_mamba_state_copy_func = NemotronHForCausalLM.get_mamba_state_copy_func

    # The sidecar passes this in ``SamplingParams.stop_token_ids`` so the
    # standard vLLM sampler terminates when ``compute_logits`` boosts it.
    VIRTUAL_EOS_TOKEN_ID: int = 0
    uses_sampler_eos: bool = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        """Construct backbone + streaming head + lm_head.

        Args:
            vllm_config: vLLM-supplied config.
            prefix: optional module-prefix string (vLLM convention).
        """
        super().__init__()
        self._model_dir = vllm_config.model_config.model
        self._vllm_config = vllm_config

        self.backbone = NemotronHModel(
            vllm_config=vllm_config,
            prefix=f"{prefix}.backbone" if prefix else "backbone",
        )
        # SmallMamba was trained with mlp_hidden_act=silu but vLLM's NemotronHMLP
        # hard-codes ReLU^2 in shared_experts. Restore SiLU.
        patch_silu_shared_experts(self.backbone)

        self.streaming_head = EasyMagpieStreamingHead()

        config = vllm_config.model_config.hf_config
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

        self._request_state: Optional[StreamingHeadState] = None
        # Pre-allocated state slot reused across requests for address stability.
        self._state_slot: Optional[StreamingHeadState] = None
        self._last_hidden_state: Optional[torch.Tensor] = None
        # Lazily-built (vocab_size,) boost vector — +1e9 at VIRTUAL_EOS_TOKEN_ID.
        self._eos_boost: Optional[torch.Tensor] = None

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        """Backbone forward + TTS-head predict in one call.

        At vLLM warmup ``self._request_state`` is ``None``; this method installs
        a dummy state so the streaming-head ops trace into the compiled graph.

        Args:
            input_ids: ``(N,)`` int64 token ids or ``None`` when using ``inputs_embeds``.
            positions: ``(N,)`` int64 absolute positions (vLLM-supplied).
            intermediate_tensors: vLLM passthrough (PP / unused here).
            inputs_embeds: optional ``(N, hidden)`` precomputed embeddings.

        Returns:
            ``(N, hidden)`` backbone hidden states.
        """
        if self._request_state is None:
            self._install_dummy_state_for_capture()

        hidden_states = self.backbone(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        self._last_hidden_state = hidden_states.detach()
        self.streaming_head.predict_codes_and_advance(hidden_states, self._request_state)
        return hidden_states

    @torch._dynamo.disable
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Per-step embedding hook for the AR decode loop.

        During prefill (no request state): trivial backbone lookup.
        During decode: build the additive ``text + phoneme + audio`` embedding
        from streaming state via :meth:`EasyMagpieStreamingHead.next_input_embedding`.

        Args:
            input_ids: ``(N,)`` int64 vLLM-sampled token ids (ignored in decode —
                the streaming state machine drives the embedding, not the sampler).

        Returns:
            ``(N, hidden)`` input embeddings.
        """
        state = self._request_state
        if state is None:
            return self.backbone.embed_input_ids(input_ids)
        device = input_ids.device
        dtype = next(self.backbone.parameters()).dtype
        return self.streaming_head.next_input_embedding(
            state, device, dtype, SMALLMAMBA["hidden_dim"],
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """LM-head logits, with a virtual-EOS boost when audio is finished.

        ``audio_finished_dev`` is an on-device tensor in ``{0.0, 1.0}`` updated
        by :meth:`EasyMagpieStreamingHead.predict_codes_and_advance`. When it
        flips to 1.0, this method adds +1e9 to ``logits[..., VIRTUAL_EOS_TOKEN_ID]``,
        the sampler picks that token, and vLLM stops the request via
        ``stop_token_ids``. No GPU→CPU sync needed.

        Args:
            hidden_states: ``(N, hidden)`` backbone output.

        Returns:
            ``(N, vocab_size)`` logits.
        """
        logits = self.logits_processor(self.lm_head, hidden_states)
        state = self._request_state
        if state is None:
            return logits

        if (self._eos_boost is None
                or self._eos_boost.device != logits.device
                or self._eos_boost.dtype != logits.dtype):
            vocab = logits.shape[-1]
            t = torch.zeros(vocab, dtype=logits.dtype, device=logits.device)
            t[self.VIRTUAL_EOS_TOKEN_ID] = 1e9
            self._eos_boost = t

        return logits + state.audio_finished_dev * self._eos_boost

    def init_streaming_request(
        self,
        *,
        audio_bos_id: int,
        audio_eos_id: int,
        phoneme_bos_id: int,
        phoneme_eos_id: int,
        streaming_speech_delay: int,
        streaming_phonemes_delay: int,
        phoneme_token_ids: list,
        text_eos_id: int,
        lt_temperature: float = 0.7,
        lt_topk: int = 80,
        phoneme_unk_token_id: int = 0,
        phoneme_confidence_unk_threshold: float = 0.0,
    ) -> None:
        """Prepare ``self._request_state`` for a new request.

        Reuses ``self._state_slot`` (from a previous request or warmup dummy)
        via ``reset_in_place`` so tensor addresses stay stable across requests
        — required for CUDA-graph replay.

        Args:
            audio_bos_id / audio_eos_id: audio codebook BOS / EOS ids.
            phoneme_bos_id / phoneme_eos_id: phoneme BOS / EOS ids.
            streaming_speech_delay / streaming_phonemes_delay: post-context delays.
            phoneme_token_ids: per-step CAS subword ids.
            text_eos_id: subword tokenizer EOS id.
            lt_temperature / lt_topk: LT sampling knobs.
            phoneme_unk_token_id / phoneme_confidence_unk_threshold: UNK config.
        """
        fields = dict(
            audio_bos_id=audio_bos_id,
            audio_eos_id=audio_eos_id,
            phoneme_bos_id=phoneme_bos_id,
            phoneme_eos_id=phoneme_eos_id,
            speech_delay=streaming_speech_delay,
            phonemes_delay=streaming_phonemes_delay,
            text_eos_id=text_eos_id,
            lt_temperature=lt_temperature,
            lt_topk=lt_topk,
            phoneme_unk_token_id=phoneme_unk_token_id,
            phoneme_confidence_unk_threshold=phoneme_confidence_unk_threshold,
            phoneme_ids=list(phoneme_token_ids),
        )
        if self._state_slot is None:
            device = next(self.streaming_head.parameters()).device
            self._state_slot = StreamingHeadState(device=device, **fields)
        else:
            self._state_slot.reset_in_place(**fields)
        self._request_state = self._state_slot

    def _install_dummy_state_for_capture(self) -> None:
        """Install a no-op state so vLLM's warmup traces the TTS path.

        Idempotent. Auto-fired by :meth:`forward` on the first call with
        ``_request_state=None``.
        """
        if self._state_slot is not None:
            self._request_state = self._state_slot
            return
        device = next(self.streaming_head.parameters()).device
        self._state_slot = StreamingHeadState.make_dummy(device=device)
        self._request_state = self._state_slot

    def wire_tokenizer(
        self,
        subword_vocab: dict,
        bos_id: int,
        eos_id: int,
        cfg_unk_token_id: int,
        subword_padding_idx: Optional[int] = None,
    ) -> None:
        """Delegate to :meth:`EasyMagpieStreamingHead.wire_tokenizer`."""
        self.streaming_head.wire_tokenizer(
            subword_vocab=subword_vocab,
            bos_id=bos_id,
            eos_id=eos_id,
            cfg_unk_token_id=cfg_unk_token_id,
            subword_padding_idx=subword_padding_idx,
        )

    def warmup_compiled_lt(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        iters: int = 3,
    ) -> None:
        """Delegate to :meth:`EasyMagpieStreamingHead.warmup_compiled_lt`."""
        self.streaming_head.warmup_compiled_lt(device=device, dtype=dtype, iters=iters)

    def embed_audio_tokens(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        """Delegate to :meth:`EasyMagpieStreamingHead.embed_audio_tokens`."""
        return self.streaming_head.embed_audio_tokens(audio_tokens)

    def embed_text_tokens(
        self, text_tokens: torch.Tensor, text_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Delegate to :meth:`EasyMagpieStreamingHead.embed_text_tokens`."""
        return self.streaming_head.embed_text_tokens(text_tokens, text_lens)

    def embed_phoneme_tokens(self, phoneme_tokens: torch.Tensor) -> torch.Tensor:
        """Delegate to :meth:`EasyMagpieStreamingHead.embed_phoneme_tokens`."""
        return self.streaming_head.embed_phoneme_tokens(phoneme_tokens)

    def encode_speaker(
        self, audio_embedded: torch.Tensor, audio_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Delegate to :meth:`EasyMagpieStreamingHead.encode_speaker`."""
        return self.streaming_head.encode_speaker(audio_embedded, audio_lens)

    def build_prefill_combined_embeddings(
        self,
        context_audio_codes: torch.Tensor,
        context_audio_codes_lens: torch.Tensor,
        context_text_tokens: torch.Tensor,
        context_text_tokens_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build ``[speaker_encoder(audio) | cas_encoder(text)]`` prefill.

        Args:
            context_audio_codes: ``(B, n_tables, T_ctx)`` int64 audio codes.
            context_audio_codes_lens: ``(B,)`` int64 valid lengths.
            context_text_tokens: ``(B, L_text)`` int64 subword ids.
            context_text_tokens_lens: ``(B,)`` int64 valid lengths.

        Returns:
            Tuple of:
              * combined embedding ``(B, T_ctx + L_text, embedding_dim)``.
              * combined lengths ``(B,)``.
        """
        audio_emb = self.streaming_head.embed_audio_tokens(context_audio_codes)
        audio_emb = self.streaming_head.encode_speaker(audio_emb, context_audio_codes_lens)
        text_emb = self.streaming_head.embed_text_tokens(
            context_text_tokens, context_text_tokens_lens,
        )
        ctx_embedding = torch.cat([audio_emb, text_emb], dim=1)
        ctx_lens = context_audio_codes_lens + context_text_tokens_lens
        return ctx_embedding, ctx_lens

    def load_weights(self, weights) -> set:
        """Load backbone + ``lm_head`` from safetensors, then ``tts_extras.pt``.

        Args:
            weights: iterable of ``(name, tensor)`` from vLLM's safetensors loader.

        Returns:
            Set of fully-qualified parameter names that were loaded.
        """
        loaded_lm_head: set[str] = set()

        def _rewrite(weights_iter):
            for name, tensor in weights_iter:
                if name == "lm_head.weight":
                    self.lm_head.weight.data.copy_(tensor.to(self.lm_head.weight.dtype))
                    loaded_lm_head.add("lm_head.weight")
                    continue
                if name.startswith("model."):
                    yield name[len("model."):], tensor
                else:
                    yield name, tensor

        loaded_backbone = self.backbone.load_weights(_rewrite(weights))
        loaded_backbone = {"backbone." + n for n in loaded_backbone}
        loaded_backbone |= loaded_lm_head

        tts_extras_path = os.path.join(self._model_dir, "tts_extras.pt")
        if os.path.isfile(tts_extras_path):
            loaded_tts = self.load_tts_extras(tts_extras_path)
        else:
            logger.warning(
                "tts_extras.pt not found at %s; TTS submodules left uninitialized",
                tts_extras_path,
            )
            loaded_tts = set()

        return loaded_backbone | set(loaded_tts)

    def load_tts_extras(self, tts_extras_path: str) -> set[str]:
        """Load TTS submodule weights from ``tts_extras.pt`` under ``streaming_head.``.

        Args:
            tts_extras_path: filesystem path to the ``tts_extras.pt`` checkpoint.

        Returns:
            Set of parameter names loaded (already ``streaming_head.``-prefixed).
        """
        tensors: dict[str, torch.Tensor] = torch.load(
            tts_extras_path, map_location="cpu", weights_only=True,
        )
        logger.info("Loading %d TTS-extras tensors from %s", len(tensors), tts_extras_path)

        own = dict(self.named_parameters())
        own_state = dict(self.state_dict())
        loaded: set[str] = set()
        unmatched: list[str] = []

        for k, v in tensors.items():
            mapped = "streaming_head." + k
            # CAS-encoder embed_tokens is sized lazily to match the trained vocab.
            if k == "cas_encoder.embed_tokens.weight":
                cur = own.get(mapped)
                if cur is not None and cur.shape != v.shape:
                    new = nn.Embedding(v.shape[0], v.shape[1])
                    self.streaming_head.cas_encoder.embed_tokens = new.to(cur.device)
                    own = dict(self.named_parameters())
                    own_state = dict(self.state_dict())

            target = own_state.get(mapped)
            if target is None:
                unmatched.append(k)
                continue
            if target.shape != v.shape:
                raise RuntimeError(
                    f"TTS extras shape mismatch at {mapped!r}: "
                    f"ckpt {tuple(v.shape)} vs model {tuple(target.shape)}"
                )
            target.data.copy_(v.to(target.dtype))
            loaded.add(mapped)

        if unmatched:
            logger.warning("%d TTS-extras keys unmatched: %s", len(unmatched), unmatched[:5])
        logger.info("TTS extras: loaded %d / %d", len(loaded), len(tensors))
        return loaded
