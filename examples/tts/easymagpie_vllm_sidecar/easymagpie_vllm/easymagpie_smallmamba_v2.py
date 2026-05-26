"""EasyMagpie SmallMamba — composition-based V2.

Top-level thin coordinator. Three composed pieces:

  * ``self.backbone``       — ``NemotronHModel`` (vLLM's standard hybrid
                              Mamba2+Attention+MoE backbone). Same code, same
                              kernels, same weights as production.
  * ``self.streaming_head`` — ``EasyMagpieStreamingHead`` holding the TTS
                              submodules (audio_embeddings, cas_encoder,
                              speaker_encoder, local_transformer + projections,
                              phoneme heads) AND the per-step state machine.
  * ``self.lm_head``        — ``ParallelLMHead`` (real logits — Phase 3 added
                              this so vLLM's sampler can EOS-stop via
                              ``VIRTUAL_EOS_TOKEN_ID``).
  * ``self._request_state`` — current request's ``StreamingHeadState``
                              (``None`` outside a request).

Sidecar API surface (``easymagpie_server/server.py`` calls these on the
model via ``apply_model``):
  ``init_streaming_request``, ``wire_tokenizer``, ``warmup_compiled_lt``,
  ``build_prefill_combined_embeddings``, ``embed_audio_tokens``,
  ``embed_text_tokens``, ``encode_speaker``, ``_streaming_codes``.

Quirks worth knowing:
  * ``backbone.layers.0.mixer.A`` is fp32 by vLLM convention (loader applies
    ``-exp(A_log)``). Most other backbone params are bf16. TTS submodules
    are fp32 at rest (explicit ``torch.set_default_dtype(fp32)`` during
    construction — see streaming_head.py).
  * Per-request streaming state lives in a dataclass on
    ``self._request_state``. Each per-step hook (`embed_input_ids`,
    `forward`, `compute_logits`) is ``@torch._dynamo.disable``-d so the
    Python state machine isn't traced/specialized by torch.compile.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from torch import nn

from vllm.model_executor.models.nemotron_h import (
    NemotronHModel,
    NemotronHForCausalLM,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    SupportsMambaPrefixCaching,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.utils import maybe_prefix
from vllm.config import VllmConfig

from .backbone_patches import patch_silu_shared_experts
from .streaming_head import EasyMagpieStreamingHead, StreamingHeadState


logger = logging.getLogger(__name__)


SMALLMAMBA = dict(
    hidden_dim=1536,
    embedding_dim=1536,
    audio_embedding_dim=1536,
    num_audio_codebooks=8,
    num_all_tokens_per_codebook=1032,
    frame_stacking_factor=2,
    phoneme_stacking_factor=1,
    phoneme_vocab_size=2051,
)


class EasyMagpieSmallMambaV2(
    nn.Module,
    HasInnerState,
    IsHybrid,
    SupportsMambaPrefixCaching,
):
    # vLLM-side classmethod contract (see Phase 1 notes).
    get_mamba_state_dtype_from_config = (
        NemotronHForCausalLM.get_mamba_state_dtype_from_config
    )
    get_mamba_state_shape_from_config = (
        NemotronHForCausalLM.get_mamba_state_shape_from_config
    )
    get_mamba_state_copy_func = (
        NemotronHForCausalLM.get_mamba_state_copy_func
    )

    # Phase 3: V2 uses the standard vLLM sampler path for EOS.
    # The sidecar checks this attr at startup to decide between the legacy
    # ``ignore_eos=True + engine.abort_request`` flow (V1) and the new
    # ``ignore_eos=False + stop_token_ids=[VIRTUAL_EOS_TOKEN_ID]`` flow (V2).
    uses_sampler_eos: bool = True

    # Virtual EOS token id we boost when streaming state's audio_finished
    # flips True. Picked at 0 (safely inside the 131072-wide vocab). The
    # sidecar must mirror this id in ``SamplingParams.stop_token_ids``.
    VIRTUAL_EOS_TOKEN_ID: int = 0

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        self._model_dir = vllm_config.model_config.model
        self._vllm_config = vllm_config

        # ---- Backbone via composition (Phase 1) ----
        self.backbone = NemotronHModel(
            vllm_config=vllm_config,
            prefix=f"{prefix}.backbone" if prefix else "backbone",
        )
        # Architectural quirk fix: vLLM's NemotronHMLP hard-codes
        # ReLU^2 for shared_experts.act_fn, ignoring config.mlp_hidden_act.
        # SmallMamba's checkpoint trained with SiLU — restore it.
        patch_silu_shared_experts(self.backbone)

        # ---- Streaming head: all TTS submodules + state machine (Phase 2) ----
        self.streaming_head = EasyMagpieStreamingHead()

        # ---- lm_head + LogitsProcessor (Phase 3) ----
        # Real lm_head loaded from safetensors (the ``lm_head.weight`` we
        # previously dropped). ``compute_logits`` returns these logits
        # (boosted on VIRTUAL_EOS_TOKEN_ID when streaming state's
        # audio_finished flips) so vLLM's standard sampler can EOS-stop the
        # request — no more ``engine.abort_request`` hack.
        config = vllm_config.model_config.hf_config
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

        # Per-request state. None until init_streaming_request is called.
        self._request_state: Optional[StreamingHeadState] = None

        # Last-step hidden state cache (Phase 1 — Phase 4 will pre-allocate).
        self._last_hidden_state: Optional[torch.Tensor] = None

        # Lazily-built (vocab_size,) tensor with +1e9 at VIRTUAL_EOS_TOKEN_ID
        # and 0 elsewhere. ``compute_logits`` adds
        # ``state.audio_finished_dev * self._eos_boost`` to the raw logits.
        # When audio_finished_dev=0 → no change; when =1 → EOS dominates and
        # vLLM's sampler picks token 0, hitting the request's stop_token_ids.
        # Built lazily because we need device + dtype from the first call.
        self._eos_boost: Optional[torch.Tensor] = None

    # ====================================================================
    # vLLM expected interface
    # ====================================================================

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        """Backbone forward + TTS heads in one module call.

        Mirrors V1's Option-3 architecture: the LT + phoneme prediction
        runs inside the model's forward, right after the backbone. This
        keeps both halves under one CUDA-graph capture once eager mode
        can be turned off (Phase 4).
        """
        hidden_states = self.backbone(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        self._last_hidden_state = hidden_states.detach()

        # Always call the TTS-heads hook (it returns immediately when no
        # streaming request is active). The hook itself is
        # ``@torch._dynamo.disable``-d, so torch.compile sees an opaque call.
        # A Python ``if state is not None`` HERE would specialise to False
        # at trace time (warmup has no state), permanently omitting the
        # call from the compiled forward graph.
        self.streaming_head.predict_codes_and_advance(
            hidden_states, self._request_state,
        )

        return hidden_states

    @torch._dynamo.disable
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """During prefill: trivial backbone lookup (Phase 1 behaviour).
        During streaming decode: build the next AR-step embedding from the
        streaming state machine (text + phoneme + audio additive combo),
        ignoring the sampled token id (we don't use vLLM's sampler).

        @torch._dynamo.disable: keep the per-request branch (state is None vs
        decode-step) out of torch.compile's tracing — see streaming_head.py.
        """
        state = self._request_state
        if state is None:
            return self.backbone.embed_input_ids(input_ids)

        device = input_ids.device
        dtype = self.streaming_head.audio_embeddings[0].weight.dtype
        backbone_param = next(self.backbone.parameters())
        dtype = backbone_param.dtype
        hidden = SMALLMAMBA["hidden_dim"]
        return self.streaming_head.next_input_embedding(
            state, device, dtype, hidden,
        )

    @torch._dynamo.disable
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Real lm_head logits + virtual-EOS boost on ``audio_finished``.

        Phase 3: V2 returns real (B, vocab_size) logits so vLLM's standard
        sampler drives termination. When the streaming state machine has
        flagged audio_finished (any LT codebook sampled audio_eos), we
        boost the logit at ``VIRTUAL_EOS_TOKEN_ID`` by +1e9 — that token
        wins the sampler's argmax/topk, and since the sidecar sets it as
        a ``stop_token_id``, the engine finishes the request naturally.

        CUDA-graph clean: ``audio_finished_dev`` is an on-device tensor
        and the boost is a pure tensor multiply + add, no Python branches
        on tensor values. The ``state is None`` branch is a Python-bool
        check on per-request setup (out-of-graph), not a hot-path branch.
        """
        logits = self.logits_processor(self.lm_head, hidden_states)
        state = self._request_state
        if state is None or state.audio_finished_dev is None:
            return logits

        # Lazily build the EOS-boost vector now that we know logits.shape +
        # device + dtype. One-time per process; reused across requests.
        if (self._eos_boost is None
                or self._eos_boost.device != logits.device
                or self._eos_boost.dtype != logits.dtype):
            vocab = logits.shape[-1]
            t = torch.zeros(vocab, dtype=logits.dtype, device=logits.device)
            t[self.VIRTUAL_EOS_TOKEN_ID] = 1e9
            self._eos_boost = t

        # audio_finished_dev: (1,) float in {0.0, 1.0}.
        # _eos_boost: (vocab,).
        # Broadcast → adds 1e9 at idx 0 when finished, 0 otherwise.
        return logits + state.audio_finished_dev * self._eos_boost

    # ====================================================================
    # Per-request lifecycle (called by sidecar server.py)
    # ====================================================================

    def init_streaming_request(
        self, *,
        audio_bos_id: int, audio_eos_id: int,
        phoneme_bos_id: int, phoneme_eos_id: int,
        streaming_speech_delay: int, streaming_phonemes_delay: int,
        phoneme_token_ids: list, text_eos_id: int,
        lt_temperature: float = 0.7, lt_topk: int = 80,
        phoneme_unk_token_id: int = 0,
        phoneme_confidence_unk_threshold: float = 0.0,
    ) -> None:
        """Build a fresh per-request streaming state. Mirrors V1's
        ``init_streaming_request`` but routes through the streaming head.
        """
        self._request_state = self.streaming_head.init_request(
            audio_bos_id=audio_bos_id,
            audio_eos_id=audio_eos_id,
            phoneme_bos_id=phoneme_bos_id,
            phoneme_eos_id=phoneme_eos_id,
            streaming_speech_delay=streaming_speech_delay,
            streaming_phonemes_delay=streaming_phonemes_delay,
            phoneme_token_ids=phoneme_token_ids,
            text_eos_id=text_eos_id,
            lt_temperature=lt_temperature,
            lt_topk=lt_topk,
            phoneme_unk_token_id=phoneme_unk_token_id,
            phoneme_confidence_unk_threshold=phoneme_confidence_unk_threshold,
        )

    # --- Server-readable streaming accumulator ---
    # ``_streaming_codes`` is the per-frame audio-codes accumulator the
    # sidecar's ``_pull_codes`` helper drains after each engine.step().
    # The setter is how server.py clears the list after pulling.
    @property
    def _streaming_codes(self):
        state = self._request_state
        return state.emitted_codes if state is not None else []

    @_streaming_codes.setter
    def _streaming_codes(self, value):
        if self._request_state is not None:
            self._request_state.emitted_codes = list(value)

    # ====================================================================
    # Sidecar-facing helpers — delegate to streaming_head
    # ====================================================================

    def wire_tokenizer(
        self,
        subword_vocab: dict,
        bos_id: int,
        eos_id: int,
        cfg_unk_token_id: int,
        subword_padding_idx: Optional[int] = None,
    ) -> None:
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
        self.streaming_head.warmup_compiled_lt(
            device=device, dtype=dtype, iters=iters,
        )

    def embed_audio_tokens(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        return self.streaming_head.embed_audio_tokens(audio_tokens)

    def embed_text_tokens(
        self,
        text_tokens: torch.Tensor,
        text_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.streaming_head.embed_text_tokens(text_tokens, text_lens)

    def embed_phoneme_tokens(self, phoneme_tokens: torch.Tensor) -> torch.Tensor:
        return self.streaming_head.embed_phoneme_tokens(phoneme_tokens)

    def encode_speaker(
        self, audio_embedded: torch.Tensor, audio_lens: torch.Tensor,
    ) -> torch.Tensor:
        return self.streaming_head.encode_speaker(audio_embedded, audio_lens)

    def build_prefill_combined_embeddings(
        self,
        context_audio_codes: torch.Tensor,
        context_audio_codes_lens: torch.Tensor,
        context_text_tokens: torch.Tensor,
        context_text_tokens_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``[speaker_encoder(audio_embed(codes)) | cas_encoder(text)]``.
        Mirrors V1.build_prefill_combined_embeddings.
        """
        audio_emb = self.streaming_head.embed_audio_tokens(context_audio_codes)
        audio_emb = self.streaming_head.encode_speaker(
            audio_emb, context_audio_codes_lens,
        )
        text_emb = self.streaming_head.embed_text_tokens(
            context_text_tokens, context_text_tokens_lens,
        )
        ctx_embedding = torch.cat([audio_emb, text_emb], dim=1)
        ctx_lens = context_audio_codes_lens + context_text_tokens_lens
        return ctx_embedding, ctx_lens

    # ====================================================================
    # Weight loading
    # ====================================================================

    def load_weights(self, weights) -> set:
        """Strip ``model.`` prefix from backbone tensors, intercept
        ``lm_head.weight`` (Phase 3) and copy into ``self.lm_head``,
        delegate everything else to ``self.backbone.load_weights``,
        re-prefix returned names with ``backbone.``, then load tts_extras.
        """
        loaded_lm_head: set[str] = set()

        def _rewrite(weights_iter):
            for name, tensor in weights_iter:
                if name == "lm_head.weight":
                    # Phase 3: V2 has a real lm_head. Copy directly and
                    # mark loaded — don't forward to backbone (which has no
                    # lm_head of its own).
                    self.lm_head.weight.data.copy_(
                        tensor.to(self.lm_head.weight.dtype)
                    )
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
                "tts_extras.pt not found at %s; TTS submodules left "
                "uninitialized (forward will produce garbage)", tts_extras_path)
            loaded_tts = set()

        return loaded_backbone | set(loaded_tts)

    def load_tts_extras(self, tts_extras_path: str) -> set[str]:
        """Load TTS-specific tensors from tts_extras.pt with key remap:
        the checkpoint keys are flat (``cas_encoder.X``, ``audio_embeddings.Y``)
        but in V2 they live under ``streaming_head.``. We prepend the prefix
        when looking up the destination.
        """
        tensors: dict[str, torch.Tensor] = torch.load(
            tts_extras_path, map_location="cpu", weights_only=True
        )
        logger.info(
            "Loading %d TTS-extras tensors from %s (V2 streaming_head)",
            len(tensors), tts_extras_path,
        )

        own = dict(self.named_parameters())
        own_state = dict(self.state_dict())

        loaded: set[str] = set()
        unmatched: list[str] = []
        for k, v in tensors.items():
            mapped = "streaming_head." + k

            # Lazily resize cas_encoder.embed_tokens to the trained vocab.
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
            logger.warning(
                "%d TTS-extras keys not found on V2 model: %s",
                len(unmatched), unmatched[:5],
            )
        logger.info("TTS extras: loaded %d / %d", len(loaded), len(tensors))
        return loaded
