"""EasyMagpie streaming head — V2's dedicated TTS sub-module.

Owns ALL the non-vLLM TTS pieces of the model:

  * Per-request state (encapsulated in ``StreamingHeadState`` — replaces the
    ~15 scattered ``self._stream_*`` attributes V1 carries on the top-level
    model).
  * TTS submodules — audio codebook embeddings, char-aware subword encoder,
    speaker encoder, local transformer + projections, phoneme heads. Stored
    on this module so they can be loaded from ``tts_extras.pt`` with key
    remapping (parent V2 prefixes each key with ``streaming_head.`` when
    loading).
  * The two pure-function entry points the parent V2 forward+embed_input_ids
    call into:
      - ``next_input_embedding(state, device, dtype, hidden) -> Tensor``
      - ``predict_codes_and_advance(hidden_last, state) -> StreamingHeadOutput``
  * Helper machinery (LT compile/warmup, wire_tokenizer, embed_*).

The TTS submodules are built in fp32 — same rationale as V1 (see comment
in __init__): tts_extras.pt copies tensors via
``target.data.copy_(v.to(target.dtype))``, and if target is fp16, source
fp32 gets rounded *before* the copy. Constructing in fp32 keeps the trained
precision intact.

Phase-2 scope only. Real ``compute_logits`` + suppress_mask still come in
Phase 3; pre-allocated CUDA-graph-friendly buffers in Phase 4.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn


logger = logging.getLogger(__name__)


# Same constants as V1 / V2's top-level SMALLMAMBA dict. Kept here for
# locality with the submodule-construction code.
SMALLMAMBA = dict(
    hidden_dim=1536,
    embedding_dim=1536,
    audio_embedding_dim=1536,
    num_audio_codebooks=8,
    num_all_tokens_per_codebook=1032,
    frame_stacking_factor=2,
    phoneme_stacking_factor=1,
    phoneme_vocab_size=2051,
    cas_encoder_n_layers=1,
    cas_subword_vocab_size=260,
    cas_subword_padding_idx=0,
    speaker_encoder_n_layers=1,
    speaker_encoder_n_heads=12,
    speaker_encoder_d_ffn=3072,
    speaker_encoder_kernel_size=1,
    local_transformer_n_layers=3,
    local_transformer_n_heads=12,
    local_transformer_hidden_dim=1536,
    local_transformer_type="autoregressive",
)


# --------------------------------------------------------------------- #
# StreamingHeadState — one request's full state, on a single object.
# Replaces V1's 15+ ``self._stream_*`` attributes on the top-level model.
# --------------------------------------------------------------------- #
@dataclass
class StreamingHeadState:
    """Per-request streaming state machine. Instantiated fresh per request
    by ``EasyMagpieStreamingHead.init_request`` and stored on the parent
    V2 model as ``self._request_state``.
    """
    # Static per-request constants (set by init_request, never mutated).
    audio_bos_id: int = 0
    audio_eos_id: int = 0
    phoneme_bos_id: int = 0
    phoneme_eos_id: int = 0
    speech_delay: int = 0
    phonemes_delay: int = 0
    text_eos_id: int = 0
    lt_temperature: float = 0.7
    lt_topk: int = 80
    # Phoneme UNK substitution; off when threshold == 0 (current ckpt default).
    phoneme_unk_token_id: int = 0
    phoneme_confidence_unk_threshold: float = 0.0

    # Per-utterance text stream.
    phoneme_ids: list = field(default_factory=list)
    phoneme_pos: int = 0

    # Per-step counters / phase flags.
    text_tokens_seen: int = 0
    phoneme_steps: int = 0
    audio_steps: int = 0
    text_finished: bool = False
    phoneme_stream_ended: bool = False
    audio_finished: bool = False

    # Last sampled codes / phoneme tokens (fed into next-step embedding).
    last_audio_codes: Optional[torch.Tensor] = None       # (1, C*S) int64
    last_phoneme_tokens: Optional[torch.Tensor] = None    # (1, S) int64

    # Server-pullable accumulator of emitted code frames.
    emitted_codes: list = field(default_factory=list)

    # Has the AR decode loop begun? True after the first embed_input_ids call.
    # ``predict_codes_and_advance`` short-circuits while this is False (prefill
    # forward call shouldn't emit codes).
    decode_started: bool = False

    # On-device twin of ``audio_finished``, kept in sync so ``compute_logits``
    # can boost the virtual-EOS logit without a GPU→CPU sync (CUDA-graph
    # friendly). Lazily allocated on the first ``predict_codes_and_advance``
    # call (we need the hidden state's device to build it).
    audio_finished_dev: Optional[torch.Tensor] = None


# --------------------------------------------------------------------- #
# EasyMagpieStreamingHead — owns all TTS submodules + state machine logic.
# --------------------------------------------------------------------- #
class EasyMagpieStreamingHead(nn.Module):
    """Holds the TTS submodules and runs the per-step streaming logic.

    Per-request state lives in a separate ``StreamingHeadState`` instance
    that the parent V2 passes in to ``next_input_embedding`` and
    ``predict_codes_and_advance``.

    Submodule field names match V1's top-level names exactly (audio_embeddings,
    cas_encoder, speaker_encoder, local_transformer, etc.), so the existing
    ``tts_extras.pt`` checkpoint loads with a simple ``streaming_head.``
    prefix prepended.
    """

    def __init__(self) -> None:
        super().__init__()

        # Deferred nemo imports — keep module top nemo-free for plugin
        # registration in the parent's vLLM spawn-child.
        from nemo.collections.tts.modules import transformer_2501
        from nemo.collections.tts.modules.magpietts_modules import (
            CharAwareSubwordEncoder,
        )

        # Force fp32 default dtype for TTS submodule construction.
        # ``load_tts_extras`` copies via ``target.data.copy_(v.to(target.dtype))``;
        # if target is bf16/fp16, source fp32 gets rounded *before* the copy
        # and the rounded values can't be recovered by a later upcast. Keeping
        # construction in fp32 preserves trained precision. The TTS submodules
        # are small relative to the 3.3 GB backbone — negligible memory cost.
        _prev_default = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            c = SMALLMAMBA
            embed_dim = c["embedding_dim"]
            hidden = c["hidden_dim"]
            n_books = c["num_audio_codebooks"]
            n_tokens = c["num_all_tokens_per_codebook"]
            stack = c["frame_stacking_factor"]
            n_tables = n_books * stack

            self.audio_embeddings = nn.ModuleList(
                [nn.Embedding(n_tokens, c["audio_embedding_dim"])
                 for _ in range(n_tables)]
            )
            if c["audio_embedding_dim"] != embed_dim:
                self.audio_in_projection = nn.Linear(
                    c["audio_embedding_dim"], embed_dim)
                self.audio_out_projection = nn.Linear(
                    hidden, c["audio_embedding_dim"])
            else:
                self.audio_in_projection = nn.Identity()
                self.audio_out_projection = nn.Identity()

            self.speaker_encoder = transformer_2501.Transformer(
                n_layers=c["speaker_encoder_n_layers"],
                d_model=embed_dim,
                d_ffn=c["speaker_encoder_d_ffn"],
                sa_n_heads=c["speaker_encoder_n_heads"],
                kernel_size=c["speaker_encoder_kernel_size"],
                p_dropout=0.0,
                is_causal=False,
                use_learnable_pos_emb=True,
            )

            _ascii = "abcdefghijklmnopqrstuvwxyz"
            self.cas_encoder = CharAwareSubwordEncoder(
                d_embed=embed_dim,
                llm_tokenizer_vocab={ch: i for i, ch in enumerate(_ascii)},
                subword_padding_idx=c["cas_subword_padding_idx"],
                special_vocab=None,
                n_layers=c["cas_encoder_n_layers"],
            )
            self.cas_encoder.float()

            self.final_proj = nn.Linear(
                c["audio_embedding_dim"],
                n_books * n_tokens * stack,
            )
            self.phoneme_embeddings = nn.ModuleList(
                [nn.Embedding(c["phoneme_vocab_size"], embed_dim)
                 for _ in range(c["phoneme_stacking_factor"])]
            )
            self.phoneme_final_proj = nn.Linear(
                hidden,
                c["phoneme_vocab_size"] * c["phoneme_stacking_factor"]
            )

            lt_hidden = c["local_transformer_hidden_dim"]
            if lt_hidden != hidden:
                self.local_transformer_in_projection = nn.Linear(hidden, lt_hidden)
            else:
                self.local_transformer_in_projection = nn.Identity()

            self.local_transformer = transformer_2501.Transformer(
                n_layers=c["local_transformer_n_layers"],
                d_model=lt_hidden,
                d_ffn=lt_hidden * 4,
                sa_n_heads=c["local_transformer_n_heads"],
                kernel_size=1,
                p_dropout=0.0,
                is_causal=(c["local_transformer_type"] == "autoregressive"),
                max_length_causal_mask=n_books * stack + 2,
                use_learnable_pos_emb=True,
            )

            if lt_hidden != c["audio_embedding_dim"]:
                self.local_transformer_audio_out_projection = nn.Linear(
                    lt_hidden, c["audio_embedding_dim"]
                )
            else:
                self.local_transformer_audio_out_projection = nn.Identity()

            self.local_transformer_out_projections = nn.ModuleList(
                [nn.Linear(lt_hidden, n_tokens) for _ in range(n_tables)]
            )
        finally:
            torch.set_default_dtype(_prev_default)

    # ----------------------- per-request lifecycle ---------------------- #

    def init_request(
        self, *,
        audio_bos_id: int, audio_eos_id: int,
        phoneme_bos_id: int, phoneme_eos_id: int,
        streaming_speech_delay: int, streaming_phonemes_delay: int,
        phoneme_token_ids: list, text_eos_id: int,
        lt_temperature: float = 0.7, lt_topk: int = 80,
        phoneme_unk_token_id: int = 0,
        phoneme_confidence_unk_threshold: float = 0.0,
    ) -> StreamingHeadState:
        """Build a fresh per-request state machine. Returns the new state
        object the parent V2 should store as ``self._request_state``.
        """
        return StreamingHeadState(
            audio_bos_id=int(audio_bos_id),
            audio_eos_id=int(audio_eos_id),
            phoneme_bos_id=int(phoneme_bos_id),
            phoneme_eos_id=int(phoneme_eos_id),
            speech_delay=int(streaming_speech_delay),
            phonemes_delay=int(streaming_phonemes_delay),
            text_eos_id=int(text_eos_id),
            lt_temperature=float(lt_temperature),
            lt_topk=int(lt_topk),
            phoneme_ids=list(phoneme_token_ids),
            phoneme_unk_token_id=int(phoneme_unk_token_id),
            phoneme_confidence_unk_threshold=float(phoneme_confidence_unk_threshold),
        )

    # ----------------------- phase masks -------------------------------- #
    @staticmethod
    def _phase_masks(state: StreamingHeadState) -> tuple[bool, bool, bool]:
        """Return (needs_text, needs_phoneme, needs_audio). Mirrors the
        production ``_prepare_streaming_input`` phase logic (lines 1391-1396
        of easy_magpietts_inference.py).
        """
        needs_text = not state.text_finished
        needs_phoneme = (
            state.text_tokens_seen >= state.phonemes_delay
            and not state.phoneme_stream_ended
        )
        needs_audio = (
            state.text_tokens_seen >= state.speech_delay
            and not state.audio_finished
        )
        return needs_text, needs_phoneme, needs_audio

    # ----------------------- embedding helpers -------------------------- #

    def embed_audio_tokens(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        """``(B, C*S, T)`` int64 -> ``(B, T, embedding_dim)``. Sum-average over
        codebooks, then project. Mirrors V1.embed_audio_tokens."""
        n_tables = audio_tokens.size(1)
        audio_embedding: Optional[torch.Tensor] = None
        for c in range(n_tables):
            e = self.audio_embeddings[c](audio_tokens[:, c, :])
            audio_embedding = e if audio_embedding is None else audio_embedding + e
        assert audio_embedding is not None
        audio_embedding = audio_embedding / n_tables
        return self.audio_in_projection(audio_embedding)

    def encode_speaker(
        self, audio_embedded: torch.Tensor, audio_lens: torch.Tensor
    ) -> torch.Tensor:
        from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
        ctx_mask = get_mask_from_lengths(audio_lens)
        return self.speaker_encoder(
            audio_embedded, ctx_mask, cond=None, cond_mask=None
        )["output"]

    def embed_text_tokens(
        self,
        text_tokens: torch.Tensor,
        text_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """CAS-only text embedding (``disable_subword_embedding=True`` path).
        Mirrors V1.embed_text_tokens.
        """
        if text_lens is None:
            text_lens = torch.full(
                (text_tokens.size(0),),
                text_tokens.size(1),
                dtype=torch.long,
                device=text_tokens.device,
            )
        from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
        text_mask = get_mask_from_lengths(text_lens)
        out = self.cas_encoder(
            text_tokens.to(text_tokens.device),
            subword_mask=text_mask,
        )
        return out.to(self.audio_embeddings[0].weight.dtype)

    def embed_phoneme_tokens(self, phoneme_tokens: torch.Tensor) -> torch.Tensor:
        """``(B, S, T)`` -> ``(B, T, E)``. Sum/avg across stacking factor.
        Mirrors V1.embed_phoneme_tokens.
        """
        emb = None
        for c in range(phoneme_tokens.size(1)):
            e = self.phoneme_embeddings[c](phoneme_tokens[:, c, :])
            emb = e if emb is None else emb + e
        emb = emb / phoneme_tokens.size(1)
        return emb

    # ----------------------- wire_tokenizer ----------------------------- #

    def wire_tokenizer(
        self,
        subword_vocab: dict,
        bos_id: int,
        eos_id: int,
        cfg_unk_token_id: int,
        subword_padding_idx: Optional[int] = None,
    ) -> None:
        """Reconstruct the CAS encoder's subword->char map from the live
        tokenizer. Mirrors V1.wire_tokenizer.
        """
        from nemo.collections.tts.modules.magpietts_modules import build_vocabs

        special_vocab = {
            "<BOS>": bos_id,
            "<EOS>": eos_id,
            "<CFG_UNK>": cfg_unk_token_id,
        }
        pad = (
            subword_padding_idx
            if subword_padding_idx is not None
            else SMALLMAMBA["cas_subword_padding_idx"]
        )
        subword_id_to_char_ids, char_vocab = build_vocabs(
            subword_vocab=subword_vocab,
            subword_padding_idx=pad,
            special_vocab=special_vocab,
        )
        self.cas_encoder.subword_id_to_char_ids = subword_id_to_char_ids
        self.cas_encoder.char_vocab = char_vocab
        logger.info(
            "Tokenizer wired: %d subwords mapped to %d chars",
            len(subword_id_to_char_ids), len(char_vocab),
        )

    # ----------------------- LT helpers (compiled path) ----------------- #

    def _get_lt_helper(self):
        """Lazy-build a LocalTransformerHelper holding refs to our LT
        submodules. Mirrors V1._get_lt_helper.
        """
        if getattr(self, "_lt_helper_cached", None) is not None:
            return self._lt_helper_cached
        from nemo.collections.tts.modules.magpietts_modules import (
            LocalTransformerHelper,
        )
        c = SMALLMAMBA
        n_tables = c["num_audio_codebooks"] * c["frame_stacking_factor"]
        helper = LocalTransformerHelper(
            local_transformer=self.local_transformer,
            audio_embeddings=self.audio_embeddings,
            audio_in_projection=self.audio_in_projection,
            local_transformer_in_projection=self.local_transformer_in_projection,
            local_transformer_audio_out_projection=self.local_transformer_audio_out_projection,
            local_transformer_out_projections=self.local_transformer_out_projections,
            num_audio_codebooks=n_tables,
            frame_stacking_factor=1,
            audio_eos_id=c["num_all_tokens_per_codebook"] - 8 + 1,
            mask_token_id=c["num_all_tokens_per_codebook"] - 8 + 4,
            codebook_size=c["num_all_tokens_per_codebook"] - 8,
        )
        helper._fused_temperature = float(
            os.environ.get("EM_LT_TEMPERATURE", "0.7")
        )
        helper._fused_topk = int(os.environ.get("EM_LT_TOPK", "80"))
        self._lt_helper_cached = helper
        return helper

    def _get_compiled_lt(self, device: torch.device, dtype: torch.dtype):
        """Lazy-build the compiled LT wrapper. Mirrors V1._get_compiled_lt."""
        cached = getattr(self, "_compiled_lt_cached", None)
        if cached is not None and cached["device"] == device and cached["dtype"] == dtype:
            return cached["wrapper"]
        from nemo.collections.tts.modules.magpietts_lt_fused import (
            LocalTransformerFusedModule,
        )
        helper = self._get_lt_helper()
        wrapper = LocalTransformerFusedModule(
            helper,
            temperature=helper._fused_temperature,
            topk=helper._fused_topk,
        )
        wrapper.eval().to(device=device, dtype=dtype)
        _mode = os.environ.get("EM_LT_COMPILE_MODE", "reduce-overhead").lower()
        compiled = torch.compile(wrapper, mode=_mode, fullgraph=True)
        self._compiled_lt_cached = {
            "wrapper": compiled,
            "device": device,
            "dtype": dtype,
        }
        logger.info(
            "[compiled-lt] LocalTransformerFusedModule torch.compile(mode=%s) "
            "ready (device=%s, dtype=%s, top_k=%d, temperature=%.2f)",
            _mode, device, dtype, helper._fused_topk, helper._fused_temperature,
        )
        return compiled

    def warmup_compiled_lt(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        iters: int = 3,
    ) -> None:
        """Trigger compile + CUDA-graph capture for the LT at startup."""
        if device is None:
            device = torch.device("cuda:0")
        if dtype is None:
            dtype = next(self.parameters()).dtype
        c = SMALLMAMBA
        hidden = c["hidden_dim"]
        n_tables = c["num_audio_codebooks"] * c["frame_stacking_factor"]
        helper = self._get_lt_helper()
        topk = helper._fused_topk
        wrapper = self._get_compiled_lt(device, dtype)
        for _ in range(iters):
            with torch.inference_mode():
                h = torch.randn(1, hidden, device=device, dtype=dtype)
                u = torch.rand(
                    n_tables, 1, topk, device=device, dtype=torch.float32,
                ).clamp_(min=1e-10, max=1.0 - 1e-7)
                g = -torch.log(-torch.log(u)).to(dtype)
                _ = wrapper(h, g)
                torch.cuda.synchronize(device)

    def _predict_audio_codes_compiled(
        self, hidden_state: torch.Tensor, state: StreamingHeadState,
    ) -> torch.Tensor:
        """Sample one frame's 16 audio codes via the compiled LT.

        Per-step deterministic Gumbel noise generator — same scheme as V1
        (seed = ``audio_steps * 7919 + 1``) so runs are reproducible and
        the noise sequence is identical across utterances.
        Returns ``(B, n_tables=16, 1)``.
        """
        if hidden_state.dim() == 3:
            hidden_state = hidden_state.squeeze(0)
        device = hidden_state.device
        dtype = hidden_state.dtype
        wrapper = self._get_compiled_lt(device, dtype)
        B = hidden_state.shape[0]
        c = SMALLMAMBA
        n_tables = c["num_audio_codebooks"] * c["frame_stacking_factor"]
        helper = self._get_lt_helper()
        topk = helper._fused_topk

        # Gumbel(0,1) noise via per-device deterministic generator. Seed
        # comes from state.audio_steps so the noise sequence is identical
        # across utterances and runs.
        gen_key = "_compile_gumbel_gen_" + str(device).replace(":", "_")
        gen = getattr(self, gen_key, None)
        if gen is None:
            gen = torch.Generator(device=device)
            setattr(self, gen_key, gen)
        gen.manual_seed(state.audio_steps * 7919 + 1)
        u = torch.rand(
            n_tables, B, topk, dtype=torch.float32, device=device,
            generator=gen,
        ).clamp_(min=1e-10, max=1.0 - 1e-7)
        gumbel = -torch.log(-torch.log(u)).to(dtype)
        tokens = wrapper(hidden_state, gumbel)
        return tokens.long().unsqueeze(-1)

    def _predict_audio_codes_trt_fused(
        self,
        hidden_state: torch.Tensor,
        temperature: Optional[float] = None,
        topk: Optional[int] = None,
    ) -> torch.Tensor:
        """Legacy / fallback LT path via LocalTransformerHelper. Mirrors V1."""
        helper = self._get_lt_helper()
        if temperature is None:
            temperature = helper._fused_temperature
        if topk is None:
            topk = helper._fused_topk
        if hidden_state.dim() == 3:
            hidden_state = hidden_state.squeeze(0)
        codes = helper.sample_autoregressive(
            dec_output=hidden_state,
            temperature=temperature,
            topk=topk,
            use_kv_cache=False,
        )
        if codes.dim() == 3 and codes.shape[-1] != 1:
            codes = codes.reshape(codes.shape[0], -1, 1)
        elif codes.dim() == 2:
            codes = codes.unsqueeze(-1)
        return codes.long()

    # ----------------------- per-step entry points ---------------------- #

    # @torch._dynamo.disable: keep the TTS streaming logic outside vLLM's
    # torch.compile graph. Without this, torch.compile traces forward()
    # once at warmup (with state=None), specializes the Python branches,
    # then at request time the compiled graph skips predict_codes_and_advance
    # entirely → no codes emitted, request hits max_frames cap.
    @torch._dynamo.disable
    def next_input_embedding(
        self, state: StreamingHeadState,
        device: torch.device, dtype: torch.dtype, hidden: int,
    ) -> torch.Tensor:
        """Build the next AR-step input embedding from the streaming state.

        ``next_input = text_emb*needs_text + phoneme_emb*needs_phoneme + audio_emb*needs_audio``

        Mutates ``state`` in place (advances phoneme_pos, sets text_finished).
        Returns shape (1, hidden) — the embedding for one decode token.
        """
        state.decode_started = True
        next_input = torch.zeros(1, 1, hidden, device=device, dtype=dtype)
        needs_text, needs_phoneme, needs_audio = self._phase_masks(state)

        # --- Text embedding ----
        if needs_text:
            pos = state.phoneme_pos
            if pos < len(state.phoneme_ids):
                tok_id = int(state.phoneme_ids[pos])
                state.phoneme_pos = pos + 1
                tok = torch.tensor([[tok_id]], dtype=torch.long, device=device)
                tok_lens = torch.tensor([1], dtype=torch.long, device=device)
                text_emb = self.embed_text_tokens(tok, tok_lens).to(dtype=dtype)
                next_input = next_input + text_emb
                if tok_id == state.text_eos_id:
                    state.text_finished = True

        # --- Phoneme embedding ----
        if needs_phoneme:
            S = SMALLMAMBA["phoneme_stacking_factor"]
            if state.phoneme_steps == 0:
                phon_in = torch.full(
                    (1, S, 1), state.phoneme_bos_id,
                    dtype=torch.long, device=device,
                )
                phon_emb = self.embed_phoneme_tokens(phon_in)
            elif state.last_phoneme_tokens is not None:
                phon_in = state.last_phoneme_tokens.unsqueeze(-1)
                phon_emb = self.embed_phoneme_tokens(phon_in)
            else:
                phon_emb = None
            if phon_emb is not None:
                next_input = next_input + phon_emb.to(dtype=dtype)

        # --- Audio embedding ----
        if needs_audio:
            CS = SMALLMAMBA["num_audio_codebooks"] * SMALLMAMBA["frame_stacking_factor"]
            if state.audio_steps == 0:
                audio_in = torch.full(
                    (1, CS, 1), state.audio_bos_id,
                    dtype=torch.long, device=device,
                )
                audio_emb = self.embed_audio_tokens(audio_in)
            elif state.last_audio_codes is not None:
                audio_in = state.last_audio_codes.unsqueeze(-1)
                audio_emb = self.embed_audio_tokens(audio_in)
            else:
                audio_emb = None
            if audio_emb is not None:
                next_input = next_input + audio_emb.to(dtype=dtype)

        # vLLM expects shape (N, hidden) where N = number of input tokens.
        return next_input.squeeze(0)

    @torch._dynamo.disable
    def predict_codes_and_advance(
        self,
        hidden_last: torch.Tensor,
        state,
    ) -> dict:
        """Run LT + phoneme prediction on the last-position hidden state,
        sample 16 audio codes, advance counters, detect audio_eos.

        Called unconditionally from ``V2.forward``. Short-circuits when
        there's no streaming request (``state is None``) or the first
        decode token hasn't been embedded yet (``decode_started=False``).
        The unconditional call site + internal-check pattern is what
        keeps the call OUT of torch.compile's trace-time Python branch
        specialisation (see V2.forward).
        """
        if state is None or not state.decode_started:
            return {"codes": None, "phoneme_tokens": None,
                    "audio_eos_seen": False, "phoneme_eos_seen": False}
        needs_text, needs_phoneme, needs_audio = self._phase_masks(state)
        # Last position of hidden_states.
        last_h = (
            hidden_last[-1:]
            if hidden_last.dim() == 2
            else hidden_last.reshape(-1, hidden_last.shape[-1])[-1:]
        )
        last_h_fp32 = last_h.float()

        out = {
            "codes": None, "phoneme_tokens": None,
            "audio_eos_seen": False, "phoneme_eos_seen": False,
        }

        # --- Phoneme prediction ----
        if needs_phoneme and self.phoneme_final_proj is not None:
            phon_logits = self.phoneme_final_proj(last_h_fp32)
            P = SMALLMAMBA["phoneme_vocab_size"]
            S = SMALLMAMBA["phoneme_stacking_factor"]
            phon_logits = phon_logits.view(1, S, P)
            phon_tok = phon_logits.argmax(dim=-1)
            # UNK substitution (production parity, easy_magpietts_inference.py:1666-1678).
            # Off when threshold == 0 (current SmallMamba ckpt default).
            if state.phoneme_confidence_unk_threshold > 0.0:
                max_probs = torch.softmax(phon_logits, dim=-1).max(dim=-1).values
                underconfident = (max_probs < state.phoneme_confidence_unk_threshold).any(dim=1, keepdim=True)
                eos_predicted = (phon_tok == state.phoneme_eos_id).any(dim=1, keepdim=True)
                replace = underconfident & ~eos_predicted
                phon_tok = torch.where(
                    replace, torch.full_like(phon_tok, state.phoneme_unk_token_id), phon_tok,
                )
            state.last_phoneme_tokens = phon_tok.long()
            state.phoneme_steps += 1
            if (phon_tok == state.phoneme_eos_id).any():
                state.phoneme_stream_ended = True
                out["phoneme_eos_seen"] = True
            out["phoneme_tokens"] = phon_tok.long()

        # --- Audio prediction ----
        # Lazy-init the on-device audio_finished twin. Done here (not in
        # init_request) because we only learn the device once a real
        # hidden_state arrives. CUDA-graph friendly: the .copy_() update
        # below stays inside the captured region.
        if state.audio_finished_dev is None:
            state.audio_finished_dev = torch.zeros(
                1, dtype=torch.float32, device=last_h.device,
            )

        if needs_audio:
            # LT input dtype. Default fp32 matches production (tts.py:1133).
            # EM_LT_DTYPE=fp16 enables fp16 LT compute (~2x faster on Tensor
            # Cores) at the cost of some precision; use to A/B against fp32.
            _lt_dtype = os.environ.get("EM_LT_DTYPE", "fp32").lower()
            if _lt_dtype == "fp16":
                last_h_lt = last_h.half()
            else:
                last_h_lt = last_h_fp32
            backend = os.environ.get("EASYMAGPIE_LT_BACKEND", "compile").lower()
            if backend == "compile":
                codes = self._predict_audio_codes_compiled(last_h_lt, state)
            else:
                codes = self._predict_audio_codes_trt_fused(
                    last_h_lt,
                    temperature=state.lt_temperature,
                    topk=state.lt_topk,
                )
            codes_flat = codes.squeeze(-1).long()
            state.last_audio_codes = codes_flat
            state.audio_steps += 1
            # Server-pullable accumulator (mirrors V1's _streaming_codes).
            state.emitted_codes.append(codes.detach().cpu())
            # Detect audio_eos. Update the on-device twin via .copy_() (a
            # captured kernel op) so compute_logits sees the flag without
            # any GPU→CPU sync. The Python-side bool is kept for the
            # existing branches in this method + sidecar polling.
            eos_hit = (codes_flat == state.audio_eos_id).any()
            state.audio_finished_dev.copy_(
                torch.maximum(state.audio_finished_dev, eos_hit.float().reshape(1))
            )
            if bool(eos_hit):
                state.audio_finished = True
                out["audio_eos_seen"] = True
            out["codes"] = codes_flat

        # Phase-mask advance: text_tokens_seen always ticks (we're never in
        # the context phase post-prefill).
        state.text_tokens_seen += 1
        return out
