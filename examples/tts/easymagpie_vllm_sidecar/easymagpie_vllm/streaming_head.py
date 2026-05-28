"""EasyMagpie TTS streaming head: per-request state + TTS submodules.

The model class composes one instance of this module and delegates all
TTS-specific work to it. State for the current request lives in a
:class:`StreamingHeadState` instance the model stashes as
``self._request_state``; the head's per-step entry points read and mutate
that state in place.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from torch import nn


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


class StreamingHeadState:
    """Per-request streaming state — Python fields + pre-allocated device twins.

    Address-stable: all device tensors are allocated once at construction and
    mutated in place across steps and across requests (via ``reset_in_place``).
    That is what CUDA-graph replay requires.

    Python-side fields stay as the source of truth for the phase masks
    (``_phase_masks`` reads them). Device twins exist for values that the
    captured forward / compiled graph needs to read without GPU→CPU syncs:
    ``audio_steps_dev`` (gates BOS vs last-codes embedding), ``audio_finished_dev``
    (drives sampler-EOS logit boost), ``last_audio_codes_dev`` /
    ``last_phoneme_tokens_dev`` (graph-friendly inputs to the embedding lookup),
    and ``codes_buf_dev`` / ``codes_count_dev`` (ring buffer the server drains
    after each engine step).
    """

    DEFAULT_MAX_FRAMES: int = 320
    DEFAULT_MAX_PHONEME: int = 2048

    def __init__(
        self,
        *,
        device: torch.device,
        audio_bos_id: int = 0,
        audio_eos_id: int = 0,
        phoneme_bos_id: int = 0,
        phoneme_eos_id: int = 0,
        speech_delay: int = 0,
        phonemes_delay: int = 0,
        text_eos_id: int = 0,
        lt_temperature: float = 0.7,
        lt_topk: int = 80,
        phoneme_unk_token_id: int = 0,
        phoneme_confidence_unk_threshold: float = 0.0,
        phoneme_ids: Optional[list] = None,
        max_frames: int = DEFAULT_MAX_FRAMES,
        max_phoneme: int = DEFAULT_MAX_PHONEME,
    ) -> None:
        """Allocate Python fields + device twins for one request.

        Args:
            device: CUDA device on which to allocate all twins.
            audio_bos_id / audio_eos_id: audio codebook BOS / EOS ids.
            phoneme_bos_id / phoneme_eos_id: phoneme tokenizer BOS / EOS ids.
            speech_delay / phonemes_delay: post-context AR-step delays before
                audio / phoneme emission begins.
            text_eos_id: subword tokenizer EOS id; consuming it sets
                ``text_finished = True``.
            lt_temperature / lt_topk: local-transformer sampling knobs.
            phoneme_unk_token_id / phoneme_confidence_unk_threshold: UNK
                substitution for low-confidence phoneme predictions.
            phoneme_ids: per-step CAS subword ids for the utterance.
            max_frames / max_phoneme: device-buffer capacities.
        """
        self._device = device
        n_tables = SMALLMAMBA["num_audio_codebooks"] * SMALLMAMBA["frame_stacking_factor"]
        phoneme_stack = SMALLMAMBA["phoneme_stacking_factor"]
        self._n_tables = n_tables
        self._phoneme_stack = phoneme_stack
        self._max_frames = int(max_frames)
        self._max_phoneme = int(max_phoneme)

        self.audio_bos_id = int(audio_bos_id)
        self.audio_eos_id = int(audio_eos_id)
        self.phoneme_bos_id = int(phoneme_bos_id)
        self.phoneme_eos_id = int(phoneme_eos_id)
        self.speech_delay = int(speech_delay)
        self.phonemes_delay = int(phonemes_delay)
        self.text_eos_id = int(text_eos_id)
        self.lt_temperature = float(lt_temperature)
        self.lt_topk = int(lt_topk)
        self.phoneme_unk_token_id = int(phoneme_unk_token_id)
        self.phoneme_confidence_unk_threshold = float(phoneme_confidence_unk_threshold)
        self.phoneme_ids: list = list(phoneme_ids or [])
        self.phoneme_pos: int = 0

        self.text_tokens_seen: int = 0
        self.phoneme_steps: int = 0
        self.audio_steps: int = 0
        self.text_finished: bool = False
        self.phoneme_stream_ended: bool = False
        self.audio_finished: bool = False
        self.decode_started: bool = False

        self.last_audio_codes: Optional[torch.Tensor] = None
        self.last_phoneme_tokens: Optional[torch.Tensor] = None

        # Device twins must not be inference-tensors: ``reset_in_place`` is
        # called from a real request (outside inference_mode) even when the
        # original allocation happened during vLLM warmup (inside inference_mode).
        with torch.inference_mode(False):
            i64 = torch.int64
            z = lambda *shape, dtype=i64: torch.zeros(shape, dtype=dtype, device=device)
            self.audio_steps_dev = z(dtype=i64)
            self.phoneme_steps_dev = z(dtype=i64)
            self.text_tokens_seen_dev = z(dtype=i64)
            self.audio_finished_dev = torch.zeros(1, dtype=torch.float32, device=device)
            self.last_audio_codes_dev = z(1, n_tables, dtype=i64)
            self.last_phoneme_tokens_dev = z(1, phoneme_stack, dtype=i64)
            self.codes_buf_dev = z(self._max_frames, n_tables, dtype=i64)
            self.codes_count_dev = z(dtype=i64)

    @torch._dynamo.disable
    def reset_in_place(self, **fields) -> None:
        """Re-initialise this state for a new request without reallocating.

        Tensor addresses stay identical; values are overwritten. Required for
        CUDA-graph replay across requests.

        Args:
            **fields: same keyword arguments as ``__init__`` (minus
                ``device`` / capacities). Missing fields keep their defaults.
        """
        self.audio_bos_id = int(fields.get("audio_bos_id", 0))
        self.audio_eos_id = int(fields.get("audio_eos_id", 0))
        self.phoneme_bos_id = int(fields.get("phoneme_bos_id", 0))
        self.phoneme_eos_id = int(fields.get("phoneme_eos_id", 0))
        self.speech_delay = int(fields.get("speech_delay", 0))
        self.phonemes_delay = int(fields.get("phonemes_delay", 0))
        self.text_eos_id = int(fields.get("text_eos_id", 0))
        self.lt_temperature = float(fields.get("lt_temperature", 0.7))
        self.lt_topk = int(fields.get("lt_topk", 80))
        self.phoneme_unk_token_id = int(fields.get("phoneme_unk_token_id", 0))
        self.phoneme_confidence_unk_threshold = float(
            fields.get("phoneme_confidence_unk_threshold", 0.0)
        )
        self.phoneme_ids = list(fields.get("phoneme_ids", []))
        self.phoneme_pos = 0
        self.text_tokens_seen = 0
        self.phoneme_steps = 0
        self.audio_steps = 0
        self.text_finished = False
        self.phoneme_stream_ended = False
        self.audio_finished = False
        self.decode_started = False
        self.last_audio_codes = None
        self.last_phoneme_tokens = None

        with torch.inference_mode(False):
            self.audio_steps_dev.zero_()
            self.phoneme_steps_dev.zero_()
            self.text_tokens_seen_dev.zero_()
            self.audio_finished_dev.zero_()
            self.last_audio_codes_dev.zero_()
            self.last_phoneme_tokens_dev.zero_()
            self.codes_buf_dev.zero_()
            self.codes_count_dev.zero_()

    @classmethod
    def make_dummy(cls, *, device: torch.device) -> "StreamingHeadState":
        """Build a no-op state for vLLM's warmup trace pass.

        vLLM's warmup runs ``forward()`` with ``_request_state=None`` — the
        streaming-head ops short-circuit and never get traced into the
        compiled / captured graph. Installing this dummy (with
        ``decode_started=True`` so the predict gate opens) lets the trace
        see the full TTS path and bake stable tensor addresses for the
        twins.

        Args:
            device: CUDA device for the twins.

        Returns:
            A ``StreamingHeadState`` with one phoneme and ``decode_started=True``.
        """
        state = cls(
            device=device,
            audio_eos_id=1,
            phoneme_eos_id=1,
            phoneme_ids=[0],
        )
        state.decode_started = True
        return state


class EasyMagpieStreamingHead(nn.Module):
    """All TTS submodules + per-step streaming logic for one request.

    Submodule names (``audio_embeddings``, ``cas_encoder``, ``speaker_encoder``,
    ``local_transformer``, ``phoneme_embeddings`` etc.) match the trained
    checkpoint keys so ``tts_extras.pt`` loads with a single ``streaming_head.``
    prefix prepended.

    TTS submodules are constructed in fp32: ``load_tts_extras`` copies via
    ``target.data.copy_(v.to(target.dtype))``, so a fp16 target would round
    the fp32 source before the copy and lose trained precision.
    """

    def __init__(self) -> None:
        super().__init__()

        from nemo.collections.tts.modules import transformer_2501
        from nemo.collections.tts.modules.magpietts_modules import (
            CharAwareSubwordEncoder,
        )

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
                [nn.Embedding(n_tokens, c["audio_embedding_dim"]) for _ in range(n_tables)]
            )
            if c["audio_embedding_dim"] != embed_dim:
                self.audio_in_projection = nn.Linear(c["audio_embedding_dim"], embed_dim)
                self.audio_out_projection = nn.Linear(hidden, c["audio_embedding_dim"])
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

            self.final_proj = nn.Linear(c["audio_embedding_dim"], n_books * n_tokens * stack)
            self.phoneme_embeddings = nn.ModuleList(
                [nn.Embedding(c["phoneme_vocab_size"], embed_dim)
                 for _ in range(c["phoneme_stacking_factor"])]
            )
            self.phoneme_final_proj = nn.Linear(
                hidden, c["phoneme_vocab_size"] * c["phoneme_stacking_factor"]
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

    def init_request(
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
        device: Optional[torch.device] = None,
    ) -> StreamingHeadState:
        """Build a fresh per-request state.

        Args:
            audio_bos_id / audio_eos_id: audio codebook BOS / EOS ids.
            phoneme_bos_id / phoneme_eos_id: phoneme BOS / EOS ids.
            streaming_speech_delay / streaming_phonemes_delay: post-context
                AR-step delays.
            phoneme_token_ids: per-step CAS subword ids for the utterance.
            text_eos_id: subword tokenizer EOS id.
            lt_temperature / lt_topk: LT sampling knobs.
            phoneme_unk_token_id / phoneme_confidence_unk_threshold: UNK
                substitution config.
            device: CUDA device for twins. Defaults to a parameter's device.

        Returns:
            A fresh ``StreamingHeadState``.
        """
        if device is None:
            device = next(self.parameters()).device
        return StreamingHeadState(
            device=device,
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
            phoneme_ids=phoneme_token_ids,
        )

    @staticmethod
    def _phase_masks(state: StreamingHeadState) -> tuple[bool, bool, bool]:
        """Compute which sub-paths are active at the current AR step.

        Args:
            state: current request state.

        Returns:
            Tuple ``(needs_text, needs_phoneme, needs_audio)``.
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

    def embed_audio_tokens(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        """Embed audio codes by summing across codebooks then projecting.

        Args:
            audio_tokens: ``(B, n_tables, T)`` int64 audio code ids.

        Returns:
            ``(B, T, embedding_dim)`` float embedding.
        """
        n_tables = audio_tokens.size(1)
        emb: Optional[torch.Tensor] = None
        for c in range(n_tables):
            e = self.audio_embeddings[c](audio_tokens[:, c, :])
            emb = e if emb is None else emb + e
        emb = emb / n_tables
        return self.audio_in_projection(emb)

    def encode_speaker(
        self, audio_embedded: torch.Tensor, audio_lens: torch.Tensor
    ) -> torch.Tensor:
        """Run the speaker encoder over the context audio embedding.

        Args:
            audio_embedded: ``(B, T_ctx, embedding_dim)`` from ``embed_audio_tokens``.
            audio_lens: ``(B,)`` int64 valid lengths.

        Returns:
            ``(B, T_ctx, embedding_dim)`` speaker-conditioned embedding.
        """
        from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
        ctx_mask = get_mask_from_lengths(audio_lens)
        return self.speaker_encoder(audio_embedded, ctx_mask, cond=None, cond_mask=None)["output"]

    def embed_text_tokens(
        self,
        text_tokens: torch.Tensor,
        text_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """CAS-only text embedding (``disable_subword_embedding=True`` path).

        Args:
            text_tokens: ``(B, L)`` int64 subword ids.
            text_lens: ``(B,)`` int64 valid lengths, or ``None`` for all-valid.

        Returns:
            ``(B, L, embedding_dim)`` embedding in the audio-embedding dtype.
        """
        if text_lens is None:
            text_lens = torch.full(
                (text_tokens.size(0),), text_tokens.size(1),
                dtype=torch.long, device=text_tokens.device,
            )
        from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
        text_mask = get_mask_from_lengths(text_lens)
        out = self.cas_encoder(text_tokens, subword_mask=text_mask)
        return out.to(self.audio_embeddings[0].weight.dtype)

    def embed_phoneme_tokens(self, phoneme_tokens: torch.Tensor) -> torch.Tensor:
        """Embed phoneme tokens by averaging across the stacking factor.

        Args:
            phoneme_tokens: ``(B, S, T)`` int64 phoneme ids.

        Returns:
            ``(B, T, embedding_dim)`` phoneme embedding.
        """
        emb: Optional[torch.Tensor] = None
        for c in range(phoneme_tokens.size(1)):
            e = self.phoneme_embeddings[c](phoneme_tokens[:, c, :])
            emb = e if emb is None else emb + e
        emb = emb / phoneme_tokens.size(1)
        return emb

    def wire_tokenizer(
        self,
        subword_vocab: dict,
        bos_id: int,
        eos_id: int,
        cfg_unk_token_id: int,
        subword_padding_idx: Optional[int] = None,
    ) -> None:
        """Populate the CAS encoder's subword→char map from a live BPE vocab.

        Must be called once before the first ``/tts/stream`` request,
        otherwise the CAS encoder CUDA-asserts on non-trivial context text.

        Args:
            subword_vocab: mapping subword string → int id.
            bos_id / eos_id: special-token ids in the subword vocab.
            cfg_unk_token_id: the UNK id used for classifier-free guidance.
            subword_padding_idx: padding id; defaults to the SMALLMAMBA constant.
        """
        from nemo.collections.tts.modules.magpietts_modules import build_vocabs

        special = {"<BOS>": bos_id, "<EOS>": eos_id, "<CFG_UNK>": cfg_unk_token_id}
        pad = subword_padding_idx if subword_padding_idx is not None else SMALLMAMBA["cas_subword_padding_idx"]
        subword_id_to_char_ids, char_vocab = build_vocabs(
            subword_vocab=subword_vocab,
            subword_padding_idx=pad,
            special_vocab=special,
        )
        self.cas_encoder.subword_id_to_char_ids = subword_id_to_char_ids
        self.cas_encoder.char_vocab = char_vocab
        logger.info(
            "Tokenizer wired: %d subwords mapped to %d chars",
            len(subword_id_to_char_ids), len(char_vocab),
        )

    def _get_lt_helper(self):
        """Build (and cache) a ``LocalTransformerHelper`` over our LT submodules."""
        if getattr(self, "_lt_helper_cached", None) is not None:
            return self._lt_helper_cached
        from nemo.collections.tts.modules.magpietts_modules import LocalTransformerHelper

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
        helper._fused_temperature = float(os.environ.get("EM_LT_TEMPERATURE", "0.7"))
        helper._fused_topk = int(os.environ.get("EM_LT_TOPK", "80"))
        self._lt_helper_cached = helper
        return helper

    def _get_compiled_lt(self, device: torch.device, dtype: torch.dtype):
        """Build (and cache) a torch.compile'd LT sampler for ``(device, dtype)``."""
        cached = getattr(self, "_compiled_lt_cached", None)
        if cached is not None and cached["device"] == device and cached["dtype"] == dtype:
            return cached["wrapper"]
        from nemo.collections.tts.modules.magpietts_lt_fused import LocalTransformerFusedModule

        helper = self._get_lt_helper()
        wrapper = LocalTransformerFusedModule(
            helper, temperature=helper._fused_temperature, topk=helper._fused_topk,
        )
        wrapper.eval().to(device=device, dtype=dtype)
        mode = os.environ.get("EM_LT_COMPILE_MODE", "reduce-overhead").lower()
        compiled = torch.compile(wrapper, mode=mode, fullgraph=True)
        self._compiled_lt_cached = {"wrapper": compiled, "device": device, "dtype": dtype}
        logger.info(
            "[compiled-lt] mode=%s device=%s dtype=%s topk=%d temperature=%.2f",
            mode, device, dtype, helper._fused_topk, helper._fused_temperature,
        )
        return compiled

    def warmup_compiled_lt(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        iters: int = 3,
    ) -> None:
        """Trigger torch.compile + CUDA-graph capture for the LT at startup.

        Args:
            device: target device; defaults to ``cuda:0``.
            dtype: target dtype; defaults to the head's parameter dtype.
            iters: number of dummy passes to ensure capture stabilises.
        """
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
                u = torch.rand(n_tables, 1, topk, device=device, dtype=torch.float32).clamp_(
                    min=1e-10, max=1.0 - 1e-7
                )
                g = -torch.log(-torch.log(u)).to(dtype)
                _ = wrapper(h, g)
                torch.cuda.synchronize(device)

    def _predict_audio_codes_compiled(
        self, hidden_state: torch.Tensor, state: StreamingHeadState,
    ) -> torch.Tensor:
        """Sample one frame's 16 audio codes via the compiled LT.

        Per-step deterministic Gumbel noise seeded by ``state.audio_steps``.

        Args:
            hidden_state: ``(1, hidden)`` or ``(1, 1, hidden)`` backbone output.
            state: current request state (used for seed).

        Returns:
            ``(1, n_tables=16, 1)`` int64 sampled code ids.
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

        gen_key = "_compile_gumbel_gen_" + str(device).replace(":", "_")
        gen = getattr(self, gen_key, None)
        if gen is None:
            gen = torch.Generator(device=device)
            setattr(self, gen_key, gen)
        gen.manual_seed(state.audio_steps * 7919 + 1)
        u = torch.rand(
            n_tables, B, topk, dtype=torch.float32, device=device, generator=gen,
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
        """Sample one frame via the TRT-fused LT path (``EASYMAGPIE_LT_BACKEND=trt_fused``).

        Args:
            hidden_state: ``(1, hidden)`` or ``(1, 1, hidden)`` backbone output.
            temperature / topk: optional overrides; default to baked helper values.

        Returns:
            ``(1, n_tables, 1)`` int64 sampled code ids.
        """
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

    @torch._dynamo.disable
    def next_input_embedding(
        self,
        state: StreamingHeadState,
        device: torch.device,
        dtype: torch.dtype,
        hidden: int,
    ) -> torch.Tensor:
        """Build the next AR-step input embedding from streaming state.

        ``next_input = text_emb + phoneme_emb + audio_emb`` (each gated by
        the current phase mask). Step-0 vs last-codes selection for audio
        and phoneme is done branchlessly via ``torch.where`` over the
        device-tensor twins so the selection is graph-friendly.

        Args:
            state: current request state. Mutated in place: ``decode_started``
                is set, ``phoneme_pos`` advances, ``text_finished`` flips on
                ``text_eos_id``.
            device: target device for the embedding.
            dtype: target dtype.
            hidden: backbone hidden dim.

        Returns:
            ``(1, hidden)`` embedding for one decode token.
        """
        state.decode_started = True
        next_input = torch.zeros(1, 1, hidden, device=device, dtype=dtype)
        needs_text, needs_phoneme, needs_audio = self._phase_masks(state)

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

        if needs_phoneme:
            S = SMALLMAMBA["phoneme_stacking_factor"]
            phon_bos_in = torch.full(
                (1, S, 1), state.phoneme_bos_id, dtype=torch.long, device=device,
            )
            phon_bos_emb = self.embed_phoneme_tokens(phon_bos_in)
            phon_last_in = state.last_phoneme_tokens_dev.unsqueeze(-1)
            phon_last_emb = self.embed_phoneme_tokens(phon_last_in)
            phon_emb = torch.where(
                (state.phoneme_steps_dev == 0), phon_bos_emb, phon_last_emb,
            )
            next_input = next_input + phon_emb.to(dtype=dtype)

        if needs_audio:
            CS = SMALLMAMBA["num_audio_codebooks"] * SMALLMAMBA["frame_stacking_factor"]
            audio_bos_in = torch.full(
                (1, CS, 1), state.audio_bos_id, dtype=torch.long, device=device,
            )
            audio_bos_emb = self.embed_audio_tokens(audio_bos_in)
            audio_last_in = state.last_audio_codes_dev.unsqueeze(-1)
            audio_last_emb = self.embed_audio_tokens(audio_last_in)
            audio_emb = torch.where(
                (state.audio_steps_dev == 0), audio_bos_emb, audio_last_emb,
            )
            next_input = next_input + audio_emb.to(dtype=dtype)

        return next_input.squeeze(0)

    @torch._dynamo.disable
    def predict_codes_and_advance(
        self, hidden_last: torch.Tensor, state: Optional[StreamingHeadState],
    ) -> None:
        """Sample one audio frame + one phoneme step, advance counters in place.

        Sync-free: all state mutations land on device tensors via
        ``copy_`` / ``add_`` / ``index_copy_``, and EOS detection writes the
        ``audio_finished_dev`` twin so ``compute_logits`` can boost the virtual
        EOS logit without a GPU→CPU sync. The Python ``audio_finished`` flag
        is left one step stale — the sampler-EOS path picks up the boosted
        logit on the next step regardless.

        Short-circuits on ``state is None`` (no request active) or
        ``decode_started=False`` (prefill forward — codes aren't valid yet).

        Args:
            hidden_last: ``(N, hidden)`` or ``(N, T, hidden)`` backbone output;
                the last position is used.
            state: current request state, or ``None``.

        Returns:
            ``None``. All outputs land on ``state``: ``last_audio_codes(_dev)``,
            ``last_phoneme_tokens(_dev)``, ``audio_finished_dev``,
            ``codes_buf_dev``, ``codes_count_dev``, and the Python counters.
        """
        if state is None or not state.decode_started:
            return
        needs_text, needs_phoneme, needs_audio = self._phase_masks(state)

        last_h = (
            hidden_last[-1:]
            if hidden_last.dim() == 2
            else hidden_last.reshape(-1, hidden_last.shape[-1])[-1:]
        )
        last_h_fp32 = last_h.float()

        if needs_phoneme and self.phoneme_final_proj is not None:
            phon_logits = self.phoneme_final_proj(last_h_fp32)
            P = SMALLMAMBA["phoneme_vocab_size"]
            S = SMALLMAMBA["phoneme_stacking_factor"]
            phon_logits = phon_logits.view(1, S, P)
            phon_tok = phon_logits.argmax(dim=-1).long()
            if state.phoneme_confidence_unk_threshold > 0.0:
                max_probs = torch.softmax(phon_logits, dim=-1).max(dim=-1).values
                underconfident = (max_probs < state.phoneme_confidence_unk_threshold).any(dim=1, keepdim=True)
                eos_predicted = (phon_tok == state.phoneme_eos_id).any(dim=1, keepdim=True)
                replace = underconfident & ~eos_predicted
                phon_tok = torch.where(
                    replace, torch.full_like(phon_tok, state.phoneme_unk_token_id), phon_tok,
                )
            state.last_phoneme_tokens_dev.copy_(phon_tok)
            state.phoneme_steps_dev.add_(1)
            state.last_phoneme_tokens = phon_tok
            state.phoneme_steps += 1

        if needs_audio:
            lt_dtype = os.environ.get("EM_LT_DTYPE", "fp32").lower()
            last_h_lt = last_h.half() if lt_dtype == "fp16" else last_h_fp32
            backend = os.environ.get("EASYMAGPIE_LT_BACKEND", "compile").lower()
            if backend == "compile":
                codes = self._predict_audio_codes_compiled(last_h_lt, state)
            else:
                codes = self._predict_audio_codes_trt_fused(
                    last_h_lt, temperature=state.lt_temperature, topk=state.lt_topk,
                )
            codes_flat = codes.squeeze(-1).long()        # (1, n_tables)

            state.last_audio_codes_dev.copy_(codes_flat)
            state.audio_steps_dev.add_(1)
            # Ring buffer write — server's _pull_codes drains and zeros the cursor.
            state.codes_buf_dev.index_copy_(
                0, state.codes_count_dev.unsqueeze(0), codes_flat,
            )
            state.codes_count_dev.add_(1)
            # Audio-EOS detection — tensor-only; compute_logits reads the twin.
            eos_hit = (codes_flat == state.audio_eos_id).any().float().reshape(1)
            state.audio_finished_dev.copy_(
                torch.maximum(state.audio_finished_dev, eos_hit)
            )
            state.last_audio_codes = codes_flat
            state.audio_steps += 1

        state.text_tokens_seen += 1
        state.text_tokens_seen_dev.add_(1)
