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
"""Architecture constants for the EasyMagpieTTS vLLM-Omni model.

These mirror the values baked into the reference EasyMagpieTTS SmallMamba
checkpoint (Nemotron-H hybrid Mamba2 + attention + MoE backbone, 8 codebooks,
frame-stacking ×2, 3-layer autoregressive local transformer).

The vLLM-Omni model reads the bulk of its configuration from the
``hf_config`` provided by vLLM at construction time; this dataclass captures
the TTS-specific scalars that are *not* part of a standard HF text-LM config
and provides a single, well-documented default profile.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Number of trailing special tokens appended to every audio codebook.
# Matches ``len(SpecialAudioToken)`` in
# ``nemo.collections.tts.modules.magpietts_modules`` (BOS, EOS, CONTEXT_BOS,
# CONTEXT_EOS, MASK, RESERVED_1..3).
NUM_SPECIAL_AUDIO_TOKENS: int = 8

# Offsets of the special audio tokens *within* the trailing special-token block
# (i.e. ``codebook_size + <offset>`` is the real embedding-table id).
SPECIAL_AUDIO_BOS: int = 0
SPECIAL_AUDIO_EOS: int = 1
SPECIAL_AUDIO_CONTEXT_BOS: int = 2
SPECIAL_AUDIO_CONTEXT_EOS: int = 3
SPECIAL_AUDIO_MASK: int = 4


@dataclass
class EasyMagpieOmniArch:
    """Static architecture description for an EasyMagpieTTS checkpoint.

    Attributes:
        hidden_dim: Backbone hidden size (``cfg.hidden_dim``).
        embedding_dim: Embedding size feeding the backbone (``cfg.embedding_dim``).
        audio_embedding_dim: Per-codebook audio embedding size
            (``cfg.audio_embedding_dim``); may differ from ``embedding_dim``.
        num_audio_codebooks: Number of codec codebooks (``C``).
        codebook_size: Base codec codebook size (excluding special tokens).
        frame_stacking_factor: Frame stacking factor (``S``). The model treats
            the audio stream as ``C * S`` independent "stacked" codebooks.
        phoneme_stacking_factor: Phoneme stacking factor.
        phoneme_vocab_size: Phoneme tokenizer vocabulary size.
        local_transformer_n_layers / _n_heads / _hidden_dim: local-transformer
            (intra-frame codebook predictor) sizing.
    """

    hidden_dim: int = 1536
    embedding_dim: int = 1536
    audio_embedding_dim: int = 1536

    num_audio_codebooks: int = 8
    codebook_size: int = 1024
    frame_stacking_factor: int = 2

    phoneme_stacking_factor: int = 1
    phoneme_vocab_size: int = 2051

    # Number of multi-mode task ("service token") embeddings. The reference model
    # prepends a single learned per-mode embedding to the prefill context when
    # trained with >1 mode (``cfg.training_modes``); 0 disables it (single-mode
    # checkpoints have no ``task_embedding`` table).
    num_task_embeddings: int = 0

    local_transformer_n_layers: int = 3
    local_transformer_n_heads: int = 12
    local_transformer_hidden_dim: int = 1536

    # Optional per-checkpoint overrides for backward compatibility (legacy
    # checkpoints sometimes forced special-token ids).
    forced_audio_bos_id: int | None = None
    forced_audio_eos_id: int | None = None
    forced_mask_token_id: int | None = None

    extra: dict[str, Any] = field(default_factory=dict)

    # ── Derived quantities ───────────────────────────────────────────
    @property
    def num_stacked_codebooks(self) -> int:
        """Number of independent codebooks the model autoregresses over (``C * S``)."""
        return self.num_audio_codebooks * self.frame_stacking_factor

    @property
    def num_all_tokens_per_codebook(self) -> int:
        """Per-codebook vocabulary size including the trailing special tokens."""
        return self.codebook_size + NUM_SPECIAL_AUDIO_TOKENS

    @property
    def audio_bos_id(self) -> int:
        """Embedding-table id of the audio BOS token."""
        if self.forced_audio_bos_id is not None:
            return self.forced_audio_bos_id
        return self.codebook_size + SPECIAL_AUDIO_BOS

    @property
    def audio_eos_id(self) -> int:
        """Embedding-table id of the audio EOS token."""
        if self.forced_audio_eos_id is not None:
            return self.forced_audio_eos_id
        return self.codebook_size + SPECIAL_AUDIO_EOS

    @property
    def mask_token_id(self) -> int:
        """Embedding-table id of the MaskGit MASK token."""
        if self.forced_mask_token_id is not None:
            return self.forced_mask_token_id
        return self.codebook_size + SPECIAL_AUDIO_MASK

    @classmethod
    def from_hf_config(cls, hf_config: Any) -> "EasyMagpieOmniArch":
        """Build an arch description from a vLLM ``hf_config``.

        Any attribute present on ``hf_config`` overrides the default profile;
        unknown attributes are ignored. This lets a converted checkpoint carry
        its own ``easymagpie`` block in ``config.json`` while still working
        out-of-the-box on the reference SmallMamba profile.
        """
        defaults = cls()
        kwargs: dict[str, Any] = {}
        for f in (
            "hidden_dim",
            "embedding_dim",
            "audio_embedding_dim",
            "num_audio_codebooks",
            "codebook_size",
            "frame_stacking_factor",
            "phoneme_stacking_factor",
            "phoneme_vocab_size",
            "num_task_embeddings",
            "local_transformer_n_layers",
            "local_transformer_n_heads",
            "local_transformer_hidden_dim",
            "forced_audio_bos_id",
            "forced_audio_eos_id",
            "forced_mask_token_id",
        ):
            if hasattr(hf_config, f):
                kwargs[f] = getattr(hf_config, f)
        # ``hidden_size`` is the canonical HF name for the backbone width.
        if "hidden_dim" not in kwargs and hasattr(hf_config, "hidden_size"):
            kwargs["hidden_dim"] = hf_config.hidden_size
            kwargs.setdefault("embedding_dim", hf_config.hidden_size)
        merged = {**defaults.__dict__, **kwargs}
        merged.pop("extra", None)
        return cls(**merged)


# Reference profile: Nemotron-H SmallMamba EasyMagpieTTS checkpoint.
EASYMAGPIE_SMALLMAMBA = EasyMagpieOmniArch()
