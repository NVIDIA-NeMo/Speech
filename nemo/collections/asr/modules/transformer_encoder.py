# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import and_masks, create_block_mask, flex_attention

from nemo.collections.asr.parts.submodules.multi_head_attention import (
    PositionalEncoding,
    RelPositionalEncoding,
    RelPositionMultiHeadAttention,
)
from nemo.collections.asr.parts.submodules.subsampling import FeatureStacking, StackingSubsampling
from nemo.core.classes.module import freeze, unfreeze
from nemo.utils.decorators import experimental

flex_attention_compiled = torch.compile(flex_attention, dynamic=True)


@dataclass
class TransformerEncoderConfig:
    """Configuration for ``TransformerEncoder`` and its sub-blocks.

    Args:
        feat_in: Input feature dimension (e.g. number of mel bins).
        d_model: Transformer encoder state dimension, i.e. the size of the residual stream that flows
            through every block (token/frame embedding size, attention input/output size, and
            feed-forward input/output size). Also known as ``hidden_size`` in HuggingFace
            ``transformers`` configs and ``embed_dim``/``d_model`` in PyTorch's
            ``nn.TransformerEncoderLayer``.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer blocks.
        drop_rate: Dropout probability applied inside attention and feed-forward sublayers.
        qkv_bias: If True, add a learnable bias to the fused Q/K/V projection. Many modern ASR/LM
            Transformers (e.g. HuggingFace Whisper) drop the bias on the K projection because a
            constant K bias adds the same scalar to every key and is wiped out by softmax's
            shift-invariance, making it a redundant parameter. Default ``False`` matches that style.
        qk_norm: If True, apply per-head ``LayerNorm`` to Q and K before the dot product. Stabilizes
            training by preventing exponential Q/K-norm growth and "attention entropy collapse"
            (Henry et al. 2020; used in OLMo 2, Gemma 3, Qwen 3). Cheap, ~no-op for inference.
        ff_expansion: Multiplier for the per-block FFN inner hidden size:
            ``ffn_hidden_size = int(ff_expansion * d_model)``. Only widens the intermediate FFN
            projection; FFN input/output stays at ``d_model``. Typical value ``4.0``; ``float``
            allows sub-1x experts for MoE. Equivalent to ``intermediate_size / hidden_size`` in
            HuggingFace and ``dim_feedforward / d_model`` in PyTorch's ``nn.TransformerEncoderLayer``.
        pre_block_norm: If True, apply ``LayerNorm`` to embeddings before the first Transformer block
            (BERT/ViT-style). Set False to match pre-norm Transformers such as Whisper or GPT-2.
        subsampling_factor: Frame-level subsampling factor performed by the pre-encoder.
        attn_mode: Attention pattern. Currently only ``"full"`` (bidirectional) is supported.
            Future modes: ``"causal"``, ``"lookahead"``, ``"local"``, ``"sliding_window"``.
        self_attention_model: Positional encoding / attention scoring scheme.

            - ``"rel_pos"`` (default): Transformer-XL relative positional encoding
              (https://arxiv.org/abs/1901.02860). The (b)+(d) cross/positional bias is computed
              from the relative-position embedding and injected into FlexAttention via a
              ``score_mod`` closure; the (c) global-content bias is folded into the query as
              ``Q + pos_bias_u``.
            - ``"abs_pos"``: sinusoidal absolute positional encoding added to embeddings
              before the first block; standard scaled dot-product attention.
            - ``"no_pos"`` (or ``None``): no positional encoding at all. The pre-encoder output
              is consumed directly by the Transformer blocks. ``xscaling``, ``pos_emb_max_len``,
              ``dropout_pre_encoder`` and ``dropout_emb`` are unused in this mode.
    """

    feat_in: int = 128
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 17
    drop_rate: float = 0.1
    qkv_bias: bool = False
    qk_norm: bool = False
    ff_expansion: float = 4.0
    pre_block_norm: bool = True
    subsampling_factor: int = 4
    # Attention mode: "full" (bidirectional) or "causal" (each token only attends to itself and earlier tokens).
    # Future: "lookahead", "local", "sliding_window".
    attn_mode: str = "full"
    self_attention_model: str = "rel_pos"


def _make_padding_mod(lengths):
    """Mask out padding positions based on per-sample lengths."""

    def pad_mask(b, h, q_idx, kv_idx):
        return kv_idx < lengths[b]

    return pad_mask


def _make_causal_mod():
    """Strictly causal — each query only attends to its own and earlier kv positions."""

    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    return causal


_SUPPORTED_ATTENTION_MODES = ("full", "causal")
_SUPPORTED_SELF_ATTENTION_MODELS = ("abs_pos", "rel_pos", "no_pos")


class FeedForward(nn.Module):
    def __init__(self, cfg: TransformerEncoderConfig):
        super().__init__()
        ff_hidden = int(cfg.ff_expansion * cfg.d_model)
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, ff_hidden),
            nn.GELU(),
            nn.Dropout(cfg.drop_rate),
            nn.Linear(ff_hidden, cfg.d_model),
            nn.Dropout(cfg.drop_rate),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: TransformerEncoderConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model
        self.self_attention_model = cfg.self_attention_model
        self._uses_rel_pos = self.self_attention_model == "rel_pos"
        if self.self_attention_model not in _SUPPORTED_SELF_ATTENTION_MODELS:
            raise ValueError(
                f"self_attention_model='{self.self_attention_model}' is not supported. "
                f"Supported modes: {_SUPPORTED_SELF_ATTENTION_MODELS}."
            )
        if self.head_dim < 16:
            raise ValueError(
                "PyTorch FlexAttention CUDA backend requires per-head embedding dimension >= 16, "
                f"but got head_dim={self.head_dim} from d_model={self.d_model}, n_heads={self.n_heads}."
            )

        self.w_qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.qkv_bias)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)

        self.qk_norm = cfg.qk_norm
        if cfg.qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

        # Transformer-XL relative-position parameters (matrix b and matrix d from
        # https://arxiv.org/abs/1901.02860 Section 3.3). The "matrix c" term `u @ K^T` is
        # absorbed by passing `Q + pos_bias_u` as the query to FlexAttention.
        if self._uses_rel_pos:
            self.linear_pos = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            self.pos_bias_u = nn.Parameter(torch.zeros(self.n_heads, self.head_dim))
            self.pos_bias_v = nn.Parameter(torch.zeros(self.n_heads, self.head_dim))
        else:
            self.linear_pos = None
            self.pos_bias_u = None
            self.pos_bias_v = None

    def _rel_shift(self, x):
        """Transformer-XL relative-position shift.

        Delegates to ``RelPositionMultiHeadAttention.rel_shift`` (which does not reference
        ``self``) so the logic lives in a single place — NeMo's existing reference
        implementation in ``parts/submodules/multi_head_attention.py``.
        """
        return RelPositionMultiHeadAttention.rel_shift(None, x)

    def _build_rel_pos_score_mod(self, q, pos_emb):
        """Build the FlexAttention inputs that realize Transformer-XL relative attention.

        Implements the (b), (c), (d) terms of Transformer-XL Section 3.3
        (https://arxiv.org/abs/1901.02860) on top of FlexAttention:

        - Matrices (b) + (d) — the position-dependent score bias ``(Q + v) @ R^T`` rel-
          shifted into ``(q_idx, kv_idx)`` coordinates — are precomputed into a
          ``(B, H, T, T)`` tensor, scaled by ``1/sqrt(D)`` (to match FlexAttention's
          already-scaled ``QK^T`` scores), and captured by a local ``score_mod`` closure.
          FlexAttention's ``score_mod`` API fixes the callable signature, so the per-forward
          tensor is threaded in via closure capture rather than as an explicit argument.
          Keeping it local (instead of on ``self``) lets the ``(B, H, T, T)`` bias be freed
          as soon as the layer's attention call returns, so peak memory holds at most one
          layer's bias rather than all layers' biases at once.
        - Matrix (c) — the global-content bias ``u @ K^T`` — is folded into FlexAttention
          by rewriting the query as ``Q + pos_bias_u``, which is returned.

        Args:
            q: Query tensor with shape ``(B, H, T, D)``.
            pos_emb: Relative positional embedding ``(1, 2T - 1, d_model)`` produced by
                ``RelPositionalEncoding``.

        Returns:
            score_mod: Callable to pass as ``flex_attention(..., score_mod=...)``.
            q_with_bias_u: ``Q + pos_bias_u`` — the (c) "matrix c" query rewrite.
        """
        H, D = self.n_heads, self.head_dim
        T = q.size(-2)
        # pos_emb: (1, 2T - 1, d_model) -> p: (1, H, 2T - 1, D)
        p = self.linear_pos(pos_emb).view(pos_emb.size(0), -1, H, D).transpose(1, 2)
        # pos_bias_{u,v}: (H, D) -> (1, H, 1, D) so they broadcast over the (B, H, T, D)
        # Q tensor against the head/depth axes rather than (incorrectly) against time.
        # Match dtype under AMP so fp32 bias params do not upcast q before FlexAttention.
        bias_u = self.pos_bias_u.view(1, H, 1, D).to(dtype=q.dtype)
        bias_v = self.pos_bias_v.view(1, H, 1, D).to(dtype=q.dtype)
        # Matrix b + d: ((Q + v) @ R^T) shifted into (q_idx, kv_idx) space, then scaled
        # by 1/sqrt(D) so it can be added directly to FlexAttention's already-scaled scores.
        q_with_bias_v = q + bias_v
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))  # (B, H, T, 2T - 1)
        # rel_shift converts absolute-relative-position columns into (query, key) columns;
        # keep the first T to land in (B, H, T, T) bias space.
        rel_pos_bias = self._rel_shift(matrix_bd)[..., :T] * (D**-0.5)

        def score_mod(score, b, h, q_idx, kv_idx):
            return score + rel_pos_bias[b, h, q_idx, kv_idx]

        # Matrix c: fold u @ K^T into FlexAttention by rewriting Q as (Q + u).
        return score_mod, q + bias_u

    def forward(self, x, block_mask=None, pos_emb=None):
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim

        qkv = self.w_qkv(x).view(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.qk_norm:
            q = self.q_norm(q).to(v.dtype)
            k = self.k_norm(k).to(v.dtype)

        score_mod = None
        if self._uses_rel_pos:
            score_mod, q = self._build_rel_pos_score_mod(q, pos_emb)

        attn_fn = flex_attention_compiled if q.is_cuda else flex_attention
        out = attn_fn(q, k, v, block_mask=block_mask, score_mod=score_mod)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerEncoderConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.drop = nn.Dropout(cfg.drop_rate)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(self, x, block_mask=None, pos_emb=None):
        x = x + self.drop(self.attn(self.norm1(x), block_mask=block_mask, pos_emb=pos_emb))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


@experimental
class TransformerEncoder(nn.Module):
    """Pre-norm Transformer encoder for ASR.

    Architecture: PreEncode -> PositionalEncoding -> LayerNorm -> N x TransformerBlock -> FinalNorm

    Uses PyTorch FlexAttention for attention computation. On CUDA, mask functions
    are compiled into fused Triton kernels with block-sparse optimization. On CPU,
    FlexAttention falls back to an unfused implementation automatically.

    Args:
        feat_in: Input feature dimension (number of mel bins).
        d_model: Transformer encoder state dimension, i.e. the size of the residual stream that flows
            through every block (token/frame embedding size, attention input/output size, and
            feed-forward input/output size). Also known as ``hidden_size`` in HuggingFace
            ``transformers`` configs and ``embed_dim``/``d_model`` in PyTorch's
            ``nn.TransformerEncoderLayer``.

        n_heads: Number of attention heads.
        n_layers: Number of Transformer blocks.
        feat_out: Output feature dimension. Defaults to ``d_model``.
        subsampling: Subsampling method. Supports ``feature_stacking`` for the
            Transformer-native ``FeatureStacking`` module, plus ``stacking`` and
            ``stacking_norm`` for linear frame stacking.
        subsampling_factor: Subsampling factor for the pre-encoder.
        drop_rate: Dropout probability.
        dropout_pre_encoder: Dropout probability after positional encoding. Defaults to ``drop_rate``.
        dropout_emb: Dropout probability for positional embeddings.
        qkv_bias: If True, add a learnable bias to the fused Q/K/V projection. Many modern ASR/LM
            Transformers (e.g. HuggingFace Whisper) drop the bias on the K projection because a
            constant K bias adds the same scalar to every key and is wiped out by softmax's
            shift-invariance, making it a redundant parameter. Default ``False`` matches that style.
        qk_norm: If True, apply per-head ``LayerNorm`` to Q and K before the dot product. Stabilizes
            training by preventing exponential Q/K-norm growth and "attention entropy collapse"
            (Henry et al. 2020; used in OLMo 2, Gemma 3, Qwen 3). Cheap, ~no-op for inference.
        ff_expansion: Multiplier for the per-block FFN inner hidden size:
            ``ffn_hidden_size = int(ff_expansion * d_model)``. Only widens the intermediate FFN
            projection; FFN input/output stays at ``d_model``. Typical value ``4.0``; ``float``
            allows sub-1x experts for MoE. Equivalent to ``intermediate_size / hidden_size`` in
            HuggingFace and ``dim_feedforward / d_model`` in PyTorch's ``nn.TransformerEncoderLayer``.
        pre_block_norm: If True (default), apply LayerNorm to embeddings before the first
            Transformer block (BERT/ViT-style). Set False to match pre-norm Transformers
            such as Whisper or GPT-2 — required when loading pretrained weights from those
            checkpoints.
        self_attention_model: Type of positional encoding and attention scoring scheme. Mirrors
            the Conformer encoder's ``self_attention_model`` choices, plus a ``"no_pos"`` option:

            - ``"rel_pos"`` (default): Transformer-XL relative positional encoding
              (https://arxiv.org/abs/1901.02860). The relative-position bias is computed in each
              layer and injected into FlexAttention via a ``score_mod`` closure (the (b)+(d)
              terms) plus a ``Q + pos_bias_u`` query rewrite (the (c) term), so the kernel stays
              FlexAttention.
            - ``"abs_pos"``: sinusoidal absolute positional encoding added to the embeddings
              before the first block; standard ``Q @ K^T`` attention via FlexAttention.
            - ``"no_pos"`` (or ``None``): no positional encoding at all — pre-encoder output
              flows straight into ``embed_norm`` and the Transformer blocks. ``xscaling``,
              ``pos_emb_max_len``, ``dropout_pre_encoder`` and ``dropout_emb`` have no effect
              in this mode. ``None`` is accepted as a YAML-friendly alias for ``"no_pos"``
              (an unset field in a config maps to ``None``).

            ``"rel_pos_local_attn"`` is not implemented yet.
        pos_emb_max_len: Initial maximum length for sinusoidal positional embeddings.
        xscaling: If True, scale embeddings by ``sqrt(d_model)`` before adding positional encodings,
            following "Attention Is All You Need" article. Originally intended to balance the magnitude
            of small-variance token embeddings against unit-bounded sinusoidal positions and to keep
            tied input/pre-softmax logits well-scaled. With modern unit-variance ``nn.Linear``
            pre-encoders and the LayerNorm directly after the positional sum, this scaling is
            largely a no-op for activation magnitudes. Only meaningful when ``pre_block_norm=False``
            or when matching pretrained checkpoints that expect this scaling.
        attn_mode: Attention pattern — currently only "full" (bidirectional) is supported.
        sync_max_audio_length: When true, sync positional encoding allocation length across distributed ranks.
    """

    def __init__(
        self,
        feat_in: int = 128,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 17,
        feat_out: int = -1,
        subsampling: str = 'feature_stacking',
        subsampling_factor: int = 4,
        drop_rate: float = 0.1,
        dropout_pre_encoder: float = None,
        dropout_emb: float = 0.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        ff_expansion: float = 4.0,
        pre_block_norm: bool = True,
        self_attention_model: Optional[str] = "rel_pos",
        pos_emb_max_len: int = 5000,
        xscaling: bool = False,
        attn_mode: str = "full",
        sync_max_audio_length: bool = True,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        if attn_mode not in _SUPPORTED_ATTENTION_MODES:
            raise ValueError(
                f"attn_mode='{attn_mode}' is not yet supported. Supported modes: {_SUPPORTED_ATTENTION_MODES}."
            )
        # ``None`` is accepted as a YAML-friendly alias for ``"no_pos"`` (an unset field in a
        # config simply maps to None) — normalize here so the rest of the module only deals with
        # the string form.
        if self_attention_model is None:
            self_attention_model = "no_pos"
        if self_attention_model not in _SUPPORTED_SELF_ATTENTION_MODELS:
            raise ValueError(
                f"self_attention_model='{self_attention_model}' is not supported. "
                "Currently only 'abs_pos', 'rel_pos', and 'no_pos' (or None) are available."
            )
        if dropout_pre_encoder is None:
            dropout_pre_encoder = drop_rate

        cfg = TransformerEncoderConfig(
            feat_in=feat_in,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            drop_rate=drop_rate,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            ff_expansion=ff_expansion,
            pre_block_norm=pre_block_norm,
            subsampling_factor=subsampling_factor,
            attn_mode=attn_mode,
            self_attention_model=self_attention_model,
        )
        self.d_model = d_model
        self.n_layers = n_layers
        self._feat_in = feat_in
        self.subsampling = subsampling
        self.subsampling_factor = subsampling_factor
        self.sync_max_audio_length = sync_max_audio_length
        self.self_attention_model = self_attention_model
        self.attn_mode = attn_mode

        if subsampling == 'feature_stacking':
            self.pre_encode = FeatureStacking(subsampling_factor, feat_in, d_model)
        elif subsampling and subsampling_factor > 1:
            if subsampling in ['stacking', 'stacking_norm']:
                self.pre_encode = StackingSubsampling(
                    subsampling_factor=subsampling_factor,
                    feat_in=feat_in,
                    feat_out=d_model,
                    norm=True if subsampling == 'stacking_norm' else False,
                )
            else:
                raise ValueError(
                    f"subsampling='{subsampling}' is not supported. "
                    "Currently only 'feature_stacking', 'stacking', and 'stacking_norm' are available."
                )
        else:
            self.pre_encode = nn.Linear(feat_in, d_model)

        self._feat_out = d_model
        if xscaling:
            self.xscale = math.sqrt(d_model)
        else:
            self.xscale = None
        self.pos_emb_max_len = pos_emb_max_len
        if self_attention_model == "rel_pos":
            self.pos_enc = RelPositionalEncoding(
                d_model=d_model,
                dropout_rate=dropout_pre_encoder,
                max_len=pos_emb_max_len,
                xscale=self.xscale,
                dropout_rate_emb=dropout_emb,
            )
        elif self_attention_model == "abs_pos":
            self.pos_enc = PositionalEncoding(
                d_model=d_model,
                dropout_rate=dropout_pre_encoder,
                max_len=pos_emb_max_len,
                xscale=self.xscale,
                dropout_rate_emb=dropout_emb,
            )
        else:  # "no_pos"
            self.pos_enc = None
        self.embed_norm = nn.LayerNorm(d_model) if pre_block_norm else nn.Identity()
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        if feat_out > 0 and feat_out != self._feat_out:
            self.out_proj = nn.Linear(self._feat_out, feat_out)
            self._feat_out = feat_out
        else:
            self.out_proj = None

        self.set_max_audio_length(self.pos_emb_max_len)

    def forward(self, audio_signal, length, bypass_pre_encode=False):
        """
        Args:
            audio_signal: ``(B, C, T)`` mel spectrogram when ``bypass_pre_encode=False``,
                or ``(B, T, D)`` pre-encoded embeddings when ``bypass_pre_encode=True``.
            length: (B,) — valid frame counts per sample.
            bypass_pre_encode: If true, skip the pre-encoder and consume frame-level embeddings.
                               This option is used when pre-encoded embeddings are used as speaker cache
                               such as speaker diarization models.

        Returns:
            x: (B, D, T') — encoded representation (channels-first).
            length: (B,) — output lengths after subsampling.
        """
        if not bypass_pre_encode and audio_signal.shape[-2] != self._feat_in:
            raise ValueError(
                f"If bypass_pre_encode is False, audio_signal should have shape "
                f"(batch, {self._feat_in}, n_frame) but got last dimension {audio_signal.shape[-2]}."
            )
        if bypass_pre_encode and audio_signal.shape[-1] != self.d_model:
            raise ValueError(
                f"If bypass_pre_encode is True, audio_signal should have shape "
                f"(batch, n_frame, {self.d_model}) but got last dimension {audio_signal.shape[-1]}."
            )

        if bypass_pre_encode:
            self.update_max_seq_length(seq_length=audio_signal.size(1), device=audio_signal.device)
        else:
            self.update_max_seq_length(seq_length=audio_signal.size(2), device=audio_signal.device)
        return self.forward_internal(audio_signal, length, bypass_pre_encode=bypass_pre_encode)

    def forward_internal(self, audio_signal, length, bypass_pre_encode=False):
        if length is None:
            length = audio_signal.new_full(
                (audio_signal.size(0),),
                audio_signal.size(1) if bypass_pre_encode else audio_signal.size(-1),
                dtype=torch.int64,
                device=audio_signal.device,
            )

        if not bypass_pre_encode:
            if isinstance(self.pre_encode, FeatureStacking):
                x, length = self.pre_encode(audio_signal, length)
            else:
                x = torch.transpose(audio_signal, 1, 2)
            if isinstance(self.pre_encode, nn.Linear):
                x = self.pre_encode(x)
            elif not isinstance(self.pre_encode, FeatureStacking):
                x, length = self.pre_encode(x=x, lengths=length)
            length = length.to(torch.int64)
        else:
            x = audio_signal
            length = length.to(torch.int64)

        if self.pos_enc is not None:
            x, pos_emb = self.pos_enc(x=x)
        else:  # "no_pos": pre-encoder output flows in unchanged
            pos_emb = None
        x = self.embed_norm(x)

        B, T, _ = x.shape
        if self.attn_mode == "causal":
            mask_mod = and_masks(_make_causal_mod(), _make_padding_mod(length))
        else:
            mask_mod = _make_padding_mod(length)
        block_mask = create_block_mask(mask_mod, B=B, H=1, Q_LEN=T, KV_LEN=T, device=x.device)
        # For ``abs_pos`` the positional information is already baked into ``x``, so we don't
        # need to thread ``pos_emb`` through each layer; only ``rel_pos`` consumes it.
        layer_pos_emb = pos_emb if self.self_attention_model == "rel_pos" else None
        for layer in self.layers:
            x = layer(x, block_mask=block_mask, pos_emb=layer_pos_emb)

        x = self.final_norm(x)
        if self.out_proj is not None:
            x = self.out_proj(x)
        x = x.transpose(1, 2)  # (B, T, D) -> (B, D, T)
        length = length.to(dtype=torch.int64)
        return x, length

    def update_max_seq_length(self, seq_length: int, device):
        """
        Updates the maximum sequence length for positional encodings.

        Args:
            seq_length: New maximum sequence length.
            device: Device to use for computations.
        """
        if self.sync_max_audio_length and torch.distributed.is_initialized():
            global_max_len = torch.tensor([seq_length], dtype=torch.float32, device=device)
            torch.distributed.all_reduce(global_max_len, op=torch.distributed.ReduceOp.MAX)
            seq_length = global_max_len.int().item()

        if seq_length > self.max_audio_length:
            self.set_max_audio_length(seq_length)

    def set_max_audio_length(self, max_audio_length):
        """Sets maximum input length and extends positional encodings if needed."""
        self.max_audio_length = max_audio_length
        if self.pos_enc is None:  # "no_pos" mode has no buffer to extend
            return
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self.pos_enc.extend_pe(max_audio_length, device, dtype)

    def freeze(self) -> None:
        freeze(self)

    def unfreeze(self, partial: bool = False) -> None:
        unfreeze(self, partial=partial)
