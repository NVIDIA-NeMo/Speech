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

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from nemo.utils.decorators import experimental

flex_attention_compiled = torch.compile(flex_attention, dynamic=True)


@dataclass
class TransformerEncoderConfig:
    feat_in: int = 80
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 17
    drop_rate: float = 0.1
    qkv_bias: bool = False
    qk_norm: bool = False
    ff_expansion: float = 4.0
    pre_block_norm: bool = True
    subsampling_factor: int = 4
    # Attention mode — currently only "full" is supported.
    # Future: "causal", "lookahead", "local", "sliding_window"
    attn_mode: str = "full"


def _make_padding_mod(lengths):
    """Mask out padding positions based on per-sample lengths."""

    def pad_mask(b, h, q_idx, kv_idx):
        return kv_idx < lengths[b]

    return pad_mask


class FeatureStacking(nn.Module):
    """Stacks consecutive input frames and projects to model dimension.

    Reduces the temporal resolution by ``subsampling_factor`` while increasing
    the feature dimension proportionally, then linearly projects back to
    ``feat_out``.

    Args:
        subsampling_factor: Number of consecutive frames to stack.
        feat_in: Input feature dimension (e.g. number of mel bins).
        feat_out: Output feature dimension (model hidden size).
    """

    def __init__(self, subsampling_factor: int, feat_in: int, feat_out: int):
        super().__init__()
        self.subsampling_factor = subsampling_factor
        self.proj = nn.Linear(subsampling_factor * feat_in, feat_out, bias=False)

    def compute_num_out_frames(self, in_frames):
        return (in_frames + self.subsampling_factor - 1) // self.subsampling_factor

    def forward(self, x, lengths):
        """
        Args:
            x: (B, C, T) — input features (channels-first from preprocessor).
            lengths: (B,) — valid lengths per sample.
        Returns:
            x: (B, T', feat_out) — stacked and projected features.
            lengths: (B,) — updated lengths after subsampling.
        """
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        b, t, c = x.size()
        pad_size = (self.subsampling_factor - (t % self.subsampling_factor)) % self.subsampling_factor
        if pad_size > 0:
            x = nn.functional.pad(x, (0, 0, 0, pad_size))
        t_new = (t + pad_size) // self.subsampling_factor
        x = x.reshape(b, t_new, c * self.subsampling_factor)
        x = self.proj(x)
        lengths = self.compute_num_out_frames(lengths)
        return x, lengths


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

        self.w_qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.qkv_bias)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)

        self.qk_norm = cfg.qk_norm
        if cfg.qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(self, x, block_mask=None):
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim

        qkv = self.w_qkv(x).view(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.qk_norm:
            q = self.q_norm(q).to(v.dtype)
            k = self.k_norm(k).to(v.dtype)

        out = flex_attention_compiled(q, k, v, block_mask=block_mask)
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

    def forward(self, x, block_mask=None):
        x = x + self.drop(self.attn(self.norm1(x), block_mask=block_mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


@experimental
class TransformerEncoder(nn.Module):
    """Pre-norm Transformer encoder for ASR.

    Architecture: FeatureStacking -> LayerNorm -> N x TransformerBlock -> FinalNorm

    Uses PyTorch FlexAttention for attention computation. On CUDA, mask functions
    are compiled into fused Triton kernels with block-sparse optimization. On CPU,
    FlexAttention falls back to an unfused implementation automatically.

    Args:
        feat_in: Input feature dimension (number of mel bins).
        d_model: Transformer hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer blocks.
        drop_rate: Dropout probability.
        qkv_bias: Whether to use bias in Q/K/V projections.
        qk_norm: Whether to apply per-head LayerNorm to Q and K before the dot product.
        ff_expansion: Feed-forward expansion factor (float to support sub-1x for MoE).
        pre_block_norm: If True (default), apply LayerNorm to embeddings before the first
            transformer block (BERT/ViT-style). Set False to match pre-norm transformers
            such as Whisper or GPT-2 — required when loading pretrained weights from those
            checkpoints.
        subsampling_factor: Frame stacking factor for the pre-encoder.
        attn_mode: Attention pattern — currently only "full" (bidirectional) is supported.
    """

    def __init__(
        self,
        feat_in: int = 80,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 17,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        ff_expansion: float = 4.0,
        pre_block_norm: bool = True,
        subsampling_factor: int = 4,
        attn_mode: str = "full",
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        if attn_mode != "full":
            raise ValueError(f"attn_mode='{attn_mode}' is not yet supported. Currently only 'full' is available.")

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
        )
        self.d_model = d_model

        self.pre_encode = FeatureStacking(subsampling_factor, feat_in, d_model)
        self.embed_norm = nn.LayerNorm(d_model) if pre_block_norm else nn.Identity()
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, audio_signal, length):
        """
        Args:
            audio_signal: (B, C, T) — mel spectrogram from preprocessor.
            length: (B,) — valid frame counts per sample.
        Returns:
            x: (B, D, T') — encoded representation (channels-first).
            length: (B,) — output lengths after subsampling.
        """
        x, length = self.pre_encode(audio_signal, length)

        x = self.embed_norm(x)

        B, T, _ = x.shape
        block_mask = create_block_mask(_make_padding_mod(length), B=B, H=1, Q_LEN=T, KV_LEN=T, device=x.device)

        for layer in self.layers:
            x = layer(x, block_mask=block_mask)

        x = self.final_norm(x)
        x = x.transpose(1, 2)  # (B, T, D) -> (B, D, T)
        return x, length

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def unfreeze(self, partial: bool = False):
        for p in self.parameters():
            p.requires_grad = True
        if not partial:
            self.train()
