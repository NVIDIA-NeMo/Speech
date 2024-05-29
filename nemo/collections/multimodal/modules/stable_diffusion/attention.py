# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import os
from inspect import isfunction

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn
from torch._dynamo import disable

if os.environ.get("USE_NATIVE_GROUP_NORM", "0") == "1":
    from nemo.gn_native import GroupNormNormlization as GroupNorm
else:
    try:
        from apex.contrib.group_norm import GroupNorm

        OPT_GROUP_NORM = True
    except Exception:
        print('Fused optimized group norm has not been installed.')
        OPT_GROUP_NORM = False

from nemo.collections.multimodal.modules.stable_diffusion.diffusionmodules.util import checkpoint
from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    AdapterName,
    ParallelLinearAdapterConfig,
)
from nemo.core import adapter_mixins
from nemo.utils import logging

try:
    from transformer_engine.pytorch.module import LayerNormLinear, LayerNormMLP

    HAVE_TE = True

except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


def check_cuda():
    if not torch.cuda.is_available():
        raise ImportError('CUDA is not available')
    cur_device = torch.cuda.current_device()
    dprops = torch.cuda.get_device_properties(cur_device)

    is_sm75 = dprops.major == 7 and dprops.minor == 5
    is_sm8x = dprops.major == 8 and dprops.minor >= 0
    is_sm90 = dprops.major == 9 and dprops.minor >= 0

    return is_sm8x or is_sm75 or is_sm90


try:
    import torch.nn as nn
    from flash_attn.modules.mha import FlashCrossAttention, FlashSelfAttention

    flash_attn_installed = check_cuda()

    # Disable TorchDynamo on FlashAttention
    FlashSelfAttention.forward = disable(FlashSelfAttention.forward)
    FlashCrossAttention.forward = disable(FlashCrossAttention.forward)
except ImportError:
    flash_attn_installed = False


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    if isinstance(d, (torch.Tensor, float, int)):
        return d
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = LinearWrapper(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0, use_te=False):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if use_te:
            activation = 'gelu' if not glu else 'geglu'
            # TODO: more parameters to be confirmed, dropout, seq_length
            self.net = LayerNormMLP(
                hidden_size=dim,
                ffn_hidden_size=inner_dim,
                activation=activation,
            )
        else:
            norm = nn.LayerNorm(dim)
            project_in = nn.Sequential(LinearWrapper(dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)
            self.net = nn.Sequential(norm, project_in, nn.Dropout(dropout), LinearWrapper(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels, num_groups=32, act=""):
    return GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True, act=act)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x + h_


# b n (h d) -> (b h) n d
def rearrange_heads_outer(t: torch.Tensor, h: int) -> torch.Tensor:
    b, n, ch = t.shape
    return t.view(b, n, h, -1).transpose(1, 2).reshape(b * h, n, -1)


# (b h) n d -> b n (h d)
def rearrange_heads_inner(t: torch.Tensor, h: int) -> torch.Tensor:
    b = t.shape[0] // h
    n = t.shape[1]
    return t.view(b, h, n, -1).transpose(1, 2).reshape(b, n, -1)


class LinearWrapper(nn.Linear, adapter_mixins.AdapterModuleMixin):
    def __init__(self, in_features, out_features, bias=True, lora_network_alpha=None):
        super().__init__(in_features, out_features, bias)
        self.set_accepted_adapter_types([ParallelLinearAdapterConfig._target_])
        self.lora_network_alpha = lora_network_alpha

    def forward(self, x):
        mixed_x = super().forward(x)
        if self.is_adapter_available():
            lora_linear_adapter = self.get_adapter_module(AdapterName.PARALLEL_LINEAR_ADAPTER)
            lora_mixed_x = lora_linear_adapter(x)
            # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
            # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
            if self.lora_network_alpha:
                mixed_x = mixed_x + lora_mixed_x * (self.lora_network_alpha / lora_linear_adapter.dim)
            else:
                mixed_x = mixed_x + lora_mixed_x
        return mixed_x

    def add_adapter(self, name, cfg, **kwargs):
        self.lora_network_alpha = cfg.network_alpha
        kwargs = {}
        adapter_mixins.AdapterModuleMixin.add_adapter(self, name, cfg, **kwargs)


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        use_flash_attention=False,
        lora_network_alpha=None,
        use_te=False,
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        if context_dim is None:
            self.is_self_attn = True
        else:
            self.is_self_attn = False  # cross-attention
        context_dim = default(context_dim, query_dim)
        # make attention part be aware of self-attention/cross-attention
        self.context_dim = context_dim
        self.query_dim = query_dim
        self.dim_head = dim_head

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_k = LinearWrapper(context_dim, self.inner_dim, bias=False, lora_network_alpha=lora_network_alpha)
        self.to_v = LinearWrapper(context_dim, self.inner_dim, bias=False, lora_network_alpha=lora_network_alpha)

        self.use_te = use_te
        if use_te:
            return_layernorm_output = True if self.is_self_attn else False
            self.norm_to_q = LayerNormLinear(
                query_dim, self.inner_dim, bias=False, return_layernorm_output=return_layernorm_output
            )
        else:
            self.norm = nn.LayerNorm(query_dim)
            self.to_q = LinearWrapper(query_dim, self.inner_dim, bias=False)

        self.to_out = nn.Sequential(
            LinearWrapper(self.inner_dim, query_dim, lora_network_alpha=lora_network_alpha), nn.Dropout(dropout)
        )
        self.use_flash_attention = use_flash_attention

        if dim_head <= 160 and (dim_head % 8) == 0 and flash_attn_installed:
            if context_dim == query_dim:
                self.flash_attn = FlashSelfAttention(softmax_scale=self.scale)
            else:
                self.flash_attn = FlashCrossAttention(softmax_scale=self.scale)

    def forward(self, x, context=None, mask=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        h = self.heads

        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)

        if self.use_te:
            q_out = self.norm_to_q(x)
            if self.is_self_attn:
                q, ln_out = q_out
                context = default(context, ln_out)
            else:
                q = q_out
                context = default(context, x)
        else:
            x = self.norm(x)
            q = self.to_q(x)
            context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            n_cp = x.shape[0] // n_times_crossframe_attn_in_self
            k = repeat(k[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp)
            v = repeat(v[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp)

        out = self._attention(q, k, v, mask, additional_tokens=None)

        return self.to_out(out)

    def _attention(self, q, k, v, mask=None, additional_tokens=None):
        h = self.heads

        if (
            not flash_attn_installed
            or not self.use_flash_attention
            or q.dtype == torch.float32
            or (self.dim_head > 160 or (self.dim_head % 8) != 0)
            or mask is not None
        ):
            # original implementation
            # b n (h d) -> (b h) n d
            q = rearrange_heads_outer(q, h)
            k = rearrange_heads_outer(k, h)
            v = rearrange_heads_outer(v, h)

            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

            if exists(mask):
                # standard stable diffusion does not run into here
                mask = mask.view(mask.shape[0], -1)
                b, j = mask.shape
                mask = mask.unsqueeze(1).expand(b, h, j).reshape(b * h, 1, j)  # b j -> (b h) () j
                sim.masked_fill_(~mask, self.max_neg[sim.dtype])

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)

            out = einsum('b i j, b j d -> b i d', attn, v)

            # (b h) n d -> b n (h d)
            out = rearrange_heads_inner(out, h)
        elif self.context_dim == self.query_dim:
            # self-attention
            qkv = torch.stack([q, k, v], dim=2)
            b, s, t, hd = qkv.shape
            d = hd // h
            qkv = qkv.view(b, s, t, h, d)

            out = self.flash_attn(qkv)
            out = out.view(b, s, hd)
        else:
            # cross-attention
            kv = torch.stack([k, v], dim=2)

            s_q = q.shape[1]
            b, s_kv, t, hd = kv.shape
            d = hd // h

            q = q.view(b, s_q, h, d)
            kv = kv.view(b, s_kv, t, h, d)

            out = self.flash_attn(q, kv)
            out = out.view(b, s_q, hd)
        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        return out


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        use_checkpoint=False,
        use_flash_attention=False,
        disable_self_attn=False,
        lora_network_alpha=None,
        use_te=False,
    ):
        super().__init__()
        self.disable_self_attn = disable_self_attn
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            use_flash_attention=use_flash_attention,
            context_dim=context_dim if self.disable_self_attn else None,
            lora_network_alpha=lora_network_alpha,
            use_te=use_te,
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff, use_te=use_te)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            use_flash_attention=use_flash_attention,
            lora_network_alpha=lora_network_alpha,
            use_te=use_te,
        )  # is self-attn if context is none
        self.use_checkpoint = use_checkpoint

    def forward(self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        kwargs = {"x": x}

        if context is not None:
            kwargs.update({"context": context})
        if additional_tokens is not None:
            kwargs.update({"additional_tokens": additional_tokens})

        if n_times_crossframe_attn_in_self:
            kwargs.update({"n_times_crossframe_attn_in_self": n_times_crossframe_attn_in_self})

        if self.use_checkpoint:
            return checkpoint(self._forward, (x, context), self.parameters(), self.use_checkpoint)
        else:
            return self._forward(x, context)

    def _forward(self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        x = (
            self.attn1(
                x,
                context=context if self.disable_self_attn else None,
                additional_tokens=additional_tokens,
                n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self if not self.disable_self_attn else 0,
            )
            + x
        )
        x = self.attn2(x, context=context, additional_tokens=additional_tokens) + x
        x = self.ff(x) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        use_checkpoint=False,
        use_flash_attention=False,
        lora_network_alpha=None,
        use_te=False,
    ):
        super().__init__()
        logging.info(
            f"constructing {self.__class__.__name__} of depth {depth} w/ {in_channels} channels and {n_heads} heads"
        )
        from omegaconf import ListConfig

        if exists(context_dim) and not isinstance(context_dim, (list, ListConfig)):
            context_dim = [context_dim]
        if exists(context_dim) and isinstance(context_dim, list):
            if depth != len(context_dim):
                logging.info(
                    f"WARNING: {self.__class__.__name__}: Found context dims {context_dim} of depth {len(context_dim)}, "
                    f"which does not match the specified 'depth' of {depth}. Setting context_dim to {depth * [context_dim[0]]} now."
                )
                # depth does not match context dims.
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), "need homogenous context_dim to match depth automatically"
                context_dim = depth * [context_dim[0]]
        elif context_dim is None:
            context_dim = [None] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    use_checkpoint=use_checkpoint,
                    use_flash_attention=use_flash_attention,
                    disable_self_attn=disable_self_attn,
                    lora_network_alpha=lora_network_alpha,
                    use_te=use_te,
                )
                for d in range(depth)
            ]
        )

        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
            # self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
            # Usually inner_dim is the same as in_channels.
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = x.view(b, c, -1).transpose(1, 2)  # b c h w -> b (h w) c
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            if i > 0 and len(context) == 1:
                i = 0  # use same context for each block
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = x.transpose(1, 2).view(b, c, h, w)  # b (h w) c -> b c h w
        if not self.use_linear:
            x = self.proj_out(x)
        return x_in + x
