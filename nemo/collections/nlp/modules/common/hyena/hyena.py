# Implementation of Hyena operator
#
# Michael Poli and Stefano Massaroli and Eric Nguyen and Daniel Y Fu and Tri Dao and Stephen Baccus and
# Yoshua Bengio and Stefano Ermon and Christopher Re,
# Hyena Hierarchy: Towards Larger Convolutional Language Models
# 2023, https://arxiv.org/abs/2302.10866
#
# Multi-head variant introduced in:
#
# Stefano Massaroli and Michael Poli and Daniel Y Fu and Hermann Kumbong and Rom Nishijima Parnichkun and
# David W. Romero and Aman Timalsina and Quinn McIntyre and Beidi Chen and Atri Rudra and Ce Zhang and
# Christopher Re and Stefano Ermon and Yoshua Bengio,
# Laughing Hyena Distillery: Extracting Compact Recurrences From Convolutions
# NeurIPS 2023, https://arxiv.org/abs/2310.18780
#
# Code is heavily based on the reference implementations from:
# https://github.com/HazyResearch/safari/blob/flashfftconv/src/models/sequence/hyena.py
# https://github.com/athms/mad-lab/blob/main/mad/model/layers/hyena.py

from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
from einops import rearrange
from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear, TERowParallelLinear
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig

from nemo.collections.common.parts.utils import activation_registry
from nemo.collections.nlp.modules.common.hyena.hyena_filter import HyenaFilter, HyenaFilterSubmodules
from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision
from nemo.utils.metaclasses import Singleton

try:
    from nemo.collections.nlp.modules.common.hyena.fftconv_wrapper import fftconv_func as safari_fftconv_fn

    HAVE_SAFARI_FFTCONV = True
except ImportError:
    HAVE_SAFARI_FFTCONV = False

try:
    from flashfftconv import FlashFFTConv as FlashFFTConvImpl

    HAVE_FLASHFFTCONV = True

    class FlashFFTConv(metaclass=Singleton):
        # Recommendation is to create single instance per model
        # https://github.com/HazyResearch/flash-fft-conv?tab=readme-ov-file#example-model
        def __init__(self, seqlen, dtype):
            self.flashfftconv = FlashFFTConvImpl(seqlen, dtype)

except ImportError:
    HAVE_FLASHFFTCONV = False

try:
    from causal_conv1d import causal_conv1d_fn

    HAVE_CAUSAL_CONV1D = True
except ImportError:
    HAVE_CAUSAL_CONV1D = False


@dataclass
class HyenaOperatorSubmodules:
    in_proj: Union[ModuleSpec, type] = IdentityOp
    short_filter: Union[ModuleSpec, type] = IdentityFuncOp
    implicit_filter: Union[ModuleSpec, type] = IdentityOp
    out_proj: Union[ModuleSpec, type] = IdentityOp


def auto_assign_attrs(cls, **kwargs):
    for k, v in kwargs.items():
        setattr(cls, k, v)


class CausalDepthWiseConv1d(nn.Module):
    def __init__(self, channels, width, bias=True):
        if not HAVE_CAUSAL_CONV1D:
            raise ImportError("Missing causal-conv1d library, please run 'pip install causal-conv1d'")

        super().__init__()
        self.channels = channels
        self.width = width
        self._conv_1d = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=width,
            padding=width - 1,
            groups=channels,
            bias=bias,
        )

    def forward(self, x):
        return causal_conv1d_fn(x, self._conv_1d.weight.squeeze(1), self._conv_1d.bias)


class HyenaConv(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_seq_length: int,
        order: int,
        bias: bool = True,
        filter_cls: Union[ModuleSpec, type] = HyenaFilter,
        filter_submodules: HyenaFilterSubmodules = None,
        **filter_kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.order = order
        self.max_seq_length = max_seq_length
        self.use_bias = bias
        bias_shape = self.d_model * (self.order - 1)
        if self.use_bias:
            self.bias = nn.Parameter(torch.randn(bias_shape))
        else:
            self.bias = torch.zeros(bias_shape)

        self.filter = build_module(
            filter_cls,
            self.d_model * (self.order - 1),
            submodules=filter_submodules,
            seq_len=max_seq_length,
            **filter_kwargs,
        )


class SingleHeadHyenaConv(HyenaConv):
    def __init__(
        self,
        d_model: int,
        max_seq_length: int,
        order: int,
        bias: bool = True,
        filter_cls: Union[ModuleSpec, type] = HyenaFilter,
        filter_submodules: HyenaFilterSubmodules = None,
        fftconv_type: str = None,
        precision: str = 'bf16',
        **filter_kwargs,
    ):
        super().__init__(
            d_model,
            max_seq_length,
            order,
            bias=bias,
            filter_cls=filter_cls,
            filter_submodules=filter_submodules,
            **filter_kwargs,
        )

        if fftconv_type is None:
            if max_seq_length <= 8192 and HAVE_SAFARI_FFTCONV:
                # safari-fftconv supports seq-len <= 8192 and is a bit faster vs. flashfftconv
                fftconv_type = 'safari'
            else:
                fftconv_type = 'flash'

        if fftconv_type not in ['safari', 'flash']:
            raise ValueError("fftconv_type must be one of ['safari', 'flash']")
        if fftconv_type == 'safari' and max_seq_length > 8192:
            raise ValueError('Safari-fftconv only supports sequence length up to 8192')
        if fftconv_type == 'safari' and not HAVE_SAFARI_FFTCONV:
            raise ImportError('Safari-fftconv library not found. Please see README at <tbd> for instructions.')
        if fftconv_type == 'flash' and not HAVE_FLASHFFTCONV:
            raise ImportError('flashfftconv library not found. Please see README at <tbd> for instructions.')

        if fftconv_type == 'safari':
            self.fftconv_fn = self._safari_fft
        else:  # fftconv_type == 'flash'
            self.flashfftconv = FlashFFTConv(
                2 * self.max_seq_length, torch_dtype_from_precision(precision)
            ).flashfftconv
            self.fftconv_fn = self._flash_fft

    def _safari_fft(self, x, k, bias):
        bias = bias.to(dtype=torch.float32)
        return safari_fftconv_fn(x, k, bias, gelu=False)

    def _flash_fft(self, x, k, bias):
        x = x.contiguous()
        y = self.flashfftconv(x, k) + x * bias.unsqueeze(dim=1)
        return y

    def forward(self, x, k, recurrence_idx):
        bias = rearrange(self.bias, '(v o) -> o v', v=self.d_model, o=self.order - 1)[recurrence_idx]
        y = self.fftconv_fn(x, k, bias)
        return y


class MultiHeadHyenaConv(HyenaConv):
    def __init__(
        self,
        d_model: int,
        max_seq_length: int,
        order: int,
        num_heads: int,
        bias: bool = True,
        filter_cls: Union[ModuleSpec, type] = HyenaFilter,
        filter_submodules: HyenaFilterSubmodules = None,
        fftconv_type: str = None,
        precision: str = 'bf16',
        **filter_kwargs,
    ):
        if num_heads == 1:
            raise ValueError('Expecting num_heads > 1')
        if order != 2:
            raise ValueError(f'Multi-head supported only with order == 2 (got order {self.order})')
        if not HAVE_SAFARI_FFTCONV:
            raise ImportError('Safari-fftconv library not found. Please see README at <tbd> for instructions.')

        super().__init__(
            d_model,
            max_seq_length,
            order,
            bias=bias,
            filter_cls=filter_cls,
            filter_submodules=filter_submodules,
            **filter_kwargs,
        )
        self.num_heads = num_heads

    def forward(self, v, k, x1, x2):
        bias = self.bias.to(dtype=torch.float32)
        y = safari_fftconv_fn(v, k, bias, gelu=False, output_hbl_layout=True, v=x2, head_dim=self.num_heads, q=x1)
        return y


class HyenaOperator(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        max_seq_length: int,
        order: int = 2,
        num_heads: int = 1,
        dropout: float = 0.0,
        short_filter_order: int = 3,
        activation: str = "identity",
        submodules: HyenaOperatorSubmodules = None,
        layer_number=None,
        **long_conv_kwargs,
    ):
        r"""
        Hyena operator described in the paper https://arxiv.org/pdf/2302.10866.pdf

        Args:
            max_seq_length: (int): Maximum input sequence length.
            order: (int): Depth of the Hyena recurrence. Defaults to 2
            num_heads: (int): Number of heads. Defaults to 1
            dropout: (float): Dropout probability. Defaults to 0.0
            short_filter_order: (int): Length of the explicit input convolutional filter. Defaults to 3
            activation: (str): type of act between kernel output and output projection (default identity)
        """
        super().__init__()

        if submodules is None:
            submodules = HyenaOperatorSubmodules(
                in_proj=TELayerNormColumnParallelLinear,
                short_filter=CausalDepthWiseConv1d,
                implicit_filter=HyenaFilter,
                out_proj=TERowParallelLinear,
            )

        if order < 2:
            raise ValueError(f'Order must be at least 2, (got {self.order})')

        d_model = config.hidden_size
        if d_model % num_heads != 0:
            raise ValueError(f'Model dimension {d_model} must be divisible by num heads {num_heads}')
        head_dim = d_model // num_heads

        auto_assign_attrs(
            self,
            d_model=d_model,
            order=order,
            max_seq_length=max_seq_length,
            num_heads=num_heads,
            head_dim=head_dim,
            short_filter_order=short_filter_order,
            activation=activation,
            mcore_config=config,
        )
        self.activation = activation_registry[activation]()
        self.dropout = nn.Dropout(dropout)

        # Setup input and output projections (over the width dimension)
        self.in_proj = build_module(
            submodules.in_proj,
            self.d_model,
            (self.order + 1) * self.d_model,
            config=self.mcore_config,
            init_method=self.mcore_config.init_method,
            gather_output=False,
            bias=True,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='in_proj',
        )

        self.out_proj = build_module(
            submodules.out_proj,
            self.d_model,
            self.d_model,
            config=self.mcore_config,
            init_method=self.mcore_config.output_layer_init_method,
            bias=True,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='out_proj',
        )

        # Setup short filter
        total_width = self.d_model * (self.order + 1)
        self.short_filter = build_module(submodules.short_filter, total_width, self.short_filter_order)

        # Setup long convolution with implicit filter
        long_conv_args = [self.head_dim, self.max_seq_length, self.order]
        long_conv_kwargs['filter_cls'] = submodules.implicit_filter
        long_conv_kwargs['filter_submodules'] = submodules.implicit_filter.submodules
        if self.num_heads == 1:
            self.long_conv = SingleHeadHyenaConv(*long_conv_args, **long_conv_kwargs)
            self.conv_fwd_fn = self.conv_single_head
        else:
            long_conv_args.append(self.num_heads)
            self.long_conv = MultiHeadHyenaConv(*long_conv_args, **long_conv_kwargs)
            self.conv_fwd_fn = self.conv_multi_head

    def forward(self, u, *args, **kwargs):
        l = u.size(0)
        l_filter = min(l, self.max_seq_length)
        u = self.in_proj(u)
        u = u[0] if isinstance(u, tuple) else u
        u = rearrange(u, 'l b d -> b d l')  # In MCore the leading dimension is the sequence dimension

        k = self.long_conv.filter(l_filter)
        # `c` is always 1 by default
        k = rearrange(k, 'c l v -> c v l', v=self.head_dim)[0]

        uc = self.short_filter(u)[..., :l_filter]

        k = k.to(dtype=torch.float32)
        y = self.conv_fwd_fn(uc, k)

        y = rearrange(y, 'b d l -> b l d')
        y = self.activation(y)
        y = self.out_proj(y)
        if isinstance(y, tuple):
            y, bias = y
        else:
            bias = None

        # Convert back to sequence-first for MCore
        y = rearrange(y, 'b l d -> l b d')

        # MCore TransformerLayer expects tuple where 2nd element represents the bias, it can be None
        return y, bias

    def conv_single_head(self, uc, k):
        k = rearrange(k, '(o v) l -> o v l', v=self.head_dim, o=self.order - 1)

        *x, v = uc.split(self.d_model, dim=1)
        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)
            v = self.long_conv(v, k=k[o], recurrence_idx=o)

        y = v * x[0]
        return y

    def conv_multi_head(self, uc, k):
        x1, x2, v = uc.split(self.d_model, dim=1)
        x1 = x1.contiguous()
        x2 = x2.contiguous()
        v = v.contiguous()

        y = self.long_conv(v, k, x1, x2)
        return y
