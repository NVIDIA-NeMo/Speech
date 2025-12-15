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
from abc import abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from nemo.utils import logging

# TODO: Move the cache implementation out of the Module class, and pass it as part of the forward so we can reset
# as needed in the inference pipeline.


class ConvolutionLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        bias: bool = True,
        is_causal: bool = False,
    ):
        """
        A convolutional layer that supports causal convolutions with padding. Replaces the standard MLP layer used in
        the original transformer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolving kernel.
            stride (int): Stride of the convolution.
            padding (Optional[int]): Padding added to both sides of the input. If None, it's calculated automatically.
            dilation (int): Spacing between kernel elements.
            bias (bool): If True, adds a learnable bias to the output.
            is_causal (bool): If True, uses causal convolution.
        """
        super().__init__()

        # Setup up padding; should be 0 if set to causal
        # If not causal and padding is None, set an appropriate value for padding
        self.causal_padding = None
        if is_causal:
            self.causal_padding = ((kernel_size - 1) * dilation, 0)
            if padding is not None:
                logging.warning(
                    f'{self} was initialized with is_causal set to True, and padding set to {padding}. '
                    f'The provided padding value will be ignored and set to {self.causal_padding}.'
                )
            padding = 0
        elif padding is None:
            if kernel_size % 2 == 0:
                raise ValueError("`kernel_size` must be odd when `padding` is None.")
            else:
                padding = int(dilation * (kernel_size - 1) / 2)

        self.is_causal = is_causal
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, signal, signal_mask):
        # signal: (B, C, T)
        # signal_mask: (B, T)
        signal = signal * signal_mask.unsqueeze(1)
        if self.is_causal:  # TODO: maybe replace with identify rather than keep conditional if in forward
            signal = F.pad(signal, self.causal_padding)

        conv_signal = self.conv(signal)
        conv_signal = conv_signal * signal_mask.unsqueeze(1)

        return conv_signal


class PositionwiseConvFF(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        p_dropout: float,
        kernel_size: int = 1,
        bias: bool = False,
        is_causal: bool = True,
        non_linearity: Callable = torch.nn.GELU(approximate="tanh"),
    ):
        """
        Positionwise Convolutional Feed-Forward layer to replace the MLP layer in transformers.

        Module will take the input with d_model hidden state, project it to d_ffn hidden dimension, perform nonlinear
        transformation, and project the state back into d_model hidden dimension. Finally, it applied dropout.

        Args:
            d_model (int): Input and output dimension of the model.
            d_ffn (int): Hidden dimension of the feed-forward network (usually 4 * d_model).
            p_dropout (float): Dropout probability.
            kernel_size (int): Size of the convolving kernel.
            bias (bool): If True, adds a learnable bias to the convolution layers.
            is_causal (bool): If True, uses causal convolution.
            non_linearity (Callable): Activation function to use (default: GELU).
        """
        super().__init__()
        # d_ffn is usually 4*d_model
        self.d_model = d_model
        self.non_linearity = non_linearity

        self.proj = ConvolutionLayer(d_model, d_ffn, bias=bias, kernel_size=kernel_size, is_causal=is_causal)
        self.o_net = ConvolutionLayer(d_ffn, d_model, bias=bias, kernel_size=kernel_size, is_causal=is_causal)
        self.dropout = torch.nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        """
        x (B, T, C)
        x_mask (B, T)
        """
        x = self.non_linearity(self.proj(x.transpose(1, 2), x_mask))
        x = self.dropout(self.o_net(x, x_mask).transpose(1, 2))
        return x


class Attention(torch.nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        p_dropout: float,
        is_causal: bool = True,
        d_head: Optional[int] = None,
    ):
        """
        Base Attention parent class. Users should not be instantiating this class, but rather use SelfAttention or
        CrossAttention classes as appropriate.
        Does DotProductionAttention and additionally dropout inside the module. The class does not currently support
        RoPE nor ALiBi.

        Args:
            n_heads (int): Number of attention heads.
            d_model (int): Dimension of the model.
            p_dropout (float): Dropout probability.
            is_causal (bool): Whether to use causal attention. Only supported when used in SelfAttention.
            d_head (int): Head dimension. Defaults to d_model // n_heads.
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model % n_head != 0"
        self.d_head = d_head if d_head is not None else d_model // n_heads
        self.n_heads = n_heads
        self.d_model = d_model
        self.scale = self.d_head**-0.5
        self.is_causal = is_causal
        self.o_net = torch.nn.Linear(n_heads * self.d_head, d_model, bias=False)
        self.dropout = torch.nn.Dropout(p_dropout)
        self.use_cache = False
        self.cache = self._init_cache()

    @abstractmethod
    def compute_qkv_and_mask(
        self,
        query: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ):
        pass

    @staticmethod
    def _init_cache() -> Dict[str, Optional[Union[bool, torch.Tensor]]]:
        return {
            'is_initialized': False,
            'self_k': None,
            'self_v': None,
            'cross_kv': None,
            'cross_k': None,
            'cross_v': None,
        }

    def reset_cache(self, use_cache: bool = False):
        self.use_cache = use_cache
        self.cache = self._init_cache()

    def attn_naive(
        self,
        query: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        attn_prior: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if self.use_cache:
            if self.cache['is_initialized']:
                query = query[:, -1:, :]
                query_mask = query_mask[:, -1:] if query_mask is not None else None
            else:
                self.cache['is_initialized'] = True

        # Calls into children classes to compute qkv tensors and mask tensor
        q, k, v, mask = self.compute_qkv_and_mask(
            query=query, query_mask=query_mask, memory=memory, memory_mask=memory_mask
        )

        # (B, T, nh, dh) -> (B, nh, T, dh)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        B, T, _ = query.shape
        attn_score = torch.matmul(q, k.transpose(2, 3)) * self.scale
        if mask is not None:
            # assumes there's at least one mask
            attn_score.masked_fill_(mask == 0, float('-inf'))
        if self.is_causal:
            attn_score.masked_fill_(self.causal_mask[..., :T, :T] == 0, float('-inf'))

        # attn_prior or square mask or vanilla attention
        if attn_prior is not None:
            eps = torch.finfo(attn_prior.dtype).tiny
            attn_prior = attn_prior[:, :T]  # trim for inference
            attn_prior = attn_prior[:, None] + eps
            # Use PyTorch's built-in training flag to branch behavior
            if self.training:
                attn_prior_log = torch.log(attn_prior)
                attn_score_log = F.log_softmax(attn_score, dim=-1) + attn_prior_log
                if self.make_prior_window_strict:
                    # Make sure attention scores are lowest (eps) where prior is zero.
                    min_score = torch.log(torch.tensor(eps)).to(attn_score_log.device)
                    attn_score_log = attn_score_log.masked_fill(
                        attn_prior == 0, min_score
                    )  # Wherever prior is zero, set scores to eps.
                    attn_score_log = torch.clamp(
                        attn_score_log, min=min_score
                    )  # Make sure scores are not less than eps.
                attn_prob = F.softmax(attn_score_log, dim=-1)
            else:
                attn_prob = F.softmax(attn_score, dim=-1)
                attn_prob = attn_prob * attn_prior
                attn_prob = attn_prob / (attn_prob.sum(dim=-1, keepdim=True))  # normalize
        else:
            attn_prob = F.softmax(attn_score, dim=-1)

        if mask is not None:
            attn_prob = attn_prob.masked_fill(mask == 0, 0.0)
        attn_prob = self.dropout(attn_prob)

        y = torch.matmul(attn_prob, v)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)

        return y, [attn_prob, attn_score]

    def forward(
        self,
        query: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        attn_prior: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of the Attention module.

        Args:
            query (torch.Tensor): Input tensor of shape (B, T1, C).
            query_mask (Optional[torch.Tensor]): Mask for query tensor of shape (B, T1).
            memory (Optional[torch.Tensor]): Memory tensor for cross-attention of shape (B, T2, C).
            memory_mask (Optional[torch.Tensor]): Mask for memory tensor of shape (B, T2).
            attn_prior (Optional[torch.Tensor]): Prior attention weights of shape (B, T1, T2).

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                - y: Attention module tensor output of shape (B, T1, C).
                - attn_prob: List containing attention probabilities and scores. returned only in attn_naive.
                    [0]: Attention probabilities used for logging during validation.
                    [1]: Attention scores used for CTC loss (only in naive attention).
        """

        y, attn_prob = self.attn_naive(query, query_mask, memory, memory_mask, attn_prior)
        y = self.dropout(self.o_net(y))

        return y, attn_prob


class SelfAttention(Attention):
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        p_dropout: float,
        is_causal: bool = True,
        max_length_causal_mask: int = 4096,
    ):
        """
        Implements SelfAttention. See parent class for forward implementation.

        Args:
            n_heads (int): Number of attention heads.
            d_model (int): Dimension of the model.
            p_dropout (float): Dropout probability.
            is_causal (bool): Whether to use causal attention. Only supported when used in SelfAttention.
            max_length_causal_mask (int): Maximum sequence length for Attention module.
        """
        super().__init__(
            n_heads=n_heads,
            d_model=d_model,
            p_dropout=p_dropout,
            is_causal=is_causal,
        )
        if is_causal:
            if max_length_causal_mask is None or max_length_causal_mask < 0:
                raise ValueError(
                    "Self Attention was called with is_causal True, but received an inappropriate value"
                    f"of {max_length_causal_mask} for max_length_causal_mask"
                )
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(max_length_causal_mask, max_length_causal_mask)).view(
                    1, 1, max_length_causal_mask, max_length_causal_mask
                ),
            )
        self.qkv_net = torch.nn.Linear(d_model, 3 * n_heads * self.d_head, bias=False)

    def compute_qkv_and_mask(
        self,
        query: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ):
        B, T, _ = query.shape
        qkv = self.qkv_net(query).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.chunk(3, dim=2)
        q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)
        if self.use_cache:
            if self.cache['self_k'] is not None:
                k = torch.cat([self.cache['self_k'], k], dim=1)
                v = torch.cat([self.cache['self_v'], v], dim=1)
            self.cache['self_k'] = k
            self.cache['self_v'] = v

        mask = None
        if query_mask is not None:
            # query_mask is a boolean mask of shape (B, T)
            # mask should be of shape (B, 1, T, T) where mask[:,0,i,:] == mask[:,0,:,i] == query_mask
            mask = query_mask.unsqueeze(1) * query_mask.unsqueeze(2)
            mask = mask.unsqueeze(1)

        return q, k, v, mask


class CrossAttention(Attention):
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        d_memory: int,
        p_dropout: float,
        make_prior_window_strict: bool = False,
        d_head: Optional[int] = None,
    ):
        """
        Implements CrossAttention. See parent class for forward implementation. Must be non-causal.

        Args:
            n_heads (int): Number of attention heads.
            d_model (int): Dimension of the model.
            d_memory (int): Dimension of the conditioning / cross-attention input.
            p_dropout (float): Dropout probability.
            make_prior_window_strict (bool): Make attention scores lowest where prior is zero.
            d_head (int): Head dimension. if None, defaults to d_model // n_heads in parent class.
        """
        super().__init__(
            n_heads=n_heads,
            d_model=d_model,
            p_dropout=p_dropout,
            is_causal=False,
            d_head=d_head,
        )
        self.q_net = torch.nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.kv_net = torch.nn.Linear(d_memory, 2 * n_heads * self.d_head, bias=False)
        self.make_prior_window_strict = make_prior_window_strict

    def compute_qkv_and_mask(
        self,
        query: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ):
        Bq, Tq, _ = query.shape
        Bkv, Tkv, _ = memory.shape
        q = self.q_net(query).reshape(Bq, Tq, self.n_heads, self.d_head)
        if self.use_cache and self.cache['cross_kv'] is not None:
            kv = self.cache['cross_kv']
        else:
            kv = self.kv_net(memory).reshape(Bkv, Tkv, 2, self.n_heads, self.d_head)

        if self.use_cache and self.cache['cross_k'] is not None:
            k = self.cache['cross_k']
            v = self.cache['cross_v']
        else:
            k, v = kv.chunk(2, dim=2)
            k, v = k.squeeze(2), v.squeeze(2)
            if self.use_cache:
                self.cache['cross_kv'] = kv
                self.cache['cross_k'] = k
                self.cache['cross_v'] = v

        mask = memory_mask[:, None, None] if memory_mask is not None else None
        return q, k, v, mask


class TransformerLayer(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        sa_n_heads: int,
        kernel_size: int,
        p_dropout: float,
        has_xattn: bool,
        xa_d_memory: Optional[int] = None,
        xa_n_heads: Optional[int] = None,
        xa_d_head: Optional[int] = None,
        is_causal: bool = True,
        apply_norm_to_cond: bool = True,
        max_length_causal_mask: int = 4096,
        conv_non_linearity: Callable = torch.nn.GELU(approximate="tanh"),
        make_prior_window_strict: bool = False,
    ):
        """
        One layer of the Transformer.
        Args:
            d_model <int>: Model dimension
            d_ffn <int>: Feed forward dimension (usually 4*d_model)
            sa_n_heads <int>: Number of attention heads used in self-attention
            kernel_size <int>: Convolution kernel size for FFN
            p_dropout <float>: Dropout probability
            has_xattn <bool>: Whether to use cross attention
            xa_d_memory <int>: Hidden dimension for cross attention
            xa_n_heads <int>: Number of attention heads used in cross attention
            xa_d_head <int>: Head dimension for cross attention. if None, defaults to d_model // xa_n_heads in Attention class.
            is_causal <bool>: Whether to use causal attention
            apply_norm_to_cond <bool>: Whether to apply normalization to conditioning tensor
            max_length_causal_mask <int>: Maximum length of causal mask
            conv_non_linearity <Callable>: Convolution non-linearity
            make_prior_window_strict <bool>: Make attention scores lowest where prior is zero.
        """
        super().__init__()
        self.has_xattn = has_xattn

        self.norm_self = torch.nn.LayerNorm(d_model, bias=False)
        self.self_attention = SelfAttention(
            n_heads=sa_n_heads,
            d_model=d_model,
            p_dropout=p_dropout,
            max_length_causal_mask=max_length_causal_mask,
            is_causal=is_causal,
        )

        if self.has_xattn:
            self.norm_xattn_query = torch.nn.LayerNorm(d_model, bias=False)
            self.cross_attention = CrossAttention(
                n_heads=xa_n_heads,
                d_model=d_model,
                d_memory=xa_d_memory,
                p_dropout=p_dropout,
                make_prior_window_strict=make_prior_window_strict,
                d_head=xa_d_head,
            )

            self.norm_xattn_memory = torch.nn.Identity()
            if apply_norm_to_cond:
                self.norm_xattn_memory = torch.nn.LayerNorm(xa_d_memory, bias=False)

        self.norm_pos_ff = torch.nn.LayerNorm(d_model, bias=False)
        self.pos_ff = PositionwiseConvFF(
            d_model, d_ffn, p_dropout, kernel_size=kernel_size, is_causal=is_causal, non_linearity=conv_non_linearity
        )

        self.use_cache = False
        self.cache = self._init_cache()

    @staticmethod
    def _init_cache() -> Dict:
        return {
            'self_attn_output': None,
            'cross_attn_output': None,
            'memory': None,
        }

    def reset_cache(self, use_cache=False):
        self.use_cache = use_cache
        self.cache = self._init_cache()
        self.self_attention.reset_cache(use_cache)
        if self.has_xattn:
            self.cross_attention.reset_cache(use_cache)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        cond_mask: Optional[torch.Tensor] = None,
        attn_prior: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Args:
            x <torch tensor> (B, T1, C): Input tensor
            x_mask <bool mask> (B, T1): Multiplicative mask where True means we keep the input, False we zero it out.
                Mask for self attention input.
            cond <torch tensor> (B, T2, C): Conditioning tensor
            cond_mask <bool mask> (B, T2): Multiplicative mask where True means we keep the input, False we zero
                it out. Mask for cross attention input if it exists.

        Returns dict with keys
            output <torch tensor> (B, T1, C): Output tensor
            attn_probabilities <dict>: Attention probabilities
        """
        x = x * x_mask.unsqueeze(-1)
        x_, s_attn_prob = self.self_attention(query=self.norm_self(x), query_mask=x_mask)
        if self.use_cache:
            if self.cache['self_attn_output'] is not None:
                x_ = torch.cat([self.cache['self_attn_output'], x_], dim=1)
            self.cache['self_attn_output'] = x_
        x = x + x_

        x_attn_prob = None
        if self.has_xattn and cond is not None:
            x_normed = self.norm_xattn_query(x)
            if self.use_cache and self.cache['memory'] is not None:
                memory = self.cache['memory']
            else:
                memory = self.norm_xattn_memory(cond)
                if self.use_cache:
                    self.cache['memory'] = memory

            x_res, x_attn_prob = self.cross_attention(
                query=x_normed, query_mask=x_mask, memory=memory, memory_mask=cond_mask, attn_prior=attn_prior
            )
            if self.use_cache:
                if self.cache['cross_attn_output'] is not None:
                    x_res = torch.cat([self.cache['cross_attn_output'], x_res], dim=1)
                self.cache['cross_attn_output'] = x_res
            x = x + x_res

        # mlp final projection
        x = x + self.pos_ff(self.norm_pos_ff(x), x_mask)
        x = x * x_mask.unsqueeze(-1)

        return {
            'output': x,
            'attn_probabilities': {'self_attn_probabilities': s_attn_prob, 'cross_attn_probabilities': x_attn_prob},
        }


class Transformer(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        d_ffn: int,
        sa_n_heads: int,
        kernel_size: int,
        p_dropout: float = 0.0,
        p_dropout_out: float = 0.0,
        has_xattn: bool = False,
        xa_d_memory: Optional[int] = None,
        xa_n_heads: Optional[int] = None,
        xa_d_head: Optional[int] = None,
        is_causal: bool = True,
        apply_norm_to_cond: bool = True,
        apply_norm_out: bool = False,
        max_length_causal_mask: int = 4096,
        use_learnable_pos_emb: bool = False,
        conv_non_linearity: Callable = torch.nn.GELU(approximate="tanh"),
        make_prior_window_strict: bool = False,
    ):
        """
        Initializes a stack of transformer layers. Can be used for both encoder and decoder.
        Set is_causal is True for autoregressive models. Equivalent to TransformerBlock from Megatron-LM
        Args:
            n_layers <int>: Number of transformer layers
            d_model <int>: Model dimension
            d_ffn <int>: Feed forward dimension (usually 4*d_model)
            sa_n_heads <int>: Number of attention heads used in self-attention
            kernel_size <int>: Convolution kernel size for FFN
            p_dropout <float>: Dropout probability
            p_dropout_out <float>: Dropout probability for output
            has_xattn <bool>: Whether to use cross attention
            xa_d_memory <int>: Hidden dimension for cross attention; required if has_xattn is True
            xa_n_heads <int>: Number of attention heads used in cross attention; required if has_xattn is True
            xa_d_head <int>: Head dimension for cross attention. if None, defaults to d_model // xa_n_heads in Attention class.
            is_causal <bool>: Whether to make attention and the convolution feedforward networks causal.
            apply_norm_to_cond <bool>: Whether to apply normalization to conditioning tensor; conditioning tensor being
                the input to the memory part of cross-attention.
            apply_norm_out <bool>: Whether to apply normalization to output
            max_length_causal_mask <int>: Maximum length of causal mask
            use_learnable_pos_emb <bool>: Whether to add a learnable positionable embedding inside the class
            conv_non_linearity <Callable>: Convolution non-linearity
            make_prior_window_strict <bool>: Make attention scores lowest where prior is zero
        """
        if has_xattn and (xa_d_memory is None or xa_n_heads is None):
            raise ValueError("It requires that `xa_d_memory` and `xa_n_heads` are specified when `has_xattn` is True!")

        super().__init__()
        self.n_layers = n_layers
        self.dropout = torch.nn.Dropout(p_dropout)
        self.p_dropout_out = p_dropout_out

        self.dropout_out = torch.nn.Identity()
        if self.p_dropout_out > 0.0:
            self.dropout_out = torch.nn.Dropout(self.p_dropout_out)

        self.norm_out = torch.nn.Identity()
        if apply_norm_out:
            self.norm_out = torch.nn.LayerNorm(d_model, bias=False)

        self.layers = torch.nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(
                TransformerLayer(
                    d_model=d_model,
                    d_ffn=d_ffn,
                    sa_n_heads=sa_n_heads,
                    kernel_size=kernel_size,
                    p_dropout=p_dropout,
                    has_xattn=has_xattn,
                    xa_d_memory=xa_d_memory,
                    xa_n_heads=xa_n_heads,
                    xa_d_head=xa_d_head,
                    is_causal=is_causal,
                    apply_norm_to_cond=apply_norm_to_cond,
                    max_length_causal_mask=max_length_causal_mask,
                    conv_non_linearity=conv_non_linearity,
                    make_prior_window_strict=make_prior_window_strict,
                )
            )

        self.use_learnable_pos_emb = use_learnable_pos_emb
        self.position_embeddings = None
        if self.use_learnable_pos_emb:
            self.position_embeddings = torch.nn.Embedding(max_length_causal_mask, d_model)
        # Apply random uniform init for all layers, except for output layers: The second of the two layers in the MLP
        # and the last linear projection in dot product attention. The output layers are scaled depending on the
        # number of layers
        self.apply(self._init_weights_gpt2)
        for name, param in self.named_parameters():
            if 'o_net' in name and name.endswith('weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers))

    def reset_cache(self, use_cache=False):
        for layer in self.layers:
            layer.reset_cache(use_cache)

    @staticmethod
    def _init_weights_gpt2(module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv1d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    @staticmethod
    def _get_layer_inputs(
        idx: int,
        cond: Optional[Union[torch.Tensor, List[torch.Tensor]]],
        cond_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]],
        attn_prior: Optional[Union[torch.Tensor, List[torch.Tensor]]],
        multi_encoder_mapping: Optional[List[Optional[int]]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if multi_encoder_mapping is not None:
            if multi_encoder_mapping[idx] is None:
                return None, None, None
            else:
                _attn_prior = attn_prior[multi_encoder_mapping[idx]] if attn_prior is not None else None
                if isinstance(_attn_prior, list):
                    # @pneekhara: This means, we are passing layerwise attn_prior
                    _attn_prior = _attn_prior[idx]
                return (
                    cond[multi_encoder_mapping[idx]],
                    cond_mask[multi_encoder_mapping[idx]] if cond_mask is not None else None,
                    _attn_prior,
                )
        else:
            if isinstance(attn_prior, list):
                # @pneekhara: This means, we are passing layerwise attn_prior
                _attn_prior = attn_prior[idx]
            else:
                _attn_prior = attn_prior
            return cond, cond_mask, _attn_prior

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        cond: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        cond_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        attn_prior: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        multi_encoder_mapping: Optional[List[Optional[int]]] = None,
        max_layer_idx: Optional[int] = None,
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """
        Args:
            x <torch tensor> (B, T1, C):
            x_mask <bool mask> (B, T1): Multiplicative mask where True means we keep the input, False we zero it out.
                Mostly used in non-causal self-attention to zero out padding values. In causal self-attention, the
                causal mask will be used in place of this.
            cond <torch tensor> (B, T2, C) or list of such tensors (from different encoders)
            cond_mask <bool mask> (B, T2): Multiplicative mask where True means we keep the input, False we zero it
                out or list of such tensors (from different encoders) output <torch tensor> (B, T1, C)
            multi_encoder_mapping <list> <int>: None or Same size as n_layers, value indicates which cond input to use
                for this layer

        Returns dict with keys:
            output <torch tensor> (B, T1, C): Output tensor
            attn_probabilities <list>: Attention probabilities of each layer
        """
        if isinstance(cond, list) and len(self.layers) < len(cond):
            raise ValueError(
                f"Insufficient Transformer layers for multiple conditionals. Each layer must cross-attend one conditional."
                f"Found {len(self.layers)} layers for {len(cond)} conditionals."
            )

        if self.use_learnable_pos_emb:
            positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            x = x + self.position_embeddings(positions)

        attn_probabilities = []
        x = self.dropout(x)
        for idx, layer in enumerate(self.layers):
            _cond, _cond_mask, _attn_prior = self._get_layer_inputs(
                idx, cond, cond_mask, attn_prior, multi_encoder_mapping
            )
            out_dict = layer(x, x_mask, _cond, _cond_mask, attn_prior=_attn_prior)
            x = out_dict['output']
            attn_probabilities.append(out_dict['attn_probabilities'])

            if max_layer_idx is not None and idx == max_layer_idx:
                break

        x = self.norm_out(x)
        x = self.dropout_out(x)
        return {'output': x, 'attn_probabilities': attn_probabilities}
