# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Tuple

import torch
from torch import Tensor
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nemo.core.classes import NeuralModule, adapter_mixins
from nemo.core.neural_types.elements import EncodedRepresentation, Index, LengthsType, MelSpectrogramType
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging


SUPPORTED_CONDITION_TYPES = ["add", "concat", "layernorm"]


def check_support_condition_types(condition_types):
    for tp in condition_types:
        if tp not in SUPPORTED_CONDITION_TYPES:
            raise ValueError(f"Unknown conditioning type {tp}")


def masked_instance_norm(
    input: Tensor, mask: Tensor, weight: Tensor, bias: Tensor, momentum: float, eps: float = 1e-5,
) -> Tensor:
    r"""Applies Masked Instance Normalization for each channel in each data sample in a batch.

    See :class:`~MaskedInstanceNorm1d` for details.
    """
    lengths = mask.sum((-1,))
    mean = (input * mask).sum((-1,)) / lengths  # (N, C)
    var = (((input - mean[(..., None)]) * mask) ** 2).sum((-1,)) / lengths  # (N, C)
    out = (input - mean[(..., None)]) / torch.sqrt(var[(..., None)] + eps)  # (N, C, ...)
    out = out * weight[None, :][(..., None)] + bias[None, :][(..., None)]

    return out


class MaskedInstanceNorm1d(torch.nn.InstanceNorm1d):
    r"""Applies Instance Normalization over a masked 3D input
    (a mini-batch of 1D inputs with additional channel dimension)..

    See documentation of :class:`~torch.nn.InstanceNorm1d` for details.

    Shape:
        - Input: :math:`(N, C, L)`
        - Mask: :math:`(N, 1, L)`
        - Output: :math:`(N, C, L)` (same shape as input)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
    ) -> None:
        super(MaskedInstanceNorm1d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input: Tensor, mask: Tensor) -> Tensor:
        return masked_instance_norm(input, mask, self.weight, self.bias, self.momentum, self.eps,)


class PartialConv1d(torch.nn.Conv1d):
    """
    Zero padding creates a unique identifier for where the edge of the data is, such that the model can almost always identify
    exactly where it is relative to either edge given a sufficient receptive field. Partial padding goes to some lengths to remove 
    this affect.
    """

    __constants__ = ['slide_winsize']
    slide_winsize: float

    def __init__(self, *args, **kwargs):
        super(PartialConv1d, self).__init__(*args, **kwargs)
        weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0])
        self.register_buffer("weight_maskUpdater", weight_maskUpdater, persistent=False)
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2]

    def forward(self, input, mask_in):
        if mask_in is None:
            mask = torch.ones(1, 1, input.shape[2], dtype=input.dtype, device=input.device)
        else:
            mask = mask_in
            input = torch.mul(input, mask)
        with torch.no_grad():
            update_mask = F.conv1d(
                mask,
                self.weight_maskUpdater,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=1,
            )
            update_mask_filled = torch.masked_fill(update_mask, update_mask == 0, self.slide_winsize)
            mask_ratio = self.slide_winsize / update_mask_filled
            update_mask = torch.clamp(update_mask, 0, 1)
            mask_ratio = torch.mul(mask_ratio, update_mask)

        raw_out = self._conv_forward(input, self.weight, self.bias)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1)
            output = torch.mul(raw_out - bias_view, mask_ratio) + bias_view
            output = torch.mul(output, update_mask)
        else:
            output = torch.mul(raw_out, mask_ratio)

        return output


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module, adapter_mixins.AdapterModuleMixin):
    __constants__ = ['use_partial_padding']
    use_partial_padding: bool

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain='linear',
        use_partial_padding=False,
        use_weight_norm=False,
        norm_fn=None,
    ):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.use_partial_padding = use_partial_padding
        conv_fn = torch.nn.Conv1d
        if use_partial_padding:
            conv_fn = PartialConv1d
        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
        if use_weight_norm:
            self.conv = torch.nn.utils.weight_norm(self.conv)
        if norm_fn is not None:
            self.norm = norm_fn(out_channels, affine=True)
        else:
            self.norm = None

    def forward(self, signal, mask=None):
        if self.use_partial_padding:
            ret = self.conv(signal, mask)
            if self.norm is not None:
                ret = self.norm(ret, mask)
        else:
            if mask is not None:
                signal = signal.mul(mask)
            ret = self.conv(signal)
            if self.norm is not None:
                ret = self.norm(ret)

        if self.is_adapter_available():
            ret = self.forward_enabled_adapters(ret.transpose(1, 2)).transpose(1, 2)

        return ret


class LocationLayer(torch.nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        super().__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(
            2,
            attention_n_filters,
            kernel_size=attention_kernel_size,
            padding=padding,
            bias=False,
            stride=1,
            dilation=1,
        )
        self.location_dense = LinearNorm(attention_n_filters, attention_dim, bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(torch.nn.Module):
    def __init__(
        self,
        attention_rnn_dim,
        embedding_dim,
        attention_dim,
        attention_location_n_filters,
        attention_location_kernel_size,
    ):
        super().__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim, bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False, w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(
            attention_location_n_filters, attention_location_kernel_size, attention_dim,
        )
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory, attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(
        self, attention_hidden_state, memory, processed_memory, attention_weights_cat, mask,
    ):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(torch.nn.Module):
    def __init__(self, in_dim, sizes, p_dropout=0.5):
        super().__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.p_dropout = p_dropout
        self.layers = torch.nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False) for (in_size, out_size) in zip(in_sizes, sizes)]
        )

    def forward(self, x, inference=False):
        if inference:
            for linear in self.layers:
                x = F.relu(linear(x))
                x0 = x[0].unsqueeze(0)
                mask = torch.autograd.Variable(torch.bernoulli(x0.data.new(x0.data.size()).fill_(1 - self.p_dropout)))
                mask = mask.expand(x.size(0), x.size(1))
                x = x * mask * 1 / (1 - self.p_dropout)
        else:
            for linear in self.layers:
                x = F.dropout(F.relu(linear(x)), p=self.p_dropout, training=True)
        return x


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels_int):
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class Invertible1x1Conv(torch.nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """

    def __init__(self, c):
        super().__init__()
        self.conv = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0, bias=False)

        # Sample a random orthonormal matrix to initialize weights
        W = torch.linalg.qr(torch.FloatTensor(c, c).normal_())[0]

        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:, 0] = -1 * W[:, 0]
        W = W.view(c, c, 1)
        self.conv.weight.data = W
        self.inv_conv = None

    def forward(self, z, reverse: bool = False):
        if reverse:
            if self.inv_conv is None:
                # Inverse convolution - initialized here only for backwards
                # compatibility with weights from existing checkpoints.
                # Should be moved to init() with next incompatible change.
                self.inv_conv = torch.nn.Conv1d(
                    self.conv.in_channels, self.conv.out_channels, kernel_size=1, stride=1, padding=0, bias=False
                )
                W_inverse = self.conv.weight.squeeze().data.float().inverse()
                W_inverse = Variable(W_inverse[..., None])
                self.inv_conv.weight.data = W_inverse
                self.inv_conv.to(device=self.conv.weight.device, dtype=self.conv.weight.dtype)
            return self.inv_conv(z)
        else:
            # Forward computation
            # shape
            W = self.conv.weight.squeeze()
            batch_size, group_size, n_of_groups = z.size()
            log_det_W = batch_size * n_of_groups * torch.logdet(W.float())
            z = self.conv(z)
            return (
                z,
                log_det_W,
            )


class WaveNet(torch.nn.Module):
    """
    This is the WaveNet like layer for the affine coupling.  The primary
    difference from WaveNet is the convolutions need not be causal.  There is
    also no dilation size reset.  The dilation only doubles on each layer
    """

    def __init__(self, n_in_channels, n_mel_channels, n_layers, n_channels, kernel_size):
        super().__init__()
        assert kernel_size % 2 == 1
        assert n_channels % 2 == 0
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(n_in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = torch.nn.Conv1d(n_channels, 2 * n_in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        cond_layer = torch.nn.Conv1d(n_mel_channels, 2 * n_channels * n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

        for i in range(n_layers):
            dilation = 2 ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(n_channels, 2 * n_channels, kernel_size, dilation=dilation, padding=padding,)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, forward_input: Tuple[torch.Tensor, torch.Tensor]):
        audio, spect = forward_input[0], forward_input[1]
        audio = self.start(audio)
        output = torch.zeros_like(audio)

        spect = self.cond_layer(spect)

        for i in range(self.n_layers):
            spect_offset = i * 2 * self.n_channels
            acts = fused_add_tanh_sigmoid_multiply(
                self.in_layers[i](audio),
                spect[:, spect_offset : spect_offset + 2 * self.n_channels, :],
                self.n_channels,
            )

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = audio + res_skip_acts[:, : self.n_channels, :]
                output = output + res_skip_acts[:, self.n_channels :, :]
            else:
                output = output + res_skip_acts

        return self.end(output)


class ConditionalLayerNorm(torch.nn.LayerNorm):
    """
    This module is used to condition torch.nn.LayerNorm.
    If we don't have any conditions, this will be a normal LayerNorm.
    """

    def __init__(self, hidden_dim, condition_dim=None, condition_types=[]):
        check_support_condition_types(condition_types)
        self.condition = "layernorm" in condition_types
        super().__init__(hidden_dim, elementwise_affine=not self.condition)

        if self.condition:
            self.cond_weight = torch.nn.Linear(condition_dim, hidden_dim)
            self.cond_bias = torch.nn.Linear(condition_dim, hidden_dim)
            self.init_parameters()

    def init_parameters(self):
        torch.nn.init.constant_(self.cond_weight.weight, 0.0)
        torch.nn.init.constant_(self.cond_weight.bias, 1.0)
        torch.nn.init.constant_(self.cond_bias.weight, 0.0)
        torch.nn.init.constant_(self.cond_bias.bias, 0.0)

    def forward(self, inputs, conditioning=None):
        inputs = super().forward(inputs)

        # Normalize along channel
        if self.condition:
            if conditioning is None:
                raise ValueError(
                    """You should add additional data types as conditions (e.g. speaker id or reference audio) 
                                 and define speaker_encoder in your config."""
                )

            inputs = inputs * self.cond_weight(conditioning)
            inputs = inputs + self.cond_bias(conditioning)

        return inputs


class ConditionalInput(torch.nn.Module):
    """
    This module is used to condition any model inputs.
    If we don't have any conditions, this will be a normal pass.
    """

    def __init__(self, hidden_dim, condition_dim, condition_types=[]):
        check_support_condition_types(condition_types)
        super().__init__()
        self.support_types = ["add", "concat"]
        self.condition_types = [tp for tp in condition_types if tp in self.support_types]
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim

        if "add" in self.condition_types and condition_dim != hidden_dim:
            self.add_proj = torch.nn.Linear(condition_dim, hidden_dim)

        if "concat" in self.condition_types:
            self.concat_proj = torch.nn.Linear(hidden_dim + condition_dim, hidden_dim)

    def forward(self, inputs, conditioning=None):
        """
        Args:
            inputs (torch.tensor): B x T x H tensor.
            conditioning (torch.tensor): B x 1 x C conditioning embedding.
        """
        if len(self.condition_types) > 0:
            if conditioning is None:
                raise ValueError(
                    """You should add additional data types as conditions (e.g. speaker id or reference audio) 
                                 and define speaker_encoder in your config."""
                )

            if "add" in self.condition_types:
                if self.condition_dim != self.hidden_dim:
                    conditioning = self.add_proj(conditioning)
                inputs = inputs + conditioning

            if "concat" in self.condition_types:
                conditioning = conditioning.repeat(1, inputs.shape[1], 1)
                inputs = torch.cat([inputs, conditioning], dim=-1)
                inputs = self.concat_proj(inputs)

        return inputs


class StyleAttention(NeuralModule):
    def __init__(self, gst_size=128, n_style_token=10, n_style_attn_head=4):
        super(StyleAttention, self).__init__()

        token_size = gst_size // n_style_attn_head
        self.tokens = torch.nn.Parameter(torch.FloatTensor(n_style_token, token_size))
        self.mha = torch.nn.MultiheadAttention(
            embed_dim=gst_size,
            num_heads=n_style_attn_head,
            dropout=0.0,
            bias=True,
            kdim=token_size,
            vdim=token_size,
            batch_first=True,
        )
        torch.nn.init.normal_(self.tokens)

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'D'), EncodedRepresentation()),
            "token_id": NeuralType(('B'), Index(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "style_emb": NeuralType(('B', 'D'), EncodedRepresentation()),
        }

    def forward(self, inputs):
        batch_size = inputs.size(0)
        query = inputs.unsqueeze(1)
        tokens = F.tanh(self.tokens).unsqueeze(0).expand(batch_size, -1, -1)

        style_emb, _ = self.mha(query=query, key=tokens, value=tokens)
        style_emb = style_emb.squeeze(1)
        return style_emb


class Conv2DReLUNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True, dropout=0.0):
        super(Conv2DReLUNorm, self).__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.norm = torch.nn.LayerNorm(out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, x_mask=None):
        if x_mask is not None:
            x = x * x_mask

        # bhwc -> bchw
        x = x.contiguous().permute(0, 3, 1, 2)
        x = F.relu(self.conv(x))
        # bchw -> bhwc
        x = x.contiguous().permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class ReferenceEncoder(NeuralModule):
    """
    Encode mel-spectrograms to an utterance level feature
    """

    def __init__(self, n_mels, cnn_filters, dropout, gru_hidden, kernel_size, stride, padding, bias):
        super(ReferenceEncoder, self).__init__()
        self.filter_size = [1] + list(cnn_filters)
        self.layers = torch.nn.ModuleList(
            [
                Conv2DReLUNorm(
                    in_channels=int(self.filter_size[i]),
                    out_channels=int(self.filter_size[i + 1]),
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                    dropout=dropout,
                )
                for i in range(len(cnn_filters))
            ]
        )
        post_conv_height = self.calculate_post_conv_lengths(n_mels, n_convs=len(cnn_filters))
        self.gru = torch.nn.GRU(
            input_size=cnn_filters[-1] * post_conv_height, hidden_size=gru_hidden, batch_first=True,
        )

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType()),
            "inputs_lengths": NeuralType(('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'D'), EncodedRepresentation()),
        }

    def forward(self, inputs, inputs_lengths):
        # BMW -> BWMC (M: mels)
        x = inputs.transpose(1, 2).unsqueeze(3)
        x_lens = inputs_lengths
        x_masks = self.lengths_to_masks(x_lens).unsqueeze(2).unsqueeze(3)

        for layer in self.layers:
            x = layer(x, x_masks)
            x_lens = self.calculate_post_conv_lengths(x_lens)
            x_masks = self.lengths_to_masks(x_lens).unsqueeze(2).unsqueeze(3)

        # BWMC -> BWC
        x = x.contiguous().view(x.shape[0], x.shape[1], -1)

        self.gru.flatten_parameters()
        packed_x = pack_padded_sequence(x, x_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_x, _ = self.gru(packed_x)
        x, x_lens = pad_packed_sequence(packed_x, batch_first=True)
        x = x[torch.arange(len(x_lens)), (x_lens - 1), :]
        return x

    @staticmethod
    def calculate_post_conv_lengths(lengths, n_convs=1, kernel_size=3, stride=2, pad=1):
        """Batch lengths after n convolution with fixed kernel/stride/pad."""
        for _ in range(n_convs):
            lengths = (lengths - kernel_size + 2 * pad) // stride + 1
        return lengths

    @staticmethod
    def lengths_to_masks(lengths):
        """Batch of lengths to batch of masks"""
        # B -> BxT
        masks = torch.arange(lengths.max()).to(lengths.device).expand(
            lengths.shape[0], lengths.max()
        ) < lengths.unsqueeze(1)
        return masks


class GlobalStyleToken(NeuralModule):
    """
    Global Style Token based Speaker Embedding
    """

    def __init__(
        self, reference_encoder, gst_size=128, n_style_token=10, n_style_attn_head=4,
    ):
        super(GlobalStyleToken, self).__init__()
        self.reference_encoder = reference_encoder
        self.style_attention = StyleAttention(
            gst_size=gst_size, n_style_token=n_style_token, n_style_attn_head=n_style_attn_head
        )

    @property
    def input_types(self):
        return {
            "inp": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType()),
            "inp_lengths": NeuralType(('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "gst": NeuralType(('B', 'D'), EncodedRepresentation()),
        }

    def forward(self, inp, inp_lengths):
        style_embedding = self.reference_encoder(inp, inp_lengths)
        gst = self.style_attention(style_embedding)
        return gst


class SpeakerLookupTable(torch.nn.Module):
    """
    LookupTable based Speaker Embedding
    """

    def __init__(self, n_speakers, embedding_dim):
        super(SpeakerLookupTable, self).__init__()
        self.table = torch.nn.Embedding(n_speakers, embedding_dim)

    def forward(self, speaker):
        return self.table(speaker)


class SpeakerEncoder(NeuralModule):
    """
    class SpeakerEncoder represents speakers representation. 
    This module can combine GST (global style token) based speaker embeddings and lookup table speaker embeddings.
    """

    def __init__(self, lookup_module=None, gst_module=None, precomputed_embedding_dim=None):
        """
        lookup_module: Torch module to get lookup based speaker embedding
        gst_module: Neural module to get GST based speaker embedding
        precomputed_embedding_dim: Give precomputed speaker embedding dimension to use precompute speaker embedding
        """
        super(SpeakerEncoder, self).__init__()

        # Multi-speaker embedding
        self.lookup_module = lookup_module

        # Reference speaker embedding
        self.gst_module = gst_module

        if precomputed_embedding_dim is not None:
            self.precomputed_emb = torch.nn.Parameter(torch.empty(precomputed_embedding_dim))
        else:
            self.precomputed_emb = None

    @property
    def input_types(self):
        return {
            "batch_size": NeuralType(optional=True),
            "speaker": NeuralType(('B'), Index(), optional=True),
            "reference_spec": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType(), optional=True),
            "reference_spec_lens": NeuralType(('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "embs": NeuralType(('B', 'D'), EncodedRepresentation()),
        }

    def overwrite_precomputed_emb(self, emb):
        self.precomputed_emb = torch.nn.Parameter(emb)

    def forward(self, batch_size=None, speaker=None, reference_spec=None, reference_spec_lens=None):
        embs = None

        # Get Precomputed speaker embedding
        if self.precomputed_emb is not None:
            return self.precomputed_emb.unsqueeze(0).repeat(batch_size, 1)

        # Get Lookup table speaker embedding
        if self.lookup_module is not None and speaker is not None:
            embs = self.lookup_module(speaker)

        # Get GST based speaker embedding
        if reference_spec is not None and reference_spec_lens is not None:
            if self.gst_module is not None:
                out = self.gst_module(reference_spec, reference_spec_lens)
                embs = out if embs is None else embs + out
            else:
                logging.warning("You may add `gst_module` in speaker_encoder to use reference_audio.")

        return embs
