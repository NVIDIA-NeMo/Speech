# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.nn import LayerNorm

from nemo.collections.asr.parts.packed_sequence import (
    PackedEncoderActivations,
    PackedSlotLayout,
    _lengths_to_cu_seqlens,
    _new_packed_encoder_activations,
    packed_slot_layout,
)
from nemo.collections.asr.parts.submodules.causal_convs import CausalConv1D, CausalConv2D
from nemo.core.utils.optional_libs import TRITON_AVAILABLE, triton_required
from nemo.utils import logging

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.triton.depthwise_conv import dw_conv2d
    from nemo.collections.asr.parts.triton.subsampling import fused_conv_relu_dw


class FeatureStacking(nn.Module):
    """Stacks consecutive input frames and projects to model dimension.

    Reduces the temporal resolution by ``subsampling_factor`` while increasing
    the feature dimension proportionally, then linearly projects back to
    ``feat_out``.

    Args:
        subsampling_factor: Number of consecutive frames to stack.
        feat_in: Input feature dimension.
        feat_out: Output feature dimension.
    """

    def __init__(self, subsampling_factor: int, feat_in: int, feat_out: int):
        super().__init__()
        self.subsampling_factor = subsampling_factor
        self.proj = nn.Linear(subsampling_factor * feat_in, feat_out, bias=False)

    def compute_num_out_frames(self, in_frames):
        return (in_frames + self.subsampling_factor - 1) // self.subsampling_factor

    def get_sampling_frames(self):
        """Input frames consumed per output frame — probed by the cache-aware streaming utils."""
        return self.subsampling_factor

    def get_streaming_cache_size(self):
        """Input-frame look-back needed to reproduce the offline output. Stacking is
        non-overlapping, so a chunk aligned to ``subsampling_factor`` needs none."""
        return 0

    def forward(self, x, lengths=None):
        """
        Args:
            x: (B, C, T) input features.
            lengths: (B,) valid lengths per sample.
        Returns:
            x: (B, T', feat_out) stacked and projected features.
            lengths: (B,) updated lengths after subsampling.
        """
        if isinstance(x, PackedEncoderActivations):
            if lengths is not None:
                raise ValueError("lengths must be omitted when x is PackedEncoderActivations.")
            return self.forward_packed(x)
        if lengths is None:
            raise ValueError("lengths are required for padded FeatureStacking input.")
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        b, t, c = x.size()
        t_new = self.compute_num_out_frames(t)
        pad_size = t_new * self.subsampling_factor - t
        # Keep padding in the exported graph even when the example length is divisible.
        if torch.jit.is_tracing() or torch.compiler.is_compiling() or pad_size > 0:
            x = nn.functional.pad(x, (0, 0, 0, pad_size))
        x = x.reshape(b, t_new, c * self.subsampling_factor)
        x = self.proj(x)
        lengths = self.compute_num_out_frames(lengths)
        return x, lengths

    def forward_packed(self, packed: PackedEncoderActivations) -> PackedEncoderActivations:
        """Stack and project token-flat feature sequences without batch padding."""
        stacked = self.stack_packed(packed)
        return stacked.with_data(self.proj(stacked.data))

    def stack_packed(self, packed: PackedEncoderActivations) -> PackedEncoderActivations:
        """Stack token-flat frames, retaining packed metadata for grouped projection."""
        if packed.data.shape[1] * self.subsampling_factor != self.proj.in_features:
            raise ValueError(
                f"Expected packed feature width {self.proj.in_features // self.subsampling_factor}, "
                f"got {packed.data.shape[1]}."
            )
        output_lengths = self.compute_num_out_frames(packed.lengths)
        output_cu_seqlens = torch.cat(
            [output_lengths.new_zeros(1, dtype=torch.int32), output_lengths.cumsum(0, dtype=torch.int32)]
        )
        slot_count = int(output_cu_seqlens[-1].item()) * self.subsampling_factor
        slot_indices = torch.arange(slot_count, device=packed.data.device)
        slot_boundaries = output_cu_seqlens[1:].to(torch.int64) * self.subsampling_factor
        slot_sequence_ids = torch.bucketize(slot_indices, slot_boundaries, right=True)
        if isinstance(packed.padding_value, torch.Tensor):
            slots = packed.padding_value.to(packed.data)[slot_sequence_ids].clone()
        else:
            slots = packed.data.new_full((slot_count, packed.data.shape[1]), packed.padding_value)
        if packed.padded_length is not None and slot_count:
            slot_starts = output_cu_seqlens[slot_sequence_ids].to(torch.int64) * self.subsampling_factor
            local_slots = slot_indices - slot_starts
            slots.masked_fill_(local_slots.unsqueeze(1) >= packed.padded_length, 0.0)
        if packed.total_tokens:
            token_indices = torch.arange(packed.total_tokens, device=packed.data.device)
            sequence_ids = torch.bucketize(token_indices, packed.cu_seqlens[1:], right=True)
            local_positions = token_indices - packed.cu_seqlens[sequence_ids]
            slot_positions = output_cu_seqlens[sequence_ids].to(torch.int64) * self.subsampling_factor
            slots[slot_positions + local_positions] = packed.data
        stacked = slots.reshape(-1, self.proj.in_features)
        max_seqlen = self.compute_num_out_frames(packed.max_seqlen)
        return _new_packed_encoder_activations(stacked, output_lengths, output_cu_seqlens, max_seqlen)


class StackingSubsampling(torch.nn.Module):
    """Stacking subsampling which simply stacks consecutive frames to reduce the sampling rate
    Args:
        subsampling_factor (int): The subsampling factor
        feat_in (int): size of the input features
        feat_out (int): size of the output features
        norm (bool): whether to use an MLP layer after the stacking along with normalization. default is False.
    """

    def __init__(self, subsampling_factor, feat_in, feat_out, norm=False):
        super(StackingSubsampling, self).__init__()
        self.subsampling_factor = subsampling_factor
        self.proj_out = torch.nn.Linear(subsampling_factor * feat_in, feat_out)
        if norm:
            self.pre_norm = LayerNorm(feat_in)
        else:
            self.pre_norm = None

    def get_sampling_frames(self):
        return self.subsampling_factor

    def get_streaming_cache_size(self):
        return 0

    def forward(self, x, lengths):
        b, t, h = x.size()
        t_new = (t + self.subsampling_factor - 1) // self.subsampling_factor
        pad_size = t_new * self.subsampling_factor - t
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_size))
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        x = torch.reshape(x, (b, t_new, h * self.subsampling_factor))
        x = self.proj_out(x)
        lengths = torch.div(lengths + pad_size, self.subsampling_factor, rounding_mode='floor')
        return x, lengths


# Conv kernels index elements with 32-bit ints, so a tensor entering or leaving a conv must
# hold fewer than this many elements. See https://github.com/pytorch/pytorch/issues/80020
_MAX_CONV_NUMEL_32BIT = 2**31 - 1


class ConvSubsampling(torch.nn.Module):
    """Convolutional subsampling which supports VGGNet and striding approach introduced in:
    VGGNet Subsampling: Transformer-transducer: end-to-end speech recognition with self-attention (https://arxiv.org/pdf/1910.12977.pdf)
    Striding Subsampling: "Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition" by Linhao Dong et al. (https://ieeexplore.ieee.org/document/8462506)
    Args:
        subsampling (str): The subsampling technique from {"vggnet", "striding", "dw-striding"}
        subsampling_factor (int): The subsampling factor, a power of 2 for the default layout;
            with `conv_strides`, the product of the per-stage time strides
        subsampling_conv_chunking_factor (int): Input chunking factor which can be -1 (no chunking)
        1 (auto) or a power of 2. Default is 1
        feat_in (int): size of the input features
        feat_out (int): size of the output features
        conv_channels: number of channels of the convolution layers,
            one value for all stages, or one per stage together with `conv_strides`
        activation (Module): activation function, default is nn.ReLU()
        conv_strides: for "striding", the time stride of each 3x3 stage instead of stride 2 on both axes;
            the frequency axis is then reduced only by `freq_pools`,
            and the product of the strides must equal `subsampling_factor`
        freq_pools: with `conv_strides`, a max-pool over the frequency axis after each stage, 1 for none
        out_bias: bias of the output projection
    """

    def __init__(
        self,
        subsampling,
        subsampling_factor,
        feat_in,
        feat_out,
        conv_channels: int | Sequence[int],
        subsampling_conv_chunking_factor=1,
        activation=nn.ReLU(),
        is_causal=False,
        use_triton: bool | None = None,
        conv_strides: Sequence[int] | None = None,
        freq_pools: Sequence[int] | None = None,
        out_bias: bool = True,
    ):
        super(ConvSubsampling, self).__init__()
        self._subsampling = subsampling
        # per-stage channel lists arrive as plain lists or as OmegaConf sequences from Hydra instantiation
        channels_per_stage = isinstance(conv_channels, Sequence) and not isinstance(conv_channels, str)
        if channels_per_stage and conv_strides is None:
            raise ValueError(f"conv_channels per stage ({list(conv_channels)}) needs conv_strides")
        # the 32-bit indexing guard sizes the output of the first convolution
        self._conv_channels = conv_channels[0] if channels_per_stage else conv_channels
        self._feat_in = feat_in
        self._feat_out = feat_out
        self._time_strides = list(conv_strides) if conv_strides is not None else None
        self._freq_pools = list(freq_pools) if freq_pools is not None else None

        if self._time_strides is not None:
            if subsampling != 'striding':
                raise ValueError("conv_strides is only supported with subsampling='striding'")
            if is_causal:
                raise ValueError("conv_strides is not supported with causal subsampling")
            if any(not isinstance(stride, int) or stride < 1 for stride in self._time_strides):
                raise ValueError(f"conv_strides must be positive integers, got {self._time_strides}")
            if self._freq_pools is not None and any(
                not isinstance(pool, int) or pool < 1 for pool in self._freq_pools
            ):
                raise ValueError(f"freq_pools must be positive integers, got {self._freq_pools}")
            if math.prod(self._time_strides) != subsampling_factor:
                raise ValueError(
                    f"The product of conv_strides {self._time_strides} must equal subsampling_factor {subsampling_factor}"
                )
            self._sampling_num = len(self._time_strides)
            if self._freq_pools is None:
                self._freq_pools = [1] * self._sampling_num
        else:
            if subsampling_factor % 2 != 0:
                raise ValueError("Sampling factor should be a multiply of 2!")
            self._sampling_num = int(math.log(subsampling_factor, 2))
        self.subsampling_factor = subsampling_factor
        self.is_causal = is_causal

        if (
            subsampling_conv_chunking_factor != -1
            and subsampling_conv_chunking_factor != 1
            and subsampling_conv_chunking_factor % 2 != 0
        ):
            raise ValueError("subsampling_conv_chunking_factor should be -1, 1, or a power of 2")
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor

        in_channels = 1
        layers = []

        if subsampling == 'vggnet':
            self._stride = 2
            self._kernel_size = 2
            self._ceil_mode = True

            self._left_padding = 0
            self._right_padding = 0

            for i in range(self._sampling_num):
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=in_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1
                    )
                )
                layers.append(activation)
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1
                    )
                )
                layers.append(activation)
                layers.append(
                    torch.nn.MaxPool2d(
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._left_padding,
                        ceil_mode=self._ceil_mode,
                    )
                )
                in_channels = conv_channels

        elif subsampling == 'dw_striding':
            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False

            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0

            # Layer 1
            if self.is_causal:
                layers.append(
                    CausalConv2D(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=None,
                    )
                )
            else:
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._left_padding,
                    )
                )
            in_channels = conv_channels
            layers.append(activation)

            for i in range(self._sampling_num - 1):
                if self.is_causal:
                    layers.append(
                        CausalConv2D(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                            groups=in_channels,
                        )
                    )
                else:
                    layers.append(
                        torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                            groups=in_channels,
                        )
                    )

                layers.append(
                    torch.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=1,
                    )
                )
                layers.append(activation)
                in_channels = conv_channels

        elif subsampling == 'striding' and self._time_strides is not None:
            self._stride = max(self._time_strides)
            self._kernel_size = 3
            self._ceil_mode = False
            self._left_padding = 1
            self._right_padding = 1
            self._max_cache_len = 0
            channels = list(conv_channels) if channels_per_stage else [conv_channels] * self._sampling_num
            if len(channels) != self._sampling_num or len(self._freq_pools) != self._sampling_num:
                raise ValueError("conv_channels and freq_pools need one entry per stride")
            for out_channels, stride, pool in zip(channels, self._time_strides, self._freq_pools):
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=self._kernel_size,
                        stride=(stride, 1),
                        padding=self._left_padding,
                    )
                )
                layers.append(activation)
                if pool > 1:
                    layers.append(torch.nn.MaxPool2d(kernel_size=(1, pool), stride=(1, pool)))
                in_channels = out_channels

        elif subsampling == 'striding':
            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False

            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0

            for i in range(self._sampling_num):
                if self.is_causal:
                    layers.append(
                        CausalConv2D(
                            in_channels=in_channels,
                            out_channels=conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                        )
                    )
                else:
                    layers.append(
                        torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                        )
                    )
                layers.append(activation)
                in_channels = conv_channels

        elif subsampling == 'striding_conv1d':

            in_channels = feat_in

            self._stride = 2
            self._kernel_size = 5
            self._ceil_mode = False

            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0

            for i in range(self._sampling_num):
                if self.is_causal:
                    layers.append(
                        CausalConv1D(
                            in_channels=in_channels,
                            out_channels=feat_out if self._sampling_num == i + 1 else conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                        )
                    )
                else:
                    layers.append(
                        torch.nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=feat_out if self._sampling_num == i + 1 else conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                        )
                    )
                layers.append(activation)
                in_channels = conv_channels

        elif subsampling == 'dw_striding_conv1d':

            in_channels = feat_in

            self._stride = 2
            self._kernel_size = 5
            self._ceil_mode = False

            self._left_padding = (self._kernel_size - 1) // 2
            self._right_padding = (self._kernel_size - 1) // 2

            # Layer 1
            layers.extend(
                [
                    torch.nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._left_padding,
                        groups=in_channels,
                    ),
                    torch.nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=feat_out if self._sampling_num == 1 else conv_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=1,
                    ),
                ]
            )
            in_channels = conv_channels
            layers.append(activation)

            for i in range(self._sampling_num - 1):
                layers.extend(
                    [
                        torch.nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                            groups=in_channels,
                        ),
                        torch.nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=feat_out if self._sampling_num == i + 2 else conv_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            groups=1,
                        ),
                    ]
                )
                layers.append(activation)
                in_channels = conv_channels

        else:
            raise ValueError(f"Not valid sub-sampling: {subsampling}!")

        if subsampling == "striding" and self._time_strides is not None:
            # the strided stages keep the frequency axis; only the pools reduce it
            freq = feat_in
            for pool in self._freq_pools:
                freq //= pool
            self.out = torch.nn.Linear(in_channels * freq, feat_out, bias=out_bias)
            self.conv2d_subsampling = True
        elif subsampling in ["vggnet", "dw_striding", "striding"]:

            in_length = torch.tensor(feat_in, dtype=torch.float)
            out_length = calc_length(
                lengths=in_length,
                all_paddings=self._left_padding + self._right_padding,
                kernel_size=self._kernel_size,
                stride=self._stride,
                ceil_mode=self._ceil_mode,
                repeat_num=self._sampling_num,
            )
            self.out = torch.nn.Linear(conv_channels * int(out_length), feat_out, bias=out_bias)
            self.conv2d_subsampling = True
        elif subsampling in ["striding_conv1d", "dw_striding_conv1d"]:
            self.out = None
            self.conv2d_subsampling = False
        else:
            raise ValueError(f"Not valid sub-sampling: {subsampling}!")

        self.conv = MaskedConvSequential(*layers)

        # The kernels implement `dw_striding`'s layout, [conv, act] + (sampling_num - 1) x
        # [dw, pw, act], with ReLU baked in; a factor of 2 stops after [conv, act], leaving no
        # depthwise to fuse.
        supported = subsampling == 'dw_striding' and self._sampling_num >= 2 and isinstance(activation, nn.ReLU)
        if use_triton and not supported:
            logging.warning(
                "use_triton=True was requested, but the fused kernels only cover dw_striding with "
                "subsampling_factor >= 4 and a ReLU activation, falling back to PyTorch instead."
            )
        self.conv.fuse_triton = supported and (TRITON_AVAILABLE if use_triton is None else use_triton)

    def get_sampling_frames(self):
        return [1, self.subsampling_factor]

    def get_streaming_cache_size(self):
        return [0, self.subsampling_factor + 1]

    def _first_conv_output_numel(self, x):
        """Element count of the first conv's output, the largest activation of the default stacks.

        ``x`` is the ``(B, T, F)`` input before the channel dim is added. Only valid for the
        strided 'striding'/'dw_striding' variants; 'vgg' starts with stride-1 convs, so its
        largest activation is not bounded by this estimate.
        With per-stage strides a later conv can be larger, see ``_max_conv_numel``.
        """
        b, t, f = x.size()
        pad = (self._left_padding, self._right_padding)
        if self._time_strides is not None:
            # the per-stage stages stride only in time and keep the frequency axis
            out_t = calculate_conv_output_size(t, self._kernel_size, self._time_strides[0], pad)
            out_f = calculate_conv_output_size(f, self._kernel_size, 1, pad)
        else:
            out_t = calculate_conv_output_size(t, self._kernel_size, self._stride, pad)
            out_f = calculate_conv_output_size(f, self._kernel_size, self._stride, pad)
        return b * self._conv_channels * out_t * out_f

    def _max_conv_numel(self, x):
        """Largest element count among the conv input and every conv output of the stack.

        The 32-bit indexing guard and the automatic chunking bound the batch by this count.
        With per-stage strides and channels a later conv can hold more elements than the first,
        when its channels grow faster than the strides and pools shrink the map,
        so the stages are walked one by one; the default stacks keep the first-conv estimate.
        """
        b, t, f = x.size()
        largest = b * t * f
        if self._time_strides is None:
            return max(largest, self._first_conv_output_numel(x))
        pad = (self._left_padding, self._right_padding)
        convs = (layer for layer in self.conv if isinstance(layer, torch.nn.Conv2d))
        for conv, stride, pool in zip(convs, self._time_strides, self._freq_pools):
            t = calculate_conv_output_size(t, self._kernel_size, stride, pad)
            f = calculate_conv_output_size(f, self._kernel_size, 1, pad)
            # the conv output is counted before the pool that follows it
            largest = max(largest, b * conv.out_channels * t * f)
            f //= pool
        return largest

    def forward_packed(self, packed: PackedEncoderActivations) -> PackedEncoderActivations:
        """Subsample token-flat features to token-flat outputs.

        The utterances are laid out on one gapped time axis (``PackedSlotLayout``):
        every slot starts at a multiple of the subsampling factor
        and keeps at least one empty frame after the utterance at every stage,
        and the frames outside the utterances are zeroed before every layer as the padded forward masks them.
        The strided convolutions therefore keep their stride phase per utterance and read zeros beyond it,
        and the result equals ``forward`` on the padded batch at the valid frames.
        Only the non-causal ``striding`` and ``dw_striding`` stacks are supported;
        VGGNet pools with ``ceil_mode``, which the floor-mode lengths of the gapped layout do not model.

        Args:
            packed: features of all utterances, data of shape ``(sum(T_i), feat_in)``.

        Returns:
            The subsampled features, data of shape ``(sum(T'_i), feat_out)``.
        """
        if not self.conv2d_subsampling or self._subsampling == 'vggnet':
            raise NotImplementedError(f"subsampling={self._subsampling!r} has no sequence-packed form.")
        if self.is_causal:
            raise NotImplementedError("Causal subsampling has no sequence-packed form.")
        if packed.data.shape[1] != self._feat_in:
            raise ValueError(f"Packed features must have width {self._feat_in}, got {packed.data.shape[1]}.")
        self._check_packed_stack_gaps()
        if packed.batch_size == 0:
            # a rank without utterances keeps every parameter in the graph, as the packed attention does
            anchor = packed.data.sum() * 0.0
            for parameter in self.conv.parameters():
                anchor = anchor + parameter.sum().to(anchor.dtype) * 0.0
            empty = self.out(packed.data.new_zeros((0, self.out.in_features))) + anchor
            return _new_packed_encoder_activations(empty, packed.lengths, packed.cu_seqlens, 0)
        layout = packed_slot_layout(packed, alignment=self.subsampling_factor, gap=self.subsampling_factor)
        outputs = []
        lengths = []
        for first, last in self._packed_sequence_groups(layout):
            group_slots = int((layout.slot_offsets[last - 1] + layout.slot_lengths[last - 1]).item()) - int(
                layout.slot_offsets[first].item()
            )
            start, end = int(packed.cu_seqlens[first].item()), int(packed.cu_seqlens[last].item())
            data, out_lengths = self._forward_packed_group(
                packed.data[start:end],
                packed.lengths[first:last],
                layout.slot_offsets[first:last] - layout.slot_offsets[first],
                layout.token_slots[start:end] - layout.slot_offsets[first],
                group_slots,
            )
            outputs.append(data)
            lengths.append(out_lengths)
        out_lengths = torch.cat(lengths).to(torch.int64)
        max_seqlen = int(out_lengths.max().item()) if out_lengths.numel() else 0
        return _new_packed_encoder_activations(
            torch.cat(outputs), out_lengths, _lengths_to_cu_seqlens(out_lengths), max_seqlen
        )

    def _check_packed_stack_gaps(self):
        """Verify that no layer of the stack reads across utterances on the gapped time axis.

        At the input stage every slot starts at a multiple of the subsampling factor
        and leaves at least that many empty frames after its utterance.
        A layer with time kernel k (dilation included), stride s and padding (left, right) reads at most
        max(left, right) frames beyond an utterance, which must stay inside the gap,
        and leaves at least ceil((gap + k - left - right - s) / s) empty frames behind it.
        """
        gap = self.subsampling_factor
        for layer in self.conv:
            if not isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.MaxPool2d)):
                continue  # activations
            kernel, stride = _time_axis(layer.kernel_size), _time_axis(layer.stride)
            kernel = (kernel - 1) * _time_axis(layer.dilation) + 1
            left, right = _layer_padding(layer)
            if max(left, right) > gap:
                raise NotImplementedError(
                    f"{type(layer).__name__} with time padding {(left, right)} reads beyond the {gap} empty frames "
                    "between utterances; this convolution stack has no sequence-packed form."
                )
            gap = -(-(gap + kernel - left - right - stride) // stride)

    def _packed_sequence_groups(self, layout: PackedSlotLayout) -> list[tuple[int, int]]:
        """Split the utterances into groups whose convolution activations stay below the indexing limit.

        The padded forward splits the batch for the same reason (``subsampling_conv_chunking_factor``);
        here a group is a run of utterances, and a group boundary is exact because the slots are independent.
        """
        batch_size = layout.slot_lengths.numel()
        if batch_size == 0:
            return []
        if self.subsampling_conv_chunking_factor == -1:
            return [(0, batch_size)]
        if self.subsampling_conv_chunking_factor > 1:
            group_count = min(self.subsampling_conv_chunking_factor, batch_size)
            bounds = [round(index * batch_size / group_count) for index in range(group_count + 1)]
            return [
                (bounds[index], bounds[index + 1]) for index in range(group_count) if bounds[index + 1] > bounds[index]
            ]
        # the widest activation of the stack must stay indexable, see https://github.com/pytorch/pytorch/issues/80020
        channels = max(layer.out_channels for layer in self.conv if isinstance(layer, nn.Conv2d))
        slot_limit = max(1, (2**31 - 1) // (channels * self._feat_in))
        if layout.total_slots <= slot_limit:
            return [(0, batch_size)]
        groups = []
        first = 0
        slot_lengths = layout.slot_lengths.tolist()
        current = 0
        for index, slots in enumerate(slot_lengths):
            if current and current + slots > slot_limit:
                groups.append((first, index))
                first, current = index, 0
            current += slots
        groups.append((first, batch_size))
        return groups

    def _forward_packed_group(
        self,
        data: torch.Tensor,
        lengths: torch.Tensor,
        slot_offsets: torch.Tensor,
        token_slots: torch.Tensor,
        total_slots: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The convolution stack over one gapped time axis; returns the valid output frames and their lengths."""
        x = data.new_zeros((total_slots, data.shape[1])).index_copy(0, token_slots, data)
        x = x.view(1, 1, total_slots, -1)
        current_lengths = lengths.to(torch.int64)
        current_offsets = slot_offsets
        for layer in self.conv:
            x = x * self._packed_stage_mask(x, current_offsets, current_lengths)
            x = layer(x)
            if hasattr(layer, 'stride') and layer.stride != (1, 1):
                current_lengths = calculate_conv_output_size(
                    current_lengths, layer.kernel_size[0], layer.stride[0], _layer_padding(layer)
                )
                current_offsets = torch.div(current_offsets, layer.stride[0], rounding_mode='floor')
        x = x * self._packed_stage_mask(x, current_offsets, current_lengths)

        total_out = int(current_lengths.sum().item())
        out_sequence_ids = torch.repeat_interleave(
            torch.arange(current_lengths.numel(), device=x.device), current_lengths, output_size=total_out
        )
        out_starts = torch.cumsum(current_lengths, dim=0) - current_lengths
        out_positions = torch.arange(total_out, device=x.device) - out_starts[out_sequence_ids]
        out_slots = current_offsets[out_sequence_ids] + out_positions
        # the padded forward flattens channels before frequency bins
        x = x.squeeze(0).permute(1, 0, 2).reshape(x.shape[2], -1).index_select(0, out_slots)
        return self.out(x), current_lengths

    @staticmethod
    def _packed_stage_mask(x: torch.Tensor, slot_offsets: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """One for the slots inside an utterance's valid frames at this stage, zero elsewhere; shape (1, 1, S, 1)."""
        positions = torch.arange(x.shape[2], device=x.device)
        sequence_ids = torch.bucketize(positions, slot_offsets[1:], right=True)
        valid = positions - slot_offsets[sequence_ids] < lengths[sequence_ids]
        return valid.to(x.dtype).view(1, 1, -1, 1)

    def forward(self, x, lengths):
        if self._time_strides is not None:
            out_lengths = lengths
            for stride in self._time_strides:
                out_lengths = calc_length(
                    out_lengths,
                    all_paddings=self._left_padding + self._right_padding,
                    kernel_size=self._kernel_size,
                    stride=stride,
                    ceil_mode=self._ceil_mode,
                )
        else:
            out_lengths = calc_length(
                lengths,
                all_paddings=self._left_padding + self._right_padding,
                kernel_size=self._kernel_size,
                stride=self._stride,
                ceil_mode=self._ceil_mode,
                repeat_num=self._sampling_num,
            )

        # Transpose to Channel First mode
        if not self.conv2d_subsampling:
            x = x.transpose(1, 2)

        # split inputs if chunking_factor is set
        if self.subsampling_conv_chunking_factor != -1 and self.conv2d_subsampling:
            if self.subsampling_conv_chunking_factor == 1:
                # if subsampling_conv_chunking_factor is 1, we split only if needed
                # avoiding a bug / feature limiting indexing of tensors to 2**31
                # see https://github.com/pytorch/pytorch/issues/80020
                # Split on '>=': at equality the tensor already holds INT_MAX elements, the
                # value that trips canUse32BitIndexMath. Guard the conv input and every conv output.
                need_to_split = self._max_conv_numel(x) >= _MAX_CONV_NUMEL_32BIT
            else:
                # if subsampling_conv_chunking_factor > 1 we always split
                need_to_split = True

            if need_to_split:
                x, lengths, success = self.conv_split_by_batch(x, lengths)
                if not success:  # if unable to split by batch, try by channel
                    if self._subsampling == 'dw_striding':
                        # TODO: implement lengths inside conv_split_by_channel
                        x = self.conv_split_by_channel(x)
                        lengths = out_lengths
                    else:
                        x, lengths = self.conv(x, lengths)  # try anyway
            else:
                x, lengths = self.conv(x, lengths)
        else:
            x, lengths = self.conv(x)

        # Flatten Channel and Frequency Axes
        if self.conv2d_subsampling:
            b, c, t, f = x.size()
            x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        # Transpose to Channel Last mode
        else:
            x = x.transpose(1, 2)

        return x, lengths

    def reset_parameters(self):
        # initialize weights
        if self._subsampling == 'dw_striding':
            with torch.no_grad():
                # init conv
                scale = 1.0 / self._kernel_size
                dw_max = (self._kernel_size**2) ** -0.5
                pw_max = self._conv_channels**-0.5

                torch.nn.init.uniform_(self.conv[0].weight, -scale, scale)
                torch.nn.init.uniform_(self.conv[0].bias, -scale, scale)

                for idx in range(2, len(self.conv), 3):
                    torch.nn.init.uniform_(self.conv[idx].weight, -dw_max, dw_max)
                    torch.nn.init.uniform_(self.conv[idx].bias, -dw_max, dw_max)
                    torch.nn.init.uniform_(self.conv[idx + 1].weight, -pw_max, pw_max)
                    torch.nn.init.uniform_(self.conv[idx + 1].bias, -pw_max, pw_max)

                # init fc (80 * 64 = 5120 from https://github.com/kssteven418/Squeezeformer/blob/13c97d6cf92f2844d2cb3142b4c5bfa9ad1a8951/src/models/conformer_encoder.py#L487
                fc_scale = (self._feat_out * self._feat_in / self._sampling_num) ** -0.5
                torch.nn.init.uniform_(self.out.weight, -fc_scale, fc_scale)
                if self.out.bias is not None:
                    torch.nn.init.uniform_(self.out.bias, -fc_scale, fc_scale)

    def conv_split_by_batch(self, x, lengths):
        """Tries to split input by batch, run conv and concat results"""
        b, *_ = x.size()
        if b == 1:  # can't split if batch size is 1
            return x, lengths, False

        if self.subsampling_conv_chunking_factor > 1:
            cf = self.subsampling_conv_chunking_factor
            logging.debug(f'using manually set chunking factor: {cf}')
        else:
            # avoiding a bug / feature limiting indexing of tensors to 2**31
            # see https://github.com/pytorch/pytorch/issues/80020
            # Smallest power-of-two split with each chunk strictly below the limit (+1 forces
            # strict); size against the largest of conv input and conv outputs.
            numel = self._max_conv_numel(x)
            cf = 2 ** math.ceil(math.log(numel // _MAX_CONV_NUMEL_32BIT + 1, 2))
            logging.debug(f'using auto set chunking factor: {cf}')

        new_batch_size = b // cf
        if new_batch_size == 0:
            # If cf > b and one sample fits, use single-sample batches rather than the channel
            # fallback (which runs the full first conv up front).
            if self._max_conv_numel(x[:1]) >= _MAX_CONV_NUMEL_32BIT:
                return x, lengths, False
            new_batch_size = 1

        logging.debug(f'conv subsampling: using split batch size {new_batch_size}')

        ans = [
            self.conv(chunk, ln)
            for chunk, ln in zip(
                torch.split(x, new_batch_size, 0),
                torch.split(lengths, new_batch_size, 0),
            )
        ]
        return torch.cat([a[0] for a in ans]), torch.cat([a[1] for a in ans]), True

    def conv_split_by_channel(self, x):
        """For dw convs, tries to split input by time, run conv and concat results"""

        # Note: this method doesn't use the convolution masking implemented in MaskedConvolutionSequential
        x = x.unsqueeze(0)
        x = self.conv[0](x)  # full conv2D
        x = self.conv[1](x)  # activation

        for i in range(self._sampling_num - 1):
            _, c, t, _ = x.size()

            if self.subsampling_conv_chunking_factor > 1:
                cf = self.subsampling_conv_chunking_factor
                logging.debug(f'using manually set chunking factor: {cf}')
            else:
                # avoiding a bug / feature limiting indexing of tensors to 2**31
                # see https://github.com/pytorch/pytorch/issues/80020
                # +1 keeps each chunk strictly below the limit and avoids a fractional factor.
                cf = 2 ** math.ceil(math.log(torch.numel(x) // _MAX_CONV_NUMEL_32BIT + 1, 2))
                logging.debug(f'using auto set chunking factor: {cf}')

            new_c = int(c // cf)
            if new_c == 0:
                logging.warning(f'chunking factor {cf} is too high; splitting down to one channel.')
                new_c = 1

            new_t = int(t // cf)
            if new_t == 0:
                logging.warning(f'chunking factor {cf} is too high; splitting down to one timestep.')
                new_t = 1

            logging.debug(f'conv dw subsampling: using split C size {new_c} and split T size {new_t}')
            x = self.channel_chunked_conv(self.conv[i * 3 + 2], new_c, x)  # conv2D, depthwise

            # splitting pointwise convs by time
            x = torch.cat([self.conv[i * 3 + 3](chunk) for chunk in torch.split(x, new_t, 2)], 2)  # conv2D, pointwise
            x = self.conv[i * 3 + 4](x)  # activation
        return x

    def channel_chunked_conv(self, conv, chunk_size, x):
        """Performs channel chunked convolution"""

        ind = 0
        out_chunks = []
        for chunk in torch.split(x, chunk_size, 1):
            step = chunk.size()[1]

            if self.is_causal:
                chunk = nn.functional.pad(
                    chunk, pad=(self._kernel_size - 1, self._stride - 1, self._kernel_size - 1, self._stride - 1)
                )
                ch_out = nn.functional.conv2d(
                    chunk,
                    conv.weight[ind : ind + step, :, :, :],
                    bias=conv.bias[ind : ind + step],
                    stride=self._stride,
                    padding=0,
                    groups=step,
                )
            else:
                ch_out = nn.functional.conv2d(
                    chunk,
                    conv.weight[ind : ind + step, :, :, :],
                    bias=conv.bias[ind : ind + step],
                    stride=self._stride,
                    padding=self._left_padding,
                    groups=step,
                )
            out_chunks.append(ch_out)
            ind += step

        return torch.cat(out_chunks, 1)

    def change_subsampling_conv_chunking_factor(self, subsampling_conv_chunking_factor: int):
        if (
            subsampling_conv_chunking_factor != -1
            and subsampling_conv_chunking_factor != 1
            and subsampling_conv_chunking_factor % 2 != 0
        ):
            raise ValueError("subsampling_conv_chunking_factor should be -1, 1, or a power of 2")
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor


def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
    """Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)


class SubsamplingReductionModule(nn.Module):
    """Downsamples the audio signal in time dimension."""

    def __init__(self, reduction: str, d_model: int, reduction_factor: int = 2):
        super().__init__()

        assert reduction in ['pooling', 'striding']

        self.reduction = reduction
        self.d_model = d_model

        if reduction == 'pooling':
            self.reduction_enc = nn.MaxPool1d(kernel_size=reduction_factor)
            self.padding = 0
            self.kernel_size = self.reduction_enc.kernel_size
            self.stride = self.reduction_enc.stride
        elif reduction == 'striding':
            self.reduction_enc = ConvSubsampling(
                subsampling='striding',
                subsampling_factor=reduction_factor,
                feat_in=d_model,
                feat_out=d_model,
                conv_channels=d_model,
                activation=nn.ReLU(),
                is_causal=False,
            )

    def forward(self, x, lengths):
        """Shapes:
        - x: [B, T, C]
        - lengths: [B]
        """

        if self.reduction == 'striding':
            x, lengths = self.reduction_enc(x=x, lengths=lengths)
        else:
            x = torch.transpose(x, 1, 2)  # [B, C, T]
            lengths = calc_length(
                lengths=lengths,
                all_paddings=self.padding,
                kernel_size=self.kernel_size,
                stride=self.stride,
                ceil_mode=False,
                repeat_num=1,  # a single MaxPool1d(kernel_size=reduction_factor) is applied below
            )
            x = self.reduction_enc(x)
            x = torch.transpose(x, 1, 2)  # [B, T, C]

        return x, lengths


def apply_channel_mask(tensor, mask):
    """Apply mask to tensor with channel dimension."""
    # tensor: (batch, channels, time, features)
    # mask: (batch, time, features)
    batch_size, channels, time, features = tensor.shape
    expanded_mask = mask.unsqueeze(1).expand(batch_size, channels, time, features)
    return tensor * expanded_mask


def calculate_conv_output_size(input_size: torch.Tensor, kernel_size: int, stride: int, padding: tuple[int, int]):
    """Calculate exact output size after convolution."""
    return (input_size + padding[0] + padding[1] - kernel_size) // stride + 1


class MaskedConvSequential(nn.Sequential):
    # Set by ConvSubsampling; off by default, so every other subsampling type stays on PyTorch.
    fuse_triton = False

    def forward(self, x, lengths):
        # Convert input (batch, time, features) to conv format
        x = x.unsqueeze(1)  # (batch, 1, time, features)
        current_lengths = lengths

        # Tracing and export cannot capture a Triton launch, the fused kernel returns no input
        # gradient, and its weight gradients accumulate through atomics, so their summation order
        # varies between runs.
        if (
            self.fuse_triton
            and x.is_cuda
            and not x.requires_grad
            and not torch.are_deterministic_algorithms_enabled()
            and not (torch.jit.is_tracing() or torch.compiler.is_exporting())
        ):
            x, current_lengths, mask = self._forward_fused(x, current_lengths)
        else:
            x, current_lengths, mask = self._forward_torch(x, current_lengths)

        # Final masking
        x = apply_channel_mask(x, mask)
        return x, current_lengths.long()

    def _forward_torch(self, x, current_lengths):
        mask = self._create_mask(x, current_lengths.long())

        # Process through each layer with mask propagation
        for i, layer in enumerate(self):
            # Apply current mask before layer
            x = apply_channel_mask(x, mask)

            # Apply layer
            x = layer(x)

            # Update lengths for stride operations with proper padding
            if hasattr(layer, 'stride') and layer.stride != (1, 1):
                current_lengths = calculate_conv_output_size(
                    current_lengths, layer.kernel_size[0], layer.stride[0], _layer_padding(layer)
                )
                mask = self._create_mask(x, current_lengths.long())

        return x, current_lengths, mask

    @triton_required
    def _forward_fused(self, x, current_lengths):
        """The `dw_striding` stack, with conv0 and the depthwise layers as Triton kernels.

        The stack is `[conv, act] + (sampling_num - 1) x [dw, pw, act]`. One kernel covers the
        leading `conv, act, dw`; the loop over `self[3:]` runs each depthwise as a kernel, each
        pointwise as a linear, and every other layer as itself. Lengths change only at the
        depthwise layers.

        Tensors are channels-last throughout, `(batch, time, freq, channels)`, and one permute at
        the end returns the `(batch, channels, time, freq)` the caller expects.

        The kernels read zeros beyond their input lengths and write zeros beyond their output
        lengths. Only the trailing pointwise and activation touch the padded tail, which
        `apply_channel_mask` clears at the end of `forward`.
        """
        conv0, _, first_depthwise, first_pointwise, activation = self[:5]
        # conv -> ReLU -> depthwise in one kernel; the intermediate never reaches memory.
        x, current_lengths = fused_conv_relu_dw(
            x,
            conv0.weight,
            conv0.bias,
            first_depthwise.weight,
            first_depthwise.bias,
            *_layer_padding(conv0),
            current_lengths,
        )
        x = _pointwise_block(x, first_pointwise, activation)

        body = self[5:]
        for i in range(0, len(body), 3):
            depthwise, pointwise, activation = body[i : i + 3]
            # The kernel masks its own output, so it needs the post-stride lengths.
            next_lengths = calculate_conv_output_size(
                current_lengths, depthwise.kernel_size[0], depthwise.stride[0], _layer_padding(depthwise)
            )
            x = dw_conv2d(
                x,
                depthwise.weight,
                depthwise.bias,
                depthwise.stride,
                *_layer_padding(depthwise),
                current_lengths,
                next_lengths,
            )
            current_lengths = next_lengths
            x = _pointwise_block(x, pointwise, activation)

        x = x.permute(0, 3, 1, 2)
        return x, current_lengths, self._create_mask(x, current_lengths.long())

    def _create_mask(self, tensor, lengths):
        """Create mask matching tensor dimensions."""
        batch_size, channels, time, features = tensor.shape
        time_mask = torch.arange(time, device=tensor.device).expand(batch_size, time) < lengths.unsqueeze(1)
        return time_mask.unsqueeze(-1).expand(batch_size, time, features).to(tensor.dtype)


def _layer_padding(layer):
    """The (start, end) padding of a convolution.

    nn.Conv2d's `.padding` is (pad_h, pad_w), one value per axis and symmetric within it, so the
    height value is both edges. CausalConv2D keeps its two edges on private attributes.
    """
    if hasattr(layer, "_left_padding"):
        return layer._left_padding, layer._right_padding
    if isinstance(layer.padding, int):  # pooling layers keep one value for both axes
        return layer.padding, layer.padding
    return layer.padding[0], layer.padding[0]


def _time_axis(value):
    """The time (height) entry of a per-axis layer attribute, which pooling layers may keep as one int."""
    return value[0] if isinstance(value, (tuple, list)) else value


def _is_depthwise(layer):
    """A depthwise convolution: one group per channel."""
    return isinstance(layer, nn.Conv2d) and layer.groups > 1


def _is_pointwise(layer):
    """A 1x1 convolution over all channels, which is a contraction over the channel axis alone."""
    return isinstance(layer, nn.Conv2d) and layer.groups == 1 and layer.kernel_size == (1, 1)


def _pointwise_block(x, conv, activation):
    # kernel_size=1 convs are pointwise, i.e. linear, but nn.Conv2d dispatches to much slower
    # cuBLAS kernels. flatten(1) on the weight is a free view, so checkpoints are unchanged.
    # F.linear on an N-D input returns a view of its 2D result. An in-place activation on a
    # view copies the whole tensor in backward, so x is flattened and the activation runs on
    # the 2D result itself.
    # TODO: remove the shape manipulation once https://github.com/pytorch/pytorch/pull/194077
    # is in the minimum required PyTorch version.
    b, t, f, c = x.shape
    x = nn.functional.linear(x.view(-1, c), conv.weight.flatten(1), conv.bias)
    return activation(x).view(b, t, f, -1)
