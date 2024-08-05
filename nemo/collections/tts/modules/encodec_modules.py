# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

# MIT License
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

from nemo.collections.common.parts.utils import mask_sequence_tensor
from nemo.collections.tts.losses.audio_codec_loss import MaskedMSELoss
from nemo.collections.tts.modules.audio_codec_modules import (
    CodecActivation,
    Conv1dNorm,
    Conv2dNorm,
    ConvTranspose1dNorm,
    VectorQuantizerBase,
    get_down_sample_padding,
)
from nemo.collections.tts.parts.utils.distributed import broadcast_tensors
from nemo.core.classes.common import typecheck
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types.elements import AudioSignal, EncodedRepresentation, Index, LengthsType, LossType, VoidType
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging
from nemo.utils.decorators import experimental


class SEANetResnetBlock(NeuralModule):
    def __init__(self, channels: int, activation: str = "elu"):
        super().__init__()
        self.pre_activation = CodecActivation(activation=activation, channels=channels)
        hidden_channels = channels // 2
        self.pre_conv = Conv1dNorm(in_channels=channels, out_channels=channels, kernel_size=1)
        self.res_conv1 = Conv1dNorm(in_channels=channels, out_channels=hidden_channels, kernel_size=3)
        self.post_activation = CodecActivation(activation=activation, channels=hidden_channels)
        self.res_conv2 = Conv1dNorm(in_channels=hidden_channels, out_channels=channels, kernel_size=1)

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'C', 'T_input'), VoidType()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'C', 'T_out'), VoidType()),
        }

    def remove_weight_norm(self):
        self.pre_conv.remove_weight_norm()
        self.res_conv1.remove_weight_norm()
        self.res_conv2.remove_weight_norm()

    @typecheck()
    def forward(self, inputs, input_len):
        res = self.pre_activation(inputs)
        res = self.res_conv1(inputs=res, input_len=input_len)
        res = self.post_activation(res)
        res = self.res_conv2(inputs=res, input_len=input_len)

        out = self.pre_conv(inputs=inputs, input_len=input_len) + res
        out = mask_sequence_tensor(out, input_len)
        return out


class SEANetRNN(NeuralModule):
    def __init__(self, dim: int, num_layers: int, rnn_type: str = "lstm", use_skip: bool = False):
        super().__init__()
        self.use_skip = use_skip
        if rnn_type == "lstm":
            self.rnn = torch.nn.LSTM(input_size=dim, hidden_size=dim, num_layers=num_layers)
        elif rnn_type == "gru":
            self.rnn = torch.nn.GRU(input_size=dim, hidden_size=dim, num_layers=num_layers)
        else:
            raise ValueError(f"Unknown RNN type {rnn_type}")

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'C', 'T'), VoidType()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'C', 'T'), VoidType()),
        }

    @typecheck()
    def forward(self, inputs, input_len):
        inputs = rearrange(inputs, "B C T -> B T C")

        packed_inputs = nn.utils.rnn.pack_padded_sequence(
            inputs, lengths=input_len.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed_inputs)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        if self.use_skip:
            out = out + inputs

        out = rearrange(out, "B T C -> B C T")
        return out


class SEANetEncoder(NeuralModule):
    def __init__(
        self,
        down_sample_rates: Iterable[int] = (2, 4, 5, 8),
        base_channels: int = 32,
        in_kernel_size: int = 7,
        out_kernel_size: int = 7,
        encoded_dim: int = 128,
        activation: str = "elu",
        rnn_layers: int = 2,
        rnn_type: str = "lstm",
        rnn_skip: bool = True,
    ):
        assert in_kernel_size > 0
        assert out_kernel_size > 0

        super().__init__()

        self.down_sample_rates = down_sample_rates
        self.pre_conv = Conv1dNorm(in_channels=1, out_channels=base_channels, kernel_size=in_kernel_size)

        in_channels = base_channels
        self.res_blocks = nn.ModuleList([])
        self.down_sample_conv_layers = nn.ModuleList([])
        self.activations = nn.ModuleList([])
        for i, down_sample_rate in enumerate(self.down_sample_rates):
            res_block = SEANetResnetBlock(channels=in_channels)
            self.res_blocks.append(res_block)
            self.activations.append(CodecActivation(activation=activation, channels=in_channels))

            out_channels = 2 * in_channels
            kernel_size = 2 * down_sample_rate
            down_sample_conv = Conv1dNorm(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=down_sample_rate,
                padding=get_down_sample_padding(kernel_size, down_sample_rate),
            )
            in_channels = out_channels
            self.down_sample_conv_layers.append(down_sample_conv)

        self.post_activation = CodecActivation(activation=activation, channels=in_channels)
        self.rnn = SEANetRNN(dim=in_channels, num_layers=rnn_layers, rnn_type=rnn_type, use_skip=rnn_skip)
        self.post_conv = Conv1dNorm(in_channels=in_channels, out_channels=encoded_dim, kernel_size=out_kernel_size)

    @property
    def input_types(self):
        return {
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "encoded": NeuralType(('B', 'D', 'T_encoded'), EncodedRepresentation()),
            "encoded_len": NeuralType(tuple('B'), LengthsType()),
        }

    def remove_weight_norm(self):
        self.pre_conv.remove_weight_norm()
        self.post_conv.remove_weight_norm()
        for res_block in self.res_blocks:
            res_block.remove_weight_norm()
        for down_sample_conv in self.down_sample_conv_layers:
            down_sample_conv.remove_weight_norm()

    @typecheck()
    def forward(self, audio, audio_len):
        encoded_len = audio_len
        audio = rearrange(audio, "B T -> B 1 T")
        # [B, C, T_audio]
        out = self.pre_conv(inputs=audio, input_len=encoded_len)
        for res_block, down_sample_conv, down_sample_rate, activation in zip(
            self.res_blocks, self.down_sample_conv_layers, self.down_sample_rates, self.activations
        ):
            # [B, C, T]
            out = res_block(inputs=out, input_len=encoded_len)
            out = activation(out)

            encoded_len = encoded_len // down_sample_rate
            # [B, 2 * C, T / down_sample_rate]
            out = down_sample_conv(inputs=out, input_len=encoded_len)

        out = self.rnn(inputs=out, input_len=encoded_len)
        out = self.post_activation(out)
        # [B, encoded_dim, T_encoded]
        encoded = self.post_conv(inputs=out, input_len=encoded_len)
        return encoded, encoded_len


class SEANetDecoder(NeuralModule):
    def __init__(
        self,
        up_sample_rates: Iterable[int] = (8, 5, 4, 2),
        base_channels: int = 512,
        in_kernel_size: int = 7,
        out_kernel_size: int = 3,
        encoded_dim: int = 128,
        activation: str = "elu",
        rnn_layers: int = 2,
        rnn_type: str = "lstm",
        rnn_skip: bool = True,
    ):
        assert in_kernel_size > 0
        assert out_kernel_size > 0

        super().__init__()

        self.up_sample_rates = up_sample_rates
        self.pre_conv = Conv1dNorm(in_channels=encoded_dim, out_channels=base_channels, kernel_size=in_kernel_size)
        self.rnn = SEANetRNN(dim=base_channels, num_layers=rnn_layers, rnn_type=rnn_type, use_skip=rnn_skip)

        in_channels = base_channels
        self.res_blocks = nn.ModuleList([])
        self.up_sample_conv_layers = nn.ModuleList([])
        self.activations = nn.ModuleList([])
        for i, up_sample_rate in enumerate(self.up_sample_rates):
            self.activations.append(CodecActivation(activation=activation, channels=in_channels))
            out_channels = in_channels // 2
            kernel_size = 2 * up_sample_rate
            up_sample_conv = ConvTranspose1dNorm(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=up_sample_rate,
            )
            in_channels = out_channels
            self.up_sample_conv_layers.append(up_sample_conv)

            res_block = SEANetResnetBlock(channels=in_channels)
            self.res_blocks.append(res_block)

        self.post_activation = CodecActivation(activation=activation, channels=in_channels)
        self.post_conv = Conv1dNorm(in_channels=in_channels, out_channels=1, kernel_size=out_kernel_size)
        self.out_activation = nn.Tanh()

    @property
    def input_types(self):
        return {
            "inputs": [NeuralType(('B', 'D', 'T_encoded'), EncodedRepresentation())],
            "input_len": [NeuralType(tuple('B'), LengthsType())],
        }

    @property
    def output_types(self):
        return {
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    def remove_weight_norm(self):
        self.pre_conv.remove_weight_norm()
        for up_sample_conv in self.up_sample_conv_layers:
            up_sample_conv.remove_weight_norm()
        for res_block in self.res_blocks:
            res_block.remove_weight_norm()

    @typecheck()
    def forward(self, inputs, input_len):
        audio_len = input_len
        # [B, C, T_encoded]
        out = self.pre_conv(inputs=inputs, input_len=audio_len)
        out = self.rnn(inputs=out, input_len=audio_len)
        for res_block, up_sample_conv, up_sample_rate, activation in zip(
            self.res_blocks, self.up_sample_conv_layers, self.up_sample_rates, self.activations
        ):
            audio_len = audio_len * up_sample_rate
            out = activation(out)
            # [B, C / 2, T * up_sample_rate]
            out = up_sample_conv(inputs=out, input_len=audio_len)
            out = res_block(inputs=out, input_len=audio_len)

        out = self.post_activation(out)
        # [B, 1, T_audio]
        out = self.post_conv(inputs=out, input_len=audio_len)
        audio = self.out_activation(out)
        audio = rearrange(audio, "B 1 T -> B T")
        return audio, audio_len


class DiscriminatorSTFT(NeuralModule):
    def __init__(self, resolution, lrelu_slope=0.1):
        super().__init__()

        self.n_fft, self.hop_length, self.win_length = resolution
        self.register_buffer("window", torch.hann_window(self.win_length, periodic=False))
        self.activation = nn.LeakyReLU(lrelu_slope)

        self.conv_layers = nn.ModuleList(
            [
                Conv2dNorm(2, 32, kernel_size=(3, 9)),
                Conv2dNorm(32, 32, kernel_size=(3, 9), dilation=(1, 1), stride=(1, 2)),
                Conv2dNorm(32, 32, kernel_size=(3, 9), dilation=(2, 1), stride=(1, 2)),
                Conv2dNorm(32, 32, kernel_size=(3, 9), dilation=(4, 1), stride=(1, 2)),
                Conv2dNorm(32, 32, kernel_size=(3, 3)),
            ]
        )
        self.conv_post = Conv2dNorm(32, 1, kernel_size=(3, 3))

    def stft(self, audio):
        # [B, fft, T_spec]
        out = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            normalized=True,
            center=True,
            return_complex=True,
        )
        out = rearrange(out, "B fft T -> B 1 T fft")
        # [batch, 2, T_spec, fft]
        out = torch.cat([out.real, out.imag], dim=1)
        return out

    @property
    def input_types(self):
        return {
            "audio": NeuralType(('B', 'T_audio'), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "scores": NeuralType(('B', 'C', 'T_spec'), VoidType()),
            "fmap": [NeuralType(('B', 'D', 'T_spec', 'C'), VoidType())],
        }

    @typecheck()
    def forward(self, audio):
        fmap = []

        # [batch, 2, T_spec, fft]
        out = self.stft(audio)
        for conv in self.conv_layers:
            # [batch, filters, T_spec, fft // 2**i]
            out = conv(inputs=out)
            out = self.activation(out)
            fmap.append(out)
        # [batch, 1, T_spec, fft // 8]
        scores = self.conv_post(inputs=out)
        fmap.append(scores)
        scores = rearrange(scores, "B 1 T C -> B C T")

        return scores, fmap


class MultiResolutionDiscriminatorSTFT(NeuralModule):
    def __init__(self, resolutions):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorSTFT(res) for res in resolutions])

    @property
    def input_types(self):
        return {
            "audio_real": NeuralType(('B', 'T_audio'), AudioSignal()),
            "audio_gen": NeuralType(('B', 'T_audio'), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "scores_real": [NeuralType(('B', 'C', 'T_spec'), VoidType())],
            "scores_gen": [NeuralType(('B', 'C', 'T_spec'), VoidType())],
            "fmaps_real": [[NeuralType(('B', 'D', 'T_spec', 'C'), VoidType())]],
            "fmaps_gen": [[NeuralType(('B', 'D', 'T_spec', 'C'), VoidType())]],
        }

    @typecheck()
    def forward(self, audio_real, audio_gen):
        scores_real = []
        scores_gen = []
        fmaps_real = []
        fmaps_gen = []

        for disc in self.discriminators:
            score_real, fmap_real = disc(audio=audio_real)
            scores_real.append(score_real)
            fmaps_real.append(fmap_real)

            score_gen, fmap_gen = disc(audio=audio_gen)
            scores_gen.append(score_gen)
            fmaps_gen.append(fmap_gen)

        return scores_real, scores_gen, fmaps_real, fmaps_gen


def _ema_inplace(moving_avg: Tensor, new: Tensor, decay: float) -> None:
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def _laplace_smoothing(inputs: Tensor, n_categories: int, epsilon: float = 1e-5) -> Tensor:
    input_sum = inputs.sum()
    smoothed = (inputs + epsilon) / (input_sum + n_categories * epsilon)
    return input_sum * smoothed


def _compute_distances(input1: Tensor, input2: Tensor) -> Tensor:
    """
    Compute pairwise L2 distance between two input tensors

    Args:
        input1: [B, D] first tensor.
        input2: [N, D] second tensor.

    Returns:
        [(B, D)] tensor of distances.
    """
    input2 = rearrange(input2, "N D -> D N")
    distances = input1.pow(2).sum(1, keepdim=True) - (2 * input1 @ input2) + input2.pow(2).sum(0, keepdim=True)
    return distances


def _sample_vectors(samples: Tensor, num_sample: int) -> Tensor:
    """
    Randomly sample from the input batch.

    Args:
        samples: [B, D] tensor with features to sample.
        num_sample: Number of samples to draw.
            If the value is less than or equal to B, then the samples will be unique.
            If the value is greater than B, then samples will be drawn with replacement.

    Returns:
        Tensor with num_sample values randomly sampled from the input batch.
    """
    device = samples.device
    total_samples = samples.shape[0]

    if total_samples >= num_sample:
        indices = torch.randperm(total_samples, device=device)[:num_sample]
    else:
        indices = torch.randint(low=0, high=total_samples, size=(num_sample,), device=device)

    return samples[indices]


def _k_means(samples: Tensor, num_clusters: int, num_iters: int = 10) -> Tuple[Tensor, Tensor]:
    """
    K-means clustering algorithm.

    Args:
        samples: [B, D] tensor with features to cluster
        num_clusters: K, the number of clusters.
        num_iters: Number of iterations of K-means to run.

    Returns:
        [K, D] cluster means and [K] bins counting how many input samples belong to each cluster
    """
    assert num_iters > 0

    input_dim = samples.shape[1]
    # [K, D]
    means = _sample_vectors(samples=samples, num_sample=num_clusters)

    for _ in range(num_iters):
        # [B, K]
        dists = _compute_distances(samples, means)

        # [N]
        buckets = dists.min(dim=1).indices
        buckets_repeated = repeat(buckets, "B -> B D", D=input_dim)
        # [K]
        bin_counts = torch.bincount(buckets, minlength=num_clusters)
        bin_counts_expanded = rearrange(bin_counts, "K -> K ()")

        # [K, D]
        new_means = buckets.new_zeros(num_clusters, input_dim, dtype=samples.dtype)
        new_means.scatter_add_(dim=0, index=buckets_repeated, src=samples)
        new_means = new_means / torch.clamp(bin_counts_expanded, min=1)
        means = torch.where(bin_counts_expanded == 0, means, new_means)

    return means, bin_counts


def _mask_3d(tensor: Tensor, lengths: Tensor):
    """
    Mask 3d tensor with time on 1st axis.

    Args:
        tensor: tensor of shape (B, T, D)
        lengths: LongTensor of shape (B,)
    Returns:
        Masked Tensor (B, T, D)
    """
    batch_size, max_lengths, _ = tensor.shape
    mask = torch.ones(batch_size, max_lengths, 1).cumsum(dim=1).type_as(lengths)
    mask = mask <= rearrange(lengths, "b -> b 1 1")
    return tensor * mask


@experimental
class EuclideanCodebook(NeuralModule):
    """
    Codebook with Euclidean distance.

    Args:
        codebook_size: Number of codes to use.
        codebook_dim: Dimension of each code.
        decay: Decay for exponential moving average over the codebooks.
        threshold_ema_dead_code: Threshold for dead code expiration.
            During every iteration, replace codes with exponential moving average cluster size less than threshold
            with randomly selected values from the current batch.
        kmeans_iters: Optional int, if provided codes will be initialized from the centroids learned from
            kmeans_iters iterations of k-means clustering on the first training batch.
    """

    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,
        decay: float = 0.99,
        threshold_ema_dead_code: Optional[float] = 2.0,
        kmeans_iters: Optional[int] = 50,
    ):
        super().__init__()
        self.decay = decay

        if kmeans_iters:
            codes = nn.init.kaiming_uniform_(torch.empty(codebook_size, codebook_dim))
        else:
            codes = torch.zeros(codebook_size, codebook_dim)

        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.register_buffer("initialized", Tensor([not kmeans_iters]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("codes", codes)
        self.register_buffer("codes_avg", codes.clone())

    @torch.jit.ignore
    def _init_codes(self, data):
        if self.initialized:
            return

        codes, cluster_size = _k_means(samples=data, num_clusters=self.codebook_size, num_iters=self.kmeans_iters)
        self.codes.data.copy_(codes)
        self.codes_avg.data.copy_(codes.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.initialized.data.copy_(Tensor([True]))
        broadcast_tensors(self.buffers())

    def _expire_codes(self, inputs: Tensor) -> None:
        if not self.threshold_ema_dead_code:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return

        samples = _sample_vectors(samples=inputs, num_sample=self.codebook_size)
        expired_codes = rearrange(expired_codes, "K -> K ()")
        modified_codes = torch.where(expired_codes, samples, self.codes)
        self.codes.data.copy_(modified_codes)

        broadcast_tensors(self.buffers())

    def _update_codes(self, inputs: Tensor, indices: Tensor) -> None:
        code_onehot = F.one_hot(indices, self.codebook_size).type(inputs.dtype)
        code_onehot = rearrange(code_onehot, "B N -> N B")
        # [N]
        code_counts = code_onehot.sum(1)
        _ema_inplace(moving_avg=self.cluster_size, new=code_counts, decay=self.decay)
        # [N, D]
        code_sum = code_onehot @ inputs
        _ema_inplace(moving_avg=self.codes_avg, new=code_sum, decay=self.decay)

        cluster_size_smoothed = _laplace_smoothing(self.cluster_size, n_categories=self.codebook_size)
        cluster_size_smoothed = rearrange(cluster_size_smoothed, "N -> N ()")
        codes_normalized = self.codes_avg / cluster_size_smoothed
        self.codes.data.copy_(codes_normalized)

    def _quantize(self, inputs: Tensor) -> Tensor:
        # [B, N]
        dist = _compute_distances(inputs, self.codes)
        # [B]
        indices = dist.min(dim=1).indices
        return indices

    def _dequantize(self, indices: Tensor) -> Tensor:
        # [B, D]
        dequantized = F.embedding(indices, self.codes)
        return dequantized

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "dequantized": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
            "indices": NeuralType(('B', 'T'), Index()),
        }

    @typecheck()
    def forward(self, inputs, input_len):
        input_flat = rearrange(inputs, "B T D -> (B T) D")
        self._init_codes(input_flat)
        # [(B T)]
        indices_flat = self._quantize(inputs=input_flat)
        # [B, T]
        indices = indices_flat.view(*inputs.shape[:-1])
        # [B, T, D]
        dequantized = self._dequantize(indices=indices)

        if self.training:
            # We do expiry of codes here because buffers are in sync and all the workers will make the same decision.
            self._expire_codes(inputs=input_flat)
            self._update_codes(inputs=input_flat, indices=indices_flat)

        dequantized = _mask_3d(dequantized, input_len)
        indices = mask_sequence_tensor(indices, input_len)
        return dequantized, indices

    @typecheck(
        input_types={
            "inputs": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={"indices": NeuralType(('B', 'T'), Index())},
    )
    def encode(self, inputs, input_len):
        input_flat = rearrange(inputs, "B T D -> (B T) D")
        # [(B T)]
        indices_flat = self._quantize(inputs=input_flat)
        # [B, T]
        indices = indices_flat.view(*inputs.shape[:-1])
        indices = mask_sequence_tensor(indices, input_len)
        return indices

    @typecheck(
        input_types={
            "indices": NeuralType(('B', 'T'), Index()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={"dequantized": NeuralType(('B', 'T', 'D'), EncodedRepresentation())},
    )
    def decode(self, indices, input_len):
        # [B, T, D]
        dequantized = self._dequantize(indices=indices)
        dequantized = _mask_3d(dequantized, input_len)
        return dequantized


class ResidualVectorQuantizer(VectorQuantizerBase):
    """
    Residual vector quantization (RVQ) algorithm as described in https://arxiv.org/pdf/2107.03312.pdf.

    Args:
        num_codebooks: Number of codebooks to use.
        codebook_size: Number of codes to use for each codebook.
        codebook_dim: Dimension of each code.
        decay: Decay for exponential moving average over the codebooks.
        threshold_ema_dead_code: Threshold for dead code expiration.
            During every iteration, replace codes with exponential moving average cluster size less than threshold
            with randomly selected values from the current batch.
        kmeans_iters: Optional int, if provided codes will be initialized from the centroids learned from
            kmeans_iters iterations of k-means clustering on the first training batch.
    """

    def __init__(
        self,
        num_codebooks: int,
        codebook_size: int = 1024,
        codebook_dim: int = 128,
        decay: float = 0.99,
        threshold_ema_dead_code: Optional[float] = 2.0,
        kmeans_iters: Optional[int] = 50,
    ):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.commit_loss_fn = MaskedMSELoss()

        self.codebooks = nn.ModuleList(
            [
                EuclideanCodebook(
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    decay=decay,
                    threshold_ema_dead_code=threshold_ema_dead_code,
                    kmeans_iters=kmeans_iters,
                )
                for _ in range(num_codebooks)
            ]
        )

    # Override output types, since this quantizer returns commit_loss
    @property
    def output_types(self):
        return {
            "dequantized": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "indices": NeuralType(('D', 'B', 'T'), Index()),
            "commit_loss": NeuralType((), LossType()),
        }

    @typecheck()
    def forward(self, inputs: Tensor, input_len: Tensor) -> Tuple[Tensor, Tensor, float]:
        commit_loss = 0.0
        residual = rearrange(inputs, "B D T -> B T D")

        index_list = []
        dequantized = torch.zeros_like(residual)
        for codebook in self.codebooks:
            dequantized_i, indices_i = codebook(inputs=residual, input_len=input_len)

            if self.training:
                dequantized_i_const = dequantized_i.detach()

                commit_loss_i = self.commit_loss_fn(
                    predicted=rearrange(residual, "B T D -> B D T"),
                    target=rearrange(dequantized_i_const, "B T D -> B D T"),
                    target_len=input_len,
                )
                commit_loss = commit_loss + commit_loss_i

                residual = residual - dequantized_i_const
                dequantized_i = residual + (dequantized_i - residual).detach()
            else:
                residual = residual - dequantized_i

            dequantized = dequantized + dequantized_i
            index_list.append(indices_i)

        # [N, B, T], first dimension is number of codebooks
        indices = torch.stack(index_list)
        dequantized = rearrange(dequantized, "B T D -> B D T")
        return dequantized, indices, commit_loss

    @typecheck(
        input_types={
            "inputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={"indices": NeuralType(('D', 'B', 'T'), Index())},
    )
    def encode(self, inputs: Tensor, input_len: Tensor) -> Tensor:
        residual = rearrange(inputs, "B D T -> B T D")
        index_list = []
        for codebook in self.codebooks:
            # [B, T]
            indices_i = codebook.encode(inputs=residual, input_len=input_len)
            # [B, D, T]
            dequantized_i = codebook.decode(indices=indices_i, input_len=input_len)
            residual = residual - dequantized_i
            index_list.append(indices_i)
        # [N, B, T]
        indices = torch.stack(index_list)
        return indices

    @typecheck(
        input_types={
            "indices": NeuralType(('D', 'B', 'T'), Index()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={
            "dequantized": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
        },
    )
    def decode(self, indices: Tensor, input_len: Tensor) -> Tensor:
        # [B, T, D]
        dequantized = torch.zeros([indices.shape[1], indices.shape[2], self.codebook_dim], device=indices.device)
        for codebook_indices, codebook in zip(indices, self.codebooks):
            dequantized_i = codebook.decode(indices=codebook_indices, input_len=input_len)
            dequantized = dequantized + dequantized_i
        dequantized = rearrange(dequantized, "B T D -> B D T")
        return dequantized


class GroupResidualVectorQuantizer(VectorQuantizerBase):
    """Split the input vector into groups and apply RVQ on each group separately.

    Args:
        num_codebooks: total number of codebooks
        num_groups: number of groups to split the input into, each group will be quantized separately using num_codebooks//num_groups codebooks
        codebook_dim: embedding dimension, will be split into num_groups
        **kwargs: parameters of ResidualVectorQuantizer

    References:
        Yang et al, HiFi-Codec: Group-residual Vector quantization for High Fidelity Audio Codec, 2023 (http://arxiv.org/abs/2305.02765).
    """

    def __init__(self, num_codebooks: int, num_groups: int, codebook_dim: int, **kwargs):
        super().__init__()

        self.num_codebooks = num_codebooks
        self.num_groups = num_groups
        self.codebook_dim = codebook_dim

        # Initialize RVQ for each group
        self.rvqs = torch.nn.ModuleList(
            [
                ResidualVectorQuantizer(
                    num_codebooks=self.num_codebooks_per_group, codebook_dim=self.codebook_dim_per_group, **kwargs
                )
                for _ in range(self.num_groups)
            ]
        )

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tnum_codebooks:           %d', self.num_codebooks)
        logging.debug('\tnum_groups:              %d', self.num_groups)
        logging.debug('\tcodebook_dim:            %d', self.codebook_dim)
        logging.debug('\tnum_codebooks_per_group: %d', self.num_codebooks_per_group)
        logging.debug('\tcodebook_dim_per_group:  %d', self.codebook_dim_per_group)

    @property
    def num_codebooks_per_group(self):
        """Number of codebooks for each group."""
        if self.num_codebooks % self.num_groups != 0:
            raise ValueError(
                f'num_codebooks ({self.num_codebooks}) must be divisible by num_groups ({self.num_groups})'
            )

        return self.num_codebooks // self.num_groups

    @property
    def codebook_dim_per_group(self):
        """Input vector dimension for each group."""
        if self.codebook_dim % self.num_groups != 0:
            raise ValueError(f'codebook_dim ({self.codebook_dim}) must be divisible by num_groups ({self.num_groups})')

        return self.codebook_dim // self.num_groups

    # Override output types, since this quantizer returns commit_loss
    @property
    def output_types(self):
        return {
            "dequantized": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "indices": NeuralType(('D', 'B', 'T'), Index()),
            "commit_loss": NeuralType((), LossType()),
        }

    @typecheck()
    def forward(self, inputs, input_len):
        """Quantize each group separately, then concatenate the results."""
        inputs_grouped = inputs.chunk(self.num_groups, dim=1)

        dequantized, indices = [], []
        commit_loss = 0

        for in_group, rvq_group in zip(inputs_grouped, self.rvqs):
            dequantized_group, indices_group, commit_loss_group = rvq_group(inputs=in_group, input_len=input_len)
            dequantized.append(dequantized_group)
            indices.append(indices_group)
            commit_loss += commit_loss_group

        # concatenate along the feature dimension
        dequantized = torch.cat(dequantized, dim=1)

        # concatente along the codebook dimension
        indices = torch.cat(indices, dim=0)

        return dequantized, indices, commit_loss

    @typecheck(
        input_types={
            "inputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={"indices": NeuralType(('D', 'B', 'T'), Index())},
    )
    def encode(self, inputs: Tensor, input_len: Tensor) -> Tensor:
        """Input is split into groups, each group is encoded separately, then the results are concatenated."""
        inputs_grouped = inputs.chunk(self.num_groups, dim=1)
        indices = []

        for in_group, rvq_group in zip(inputs_grouped, self.rvqs):
            indices_group = rvq_group.encode(inputs=in_group, input_len=input_len)
            indices.append(indices_group)

        # concatenate along the codebook dimension
        indices = torch.cat(indices, dim=0)

        return indices

    @typecheck(
        input_types={
            "indices": NeuralType(('D', 'B', 'T'), Index()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={
            "dequantized": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
        },
    )
    def decode(self, indices: Tensor, input_len: Tensor) -> Tensor:
        """Input indices are split into groups, each group is decoded separately, then the results are concatenated."""
        indices_grouped = indices.chunk(self.num_groups, dim=0)
        dequantized = []

        for indices_group, rvq_group in zip(indices_grouped, self.rvqs):
            dequantized_group = rvq_group.decode(indices=indices_group, input_len=input_len)
            dequantized.append(dequantized_group)

        # concatenate along the feature dimension
        dequantized = torch.cat(dequantized, dim=1)

        return dequantized
