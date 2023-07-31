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

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

from nemo.collections.tts.losses.audio_codec_loss import MaskedMSELoss
from nemo.collections.tts.parts.utils.distributed import broadcast_tensors
from nemo.collections.tts.parts.utils.helpers import mask_sequence_tensor
from nemo.core.classes.common import typecheck
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types.elements import EncodedRepresentation, Index, LengthsType, LossType
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils.decorators import experimental


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
        threshold_ema_dead_code: Optional[int] = 2,
        kmeans_iters: Optional[int] = None,
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
        quantized = F.embedding(indices, self.codes)
        return quantized

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "quantized": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
            "indices": NeuralType(('B', 'T'), Index()),
        }

    def forward(self, inputs, input_len):
        input_flat = rearrange(inputs, "B T D -> (B T) D")
        self._init_codes(input_flat)
        # [(B T)]
        indices_flat = self._quantize(inputs=input_flat)
        # [B, T]
        indices = indices_flat.view(*inputs.shape[:-1])
        # [B, T, D]
        quantized = self._dequantize(indices=indices)

        if self.training:
            # We do expiry of codes here because buffers are in sync and all the workers will make the same decision.
            self._expire_codes(inputs=input_flat)
            self._update_codes(inputs=input_flat, indices=indices_flat)

        quantized = _mask_3d(quantized, input_len)
        indices = mask_sequence_tensor(indices, input_len)
        return quantized, indices

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
        input_types={"indices": NeuralType(('B', 'T'), Index()), "input_len": NeuralType(tuple('B'), LengthsType()),},
        output_types={"quantized": NeuralType(('B', 'T', 'D'), EncodedRepresentation())},
    )
    def decode(self, indices, input_len):
        # [B, T, D]
        quantized = self._dequantize(indices=indices)
        quantized = _mask_3d(quantized, input_len)
        return quantized


class ResidualVectorQuantizer(NeuralModule):
    """
    Residual vector quantization (RVQ) algorithm as described in https://arxiv.org/pdf/2107.03312.pdf.

    Args:
        num_codebooks: Number of codebooks to use.
        commit_loss_scale: Loss scale for codebook commit loss.
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
        commit_loss_scale: float = 1.0,
        codebook_size: int = 1024,
        codebook_dim: int = 128,
        decay: float = 0.99,
        threshold_ema_dead_code: Optional[int] = 2,
        kmeans_iters: Optional[int] = 50,
    ):
        super().__init__()
        self.codebook_dim = codebook_dim

        if commit_loss_scale:
            self.commit_loss_fn = MaskedMSELoss(loss_scale=commit_loss_scale)
        else:
            self.commit_loss_fn = None

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

    def _commit_loss(self, input, target, input_len):
        if not self.commit_loss_fn:
            return 0.0

        return self.commit_loss_fn(
            predicted=rearrange(input, "B T D -> B D T"),
            target=rearrange(target, "B T D -> B D T"),
            target_len=input_len,
        )

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "quantized": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "indices": NeuralType(('B', 'T'), Index()),
            "commit_loss": NeuralType((), LossType()),
        }

    def forward(self, inputs: Tensor, input_len: Tensor) -> Tuple[Tensor, Tensor, float]:
        commit_loss = 0.0
        residual = rearrange(inputs, "B D T -> B T D")

        index_list = []
        quantized = torch.zeros_like(residual)
        for codebook in self.codebooks:
            quantized_i, indices_i = codebook(inputs=residual, input_len=input_len)

            if self.training:
                quantized_i = residual + (quantized_i - residual).detach()
                quantized_i_const = quantized_i.detach()
                commit_loss_i = self._commit_loss(input=residual, target=quantized_i_const, input_len=input_len)
                commit_loss = commit_loss + commit_loss_i

                residual = residual - quantized_i_const

            else:
                residual = residual - quantized_i

            quantized = quantized + quantized_i
            index_list.append(indices_i)

        # [N, B, T]
        indices = torch.stack(index_list)
        quantized = rearrange(quantized, "B T D -> B D T")
        return quantized, indices, commit_loss

    @typecheck(
        input_types={
            "inputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={"indices": NeuralType(('N', 'B', 'T'), Index())},
    )
    def encode(self, inputs: Tensor, input_len: Tensor) -> Tensor:
        residual = rearrange(inputs, "B D T -> B T D")
        index_list = []
        for codebook in self.codebooks:
            # [B, T]
            indices_i = codebook.encode(inputs=residual, input_len=input_len)
            # [B, D, T]
            quantized_i = codebook.decode(indices=indices_i, input_len=input_len)
            residual = residual - quantized_i
            index_list.append(indices_i)
        # [N, B, T]
        indices = torch.stack(index_list)
        return indices

    @typecheck(
        input_types={
            "indices": NeuralType(('N', 'B', 'T'), Index()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={"quantized": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),},
    )
    def decode(self, indices: Tensor, input_len: Tensor) -> Tensor:
        # [B, T, D]
        quantized = torch.zeros([indices.shape[1], indices.shape[2], self.codebook_dim], device=indices.device)
        for codebook_indices, codebook in zip(indices, self.codebooks):
            quantized_i = codebook.decode(indices=codebook_indices, input_len=input_len)
            quantized = quantized + quantized_i
        quantized = rearrange(quantized, "B T D -> B D T")
        return quantized
