# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import bisect
from typing import List

import torch
from megatron.core.packed_seq_params import PackedSeqParams


# pylint:disable=line-too-long
# Based on https://github.com/hiyouga/LLaMA-Factory/blob/641d0dab08d96a93c34657742213d8994d9ed476/src/llamafactory/data/processors/processor_utils.py#L19
# Copyright (c) 2024 LLaMA-Factory. Apache license 2.0.
def search_for_fit(numbers: List[int], capacity: int) -> int:
    """Finds the index of largest number that fits into the knapsack with the given capacity."""
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)


# pylint: disable=line-too-long
# Based on https://github.com/hiyouga/LLaMA-Factory/blob/641d0dab08d96a93c34657742213d8994d9ed476/src/llamafactory/data/processors/processor_utils.py#L27
# Copyright (c) 2024 LLaMA-Factory. Apache license 2.0.
def greedy_knapsack(item_sizes: List[int], samples: List, max_capacity: int) -> List:
    """Greedy algorithm with binary search for the knapsack problem.

    Pack as many samples as possible given a maximum capacity and capacities of individual samples.
    Used if sequence packing is enabled.
    """
    assert len(item_sizes) == len(samples), "sample lengths and samples must have the same length."

    knapsacks = []

    if len(item_sizes) == 0:
        return knapsacks

    # Sort sample lengths and samples together.
    sorted_item_sizes, sorted_samples = zip(*sorted(zip(item_sizes, samples), key=lambda x: x[0]))
    sorted_item_sizes = list(sorted_item_sizes)
    sorted_samples = list(sorted_samples)

    # Check if all samples fit in the knapsack capacity.
    if sorted_item_sizes[-1] > max_capacity:
        raise ValueError(
            f"knapsack: A sample is larger {sorted_item_sizes[-1]} than the max_sequence_length {max_capacity}."
        )

    while sorted_item_sizes:
        current_knapsack = []
        remaining_capacity = max_capacity

        while True:
            idx = search_for_fit(sorted_item_sizes, remaining_capacity)
            if idx == -1:
                break  # Can't fit more samples.

            remaining_capacity -= sorted_item_sizes[idx]

            sorted_item_sizes.pop(idx)
            sample = sorted_samples.pop(idx)
            current_knapsack.append(sample)

        knapsacks.append(current_knapsack)

    return knapsacks


def predict_seq_len(instance_tokens: torch.Tensor, num_image_embeddings_per_tile: int, media_token_index: int) -> int:
    """
    Predict the effective sequence length, accounting for media embeddings.

    Args:
        instance_tokens (torch.Tensor): Token tensor for a single instance.
        num_image_embeddings_per_tile (int): Number of image embeddings per tile.
        media_token_index (int): Token ID representing media.

    Returns:
        int: Effective sequence length.
    """
    num_images = torch.sum(instance_tokens == media_token_index).item()
    seqlen = len(instance_tokens) + (num_image_embeddings_per_tile - 1) * num_images
    return seqlen


def predict_seq_len_with_padding(instance_tokens: torch.Tensor, pad_to_multiple_of: int = 64) -> int:
    """Get seqlen with padding"""
    seqlen = len(instance_tokens)
    seqlen_padded = (seqlen + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of
    return seqlen_padded


def convert_to_packed(
    tokens: List[torch.Tensor],
    labels: List[torch.Tensor],
    seqlens: List[int],
):
    """
    Convert tokens, labels, and associated inputs into a packed version.

    Args:
        tokens (list[torch.Tensor]): List of token tensors for each instance.
        labels (list[torch.Tensor]): List of label tensors for each instance.
        seqlens (list[int]): List of sequence lengths for each instance before padding.
    """
    packed_tokens = []
    packed_labels = []
    packed_position_ids = []
    seqlens_padded = []
    cu_seqlens = [0]
    cu_seqlens_padded = [0]

    for instance_tokens, instance_labels, seqlen in zip(tokens, labels, seqlens):

        packed_tokens.append(instance_tokens)
        packed_labels.append(instance_labels)
        packed_position_ids.append(torch.arange(len(instance_tokens), dtype=torch.int, device=instance_tokens.device))
        seqlen_padded = len(instance_tokens)
        seqlens_padded.append(seqlen_padded)
        cu_seqlens.append(cu_seqlens[-1] + seqlen)
        cu_seqlens_padded.append(cu_seqlens_padded[-1] + seqlen_padded)

    packed_tokens = torch.cat(packed_tokens, dim=0).unsqueeze(0)
    packed_labels = torch.cat(packed_labels, dim=0).unsqueeze(0)
    packed_position_ids = torch.cat(packed_position_ids, dim=0).unsqueeze(0)
    packed_loss_mask = torch.ones_like(packed_labels, dtype=torch.float, device=packed_labels.device)
    packed_loss_mask[packed_labels < 0] = 0.0

    cu_seqlens = torch.IntTensor(cu_seqlens)
    cu_seqlens_padded = torch.IntTensor(cu_seqlens_padded)

    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=int(max(seqlens_padded)),
        max_seqlen_kv=int(max(seqlens_padded)),
        qkv_format='thd',
    )

    return packed_tokens, packed_labels, packed_position_ids, packed_loss_mask, packed_seq_params


def _find_pattern_indices(template, pattern, search_start_index=0, allow_first_token_mismatch=False):
    template_len = len(template)
    pattern_len = len(pattern)
    for i in range(search_start_index, template_len - pattern_len + 1):
        match = template[i : i + pattern_len] == pattern
        if torch.all(match) or (allow_first_token_mismatch and torch.all(match[1:])):
            return i, i + pattern_len
    return -1, -1
