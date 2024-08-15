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

import logging
import math
import os
from typing import Iterable, List

logger = logging.getLogger(__name__)

import einops
import torch
import torch.nn as nn

__all__ = ['if_exist', '_compute_softmax', 'flatten']

activation_registry = {
    "identity": nn.Identity,
    "hardtanh": nn.Hardtanh,
    "relu": nn.ReLU,
    "selu": nn.SELU,
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}


def if_exist(outfold: str, files: List[str]):
    """
    Returns true if all given files exist in the given folder
    Args:
        outfold: folder path
        files: list of file names relative to outfold
    """
    if not os.path.exists(outfold):
        return False
    for file in files:
        if not os.path.exists(f'{outfold}/{file}'):
            return False
    return True


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def flatten_iterable(iter: Iterable) -> Iterable:
    """Flatten an iterable which contains values or
    iterables with values.

    Args:
        iter: iterable containing values at the deepest level.

    Returns:
        A flat iterable containing values.
    """
    for it in iter:
        if isinstance(it, str) or not isinstance(it, Iterable):
            yield it
        else:
            yield from flatten_iterable(it)


def flatten(list_in: List) -> List:
    """Flatten a list of (nested lists of) values into a flat list.

    Args:
        list_in: list of values, possibly nested

    Returns:
        A flat list of values.
    """
    return list(flatten_iterable(list_in))


def extend_instance(obj, mixin):
    """Apply mixins to a class instance after creation"""
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(
        base_cls_name, (mixin, base_cls), {}
    )  # mixin needs to go first for our forward() logic to work


def apply_rope_scaling(freqs):
    # Apply scaling for RoPE frequencies
    logger.info("apply rope scaling ...")
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def mask_sequence_tensor(tensor: torch.Tensor, lengths: torch.Tensor):
    """
    For tensors containing sequences, zero out out-of-bound elements given lengths of every element in the batch.

    tensor: tensor of shape (B, L), (B, D, L) or (B, D1, D2, L),
    lengths: LongTensor of shape (B,)
    """
    batch_size, *_, max_lengths = tensor.shape

    if len(tensor.shape) == 2:
        mask = torch.ones(batch_size, max_lengths).cumsum(dim=-1).type_as(lengths)
        mask = mask <= einops.rearrange(lengths, 'B -> B 1')
    elif len(tensor.shape) == 3:
        mask = torch.ones(batch_size, 1, max_lengths).cumsum(dim=-1).type_as(lengths)
        mask = mask <= einops.rearrange(lengths, 'B -> B 1 1')
    elif len(tensor.shape) == 4:
        mask = torch.ones(batch_size, 1, 1, max_lengths).cumsum(dim=-1).type_as(lengths)
        mask = mask <= einops.rearrange(lengths, 'B -> B 1 1 1')
    else:
        raise ValueError('Can only mask tensors of shape B x L, B x D x L and B x D1 x D2 x L')

    return tensor * mask
