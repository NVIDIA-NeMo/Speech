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

import math
import os
from typing import Iterable, List

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
