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
import torch
import torch.nn.functional as F

_compiled_compute_cross_entropy = None


def compute_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index=-100,
):
    """
    Computes the cross-entropy loss between logits and targets.

    Args:
        logits (torch.Tensor): Model predictions of shape (sequence_length, num_classes).
        targets (torch.Tensor): Ground-truth labels of shape (sequence_length,).
        ignore_index (int, optional): Target value that is ignored when computing the loss.
            Defaults to -100.

    Returns:
        torch.Tensor: The sum of cross-entropy losses over the sequence.
    """
    return F.cross_entropy(logits.float(), targets, ignore_index=ignore_index, reduction="sum")


def chunked_cross_entropy(logits, targets, mask=None, chunk_len=32, compile=True, ignore_index=-100):
    """
    Computes cross-entropy loss in chunks to handle long sequences more efficiently.

    Args:
        logits (torch.Tensor): Model output logits of shape (sequence_length, num_classes).
        targets (torch.Tensor): Ground-truth labels of shape (sequence_length,).
        mask (torch.Tensor, optional): Boolean mask indicating valid positions (1) and
            positions to ignore (0). Defaults to None.
        chunk_len (int, optional): The size of each chunk. The sequence will be split
            along the first dimension in chunks of this length. Defaults to 32.
        compile (bool, optional): If True, uses the compiled compute_cross_entropy function.
            Defaults to True.
        ignore_index (int, optional): Target value that is ignored when computing the loss.
            Defaults to -100.

    Returns:
        torch.Tensor: The average cross-entropy loss across the valid tokens in the sequence.
    """
    # copied the following block from masked_ce
    # this may happen with CPUOffloadPolicy
    if targets.device != logits.device:
        targets = targets.to(logits.device)
    if mask is not None:
        with torch.no_grad():
            if mask.device != targets.device:
                mask = mask.to(targets.device)
            targets.masked_fill_(mask.view(-1) == 0, ignore_index)
            del mask

    # maybe refactor if this is moved to a class?
    global _compiled_compute_cross_entropy
    if _compiled_compute_cross_entropy is None:
        _compiled_compute_cross_entropy = torch.compile(compute_cross_entropy, dynamic=True)

    seq_len = logits.shape[0]
    num_chunks = (seq_len + chunk_len - 1) // chunk_len
    loss = 0.0
    for logits_chunk, targets_chunk in zip(logits.chunk(num_chunks, dim=0), targets.chunk(num_chunks, dim=0)):
        loss += _compiled_compute_cross_entropy(logits_chunk, targets_chunk, ignore_index)
    # normalize
    num_tokens = (targets != ignore_index).sum().detach()
    return loss / num_tokens
