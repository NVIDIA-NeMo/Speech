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
import pytest
import torch
import torch.nn.functional as F

from nemo.automodel.loss.chunked_ce import chunked_cross_entropy, compute_cross_entropy


def test_compute_cross_entropy_basic():
    """
    Tests compute_cross_entropy with a small set of logits and targets.
    Verifies results match PyTorch's built-in cross_entropy.
    """
    # Create sample logits and targets
    logits = torch.tensor([[2.0, 0.5, 0.3], [1.0, 2.0, 0.1]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)

    # Expected cross_entropy from PyTorch (sum reduction for direct comparison)
    expected_loss = F.cross_entropy(logits, targets, reduction="sum")

    # Compare function output
    actual_loss = compute_cross_entropy(logits, targets)
    assert torch.allclose(
        actual_loss, expected_loss, atol=1e-6
    ), f"Expected loss {expected_loss.item()}, but got {actual_loss.item()}."


def test_compute_cross_entropy_ignore_index():
    """
    Tests compute_cross_entropy with ignore_index to ensure ignored targets
    don't contribute to the loss.
    """
    # Create sample logits and targets with ignore_index
    logits = torch.tensor([[0.0, 0.0], [2.0, 3.0], [1.0, 1.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1, -100], dtype=torch.long)  # -100 will be ignored

    # Compute expected loss with PyTorch
    expected_loss = F.cross_entropy(logits, targets, reduction="sum", ignore_index=-100)

    # Compare function output
    actual_loss = compute_cross_entropy(logits, targets, ignore_index=-100)
    assert torch.allclose(
        actual_loss, expected_loss, atol=1e-6
    ), f"Expected loss {expected_loss.item()}, but got {actual_loss.item()}."


def test_chunked_cross_entropy_matches_compute_cross_entropy():
    """
    Tests that chunked_cross_entropy produces the same result as compute_cross_entropy
    when the entire sequence is processed in one chunk.
    """
    # Create random test data
    batch_size = 4
    seq_len = 16
    num_classes = 8

    logits = torch.randn(seq_len, num_classes)
    targets = torch.randint(0, num_classes, (seq_len,))

    # Loss from normal compute_cross_entropy
    loss_ref = compute_cross_entropy(logits, targets) / (targets != -100).sum().detach()

    # Loss from chunked_cross_entropy when chunk_len = seq_len (effectively one chunk)
    from math import ceil

    chunk_len = seq_len  # so there's only one chunk
    loss_chunked = chunked_cross_entropy(logits, targets, chunk_len=chunk_len)

    assert torch.allclose(
        loss_chunked, loss_ref, atol=1e-6
    ), f"Expected chunked loss {loss_ref.item()}, but got {loss_chunked.item()}."


def test_chunked_cross_entropy_ignore_index_and_mask():
    """
    Tests that chunked_cross_entropy properly ignores indices and respects masks.
    Verifies consistency with compute_cross_entropy.
    """
    seq_len = 10
    num_classes = 5
    logits = torch.randn(seq_len, num_classes)
    targets = torch.randint(0, num_classes, (seq_len,))

    # Randomly zero out entries in a mask
    mask = torch.randint(0, 2, (seq_len,))  # 0 or 1
    mask[0] = 1  # make sure at least one entry is 1
    ignore_idx = -100

    # First compute the reference loss by manually applying ignore_index
    masked_targets = targets.clone()
    masked_targets[mask == 0] = ignore_idx
    loss_ref = compute_cross_entropy(logits, masked_targets, ignore_index=ignore_idx)
    loss_ref /= (masked_targets != ignore_idx).sum().detach()

    # Now compute chunked CE with mask
    chunk_len = 3  # just an arbitrary small chunk size
    loss_chunked = chunked_cross_entropy(logits, targets, mask=mask, chunk_len=chunk_len, ignore_index=ignore_idx)

    assert torch.allclose(
        loss_chunked, loss_ref, atol=1e-6
    ), f"Expected chunked loss {loss_ref.item()}, but got {loss_chunked.item()}."
