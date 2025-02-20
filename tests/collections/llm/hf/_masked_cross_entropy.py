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

import pytest
import torch
import torch.nn.functional as F

from nemo.collections.llm.gpt.model.hf_auto_model_for_causal_lm import masked_cross_entropy


def test_no_mask():
    # Create sample logits and targets.
    logits = torch.tensor([[0.2, 0.8], [0.5, 0.5], [0.9, 0.1]], dtype=torch.float32)
    targets = torch.tensor([1, 0, 1], dtype=torch.long)

    # Compute loss with masked_cross_entropy without providing a mask.
    loss_function = masked_cross_entropy
    loss = loss_function(logits, targets.clone())

    # Expected loss is the standard cross_entropy.
    expected_loss = F.cross_entropy(logits, targets)

    # Use allclose to check they match.
    assert torch.allclose(loss, expected_loss), "Loss without mask does not match expected cross_entropy loss."


def test_with_mask():
    # Create sample logits and targets.
    logits = torch.tensor([[0.2, 0.8], [0.5, 0.5], [0.9, 0.1]], dtype=torch.float32)
    # Original targets.
    targets = torch.tensor([1, 0, 1], dtype=torch.long)
    # Create a mask that ignores the second example.
    mask = torch.tensor([1, 0, 1], dtype=torch.uint8)

    # Make copies since the function modifies targets in-place.
    logits_copy = logits.clone()
    targets_copy = targets.clone()
    mask_copy = mask.clone()

    # Compute loss using our function.
    loss = masked_cross_entropy(logits_copy, targets_copy, mask_copy)

    # Prepare the expected targets: positions with mask=0 (second example) become -100 (ignored).
    expected_targets = targets.clone()
    expected_targets.masked_fill_(mask.view(-1) == 0, -100)

    # Compute the expected loss with F.cross_entropy which ignores targets with -100.
    expected_loss = F.cross_entropy(logits, expected_targets)

    # Check that the loss matches.
    assert torch.allclose(loss, expected_loss), "Loss with mask does not match expected masked cross_entropy loss."


def test_all_masked_out():
    # Test the case where all elements are masked out.
    logits = torch.tensor([[2.0, 1.0], [0.5, 1.5]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)
    # All items masked
    mask = torch.tensor([0, 0], dtype=torch.uint8)

    logits_copy = logits.clone()
    targets_copy = targets.clone()
    mask_copy = mask.clone()

    loss = masked_cross_entropy(logits_copy, targets_copy, mask_copy)
    assert torch.isnan(loss), "Expected loss to be nan"


if __name__ == "main":
    pytest.main()
