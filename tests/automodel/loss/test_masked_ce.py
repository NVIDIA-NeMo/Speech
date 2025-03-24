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
from nemo.automodel.loss.masked_ce import masked_cross_entropy


def test_masked_cross_entropy_no_mask():
    # Create dummy data
    batch_size = 4
    num_classes = 3
    torch.manual_seed(0)
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(high=num_classes, size=(batch_size,))

    # Compute loss with our function
    loss_custom = masked_cross_entropy(logits, targets, mask=None)

    # Compute baseline cross-entropy
    loss_ref = F.cross_entropy(logits, targets)

    # They should be very close
    assert torch.allclose(
        loss_custom, loss_ref
    ), f"Loss without mask expected {loss_ref.item():.4f}, but got {loss_custom.item():.4f}"


def test_masked_cross_entropy_with_mask():
    # Create dummy data
    batch_size = 4
    num_classes = 3
    torch.manual_seed(0)
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(high=num_classes, size=(batch_size,))
    mask = torch.tensor([1, 0, 1, 0])  # Only positions 0 and 2 are used

    # Our loss
    loss_custom = masked_cross_entropy(logits, targets, mask=mask)

    # Reference: Manually mask out positions by setting target to -100
    targets_ref = targets.clone()
    targets_ref[mask == 0] = -100
    loss_ref = F.cross_entropy(logits, targets_ref)

    assert torch.allclose(
        loss_custom, loss_ref
    ), f"Loss with mask expected {loss_ref.item():.4f}, but got {loss_custom.item():.4f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_masked_cross_entropy_gpu():
    # Same test as above, but on GPU
    device = torch.device("cuda")
    batch_size = 4
    num_classes = 3
    torch.manual_seed(0)
    logits = torch.randn(batch_size, num_classes, device=device)
    targets = torch.randint(high=num_classes, size=(batch_size,), device=device)
    mask = torch.tensor([1, 0, 1, 1], device=device)

    loss_gpu = masked_cross_entropy(logits, targets, mask=mask)
    assert loss_gpu.dtype == torch.float32  # By default it should be FP32 once cast

    # Double-check it runs without error
    assert loss_gpu is not None
