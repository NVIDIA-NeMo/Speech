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


def masked_cross_entropy(logits, targets, mask=None, fp32_upcast=True, ignore_index=-100):
    """
    Compute the masked cross-entropy loss between logits and targets.

    If a mask is provided, the loss is computed per element, multiplied by the mask,
    and then averaged. If no mask is provided, the standard cross-entropy loss is used.

    Args:
        logits (torch.Tensor): The predicted logits with shape (N, C) where C is the number of classes.
        targets (torch.Tensor): The ground truth class indices with shape (N,).
        mask (torch.Tensor, optional): A tensor that masks the loss computation. Items marked with
            1 will be used to calculate loss, otherwise ignored. Must be broadcastable to the shape
            of the loss. Defaults to None.
        fp32_upcast (bool, optional): if True it will cast logits to float32 before computing
        cross entropy. Default: True.
    Returns:
        torch.Tensor: The computed loss as a scalar tensor.
    """
    # this may happen with CPUOffloadPolicy
    if targets.device != logits.device:
        targets = targets.to(logits.device)
    if mask is not None:
        with torch.no_grad():
            if mask.device != targets.device:
                mask = mask.to(targets.device)
            targets.masked_fill_(mask.view(-1) == 0, ignore_index)
            del mask
    if fp32_upcast:
        logits = logits.float()
    return F.cross_entropy(logits, targets)
