# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.asr.parts.k2.utils import make_non_pad_mask


class GradExpNormalize(torch.autograd.Function):
    """Function for fast gradient normalization.
    Typical use case is normalization for mle loss.
    """

    @staticmethod
    def forward(
        ctx, log_probs: torch.Tensor, input_lengths: torch.Tensor, reduction: str = "mean",
    ):
        mask = make_non_pad_mask(input_lengths, log_probs.shape[1])
        probs = log_probs.exp()
        norm_probs = torch.zeros_like(log_probs)
        norm_probs[mask] += probs[mask]
        if reduction == "mean":
            norm_probs /= norm_probs.shape[0]
        ctx.save_for_backward(norm_probs)
        return log_probs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output - grad_output.sum(-1).unsqueeze(-1) * ctx.saved_tensors[0], None, None


class GradInsert(torch.autograd.Function):
    """Function to attach a pre-computed gradient to a tensor.
    Typical use case is gradient computation before calling loss.backward().
    """

    @staticmethod
    def forward(
        ctx, input_tensor: torch.Tensor, output_tensor: torch.Tensor, grad: torch.Tensor, mask: torch.Tensor,
    ):
        assert input_tensor.requires_grad
        assert not output_tensor.requires_grad and not grad.requires_grad

        ctx.save_for_backward(grad, mask)
        return output_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved_grad, mask = ctx.saved_tensors
        # TODO (alaptev): make it work for grad_output with arbitrary shape
        padded_grad_output = torch.zeros(saved_grad.shape[0], dtype=grad_output.dtype, device=grad_output.device)
        padded_grad_output[mask] = grad_output
        return (padded_grad_output * saved_grad.T).T, None, None, None


class PartialGrad(torch.nn.Module):
    """Module for partial gradient computation.
    Useful when computing loss on batch splits to save memory.
    """

    def __init__(self, func: torch.nn.Module):
        super().__init__()
        self.func = func

    def forward(
        self,
        input_tensor: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ):
        # break the gradient chain
        loc_tensor = input_tensor.detach()
        loc_tensor.requires_grad_(True)

        new_tensor, mask = self.func(loc_tensor, targets, input_lengths, target_lengths)
        loc_new_tensor = new_tensor.detach()

        new_tensor.sum().backward()
        grad = loc_tensor.grad

        return GradInsert.apply(input_tensor, loc_new_tensor, grad, mask), mask
