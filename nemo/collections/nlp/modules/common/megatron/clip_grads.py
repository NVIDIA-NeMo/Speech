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

"""Gradient clipping."""

import itertools

import torch
from torch import inf

from nemo.collections.nlp.modules.common.megatron.module import param_is_not_shared
from nemo.utils import logging

try:
    import amp_C
    from apex.multi_tensor_apply import multi_tensor_applier

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

HAVE_APEX_DISTRIBUTED_ADAM = False

if HAVE_APEX:
    try:
        from apex.contrib.optimizers.distributed_fused_adam import DistributedFusedAdam

        HAVE_APEX_DISTRIBUTED_ADAM = True

    except (ImportError, ModuleNotFoundError):
        pass

try:
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.layers import param_is_not_tensor_parallel_duplicate

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


def clip_grad_norm_fp32(parameters, max_norm, norm_type=2):
    """Clips gradient norm of an iterable of parameters whose gradients
       are in fp32.
    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place.
    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # Filter parameters based on:
    #   - grad should not be none
    #   - parameter should not be shared
    #   - should not be a replica due to tensor model parallelism
    grads = []
    grads_for_norm = []
    for param in parameters:
        grad_not_none = param.grad is not None
        is_not_shared = param_is_not_shared(param)
        is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
        if grad_not_none:
            grad = param.grad.detach()
            # Make sure the grads are in fp32
            assert isinstance(param.grad, torch.cuda.FloatTensor)
            grads.append(grad)
        if grad_not_none and is_not_shared and is_not_tp_duplicate:
            grads_for_norm.append(grad)

    if not grads_for_norm:
        logging.warning("No grads found, consider disabling gradient clipping")

    # Norm parameters.
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    total_norm = 0.0

    # Calculate norm.
    if norm_type == inf:
        if grads_for_norm:  # (@adithyare) grads_for_norm can be empty for adapter training with pp>1
            total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        # Take max across all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_model_parallel_group()
        )
        total_norm = total_norm_cuda[0].item()

    else:
        if norm_type == 2.0:
            dummy_overflow_buf = torch.cuda.IntTensor([0])
            # Use apex's multi-tensor applier for efficiency reasons.
            # Multi-tensor applier takes a function and a list of list
            # and performs the operation on that list all in one kernel.
            if grads_for_norm:  # (@adithyare) grads_for_norm can be empty for adapter training with pp>1
                grad_norm, _ = multi_tensor_applier(
                    amp_C.multi_tensor_l2norm, dummy_overflow_buf, [grads_for_norm], False  # no per-parameter norm
                )
            else:
                grad_norm = 0.0
            # Since we will be summing across data parallel groups,
            # we need the pow(norm-type).
            total_norm = grad_norm ** norm_type

        else:
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm ** norm_type

        # Sum across all model-parallel GPUs.
        total_norm_cuda = torch.cuda.FloatTensor(
            [float(total_norm)]
        )  # (@adithyare) total_norm can be a float at this point so we convert it to cuda.FloatTensor
        torch.distributed.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_model_parallel_group()
        )
        total_norm = total_norm_cuda[0].item()
        total_norm = total_norm ** (1.0 / norm_type)

    # Scale.
    clip_coeff = max_norm / (total_norm + 1.0e-6)
    if clip_coeff < 1.0 and grads:  # (@adithyare) grads can be empty for adapter training.
        dummy_overflow_buf = torch.cuda.IntTensor([0])
        multi_tensor_applier(amp_C.multi_tensor_scale, dummy_overflow_buf, [grads, grads], clip_coeff)

    return total_norm


def count_zeros_fp32(parameters):

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # Filter parameters based on:
    #   - grad should not be none
    #   - parameter should not be shared
    #   - should not be a replica due to tensor model parallelism
    total_num_zeros = 0.0
    for param in parameters:
        grad_not_none = param.grad is not None
        is_not_shared = param_is_not_shared(param)
        is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
        if grad_not_none and is_not_shared and is_not_tp_duplicate:
            grad = param.grad.detach()
            num_zeros = grad.numel() - torch.count_nonzero(grad)
            total_num_zeros = num_zeros + total_num_zeros

    # Sum across all model-parallel GPUs.
    torch.distributed.all_reduce(
        total_num_zeros, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_model_parallel_group()
    )
    total_num_zeros = total_num_zeros.item()

    return total_num_zeros


def clip_grad_norm_distributed_optimizer(optimizer, max_norm, norm_type=2):
    """Clips gradient norm of parameters in distributed optimizer

    This is a wrapper around DistributedFusedAdam.clip_grad_norm with
    added functionality to handle model parallel parameters.

    Arguments:
        parameters (DistributedFusedAdam): distributed optimizer
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Currently
            only 2-norm is supported.

    Returns:
        Total norm of the parameters (viewed as a single vector).

    """
    assert isinstance(optimizer, DistributedFusedAdam)

    # Filter parameters based on:
    #   - parameter should not be shared
    #   - should not be a replica due to tensor model parallelism
    params_for_norm = []
    for param in optimizer.parameters(with_fp32_optim_params=True):
        is_not_shared = param_is_not_shared(param)
        is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
        if is_not_shared and is_not_tp_duplicate:
            params_for_norm.append(param)

    # Compute grad norm
    # Note: DistributedFusedAdam caches grad norm to avoid redundant
    # communication.
    optimizer.grad_norm(parameters=params_for_norm, norm_type=norm_type)

    return optimizer.clip_grad_norm(max_norm, norm_type=norm_type)
