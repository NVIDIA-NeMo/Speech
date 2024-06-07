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

"""Utilities for models."""
import itertools
import math
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from nemo.utils import logging, logging_mode

try:
    from apex.normalization import MixedFusedRMSNorm
    from apex.normalization.fused_layer_norm import FusedLayerNorm  # NOQA
    from apex.transformer.enums import AttnMaskType
    from apex.transformer.layers.layer_norm import FastLayerNorm
    from apex.transformer.pipeline_parallel.schedules.common import listify_model

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

try:
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.tensor_parallel.layers import linear_with_grad_accumulation_and_async_allreduce

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


def ApproxGELUActivation(input: Tensor):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """
    return input * torch.sigmoid(1.702 * input)


class ApexGuardDefaults(object):
    """
    This class can be used to replace missing classes when apex is missing.
    """

    def __init__(self):
        super().__init__()

    def __getattr__(self, item):
        return None


def parallel_lm_logits(
    input_: torch.Tensor,
    word_embeddings_weight: torch.Tensor,
    parallel_output: bool,
    bias: torch.Tensor = None,
    async_tensor_model_parallel_allreduce: bool = False,
    sequence_parallel: bool = False,
    gradient_accumulation_fusion: bool = False,
):
    """Language Model logits using word embedding weights.

    Args:
        input_ (torch.Tensor): [b, s, h]
        word_embeddings_weight (torch.Tensor): [(padded) vocab size, h]
        parallel_output (bool): False will gather logits from tensor model parallel region
        bias (torch.Tensor, optional): bias tensor. Defaults to None.
        async_tensor_model_parallel_allreduce (bool, optional): Defaults to False.
        sequence_parallel (bool, optional): If True will use sequence parallelism. Defaults to False.
        gradient_accumulation_fusioa (bool, optional): If True fuse gradient accumulation to WGRAD GEMM

    Returns:
        torch.Tensor: [b, s, (padded) vocab size]
    """

    tensor_model_parallel = parallel_state.get_tensor_model_parallel_world_size() > 1

    # async grad allreduce can only be used when not using sequence parallelism
    async_grad_allreduce = async_tensor_model_parallel_allreduce and tensor_model_parallel and not sequence_parallel

    # copy input_ to model parallel region if needed
    if async_tensor_model_parallel_allreduce or sequence_parallel:
        input_parallel = input_

    else:
        input_parallel = tensor_parallel.copy_to_tensor_model_parallel_region(input_)

    # Matrix multiply.
    logits_parallel = linear_with_grad_accumulation_and_async_allreduce(
        input=input_parallel,
        weight=word_embeddings_weight,
        bias=bias,
        gradient_accumulation_fusion=gradient_accumulation_fusion,
        async_grad_allreduce=async_grad_allreduce,
        sequence_parallel=sequence_parallel,
    )

    # Gather if needed.
    if parallel_output:
        return logits_parallel
    else:
        return tensor_parallel.gather_from_tensor_model_parallel_region(logits_parallel)


def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def init_method_kaiming_uniform(val):
    def init_(tensor):
        return torch.nn.init.kaiming_uniform_(tensor, a=val)

    return init_


def init_method_const(val):
    def init_(tensor):
        return torch.nn.init.constant_(tensor, val)

    return init_


def scaled_init_method_normal(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


def get_linear_layer(rows, columns, init_method):
    """Simple linear layer with weight initialization."""
    layer = torch.nn.Linear(rows, columns)
    init_method(layer.weight)
    with torch.no_grad():
        layer.bias.zero_()
    return layer


@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))


def openai_gelu(x):
    return gelu_impl(x)


try:
    jit_fuser = torch.compile
except:
    jit_fuser = torch.jit.script


@jit_fuser
def squared_relu(x):
    return torch.pow(torch.nn.functional.relu(x), 2)


# This is actually Python equivalent of torch.nn.functional.gelu(), also with type hints for ONNX exporter
@torch.jit.script
def erf_gelu(x):
    return x * 0.5 * (torch.erf(x / 1.41421).to(dtype=x.dtype) + torch.ones_like(x).to(dtype=x.dtype))


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat([loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses, group=parallel_state.get_data_parallel_group())
    averaged_losses = averaged_losses / torch.distributed.get_world_size(
        group=parallel_state.get_data_parallel_group()
    )

    return averaged_losses


def get_ltor_masks_and_position_ids(
    data, eod_token, reset_position_ids, reset_attention_mask, eod_mask_loss, compute_attention_mask=True
):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1

    attention_mask = None
    if compute_attention_mask:
        attention_mask = torch.tril(torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length
        )

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).repeat(micro_batch_size, 1)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indicies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1) :, : (i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1) :] -= i + 1 - prev_index
                    prev_index = i + 1

    if compute_attention_mask:
        # Convert attention mask to binary:
        attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


def attn_mask_postprocess(attn_mask):
    # [b, 1, s, s]
    # Attn_masks for enc-dec attn and dec attn is None when trying to get just the encoder hidden states.
    if attn_mask is None:
        return None
    extended_attention_mask = attn_mask.unsqueeze(1)
    return extended_attention_mask


def enc_dec_extended_attention_mask(attention_mask_list):

    return [attn_mask_postprocess(attn_mask) for attn_mask in attention_mask_list]


def build_position_ids(token_ids):
    # Create position ids
    seq_length = token_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(token_ids).clone()

    return position_ids


def make_attention_mask_3d(source_mask, target_mask):
    """
    Returns a 3-dimensional (3-D) attention mask
    :param source_block: 2-D array
    :param target_block: 2-D array
    """
    mask = target_mask[:, None, :] * source_mask[:, :, None]
    return mask


def make_inference_attention_mask_3d(source_block, target_block, pad_id):
    """
    Returns a 3-dimensional (3-D) attention mask
    :param source_block: 2-D array
    :param target_block: 2-D array
    """
    # mask = (target_block[:, None, :] != pad_id) * (source_block[:, :, None] != pad_id)
    return make_attention_mask_3d(source_block != pad_id, target_block != pad_id)


def make_inference_history_mask_3d(block):
    batch, length = block.shape
    arange = torch.arange(length, device=block.device)
    history_mask = (arange[None,] <= arange[:, None])[None,]
    history_mask = history_mask.expand(batch, length, length)
    return history_mask


def build_attention_mask_3d_padding(source_mask, target_mask):
    """
    Returns a 3D joint attention mask for Megatron given two 2D masks
    :param source_mask - True for non-masked, else masked [batch, src length]
    :param target_mask - True for non-masked, else masked [batch, tgt length]
    """
    mask = make_attention_mask_3d(source_mask, target_mask)
    # invert mask for Megatron
    return mask < 0.5


def build_attention_mask_3d_causal(source_mask, target_mask):
    """
    Returns a 3D joint attention mask for Megatron given two 2D masks
    :param source_mask - True for non-masked, else masked [batch, src length]
    :param target_mask - True for non-masked, else masked [batch, tgt length]
    """
    causal_mask = make_inference_history_mask_3d(target_mask)
    mask = make_attention_mask_3d(source_mask, target_mask)
    mask = mask * causal_mask
    # invert mask for Megatron
    return mask < 0.5


def build_attention_mask_3d(source_mask, target_mask, attn_mask_type):
    """
    Returns a 3D attention mask for Megatron given two 2D masks
    :param source_mask - < 0.5 for non-masked, else masked [batch, src length]
    :param target_mask - < 0.5 for non-masked, else masked [batch, tgt length]
    :param attn_mask_type - AttnMaskType enum
    """
    if attn_mask_type == AttnMaskType.padding:
        mask = build_attention_mask_3d_padding(source_mask, target_mask)
    elif attn_mask_type == AttnMaskType.causal:
        mask = build_attention_mask_3d_causal(source_mask, target_mask)
    else:
        raise ValueError(f"Unsupported attention mask attn_mask_type = {attn_mask_type}")

    return mask


def get_params_for_weight_decay_optimization(
    model: Union[torch.nn.Module, List[torch.nn.Module]],
) -> Dict[str, torch.nn.Parameter]:
    """Divide params into with-weight-decay and without-weight-decay groups.

    Layernorms and biases will have no weight decay but the rest will.
    """
    modules = listify_model(model)
    weight_decay_params = {'params': [], 'is_expert': False}
    weight_decay_expert_params = {'params': [], 'is_expert': True}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0, 'is_expert': False}
    # EP params have the 'allreduce' attr set.
    is_expert = lambda param: not getattr(param, 'allreduce', True)
    # Do the actual param classification
    for module in modules:
        for module_ in module.modules():
            if isinstance(module_, (FusedLayerNorm, FastLayerNorm, MixedFusedRMSNorm)):
                no_weight_decay_params['params'].extend(
                    list(filter(lambda p: p is not None, module_._parameters.values()))
                )
            else:
                for name, param in module_._parameters.items():
                    if param is None:
                        continue
                    if name.endswith('bias'):
                        no_weight_decay_params['params'].extend([param])
                    else:
                        if is_expert(param):
                            weight_decay_expert_params['params'].extend([param])
                        else:
                            weight_decay_params['params'].extend([param])

    param_groups = [weight_decay_params, weight_decay_expert_params, no_weight_decay_params]
    return tuple(filter(lambda g: len(g['params']) > 0, param_groups))


def get_all_params_for_weight_decay_optimization(
    model: Union[torch.nn.Module, List[torch.nn.Module]],
) -> Tuple[Dict[str, List[torch.nn.Parameter]]]:
    """Use all params for weight decay."""
    modules = listify_model(model)

    weight_decay_params = {'params': [], 'is_expert': False}
    weight_decay_expert_params = {'params': [], 'is_expert': True}

    # populate with params
    is_expert = lambda param: not getattr(param, 'allreduce', True)
    for module in modules:
        weight_decay_params['params'] += list(filter(lambda x: not is_expert(x), module.parameters()))
        weight_decay_expert_params['params'] += list(filter(is_expert, module.parameters()))

    param_groups = [weight_decay_params, weight_decay_expert_params]
    return tuple(filter(lambda g: len(g['params']) > 0, param_groups))


def split_list(inputs, num_chunks, enforce_divisible_batch: Optional[bool] = True):
    """
    Split a list into equal sized chunks
    """
    chunk_size = len(inputs) // num_chunks
    if enforce_divisible_batch:
        assert len(inputs) % chunk_size == 0, "Issue with batch size configuration!"
    return [inputs[i : i + chunk_size] for i in range(0, len(inputs), chunk_size)]


def get_iterator_k_split(
    batch: Union[Dict, List[torch.Tensor]], num_microbatches: int, enforce_divisible_batch: Optional[bool] = True
) -> Iterator:
    """
    Split a batch into k microbatches, where the batch size is divisible by k. Batch could be
    a dictionary of tensors or a list of tensors. A dictionary batch could also have items of List type,
    as long as the length of that list is the same as the batch size.
    """
    if isinstance(batch, dict):
        discard_items = [k for k, v in batch.items() if not isinstance(v, (torch.Tensor, list))]
        if len(discard_items) > 0:
            logging.warning(
                f"Only support splitting torch.Tensor and List[torch.Tensor]. Discarding the following keys from the batch: {discard_items}",
                mode=logging_mode.ONCE,
            )

        batch = {k: v for k, v in batch.items() if isinstance(v, (torch.Tensor, list))}
        tensor_items = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
        list_items = {k: v for k, v in batch.items() if isinstance(v, list)}

        # Split tensor items
        items = list(tensor_items.items())
        if enforce_divisible_batch:
            assert items[0][1].shape[0] % num_microbatches == 0, "Issue with batch size configuration!"
        split_batch = [torch.tensor_split(item[1], num_microbatches, dim=0) for item in items]
        # handle the case where the batch size from dynamic bucketting is not divisible
        if items[0][1].shape[0] % num_microbatches != 0:
            chunk_size = split_batch[0][-1].shape[0]
            split_batch = [[j[:chunk_size] for j in i] for i in split_batch]

        if len(list_items) == 0:
            # Only have tensor items
            microbatches = [
                [(items[i][0], split_batch[i][j]) for i in range(len(items))] for j in range(num_microbatches)
            ]
        else:
            # Split list items
            list_items = list(list_items.items())
            split_list_batch = [
                split_list(item[1], num_microbatches, enforce_divisible_batch=enforce_divisible_batch)
                for item in list_items
            ]
            # Merge tensor and list items
            all_keys = [item[0] for item in items] + [item[0] for item in list_items]
            all_split_batch = split_batch + split_list_batch
            microbatches = [
                [(all_keys[i], all_split_batch[i][j]) for i in range(len(all_keys))] for j in range(num_microbatches)
            ]
        microbatches = [dict(elem) for elem in microbatches]
    else:
        # Split a list of torch tensors
        assert batch[0].shape[0] % num_microbatches == 0, "Issue with batch size configuration!"
        split_batch = [
            torch.tensor_split(item, num_microbatches, dim=0) if torch.is_tensor(item) else item for item in batch
        ]
        microbatches = [
            [elem[i] if elem is not None else elem for elem in split_batch] for i in range(num_microbatches)
        ]

    return itertools.chain(microbatches)


def _cast_if_autocast_enabled(tensor):
    if torch.is_autocast_enabled():
        if isinstance(tensor, torch.Tensor):
            if tensor.device.type == 'cuda':
                dtype = torch.get_autocast_gpu_dtype()
            elif tensor.device.type == 'cpu':
                dtype = torch.get_autocast_cpu_dtype()
            else:
                raise NotImplementedError()
            return tensor.to(dtype=dtype)
    return tensor
