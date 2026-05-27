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

# flake8: noqa
# pylint: skip-file

"""Utilities for models."""
import itertools
from typing import Dict, Iterator, List, Optional, Union

import torch
from torch import Tensor

from nemo.utils import logging, logging_mode

try:
    from apex.transformer.enums import AttnMaskType

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

try:
    from megatron.core import parallel_state

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


def init_method_kaiming_uniform(val):
    def init_(tensor):
        return torch.nn.init.kaiming_uniform_(tensor, a=val)

    return init_


def init_method_const(val):
    def init_(tensor):
        return torch.nn.init.constant_(tensor, val)

    return init_


def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


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
            if items[0][1].shape[0] % num_microbatches != 0:
                raise ValueError(
                    f"Issue with batch size configuration: batch size {items[0][1].shape[0]} is not divisible by {num_microbatches}!"
                )

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
        split_batch = []
        for item in batch:
            if torch.is_tensor(item):
                split_batch.append(torch.tensor_split(item, num_microbatches, dim=0))
            elif isinstance(item, list):
                if isinstance(item[0], torch.Tensor):
                    split_tensors = [torch.tensor_split(elem, num_microbatches, dim=0) for elem in item]
                    split_tuple = []
                    for mbi in range(num_microbatches):
                        split_tuple.append([split_tensors[i][mbi] for i in range(len(split_tensors))])
                    split_tuple = tuple(split_tuple)
                    split_batch.append(split_tuple)
                else:
                    split_batch.append(split_list(item, num_microbatches))
            elif item is None:
                split_batch.append(item)
            else:
                raise ValueError(f"Unsupported item type: {type(item)}")

        microbatches = [
            [elem[i] if elem is not None else elem for elem in split_batch] for i in range(num_microbatches)
        ]

    return itertools.chain(microbatches)
