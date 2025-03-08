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

from typing import List

import torch
import torch.nn.functional as F
from megatron.core.packed_seq_params import PackedSeqParams


def convert_to_packed_llava_next(
    tokens: List[torch.Tensor],
    labels: List[torch.Tensor],
    ignore_index: int,
    pad_to_multiple_of: int = 64,
    final_padding_to: int | None = None,
):
    """
    Convert tokens, labels, and associated inputs into a packed version with padded sequence parameters.

    Args:
        tokens (list[torch.Tensor]): List of token tensors for each instance.
        labels (list[torch.Tensor]): List of label tensors for each instance.
        num_image_embeddings_per_tile (int): Number of image embeddings per tile.
        media_token_index (int): Token ID representing media.
        ignore_index (int): Value to use for padding labels.
        pad_to_multiple_of (int): Sequence length will be padded to a multiple of this value. Default is 8.
        final_padding_to(int): Pad the final seq to make everything to same size
    """
    packed_tokens = []
    packed_labels = []
    packed_position_ids = []
    seqlens_padded = []
    cu_seqlens = [0]
    cu_seqlens_padded = [0]

    for i, (instance_tokens, instance_labels) in enumerate(zip(tokens, labels)):
        seqlen = len(instance_tokens)
        seqlen_padded = (seqlen + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of
        if i == len(tokens) - 1 and final_padding_to:
            assert final_padding_to >= (cu_seqlens_padded[-1] + seqlen_padded), (
                f"{final_padding_to = }, " f"{(cu_seqlens_padded[-1] + seqlen_padded)}"
            )
            seqlen_padded = final_padding_to - cu_seqlens_padded[-1]
        pad_len = seqlen_padded - seqlen

        if pad_len > 0:
            instance_tokens = F.pad(instance_tokens, (0, pad_len), 'constant', 0)
            instance_labels = F.pad(instance_labels, (0, pad_len), 'constant', ignore_index)

        packed_tokens.append(instance_tokens)
        packed_labels.append(instance_labels)
        packed_position_ids.append(torch.arange(len(instance_tokens), dtype=torch.int, device=instance_tokens.device))
        seqlens_padded.append(seqlen_padded)
        cu_seqlens.append(cu_seqlens[-1] + seqlen)
        cu_seqlens_padded.append(cu_seqlens_padded[-1] + seqlen_padded)

    packed_tokens = torch.cat(packed_tokens, dim=0).unsqueeze(0)
    packed_labels = torch.cat(packed_labels, dim=0).unsqueeze(0)
    packed_position_ids = torch.cat(packed_position_ids, dim=0).unsqueeze(0)
    packed_loss_mask = torch.ones_like(packed_labels, dtype=torch.float, device=packed_labels.device)
    packed_loss_mask[packed_labels < 0] = 0.0

    cu_seqlens = torch.IntTensor(cu_seqlens)
    cu_seqlens_padded = torch.IntTensor(cu_seqlens_padded)

    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=int(max(seqlens_padded)),
        max_seqlen_kv=int(max(seqlens_padded)),
        qkv_format='thd',
    )

    return packed_tokens, packed_labels, packed_position_ids, packed_loss_mask, packed_seq_params
