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

import torch
import torch.nn as nn

__all__ = ['NEG_INF', 'form_attention_mask', 'transformer_weights_init', 'mask_padded_tokens']

NEG_INF = -10000.0


def form_attention_mask(input_mask, diagonal=None):
    """
    Build attention mask with optional masking of future tokens we forbid
    to attend to (e.g. as it is in Transformer decoder).

    Args:
        input_mask: binary mask of size B x L with 1s corresponding to valid
            tokens and 0s corresponding to padding tokens
        diagonal: diagonal where triangular future mask starts
            None -- do not mask anything
            0 -- regular translation or language modeling future masking
            1 -- query stream masking as in XLNet architecture
    Returns:
        attention_mask: mask of size B x 1 x L x L with 0s corresponding to
            tokens we plan to attend to and -10000 otherwise
    """

    if input_mask is None:
        return None
    attn_shape = (1, input_mask.shape[1], input_mask.shape[1])
    attn_mask = input_mask.to(dtype=bool).unsqueeze(1)
    if diagonal is not None:
        future_mask = torch.tril(torch.ones(attn_shape, dtype=torch.bool, device=input_mask.device), diagonal)
        attn_mask = attn_mask & future_mask
    attention_mask = (1 - attn_mask.to(torch.float)) * NEG_INF
    return attention_mask.unsqueeze(1)


def transformer_weights_init(module, std_init_range=0.02, xavier=True):
    """
    Initialize different weights in Transformer model.

    Args:
        module: torch.nn.Module to be initialized
        std_init_range: standard deviation of normal initializer
        xavier: if True, xavier initializer will be used in Linear layers
            as was proposed in AIAYN paper, otherwise normal initializer
            will be used (like in BERT paper)
    """

    if isinstance(module, nn.Linear):
        if xavier:
            nn.init.xavier_uniform_(module.weight)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=std_init_range)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std_init_range)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)


def mask_padded_tokens(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    Args:
        tokens: input tokens
        pad_id: padding token id
    Returns:
        mask: binary mask of size B x L with 1s corresponding to valid tokens and 0s corresponding to padding tokens
    """

    mask = tokens != pad_id
    return mask
