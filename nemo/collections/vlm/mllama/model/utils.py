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


from typing import Tuple

import torch


def _pad_attention_masks(
    masks: torch.Tensor,
    num_chunks: torch.Tensor,
    total_length: int,
    max_chunks: int,
    device: torch.device,
    dtype=torch.bfloat16,
) -> torch.Tensor:
    """
    Pads the provided masks to a uniform shape for batching.

    Args:
        masks (torch.Tensor): List of tensors containing attention masks for each batch.
        num_chunks (torch.Tensor): Tensor containing the number of chunks for each mask.
        total_length (int): Total sequence length for padding.
        max_chunks (int): Maximum number of chunks to pad each mask to.
        device (torch.device): Device to place the output tensor on.
        dtype (torch.dtype): Data type for the output tensor. Default is `torch.bfloat16`.

    Returns:
        torch.Tensor: A padded tensor of shape [B, total_length, max_num_media, max_chunks]
        where `B` is the batch size.
    """
    mask_value = 1.0
    batch_size = len(masks)
    max_num_media = max([len(m) for m in masks])

    padded_masks = torch.full(
        (batch_size, total_length, max_num_media, max_chunks),
        mask_value,
        dtype=dtype,
        device=device,
    )

    for idx, (mask_group, chunks) in enumerate(zip(masks, num_chunks)):
        for media_idx, (mask, chunk_count) in enumerate(zip(mask_group, chunks)):
            if len(mask) == 2:
                mask[1] = min(mask[1], total_length)
                if mask[1] == -1:
                    mask[1] = total_length
                padded_masks[idx, mask[0] : mask[1], media_idx, :chunk_count].fill_(0.0)

    return padded_masks


def _get_full_row_masked_out_mask(
    attention_bias: torch.Tensor,
    mask_value: float,
):
    """
    Determines whether each row in the attention bias tensor contains masked values.

    Args:
        attention_bias (torch.Tensor): A 4D tensor of shape [B, H, S1, S2], where:
            - B: Batch size.
            - H: Number of attention heads.
            - S1: Length of the first sequence.
            - S2: Length of the second sequence.
        mask_value (float): The value used to represent masked positions in `attention_bias`.

    Returns:
        torch.Tensor: A 4D tensor of shape [B, H, S1, 1], containing boolean values (as a tensor)
        indicating if each row in the last dimension is fully masked (0 if fully masked, 1 otherwise).
    """
    return (attention_bias != mask_value).any(dim=-1).type_as(attention_bias)[..., None]


def _generate_cross_attention_mask(
    text_token_count: int,
    text_device: torch.device,
    text_dtype: torch.dtype,
    vision_tokens: torch.Tensor,
    cross_attention_masks: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a cross-attention mask for aligning text and vision tokens.

    Args:
        text_token_count (int): Number of tokens in the text sequence.
        text_device (torch.device): Device to place the output tensor on.
        text_dtype (torch.dtype): Data type for the output tensor.
        vision_tokens (torch.Tensor): Vision tokens tensor of shape [B, I, T, D] where:
            - B: Batch size.
            - I: Number of images.
            - T: Number of image tokens per image.
            - D: Dimension of each image token.
        cross_attention_masks (torch.Tensor): Cross attention masks of shape [B, N, I, C], where:
            - B: Batch size.
            - N: Number of text tokens.
            - I: Number of images.
            - C: Number of chunks.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The adjusted cross-attention masks of shape [B, 1, N, I * T].
            - The full row mask status tensor of shape [B, 1, N, 1].
    """
    assert vision_tokens is not None, "Vision tokens must be provided"
    vision_token_length = vision_tokens.shape[3]
    assert (
        vision_tokens.shape[1] == cross_attention_masks.shape[2]
    ), f"Mismatch in number of images given and number of masks provided: {vision_tokens.shape} vs {cross_attention_masks.shape}"
    assert (
        vision_tokens.shape[2] == cross_attention_masks.shape[3]
    ), f"Mismatch between vision tokens and cross-attention masks: {vision_tokens.shape} vs {cross_attention_masks.shape}"
    assert (
        text_token_count == cross_attention_masks.shape[1]
    ), f"Text sequence length {text_token_count} does not match cross-attention mask length {cross_attention_masks.shape[1]}"

    batch_size, _, num_images, num_chunks = cross_attention_masks.shape
    cross_attention_masks = cross_attention_masks.view(batch_size, text_token_count, -1).unsqueeze(1)

    full_row_mask_status = _get_full_row_masked_out_mask(cross_attention_masks, mask_value=1.0)
    cross_attention_masks = cross_attention_masks.repeat_interleave(vision_token_length, dim=3)
    cross_attention_masks *= full_row_mask_status

    return (
        cross_attention_masks.to(device=text_device, dtype=text_dtype),
        full_row_mask_status.to(device=text_device, dtype=text_dtype),
    )


def create_vision_mask_tensor(tokens: torch.Tensor, vision_token_id: int = 128256) -> torch.Tensor:
    """
    Create a vision mask from a tensor of tokens and a vision token ID.

    Args:
        tokens (torch.Tensor): A 1D tensor of token IDs.
        vision_token_id (int): The ID of the vision token.

    Returns:
        torch.Tensor: A tensor containing vision masks in the format [start, end].
    """
    # Get the locations of the vision tokens
    vision_token_locations = (tokens == vision_token_id).nonzero(as_tuple=False).squeeze()

    # If no vision token found, return an empty tensor
    if vision_token_locations.numel() == 0:
        return torch.empty(1, 2, dtype=torch.long)

    vision_masks = []

    # Handle case with only one vision token
    if vision_token_locations.numel() == 1:
        vision_masks.append([vision_token_locations.item(), len(tokens)])
    else:
        # Multiple vision tokens, pairwise masks
        for i in range(len(vision_token_locations) - 1):
            vision_masks.append([vision_token_locations[i].item(), vision_token_locations[i + 1].item()])
        # Last vision token attends to all subsequent text
        vision_masks.append([vision_token_locations[-1].item(), len(tokens)])

    # Handle consecutive vision tokens
    last_mask_end = vision_masks[-1][1]
    for vision_mask in reversed(vision_masks):
        if vision_mask[0] == vision_mask[1] - 1:
            vision_mask[1] = last_mask_end
        last_mask_end = vision_mask[1]

    return torch.tensor(vision_masks, dtype=torch.long)
