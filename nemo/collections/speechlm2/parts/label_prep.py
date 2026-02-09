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

"""Utility functions for preparing model inputs including text and ASR channels."""

import torch


def delay_eos(tokens, eos_token_id, pad_token_id, shift=10):
    """
    Delays each EOS token by `shift` steps forward. Replaces original EOS with PAD.
    Skips move if it would go out of bounds or overwrite another EOS/PAD.
    Safe for GPU execution.
    """
    _, T = tokens.shape
    tokens = tokens.clone()

    # Find all EOS positions
    eos_mask = tokens == eos_token_id
    if not eos_mask.any():
        return tokens

    # Flattened indices of EOS tokens
    eos_indices = eos_mask.nonzero(as_tuple=False)  # [N, 2]
    b_idx = eos_indices[:, 0]  # [N]
    eos_pos = eos_indices[:, 1]  # [N]
    new_pos = eos_pos + shift  # [N]

    # Filter: new position must be in bounds and not overwrite EOS or PAD
    valid = new_pos < T
    if valid.any():
        b_idx = b_idx[valid]
        old_pos = eos_pos[valid]
        new_pos = new_pos[valid]

        # Now, check overwrite safety in new positions
        target_vals = tokens[b_idx, new_pos]
        safe = target_vals != eos_token_id

        if safe.any():
            b_idx = b_idx[safe]
            old_pos = old_pos[safe]
            new_pos = new_pos[safe]
            # Move EOS token: clear original, set new
            tokens[b_idx, old_pos] = pad_token_id
            tokens[b_idx, new_pos] = eos_token_id
    return tokens


def prepare_text_and_asr_labels(
    batch,
    target_tokens,
    source_encoded,
    cfg,
    predict_user_text,
    user_eos_id,
    text_pad_id,
    text_bos_id,
    text_eos_id,
    advance_text_channel_by=None,
    use_tp=False,
    device_mesh=None,
):
    """
    Prepare text and ASR labels for duplex STT model training.

    This function handles:
    - Text channel delay/advance adjustments (for speech-text alignment)
    - User text prediction with delayed source tokens (ASR channel)
    - User turn masking and agent turn boundary preservation
    - ASR head processing for conversational models
    - Tensor parallelism adjustments for distributed training

    Args:
        batch: Dictionary containing batch data including source_tokens, target_tokens, etc.
        target_tokens: Target text tokens (B, T)
        source_encoded: Encoded source audio features (B, T, D)
        cfg: Configuration object with model settings
        predict_user_text: Whether to predict user text in addition to agent text
        user_eos_id: Token ID for user turn ending
        text_pad_id: Token ID for text padding
        text_bos_id: Token ID for agent text beginning
        text_eos_id: Token ID for agent text ending
        advance_text_channel_by: Number of frames to advance text channel prediction
        use_tp: Whether tensor parallelism is enabled
        device_mesh: Device mesh for tensor parallelism

    Returns:
        dict: Dictionary containing:
            - text_inputs: Text input tokens (B, T-1)
            - text_labels: Text label tokens (B, T-1)
            - asr_inputs: ASR input tokens (B, T-1) if predict_user_text is True
            - asr_labels: ASR label tokens (B, T-1) if predict_user_text is True

    Example:
        Input conversation (simplified):
            User: "Hello" → source_tokens: [PAD, USER_BOS, 123, 456, USER_EOS, PAD, PAD, PAD]
            Agent: "Hi there" → target_tokens: [PAD, PAD, PAD, TEXT_BOS, 789, 101, TEXT_EOS, PAD]

        With predict_user_text=True:
            Input:
                target_tokens: [[0, 0, 0, 2, 789, 101, 3, 0]]  # Shape: (1, 8)
                source_tokens: [[0, 1, 123, 456, 4, 0, 0, 0]]  # From batch
                source_encoded: Tensor of shape (1, 8, 1024)

            Output:
                {
                    'text_inputs': [[0, 0, 0, 2, 789, 101, 3]],    # (1, 7) - agent text input
                    'text_labels': [[0, 0, 2, 789, 101, 3, 0]],    # (1, 7) - agent text target
                    'asr_inputs': [[0, 1, 123, 456, 3, 0, 0]],     # (1, 7) - user text input (user_eos→text_eos)
                    'asr_labels': [[1, 123, 456, 3, 0, 0, 0]],     # (1, 7) - user text target
                    'source_encoded': Tensor of shape (1, 8, 1024),
                    'source_token_lens': [...],
                    'target_token_lens': [...]
                }

        With predict_user_text=False (single channel):
            Input:
                target_tokens: [[2, 789, 101, 3, 0, 0, 0, 0]]  # Shape: (1, 8)
                source_encoded: Tensor of shape (1, 8, 1024)

            Output:
                {
                    'text_inputs': [[2, 789, 101, 3, 0, 0, 0]],    # (1, 7)
                    'text_labels': [[789, 101, 3, 0, 0, 0, 0]],    # (1, 7)
                    'source_encoded': Tensor of shape (1, 8, 1024)
                }

        With delay_text_channel_by=2:
            Shifts target_tokens right by 2 positions, adding padding at start:
                Before: [[2, 789, 101, 3, 0, 0, 0, 0]]
                After:  [[0, 0, 2, 789, 101, 3, 0, 0]]
    """

    # Apply text channel delay and advance adjustments
    # move back text channel by x, in inference it advance the text channel prediction
    # it is the oposite of speech delay applied on text channel
    if advance_text_channel_by:
        pad = torch.full(
            (target_tokens.shape[0], advance_text_channel_by),
            fill_value=text_pad_id,
            device=target_tokens.device,
            dtype=torch.long,
        )
        target_tokens = torch.cat([target_tokens[:, advance_text_channel_by:], pad], dim=-1)
        # make sure that eos/bos is in the place (it can cut tokens from the first advance_text_channel_by tokens and this will breaks everything)

    if cfg.get("delay_text_channel_by", 0) > 0:
        delay_by = cfg.get("delay_text_channel_by", 0)

        eos_mask = (target_tokens == text_eos_id) & (
            torch.arange(target_tokens.size(1), device=target_tokens.device).unsqueeze(0)
            >= (target_tokens.size(1) - delay_by)
        )
        for i in range(target_tokens.size(0)):
            if eos_mask[i].any():
                target_tokens[i, -(delay_by)] = text_eos_id
        target_tokens = torch.where(eos_mask, text_pad_id, target_tokens)
        pad = torch.full(
            (target_tokens.shape[0], delay_by),
            fill_value=text_pad_id,
            device=target_tokens.device,
            dtype=torch.long,
        )
        target_tokens = torch.cat([pad, target_tokens[:, :-delay_by]], dim=-1)

    if cfg.get("delay_text_eos_by", None):
        target_tokens = delay_eos(target_tokens, text_eos_id, text_pad_id, shift=cfg.delay_text_eos_by)

    if cfg.get("delay_text_bos_by", None):
        target_tokens = delay_eos(target_tokens, text_bos_id, text_pad_id, shift=cfg.delay_text_bos_by)

    if predict_user_text:
        source_tokens = batch["source_tokens"]

        if source_tokens.shape != target_tokens.shape:
            min_len = min(source_tokens.shape[1], target_tokens.shape[1])
            source_tokens = source_tokens[:, :min_len]
            target_tokens = target_tokens[:, :min_len]
            source_encoded = source_encoded[:, :min_len]

        # Optionally delay the prediction of source_tokens by a flag
        delay_source_text_by = cfg.get("delay_source_text_by", 0)
        if delay_source_text_by > 0:
            pad = torch.full(
                (source_tokens.shape[0], delay_source_text_by),
                fill_value=text_pad_id,
                device=source_tokens.device,
                dtype=torch.long,
            )
            source_tokens_delayed = torch.cat([pad, source_tokens[:, :-delay_source_text_by]], dim=-1)
        else:
            source_tokens_delayed = source_tokens

        source_tokens_flat = source_tokens_delayed.clone()
        target_tokens_flat = target_tokens.clone()

        # Keep user and agent text in separate channels and allow overlap between them
        # To be consistent with the single channel case, replace the user_eos_id with agent_eos_id
        source_tokens_flat = source_tokens_flat.clone()
        source_tokens_flat[source_tokens_flat == user_eos_id] = text_eos_id
        asr_inputs = source_tokens_flat[:, :-1]
        asr_labels = source_tokens_flat[:, 1:]
        text_inputs = target_tokens_flat[:, :-1]
        text_labels = target_tokens_flat[:, 1:]

        result = {
            "asr_inputs": asr_inputs,
            "asr_labels": asr_labels,
            "source_token_lens": batch["source_token_lens"],
            "text_inputs": text_inputs,
            "text_labels": text_labels,
            "target_token_lens": batch["target_token_lens"],
            "source_encoded": source_encoded,
        }
        return result
    else:
        target_tokens_flat = target_tokens

    if use_tp:
        tp_world_size = device_mesh["tensor_parallel"].size()
        if (remainder := (target_tokens.shape[1] - 1) % tp_world_size) != 0:
            target_tokens = target_tokens[:, :-remainder]
            source_encoded = source_encoded[:, :-remainder]

    text_inputs = target_tokens[:, :-1]
    text_labels = target_tokens[:, 1:]

    result = {
        "text_inputs": text_inputs,
        "text_labels": text_labels,
    }

    # Split the merged text channel into asr and text channels (no overlap between them)
    if cfg.get("predict_user_text", False):
        asr_ids = target_tokens.clone()
        asr_inputs = asr_ids[:, :-1]
        asr_labels = asr_ids[:, 1:]

        result["asr_inputs"] = asr_inputs
        result["asr_labels"] = asr_labels

    result["source_encoded"] = source_encoded

    return result
