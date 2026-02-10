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
    text_pad_id,
    text_bos_id,
    text_eos_id,
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
        cfg: Configuration object with model settings. Reads the following keys:
            - predict_user_text (bool): Whether to predict user text in addition to agent text
            - advance_text_channel_by (int, optional): Number of frames to advance text channel
            - delay_text_channel_by (int, optional): Number of frames to delay text channel
            - delay_text_eos_by (int, optional): Number of frames to delay EOS tokens
            - delay_text_bos_by (int, optional): Number of frames to delay BOS tokens
            - delay_source_text_by (int, optional): Number of frames to delay source text
        text_pad_id: Token ID for text padding
        text_bos_id: Token ID for text beginning
        text_eos_id: Token ID for text ending
        use_tp: Whether tensor parallelism is enabled
        device_mesh: Device mesh for tensor parallelism

    Returns:
        dict: Dictionary containing:
            - text_inputs: Text input tokens (B, T-1)
            - text_labels: Text label tokens (B, T-1)
            - target_token_lens: Adjusted target token lengths (B,)
            - source_encoded: Encoded source features (B, T, D)
            - asr_inputs: ASR input tokens (B, T-1) if predict_user_text is True
            - asr_labels: ASR label tokens (B, T-1) if predict_user_text is True

    Example:
        Input conversation (simplified):
            User: "Hello" → source_tokens: [PAD, BOS, 123, 456, EOS, PAD, PAD, PAD]
            Agent: "Hi there" → target_tokens: [PAD, PAD, PAD, BOS, 789, 101, EOS, PAD]

        With predict_user_text=True:
            Input:
                target_tokens: [[0, 0, 0, 2, 789, 101, 3, 0]]  # Shape: (1, 8)
                source_tokens: [[0, 2, 123, 456, 3, 0, 0, 0]]  # From batch
                source_encoded: Tensor of shape (1, 8, 1024)

            Output:
                {
                    'text_inputs': [[0, 0, 0, 2, 789, 101, 3]],       # (1, 7) - agent text input
                    'text_labels': [[0, 0, 2, 789, 101, 3, 0]],       # (1, 7) - agent text target
                    'asr_inputs': [[0, 2, 123, 456, 3, 0, 0]],        # (1, 7) - user text input
                    'asr_labels': [[2, 123, 456, 3, 0, 0, 0]],        # (1, 7) - user text target
                    'target_token_lens': tensor([7]),                 # Adjusted lengths
                    'source_encoded': Tensor of shape (1, 8, 1024),
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
    predict_user_text = cfg.get("predict_user_text", False)
    advance_text_channel_by = cfg.get("advance_text_channel_by", None)

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

    delay_by = cfg.get("delay_text_channel_by", 0)
    if delay_by > 0:
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

    # Clone lengths for adjustment (needed for both single and dual-channel due to TP trimming)
    target_token_lens = batch["target_token_lens"].clone()

    # Handle dual-channel specific processing
    source_tokens_delayed = None

    if predict_user_text:
        source_tokens = batch["source_tokens"]

        if source_tokens.shape != target_tokens.shape:
            min_len = min(source_tokens.shape[1], target_tokens.shape[1])
            source_tokens = source_tokens[:, :min_len]
            target_tokens = target_tokens[:, :min_len]
            source_encoded = source_encoded[:, :min_len]
            # Update lengths after truncation
            target_token_lens = torch.clamp(target_token_lens, max=min_len)

        # Optionally delay the prediction of source_tokens
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

    # Apply tensor parallelism alignment (common for both single and dual-channel)
    if use_tp:
        tp_world_size = device_mesh["tensor_parallel"].size()
        if (remainder := (target_tokens.shape[1] - 1) % tp_world_size) != 0:
            target_tokens = target_tokens[:, :-remainder]
            source_encoded = source_encoded[:, :-remainder]
            if source_tokens_delayed is not None:
                source_tokens_delayed = source_tokens_delayed[:, :-remainder]
            # Update lengths after TP trimming
            target_token_lens = torch.clamp(target_token_lens, max=target_tokens.shape[1])

    # Create input/label pairs (common slicing logic)
    text_inputs = target_tokens[:, :-1]
    text_labels = target_tokens[:, 1:]

    result = {
        "text_inputs": text_inputs,
        "text_labels": text_labels,
        "source_encoded": source_encoded,
        "target_token_lens": target_token_lens,  # Always return adjusted lengths
    }

    # Add dual-channel outputs if enabled
    if predict_user_text:
        result.update(
            {
                "asr_inputs": source_tokens_delayed[:, :-1],
                "asr_labels": source_tokens_delayed[:, 1:],
            }
        )

    return result


def maybe_prepend_prompt_tokens(
    batch,
    embed_fn,
    source_encoded,
    source_encoded_lens,
    text_pad_id,
):
    """Optionally prepend prompt embeddings to source/target sequences.

    If ``batch`` does not contain ``"prompt_tokens"``, returns inputs unchanged.

    When prompt tokens are present, creates new tensors with space for prompt + data, then copies:
    1. Prompt embeddings at the beginning of source_encoded
    2. Audio encodings after the prompt
    3. Target tokens (padded) aligned with the audio + prompt timeline
    4. Source tokens (if present) aligned similarly for ASR head

    All lengths are updated to account for the prompt prefix.
    Batch entries (target_token_lens, source_tokens, source_token_lens) are updated in place.

    Args:
        batch: Dictionary containing batch data. Must have "target_tokens" and "target_token_lens".
            Optionally contains "prompt_tokens", "prompt_token_lens", "source_tokens", "source_token_lens".
        embed_fn: Callable to embed prompt token IDs into embeddings (e.g. model.embed_tokens).
        source_encoded: Encoded source audio features (B, T_src, H).
        source_encoded_lens: Source encoding lengths (B,).
        text_pad_id: Token ID for padding.

    Returns:
        tuple of (source_encoded, source_encoded_lens, target_tokens).
    """
    target_tokens = batch["target_tokens"]

    if "prompt_tokens" not in batch:
        return source_encoded, source_encoded_lens, target_tokens

    prompt_embedded = embed_fn(batch["prompt_tokens"])
    prompt_token_lens = batch["prompt_token_lens"]
    target_token_lens = batch["target_token_lens"]
    source_tokens = batch.get("source_tokens")
    source_token_lens = batch.get("source_token_lens")

    B, max_prompt_len, H = prompt_embedded.shape
    T_src = source_encoded.shape[1]
    T_tgt = target_tokens.shape[1]

    new_source_encoded = torch.zeros(
        B, max_prompt_len + T_src, H, dtype=source_encoded.dtype, device=source_encoded.device
    )
    new_target_tokens = torch.full(
        (B, max_prompt_len + T_tgt), text_pad_id, dtype=target_tokens.dtype, device=target_tokens.device
    )

    new_source_tokens = None
    if source_tokens is not None:
        T_src_tok = source_tokens.shape[1]
        new_source_tokens = torch.full(
            (B, max_prompt_len + T_src_tok),
            text_pad_id,
            dtype=source_tokens.dtype,
            device=source_tokens.device,
        )

    source_encoded_lens = source_encoded_lens.clone()
    target_token_lens = target_token_lens.clone()
    if source_token_lens is not None:
        source_token_lens = source_token_lens.clone()

    for i, prompt_len in enumerate(prompt_token_lens):
        prompt_len = prompt_len.item()

        if prompt_len > 0:
            new_source_encoded[i, :prompt_len, :] = prompt_embedded[i, :prompt_len, :]

        src_len = source_encoded_lens[i].item()
        new_source_encoded[i, prompt_len : prompt_len + src_len, :] = source_encoded[i, :src_len, :]

        tgt_len = target_token_lens[i].item()
        new_target_tokens[i, prompt_len : prompt_len + tgt_len] = target_tokens[i, :tgt_len]

        source_encoded_lens[i] = prompt_len + src_len
        target_token_lens[i] = prompt_len + tgt_len

        if new_source_tokens is not None:
            src_tok_len = source_token_lens[i].item()
            new_source_tokens[i, prompt_len : prompt_len + src_tok_len] = source_tokens[i, :src_tok_len]
            source_token_lens[i] = prompt_len + src_tok_len

    batch["target_token_lens"] = target_token_lens
    if new_source_tokens is not None:
        batch["source_tokens"] = new_source_tokens
        batch["source_token_lens"] = source_token_lens

    return new_source_encoded, source_encoded_lens, new_target_tokens
