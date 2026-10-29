# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional online attention window using observed text keys and no acoustic teacher."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from nemo.collections.tts.modules.attention_alignment import _mask_visibility, _select_attention_mask


class AttentionAlignmentInferenceState:
    """Per-conversation progress for one selected attention layer and head.

    State indices are conditional decoder rows; any CFG rows appended after them
    remain untouched. Positions refer to the absolute KV-cache timeline. A turn
    reset forgets only this metadata, preserving the decoder's conversation cache.
    """

    def __init__(
        self,
        batch_size: int,
        look_back_tokens: int = 1,
        look_ahead_tokens: int = 5,
        prior_epsilon: float = 0.1,
        layer: str | int = 'middle',
        head: int = 0,
    ):
        for name, value, minimum in (
            ('batch_size', batch_size, 1),
            ('look_back_tokens', look_back_tokens, 0),
            ('look_ahead_tokens', look_ahead_tokens, 0),
            ('head', head, 0),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if (
            isinstance(prior_epsilon, bool)
            or not isinstance(prior_epsilon, (int, float))
            or not math.isfinite(prior_epsilon)
            or not 0 < prior_epsilon <= 1
        ):
            raise ValueError('prior_epsilon must be finite and in (0, 1]')
        if layer != 'middle' and (type(layer) is not int or layer < 0):
            raise ValueError('layer must be middle or a nonnegative attention-layer ordinal')
        self.batch_size = batch_size
        self.look_back_tokens = look_back_tokens
        self.look_ahead_tokens = look_ahead_tokens
        self.prior_epsilon = float(prior_epsilon)
        self.layer = layer
        self.head = head
        self.text_key_positions: list[list[int]] = [[] for _ in range(batch_size)]
        self.anchors = [0] * batch_size
        self.last_attended_tokens = [-1] * batch_size

    def add_text_token(self, batch_index: int, key_position: int):
        """Register one actual conditioning BPE key, excluding controls and padding."""
        self._validate_batch_index(batch_index)
        if type(key_position) is not int or key_position < 0:
            raise ValueError('key_position must be a nonnegative integer')
        positions = self.text_key_positions[batch_index]
        if positions and key_position <= positions[-1]:
            raise ValueError('Text key positions must be strictly increasing within a turn')
        positions.append(key_position)

    def reset_turn(self, batch_index: int):
        """Start a new turn without changing any decoder KV cache."""
        self._validate_batch_index(batch_index)
        self.text_key_positions[batch_index] = []
        self.anchors[batch_index] = 0
        self.last_attended_tokens[batch_index] = -1

    def request(self, active_batch_indices=None):
        """Build an immutable request selecting conditional rows active at this step."""
        if active_batch_indices is None:
            active_batch_indices = tuple(range(self.batch_size))
        return AttentionAlignmentInferenceRequest(self, active_batch_indices)

    def resolve_layer(self, num_attention_layers: int) -> int:
        """Resolve the selected attention ordinal, excluding other hybrid blocks."""
        if type(num_attention_layers) is not int or num_attention_layers < 1:
            raise ValueError('Attention inference guidance requires at least one attention layer')
        ordinal = num_attention_layers // 2 if self.layer == 'middle' else self.layer
        if ordinal >= num_attention_layers:
            raise ValueError('Inference layer exceeds the decoder attention-layer count')
        return ordinal

    def _validate_batch_index(self, batch_index):
        if type(batch_index) is not int or not 0 <= batch_index < self.batch_size:
            raise ValueError('Conditional batch index is out of range')


@dataclass(frozen=True)
class AttentionAlignmentInferenceRequest:
    """One cached decoder step; mutable progress belongs to the referenced state."""

    state: AttentionAlignmentInferenceState
    batch_indices: tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.batch_indices, (tuple, list)):
            raise ValueError('active_batch_indices must be a list or tuple of conditional row indices')
        indices = tuple(self.batch_indices)
        for index in indices:
            self.state._validate_batch_index(index)
        if len(set(indices)) != len(indices):
            raise ValueError('Conditional batch indices must be distinct')
        object.__setattr__(self, 'batch_indices', indices)


def apply_attention_alignment_inference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    ordinary_output: torch.Tensor,
    request: AttentionAlignmentInferenceRequest,
    cache_position: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Bias one head's current-turn text keys while retaining all ordinary non-text keys.

    Inputs use ``[B,H,Q,D]`` with expanded grouped K/V and exactly one query.
    The finite prior scales only current-turn text keys outside the previous
    progress window. It never reveals future/padded keys. The new progress
    anchor is chosen within that window and cannot move backwards; actual
    attention remains free to look back or overcome the finite prior.
    """
    state = request.state
    if (
        query.ndim != 4
        or query.size(2) != 1
        or key.shape != value.shape
        or ordinary_output.shape != query.shape
        or key.shape[:2] != query.shape[:2]
        or key.size(-1) != query.size(-1)
    ):
        raise ValueError('Inference attention guidance requires expanded Q/K/V [B,H,1,D] with cached keys')
    if state.batch_size > query.size(0) or state.head >= query.size(1):
        raise ValueError('Inference state batch size or selected head exceeds the decoder dimensions')
    if attention_mask is not None and attention_mask.ndim not in (2, 4):
        raise ValueError('Inference attention guidance requires a 2D padding or 4D attention mask')
    if cache_position is None:
        position = key.size(2) - 1
    else:
        if cache_position.numel() != 1 or cache_position.dtype not in (torch.int32, torch.int64):
            raise ValueError('Inference guidance requires one integer absolute cache_position')
        position = int(cache_position.item())
    if not 0 <= position < key.size(2):
        raise ValueError('Inference cache_position must be within the absolute cached key timeline')
    output = ordinary_output.clone()
    rows = torch.tensor([position], device=query.device, dtype=torch.long)
    heads = torch.tensor([state.head], device=query.device, dtype=torch.long)
    key_indices = torch.arange(key.size(2), device=query.device)
    for batch_index in request.batch_indices:
        positions = state.text_key_positions[batch_index]
        if not positions:
            continue
        if attention_mask is not None and attention_mask.ndim == 2 and not attention_mask[batch_index, position]:
            continue
        text_keys = torch.tensor(positions, device=query.device, dtype=torch.long)
        ordinals = torch.arange(len(positions), device=query.device)
        observed = (text_keys < key.size(2)) & (text_keys <= position)
        text_keys = text_keys[observed]
        ordinals = ordinals[observed]
        if not text_keys.numel():
            continue
        mask = None
        visible = key_indices <= position
        if attention_mask is not None:
            mask = _select_attention_mask(attention_mask, batch_index, heads, rows, key.size(2))[0, 0]
            visible = visible & _mask_visibility(mask)
        text_visible = visible.index_select(0, text_keys)
        text_keys = text_keys[text_visible]
        ordinals = ordinals[text_visible]
        if not text_keys.numel():
            continue
        with torch.autocast(device_type=query.device.type, enabled=False):
            scores = (query[batch_index, state.head, 0].float() @ key[batch_index, state.head].float().T) / math.sqrt(
                query.size(-1)
            )
            if mask is not None and mask.dtype != torch.bool:
                scores = scores + mask.float()
            scores = scores.masked_fill(~visible, -torch.inf)
            anchor = state.anchors[batch_index]
            inside = (ordinals >= anchor - state.look_back_tokens) & (ordinals <= anchor + state.look_ahead_tokens)
            prior = torch.ones_like(scores)
            outside_keys = text_keys[~inside]
            prior[outside_keys] = state.prior_epsilon
            probabilities = F.softmax(scores + prior.log(), dim=-1)
            peak = ordinals[probabilities.index_select(0, text_keys).argmax()]
            state.last_attended_tokens[batch_index] = int(peak.item())
            if inside.any():
                candidates = ordinals[inside]
                candidate_keys = text_keys[inside]
                candidate = int(candidates[probabilities.index_select(0, candidate_keys).argmax()].item())
                state.anchors[batch_index] = max(anchor, candidate)
            if state.prior_epsilon < 1 and outside_keys.numel():
                output[batch_index, state.head, 0] = (probabilities @ value[batch_index, state.head].float()).to(
                    dtype=query.dtype
                )
    return output
