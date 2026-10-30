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
    ``enabled=False`` observes attention without changing decoder outputs or the
    progress anchor. Text-head guidance requires a matching raw input/output BPE
    vocabulary and consumes the current frame's logits after its forward pass.
    """

    def __init__(
        self,
        batch_size: int,
        look_back_tokens: int = 1,
        look_ahead_tokens: int = 5,
        prior_epsilon: float = 0.1,
        layer: str | int = 'middle',
        head: int = 0,
        source: str = 'attention',
        enabled: bool = True,
        record_alignments: bool = False,
        min_confidence: float = 0.0,
        min_margin: float = 0.0,
        min_token_frames: int = 1,
    ):
        for name, value, minimum in (
            ('batch_size', batch_size, 1),
            ('look_back_tokens', look_back_tokens, 0),
            ('look_ahead_tokens', look_ahead_tokens, 0),
            ('head', head, 0),
            ('min_token_frames', min_token_frames, 1),
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
        if source not in ('attention', 'text_head'):
            raise ValueError('source must be attention or text_head')
        for name, value in (('enabled', enabled), ('record_alignments', record_alignments)):
            if type(value) is not bool:
                raise ValueError(f'{name} must be a boolean')
        for name, value in (('min_confidence', min_confidence), ('min_margin', min_margin)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f'{name} must be finite and in [0, 1]')
        self.batch_size = batch_size
        self.look_back_tokens = look_back_tokens
        self.look_ahead_tokens = look_ahead_tokens
        self.prior_epsilon = float(prior_epsilon)
        self.layer = layer
        self.head = head
        self.source = source
        self.enabled = enabled
        self.record_alignments = record_alignments
        self.min_confidence = float(min_confidence)
        self.min_margin = float(min_margin)
        self.min_token_frames = min_token_frames
        self.text_key_positions: list[list[int]] = [[] for _ in range(batch_size)]
        self.text_token_ids: list[list[int | None]] = [[] for _ in range(batch_size)]
        self.anchors = [0] * batch_size
        self.last_attended_tokens = [-1] * batch_size
        self.last_observations: list[dict | None] = [None] * batch_size
        self.last_head_observations: list[dict | None] = [None] * batch_size
        self._visible_token_ordinals: list[list[int]] = [[] for _ in range(batch_size)]
        self._anchor_frames = [0] * batch_size

    def add_text_token(self, batch_index: int, key_position: int, token_id: int | None = None):
        """Register one actual conditioning BPE key, excluding controls and padding."""
        self._validate_batch_index(batch_index)
        if type(key_position) is not int or key_position < 0:
            raise ValueError('key_position must be a nonnegative integer')
        if token_id is not None and (type(token_id) is not int or token_id < 0):
            raise ValueError('token_id must be a nonnegative integer or None')
        positions = self.text_key_positions[batch_index]
        if positions and key_position <= positions[-1]:
            raise ValueError('Text key positions must be strictly increasing within a turn')
        positions.append(key_position)
        self.text_token_ids[batch_index].append(token_id)

    def reset_turn(self, batch_index: int):
        """Start a new turn without changing any decoder KV cache."""
        self._validate_batch_index(batch_index)
        self.text_key_positions[batch_index] = []
        self.text_token_ids[batch_index] = []
        self.anchors[batch_index] = 0
        self.last_attended_tokens[batch_index] = -1
        self.last_observations[batch_index] = None
        self.last_head_observations[batch_index] = None
        self._visible_token_ordinals[batch_index] = []
        self._anchor_frames[batch_index] = 0

    def observe_text_logits(self, logits, active_batch_indices=None, ignored_token_ids=()):
        """Observe this audio frame's BPE head and update the next frame's window.

        Logits are the conditional rows, with the head's ordinary output mask
        already applied. Confidence is measured over that full vocabulary, not
        renormalized over the input text. Only positions visible to the preceding
        attention query may be matched. Adjacent identical input IDs cannot be
        distinguished by this head; keep the current occurrence when possible
        and report that ambiguity instead of inventing a boundary.
        """
        if logits.ndim != 2 or logits.size(0) != self.batch_size or logits.size(1) < 1:
            raise ValueError('Text-head logits must have shape [conditional_batch_size, vocabulary_size]')
        indices = self.request(active_batch_indices).batch_indices
        ignored = set(ignored_token_ids)
        for batch_index in indices:
            probabilities = F.softmax(logits[batch_index].detach().float(), dim=-1)
            top = probabilities.topk(min(2, probabilities.numel()))
            # Match the ordinary output head's argmax, including its earliest-ID
            # tie break. topk is used only for the probability margin.
            token_id = int(logits[batch_index].argmax().item())
            confidence = float(top.values[0].item())
            margin = confidence - (float(top.values[1].item()) if top.values.numel() > 1 else 0.0)
            anchor = self.anchors[batch_index]
            available = self._visible_token_ordinals[batch_index]
            candidates = [index for index in available if anchor <= index <= anchor + self.look_ahead_tokens]
            matches = [index for index in candidates if self.text_token_ids[batch_index][index] == token_id]
            candidate = matches[0] if matches else None
            ambiguous = len(matches) > 1
            accepted = False
            if self.source == 'text_head':
                self._anchor_frames[batch_index] += 1
            if not math.isfinite(confidence) or not math.isfinite(margin):
                reason = 'invalid_logits'
                confidence = 0.0
                margin = 0.0
            elif token_id in ignored:
                reason = 'ignored_token'
            elif candidate is None:
                reason = 'unmatched_token'
            elif confidence < self.min_confidence:
                reason = 'low_confidence'
            elif margin < self.min_margin:
                reason = 'low_margin'
            elif candidate > anchor and self._anchor_frames[batch_index] < self.min_token_frames:
                reason = 'minimum_frames'
            else:
                accepted = True
                reason = 'ambiguous_token' if ambiguous else 'matched_token'
                if self.enabled and self.source == 'text_head' and candidate > anchor:
                    self.anchors[batch_index] = candidate
                    self._anchor_frames[batch_index] = 0
            self.last_head_observations[batch_index] = {
                'token_id': token_id,
                'probability': confidence,
                'margin': margin,
                'candidate_token_index': candidate,
                'anchor_before': anchor,
                'anchor': self.anchors[batch_index],
                'accepted': accepted,
                'ambiguous': ambiguous,
                'reason': reason,
            }
        return self.last_head_observations

    def begin_step(self):
        """Clear per-query observations so inactive rows never reuse stale peaks."""
        self.last_observations = [None] * self.batch_size
        self.last_head_observations = [None] * self.batch_size
        self._visible_token_ordinals = [[] for _ in range(self.batch_size)]

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
    progress window. It never reveals future/padded keys. For an attention-driven
    controller the new anchor is chosen within that window; a text-head-driven
    controller updates it separately after the current forward pass. The anchor
    cannot move backwards; actual attention remains free to look back or overcome
    the finite prior. Record-only requests preserve ``ordinary_output`` exactly.
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
        state.last_observations[batch_index] = None
        state._visible_token_ordinals[batch_index] = []
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
        state._visible_token_ordinals[batch_index] = ordinals.tolist()
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
            if state.enabled:
                prior[outside_keys] = state.prior_epsilon
            raw_probabilities = F.softmax(scores, dim=-1)
            probabilities = F.softmax(scores + prior.log(), dim=-1)
            text_probabilities = probabilities.index_select(0, text_keys)
            raw_text_probabilities = raw_probabilities.index_select(0, text_keys)
            raw_peak_index = int(raw_text_probabilities.argmax().item())
            peak_index = int(text_probabilities.argmax().item())
            peak = ordinals[peak_index]
            state.last_attended_tokens[batch_index] = int(peak.item())
            accepted = False
            confidence = 0.0
            margin = 0.0
            candidate = None
            if state.source == 'attention':
                state._anchor_frames[batch_index] += 1
            if inside.any():
                candidates = ordinals[inside]
                candidate_keys = text_keys[inside]
                candidate_probabilities = raw_probabilities.index_select(0, candidate_keys)
                top = candidate_probabilities.topk(min(2, candidate_probabilities.numel()))
                # argmax preserves the earliest-position tie break of the legacy
                # controller; topk is used only for the confidence margin.
                candidate = int(candidates[candidate_probabilities.argmax()].item())
                confidence = float(top.values[0].item())
                margin = confidence - (float(top.values[1].item()) if top.values.numel() > 1 else 0.0)
                accepted = (
                    confidence >= state.min_confidence
                    and margin >= state.min_margin
                    and (candidate <= anchor or state._anchor_frames[batch_index] >= state.min_token_frames)
                )
                if state.enabled and state.source == 'attention' and accepted and candidate > anchor:
                    state.anchors[batch_index] = candidate
                    state._anchor_frames[batch_index] = 0
            state.last_observations[batch_index] = {
                'attention_token_index': int(ordinals[raw_peak_index].item()),
                'attention_probability': float(raw_text_probabilities[raw_peak_index].item()),
                'guided_attention_token_index': int(peak.item()),
                'guided_attention_probability': float(text_probabilities[peak_index].item()),
                'text_attention_mass': float(raw_text_probabilities.sum().item()),
                'guided_text_attention_mass': float(text_probabilities.sum().item()),
                'anchor_before': anchor,
                'anchor': state.anchors[batch_index],
                'candidate_token_index': candidate,
                'confidence': confidence,
                'margin': margin,
                'accepted': accepted,
            }
            if state.enabled and state.prior_epsilon < 1 and outside_keys.numel():
                output[batch_index, state.head, 0] = (probabilities @ value[batch_index, state.head].float()).to(
                    dtype=query.dtype
                )
    return output
