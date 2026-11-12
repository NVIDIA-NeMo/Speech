# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attention alignment supervision and causal streaming token timestamps.

Training targets map acoustic alignment positions to conditioning text keys.
Streaming timestamps observe the ordinary full-key attention probabilities and
retain token spans, incremental events, and optional frame records. Use
``build_token_timestamps`` to crop records to a waveform and
``save_token_timestamps`` to save them beside the audio.
"""

import copy
import json
import math
import operator
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class AttentionAlignmentTurn:
    """A complete output turn on the decoder's absolute, pre-shifted timeline."""

    batch_index: int
    query_positions: torch.Tensor
    key_positions: torch.Tensor
    token_indices: torch.Tensor


@dataclass
class AttentionAlignmentGuidance:
    """No parameters or mutable loss accumulators are stored in this request.

    ``prior_strength=0`` preserves decoder outputs exactly while still computing
    the auxiliary loss. One is a soft text-band prior; intermediate values mix
    that prior with all ones. Heads not listed here always retain ordinary
    attention. ``heads=None`` supervises all heads. ``layers`` selects zero-based
    attention-layer ordinals, excluding intervening Mamba/MLP blocks. ``'middle'``
    selects ordinal ``num_attention_layers // 2``; ``None`` selects every layer.
    Optional look-back/look-ahead token counts give the soft band a flat local
    tolerance window before Gaussian falloff; causal visibility still applies.
    """

    turns: list[AttentionAlignmentTurn]
    prior_strength: float = 0.0
    band_width: float = 1.0
    heads: tuple[int, ...] | None = (0,)
    query_chunk_size: int = 128
    layers: str | tuple[int, ...] | None = 'middle'
    look_back_tokens: int = 0
    look_ahead_tokens: int = 0

    def __post_init__(self):
        if not math.isfinite(self.prior_strength) or not 0 <= self.prior_strength <= 1:
            raise ValueError('prior_strength must be finite and in [0, 1]')
        if not math.isfinite(self.band_width) or self.band_width <= 0:
            raise ValueError('band_width must be finite and positive')
        if type(self.query_chunk_size) is not int or self.query_chunk_size < 1:
            raise ValueError('query_chunk_size must be a positive integer')
        for name in ('look_back_tokens', 'look_ahead_tokens'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.heads is not None:
            self.heads = tuple(self.heads)
            if (
                not self.heads
                or any(type(head) is not int or head < 0 for head in self.heads)
                or len(set(self.heads)) != len(self.heads)
            ):
                raise ValueError('heads must be null or distinct nonnegative head indices')
        if isinstance(self.layers, str):
            if self.layers != 'middle':
                raise ValueError('layers must be middle, null or distinct nonnegative attention-layer indices')
        elif self.layers is not None:
            if not isinstance(self.layers, (list, tuple)):
                raise ValueError('layers must be middle, null or distinct nonnegative attention-layer indices')
            self.layers = tuple(self.layers)
            if (
                not self.layers
                or any(type(layer) is not int or layer < 0 for layer in self.layers)
                or len(set(self.layers)) != len(self.layers)
            ):
                raise ValueError('layers must be middle, null or distinct nonnegative attention-layer indices')

    def resolve_layers(self, num_attention_layers: int) -> tuple[int, ...]:
        """Resolve the selected ordinals against the decoder's actual attention count."""
        if type(num_attention_layers) is not int or num_attention_layers < 1:
            raise ValueError('Attention alignment requires at least one attention layer')
        if self.layers == 'middle':
            return (num_attention_layers // 2,)
        if self.layers is None:
            return tuple(range(num_attention_layers))
        if any(layer >= num_attention_layers for layer in self.layers):
            raise ValueError('An attention alignment layer index exceeds the decoder attention-layer count')
        return self.layers


def apply_attention_alignment(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    ordinary_output: torch.Tensor,
    guidance: AttentionAlignmentGuidance,
    attention_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Guide selected heads without retaining a conversation-square matrix.

    Q/K/V and ordinary output have shape ``[B,H,T,D]``; grouped K/V must have
    already been repeated to the query head count. Cached decoding is deliberately
    unsupported: this teacher is only available for complete training/validation
    examples. Chunk-level non-reentrant checkpointing bounds retained attention
    probabilities while preserving gradients through the guidance and prior.
    """
    if query.ndim != 4 or key.shape != value.shape or ordinary_output.shape != query.shape:
        raise ValueError('Expected Q/K/V and attention output [B,H,T,D]')
    if key.shape[:2] != query.shape[:2] or key.size(-1) != query.size(-1):
        raise ValueError('Attention guidance requires expanded K/V with matching query heads')
    if key.size(2) != query.size(2):
        raise ValueError('Attention guidance does not support cached decoding')
    if attention_mask is not None and attention_mask.ndim not in (2, 4):
        raise ValueError('Attention guidance requires a 2D padding or 4D attention mask')

    batch_size, num_heads, query_length, _ = query.shape
    head_indices = tuple(range(num_heads)) if guidance.heads is None else guidance.heads
    if any(head >= num_heads for head in head_indices):
        raise ValueError('An attention alignment head index exceeds the decoder head count')
    selected_heads = torch.tensor(head_indices, device=query.device, dtype=torch.long)
    # Retain the original tensor and arithmetic exactly when the prior is off.
    output = ordinary_output.clone() if guidance.prior_strength else ordinary_output
    loss_sum = query.float().sum() * 0.0 + key.float().sum() * 0.0
    count = 0
    seen_queries = set()
    for turn in guidance.turns:
        _validate_turn(turn, query, query_length, batch_size)
        if not turn.query_positions.numel() or not turn.key_positions.numel():
            continue
        positions = turn.query_positions.detach().long()
        keys = turn.key_positions.detach().long()
        centers = turn.token_indices.detach().long()
        target_exists = (centers >= 0) & (centers < keys.numel())
        center_keys = keys[centers.clamp(0, keys.numel() - 1)]
        target_exists = target_exists & (center_keys <= positions)
        if attention_mask is not None and attention_mask.ndim == 2:
            target_exists = target_exists & attention_mask[turn.batch_index, positions].bool()
        # Validate overlapping output spans once, outside checkpointed arithmetic.
        for position in positions.detach().cpu().tolist():
            identity = (turn.batch_index, position)
            if identity in seen_queries:
                raise ValueError('Attention alignment query spans must not overlap')
            seen_queries.add(identity)
        if attention_mask is not None:
            center_mask = _select_center_mask(attention_mask, turn.batch_index, selected_heads, positions, center_keys)
            target_exists = target_exists & _mask_visibility(center_mask).all(dim=0)
        positions = positions[target_exists]
        centers = centers[target_exists]
        if not positions.numel():
            continue
        selected_query = query[turn.batch_index].index_select(0, selected_heads)
        selected_key = key[turn.batch_index].index_select(0, selected_heads)
        selected_value = value[turn.batch_index].index_select(0, selected_heads)
        for start in range(0, positions.numel(), guidance.query_chunk_size):
            rows = positions[start : start + guidance.query_chunk_size]
            row_centers = centers[start : start + guidance.query_chunk_size]
            q = selected_query.index_select(1, rows)
            if attention_mask is None:
                # Keep checkpoint inputs compact; the causal query axis is
                # reconstructed inside the chunk rather than retained per row.
                mask = torch.ones((1, 1, key.size(2)), device=q.device, dtype=torch.bool)
            else:
                mask = _select_attention_mask(attention_mask, turn.batch_index, selected_heads, rows, key.size(2))
            args = (
                q,
                selected_key,
                selected_value,
                rows,
                keys,
                row_centers,
                mask,
                guidance.prior_strength,
                guidance.band_width,
                dropout_p,
                guidance.look_back_tokens,
                guidance.look_ahead_tokens,
            )
            if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (q, selected_key, selected_value)):
                chunk_loss, guided_output = checkpoint(_alignment_chunk, *args, use_reentrant=False)
            else:
                chunk_loss, guided_output = _alignment_chunk(*args)
            loss_sum = loss_sum + chunk_loss
            count += len(head_indices) * rows.numel()
            if guidance.prior_strength:
                output[turn.batch_index, selected_heads[:, None], rows[None, :], :] = guided_output
    return output, loss_sum / max(count, 1)


class AttentionTimestampTracker:
    """Convert streaming attention into timestamp records and incremental events.

    Args:
        confirmation_frames: Consecutive frames confirming a new position. One
            confirms the first observed frame. Short forward runs are retained
            as ``short_evidence`` when this value is greater than one.
        min_confidence: Minimum probability of the selected visible text key,
            measured in the original full-key attention distribution.
        min_text_mass: Minimum attention mass on registered visible text keys.
        min_margin: Minimum selected-key probability minus the strongest other
            visible key. Zero disables this gate in monotonic mode, permitting
            weak positions recovered from the sequence posterior.
        method: Causal forward-only posterior path, ``monotonic``.
        force_progress: Permit approximate forward progress when attention has
            left the current position. Forced portions have zero confidence.
        progress_threshold: Relative support decay needed to escape a stall.
        skip_penalty: Log-score cost per skipped text occurrence. Skips allow
            zero-evidence positions to stay unaligned instead of stalling.
        record_frames: Enqueue one instantaneous alignment record per observed
            audio query for ``drain_frames``. Drain this queue regularly.

    ``update`` returns newly generated events and also enqueues them for
    ``drain_events``. Clients should use either delivery mechanism, not both.
    Drain the event queue regularly during long streams. Only aggregated spans,
    one current span and input-position metadata are retained. Events are
    returned on every update, without buffering future audio or attention frames.
    """

    def __init__(
        self,
        confirmation_frames: int = 2,
        min_confidence: float = 0.0,
        min_text_mass: float = 0.0,
        min_margin: float = 0.0,
        method: str = 'monotonic',
        force_progress: bool = True,
        progress_threshold: float = 0.5,
        skip_penalty: float = 6.0,
        record_frames: bool = False,
    ) -> None:
        self.confirmation_frames = _index(confirmation_frames, 'confirmation_frames', minimum=1)
        self.min_confidence = _probability_setting(min_confidence, 'min_confidence')
        self.min_text_mass = _probability_setting(min_text_mass, 'min_text_mass')
        self.min_margin = _probability_setting(min_margin, 'min_margin')
        if method != 'monotonic':
            raise ValueError('method must be monotonic')
        self.method = method
        if not isinstance(force_progress, bool):
            raise ValueError('force_progress must be a boolean')
        self.force_progress = force_progress
        self.progress_threshold = _probability_setting(progress_threshold, 'progress_threshold')
        self.skip_penalty = _nonnegative_setting(skip_penalty, 'skip_penalty')
        if not isinstance(record_frames, bool):
            raise ValueError('record_frames must be a boolean')
        self.record_frames = record_frames
        self._path_decoder = StreamingAttentionPathDecoder(
            skip_penalty=self.skip_penalty,
            force_progress=self.force_progress,
            progress_threshold=self.progress_threshold,
            min_confidence=self.min_confidence,
            min_text_mass=self.min_text_mass,
            min_margin=self.min_margin,
        )
        self._tokens = {}
        self._current = None
        self._last_end = None
        self._input_end = None
        self._duration = None
        self._events = []
        self._next_span_id = 0
        self._frames = []
        self._next_frame_index = 0

    def register_token(self, token_index: int, token_id: int, token: str | None = None) -> None:
        """Register one input occurrence, preserving adjacent repeated token IDs."""
        self._require_open()
        token_index = _index(token_index, 'token_index')
        token_id = _index(token_id, 'token_id')
        if token is not None and not isinstance(token, str):
            raise ValueError('token must be a string or None')
        previous = self._tokens.get(token_index)
        if previous is not None:
            if previous['token_id'] != token_id or (token is not None and previous['token'] not in (None, token)):
                raise ValueError('A registered token position cannot change its identity')
            if token is not None:
                previous['token'] = token
            return
        self._tokens[token_index] = {
            'token_id': token_id,
            'token': token,
            'spans': [],
            'observed_frames': 0,
            'posterior_support': 0.0,
            'max_probability': 0.0,
            'candidate_frames': 0,
        }

    def update(
        self,
        probabilities: Iterable[float],
        *,
        start: float,
        end: float,
        token_indices: Iterable[int] | None = None,
    ) -> list[dict]:
        """Observe one query covering ``[start, end)`` on the audio clock.

        ``token_indices`` maps probabilities to registered input positions. When
        omitted, probabilities must cover every registered position in sorted
        order. Explicit unregistered positions are ignored, so unseen future text
        cannot receive a timestamp. Callers must omit padded/context queries and
        clip a partial final frame before feeding it. Zero-length intervals are
        ignored. Gaps and invalid/insufficient evidence never extend a label.
        """
        self._require_open()
        start, end = _time(start, 'start'), _time(end, 'end')
        if end < start:
            raise ValueError('end must be >= start')
        if self._input_end is not None and start < self._input_end - 1e-9:
            raise ValueError('Audio intervals must be nonoverlapping and ordered')
        try:
            values = [float(value) for value in probabilities]
        except (TypeError, ValueError, OverflowError):
            raise ValueError('probabilities must be a one-dimensional iterable of numeric values') from None
        if token_indices is None:
            indices = sorted(self._tokens)
        else:
            indices = [_index(index, 'token_indices') for index in token_indices]
        if len(values) != len(indices) or len(set(indices)) != len(indices):
            raise ValueError('probabilities and distinct token_indices must have matching lengths')
        if end == start:
            return []
        event_start = len(self._events)
        self._input_end = end
        visible = self._observe_probabilities(values, indices, end - start)
        decisions = self._path_decoder.update(
            [value for value, _ in visible] if visible else [],
            token_indices=[index for _, index in visible] if visible else [],
            start=start,
            end=end,
        )
        assignments = self._consume_decisions(decisions)
        if self.record_frames:
            self._record_frame(assignments, start, end)
        return copy.deepcopy(self._events[event_start:])

    def update_many(
        self,
        probabilities: Iterable[Iterable[float]],
        *,
        starts: Iterable[float],
        ends: Iterable[float],
        token_indices: Iterable[int] | None = None,
    ) -> list[dict]:
        """Consume a chunk of queries with the same registered text-key mapping.

        This is equivalent to consecutive ``update`` calls, so chunk size cannot
        change boundaries or confirmation latency. Future keys in a dense chunk
        must retain zero probability until they are causally available.
        """
        rows, starts, ends = list(probabilities), list(starts), list(ends)
        if len(rows) != len(starts) or len(rows) != len(ends):
            raise ValueError('Attention rows, starts and ends must have matching lengths')
        token_indices = None if token_indices is None else list(token_indices)
        events = []
        for row, start, end in zip(rows, starts, ends):
            events.extend(self.update(row, start=start, end=end, token_indices=token_indices))
        return events

    def snapshot(self) -> list[dict]:
        """Return independent JSON-ready records, including active/provisional spans."""
        result = []
        for index, token in sorted(self._tokens.items()):
            spans = [self._span_record(span, 'finalized') for span in token['spans']]
            status = 'observed' if spans else 'unaligned'
            if self._current is not None and self._current['token_index'] == index:
                status = 'active' if self._current['confirmed'] else 'provisional'
                spans.append(self._span_record(self._current, status))
            if self._duration is not None and spans:
                status = 'finalized'
            primary_index = max(range(len(spans)), key=lambda i: spans[i]['support']) if spans else None
            primary = spans[primary_index] if primary_index is not None else None
            forced = any(span['forced'] for span in spans)
            result.append(
                {
                    'source': 'attention',
                    'method': self.method,
                    'token_index': index,
                    'token_id': token['token_id'],
                    'token': token['token'],
                    'start': primary['start'] if primary else None,
                    'end': primary['end'] if primary else None,
                    'confidence': 0.0 if forced else primary['confidence'] if primary else None,
                    'attention_confidence': primary['attention_confidence'] if primary else None,
                    'forced': forced,
                    'text_attention_mass': primary['text_attention_mass'] if primary else None,
                    'status': status,
                    'ambiguous': len(spans) > 1,
                    'short_evidence': primary['short_evidence'] if primary else False,
                    'primary_span_index': primary_index,
                    'reason': None if spans else self._unaligned_reason(token),
                    'observed_frames': token['observed_frames'],
                    'posterior_support': token['posterior_support'],
                    'max_probability': token['max_probability'],
                    'candidate_frames': token['candidate_frames'],
                    'spans': spans,
                }
            )
        return result

    def drain_events(self) -> list[dict]:
        """Consume pending incremental events; finalized spans themselves are immutable."""
        events, self._events = self._events, []
        return events

    def drain_frames(self) -> list[dict]:
        """Consume instantaneous query records independently of token-span events.

        Records are available immediately after ``update``. Each covers the
        supplied audio interval, with its endpoint assignment and any approximate
        subdivisions in ``assignments``. These causal observations remain unchanged
        when aggregate spans are confirmed or finalized.
        """
        frames, self._frames = self._frames, []
        return frames

    def finalize(self, duration: float) -> list[dict]:
        """Close the stream without extending observations through an unobserved tail.

        ``duration`` is an endpoint on the same clock as update intervals and
        must not precede an already supplied interval. Supplying valid clipped
        audio intervals before update keeps emitted finalized spans immutable.
        Repeated finalization with the same endpoint is idempotent.
        """
        duration = _time(duration, 'duration')
        if self._duration is not None:
            if abs(duration - self._duration) > 1e-9:
                raise ValueError('A finalized stream cannot change duration')
            return self.snapshot()
        if self._input_end is not None and duration < self._input_end - 1e-9:
            raise ValueError('duration cannot precede an observed audio interval; clip before update')
        self._path_decoder.finalize(duration)
        self._end_current(duration)
        self._duration = duration
        return self.snapshot()

    def _apply_candidate(self, candidate, start, end, emitted_at):
        if self._last_end is not None and start > self._last_end + 1e-9:
            self._end_current(emitted_at)
        self._last_end = end
        if candidate is None:
            self._end_current(emitted_at)
            return
        index, confidence, text_mass, forced = candidate
        if self._current is not None and self._current['token_index'] != index:
            self._end_current(emitted_at)
        if self._current is None:
            self._begin(index, start, end, confidence, text_mass, forced, emitted_at)
            return
        self._extend(self._current, end, confidence, text_mass, forced)
        event = 'update'
        if not self._current['confirmed'] and self._current['frame_count'] >= self.confirmation_frames:
            self._current['confirmed'] = True
            event = 'confirmed'
        self._emit(event, self._current, emitted_at)

    def _require_open(self):
        if self._duration is not None:
            raise ValueError('The timestamp stream is already finalized')

    def _observe_probabilities(self, values, indices, duration):
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            return None
        if math.fsum(values) > 1 + 1e-5:
            return None
        visible = [(value, index) for value, index in zip(values, indices) if index in self._tokens]
        if not visible:
            return None
        for probability, index in visible:
            token = self._tokens[index]
            token['observed_frames'] += 1
            token['posterior_support'] += probability * duration
            token['max_probability'] = max(token['max_probability'], probability)
        return visible

    def _consume_decisions(self, decisions):
        assignments = []
        for decision in decisions:
            index = decision['token_index']
            candidate = None
            if index is not None:
                self._tokens[index]['candidate_frames'] += 1
                candidate = index, decision['probability'], decision['text_attention_mass'], decision['forced']
            self._apply_candidate(candidate, decision['start'], decision['end'], decision['emitted_at'])
            if self.record_frames:
                assignments.append(
                    self._frame_assignment(
                        candidate,
                        decision['start'],
                        decision['end'],
                        decision['emitted_at'],
                        decision['margin'],
                        decision['text_attention_mass'],
                    )
                )
        return assignments

    def _frame_assignment(self, candidate, start, end, emitted_at, margin, text_mass):
        index = candidate[0] if candidate is not None else None
        token = self._tokens[index] if index is not None else None
        probability = candidate[1] if candidate is not None else None
        forced = candidate[3] if candidate is not None else False
        return {
            'source': 'attention',
            'method': self.method,
            'start': start,
            'end': end,
            'emitted_at': emitted_at,
            'token_index': index,
            'token_id': token['token_id'] if token is not None else None,
            'token': token['token'] if token is not None else None,
            'confidence': 0.0 if forced else probability,
            'attention_confidence': probability,
            'forced': forced,
            'text_attention_mass': text_mass,
            'margin': margin if candidate is not None else None,
            'status': 'aligned' if candidate is not None else 'unaligned',
        }

    def _record_frame(self, assignments, start, end):
        frame = {
            **assignments[-1],
            'frame_index': self._next_frame_index,
            'start': start,
            'end': end,
            'emitted_at': end,
            'assignments': assignments,
        }
        self._next_frame_index += 1
        self._frames.append(frame)

    def _begin(self, index, start, end, confidence, text_mass, forced, emitted_at):
        span = {
            'span_id': self._next_span_id,
            'token_index': index,
            'start': start,
            'end': start,
            'probability_weight': 0.0,
            'mass_weight': 0.0,
            'frame_count': 0,
            'confirmed': self.confirmation_frames == 1,
            'forced': False,
        }
        self._next_span_id += 1
        self._extend(span, end, confidence, text_mass, forced)
        self._current = span
        self._emit('start', span, emitted_at)

    @staticmethod
    def _extend(span, end, confidence, text_mass, forced=False):
        duration = end - span['end']
        span['probability_weight'] += confidence * duration
        span['mass_weight'] += text_mass * duration
        span['end'] = end
        span['frame_count'] += 1
        span['forced'] = span['forced'] or forced

    def _end_current(self, emitted_at):
        if self._current is not None:
            span, self._current = self._current, None
            self._tokens[span['token_index']]['spans'].append(span)
            self._emit('end', span, emitted_at)

    @staticmethod
    def _unaligned_reason(token):
        if not token['observed_frames']:
            return 'not_observed'
        if not token['max_probability']:
            return 'no_positive_attention'
        return 'never_selected'

    @staticmethod
    def _span_record(span, status):
        duration = span['end'] - span['start']
        return {
            'span_id': span['span_id'],
            'start': span['start'],
            'end': span['end'],
            'confidence': 0.0 if span['forced'] else span['probability_weight'] / duration,
            'attention_confidence': span['probability_weight'] / duration,
            'forced': span['forced'],
            'text_attention_mass': span['mass_weight'] / duration,
            'frame_count': span['frame_count'],
            'confirmed': span['confirmed'],
            'short_evidence': not span['confirmed'],
            'support': span['probability_weight'],
            'status': status,
        }

    def _emit(self, event, span, emitted_at):
        token = self._tokens[span['token_index']]
        status = 'active' if span['confirmed'] else 'provisional'
        if event == 'end':
            status = 'finalized'
        boundary = span['start'] if event in ('start', 'confirmed') else span['end']
        self._events.append(
            {
                **self._span_record(span, status),
                'event': event,
                'source': 'attention',
                'token_index': span['token_index'],
                'token_id': token['token_id'],
                'token': token['token'],
                'emitted_at': emitted_at,
                'latency_seconds': max(0.0, emitted_at - boundary),
            }
        )


class StreamingAttentionPathDecoder:
    """Causal monotonic attention decoding with explicitly marked estimates.

    Ordinary transitions stay, advance, or pay ``skip_penalty`` per crossed
    occurrence. A decayed current peak plus stronger forward evidence can escape
    that skip penalty. Crossed positive-support positions share only the current
    observed frame, with ``forced=True``; no prior boundary is revised. The
    relative-support and text-mass checks avoid advancing solely because the
    head stopped attending to text. Zero-evidence positions and unobserved tails
    stay unaligned. Confidence, text-mass, and margin gates are applied before
    committing progress, so rejected evidence cannot skip a later valid token.
    """

    def __init__(
        self,
        skip_penalty: float = 6.0,
        force_progress: bool = True,
        progress_threshold: float = 0.5,
        min_confidence: float = 0.0,
        min_text_mass: float = 0.0,
        min_margin: float = 0.0,
    ) -> None:
        self.skip_penalty = _nonnegative_setting(skip_penalty, 'skip_penalty')
        if not isinstance(force_progress, bool):
            raise ValueError('force_progress must be a boolean')
        self.force_progress = force_progress
        self.progress_threshold = _probability_setting(progress_threshold, 'progress_threshold')
        self.min_confidence = _probability_setting(min_confidence, 'min_confidence')
        self.min_text_mass = _probability_setting(min_text_mass, 'min_text_mass')
        self.min_margin = _probability_setting(min_margin, 'min_margin')
        self._known_indices = set()
        self._last_index = None
        self._peak_current_support = 0.0
        self._peak_text_mass = 0.0
        self._last_end = None
        self._duration = None

    def update(
        self,
        probabilities: Iterable[float],
        *,
        start: float,
        end: float,
        token_indices: Iterable[int] | None = None,
    ) -> list[dict]:
        if self._duration is not None:
            raise ValueError('The attention path is already finalized')
        start, end = _time(start, 'start'), _time(end, 'end')
        if end < start or (self._last_end is not None and start < self._last_end - 1e-9):
            raise ValueError('Audio intervals must be nonoverlapping and ordered')
        values = [float(value) for value in probabilities]
        indices = (
            list(range(len(values))) if token_indices is None else [_index(i, 'token_indices') for i in token_indices]
        )
        if len(values) != len(indices) or len(set(indices)) != len(indices):
            raise ValueError('probabilities and distinct token_indices must have matching lengths')
        if end == start:
            return []
        self._last_end = end
        valid = (
            bool(values)
            and all(math.isfinite(value) and 0 <= value <= 1 for value in values)
            and math.fsum(values) <= 1 + 1e-5
        )
        observed = dict(zip(indices, values)) if valid else {}
        self._known_indices.update(observed)
        text_mass = math.fsum(observed.values())
        eligible = {
            index for index, probability in observed.items() if probability > 0 and probability >= self.min_confidence
        }
        if text_mass < self.min_text_mass:
            eligible.clear()
        if self.min_margin > 0 and eligible:
            ranked = sorted(observed, key=lambda index: (-observed[index], index))
            best = ranked[0]
            runner_up = observed[ranked[1]] if len(ranked) > 1 else 0.0
            eligible.intersection_update([best] if observed[best] - runner_up >= self.min_margin else [])
        if eligible:
            self._peak_text_mass = max(self._peak_text_mass, text_mass)
        ranks = {index: rank for rank, index in enumerate(sorted(self._known_indices))}
        current_rank = ranks[self._last_index] if self._last_index is not None else -1
        emissions = {index: observed[index] for index in eligible if ranks[index] >= current_rank}
        if not emissions:
            return [self._decision(None, observed, start, end, end)]
        selected = max(
            emissions,
            key=lambda index: (
                math.log(emissions[index]) - self.skip_penalty * max(0, ranks[index] - current_rank - 1),
                -index,
            ),
        )
        forced = False
        force_allowed = self.force_progress and text_mass >= self.progress_threshold * self._peak_text_mass
        if force_allowed and self._last_index is not None:
            current_support = observed.get(self._last_index, 0.0) / text_mass
            forward = [index for index in emissions if ranks[index] > current_rank]
            if forward:
                ahead = max(forward, key=lambda index: (emissions[index], -index))
                ahead_support = emissions[ahead] / text_mass
                threshold = self.progress_threshold
                if (
                    current_support <= threshold * self._peak_current_support
                    and ahead_support > current_support
                    and ahead_support >= threshold * self._peak_current_support
                    and ranks[ahead] > ranks[selected]
                ):
                    selected, forced = ahead, True
        crossed = []
        if force_allowed and self._last_index is not None:
            crossed = sorted(index for index in emissions if current_rank < ranks[index] < ranks[selected])
        assigned = crossed + [selected]
        forced = forced or bool(crossed)
        # Current-frame subdivision is approximate. Mark the selected endpoint
        # as forced too, since its boundary depends on the same subdivision.
        step = (end - start) / len(assigned)
        decisions = [
            self._decision(
                index,
                observed,
                start + offset * step,
                end if offset == len(assigned) - 1 else start + (offset + 1) * step,
                end,
                forced=forced,
            )
            for offset, index in enumerate(assigned)
        ]
        support = emissions[selected] / text_mass
        self._peak_current_support = (
            max(self._peak_current_support, support) if selected == self._last_index else support
        )
        self._last_index = selected
        return decisions

    def finalize(self, duration: float) -> list[dict]:
        duration = _time(duration, 'duration')
        if self._duration is not None and abs(duration - self._duration) > 1e-9:
            raise ValueError('A finalized attention path cannot change duration')
        if self._last_end is not None and duration < self._last_end - 1e-9:
            raise ValueError('duration cannot precede an observed audio interval')
        self._duration = duration
        return []

    @staticmethod
    def _decision(index, probabilities, start, end, emitted_at, forced=False):
        probability = probabilities.get(index, 0.0) if index is not None else 0.0
        alternatives = [p for key, p in probabilities.items() if key != index]
        return {
            'token_index': index,
            'probability': probability,
            'forced': forced,
            'text_attention_mass': math.fsum(probabilities.values()),
            'margin': probability - max(alternatives, default=0.0),
            'start': start,
            'end': end,
            'emitted_at': emitted_at,
        }


def build_token_timestamps(
    trace,
    tokenizer=None,
    start_time=0.0,
    end_time=None,
    turn_index=None,
    expected_input_text=None,
    expected_input_token_ids=None,
):
    """Clip timestamp records to a saved waveform without manufacturing boundaries.

    ``trace`` is one batch item's timestamp output. Times are relative to its
    returned waveform. A multiturn caller selects a turn and crops that waveform;
    all saved times are then rebased to the cropped audio. Requested but unseen
    input occurrences remain present with null times and ``status=unobserved``.
    Heuristically forced spans have zero ``confidence``; ``attention_confidence``
    retains their underlying attention probability for debugging. Any retained
    forced span makes the containing token forced, even if its primary span has
    direct attention support. Frame records preserve each causal decoder query,
    including its instantaneous score and availability time. These scores are
    not calibrated correctness odds.
    """
    if trace is None:
        raise ValueError('Timestamp recording was requested but the model returned no timestamps')
    duration = float(trace['audio_duration_seconds'])
    if not math.isfinite(duration) or duration < 0:
        raise ValueError('Timestamp audio duration must be finite and nonnegative')
    start_time = float(start_time)
    end_time = duration if end_time is None else float(end_time)
    if not math.isfinite(start_time) or not math.isfinite(end_time):
        raise ValueError('Timestamp crop times must be finite')
    start_time = max(0.0, min(start_time, duration))
    end_time = max(start_time, min(end_time, duration))
    result = {
        key: copy.deepcopy(value)
        for key, value in trace.items()
        if key not in ('timestamps', 'turns', 'events', 'frames')
    }
    result.update(
        schema_version=1,
        source='attention',
        timings_are_estimates=True,
        timing_kind='attention_estimate',
        audio_duration_seconds=end_time - start_time,
        audio_offset_seconds=start_time,
        source_audio_duration_seconds=duration,
        primary_span_rule='largest probability-duration support; all disjoint spans are retained',
        confidence_kind='mean_full_key_attention_probability',
        confidence_is_calibrated=False,
    )
    if turn_index is not None:
        result['source_turn_index'] = turn_index
    turns = [
        copy.deepcopy(turn)
        for turn in trace.get('turns', [])
        if turn_index is None or turn.get('turn_index', 0) == turn_index
    ]
    result['turns'] = turns
    records = [
        copy.deepcopy(token)
        for token in trace.get('timestamps', [])
        if turn_index is None or token.get('turn_index', 0) == turn_index
    ]
    timestamps = []
    for token in records:
        token.setdefault('turn_index', 0 if turn_index is None else turn_index)
        token.setdefault('observed', True)
        source_spans = token['spans']
        spans = []
        for span in source_spans:
            clipped = _crop_interval(span, start_time, end_time)
            if clipped is None:
                continue
            if clipped.get('clipped'):
                frame_step = float(trace.get('frame_step_seconds', 0.0))
                if 'frame_count' in clipped and frame_step > 0:
                    clipped['source_frame_count'] = clipped['frame_count']
                    clipped['frame_count'] = min(
                        clipped['frame_count'],
                        max(1, math.ceil((clipped['end'] - clipped['start']) / frame_step - 1e-8)),
                    )
            spans.append(clipped)
        token['spans'] = spans
        token['source'] = 'attention'
        if spans:
            primary = max(
                range(len(spans)),
                key=lambda index: (
                    float(spans[index].get('attention_confidence') or 0.0)
                    * (spans[index]['end'] - spans[index]['start']),
                    -spans[index]['start'],
                ),
            )
            forced = any(span['forced'] for span in spans)
            token.update(
                start=spans[primary]['start'],
                end=spans[primary]['end'],
                confidence=0.0 if forced else spans[primary].get('confidence'),
                attention_confidence=spans[primary].get('attention_confidence'),
                forced=forced,
                text_attention_mass=spans[primary].get('text_attention_mass', token.get('text_attention_mass')),
                primary_span_index=primary,
                ambiguous=len(spans) > 1,
                short_evidence=spans[primary].get('short_evidence', token.get('short_evidence', False)),
            )
        else:
            token.update(
                start=None,
                end=None,
                confidence=None,
                attention_confidence=None,
                forced=False,
                text_attention_mass=None,
                primary_span_index=None,
                ambiguous=False,
                short_evidence=False,
            )
            if source_spans:
                token['status'] = 'outside_audio_slice'
                token['reason'] = 'outside_audio_slice'
        token['token'] = _label(tokenizer, token['token_id'], token.get('token'))
        timestamps.append(token)
    if expected_input_token_ids is not None:
        expected = list(expected_input_token_ids)
        indices = {(token.get('turn_index', 0), token['token_index']): token for token in timestamps}
        selected_turns = {token.get('turn_index', 0) for token in timestamps}
        if turn_index is None and len(selected_turns) > 1:
            raise ValueError('Expected input IDs require selecting one turn for a multiturn timestamp trace')
        selected = turn_index if turn_index is not None else next(iter(selected_turns), 0)
        if any(token['token_index'] >= len(expected) for token in timestamps):
            raise ValueError('Timestamp trace contains positions beyond the requested input sequence')
        for index, token_id in enumerate(expected):
            token = indices.get((selected, index))
            if token is not None:
                if token['token_id'] != token_id:
                    raise ValueError(f'Timestamp token {index} does not match the requested input token')
                continue
            timestamps.append(
                {
                    'source': 'attention',
                    'turn_index': selected,
                    'token_index': index,
                    'token_id': token_id,
                    'token': _label(tokenizer, token_id),
                    'start': None,
                    'end': None,
                    'confidence': None,
                    'attention_confidence': None,
                    'forced': False,
                    'text_attention_mass': None,
                    'status': 'unobserved',
                    'reason': 'not_observed',
                    'observed': False,
                    'ambiguous': False,
                    'short_evidence': False,
                    'primary_span_index': None,
                    'spans': [],
                }
            )
        result['expected_input_token_ids'] = expected
        result['token_scope'] = 'requested_input'
    else:
        result['token_scope'] = 'observed_streaming_input'
    result['timestamps'] = sorted(timestamps, key=lambda token: (token.get('turn_index', 0), token['token_index']))
    if 'frames' in trace:
        result['frames'] = _crop_frames(trace, tokenizer, start_time, end_time, turn_index)
        result['source_frame_count'] = trace.get('frame_count', len(trace['frames']))
        history_retained = trace.get('frame_history_retained', True)
        if history_retained:
            result['frame_count'] = len(result['frames'])
        elif turn_index is not None or start_time != 0.0 or end_time != duration:
            result['frame_count'] = None
        result['frame_confidence_kind'] = 'instantaneous_full_key_attention_probability'
    if expected_input_text is not None:
        result['expected_input_text'] = expected_input_text
    return result


def save_token_timestamps(audio_path, trace, tokenizer=None, **kwargs):
    """Atomically save ``<audio stem>.timestamps.json`` and return its absolute path."""
    audio_path = Path(audio_path).resolve()
    destination = audio_path.with_suffix('.timestamps.json')
    record = build_token_timestamps(trace, tokenizer=tokenizer, **kwargs)
    record['predicted_audio_filepath'] = str(audio_path)
    temporary = destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(destination)
    return str(destination)


def resolve_timestamp_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate independent, opt-in timestamp recording configuration."""
    values = dict(
        enabled=False,
        layer='middle',
        head=0,
        confirmation_frames=2,
        min_confidence=0.0,
        min_text_mass=0.0,
        min_margin=0.0,
        method='monotonic',
        force_progress=True,
        progress_threshold=0.5,
        skip_penalty=6.0,
        retain_frame_history=True,
    )
    if config is not None and not isinstance(config, Mapping):
        raise ValueError('timestamp_config must be a mapping')
    unknown = set(config or {}) - values.keys()
    if unknown:
        raise ValueError(f'Unknown timestamp_config options: {sorted(unknown)}')
    values.update(config or {})
    if type(values['enabled']) is not bool:
        raise ValueError('timestamp_config.enabled must be a boolean')
    if type(values['retain_frame_history']) is not bool:
        raise ValueError('timestamp_config.retain_frame_history must be a boolean')
    if values['layer'] != 'middle' and (type(values['layer']) is not int or values['layer'] < 0):
        raise ValueError('timestamp_config.layer must be middle or a nonnegative main-backbone attention ordinal')
    if type(values['head']) is not int or values['head'] < 0:
        raise ValueError('timestamp_config.head must be a nonnegative integer')
    AttentionTimestampTracker(
        **{
            key: value
            for key, value in values.items()
            if key not in ('enabled', 'layer', 'head', 'retain_frame_history')
        }
    )
    return values


class StreamingTimestampSession:
    """Per-stream timestamp state, independent from inference attention policy.

    ``frame_index`` counts emitted decoder audio steps; ``codec_frame_start``
    and ``codec_frame_end`` identify the covered unstacked codec frames. Codec
    indices and times are relative to the first generated audio frame, matching
    ``streaming_finalize``. ``audio_lengths`` supplied to ``snapshots`` are seconds.
    """

    def __init__(
        self,
        batch_size: int,
        codec_frame_seconds: float,
        frame_stacking_factor: int,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        if not math.isfinite(codec_frame_seconds) or codec_frame_seconds <= 0:
            raise ValueError('codec_frame_seconds must be finite and positive')
        if type(frame_stacking_factor) is not int or frame_stacking_factor < 1:
            raise ValueError('frame_stacking_factor must be positive')
        self.batch_size = batch_size
        self.codec_frame_seconds = float(codec_frame_seconds)
        self.frame_stacking_factor = frame_stacking_factor
        self.config = resolve_timestamp_config(config)
        self._turns = [[self._new_turn(0)] for _ in range(batch_size)]
        self._origins = [None] * batch_size
        self._ends = [None] * batch_size
        self._events = []
        self._frames = []
        self._frame_history = [[] for _ in range(batch_size)]
        self._frame_counts = [0] * batch_size
        self._raw_attention = None

    def begin_step(self) -> None:
        """Clear the previous query before a new decoding step begins."""
        self._raw_attention = None

    def register_text(self, row: int, key_position: int, token_id: int, token: str | None = None) -> None:
        """Register a source token occurrence at its current decoder key position."""
        self._validate_row(row)
        turn = self._turns[row][-1]
        if turn['finalized']:
            raise ValueError('Reset the timestamp turn before registering more text')
        if type(key_position) is not int or key_position < 0:
            raise ValueError('Text key position must be a nonnegative integer')
        if turn['keys'] and key_position <= turn['keys'][-1]:
            raise ValueError('Text key positions must increase within each turn')
        index = len(turn['keys'])
        turn['tracker'].register_token(index, token_id, token)
        turn['keys'].append(key_position)
        turn['ids'].append(token_id)

    def reset_turn(self, row: int) -> None:
        """Finalize a response and start its next source-token sequence."""
        self._validate_row(row)
        turn = self._turns[row][-1]
        if not turn['keys'] and not turn['has_audio']:
            return
        self._finish_turn(row)
        self._turns[row].append(self._new_turn(turn['turn_index'] + 1))

    def get_turn_indices(self) -> list[int]:
        """Return the current response ordinal for each conditional batch row."""
        return [turns[-1]['turn_index'] for turns in self._turns]

    def observe_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> None:
        """Observe expanded Q/K in ``[batch, heads, queries/keys, head_dim]``.

        CFG's unconditional rows are excluded. The full-key softmax is used;
        text attention is never renormalized independently of other channels.
        """
        head = self.config['head']
        if head >= query.size(1):
            raise ValueError('timestamp_config.head is outside the main-backbone attention heads')
        q = query[: self.batch_size, head, -1:, :].detach().float()
        k = key[: self.batch_size, head, :, :].detach().float()
        logits = (q @ k.transpose(-1, -2)).squeeze(1) / math.sqrt(query.size(-1))
        allowed = torch.ones_like(logits, dtype=torch.bool)
        if cache_position is not None:
            position = cache_position.reshape(-1)[-1]
            allowed &= torch.arange(k.size(-2), device=k.device)[None, :] <= position
        if attention_mask is not None:
            mask = attention_mask[: self.batch_size]
            if mask.dim() == 4:
                mask = mask[:, head if mask.size(1) > 1 else 0, -1, : k.size(-2)]
                if mask.dtype == torch.bool:
                    allowed &= mask
                else:
                    logits = logits + mask.float()
                    allowed &= mask > torch.finfo(mask.dtype).min / 2
            elif mask.dim() == 2:
                allowed &= mask[:, : k.size(-2)].bool()
            else:
                raise ValueError('Timestamp observer requires a 2D padding mask or 4D attention mask')
        logits = logits.masked_fill(~allowed, -torch.inf)
        self._raw_attention = torch.softmax(logits, dim=-1).nan_to_num(0.0).cpu()

    def observe_audio(
        self,
        needs_audio: torch.Tensor | Sequence[bool],
        frame_start: int,
        valid_frames: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        """Publish current-frame events immediately, after clipping audio EOS."""
        needed_rows = _values(needs_audio)
        counts = None if valid_frames is None else _values(valid_frames)
        if len(needed_rows) != self.batch_size or (counts is not None and len(counts) != self.batch_size):
            raise ValueError('Audio masks and frame counts must match the conditional batch size')
        intervals = []
        for row, needed in enumerate(needed_rows):
            if not needed:
                continue
            count = self.frame_stacking_factor if counts is None else counts[row]
            self._validate_audio_interval(row, frame_start, count)
            intervals.append((row, count))
        for row, count in intervals:
            self.observe_row(row, frame_start, count)
        self._raw_attention = None

    def observe_row(self, row: int, frame_start: int, valid_frames: int) -> None:
        """Record one emitted audio interval on the absolute unstacked codec grid."""
        self._validate_row(row)
        self._validate_audio_interval(row, frame_start, valid_frames)
        if self._origins[row] is None:
            self._origins[row] = frame_start
        self._ends[row] = frame_start + valid_frames
        if valid_frames <= 0:
            return
        turn = self._turns[row][-1]
        turn['has_audio'] = True
        raw = self._raw_attention[row].tolist() if self._raw_attention is not None else []
        positions = [(index, position) for index, position in enumerate(turn['keys']) if position < len(raw)]
        turn['tracker'].update(
            [raw[position] for _, position in positions],
            token_indices=[index for index, _ in positions],
            start=(frame_start - self._origins[row]) * self.codec_frame_seconds,
            end=(self._ends[row] - self._origins[row]) * self.codec_frame_seconds,
        )
        self._collect(row, turn)
        for record in turn['tracker'].drain_frames():
            for assignment in record['assignments']:
                index = assignment['token_index']
                assignment['text_key_position'] = turn['keys'][index] if index is not None else None
            index = record['token_index']
            frame = dict(
                record,
                text_key_position=turn['keys'][index] if index is not None else None,
                batch_index=row,
                turn_index=turn['turn_index'],
                turn_frame_index=record['frame_index'],
                frame_index=self._frame_counts[row],
                codec_frame_start=frame_start - self._origins[row],
                codec_frame_end=self._ends[row] - self._origins[row],
                valid_codec_frames=valid_frames,
            )
            self._frame_counts[row] += 1
            self._frames.append(frame)
            if self.config['retain_frame_history']:
                self._frame_history[row].append(deepcopy(frame))

    def drain_events(self, batch_indices: Iterable[int] | None = None) -> list[dict[str, Any]]:
        """Consume token-span events for the selected conditional batch rows."""
        self._events, events = self._drain(self._events, batch_indices)
        return events

    def drain_frames(self, batch_indices: Iterable[int] | None = None) -> list[dict[str, Any]]:
        """Consume current-step decisions, independently of token-span events.

        A record is emitted for every positive-duration audio step, including
        steps with unavailable or insufficient attention evidence. History in
        ``snapshots`` remains available after draining when retention is enabled.
        """
        self._frames, frames = self._drain(self._frames, batch_indices)
        return frames

    def snapshots(
        self, finalize: bool = False, audio_lengths: torch.Tensor | Sequence[float] | None = None
    ) -> list[dict[str, Any]]:
        """Return independent traces; optional waveform lengths are in seconds."""
        results = []
        lengths = _values(audio_lengths) if audio_lengths is not None else None
        if lengths is not None:
            if len(lengths) != self.batch_size:
                raise ValueError('Audio lengths must match the conditional batch size')
            if any(not math.isfinite(length) or length < 0 for length in lengths):
                raise ValueError('Audio lengths must be finite and nonnegative')
        for row, turns in enumerate(self._turns):
            if finalize:
                self._finish_turn(row)
            duration = self._duration(row) if lengths is None else float(lengths[row])
            results.append(
                dict(
                    schema_version=1,
                    source='attention',
                    timings_are_estimates=True,
                    confidence_is_calibrated=False,
                    guidance_enabled=False,
                    causal=True,
                    audio_duration_seconds=duration,
                    frame_step_seconds=self.frame_stacking_factor * self.codec_frame_seconds,
                    attention_layer=self.config['layer'],
                    attention_head=self.config['head'],
                    codec_frame_seconds=self.codec_frame_seconds,
                    frame_count=self._frame_counts[row],
                    frame_history_retained=self.config['retain_frame_history'],
                    frames=deepcopy(self._frame_history[row]),
                    timestamps=[
                        dict(record, turn_index=turn['turn_index'])
                        for turn in turns
                        for record in turn['tracker'].snapshot()
                    ],
                    turns=[
                        dict(
                            turn_index=turn['turn_index'],
                            source_token_ids=list(turn['ids']),
                            source_key_positions=list(turn['keys']),
                        )
                        for turn in turns
                    ],
                )
            )
        return results

    def _new_turn(self, turn_index: int) -> dict[str, Any]:
        settings = {
            key: value
            for key, value in self.config.items()
            if key not in ('enabled', 'layer', 'head', 'retain_frame_history')
        }
        return dict(
            turn_index=turn_index,
            tracker=AttentionTimestampTracker(record_frames=True, **settings),
            keys=[],
            ids=[],
            finalized=False,
            has_audio=False,
        )

    def _duration(self, row: int) -> float:
        if self._origins[row] is None or self._ends[row] is None:
            return 0.0
        return max(0, self._ends[row] - self._origins[row]) * self.codec_frame_seconds

    def _finish_turn(self, row: int) -> None:
        turn = self._turns[row][-1]
        if not turn['finalized']:
            turn['tracker'].finalize(self._duration(row))
            turn['finalized'] = True
            self._collect(row, turn)

    def _collect(self, row: int, turn: dict[str, Any]) -> None:
        self._events.extend(
            dict(event, batch_index=row, turn_index=turn['turn_index']) for event in turn['tracker'].drain_events()
        )

    def _drain(self, queue, batch_indices):
        if batch_indices is None:
            return [], queue
        selected = set(batch_indices)
        for row in selected:
            self._validate_row(row)
        return (
            [record for record in queue if record['batch_index'] not in selected],
            [record for record in queue if record['batch_index'] in selected],
        )

    def _validate_row(self, row: int) -> None:
        if type(row) is not int or not 0 <= row < self.batch_size:
            raise ValueError('Timestamp batch index is outside the conditional batch')

    def _validate_audio_interval(self, row: int, frame_start: int, valid_frames: int) -> None:
        if self._turns[row][-1]['finalized']:
            raise ValueError('Reset the timestamp turn before observing more audio')
        if type(frame_start) is not int or frame_start < 0:
            raise ValueError('Audio frame start must be a nonnegative integer')
        if type(valid_frames) is not int or not 0 <= valid_frames <= self.frame_stacking_factor:
            raise ValueError('Valid audio frames must be an integer between zero and the frame stacking factor')
        if self._ends[row] is not None and frame_start < self._ends[row]:
            raise ValueError('Audio intervals must be nonoverlapping and ordered')


def _alignment_chunk(
    query,
    key,
    value,
    rows,
    text_keys,
    centers,
    mask,
    prior_strength,
    band_width,
    dropout_p,
    look_back_tokens,
    look_ahead_tokens,
):
    with torch.autocast(device_type=query.device.type, enabled=False):
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) / math.sqrt(query.size(-1))
        causal = torch.arange(key.size(1), device=query.device)[None, None, :] <= rows[None, :, None]
        visible = causal & _mask_visibility(mask)
        if mask.dtype != torch.bool:
            scores = scores + mask.float()
        scores = scores.masked_fill(~visible, -torch.inf)
        log_probs = F.log_softmax(scores, dim=-1)

        token_distance = torch.arange(text_keys.numel(), device=query.device)[None, :] - centers[:, None]
        distance = (token_distance - look_ahead_tokens).clamp_min(0) + (-token_distance - look_back_tokens).clamp_min(
            0
        )
        target = torch.exp(-0.5 * (distance.float() / band_width).square())
        text_visible = visible.expand(query.size(0), -1, -1).index_select(-1, text_keys)
        target = target[None, :, :] * text_visible
        target = (target / target.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)).detach()
        text_log_probs = log_probs.index_select(-1, text_keys)
        loss = -(target * torch.where(target > 0, text_log_probs, 0.0)).sum()

        if not prior_strength:
            return loss, query.new_empty((0,))
        prior = torch.full_like(scores, 1e-6)
        prior = prior.scatter(-1, text_keys[None, None, :].expand_as(target), target.clamp_min(1e-6))
        prior = (1.0 - prior_strength) + prior_strength * prior
        probabilities = F.softmax(scores + prior.log(), dim=-1)
        probabilities = F.dropout(probabilities, p=dropout_p, training=dropout_p > 0)
        guided_output = torch.matmul(probabilities, value.float()).to(dtype=query.dtype)
        return loss, guided_output


def _validate_turn(turn, query, query_length, batch_size):
    if type(turn.batch_index) is not int or not 0 <= turn.batch_index < batch_size:
        raise ValueError('Attention alignment batch_index is out of range')
    for field in (turn.query_positions, turn.key_positions, turn.token_indices):
        if field.ndim != 1 or field.dtype not in (torch.int32, torch.int64) or field.device != query.device:
            raise ValueError(
                'Attention alignment positions and indices must be 1D integer tensors on the query device'
            )
    if turn.query_positions.shape != turn.token_indices.shape:
        raise ValueError('Each guided query requires one teacher token index')
    for positions in (turn.query_positions, turn.key_positions):
        if positions.numel() and (
            torch.any(positions < 0)
            or torch.any(positions >= query_length)
            or torch.any(positions[1:] <= positions[:-1])
        ):
            raise ValueError(
                'Attention alignment positions must be strictly increasing and within the decoder timeline'
            )


def _select_attention_mask(mask, batch_index, heads, rows, key_length):
    if mask.ndim == 2:
        return mask[batch_index, :key_length].bool()[None, None, :].expand(heads.numel(), rows.numel(), -1)
    batch_mask = mask[batch_index if mask.size(0) > 1 else 0]
    if batch_mask.size(0) == 1:
        batch_mask = batch_mask.expand(heads.numel(), -1, -1)
    else:
        batch_mask = batch_mask.index_select(0, heads)
    if batch_mask.size(1) == 1:
        return batch_mask[:, :, :key_length].expand(-1, rows.numel(), -1)
    return batch_mask.index_select(1, rows)[:, :, :key_length]


def _select_center_mask(mask, batch_index, heads, rows, centers):
    if mask.ndim == 2:
        return mask[batch_index, centers].bool()[None, :].expand(heads.numel(), -1)
    selected_batch = batch_index if mask.size(0) > 1 else 0
    selected_heads = heads if mask.size(1) > 1 else torch.zeros_like(heads)
    selected_rows = rows if mask.size(2) > 1 else torch.zeros_like(rows)
    return mask[selected_batch, selected_heads[:, None], selected_rows[None, :], centers[None, :]]


def _mask_visibility(mask):
    if mask.dtype == torch.bool:
        return mask
    if not mask.is_floating_point():
        return mask != 0
    return torch.isfinite(mask) & (mask > torch.finfo(mask.dtype).min / 2)


def _index(value, name, minimum=0):
    try:
        result = operator.index(value)
    except TypeError:
        raise ValueError(f'{name} must be an integer >= {minimum}') from None
    if isinstance(value, bool) or result < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return result


def _probability_setting(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f'{name} must be finite and in [0, 1]')
    return float(value)


def _nonnegative_setting(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return float(value)


def _time(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be a finite nonnegative time')
    return float(value)


def _crop_frames(trace, tokenizer, start_time, end_time, turn_index):
    frames = []
    codec_step = float(trace.get('codec_frame_seconds', 0.0))
    for original in trace['frames']:
        if turn_index is not None and original.get('turn_index', 0) != turn_index:
            continue
        frame = _crop_interval(original, start_time, end_time)
        if frame is None:
            continue
        frame['source_frame_index'] = original['frame_index']
        frame['frame_index'] = len(frames)
        for name in ('codec_frame_start', 'codec_frame_end', 'valid_codec_frames'):
            if name in original:
                frame[f'source_{name}'] = original[name]
        # Seconds remain exact for arbitrary waveform crops. Integer codec
        # indices describe only intervals whose boundaries lie on that grid.
        left = _codec_grid_index(frame['start'], codec_step)
        right = _codec_grid_index(frame['end'], codec_step)
        if _codec_grid_index(start_time, codec_step) is None or left is None or right is None:
            frame.update(codec_frame_start=None, codec_frame_end=None, valid_codec_frames=None)
        else:
            frame.update(codec_frame_start=left, codec_frame_end=right, valid_codec_frames=right - left)
        assignments = []
        for assignment in original.get('assignments', []):
            clipped = _crop_interval(assignment, start_time, end_time)
            if clipped is not None:
                _label_frame(clipped, tokenizer)
                assignments.append(clipped)
        if 'assignments' in original:
            frame['assignments'] = assignments
        if assignments:
            # A partial crop may remove the source query's final subdivision.
            # Its retained endpoint is then the last surviving assignment.
            for name in (
                'token_index',
                'token_id',
                'token',
                'text_key_position',
                'confidence',
                'attention_confidence',
                'forced',
                'text_attention_mass',
                'margin',
                'status',
            ):
                if name in assignments[-1]:
                    frame[name] = assignments[-1][name]
        _label_frame(frame, tokenizer)
        frames.append(frame)
    return frames


def _crop_interval(record, start_time, end_time):
    left, right = max(float(record['start']), start_time), min(float(record['end']), end_time)
    if right <= left:
        return None
    clipped = copy.deepcopy(record)
    clipped.update(start=left - start_time, end=right - start_time)
    if left != record['start'] or right != record['end']:
        clipped['clipped'] = True
    if record.get('emitted_at') is not None:
        clipped['emitted_at'] = float(record['emitted_at']) - start_time
    if clipped.get('forced', False):
        clipped['confidence'] = 0.0
    return clipped


def _codec_grid_index(seconds, step):
    if not math.isfinite(step) or step <= 0:
        return None
    value = seconds / step
    rounded = round(value)
    return rounded if abs(value - rounded) <= 1e-7 else None


def _label_frame(record, tokenizer):
    if record.get('token_id') is not None:
        record['token'] = _label(tokenizer, record['token_id'], record.get('token'))


def _label(tokenizer, token_id, existing=None):
    if existing is not None:
        return existing
    return _token_label(tokenizer, token_id) if tokenizer is not None else str(token_id)


def _token_label(tokenizer, token_id):
    components = getattr(tokenizer, 'tokenizers', None)
    if components is not None:
        for name, component in components.items():
            offset = tokenizer.tokenizer_offsets[name]
            if offset <= token_id < offset + tokenizer.num_tokens_per_tokenizer[name]:
                tokenizer, token_id = component, token_id - offset
                break
        else:
            return str(token_id)
    convert = getattr(tokenizer, 'convert_ids_to_tokens', None)
    if callable(convert):
        return str(convert(token_id))
    tokens = getattr(tokenizer, 'tokens', None)
    if tokens is not None and 0 <= token_id < len(tokens):
        return str(tokens[token_id])
    return str(token_id)


def _values(value: torch.Tensor | Sequence[Any]) -> list[Any]:
    return value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else list(value)
