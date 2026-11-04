# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming token timing estimates from attention, independent of any decoder.

The tracker consumes probabilities already computed by a model; it never changes
attention, text inputs or audio outputs. Callers register text occurrences as they
arrive, then supply the valid audio interval for each attention query. Probabilities
are the text slice of the ordinary full-key softmax, without text renormalization.

The causal monotonic decoder uses only the current posterior and past progress.
It can approximate crossed weak positions when forward evidence exposes a stall;
such estimates are explicitly forced and have zero confidence. Confirmation
tracks consecutive evidence while retaining short initial and final tokens.
Each token retains its observed contiguous spans and uses the most supported
span as its main start/end interval.

These are model-derived estimates, not independent recognition or forced-aligned
word boundaries. Timestamp resolution and accuracy depend on the supplied model
attention and audio frame intervals.
"""

import copy
import math
import operator
from collections.abc import Iterable


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
