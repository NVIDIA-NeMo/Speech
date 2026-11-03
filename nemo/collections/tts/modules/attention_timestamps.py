# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming token timing estimates from attention, independent of any decoder.

The tracker consumes probabilities already computed by a model; it never changes
attention, text inputs or audio outputs. Callers register text occurrences as they
arrive, then supply the valid audio interval for each attention query. Probabilities
are the text slice of the ordinary full-key softmax, without text renormalization.

The optional monotonic decoder uses a bounded lookahead over full posteriors to
recover weak positions that never win framewise argmax. The argmax alternative
allows local backward revisits. Two-frame confirmation rejects isolated
A -> B -> A excursions while retaining one-frame tokens along forward progress,
and short initial/final evidence. Events are provisional until confirmed or
ended; retractions explicitly remove rejected excursions. A token's main
start/end pair describes its most supported contiguous span, rather than a
potentially misleading first-to-last envelope over disjoint visits.

These are model-derived estimates, not independent recognition or forced-aligned
word boundaries. Timestamp resolution and accuracy depend on the supplied model
attention and audio frame intervals.
"""

import copy
import math
import operator


class AttentionTimestampTracker:
    """Convert streaming attention into timestamp records and incremental events.

    Args:
        confirmation_frames: Consecutive frames confirming a new position. One
            disables excursion filtering. Short forward runs are still retained
            as ``short_evidence`` when this value is greater than one.
        min_confidence: Minimum probability of the selected visible text key,
            measured in the original full-key attention distribution.
        min_text_mass: Minimum attention mass on registered visible text keys.
        min_margin: Minimum selected-key probability minus the strongest other
            visible key. Zero disables this gate in monotonic mode, permitting
            weak positions recovered from the sequence posterior.
        method: ``argmax`` for framewise selection with local backward revisits,
            or ``monotonic`` for a fixed-lag forward-only posterior path.
        lookahead_frames: Query-frame delivery lag of the monotonic decoder.
            This does not delay or otherwise change generated audio.
        skip_penalty: Log-score cost per skipped text occurrence. Skips allow
            zero-evidence positions to stay unaligned instead of stalling.

    ``update`` returns newly generated events and also enqueues them for
    ``drain_events``. Clients should use either delivery mechanism, not both.
    Drain the event queue regularly during long streams. Only aggregated spans,
    one pending candidate and one active candidate are retained. Monotonic mode
    additionally retains a bounded ``lookahead_frames`` posterior buffer and
    one score per reachable position; no full attention history is stored.
    """

    def __init__(
        self,
        confirmation_frames: int = 2,
        min_confidence: float = 0.0,
        min_text_mass: float = 0.0,
        min_margin: float = 0.0,
        method: str = 'argmax',
        lookahead_frames: int = 4,
        skip_penalty: float = 6.0,
    ):
        self.confirmation_frames = _index(confirmation_frames, 'confirmation_frames', minimum=1)
        self.min_confidence = _probability_setting(min_confidence, 'min_confidence')
        self.min_text_mass = _probability_setting(min_text_mass, 'min_text_mass')
        self.min_margin = _probability_setting(min_margin, 'min_margin')
        if method not in ('argmax', 'monotonic'):
            raise ValueError('method must be argmax or monotonic')
        self.method = method
        self.lookahead_frames = _index(lookahead_frames, 'lookahead_frames')
        self.skip_penalty = _nonnegative_setting(skip_penalty, 'skip_penalty')
        self._path_decoder = (
            StreamingAttentionPathDecoder(lookahead_frames=self.lookahead_frames, skip_penalty=self.skip_penalty)
            if method == 'monotonic'
            else None
        )
        self._tokens = {}
        self._active = None
        self._pending = None
        self._previous_index = None
        self._previous_confirmed = False
        self._last_end = None
        self._input_end = None
        self._duration = None
        self._events = []
        self._next_span_id = 0

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
            'rejected_frames': 0,
            'retracted_spans': 0,
        }

    def update(self, probabilities, *, start: float, end: float, token_indices=None) -> list[dict]:
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
        if self._path_decoder is not None:
            decisions = self._path_decoder.update(
                [value for value, _ in visible] if visible else [],
                token_indices=[index for _, index in visible] if visible else [],
                start=start,
                end=end,
            )
            self._consume_decisions(decisions)
        else:
            self._apply_candidate(self._candidate(visible), start, end, end)
        return copy.deepcopy(self._events[event_start:])

    def _apply_candidate(self, candidate, start, end, emitted_at):
        if self._last_end is not None and start > self._last_end + 1e-9:
            self._end_pending(emitted_at, retain=True)
            self._end_active(emitted_at)
            self._previous_index = None
            self._previous_confirmed = False
        self._last_end = end
        if candidate is None:
            self._end_pending(emitted_at, retain=True)
            self._end_active(emitted_at)
            self._previous_index = None
            self._previous_confirmed = False
            return
        index, confidence, text_mass = candidate
        if self._active is not None:
            if self._active['token_index'] == index:
                self._extend(self._active, end, confidence, text_mass)
                self._emit('update', self._active, emitted_at)
                return
            self._end_active(emitted_at)
        if self._pending is not None:
            if self._pending['token_index'] == index:
                self._extend(self._pending, end, confidence, text_mass)
                if self._pending['frame_count'] >= self.confirmation_frames:
                    self._active, self._pending = self._pending, None
                    self._active['confirmed'] = True
                    self._emit('confirmed', self._active, emitted_at)
                else:
                    self._emit('update', self._pending, emitted_at)
                return
            if index == self._previous_index:
                # A single-frame excursion is not evidence that B occupied all
                # of the gap. Keep A's two observed spans separate as well.
                resume_confirmed = self._previous_confirmed
                self._end_pending(emitted_at, retain=False)
                self._begin(
                    index, start, end, confidence, text_mass, confirmed=resume_confirmed, emitted_at=emitted_at
                )
                return
            forward = self._previous_index is None or self._previous_index < self._pending['token_index'] < index
            self._end_pending(emitted_at, retain=forward)
        self._begin(index, start, end, confidence, text_mass, emitted_at=emitted_at)

    def update_many(self, probabilities, *, starts, ends, token_indices=None) -> list[dict]:
        """Consume a chunk of queries with the same registered text-key mapping.

        This is equivalent to consecutive ``update`` calls, so chunk size cannot
        change boundaries or confirmation latency. Future keys in a dense chunk
        must retain zero probability until they are causally available.
        """
        rows, starts, ends = list(probabilities), list(starts), list(ends)
        if len(rows) != len(starts) or len(rows) != len(ends):
            raise ValueError('Attention rows, starts and ends must have matching lengths')
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
            for live, live_status in ((self._active, 'active'), (self._pending, 'provisional')):
                if live is not None and live['token_index'] == index:
                    spans.append(self._span_record(live, live_status))
                    status = live_status
            if self._duration is not None and spans:
                status = 'finalized'
            primary_index = max(range(len(spans)), key=lambda i: spans[i]['support']) if spans else None
            primary = spans[primary_index] if primary_index is not None else None
            result.append(
                {
                    'source': 'attention',
                    'method': self.method,
                    'token_index': index,
                    'token_id': token['token_id'],
                    'token': token['token'],
                    'start': primary['start'] if primary else None,
                    'end': primary['end'] if primary else None,
                    'confidence': primary['confidence'] if primary else None,
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
        return copy.deepcopy(events)

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
        if self._path_decoder is not None:
            self._consume_decisions(self._path_decoder.finalize(duration))
        self._end_pending(duration, retain=True)
        self._end_active(duration)
        self._duration = duration
        return self.snapshot()

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

    def _candidate(self, visible):
        if not visible:
            return None
        text_mass = math.fsum(value for value, _ in visible)
        ranked = sorted(visible, key=lambda item: (-item[0], item[1]))
        confidence, index = ranked[0]
        runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
        if confidence <= 0 or text_mass <= 0:
            return None
        self._tokens[index]['candidate_frames'] += 1
        if (
            confidence < self.min_confidence
            or text_mass < self.min_text_mass
            or confidence - runner_up < self.min_margin
        ):
            self._tokens[index]['rejected_frames'] += 1
            return None
        return index, confidence, text_mass

    def _consume_decisions(self, decisions):
        for decision in decisions:
            index = decision['token_index']
            confidence = decision['probability']
            text_mass = decision['text_attention_mass']
            candidate = None
            if index is not None:
                self._tokens[index]['candidate_frames'] += 1
                if (
                    confidence >= self.min_confidence
                    and text_mass >= self.min_text_mass
                    and (self.min_margin == 0 or decision['margin'] >= self.min_margin)
                ):
                    candidate = index, confidence, text_mass
                else:
                    self._tokens[index]['rejected_frames'] += 1
            self._apply_candidate(candidate, decision['start'], decision['end'], decision['emitted_at'])

    def _begin(self, index, start, end, confidence, text_mass, confirmed=False, emitted_at=None):
        span = {
            'span_id': self._next_span_id,
            'token_index': index,
            'start': start,
            'end': start,
            'probability_weight': 0.0,
            'mass_weight': 0.0,
            'frame_count': 0,
            'confirmed': confirmed or self.confirmation_frames == 1,
        }
        self._next_span_id += 1
        self._extend(span, end, confidence, text_mass)
        if span['confirmed']:
            self._active = span
        else:
            self._pending = span
        self._emit('start', span, end if emitted_at is None else emitted_at)

    @staticmethod
    def _extend(span, end, confidence, text_mass):
        duration = end - span['end']
        span['probability_weight'] += confidence * duration
        span['mass_weight'] += text_mass * duration
        span['end'] = end
        span['frame_count'] += 1

    def _end_active(self, emitted_at):
        if self._active is None:
            return
        span, self._active = self._active, None
        self._tokens[span['token_index']]['spans'].append(span)
        self._previous_index = span['token_index']
        self._previous_confirmed = span['confirmed']
        self._emit('end', span, emitted_at)

    def _end_pending(self, emitted_at, retain):
        if self._pending is None:
            return
        span, self._pending = self._pending, None
        if retain:
            self._tokens[span['token_index']]['spans'].append(span)
            self._previous_index = span['token_index']
            self._previous_confirmed = span['confirmed']
        else:
            self._tokens[span['token_index']]['retracted_spans'] += 1
        self._emit('end' if retain else 'retract', span, emitted_at)

    @staticmethod
    def _unaligned_reason(token):
        if not token['observed_frames']:
            return 'not_observed'
        if not token['max_probability']:
            return 'no_positive_attention'
        if token['retracted_spans']:
            return 'unconfirmed_excursion'
        if token['rejected_frames']:
            return 'below_threshold'
        return 'never_selected'

    @staticmethod
    def _span_record(span, status):
        duration = span['end'] - span['start']
        return {
            'span_id': span['span_id'],
            'start': span['start'],
            'end': span['end'],
            'confidence': span['probability_weight'] / duration,
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
        if event in ('end', 'retract'):
            status = 'finalized' if event == 'end' else 'retracted'
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
    """Fixed-lag Viterbi decoding of an observed attention posterior.

    The path may stay, advance to the next input position, or skip positions at
    ``skip_penalty`` cost per skipped position. Emissions use the actual global
    attention probabilities. A zero-probability position is never assigned, and
    the final path is not forced to cover the end of the supplied transcript.

    Only the last ``lookahead_frames`` observations and their backpointers are
    retained, plus one score per reachable text position. Committed decisions
    constrain subsequent paths, so finalized positions never move backwards or
    change retrospectively. Increasing the lag can recover short weak positions
    that never win framewise argmax, at the cost of that many query frames of
    delivery latency. The decoder does not change model attention or synthesis.
    """

    def __init__(self, lookahead_frames: int = 4, skip_penalty: float = 6.0):
        self.lookahead_frames = _index(lookahead_frames, 'lookahead_frames')
        self.skip_penalty = _nonnegative_setting(skip_penalty, 'skip_penalty')
        self._scores = {}
        self._known_indices = set()
        self._frames = []
        self._last_end = None
        self._duration = None

    @property
    def buffered_frames(self) -> int:
        return len(self._frames)

    @property
    def buffered_state_count(self) -> int:
        """Number of retained scalar scores/backpointers/probabilities."""
        return len(self._scores) + sum(len(frame['parents']) + len(frame['probabilities']) for frame in self._frames)

    def update(self, probabilities, *, start: float, end: float, token_indices=None) -> list[dict]:
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
        emissions = {index: probability for index, probability in observed.items() if probability > 0}
        scores, parents = self._advance(emissions)
        # A gap or a posterior that only points backwards supplies no assignable
        # frame. Keep the previous progress state without filling the gap.
        if not scores:
            scores = dict(self._scores)
            parents = {index: index for index in scores}
            emissions = {}
        self._scores = _normalized_scores(scores)
        self._frames.append(
            {
                'start': start,
                'end': end,
                'parents': parents,
                'probabilities': observed,
                'assignable': bool(emissions),
            }
        )
        if len(self._frames) > self.lookahead_frames:
            return [self._commit(end)]
        return []

    def finalize(self, duration: float) -> list[dict]:
        duration = _time(duration, 'duration')
        if self._duration is not None:
            if abs(duration - self._duration) > 1e-9:
                raise ValueError('A finalized attention path cannot change duration')
            return []
        if self._last_end is not None and duration < self._last_end - 1e-9:
            raise ValueError('duration cannot precede an observed audio interval')
        decisions = []
        while self._frames:
            decisions.append(self._commit(duration))
        self._duration = duration
        return decisions

    def _advance(self, emissions):
        if not emissions:
            return {}, {}
        # Identity is the input occurrence, not its numeric decoder-key offset.
        # Adjacent supplied keys can be noncontiguous in the model sequence.
        ranks = {index: rank for rank, index in enumerate(sorted(self._known_indices))}
        if not self._scores:
            return (
                {index: math.log(p) - self.skip_penalty * ranks[index] for index, p in emissions.items()},
                {index: None for index in emissions},
            )
        previous = sorted(self._scores)
        previous_by_rank = {ranks[index]: index for index in previous}
        prefix_score, prefix_index, prefix_position = -math.inf, None, 0
        scores, parents = {}, {}
        for index, probability in sorted(emissions.items()):
            rank = ranks[index]
            while prefix_position < len(previous) and ranks[previous[prefix_position]] < rank - 1:
                candidate_index = previous[prefix_position]
                candidate_score = self._scores[candidate_index] + self.skip_penalty * ranks[candidate_index]
                if candidate_score > prefix_score:
                    prefix_score, prefix_index = candidate_score, candidate_index
                prefix_position += 1
            score, parent = self._scores.get(index, -math.inf), index
            advance_index = previous_by_rank.get(rank - 1)
            advance_score = self._scores.get(advance_index, -math.inf)
            if advance_score > score:
                score, parent = advance_score, advance_index
            skip_score = prefix_score - self.skip_penalty * (rank - 1)
            if skip_score > score:
                score, parent = skip_score, prefix_index
            if math.isfinite(score):
                scores[index] = score + math.log(probability)
                parents[index] = parent
        return scores, parents

    def _ancestor(self, endpoint):
        state = endpoint
        for frame in reversed(self._frames[1:]):
            state = frame['parents'].get(state) if state is not None else None
        return state

    def _commit(self, emitted_at):
        endpoint = max(self._scores, key=lambda index: (self._scores[index], -index)) if self._scores else None
        ancestor = self._ancestor(endpoint)
        # Retain only paths consistent with this irrevocable prefix decision.
        # Otherwise a later optimum could revise old history and cause a
        # backwards transition in already published timestamps.
        self._scores = {index: score for index, score in self._scores.items() if self._ancestor(index) == ancestor}
        self._scores = _normalized_scores(self._scores)
        frame = self._frames.pop(0)
        index = ancestor if frame['assignable'] and ancestor in frame['probabilities'] else None
        probability = frame['probabilities'].get(index, 0.0) if index is not None else 0.0
        if probability <= 0:
            index, probability = None, 0.0
        alternatives = [p for key, p in frame['probabilities'].items() if key != index]
        return {
            'token_index': index,
            'probability': probability,
            'text_attention_mass': math.fsum(frame['probabilities'].values()),
            'margin': probability - max(alternatives, default=0.0),
            'start': frame['start'],
            'end': frame['end'],
            'emitted_at': emitted_at,
        }


def _normalized_scores(scores):
    maximum = max(scores.values(), default=0.0)
    return {index: score - maximum for index, score in scores.items()}


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
