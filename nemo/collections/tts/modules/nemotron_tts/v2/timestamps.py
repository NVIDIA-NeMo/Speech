# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only streaming timestamps from one main-backbone attention head.

The observer records causal alignment decisions at each emitted audio step,
alongside aggregate token spans and optional frame history.
"""

import math
from collections.abc import Mapping
from copy import deepcopy

import torch

from nemo.collections.tts.modules.attention_timestamps import AttentionTimestampTracker


def resolve_timestamp_config(config=None):
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

    def __init__(self, batch_size, codec_frame_seconds, frame_stacking_factor, config=None):
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

    def begin_step(self):
        """Clear the previous query before a new decoding step begins."""
        self._raw_attention = None

    def register_text(self, row, key_position, token_id, token=None):
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

    def reset_turn(self, row):
        self._validate_row(row)
        turn = self._turns[row][-1]
        if not turn['keys'] and not turn['has_audio']:
            return
        self._finish_turn(row)
        self._turns[row].append(self._new_turn(turn['turn_index'] + 1))

    def observe_attention(self, query, key, attention_mask=None, cache_position=None):
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

    def observe_audio(self, needs_audio, frame_start, valid_frames=None):
        """Publish current-frame events immediately, after clipping audio EOS."""
        counts = None if valid_frames is None else _values(valid_frames)
        for row, needed in enumerate(_values(needs_audio)):
            if not needed:
                continue
            count = self.frame_stacking_factor if counts is None else int(counts[row])
            self.observe_row(row, frame_start, count)
        self._raw_attention = None

    def observe_row(self, row, frame_start, valid_frames):
        self._validate_row(row)
        if self._origins[row] is None:
            self._origins[row] = frame_start
        self._ends[row] = frame_start + valid_frames
        if valid_frames <= 0:
            return
        turn = self._turns[row][-1]
        if turn['finalized']:
            raise ValueError('Reset the timestamp turn before observing more audio')
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

    def drain_events(self, batch_indices=None):
        if batch_indices is None:
            events, self._events = self._events, []
            return events
        selected = set(batch_indices)
        for row in selected:
            self._validate_row(row)
        events = [event for event in self._events if event['batch_index'] in selected]
        self._events = [event for event in self._events if event['batch_index'] not in selected]
        return events

    def drain_frames(self, batch_indices=None):
        """Consume current-step decisions, independently of token-span events.

        A record is emitted for every positive-duration audio step, including
        steps with unavailable or insufficient attention evidence. History in
        ``snapshots`` remains available after draining when retention is enabled.
        """
        if batch_indices is None:
            frames, self._frames = self._frames, []
            return frames
        selected = set(batch_indices)
        for row in selected:
            self._validate_row(row)
        frames = [frame for frame in self._frames if frame['batch_index'] in selected]
        self._frames = [frame for frame in self._frames if frame['batch_index'] not in selected]
        return frames

    def snapshots(self, finalize=False, audio_lengths=None):
        results = []
        lengths = _values(audio_lengths) if audio_lengths is not None else None
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

    def _new_turn(self, turn_index):
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

    def _duration(self, row):
        if self._origins[row] is None or self._ends[row] is None:
            return 0.0
        return max(0, self._ends[row] - self._origins[row]) * self.codec_frame_seconds

    def _finish_turn(self, row):
        turn = self._turns[row][-1]
        if not turn['finalized']:
            turn['tracker'].finalize(self._duration(row))
            turn['finalized'] = True
            self._collect(row, turn)

    def _collect(self, row, turn):
        self._events.extend(
            dict(event, batch_index=row, turn_index=turn['turn_index']) for event in turn['tracker'].drain_events()
        )

    def _validate_row(self, row):
        if type(row) is not int or not 0 <= row < self.batch_size:
            raise ValueError('Timestamp batch index is outside the conditional batch')


def _values(value):
    return value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else list(value)
