# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Causal attention timestamp decoding shared by HTTP, WebSocket and offline clients.

The exact NeMo tracker is bundled at conversion, so serving does not import NeMo.
Character boundaries inherit their input-token interval; attention cannot recover
independent sub-token acoustic boundaries. Every mapping is occurrence-indexed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import struct
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


def bundle_timestamp_runtime(outdir: str | Path, native_module_path: str | Path) -> dict[str, str]:
    """Copy the standalone native tracker into a converted artifact, with integrity metadata."""
    data = Path(native_module_path).read_bytes()
    destination = Path(outdir) / 'attention_timestamp_runtime.py'
    destination.write_bytes(data)
    return {'timestamp_runtime_file': destination.name, 'timestamp_runtime_sha256': hashlib.sha256(data).hexdigest()}


def load_timestamp_runtime(model_path: str | Path):
    """Load the exported tracker without executing any ``nemo.__init__`` code."""
    model_path = Path(model_path)
    config = json.loads((model_path / 'config.json').read_text())
    name = config.get('timestamp_runtime_file')
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError('Timestamp recording requires a converted artifact with a bundled timestamp runtime')
    source = model_path / name
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != config.get('timestamp_runtime_sha256'):
        raise ValueError('The bundled timestamp runtime does not match its exported SHA256')
    return _load_runtime(str(source.resolve()), digest)


def pack_alignment_packet(value: dict[str, Any]):
    """Return a compact length-prefixed CPU uint8 tensor for the pinned Omni wire contract."""
    import torch

    encoded = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return torch.frombuffer(bytearray(struct.pack('<I', len(encoded)) + encoded), dtype=torch.uint8).clone()


def unpack_alignment_packets(stage_output) -> list[dict[str, Any]]:
    """Decode stage-0 packet deltas; accept both raw and wrapped Omni outputs."""
    import torch

    if isinstance(stage_output, (tuple, list)):
        return [packet for output in stage_output for packet in unpack_alignment_packets(output)]
    mm = getattr(stage_output, 'multimodal_output', None)
    if mm is None:
        outputs = getattr(getattr(stage_output, 'request_output', stage_output), 'outputs', None) or []
        mm = getattr(outputs[0], 'multimodal_output', None) if outputs else None
    value = mm.get('alignment') if hasattr(mm, 'get') else None
    if value is None:
        return []
    chunks = value if isinstance(value, list) else [value]
    raw = b''.join(chunk.detach().to(device='cpu', dtype=torch.uint8).numpy().tobytes() for chunk in chunks)
    records = []
    offset = 0
    while offset < len(raw):
        if len(raw) - offset < 4:
            raise ValueError('Truncated alignment packet header')
        length = struct.unpack_from('<I', raw, offset)[0]
        offset += 4
        if length <= 0 or offset + length > len(raw):
            raise ValueError('Truncated or invalid alignment packet body')
        records.append(json.loads(raw[offset : offset + length]))
        offset += length
    return records


class StreamingAlignmentDecoder:
    """Apply the native online tracker and expose token/frame and character/time dictionaries.

    ``update`` consumes one raw attention observation and returns only changed
    token/character records. ``finalize`` includes every supplied input token,
    preserving null boundaries for missing evidence. Adjacent repeated IDs and
    overlapping UTF-8 tokenizer offsets remain distinct input occurrences.
    """

    def __init__(self, model_path: str | Path, tokenizer=None, text: str | None = None) -> None:
        self.model_path = Path(model_path)
        self.runtime = load_timestamp_runtime(self.model_path)
        config = json.loads((self.model_path / 'config.json').read_text())
        settings = self.runtime.resolve_timestamp_config(config.get('timestamp_config'))
        self.settings = settings
        self.tracker = self.runtime.AttentionTimestampTracker(
            **{k: v for k, v in settings.items() if k not in ('enabled', 'layer', 'head', 'retain_frame_history')},
            record_frames=True,
        )
        self.tokenizer = tokenizer
        self.text = ''
        self.input_ids: list[int] = []
        self.offsets: list[tuple[int, int]] = []
        self._character_tokens: list[list[int]] = []
        self._token_characters: dict[int, list[int]] = {}
        self._input_to_alignment: dict[int, int] = {}
        self.registrations: dict[int, dict] = {}
        self.token_records: dict[int, dict] = {}
        self.span_records: dict[int, dict[int, dict]] = {}
        self.frames: list[dict] = []
        self._last_sequence = -1
        self._codec_frames = 0
        self._codec_frame_seconds: float | None = None
        self._stacking = int(config.get('frame_stacking_factor', 1))
        self._frame_step_seconds: float | None = None
        self._attention_block: int | None = None
        self._audio_eos: bool | None = None
        self._closed = False
        if text is not None:
            self.append_text(text)

    def append_text(self, text: str, expected_ids: list[int] | None = None) -> None:
        """Append one complete text chunk with offsets in the actual tokenization used for that chunk."""
        if self._closed:
            raise ValueError('Cannot append text after finalizing timestamps')
        if self.tokenizer is None:
            raise ValueError('Character timestamps require the input text tokenizer')
        ids, offsets = self.tokenizer.encode_with_offsets(text)
        if expected_ids is not None and list(ids) != list(expected_ids):
            raise ValueError('Timestamp offsets do not match the actual supplied input token IDs')
        start = len(self.text)
        token_start = len(self.input_ids)
        self.text += text
        self.input_ids.extend(ids)
        self.offsets.extend((start + a, start + b) for a, b in offsets)
        self._character_tokens.extend([] for _ in text)
        for token_index, (a, b) in enumerate(offsets, token_start):
            characters = list(range(start + a, start + b))
            self._token_characters[token_index] = characters
            for character_index in characters:
                self._character_tokens[character_index].append(token_index)

    def append_stream_text(
        self,
        text: str,
        emitted_ids: list[int],
        *,
        deferred: bool = False,
        closes_segment: bool = False,
    ) -> None:
        """Track text fed through the incremental encoder, which may hold tokens back across chunks.

        Offsets come from tokenizing the whole open segment (text since the last raw-id boundary);
        only tokens the encoder has already emitted are registered, and they must be a prefix of
        that tokenization. ``deferred`` skips registration while a pronunciation span is open.
        ``closes_segment`` requires every token to have been emitted and starts a new segment.
        """
        if self._closed:
            raise ValueError('Cannot append text after finalizing timestamps')
        if self.tokenizer is None:
            raise ValueError('Character timestamps require the input text tokenizer')
        if not hasattr(self, '_segment_text'):
            self._segment_text = ''
            self._segment_ids: list[int] = []
            self._segment_registered = 0
            self._segment_char_base = len(self.text)
            self._segment_token_base = len(self.input_ids)
        self.text += text
        self._character_tokens.extend([] for _ in text)
        self._segment_text += text
        self._segment_ids.extend(emitted_ids)
        if deferred and not closes_segment:
            return
        if not self._segment_text and not self._segment_ids:
            return
        ids, offsets = self.tokenizer.encode_with_offsets(self._segment_text)
        emitted = len(self._segment_ids)
        if list(ids[:emitted]) != list(self._segment_ids) or (closes_segment and len(ids) != emitted):
            raise ValueError('Timestamp offsets do not match the actual supplied input token IDs')
        for k in range(self._segment_registered, emitted):
            a, b = offsets[k]
            token_index = self._segment_token_base + k
            self.input_ids.append(ids[k])
            self.offsets.append((self._segment_char_base + a, self._segment_char_base + b))
            characters = list(range(self._segment_char_base + a, self._segment_char_base + b))
            self._token_characters[token_index] = characters
            for character_index in characters:
                self._character_tokens[character_index].append(token_index)
        self._segment_registered = emitted
        if closes_segment:
            del self._segment_text, self._segment_ids, self._segment_registered
            del self._segment_char_base, self._segment_token_base

    def update(self, packet: dict[str, Any]) -> dict[str, Any] | None:
        """Consume a causal frame before the next audio observation is available."""
        if self._closed:
            raise ValueError('Cannot update finalized timestamps')
        sequence = int(packet['sequence'])
        if sequence <= self._last_sequence:
            return None  # Supports cumulative outputs as well as drained deltas.
        if sequence != self._last_sequence + 1:
            raise ValueError('An alignment observation was lost or reordered')
        self._last_sequence = sequence
        if 'audio_eos' in packet:
            self._audio_eos = bool(self._audio_eos or packet['audio_eos'])
        block = packet.get('attention_block')
        if self._attention_block is None:
            self._attention_block = block
        if block != self._attention_block:
            raise ValueError('The observed attention block changed within a stream')
        seconds = float(packet['codec_frame_seconds'])
        if self._codec_frame_seconds is None:
            self._codec_frame_seconds = seconds
            self._frame_step_seconds = seconds * self._stacking
        if not math.isfinite(seconds) or seconds <= 0 or seconds != self._codec_frame_seconds:
            raise ValueError('The codec frame clock changed within a stream')
        changed = set()
        for registration in packet.get('registrations', []):
            index = int(registration['token_index'])
            if index in self.registrations:
                raise ValueError('A token occurrence was registered more than once')
            self.registrations[index] = dict(registration)
            self._input_to_alignment[int(registration['input_token_index'])] = index
            self.tracker.register_token(index, int(registration['token_id']), self._label(registration))
            changed.add(index)
        count = int(packet['valid_codec_frames'])
        if int(packet['codec_frame_start']) != self._codec_frames:
            raise ValueError('Audio codec frame observations are not contiguous')
        if not 0 <= count <= self._stacking:
            raise ValueError('An alignment observation has an invalid codec frame count')
        if count:
            self.tracker.update(
                packet['probabilities'],
                token_indices=packet['token_indices'],
                start=self._codec_frames * seconds,
                end=(self._codec_frames + count) * seconds,
            )
            self._codec_frames += count
        events = self.tracker.drain_events()
        for event in events:
            index = int(event['token_index'])
            self.span_records.setdefault(index, {})[int(event['span_id'])] = dict(event)
            changed.add(index)
        for index in changed:
            spans = list(self.span_records.get(index, {}).values())
            if spans:
                primary = max(spans, key=lambda span: span['support'])
                self.token_records[index] = dict(primary, forced=any(span['forced'] for span in spans))
        frames = []
        for frame in self.tracker.drain_frames():
            index = frame['token_index']
            frame = dict(
                frame,
                text_key_position=self.registrations[index]['text_key_position'] if index is not None else None,
                input_token_index=self.registrations[index]['input_token_index'] if index is not None else None,
                codec_frame_start=int(packet['codec_frame_start']),
                codec_frame_end=self._codec_frames,
                valid_codec_frames=count,
            )
            frames.append(frame)
            if self.settings['retain_frame_history']:
                self.frames.append(frame)
        tokens = {str(self.registrations[i]['input_token_index']): self._token_record(i) for i in sorted(changed)}
        characters = self._characters({int(index) for index in tokens})
        return {
            'type': 'alignment.delta',
            'schema_version': 1,
            'sequence': sequence,
            'frame_step_seconds': self._frame_step_seconds,
            'frames': frames,
            'events': events,
            'token_timestamps': tokens,
            'character_timestamps': characters,
        }

    def append_token_ids(self, token_ids: list[int]) -> None:
        """Append raw-token input; retain null character offsets if decoding is not losslessly reversible."""
        if self.tokenizer is None:
            raise ValueError('Decoding input IDs requires the text tokenizer')
        decoded = self.tokenizer.base_tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
        encoded, _ = self.tokenizer.encode_with_offsets(decoded)
        if encoded == list(token_ids):
            self.append_text(decoded, expected_ids=token_ids)
            return
        start = len(self.text)
        self.text += decoded
        self._character_tokens.extend([] for _ in decoded)
        self.input_ids.extend(token_ids)
        self.offsets.extend((start, start) for _ in token_ids)

    def finalize(self, duration_seconds: float | None = None) -> dict[str, Any]:
        """Return all token occurrences, characters, words and immutable observed frames."""
        observed_end = self._codec_frames * (self._codec_frame_seconds or 0.0)
        duration = observed_end if duration_seconds is None else float(duration_seconds)
        if duration + 1e-6 < observed_end:
            raise ValueError(
                f'Audio is shorter than the observed codec clock ({duration:.6f}s versus {observed_end:.6f}s); '
                'do not invent a crop after publication'
            )
        self.tracker.finalize(max(duration, observed_end))
        for record in self.tracker.snapshot():
            self.token_records[int(record['token_index'])] = record
        self._closed = True
        tokens = {str(r['input_token_index']): self._token_record(i) for i, r in self.registrations.items()}
        for index, token_id in enumerate(self.input_ids):
            tokens.setdefault(str(index), self._missing_token(index, token_id))
        characters = self._characters()
        words = []
        for match in re.finditer(r'\S+', self.text):
            covered = [characters[str(i)] for i in range(match.start(), match.end())]
            starts = [c['start'] for c in covered if c['start'] is not None]
            ends = [c['end'] for c in covered if c['end'] is not None]
            words.append(
                {
                    'text': match.group(),
                    'character_start': match.start(),
                    'character_end': match.end(),
                    'start': min(starts) if starts else None,
                    'end': max(ends) if ends else None,
                    'token_indices': sorted({i for c in covered for i in c['token_indices']}),
                    'partially_aligned': any(c['start'] is None for c in covered),
                }
            )
        return {
            'type': 'alignment.done',
            'schema_version': 1,
            'source': 'attention',
            'causal': True,
            'timings_are_estimates': True,
            'character_times_inherit_token_intervals': True,
            'confidence_is_calibrated': False,
            'tracker_settings': dict(self.settings),
            'forced_subframe_times_are_estimates': True,
            'ended_by_audio_eos': self._audio_eos,
            'generation_cap_hit': not self._audio_eos if self._audio_eos is not None else None,
            'attention_layer': self.settings['layer'],
            'attention_head': self.settings['head'],
            'attention_block': self._attention_block,
            'audio_duration_seconds': duration,
            'frame_step_seconds': self._frame_step_seconds,
            'codec_frame_seconds': self._codec_frame_seconds,
            'codec_frame_count': self._codec_frames,
            'input_text': self.text,
            'token_timestamps': dict(sorted(tokens.items(), key=lambda p: int(p[0]))),
            'character_timestamps': characters,
            'words': words,
            'frames': self.frames,
            'events': self.tracker.drain_events(),
        }

    def _label(self, registration):
        index = int(registration['input_token_index'])
        token_id = int(registration['token_id'])
        if index < len(self.input_ids):
            if self.input_ids[index] != token_id:
                raise ValueError('The observed source token does not match the supplied input')
            start, end = self.offsets[index]
            if end > start:
                return self.text[start:end]
        if self.tokenizer is not None:
            return self.tokenizer.base_tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        return None

    def _token_record(self, index):
        registration = self.registrations[index]
        record = self.token_records.get(index, {})
        start, end = record.get('start'), record.get('end')
        step = self._frame_step_seconds
        input_index = int(registration['input_token_index'])
        offsets = self.offsets[input_index] if input_index < len(self.offsets) else (None, None)
        return {
            'token_index': registration['input_token_index'],
            'alignment_token_index': index,
            'token_id': registration['token_id'],
            'decoded': self._label(registration),
            'character_start': offsets[0],
            'character_end': offsets[1],
            'text_key_position': registration['text_key_position'],
            'frame_step': _grid(start, step),
            'frame_end_exclusive': _grid(end, step, ceiling=True),
            'time_seconds': start,
            'start': start,
            'end': end,
            'confidence': 0.0 if record.get('forced') else record.get('confidence'),
            'attention_confidence': record.get('attention_confidence'),
            'forced': record.get('forced', False),
            'status': record.get('status', 'unaligned'),
            'reason': record.get('reason', None if start is not None else 'no_evidence'),
            'emitted_at': record.get('emitted_at'),
        }

    def _missing_token(self, index, token_id):
        return {
            'token_index': index,
            'alignment_token_index': None,
            'token_id': token_id,
            'decoded': self.text[slice(*self.offsets[index])],
            'text_key_position': None,
            'frame_step': None,
            'frame_end_exclusive': None,
            'time_seconds': None,
            'start': None,
            'end': None,
            'confidence': None,
            'attention_confidence': None,
            'forced': False,
            'status': 'unaligned',
            'reason': 'not_consumed_or_no_native_text_key',
            'emitted_at': None,
        }

    def _characters(self, changed_inputs=None):
        # Multiple byte-level token occurrences can cover the same Unicode character.
        indices = (
            range(len(self.text))
            if changed_inputs is None
            else sorted({index for token in changed_inputs for index in self._token_characters.get(token, [])})
        )
        result = {}
        for index in indices:
            character = self.text[index]
            covered = self._character_tokens[index]
            records = [
                self._token_record(self._input_to_alignment[i]) for i in covered if i in self._input_to_alignment
            ]
            starts = [r['start'] for r in records if r['start'] is not None]
            ends = [r['end'] for r in records if r['end'] is not None]
            result[str(index)] = {
                'character_index': index,
                'character': character,
                'token_indices': covered,
                'token_ids': [self.input_ids[i] for i in covered],
                'time_seconds': min(starts) if starts else None,
                'start': min(starts) if starts else None,
                'end': max(ends) if ends else None,
                'forced': any(r['forced'] for r in records),
                'status': 'aligned' if starts else 'unaligned',
            }
        return result


@lru_cache(maxsize=8)
def _load_runtime(source: str, digest: str):
    name = '_easymagpie_timestamp_runtime_' + digest
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolves the defining module while loading.
    spec.loader.exec_module(module)
    return module


def _grid(seconds, step, ceiling=False):
    if seconds is None or step is None:
        return None
    value = seconds / step
    return max(0, int(math.ceil(value - 1e-8) if ceiling else math.floor(value + 1e-8)))
