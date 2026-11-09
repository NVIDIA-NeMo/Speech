"""Transcribe generated speech with an independent vLLM ASR endpoint."""

import base64
import hashlib
import itertools
import math
import os
import re
import time
import wave
from pathlib import Path

from .core import (
    digest,
    file_sha,
    read_json,
    read_jsonl,
    request_json,
    run_records,
    stage_fingerprint,
    wait_ready,
    wav_info,
    write_json,
    write_pcm_wav,
)


class ASRResponseError(ValueError):
    def __init__(self, message, response):
        super().__init__(message)
        self.response = response


class ASRLengthLimit(ASRResponseError):
    def __init__(self, response):
        super().__init__('ASR did not finish normally: length', response)


def transcribe(run_dir, limit=0):
    run = Path(run_dir)
    manifest = read_json(run / 'run_manifest.json')
    spec = manifest['config']['asr']
    revision = os.environ.get('SPEECH_ROUNDTRIP_ASR_REVISION') or spec.get('revision')
    url = os.environ.get('SPEECH_ROUNDTRIP_ASR_URL', spec['url']).rstrip('/')
    wait_ready(url, spec.get('startup_timeout_s', 900))
    records = read_jsonl(run / 'inputs.jsonl')
    if limit:
        records = itertools.islice(records, limit)
    done = run_records(
        records, lambda r: _transcribe_one(run, manifest, spec, url, revision, r), spec.get('concurrency', 1)
    )
    write_json(
        run / 'asr_status.json',
        {
            'status': 'complete' if done == manifest['records'] else 'partial',
            'completed': done,
            'expected': manifest['records'],
        },
    )


def _transcribe_one(run, manifest, spec, url, revision, record):
    tts = read_json(run / 'tts' / f"{record['key']}.json")
    if tts['status'] != 'complete':
        raise ValueError('ASR requires a successful TTS artifact for every selected row')
    audio = Path(tts['audio_path'])
    info = wav_info(audio, include_rms=False)
    if info['sha256'] != tts['audio']['sha256']:
        raise ValueError('TTS audio changed after it was finalized')
    fingerprint = digest([stage_fingerprint(manifest, record, 'asr'), info['sha256'], revision])
    parts = _audio_parts(run, record, tts, audio, info, spec)
    if len(parts) > 1:
        fingerprint = digest(
            [fingerprint, 'lossless_tts_chunk_boundaries', spec.get('max_single_request_audio_s', 300), parts]
        )
    sidecar = run / 'asr' / f"{record['key']}.json"
    recovery = {'triggered': False}
    if sidecar.exists():
        old = read_json(sidecar)
        if old.get('fingerprint') != fingerprint:
            raise ValueError('ASR input/settings changed; use a new run directory')
        if old.get('status') == 'complete' and isinstance(old.get('transcript'), str):
            if hashlib.sha256(old['transcript'].encode()).hexdigest() != old.get('transcript_sha256'):
                raise ValueError('Cached ASR transcript checksum mismatch')
            return 1
        recovery['triggered'] = 'finish normally: length' in old.get('error', '')
    started = time.monotonic()
    try:
        if not record['text'].strip():
            result, transcript, segments = None, '', []
        else:
            segments = []
            offset = 0
            for part in parts:
                leaves = _decode_adaptive(run, record, spec, url, fingerprint, part, recovery)
                for leaf in leaves:
                    segments.append(
                        {
                            **leaf,
                            'global_start_frame': offset + leaf['start_frame'],
                            'global_end_frame': offset + leaf['end_frame'],
                        }
                    )
                offset += part['frames']
            if offset != info['frames'] or sum(s['frames'] for s in segments) != info['frames']:
                raise ValueError('Adaptive ASR did not cover every finalized audio frame')
            transcript = ' '.join(s['transcript'] for s in segments)
            # Preserve the original whole-reply response shape for ordinary replies.
            result = segments[0]['raw_response'] if len(segments) == 1 else None
        write_json(
            sidecar,
            {
                'id': record['id'],
                'dataset': record['dataset'],
                'status': 'complete',
                'fingerprint': fingerprint,
                'audio_sha256': info['sha256'],
                'model': spec['model'],
                'model_revision': revision,
                'endpoint': url,
                'prompt': spec['prompt'],
                'raw_response': result,
                'transcript': transcript,
                'segmentation': (
                    'adaptive_audio_ranges'
                    if recovery['triggered']
                    else ('tts_chunk_boundaries' if len(parts) > 1 else 'whole_reply')
                ),
                'decode_policy': 'duration_bounded_with_lossless_length_recovery_v2',
                'segments': segments,
                'transcript_sha256': hashlib.sha256(transcript.encode()).hexdigest(),
                'words': len(transcript.split()),
                'elapsed_s': time.monotonic() - started,
            },
        )
        print(f"ASR {record['dataset']} {record['id']}: {len(transcript.split())} words", flush=True)
        return 1
    except Exception as exc:
        write_json(
            sidecar,
            {
                'id': record['id'],
                'status': 'failed',
                'fingerprint': fingerprint,
                'error': f'{type(exc).__name__}: {exc}',
            },
        )
        raise


def _decode_audio(url, spec, path, max_tokens=None):
    data = base64.b64encode(path.read_bytes()).decode()
    payload = {
        'model': spec['model'],
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': spec['prompt']},
                    {'type': 'audio_url', 'audio_url': {'url': 'data:audio/wav;base64,' + data}},
                ],
            }
        ],
        'temperature': 0,
        'top_p': 1,
        'max_tokens': max_tokens if max_tokens is not None else spec.get('max_tokens', 16384),
        'seed': spec.get('seed', 0),
        'chat_template_kwargs': {'enable_thinking': False},
        'skip_special_tokens': False,
        'spaces_between_special_tokens': False,
    }
    result = request_json(url + '/v1/chat/completions', payload, spec.get('request_timeout_s', 600))
    choice = result['choices'][0]
    if choice.get('finish_reason') == 'length':
        raise ASRLengthLimit(result)
    if choice.get('finish_reason') != 'stop':
        raise ASRResponseError(f"ASR did not finish normally: {choice.get('finish_reason')}", result)
    message = choice['message']
    transcript = message.get('content')
    # OpenAI-compatible servers can represent a successful empty response as null.
    # Never promote reasoning, refusals or tool calls into transcript text.
    if transcript is None and not any(
        message.get(k) for k in ('reasoning', 'reasoning_content', 'refusal', 'tool_calls', 'function_call')
    ):
        transcript = ''
    if not isinstance(transcript, str):
        raise ASRResponseError('ASR did not return a clean transcript string', result)
    if re.search(r'</?think>|<\|im_(?:start|end)\|>', transcript):
        raise ASRResponseError('ASR returned control tokens instead of a clean transcript', result)
    return result, transcript


def _output_budget(seconds, spec):
    return min(spec.get('max_tokens', 16384), max(512, math.ceil(seconds * 32) + 256))


def _pause_cut(pcm, rate, start, end):
    """Choose a low-energy boundary near the midpoint; retain both sides exactly."""
    samples = memoryview(pcm).cast('h')
    middle = (start + end) // 2
    guard = max(1, rate // 4)
    lo, hi = max(start + guard, middle - rate // 2), min(end - guard, middle + rate // 2)
    window = max(1, rate // 50)
    if hi <= lo:
        return middle
    candidates = range(lo, hi, window)
    cut = min(candidates, key=lambda i: sum(x * x for x in samples[i : min(i + window, end)]))
    return max(start + 1, min(end - 1, cut + window // 2))


def _decode_adaptive(run, record, spec, url, fingerprint, part, recovery):
    root_path = Path(part['audio_path'])
    if file_sha(root_path) != part['audio_sha256']:
        raise ValueError('ASR source chunk checksum changed')
    with wave.open(str(root_path), 'rb') as w:
        rate, count = w.getframerate(), w.getnframes()
        pcm = w.readframes(count)
    if len(pcm) != count * 2 or count != part['frames']:
        raise ValueError('ASR source chunk PCM payload is incomplete')
    limit = spec.get('recovery_max_audio_s', 15)
    minimum = spec.get('minimum_recovery_audio_s', 1)
    if limit <= 0 or minimum <= 0:
        raise ValueError('ASR recovery durations must be positive')
    policy_path = run / 'asr_chunks' / record['key'] / 'recovery_policy.json'
    if policy_path.exists():
        policy = read_json(policy_path)
        if policy.get('fingerprint') != fingerprint:
            raise ValueError('ASR recovery policy cache identity mismatch')
        recovery['triggered'] = True

    def mark_recovery():
        recovery['triggered'] = True
        write_json(
            policy_path,
            {
                'fingerprint': fingerprint,
                'trigger': 'unclean_or_truncated_response',
                'max_recovery_audio_s': limit,
                'minimum_audio_s': minimum,
            },
        )

    if recovery['triggered']:
        mark_recovery()

    def split(start, end):
        if (end - start) / rate <= minimum:
            raise ValueError('ASR still hit its output limit on a minimal audio segment; no partial text accepted')
        cut = _pause_cut(pcm, rate, start, end)
        return decode(start, cut) + decode(cut, end)

    def decode(start, end):
        seconds = (end - start) / rate
        budget = _output_budget(seconds, spec)
        identity = digest([fingerprint, 'adaptive_audio_v2', part['audio_sha256'], start, end, budget])
        cache = run / 'asr_chunks' / record['key'] / f'{identity[:24]}.json'
        if cache.exists():
            saved = read_json(cache)
            if saved.get('fingerprint') != identity:
                raise ValueError('ASR segment cache fingerprint mismatch')
            if saved.get('status') == 'complete':
                if hashlib.sha256(saved['transcript'].encode()).hexdigest() != saved['transcript_sha256']:
                    raise ValueError('ASR cached segment transcript changed')
                if saved['raw_response']['choices'][0]['finish_reason'] != 'stop':
                    raise ValueError('Cached ASR segment contains an incomplete response')
                return [saved]
            if saved.get('status') == 'needs_split':
                mark_recovery()
                return split(start, end)
        if recovery['triggered'] and seconds > limit:
            return split(start, end)
        if start == 0 and end == count:
            audio = root_path
        else:
            audio = run / 'asr_audio_chunks' / record['key'] / f'{identity[:24]}.wav'
            write_pcm_wav(audio, pcm[start * 2 : end * 2], rate)
        metadata = {
            'fingerprint': identity,
            'root_audio_sha256': part['audio_sha256'],
            'root_audio_path': str(root_path),
            'audio_path': str(audio),
            'audio_sha256': file_sha(audio),
            'start_frame': start,
            'end_frame': end,
            'frames': end - start,
            'sample_rate': rate,
            'max_tokens_actual': budget,
        }
        try:
            response, text = _decode_audio(url, spec, audio, budget)
        except ASRResponseError as exc:
            write_json(
                cache,
                {
                    **metadata,
                    'status': 'needs_split',
                    'raw_response': exc.response,
                    'error': f'{exc}; response excluded from transcript',
                },
            )
            mark_recovery()
            return split(start, end)
        leaf = {
            **metadata,
            'status': 'complete',
            'raw_response': response,
            'transcript': text,
            'transcript_sha256': hashlib.sha256(text.encode()).hexdigest(),
        }
        write_json(cache, leaf)
        return [leaf]

    leaves = decode(0, count)
    if (
        not leaves
        or leaves[0]['start_frame'] != 0
        or leaves[-1]['end_frame'] != count
        or any(a['end_frame'] != b['start_frame'] for a, b in zip(leaves, leaves[1:]))
    ):
        raise ValueError('Adaptive ASR ranges contain an audio gap or overlap')
    return leaves


def _audio_parts(run, record, tts, audio, info, spec):
    """For long replies, verify that the existing synthesis chunks cover every PCM frame."""
    limit = spec.get('max_single_request_audio_s', 300)
    if not isinstance(limit, (int, float)) or limit <= 0:
        raise ValueError('max_single_request_audio_s must be positive')
    if info['duration_s'] <= limit:
        return [{'audio_path': str(audio), 'audio_sha256': info['sha256'], 'frames': info['frames']}]
    parts, frames, chunk_pcm = [], 0, hashlib.sha256()
    for i in range(len(tts['chunks'])):
        path = run / 'audio_chunks' / record['key'] / f'{i:04d}.wav'
        with wave.open(str(path), 'rb') as w:
            if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, info['sample_rate']):
                raise ValueError('TTS chunk format differs from the final audio')
            count = w.getnframes()
            if count / info['sample_rate'] > limit:
                raise ValueError('A TTS chunk exceeds the ASR duration limit; use smaller synthesis chunks')
            data = w.readframes(count)
        if not count or len(data) != count * 2:
            raise ValueError('TTS chunk has an incomplete PCM payload')
        chunk_pcm.update(data)
        frames += count
        parts.append({'audio_path': str(path), 'audio_sha256': file_sha(path), 'frames': count})
    full_pcm = hashlib.sha256()
    with wave.open(str(audio), 'rb') as w:
        while data := w.readframes(1024 * 1024):
            full_pcm.update(data)
    if not parts or frames != info['frames'] or chunk_pcm.digest() != full_pcm.digest():
        raise ValueError('ASR chunks do not exactly reconstruct the final TTS waveform')
    return parts
