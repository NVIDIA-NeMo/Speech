"""EasyMagpie vLLM-Omni speech endpoint; independent reference-only requests."""

import hashlib
import itertools
import json
import os
import time
import urllib.request
from pathlib import Path

from .core import (
    chunk_text,
    digest,
    file_sha,
    read_json,
    read_jsonl,
    run_records,
    stage_fingerprint,
    wait_ready,
    wav_info,
    write_json,
    write_pcm_wav,
)


def synthesize(run_dir, limit=0):
    run = Path(run_dir)
    manifest = read_json(run / 'run_manifest.json')
    spec = manifest['config']['tts']
    if spec.get('reference_sha256') and file_sha(spec['reference_audio']) != spec['reference_sha256']:
        raise ValueError('Reference audio checksum mismatch')
    model_sha = None
    if spec.get('converted_model'):
        model_sha = file_sha(Path(spec['converted_model']) / 'CONVERSION_COMPLETE.json')
    url = os.environ.get('SPEECH_ROUNDTRIP_TTS_URL', spec['url']).rstrip('/')
    wait_ready(url, spec.get('startup_timeout_s', 900))
    records = read_jsonl(run / 'inputs.jsonl')
    if limit:
        records = itertools.islice(records, limit)
    done = run_records(
        records, lambda r: _synthesize_one(run, manifest, spec, url, model_sha, r), spec.get('concurrency', 1)
    )
    write_json(
        run / 'tts_status.json',
        {
            'status': 'complete' if done == manifest['records'] else 'partial',
            'completed': done,
            'expected': manifest['records'],
        },
    )


def _synthesize_one(run, manifest, spec, url, model_sha, record):
    fingerprint = digest([stage_fingerprint(manifest, record, 'tts'), model_sha])
    sidecar = run / 'tts' / f"{record['key']}.json"
    output = run / 'audio' / f"{record['key']}.wav"
    if sidecar.exists():
        old = read_json(sidecar)
        if old.get('fingerprint') != fingerprint:
            raise ValueError('TTS settings changed; use a new run directory')
        if old.get('status') == 'complete' and output.exists():
            if wav_info(output)['sha256'] == old['audio']['sha256']:
                return 1
    started = time.monotonic()
    silent_chunks = []
    try:
        chunks = chunk_text(record['text'], spec.get('max_chars_per_chunk', 600))
        if not record['text'].strip():
            pcm = b'\0\0' * (spec['sample_rate'] // 10)
            chunks = []
        else:
            parts = []
            for index, text in enumerate(chunks):
                payload = {
                    'input': text,
                    'voice': spec['voice'],
                    'response_format': 'pcm',
                    'stream': True,
                    'stream_format': 'audio',
                    'max_new_tokens': spec.get('max_new_tokens', 2048),
                }
                payload.update(spec.get('request_options', {}))
                request = urllib.request.Request(
                    url + '/v1/audio/speech', json.dumps(payload).encode(), {'Content-Type': 'application/json'}
                )
                with urllib.request.urlopen(request, timeout=spec.get('request_timeout_s', 300)) as response:
                    if 'json' in response.headers.get('Content-Type', ''):
                        raise ValueError('TTS returned a JSON error instead of PCM audio')
                    part = response.read()
                if not part or len(part) % 2:
                    raise ValueError('Empty or malformed TTS PCM response')
                chunk_path = run / 'audio_chunks' / record['key'] / f'{index:04d}.wav'
                write_pcm_wav(chunk_path, part, spec['sample_rate'])
                if wav_info(chunk_path)['rms'] == 0:
                    silent_chunks.append(index)
                parts.append(part)
            pcm = b''.join(parts)
        write_pcm_wav(output, pcm, spec['sample_rate'])
        info = wav_info(output)
        write_json(
            sidecar,
            {
                'id': record['id'],
                'dataset': record['dataset'],
                'status': 'complete',
                'fingerprint': fingerprint,
                'input_sha256': record['input_sha256'],
                'text_sha256': hashlib.sha256(record['text'].encode()).hexdigest(),
                'model_manifest_sha256': model_sha,
                'voice': spec['voice'],
                'single_turn': True,
                'chunks': chunks,
                'empty_source': not bool(record['text'].strip()),
                'silent_audio': info['rms'] == 0,
                'silent_chunks': silent_chunks,
                'endpoint': url,
                'audio_path': str(output.resolve()),
                'audio': info,
                'elapsed_s': time.monotonic() - started,
            },
        )
        print(f"TTS {record['dataset']} {record['id']}: {info['duration_s']:.2f}s", flush=True)
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
