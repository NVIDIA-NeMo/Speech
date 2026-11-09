"""Configuration, input adapters, atomic artifacts, and HTTP utilities."""

import hashlib
import json
import math
import os
import re
import struct
import time
import urllib.error
import urllib.request
import wave
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import yaml


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.partial')
    try:
        with temporary.open('wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                raise ValueError(f'{path}:{line_number}: blank JSONL record')
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{line_number}: malformed JSON') from exc


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg.get('schema_version') != 1 or not cfg.get('datasets'):
        raise ValueError('Expected schema_version: 1 and nonempty datasets')
    names = [d['name'] for d in cfg['datasets']]
    if len(names) != len(set(names)):
        raise ValueError('Dataset names must be unique')
    for name in names:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in ('.', '..'):
            raise ValueError(f'Unsafe dataset name: {name}')
    return cfg


def adapt_dataset(spec):
    """Support NeMo audio-evals rollouts or a generic id/question/text JSONL."""
    seen = set()
    count = 0
    for row in read_jsonl(spec['path']):
        if spec.get('format', 'jsonl') == 'audio_evals':
            identifier = row.get('canonical_sample_id')
            if identifier is None:
                identifier = row.get('sample_id')
            question = (row.get('verifier_metadata') or {}).get('prompt')
            text = row.get('candidate_answer')
            source_status = row.get('candidate_status')
            source_valid = row.get('candidate_valid')
        elif spec.get('format', 'jsonl') == 'jsonl':
            fields = spec.get('fields', {})
            identifier = row.get(fields.get('id', 'id'))
            question = row.get(fields.get('question', 'question'))
            text = row.get(fields.get('text', 'text'))
            source_status, source_valid = 'completed', True
        else:
            raise ValueError(f"Unknown input format: {spec['format']}")
        if identifier is None or not isinstance(question, str) or not isinstance(text, str):
            raise ValueError(f"{spec['name']}: each row needs an id, question string, and text string")
        identifier = str(identifier)
        if identifier in seen:
            raise ValueError(f"Duplicate id in {spec['name']}: {identifier}")
        seen.add(identifier)
        count += 1
        record = {
            'id': identifier,
            'dataset': spec['name'],
            'question': question,
            'text': text,
            'source_status': source_status,
            'source_valid': source_valid,
        }
        record['key'] = digest([spec['name'], identifier])[:24]
        record['input_sha256'] = digest(record)
        yield record
    if spec.get('expected_rows') is not None and count != spec['expected_rows']:
        raise ValueError(f"{spec['name']}: expected {spec['expected_rows']} rows, got {count}")


def prepare(config_path, run_dir, preview_per_dataset=0):
    cfg = load_config(config_path)
    run = Path(run_dir).resolve()
    records, sources = [], []
    for spec in cfg['datasets']:
        rows = list(adapt_dataset(spec))
        sources.append(
            {
                'dataset': spec['name'],
                'path': str(Path(spec['path']).resolve()),
                'sha256': file_sha(spec['path']),
                'rows': len(rows),
            }
        )
        if preview_per_dataset:
            valid = [r for r in rows if r['source_status'] == 'completed' and r['text'].strip()]
            if not valid:
                raise ValueError(f"No completed, nonempty preview candidates in {spec['name']}")
            lengths = sorted(len(r['text'].split()) for r in valid)
            median = lengths[(len(lengths) - 1) // 2]
            rows = sorted(valid, key=lambda r: (abs(len(r['text'].split()) - median), r['id']))[:preview_per_dataset]
        records.extend(rows)
    identity = {'config': cfg, 'sources': sources, 'preview_per_dataset': preview_per_dataset}
    fingerprint = digest(identity)
    manifest_path = run / 'run_manifest.json'
    if manifest_path.exists():
        if read_json(manifest_path)['fingerprint'] != fingerprint:
            raise ValueError('Run inputs/configuration changed; choose a new run directory')
        saved = list(read_jsonl(run / 'inputs.jsonl'))
        if saved != records:
            raise ValueError('Existing input manifest is corrupt; restore it before resuming')
        return run
    run.mkdir(parents=True, exist_ok=True)
    data = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records).encode()
    atomic_bytes(run / 'inputs.jsonl', data)
    write_json(
        manifest_path,
        {
            'schema_version': 1,
            'fingerprint': fingerprint,
            **identity,
            'records': len(records),
            'scope': 'preview' if preview_per_dataset else 'full',
            'selection': (
                'completed answers closest to per-dataset median length, then id'
                if preview_per_dataset
                else 'all rows, including source failures'
            ),
            'conversation_mode': 'single_turn_reference_voice_only',
        },
    )
    for r in records:
        atomic_bytes(run / 'texts' / f"{r['key']}.txt", r['text'].encode())
    return run


def chunk_text(text, max_chars):
    """Split without deleting, inserting, or rewriting any characters."""
    if max_chars < 32:
        raise ValueError('max_chars must be at least 32')
    chunks = []
    remaining = text
    while len(remaining) > max_chars:
        window = remaining[: max_chars + 1]
        boundaries = [m.end() for m in re.finditer(r'[.!?][\s]+', window)]
        split = next((p for p in reversed(boundaries) if p >= max_chars // 2), None)
        if split is None:
            spaces = [m.end() for m in re.finditer(r'\s+', window)]
            split = spaces[-1] if spaces else None
        if not split:
            raise ValueError('An unbroken token exceeds the TTS text chunk limit')
        split = min(split, max_chars)
        chunks.append(remaining[:split])
        remaining = remaining[split:]
    if remaining:
        chunks.append(remaining)
    assert ''.join(chunks) == text
    return chunks


def request_json(url, payload=None, timeout=300):
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_ready(url, timeout=900):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            request_json(url.rstrip('/') + '/v1/models', timeout=5)
            return
        except (urllib.error.URLError, TimeoutError, ValueError):
            time.sleep(3)
    raise TimeoutError(f'Server did not become ready: {url}')


def wav_info(path, *, include_rms=True):
    with wave.open(str(path), 'rb') as f:
        if f.getnchannels() != 1 or f.getsampwidth() != 2 or not f.getnframes():
            raise ValueError('Expected nonempty mono PCM16 WAV')
        data = f.readframes(f.getnframes())
        if len(data) != f.getnframes() * 2:
            raise ValueError('Truncated WAV payload')
        info = {
            'sample_rate': f.getframerate(),
            'channels': 1,
            'frames': f.getnframes(),
            'duration_s': f.getnframes() / f.getframerate(),
            'bytes': Path(path).stat().st_size,
            'sha256': file_sha(path),
        }
        if include_rms:
            samples = memoryview(data).cast('h')
            info['rms'] = math.sqrt(sum(float(x) ** 2 for x in samples) / len(samples)) / 32768
        return info


def write_pcm_wav(path, pcm, sample_rate):
    if len(pcm) % 2 or not pcm:
        raise ValueError('Invalid PCM16 response')
    size = len(pcm)
    header = (
        b'RIFF'
        + struct.pack('<I', 36 + size)
        + b'WAVEfmt '
        + struct.pack('<IHHIIHH', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b'data'
        + struct.pack('<I', size)
    )
    atomic_bytes(path, header + pcm)


def stage_fingerprint(manifest, record, stage):
    return digest({'input': record['input_sha256'], 'spec': manifest['config'][stage]})


def run_records(records, worker, concurrency=1):
    """Bound in-flight HTTP requests; stop scheduling when any worker fails."""
    if not isinstance(concurrency, int) or not 1 <= concurrency <= 32:
        raise ValueError('concurrency must be an integer from 1 to 32')
    if concurrency == 1:
        return sum(worker(record) for record in records)
    iterator = iter(records)
    executor = ThreadPoolExecutor(max_workers=concurrency)
    pending = set()
    total = 0
    try:
        for _ in range(concurrency):
            record = next(iterator, None)
            if record is not None:
                pending.add(executor.submit(worker, record))
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            total += sum(f.result() for f in finished)
            for _ in finished:
                record = next(iterator, None)
                if record is not None:
                    pending.add(executor.submit(worker, record))
    except BaseException:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    executor.shutdown(wait=True)
    return total
