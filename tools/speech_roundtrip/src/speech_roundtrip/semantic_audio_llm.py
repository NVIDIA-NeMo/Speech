"""Judge saved TTS audio directly with the user-provided 1-to-3 audio rubric."""

import base64
import hashlib
import json
import os
import re
import statistics
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .core import atomic_bytes, digest, file_sha, read_json, read_jsonl, run_records, wav_info, write_json

DEFAULT_MODEL = 'gcp/google/gemini-3.8-flash'
DEFAULT_PROMPT = Path(__file__).parent / 'prompts' / 'semantic_audio_llm.txt'
NAME = 'semantic audio LLM'
FORMAT_RECOVERY = (
    '\n\nOutput-format recovery: The earlier response exceeded the output budget. '
    'Evaluate the entire supplied audio against the same rubric above. For long repetition runs, '
    'write a few literal repetitions in heard, followed by [repetition continues], and describe the '
    'continuation in issues. Keep other heard content literal. Return the complete heard/score/issues '
    'JSON within the budget; do not continue a repetition run indefinitely in the JSON.'
)


def parse_result(content):
    """Allow the supplied script's fenced JSON, never infer a missing grade."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError('Audio judge returned no JSON text')
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', content):
        try:
            value, _ = decoder.raw_decode(content[match.start() :])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict) or set(value) != {'heard', 'score', 'issues'}:
            continue
        if type(value['score']) is not int or value['score'] not in (1, 2, 3):
            raise ValueError('Audio judge score must be an integer 1, 2 or 3')
        if not isinstance(value['heard'], str) or not isinstance(value['issues'], str) or not value['issues'].strip():
            raise ValueError('Audio judge needs string heard and nonempty issues fields')
        return value
    raise ValueError('Audio judge needs complete heard/score/issues JSON; no fallback score assigned')


def _base_url(manifest, override):
    base = override or os.environ.get('OPENAI_API_BASE') or os.environ.get('OPENAI_BASE_URL')
    if not base:
        spec = manifest['config'].get('judge', {})
        if spec.get('script'):
            from .judging import load_judge

            base = load_judge(spec).NV_BASE_URL
    if not base:
        raise ValueError('Set OPENAI_API_BASE or pass --api-base for the inference gateway')
    base = base.rstrip('/')
    parsed = urllib.parse.urlsplit(base)
    if (
        parsed.scheme not in ('http', 'https')
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
    ):
        raise ValueError('API base must be an HTTP(S) URL without embedded credentials or query parameters')
    return base if base.endswith('/v1') else base + '/v1'


def _settings(manifest, model, concurrency, api_base):
    cfg = manifest['config'].get('semantic_audio_llm', {})
    settings = {
        'model': model or cfg.get('model', DEFAULT_MODEL),
        'api_base': _base_url(manifest, api_base or cfg.get('api_base')),
        'temperature': 0,
        'max_tokens': cfg.get('max_tokens', 1200),
        'retry_max_tokens': cfg.get('retry_max_tokens', 4000),
        'attempts': cfg.get('attempts', 5),
        'concurrency': concurrency if concurrency is not None else cfg.get('concurrency', 8),
        'rate': cfg.get('rate', 5),
        'timeout_s': cfg.get('timeout_s', 180),
        'max_audio_bytes': cfg.get('max_audio_bytes', 12 * 1024 * 1024),
        'max_audio_s': cfg.get('max_audio_s', 300),
    }
    if type(settings['concurrency']) is not int or not 1 <= settings['concurrency'] <= 32:
        raise ValueError('Audio judge concurrency must be an integer from 1 to 32')
    for key in ('max_tokens', 'retry_max_tokens', 'attempts', 'max_audio_bytes'):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if settings['retry_max_tokens'] < settings['max_tokens'] or any(
        settings[k] <= 0 for k in ('rate', 'timeout_s', 'max_audio_s')
    ):
        raise ValueError('Invalid audio judge budgets or rate limits')
    return settings


def _plan(run, row, settings):
    if (
        not re.fullmatch(r'[a-f0-9]{24}', row['key'])
        or not re.fullmatch(r'[A-Za-z0-9_.-]+', row['dataset'])
        or row['dataset'] in ('.', '..')
    ):
        raise ValueError('Unsafe audio judge input identifier')
    tts = read_json(run / 'tts' / f"{row['key']}.json")
    if tts['status'] != 'complete' or tts['text_sha256'] != hashlib.sha256(row['text'].encode()).hexdigest():
        raise ValueError('Audio judge requires finalized TTS matching the exact source text')
    audio = Path(tts['audio_path']).resolve()
    info = wav_info(audio, include_rms=False)
    if info['sha256'] != tts['audio']['sha256']:
        raise ValueError('Finalized TTS audio checksum changed')
    parts = [
        {
            'index': 0,
            'audio_path': str(audio),
            'audio_sha256': info['sha256'],
            'frames': info['frames'],
            'sample_rate': info['sample_rate'],
            'duration_s': info['duration_s'],
            'text': row['text'],
        }
    ]
    if info['bytes'] > settings['max_audio_bytes'] or info['duration_s'] > settings['max_audio_s']:
        chunks = tts.get('chunks', [])
        if not chunks or ''.join(chunks) != row['text']:
            raise ValueError('Long audio requires lossless original TTS text chunks')
        parts, reconstructed, frames = [], hashlib.sha256(), 0
        for i, text in enumerate(chunks):
            path = (run / 'audio_chunks' / row['key'] / f'{i:04d}.wav').resolve()
            chunk = wav_info(path, include_rms=False)
            if chunk['sample_rate'] != info['sample_rate'] or chunk['channels'] != info['channels']:
                raise ValueError('TTS chunk format differs from full audio')
            if chunk['bytes'] > settings['max_audio_bytes'] or chunk['duration_s'] > settings['max_audio_s']:
                raise ValueError(
                    'A TTS chunk exceeds the audio judge request limits; no audio or reference was truncated'
                )
            with wave.open(str(path), 'rb') as w:
                if w.getsampwidth() != 2:
                    raise ValueError('Expected PCM16 synthesis chunks')
                reconstructed.update(w.readframes(w.getnframes()))
            frames += chunk['frames']
            parts.append(
                {
                    'index': i,
                    'audio_path': str(path),
                    'audio_sha256': chunk['sha256'],
                    'frames': chunk['frames'],
                    'sample_rate': chunk['sample_rate'],
                    'duration_s': chunk['duration_s'],
                    'text': text,
                }
            )
        full_pcm = hashlib.sha256()
        with wave.open(str(audio), 'rb') as w:
            while data := w.readframes(1024 * 1024):
                full_pcm.update(data)
        if frames != info['frames'] or reconstructed.digest() != full_pcm.digest():
            raise ValueError('TTS chunks do not reconstruct every original audio frame')
    return {
        'id': row['id'],
        'key': row['key'],
        'dataset': row['dataset'],
        'source_status': row['source_status'],
        'text': row['text'],
        'source_sha256': tts['text_sha256'],
        'audio_path': str(audio),
        'audio_sha256': info['sha256'],
        'audio_duration_s': info['duration_s'],
        'parts': parts,
    }


def _prepare(run, output, model, concurrency, api_base, limit, keys):
    manifest = read_json(run / 'run_manifest.json')
    settings = _settings(manifest, model, concurrency, api_base)
    prompt = DEFAULT_PROMPT.read_text(encoding='utf-8')
    rows = list(read_jsonl(run / 'inputs.jsonl'))
    if len(rows) != manifest['records'] or len({r['key'] for r in rows}) != len(rows):
        raise ValueError('Input manifest denominator or identifiers are corrupt')
    if limit < 0 or (keys and limit):
        raise ValueError('Use a nonnegative --limit or --keys, not both')
    if keys:
        if len(keys) != len(set(keys)) or not set(keys) <= {r['key'] for r in rows}:
            raise ValueError('Requested audio judge keys are missing or duplicated')
        rows = [r for r in rows if r['key'] in set(keys)]
    elif limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError('No audio judge inputs selected')
    items = [_plan(run, row, settings) for row in rows]
    identity = {
        'schema_version': 1,
        'name': NAME,
        'score_scale': [1, 2, 3],
        'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
        'model': settings['model'],
        'api_base': settings['api_base'],
        'temperature': 0,
        'max_tokens': settings['max_tokens'],
        'retry_max_tokens': settings['retry_max_tokens'],
        'segment_aggregation': 'minimum score over losslessly matched TTS chunks',
        'input_manifest_sha256': file_sha(run / 'inputs.jsonl'),
        'items': json.loads(json.dumps(items)),
    }
    fingerprint = digest(identity)
    for item in items:
        item['fingerprint'] = digest([fingerprint, item['key']])
        for part in item['parts']:
            part['fingerprint'] = digest([item['fingerprint'], part])
    data = ''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in items).encode()
    saved_manifest = output / 'run_manifest.json'
    if saved_manifest.exists():
        saved = read_json(saved_manifest)
        if saved['fingerprint'] != fingerprint:
            raise ValueError('Audio judge inputs, model or prompt changed; choose a new --output-dir')
        if (
            file_sha(output / 'system_prompt.txt') != identity['prompt_sha256']
            or file_sha(output / 'inputs.jsonl') != hashlib.sha256(data).hexdigest()
        ):
            raise ValueError('Audio judge prompt/input snapshot was changed')
    else:
        atomic_bytes(output / 'system_prompt.txt', prompt.encode())
        atomic_bytes(output / 'inputs.jsonl', data)
        write_json(
            saved_manifest,
            {
                'fingerprint': fingerprint,
                'identity': identity,
                'settings': settings,
                'records': len(items),
                'scope': 'subset' if keys or limit else manifest['scope'],
                'datasets': sorted({item['dataset'] for item in items}),
                'implementation_sha256': file_sha(__file__),
                'uses_external_asr': False,
                'source_run': str(run),
                'input_snapshot_sha256': hashlib.sha256(data).hexdigest(),
            },
        )
    return settings, prompt, items


class _Pacer:
    def __init__(self, rate):
        self.interval, self.next, self.lock = 1 / rate, 0, threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next)
            self.next = slot + self.interval
        if slot > now:
            time.sleep(slot - now)


def _http_sender(settings):
    key = os.environ.get('OPENAI_API_KEY') or os.environ.get('NVIDIA_API_KEY')
    if not key:
        raise ValueError('Set OPENAI_API_KEY or NVIDIA_API_KEY in the child environment')

    def send(body):
        request = urllib.request.Request(
            settings['api_base'] + '/chat/completions',
            data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key},
        )
        try:
            with urllib.request.urlopen(request, timeout=settings['timeout_s']) as response:
                data = json.load(response)
                try:
                    cost = float(response.headers.get('x-litellm-response-cost'))
                except (TypeError, ValueError):
                    cost = None
                return data, cost
        except urllib.error.HTTPError as exc:
            detail = exc.read(400).decode(errors='replace').replace(key, '[REDACTED]')
            error = RuntimeError(f'HTTP {exc.code}: {detail}')
            error.retryable = exc.code in (429, 500, 502, 503, 504, 529)
            raise error from None

    return send


def _cached(path, fingerprint):
    if not path.exists():
        return None
    entry = read_json(path)
    if entry.get('fingerprint') != fingerprint:
        raise ValueError('Audio judge cache identity mismatch')
    if entry.get('status') != 'complete':
        return None
    result = parse_result(json.dumps(entry['result']))
    if digest(result) != entry.get('result_sha256'):
        raise ValueError('Audio judge result checksum mismatch')
    return entry


def _recovery_policy(output, settings, enable=False):
    """A recorded, compatible format retry applies only after normal length failures.

    Initial requests and the supplied scoring rubric do not change. Valid caches
    remain byte-for-byte intact; recovery changes only unscored requests after the
    original retry policy is exhausted. The appendix is separately fingerprinted.
    """
    manifest = read_json(output / 'run_manifest.json')
    path = output / 'output_recovery_policy.json'
    previous_policy = read_json(path) if path.exists() else None
    trigger_failures = []
    if previous_policy:
        scope_keys = previous_policy['scope_record_keys']
        if not set(scope_keys) <= {item['key'] for item in manifest['identity']['items']}:
            raise ValueError('Invalid output-recovery scope')
    else:
        for failed_path in sorted((output / 'segments').rglob('*.json')):
            failed = read_json(failed_path)
            history = failed.get('attempt_history', [])
            if (
                failed.get('status') == 'failed'
                and history
                and history[-1].get('raw_response', {}).get('choices', [{}])[0].get('finish_reason') == 'length'
            ):
                trigger_failures.append(failed)
        scope_keys = sorted({failed['key'] for failed in trigger_failures})
    spec = {
        'version': 1,
        'base_fingerprint': manifest['fingerprint'],
        'rubric_sha256': manifest['identity']['prompt_sha256'],
        'trigger': 'unscored segments in records with exhausted length failures; otherwise only after exhausted normal length retries',
        'scope_record_keys': scope_keys,
        'normal_attempts': settings['attempts'],
        'extra_attempts': 2,
        'max_tokens': settings['retry_max_tokens'],
        'instruction': FORMAT_RECOVERY,
        'heard_representation': 'literal excerpts for long repetition runs; full audio still judged',
    }
    if path.exists():
        policy = read_json(path)
        if policy != spec or manifest.get('compatible_output_recovery', {}).get('policy_sha256') != digest(policy):
            raise ValueError('Audio judge output-recovery policy was changed')
        return policy
    if manifest.get('compatible_output_recovery'):
        raise ValueError('Audio judge output-recovery policy is missing')
    if not enable:
        return None
    preserved = []
    for directory in ('cache', 'segments'):
        for cached in sorted((output / directory).rglob('*.json')):
            entry = read_json(cached)
            if entry.get('status') == 'complete':
                assert digest(entry['result']) == entry['result_sha256']
                if directory == 'segments':
                    assert entry['attempt_history'][-1]['raw_response']['choices'][0]['finish_reason'] == 'stop'
                preserved.append({'path': str(cached.relative_to(output)), 'sha256': file_sha(cached)})
    receipt = output / 'control/output_recovery_preservation.json'
    write_json(
        receipt,
        {
            'preserved': preserved,
            'policy': 'No successful grade or cache file is rewritten by the format-recovery extension',
        },
    )
    write_json(output / 'control/output_recovery_trigger_failures.json', trigger_failures)
    write_json(path, spec)
    manifest['compatible_output_recovery'] = {
        'policy_sha256': digest(spec),
        'preservation_receipt_sha256': file_sha(receipt),
        'implementation_sha256': file_sha(__file__),
    }
    write_json(output / 'run_manifest.json', manifest)
    return spec


def _check_recovery_entry(entry, policy, prompt, part):
    if entry.get('format_recovery'):
        expected = hashlib.sha256((prompt.format(text=part['text']) + FORMAT_RECOVERY).encode()).hexdigest()
        if (
            not policy
            or entry.get('recovery_policy_sha256') != digest(policy)
            or entry['attempt_history'][-1].get('prompt_sha256') != expected
        ):
            raise ValueError('Audio judge recovered response provenance is incomplete')


def _segment(item, part, output, settings, prompt, send, pacer, recovery=None):
    path = output / 'segments' / item['dataset'] / item['key'] / f"{part['index']:04d}.json"
    cached = _cached(path, part['fingerprint'])
    if cached:
        _check_recovery_entry(cached, recovery, prompt, part)
        return cached
    audio = Path(part['audio_path']).read_bytes()
    if hashlib.sha256(audio).hexdigest() != part['audio_sha256']:
        raise ValueError('Audio changed before its judge request')
    body = {
        'model': settings['model'],
        'temperature': 0,
        'max_tokens': settings['max_tokens'],
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt.format(text=part['text'])},
                    {
                        'type': 'input_audio',
                        'input_audio': {'data': base64.b64encode(audio).decode(), 'format': 'wav'},
                    },
                ],
            }
        ],
    }
    history = read_json(path).get('attempt_history', []) if path.exists() else []
    metadata = {
        'id': item['id'],
        'key': item['key'],
        'dataset': item['dataset'],
        'segment': part['index'],
        'fingerprint': part['fingerprint'],
        'audio_sha256': part['audio_sha256'],
        'source_sha256': hashlib.sha256(part['text'].encode()).hexdigest(),
        'model': settings['model'],
    }

    def completed(result, call, recovered=False):
        call['elapsed_s'] = time.monotonic() - started
        history.append(call)
        entry = {
            **metadata,
            'status': 'complete',
            'result': result,
            'result_sha256': digest(result),
            'attempt_history': history,
            'reported_cost_usd': sum(h.get('reported_cost_usd') or 0 for h in history),
        }
        if recovered:
            entry.update(format_recovery=True, heard_is_excerpt=True, recovery_policy_sha256=digest(recovery))
        write_json(path, entry)
        return entry

    previous_length = (
        bool(history) and history[-1].get('raw_response', {}).get('choices', [{}])[0].get('finish_reason') == 'length'
    )
    direct_recovery = bool(recovery and item['key'] in recovery['scope_record_keys'])
    normal_attempts = (
        0
        if direct_recovery or (recovery and previous_length and len(history) >= settings['attempts'])
        else settings['attempts']
    )
    for attempt in range(normal_attempts):
        pacer.wait()
        started = time.monotonic()
        call = {'attempt': len(history) + 1, 'max_tokens': body['max_tokens']}
        try:
            response, cost = send(body)
            call.update(raw_response=response, reported_cost_usd=cost)
            choice = (response.get('choices') or [{}])[0]
            if choice.get('finish_reason') != 'stop':
                raise ValueError(f"Audio judge did not finish normally: {choice.get('finish_reason')}")
            result = parse_result((choice.get('message') or {}).get('content'))
            return completed(result, call)
        except Exception as exc:
            call.update(error=f'{type(exc).__name__}: {exc}', elapsed_s=time.monotonic() - started)
            history.append(call)
            if isinstance(exc, ValueError):
                body['max_tokens'] = settings['retry_max_tokens']
            if not getattr(exc, 'retryable', True):
                break
            if attempt + 1 < settings['attempts']:
                time.sleep(min(2 ** (attempt + 1), 30))
    last_length = (
        bool(history) and history[-1].get('raw_response', {}).get('choices', [{}])[0].get('finish_reason') == 'length'
    )
    if recovery and (last_length or direct_recovery):
        body['max_tokens'] = recovery['max_tokens']
        body['messages'][0]['content'][0]['text'] = prompt.format(text=part['text']) + recovery['instruction']
        actual_prompt_sha = hashlib.sha256(body['messages'][0]['content'][0]['text'].encode()).hexdigest()
        for attempt in range(recovery['extra_attempts']):
            pacer.wait()
            started = time.monotonic()
            call = {
                'attempt': len(history) + 1,
                'max_tokens': body['max_tokens'],
                'format_recovery': True,
                'prompt_sha256': actual_prompt_sha,
            }
            try:
                response, cost = send(body)
                call.update(raw_response=response, reported_cost_usd=cost)
                choice = (response.get('choices') or [{}])[0]
                if choice.get('finish_reason') != 'stop':
                    raise ValueError(f"Audio judge format recovery did not finish: {choice.get('finish_reason')}")
                result = parse_result((choice.get('message') or {}).get('content'))
                return completed(result, call, recovered=True)
            except Exception as exc:
                call.update(error=f'{type(exc).__name__}: {exc}', elapsed_s=time.monotonic() - started)
                history.append(call)
                if not getattr(exc, 'retryable', True):
                    break
                if attempt + 1 < recovery['extra_attempts']:
                    time.sleep(2)
    write_json(path, {**metadata, 'status': 'failed', 'error': history[-1]['error'], 'attempt_history': history})
    raise ValueError(f"Missing audio judge segment for {item['dataset']}/{item['id']}; no placeholder score assigned")


def _one(item, output, settings, prompt, send, pacer, recovery=None):
    path = output / 'cache' / item['dataset'] / f"{item['key']}.json"
    if _cached(path, item['fingerprint']):
        return 1
    parts = [_segment(item, part, output, settings, prompt, send, pacer, recovery) for part in item['parts']]
    result = {
        'score': min(part['result']['score'] for part in parts),
        'heard': ' '.join(part['result']['heard'] for part in parts),
        'issues': (
            parts[0]['result']['issues']
            if len(parts) == 1
            else '; '.join(f"segment {i + 1}: {part['result']['issues']}" for i, part in enumerate(parts))
        ),
    }
    entry = {
        'id': item['id'],
        'key': item['key'],
        'dataset': item['dataset'],
        'source_status': item['source_status'],
        'status': 'complete',
        'fingerprint': item['fingerprint'],
        'model': settings['model'],
        'audio_sha256': item['audio_sha256'],
        'result': result,
        'result_sha256': digest(result),
        'segments': len(parts),
        'segment_result_sha256': [part['result_sha256'] for part in parts],
        'aggregation': 'minimum segment score' if len(parts) > 1 else 'single audio request',
        'heard_is_excerpt': any(part.get('heard_is_excerpt', False) for part in parts),
        'reported_cost_usd': sum(part['reported_cost_usd'] for part in parts),
    }
    write_json(path, entry)
    print(f"AUDIO_LLM {item['dataset']} {item['id']}: {result['score']}/3 ({len(parts)} request segments)", flush=True)
    return 1


def judge_audio_llm(
    run_dir,
    model=None,
    concurrency=None,
    output_dir=None,
    api_base=None,
    limit=0,
    keys=None,
    recover_length=False,
    *,
    send=None,
):
    run = Path(run_dir).resolve()
    output = Path(output_dir).resolve() if output_dir else run / 'semantic_audio_llm'
    settings, prompt, items = _prepare(run, output, model, concurrency, api_base, limit, keys)
    recovery = _recovery_policy(output, settings, recover_length)
    missing = [
        item
        for item in items
        if not _cached(output / 'cache' / item['dataset'] / f"{item['key']}.json", item['fingerprint'])
    ]
    if missing:
        sender = send or _http_sender(settings)
        pacer = _Pacer(settings['rate'])
        try:
            run_records(
                missing,
                lambda item: _one(item, output, settings, prompt, sender, pacer, recovery),
                settings['concurrency'],
            )
        finally:
            complete = sum(
                bool(_cached(output / 'cache' / item['dataset'] / f"{item['key']}.json", item['fingerprint']))
                for item in items
            )
            write_json(
                output / 'status.json',
                {'expected': len(items), 'complete': complete, 'missing': len(items) - complete},
            )
    else:
        write_json(output / 'status.json', {'expected': len(items), 'complete': len(items), 'missing': 0})
    result = report_audio_llm(output)
    atomic_bytes(output / 'evaluation_complete.txt', (datetime.now(timezone.utc).isoformat() + '\n').encode())
    return result


def report_audio_llm(output_dir):
    output = Path(output_dir)
    manifest = read_json(output / 'run_manifest.json')
    recovery = _recovery_policy(output, manifest['settings'])
    prompt = (output / 'system_prompt.txt').read_text(encoding='utf-8')
    if (
        file_sha(output / 'inputs.jsonl') != manifest['input_snapshot_sha256']
        or file_sha(output / 'system_prompt.txt') != manifest['identity']['prompt_sha256']
    ):
        raise ValueError('Audio judge source or prompt snapshot checksum mismatch')
    items = list(read_jsonl(output / 'inputs.jsonl'))
    grades, entries = [], []
    for item in items:
        entry = _cached(output / 'cache' / item['dataset'] / f"{item['key']}.json", item['fingerprint'])
        if entry is None:
            raise ValueError(
                f"Missing semantic audio LLM score: {item['dataset']}/{item['id']}; no incomplete report produced"
            )
        parts = [
            _cached(
                output / 'segments' / item['dataset'] / item['key'] / f"{part['index']:04d}.json", part['fingerprint']
            )
            for part in item['parts']
        ]
        if (
            any(part is None for part in parts)
            or [part['result_sha256'] for part in parts] != entry['segment_result_sha256']
        ):
            raise ValueError('Audio judge segment cache is missing or changed')
        if entry['result']['score'] != min(part['result']['score'] for part in parts):
            raise ValueError('Audio judge aggregate does not match segment scores')
        for saved, part in zip(parts, item['parts']):
            _check_recovery_entry(saved, recovery, prompt, part)
        grades.append(
            {
                'id': item['id'],
                'key': item['key'],
                'set': item['dataset'],
                'source_status': item['source_status'],
                'ref': item['text'],
                'audio_path': item['audio_path'],
                'audio_sha256': item['audio_sha256'],
                'audio_s': item['audio_duration_s'],
                **entry['result'],
                'segments': entry['segments'],
                'heard_is_excerpt': entry.get('heard_is_excerpt', False),
            }
        )
        entries.append(entry)
    if len(grades) != manifest['records']:
        raise ValueError('Audio judge denominator changed')

    def aggregate(rows):
        scores = [row['score'] for row in rows]
        return {
            'n': len(rows),
            'mean': statistics.mean(scores),
            'median': statistics.median(scores),
            'dist': {str(i): scores.count(i) for i in (1, 2, 3)},
        }

    per_set = {name: aggregate([row for row in grades if row['set'] == name]) for name in manifest['datasets']}
    pooled = aggregate(grades)
    summary = {
        'name': NAME,
        'scope': manifest['scope'],
        'model': manifest['settings']['model'],
        'score_scale': [1, 2, 3],
        'prompt_sha256': manifest['identity']['prompt_sha256'],
        'n_judged': len(grades),
        'n_failed': 0,
        'mean_score': pooled['mean'],
        'dist': pooled['dist'],
        'pooled': pooled,
        'per_set': per_set,
        'mean_of_dataset_means': statistics.mean(row['mean'] for row in per_set.values()),
        'audio_assessed': True,
        'uses_external_asr': False,
        'audio_requests': sum(entry['segments'] for entry in entries),
        'reported_judge_cost_usd': sum(entry['reported_cost_usd'] for entry in entries),
        'segmented_records': sum(entry['segments'] > 1 for entry in entries),
        'heard_excerpt_records': sum(row['heard_is_excerpt'] for row in grades),
        'output_recovery_policy': recovery,
        'segment_aggregation': manifest['identity']['segment_aggregation'],
        'source_status_counts': dict(Counter(row['source_status'] for row in grades)),
        'worst': sorted(
            [row for row in grades if row['score'] < 3], key=lambda row: (row['score'], -row['audio_s'], row['id'])
        )[:25],
    }
    write_json(output / 'judge.json', summary)
    write_json(output / 'reports/summary.json', summary)
    data = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in grades).encode()
    atomic_bytes(output / 'judge.utterances.jsonl', data)
    atomic_bytes(output / 'reports/grades.jsonl', data)
    atomic_bytes(
        output / 'judge.cache.jsonl', ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in entries).encode()
    )
    lines = [
        f"{NAME} ({manifest['scope']}, {len(grades)} replies)",
        f"Model: {summary['model']}; score scale 1-3; prompt {summary['prompt_sha256']}",
        f"Pooled mean: {pooled['mean']:.3f}/3; mean of dataset means: {summary['mean_of_dataset_means']:.3f}/3",
        f"Score counts: {pooled['dist']}; request segments: {summary['audio_requests']}",
    ]
    lines.extend(
        f"  {name}: n={row['n']}, mean={row['mean']:.3f}/3, counts={row['dist']}" for name, row in per_set.items()
    )
    lines += [
        'Reference text and original WAVs were judged directly; no external ASR transcript was supplied.',
        'Long clips use their exact original TTS chunks, with complete PCM/text coverage; overall score is the minimum segment score.',
    ]
    if recovery:
        lines.append(
            f"Output-format recovery: {summary['heard_excerpt_records']} replies contain marked heard excerpts for repetition runs; every audio segment was still judged in full."
        )
    atomic_bytes(output / 'reports/report.txt', ('\n'.join(lines) + '\n').encode())
    print('\n'.join(lines))
    return summary
