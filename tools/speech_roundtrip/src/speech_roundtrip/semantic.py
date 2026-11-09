"""Score semantic preservation of complete TTS source/ASR text pairs."""

import hashlib
import importlib
import json
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .core import atomic_bytes, digest, file_sha, read_json, read_jsonl, run_records, write_json
from .judging import load_judge


DEFAULT_PROMPT = Path(__file__).parent / 'prompts' / 'semantic_preservation.txt'
RESULT_KEYS = {
    'score_1_to_5',
    'categories_checked',
    'meaning_errors',
    'uncertainty_notes',
    'wording_notes',
    'observability_limits',
    'error_attribution',
    'audio_assessed',
    'summary',
}
ERROR_TYPES = {
    'minor_nuance_changed',
    'value_changed',
    'unit_or_currency_changed',
    'identifier_changed',
    'date_or_time_changed',
    'negation_or_modality_changed',
    'entity_or_relationship_changed',
    'omission',
    'addition',
    'wrong_contextual_interpretation',
}


def validate_result(result, source, transcript):
    """Reject malformed grades, invented quotations and contradictory score fields."""
    if not isinstance(result, dict) or set(result) != RESULT_KEYS:
        raise ValueError('Semantic result must have exactly the score-only prompt schema')
    score = result['score_1_to_5']
    if type(score) is not int or not 1 <= score <= 5:
        raise ValueError('Semantic score must always be an integer from 1 to 5')
    if result['audio_assessed'] is not False or result['error_attribution'] != 'UNDETERMINED_FROM_TEXT':
        raise ValueError('A text-pair judge cannot claim to assess audio or attribute its cause')
    for key in ('categories_checked', 'uncertainty_notes', 'wording_notes', 'observability_limits'):
        if not isinstance(result[key], list) or any(not isinstance(s, str) or not s.strip() for s in result[key]):
            raise ValueError(f'{key} must be a list of nonempty strings')
    if not isinstance(result['summary'], str) or not result['summary'].strip():
        raise ValueError('Semantic summary must be a nonempty string')
    errors = result['meaning_errors']
    if not isinstance(errors, list) or (score == 5 and errors) or (score < 5 and not errors):
        raise ValueError('Score 5 has no meaning errors; scores 1-4 require concrete errors')
    for error in errors:
        if not isinstance(error, dict) or set(error) != {'type', 'source_evidence', 'asr_evidence', 'explanation'}:
            raise ValueError('Malformed semantic error evidence')
        if error['type'] not in ERROR_TYPES or (score == 4 and error['type'] != 'minor_nuance_changed'):
            raise ValueError('Score 4 permits only minor_nuance_changed errors')
        if not isinstance(error['explanation'], str) or not error['explanation'].strip():
            raise ValueError('Every meaning error needs a concise explanation')
        evidence = 0
        for key, text in [('source_evidence', source), ('asr_evidence', transcript)]:
            quote = error[key]
            if quote is not None:
                if not isinstance(quote, str) or not quote.strip() or quote not in text:
                    raise ValueError(f'{key} must be a literal, nonempty input substring or null')
                evidence += 1
        if not evidence:
            raise ValueError('Every meaning error requires literal evidence from at least one input')
    return result


def _prepare(run, output, prompt_path, model, workers):
    manifest = read_json(run / 'run_manifest.json')
    spec = manifest['config'].get('semantic_judge', {})
    prompt_path = Path(prompt_path or spec.get('prompt') or DEFAULT_PROMPT)
    prompt_bytes = prompt_path.read_bytes()
    settings = {
        'model': model or spec.get('model') or manifest['config']['judge']['model'],
        'prompt_sha256': hashlib.sha256(prompt_bytes).hexdigest(),
        'temperature': None,
        'max_tokens': spec.get('max_tokens', 8192),
        'workers': (
            workers if workers is not None else spec.get('workers', manifest['config']['judge'].get('workers', 16))
        ),
        'rate': spec.get('rate', 5),
        'request_timeout_s': spec.get('request_timeout_s', 180),
        'attempts_per_item': spec.get('attempts_per_item', 4),
    }
    if type(settings['workers']) is not int or not 1 <= settings['workers'] <= 32:
        raise ValueError('Semantic workers must be an integer from 1 to 32')
    pairs = []
    for row in read_jsonl(run / 'inputs.jsonl'):
        tts = read_json(run / 'tts' / f"{row['key']}.json")
        asr = read_json(run / 'asr' / f"{row['key']}.json")
        source_sha = hashlib.sha256(row['text'].encode()).hexdigest()
        if tts['status'] != 'complete' or tts['text_sha256'] != source_sha:
            raise ValueError('Semantic source does not match the finalized TTS input')
        if asr['status'] != 'complete' or asr['audio_sha256'] != tts['audio']['sha256']:
            raise ValueError('Semantic judging requires complete ASR of the finalized speech')
        transcript = asr['transcript']
        transcript_sha = hashlib.sha256(transcript.encode()).hexdigest()
        if transcript_sha != asr['transcript_sha256']:
            raise ValueError('ASR transcript checksum mismatch before semantic judging')
        pairs.append(
            {
                'id': row['id'],
                'key': row['key'],
                'dataset': row['dataset'],
                'source_status': row['source_status'],
                'source_text': row['text'],
                'asr_transcript': transcript,
                'source_sha256': source_sha,
                'transcript_sha256': transcript_sha,
            }
        )
    if len(pairs) != manifest['records']:
        raise ValueError('Semantic pair denominator differs from the input manifest')
    identity = {
        'schema_version': 1,
        'model': settings['model'],
        'prompt_sha256': settings['prompt_sha256'],
        'max_tokens': settings['max_tokens'],
        'temperature': None,
        'complete_text_pairs': True,
        'input_manifest_sha256': file_sha(run / 'inputs.jsonl'),
        'pairs': [[p['key'], p['source_sha256'], p['transcript_sha256']] for p in pairs],
    }
    fingerprint = digest(identity)
    path = output / 'run_manifest.json'
    if path.exists():
        previous = read_json(path)
        if previous['fingerprint'] != fingerprint:
            raise ValueError('Semantic inputs/prompt/model changed; choose a new --output-dir')
        if file_sha(output / 'system_prompt.txt') != settings['prompt_sha256']:
            raise ValueError('Semantic system prompt snapshot checksum mismatch')
    else:
        atomic_bytes(output / 'system_prompt.txt', prompt_bytes)
        write_json(
            path,
            {
                'fingerprint': fingerprint,
                'identity': identity,
                'settings': settings,
                'records': len(pairs),
                'scope': manifest['scope'],
                'implementation_sha256': file_sha(__file__),
                'datasets': [d['name'] for d in manifest['config']['datasets']],
            },
        )
    atomic_bytes(output / 'inputs.jsonl', ''.join(json.dumps(p, ensure_ascii=False) + '\n' for p in pairs).encode())
    for pair in pairs:
        pair['fingerprint'] = digest([fingerprint, pair['key'], pair['source_sha256'], pair['transcript_sha256']])
    return manifest, settings, prompt_bytes.decode('utf-8'), pairs


def _cache_path(output, pair):
    return output / 'cache' / pair['dataset'] / f"{pair['key']}.json"


def _cached(output, pair):
    path = _cache_path(output, pair)
    if not path.exists():
        return None
    entry = read_json(path)
    if entry.get('fingerprint') != pair['fingerprint']:
        raise ValueError('Semantic cache identity mismatch')
    if entry.get('status') != 'complete':
        return None
    validate_result(entry['result'], pair['source_text'], pair['asr_transcript'])
    if entry.get('result_sha256') != digest(entry['result']):
        raise ValueError('Semantic cached result checksum mismatch')
    return entry


def _one(client, pacer, settings, prompt, output, pair):
    if _cached(output, pair) is not None:
        return 1
    path = _cache_path(output, pair)
    prior = read_json(path).get('attempt_history', []) if path.exists() else []
    history = list(prior)
    entry = {
        'id': pair['id'],
        'key': pair['key'],
        'dataset': pair['dataset'],
        'fingerprint': pair['fingerprint'],
        'model': settings['model'],
        'prompt_sha256': settings['prompt_sha256'],
        'source_status': pair['source_status'],
    }
    # Only these two strings are sent. No question, generator identity, grades or hints.
    content = json.dumps(
        {'source_text': pair['source_text'], 'asr_transcript': pair['asr_transcript']}, ensure_ascii=False
    )
    for attempt in range(settings['attempts_per_item']):
        if pacer is not None:
            pacer.wait()
        started = time.monotonic()
        call = {'attempt': len(history) + 1}
        try:
            response = client.chat.completions.with_raw_response.create(
                model=settings['model'],
                max_tokens=settings['max_tokens'],
                messages=[{'role': 'system', 'content': prompt}, {'role': 'user', 'content': content}],
            )
            parsed = response.parse()
            choice = parsed.choices[0]
            call.update(
                raw_output=choice.message.content,
                finish_reason=choice.finish_reason,
                cost_usd=(
                    float(response.headers['x-litellm-response-cost'])
                    if response.headers.get('x-litellm-response-cost')
                    else None
                ),
                input_tokens=getattr(parsed.usage, 'prompt_tokens', None),
                output_tokens=getattr(parsed.usage, 'completion_tokens', None),
            )
            if choice.finish_reason != 'stop':
                raise ValueError(f'Semantic judge did not finish normally: {choice.finish_reason}')
            result = validate_result(json.loads(choice.message.content), pair['source_text'], pair['asr_transcript'])
            call['latency_s'] = time.monotonic() - started
            history.append(call)
            if pacer is not None:
                pacer.ok()
            write_json(
                path,
                {
                    **entry,
                    'status': 'complete',
                    'result': result,
                    'result_sha256': digest(result),
                    'attempt_history': history,
                    'reported_cost_usd': sum(h.get('cost_usd') or 0 for h in history),
                },
            )
            print(f"SEM {pair['dataset']} {pair['id']}: {result['score_1_to_5']}/5", flush=True)
            return 1
        except Exception as exc:
            call.update(error=f'{type(exc).__name__}: {str(exc)[:400]}', latency_s=time.monotonic() - started)
            history.append(call)
            if pacer is not None and any(s in call['error'] for s in ('429', 'RateLimit', 'Budget')):
                pacer.throttled()
            if attempt + 1 < settings['attempts_per_item']:
                time.sleep(min(2**attempt, 8))
    write_json(path, {**entry, 'status': 'failed', 'error': history[-1]['error'], 'attempt_history': history})
    print(f"SEM failed {pair['dataset']} {pair['id']}: {history[-1]['error']}", flush=True)
    return 0


def judge_semantic(run_dir, prompt_path=None, model=None, workers=None, output_dir=None, *, client=None, pacer=None):
    run = Path(run_dir).resolve()
    output = Path(output_dir).resolve() if output_dir else run / 'semantic_judge'
    manifest, settings, prompt, pairs = _prepare(run, output, prompt_path, model, workers)
    own_client = client is None
    if own_client:
        spec = manifest['config']['judge']
        pinned = load_judge(spec)
        transport = importlib.import_module(spec.get('transport_module', 'httpx'))
        from openai import OpenAI

        http = transport.Client(
            timeout=transport.Timeout(settings['request_timeout_s'], connect=20),
            limits=transport.Limits(max_connections=settings['workers'] + 8, max_keepalive_connections=0),
        )
        client = OpenAI(api_key=pinned.get_key(), base_url=pinned.NV_BASE_URL, max_retries=0, http_client=http)
        pacer = pinned.Pacer(settings['rate'])
    try:
        complete = run_records(pairs, lambda p: _one(client, pacer, settings, prompt, output, p), settings['workers'])
    finally:
        if own_client:
            client.close()
    write_json(output / 'status.json', {'expected': len(pairs), 'complete': complete, 'failed': len(pairs) - complete})
    result = report_semantic(output)
    atomic_bytes(output / 'evaluation_complete.txt', (datetime.now(timezone.utc).isoformat() + '\n').encode())
    return result


def report_semantic(output_dir):
    output = Path(output_dir)
    manifest = read_json(output / 'run_manifest.json')
    pairs = list(read_jsonl(output / 'inputs.jsonl'))
    entries = []
    for pair in pairs:
        pair['fingerprint'] = digest(
            [manifest['fingerprint'], pair['key'], pair['source_sha256'], pair['transcript_sha256']]
        )
        entry = _cached(output, pair)
        if entry is None:
            raise ValueError(
                f"Missing semantic score for {pair['dataset']}/{pair['id']}; no incomplete report produced"
            )
        entries.append(entry)
    if len(entries) != manifest['records']:
        raise ValueError('Semantic score denominator differs from its manifest')

    def aggregate(rows):
        scores = [r['result']['score_1_to_5'] for r in rows]
        counts = Counter(scores)
        return {
            'n': len(rows),
            'mean_score_1_to_5': statistics.mean(scores) if scores else None,
            'median_score_1_to_5': statistics.median(scores) if scores else None,
            'score_counts': {str(i): counts[i] for i in range(1, 6)},
            'with_uncertainty_notes': sum(bool(r['result']['uncertainty_notes']) for r in rows),
            'with_observability_limits': sum(bool(r['result']['observability_limits']) for r in rows),
        }

    per_set = {name: aggregate([e for e in entries if e['dataset'] == name]) for name in manifest['datasets']}
    means = [v['mean_score_1_to_5'] for v in per_set.values()]
    result = {
        'scope': manifest['scope'],
        'records': len(entries),
        'judge_model': manifest['settings']['model'],
        'prompt_sha256': manifest['settings']['prompt_sha256'],
        'scale': 'ordinal integers 1 to 5',
        'aggregation': 'unweighted mean of dataset means; pooled median and score counts',
        'mean_score_1_to_5': statistics.mean(means) if means and all(x is not None for x in means) else None,
        'pooled': aggregate(entries),
        'per_set': per_set,
        'audio_assessed': False,
        'error_attribution': 'UNDETERMINED_FROM_TEXT',
        'complete_text_pairs': True,
        'reported_judge_cost_usd': sum(e.get('reported_cost_usd') or 0 for e in entries),
    }
    write_json(output / 'reports/summary.json', result)
    atomic_bytes(
        output / 'reports/grades.jsonl', ''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in entries).encode()
    )
    mean = f"{result['mean_score_1_to_5']:.3f}" if result['mean_score_1_to_5'] is not None else 'N/A'
    lines = [
        f"Semantic preservation ({manifest['scope']}, {len(entries)} replies)",
        f"Judge: {result['judge_model']}; prompt SHA256 {result['prompt_sha256']}",
        f"Mean score: {mean}/5 (unweighted dataset mean); pooled median: {result['pooled']['median_score_1_to_5']}",
        f"Score counts: {result['pooled']['score_counts']}",
    ]
    for name, item in per_set.items():
        lines.append(
            f"  {name}: mean {item['mean_score_1_to_5']}; median {item['median_score_1_to_5']}; n={item['n']}"
        )
    lines += [
        'Every reply has an integer score. Complete text pairs were supplied without truncation.',
        'Text comparison only; uncertainty and observability notes do not remove rows or replace scores.',
    ]
    if manifest['scope'] == 'preview':
        lines.append('Preview only: not a full-benchmark result.')
    atomic_bytes(output / 'reports/report.txt', ('\n'.join(lines) + '\n').encode())
    print('\n'.join(lines))
    return result
