"""Use the pinned verbal judge on text and ASR in one fresh, paired pass."""

import argparse
import hashlib
import importlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path

from .core import atomic_bytes, read_json, read_jsonl, write_json


def load_judge(spec):
    transport = spec.get('transport_module')
    if transport:
        sys.modules['httpx'] = importlib.import_module(transport)
    module_spec = importlib.util.spec_from_file_location('pinned_verbal_judge', spec['script'])
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def build_judge_runs(run):
    manifest = read_json(run / 'run_manifest.json')
    records = list(read_jsonl(run / 'inputs.jsonl'))
    datasets = [s['name'] for s in manifest['config']['datasets']]
    for label in ('text', 'tts_asr'):
        for dataset in datasets:
            rows = []
            for r in records:
                if r['dataset'] != dataset:
                    continue
                answer = r['text']
                if label == 'tts_asr':
                    asr = read_json(run / 'asr' / f"{r['key']}.json")
                    if asr['status'] != 'complete':
                        raise ValueError('Cannot grade incomplete ASR outputs')
                    answer = asr['transcript']
                    if hashlib.sha256(answer.encode()).hexdigest() != asr.get('transcript_sha256'):
                        raise ValueError('ASR transcript checksum mismatch before judging')
                rows.append(
                    {
                        'canonical_sample_id': r['id'],
                        'sample_id': r['id'],
                        'candidate_answer': answer,
                        'candidate_status': r['source_status'],
                        'candidate_valid': r['source_valid'],
                        'verifier_metadata': {'subset': dataset, 'prompt': r['question']},
                        'roundtrip_key': r['key'],
                        'pipeline_status': 'complete',
                    }
                )
            path = run / 'judge_inputs' / label / 'eval' / 'roundtrip' / 'benchmarks' / dataset / 'rollouts.jsonl'
            atomic_bytes(path, ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode())
    return datasets


def judge(run_dir):
    run = Path(run_dir).resolve()
    spec = read_json(run / 'run_manifest.json')['config']['judge']
    datasets = build_judge_runs(run)
    module = load_judge(spec)
    args = argparse.Namespace(
        run=[f"{label}={run / 'judge_inputs' / label}" for label in ('text', 'tts_asr')],
        conds=['roundtrip'],
        sets=datasets,
        out=str(run / 'judge_cache'),
        prompt=spec['prompt'],
        model=spec['model'],
        temperature='none',
        max_tokens=spec.get('max_tokens', 2048),
        workers=spec.get('workers', 16),
        rate=spec.get('rate', 5),
        request_timeout=90,
        stall_timeout=240,
        stall_rounds=3,
        no_keepalive=True,
        limit=0,
    )
    for attempt in range(spec.get('missing_grade_retries', 3)):
        module.cmd_judge(args)
        try:
            report(run)
            return
        except MissingGrades:
            if attempt + 1 == spec.get('missing_grade_retries', 3):
                raise


class MissingGrades(ValueError):
    """Incomplete grades must never be silently dropped from a report."""


def report(run_dir):
    import hashlib

    run = Path(run_dir)
    manifest = read_json(run / 'run_manifest.json')
    spec = manifest['config']['judge']
    module = load_judge(spec)
    prompt_sha = hashlib.sha256(Path(spec['prompt']).read_bytes()).hexdigest()[:8]
    result = {
        'scope': manifest['scope'],
        'records': manifest['records'],
        'conversation_mode': manifest['conversation_mode'],
        'judge_model': spec['model'],
        'prompt_sha': prompt_sha,
        'aggregation': 'unweighted mean of dataset means; pooled median words',
        'delivery_interpretation': 'spoken wording recovered by ASR; not an acoustic/prosody assessment',
        'labels': {},
    }
    for label in ('text', 'tts_asr'):
        per_set, words, total_cost = {}, [], 0.0
        for dataset in [s['name'] for s in manifest['config']['datasets']]:
            path = run / 'judge_inputs' / label / 'eval' / 'roundtrip' / 'benchmarks' / dataset / 'rollouts.jsonl'
            rows = module.load_rollouts(path)
            cache = module.load_cache(
                module.cache_path(str(run / 'judge_cache'), label, 'roundtrip', dataset), spec['model']
            )
            grades = []
            for row in rows:
                entry = cache.get((row['id'], row['sha']))
                if entry is None:
                    raise MissingGrades(f'{label}/{dataset}: missing grade for {row["id"]}')
                if entry.get('prompt_sha') != prompt_sha or entry.get('temperature') is not None:
                    raise ValueError('Judge cache prompt/temperature mismatch')
                grades.append(entry['judge'])
                total_cost += entry.get('cost_usd') or 0
                words.append(row['words'])
            per_set[dataset] = {
                'n': len(rows),
                'median_words': statistics.median(r['words'] for r in rows),
                **{axis: 20 * statistics.mean(g[axis] for g in grades) for axis in module.AX},
            }
        if len(words) != manifest['records']:
            raise ValueError('Judge denominator differs from input manifest')
        result['labels'][label] = {
            'n': len(words),
            'median_words': statistics.median(words),
            **{axis: statistics.mean(s[axis] for s in per_set.values()) for axis in module.AX},
            'per_set': per_set,
            'reported_judge_cost_usd': total_cost,
        }
    result['delta'] = {
        key: result['labels']['tts_asr'][key] - result['labels']['text'][key]
        for key in ('content', 'delivery', 'overall', 'median_words')
    }
    write_json(run / 'reports' / 'summary.json', result)
    lines = [
        f"Speech roundtrip evaluation ({manifest['scope']}, {manifest['records']} replies)",
        f"Single turn: reference voice only; judge {spec['model']}; prompt {prompt_sha}",
        'Both labels graded in the same fresh pass; no earlier grades reused.',
        '',
    ]
    for label, data in result['labels'].items():
        lines.append(
            f"{label}: content {data['content']:.1f}; delivery {data['delivery']:.1f}; "
            f"overall {data['overall']:.1f}; median words {data['median_words']:g}; n={data['n']}"
        )
        for dataset, item in data['per_set'].items():
            lines.append(f"  {dataset}: content {item['content']:.1f}; delivery {item['delivery']:.1f}; n={item['n']}")
    lines += ['', result['delivery_interpretation']]
    if manifest['scope'] == 'preview':
        lines.append('Preview only: these scores are not a full-dataset benchmark result.')
    atomic_bytes(run / 'reports' / 'report.txt', ('\n'.join(lines) + '\n').encode())
    print('\n'.join(lines))
    return result
