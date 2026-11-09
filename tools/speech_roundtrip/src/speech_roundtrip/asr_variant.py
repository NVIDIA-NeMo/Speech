"""An isolated ASR evaluation over finalized speech from an existing run."""

import argparse
import copy
import hashlib
import json
import re
import statistics
from pathlib import Path

from .core import atomic_bytes, digest, file_sha, read_json, read_jsonl, write_json
from .judging import MissingGrades, load_judge


def prepare_variant(source_run, run_dir, checkpoint, name='Parakeet-TDT', batch_size=4, max_audio_s=180):
    source, run = Path(source_run).resolve(), Path(run_dir).resolve()
    if source == run or source in run.parents:
        raise ValueError('Use a separate run directory outside the source run')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in ('.', '..'):
        raise ValueError('Unsafe evaluation label')
    if type(batch_size) is not int or batch_size < 1 or max_audio_s <= 0:
        raise ValueError('Invalid batch size or audio limit')
    checkpoint = Path(checkpoint).resolve(strict=True)
    original = read_json(source / 'run_manifest.json')
    records = list(read_jsonl(source / 'inputs.jsonl'))
    if len(records) != original['records'] or len({r['key'] for r in records}) != len(records):
        raise ValueError('Source input denominator or keys changed')
    copies = []
    for row in records:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', row['key']):
            raise ValueError('Unsafe source key')
        path = source / 'tts' / (row['key'] + '.json')
        tts = read_json(path)
        if tts['status'] != 'complete' or tts['text_sha256'] != hashlib.sha256(row['text'].encode()).hexdigest():
            raise ValueError('ASR variants require every finalized TTS source')
        copies.append((path, file_sha(path)))
    stat = checkpoint.stat()
    config = copy.deepcopy(original['config'])
    config['asr'] = {
        'backend': 'parakeet_tdt',
        'model': name,
        'checkpoint': str(checkpoint),
        'batch_size': batch_size,
        'use_lhotse': False,
        'transcript_field': 'pred_text',
        'normalizer': "get_text_processor('en').process_text_for_wer",
        'normalization_scope': 'whole_reply',
        'max_single_request_audio_s': max_audio_s,
        'speech_revision': config['tts'].get('speech_revision'),
    }
    identity = {
        'kind': 'asr_variant',
        'name': name,
        'source_run': str(source),
        'source_manifest_sha256': file_sha(source / 'run_manifest.json'),
        'input_sha256': file_sha(source / 'inputs.jsonl'),
        'config': config,
        'checkpoint_stat': {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns},
        'tts_sidecars': [[p.name, checksum] for p, checksum in copies],
    }
    manifest = {
        **original,
        'fingerprint': digest(identity),
        'config': config,
        'evaluation_name': name,
        'artifact_kind': 'asr_variant',
        'variant_identity': identity,
    }
    if (run / 'run_manifest.json').exists():
        if read_json(run / 'run_manifest.json') != manifest:
            raise ValueError('ASR variant settings or source changed; choose a new run directory')
    else:
        if run.exists() and any(run.iterdir()):
            raise ValueError('New variant directory must be empty')
        write_json(run / 'run_manifest.json', manifest)
    data = (source / 'inputs.jsonl').read_bytes()
    if (run / 'inputs.jsonl').exists() and (run / 'inputs.jsonl').read_bytes() != data:
        raise ValueError('Variant inputs were changed')
    atomic_bytes(run / 'inputs.jsonl', data)
    for path, checksum in copies:
        target = run / 'tts' / path.name
        if target.exists() and file_sha(target) != checksum:
            raise ValueError('Variant TTS metadata changed')
        if not target.exists():
            atomic_bytes(target, path.read_bytes())
    return run


def validate_transcripts(run):
    manifest = read_json(run / 'run_manifest.json')
    if manifest.get('artifact_kind') != 'asr_variant':
        raise ValueError('Expected an isolated ASR variant')
    if file_sha(run / 'inputs.jsonl') != manifest['variant_identity']['input_sha256']:
        raise ValueError('ASR variant inputs changed')
    records = list(read_jsonl(run / 'inputs.jsonl'))
    if len(records) != manifest['records']:
        raise ValueError('ASR variant denominator changed')
    frozen = dict(manifest['variant_identity']['tts_sidecars'])
    for row in records:
        if file_sha(run / 'tts' / (row['key'] + '.json')) != frozen[row['key'] + '.json']:
            raise ValueError('Frozen TTS metadata changed')
        saved = read_json(run / 'asr' / (row['key'] + '.json'))
        tts = read_json(run / 'tts' / (row['key'] + '.json'))
        if saved['status'] != 'complete' or saved['audio_sha256'] != tts['audio']['sha256']:
            raise ValueError('ASR variant has an incomplete or mismatched transcript')
        if saved['model'] != manifest['evaluation_name'] or saved['transcript_field'] != 'pred_text':
            raise ValueError('Unexpected ASR identity or transcript representation')
        for field in ['transcript', 'raw_asr_text', 'pred_text']:
            if (
                not isinstance(saved[field], str)
                or hashlib.sha256(saved[field].encode()).hexdigest() != saved[field + '_sha256']
            ):
                raise ValueError('ASR variant transcript checksum mismatch')
        if saved['transcript'] != saved['pred_text']:
            raise ValueError('Judges must receive the selected normalized pred_text')
        row['asr'] = saved
    return manifest, records


def judge_variant(run_dir):
    run = Path(run_dir).resolve()
    manifest, rows = validate_transcripts(run)
    label, spec = manifest['evaluation_name'], manifest['config']['judge']
    datasets = [d['name'] for d in manifest['config']['datasets']]
    for dataset in datasets:
        output = []
        for row in rows:
            if row['dataset'] == dataset:
                output.append(
                    {
                        'canonical_sample_id': row['id'],
                        'sample_id': row['id'],
                        'candidate_answer': row['asr']['transcript'],
                        'candidate_status': row['source_status'],
                        'candidate_valid': row['source_valid'],
                        'verifier_metadata': {'subset': dataset, 'prompt': row['question']},
                        'roundtrip_key': row['key'],
                        'pipeline_status': 'complete',
                    }
                )
        target = run / 'judge_inputs' / 'eval' / 'roundtrip' / 'benchmarks' / dataset / 'rollouts.jsonl'
        data = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in output).encode()
        if target.exists() and target.read_bytes() != data:
            raise ValueError('Verbal judge inputs changed')
        atomic_bytes(target, data)
    pinned = load_judge(spec)
    args = argparse.Namespace(
        run=[f'{label}={run / "judge_inputs"}'],
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
        pinned.cmd_judge(args)
        try:
            return report_variant(run)
        except MissingGrades:
            if attempt + 1 == spec.get('missing_grade_retries', 3):
                raise


def report_variant(run_dir):
    run = Path(run_dir).resolve()
    manifest, records = validate_transcripts(run)
    spec, label = manifest['config']['judge'], manifest['evaluation_name']
    pinned = load_judge(spec)
    prompt_sha = file_sha(spec['prompt'])[:8]
    per_set, all_words, cost = {}, [], 0.0
    for dataset in [d['name'] for d in manifest['config']['datasets']]:
        path = run / 'judge_inputs/eval/roundtrip/benchmarks' / dataset / 'rollouts.jsonl'
        rows = pinned.load_rollouts(path)
        expected = sum(r['dataset'] == dataset for r in records)
        if len(rows) != expected:
            raise ValueError('Verbal judge loader dropped ASR rows')
        cache = pinned.load_cache(
            pinned.cache_path(str(run / 'judge_cache'), label, 'roundtrip', dataset), spec['model']
        )
        grades = []
        for row in rows:
            entry = cache.get((row['id'], row['sha']))
            if entry is None:
                raise MissingGrades(f'{label}/{dataset}: missing grade for {row["id"]}')
            if entry.get('prompt_sha') != prompt_sha or entry.get('temperature') is not None:
                raise ValueError('Verbal judge cache prompt or temperature changed')
            grades.append(entry['judge'])
            cost += entry.get('cost_usd') or 0
            all_words.append(row['words'])
        per_set[dataset] = {
            'n': len(rows),
            'median_words': statistics.median(r['words'] for r in rows),
            **{axis: 20 * statistics.mean(g[axis] for g in grades) for axis in pinned.AX},
        }
    if len(all_words) != manifest['records']:
        raise ValueError('Verbal judge denominator changed')
    result = {
        'name': label,
        'scope': manifest['scope'],
        'records': len(all_words),
        'judge_model': spec['model'],
        'prompt_sha': prompt_sha,
        'asr': manifest['config']['asr'],
        'aggregation': 'unweighted mean of dataset means; pooled median words',
        'transcript_representation': 'English-normalized pred_text',
        'median_words': statistics.median(all_words),
        'per_set': per_set,
        **{axis: statistics.mean(s[axis] for s in per_set.values()) for axis in pinned.AX},
        'reported_judge_cost_usd': cost,
        'delivery_interpretation': 'Spoken wording recovered by ASR; not an acoustic/prosody assessment',
    }
    write_json(run / 'reports/verbal_summary.json', result)
    lines = [
        f'{label} ({len(all_words)} replies)',
        f"Content {result['content']:.3f}; delivery {result['delivery']:.3f}; median words {result['median_words']:g}",
        'Input: pred_text from the requested English process_text_for_wer postprocessor.',
    ]
    lines.extend(
        f"  {name}: n={r['n']}, content {r['content']:.3f}, delivery {r['delivery']:.3f}, median words {r['median_words']:g}"
        for name, r in per_set.items()
    )
    atomic_bytes(run / 'reports/verbal_report.txt', ('\n'.join(lines) + '\n').encode())
    print('\n'.join(lines))
    return result
