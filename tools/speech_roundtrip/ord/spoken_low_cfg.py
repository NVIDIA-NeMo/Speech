"""Prepare, submit and reproduce the complete spoken_low TTS/Parakeet evaluation on ORD."""

import argparse
import importlib.util
import json
import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path

import yaml

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'src'))
from speech_roundtrip.asr_variant import judge_variant, prepare_variant
from speech_roundtrip.core import file_sha, prepare, read_json, read_jsonl, write_json
from speech_roundtrip.semantic import judge_semantic
from speech_roundtrip.tts import synthesize


def tts_complete(path):
    status = path / 'tts_status.json'
    return status.exists() and read_json(status)['status'] == 'complete'


def prepare_run(root, run, recipe_path):
    root, run = Path(root).resolve(), Path(run).resolve()
    recipe = read_json(recipe_path)
    speech = PACKAGE.parents[1]
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=speech, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=speech, text=True).strip():
        raise ValueError('Commit the evaluation code before preparing an immutable run')
    model = root / recipe['model_dir']
    receipt = read_json(model / 'CONVERSION_COMPLETE.json')
    command = receipt['conversion_command']
    checkpoint = command[command.index('--nemo_file') + 1]
    old = root / 'spoken_low_replication_20261105'
    runtime = root / 'speech_roundtrip_runtime'
    config = {
        'schema_version': 1,
        'datasets': [
            {
                'name': name,
                'format': 'audio_evals',
                'path': str(root / recipe['input_run'] / 'eval/spoken_low/benchmarks' / name / 'rollouts.jsonl'),
                'expected_rows': count,
            }
            for name, count in recipe['datasets'].items()
        ],
        'tts': {
            'url': 'http://127.0.0.1:8091',
            'implementation': 'easymagpie_vllm_omni',
            'checkpoint': checkpoint,
            'converted_model': str(model),
            'reference_audio': str(root.parent / 'Mg_a_00759.wav'),
            'reference_sha256': recipe['voice_sha256'],
            'voice': recipe['voice'],
            'sample_rate': 22050,
            'speech_revision': revision,
            'vllm_version': '0.24.0',
            'vllm_omni_version': '0.24.0',
            'max_chars_per_chunk': recipe['max_chars_per_chunk'],
            'max_new_tokens': recipe['max_new_tokens'],
            'request_timeout_s': 600,
            'startup_timeout_s': 1200,
            'concurrency': recipe['concurrency'],
            'request_options': {
                'extra_params': {
                    'temperature': recipe['temperature'],
                    'top_k': recipe['top_k'],
                    'cfg_scale': recipe['cfg_scale'],
                }
            },
        },
        'asr': {'backend': 'parakeet_tdt', 'checkpoint': recipe['asr_checkpoint']},
        'judge': {
            'script': str(old / 'code/audio-reasoning-length-control/tools/judge_verbal_rollouts.py'),
            'prompt': str(old / 'code/audio-reasoning-length-control/configs/spoken/judge_verbal_open_v1.yaml'),
            'model': recipe['judge_model'],
            'transport_module': 'httpx2',
            'max_tokens': 2048,
            'workers': 16,
            'rate': 5,
            'missing_grade_retries': 3,
        },
        'semantic_judge': {'model': recipe['judge_model'], 'workers': 16, 'rate': 5, 'max_tokens': 8192},
    }
    plan = {
        'root': str(root),
        'run': str(run),
        'speech': str(speech),
        'speech_revision': revision,
        'recipe': recipe,
        'model_receipt_sha256': file_sha(model / 'CONVERSION_COMPLETE.json'),
        'module_source_sha256': {
            str(p.relative_to(PACKAGE)): file_sha(p)
            for p in PACKAGE.rglob('*')
            if p.is_file()
            and '__pycache__' not in p.parts
            and p.suffix in {'.py', '.txt', '.yaml', '.json', '.toml', '.md'}
        },
    }
    if (run / 'plan.json').exists():
        if read_json(run / 'plan.json') != plan:
            raise ValueError('Run settings changed; choose a new run directory')
    else:
        if run.exists() and any(run.iterdir()):
            raise ValueError('New run directory must be empty')
        write_json(run / 'plan.json', plan)
    for name in ['logs', 'control', 'cache']:
        (run / name).mkdir(exist_ok=True)
    config_path = run / 'config.yaml'
    text = yaml.safe_dump(config, sort_keys=False)
    if config_path.exists() and config_path.read_text() != text:
        raise ValueError('Resolved configuration changed')
    config_path.write_text(text)
    prepare(config_path, run / 'tts_run')
    prepare(config_path, run / 'preview', preview_per_dataset=1)
    deployment = yaml.safe_load((speech / 'tools/easymagpie_vllm_omni/deploy/easymagpie_cfg.yaml').read_text())
    deployment['stages'][0]['seed'] = 0
    deployment['stages'][0]['max_model_len'] = 4096
    deployment['stages'][0]['hf_overrides']['easymagpie_cfg_scale'] = recipe['cfg_scale']
    deployment['stages'][1]['max_model_len'] = 4096
    deployment['stages'][1]['max_num_batched_tokens'] = 4096
    (run / 'control/deploy.yaml').write_text(yaml.safe_dump(deployment, sort_keys=False))
    write_json(run / 'control/model_receipt.json', receipt)
    write_json(
        run / 'control/runtime_paths.json',
        {
            'runtime': str(runtime),
            'key_loader': str(old / 'scripts/with_ord_key.py'),
            'judge_python': str(old / 'code/audio-evals/.venv/bin/python'),
        },
    )
    print('PREPARED', run, 'REPLIES', read_json(run / 'tts_run/run_manifest.json')['records'])


def validate_sources(run):
    plan = read_json(run / 'plan.json')
    speech = Path(plan['speech'])
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=speech, text=True).strip() != plan['speech_revision']:
        raise ValueError('The pinned Speech revision changed')
    for name, checksum in plan['module_source_sha256'].items():
        if file_sha(PACKAGE / name) != checksum:
            raise ValueError('Evaluation source changed: ' + name)
    model = Path(plan['root']) / plan['recipe']['model_dir']
    if file_sha(model / 'CONVERSION_COMPLETE.json') != plan['model_receipt_sha256']:
        raise ValueError('Converted-model receipt changed')
    return plan


def run_gpu(run):
    import importlib.metadata

    import torch

    plan = validate_sources(run)
    root, speech, recipe = Path(plan['root']), Path(plan['speech']), plan['recipe']
    runtime = root / 'speech_roundtrip_runtime'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    model = root / recipe['model_dir']
    for name, checksum in read_json(model / 'CONVERSION_COMPLETE.json')['export_sha256'].items():
        if file_sha(model / name) != checksum:
            raise ValueError('Converted model changed: ' + name)
    write_json(
        run / 'control/gpu_runtime.json',
        {
            'gpu': torch.cuda.get_device_name(0),
            'torch': torch.__version__,
            'cuda': torch.version.cuda,
            'job_id': os.environ.get('SLURM_JOB_ID'),
            'node': socket.gethostname(),
            'vllm': importlib.metadata.version('vllm'),
            'vllm_omni': importlib.metadata.version('vllm-omni'),
            'container': 'nvcr.io/nvidia/nemo-speech:26.07.00',
            'speech_revision': plan['speech_revision'],
        },
    )
    spec = importlib.util.spec_from_file_location(
        'cfg_helpers', speech / 'tools/easymagpie_vllm_omni/scripts/benchmark_cfg.py'
    )
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    # Trace only the three diagnostic replies; use the production sampler for the full dataset.
    for traced in [True, False]:
        if not traced and tts_complete(run / 'tts_run'):
            continue
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        url = f'http://127.0.0.1:{port}'
        env = {
            **os.environ,
            'VLLM_PLUGINS': 'easymagpie_omni',
            'EASYMAGPIE_CFG_ENABLED': '1',
            'CFG_SCALE': str(recipe['cfg_scale']),
        }
        if traced:
            env['EASYMAGPIE_CFG_TRACE_PATH'] = str(run / 'control/cfg_trace.jsonl')
        else:
            env.pop('EASYMAGPIE_CFG_TRACE_PATH', None)
        if traced and tts_complete(run / 'preview'):
            continue
        command = [
            str(runtime / 'envs/tts-vllm/bin/vllm'),
            'serve',
            str(model),
            '--deploy-config',
            str(run / 'control/deploy.yaml'),
            '--host',
            '127.0.0.1',
            '--port',
            str(port),
            '--trust-remote-code',
            '--disable-log-stats',
            '--disable-uvicorn-access-log',
            '--uvicorn-log-level',
            'warning',
            '--omni',
        ]
        label = 'preview' if traced else 'full'
        write_json(run / f'control/{label}_server_command.json', command)
        with (run / f'logs/tts_{label}_server.log').open('w') as log:
            server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                helpers._wait_ready(server, url, 1200)
                os.environ['SPEECH_ROUNDTRIP_TTS_URL'] = url
                synthesize(run / ('preview' if traced else 'tts_run'))
            finally:
                helpers._stop(server)
    events = list(read_jsonl(run / 'control/cfg_trace.jsonl'))
    if not events or not all(
        e['cfg_active'] and e['cfg_scale'] == recipe['cfg_scale'] and e['guided_conditional_logits_max_abs'] > 0
        for e in events
    ):
        raise ValueError('The diagnostic run did not verify active guidance')
    variant = run / 'Parakeet-TDT'
    prepare_variant(
        run / 'tts_run', variant, recipe['asr_checkpoint'], name=recipe['name'] + '-Parakeet-TDT', batch_size=4
    )
    env = {**os.environ, 'PYTHONPATH': str(root / recipe['asr_source']) + ':' + str(PACKAGE / 'src')}
    with (run / 'logs/parakeet.log').open('a') as log:
        subprocess.run(
            [
                str(runtime / 'envs/parakeet/bin/python'),
                '-m',
                'speech_roundtrip',
                'transcribe',
                '--run-dir',
                str(variant),
            ],
            env=env,
            cwd=variant,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    write_json(
        run / 'control/gpu_complete.json',
        {
            'complete': True,
            'records': read_json(variant / 'run_manifest.json')['records'],
            'guidance_verified_requests': len(events),
            'source_revision': plan['speech_revision'],
        },
    )


def run_judges(run):
    validate_sources(run)
    if not (run / 'control/gpu_complete.json').exists():
        raise ValueError('The complete GPU stages are required before judging')
    variant = run / 'Parakeet-TDT'
    judge_variant(variant)
    judge_semantic(variant)
    semantic = read_json(variant / 'semantic_judge/reports/summary.json')
    verbal = read_json(variant / 'reports/verbal_summary.json')
    counts = semantic['pooled']['score_counts']
    write_json(
        run / 'summary.json',
        {
            'model_label': read_json(run / 'plan.json')['recipe']['model_label'],
            'records': verbal['records'],
            'content': verbal['content'],
            'delivery': verbal['delivery'],
            'median_words': verbal['median_words'],
            'semantic_mean_1_to_5': semantic['mean_score_1_to_5'],
            'semantic_score_counts': counts,
            'semantic_percent_5': 100 * counts.get('5', counts.get(5, 0)) / verbal['records'],
            'verbal': verbal,
            'semantic': semantic,
            'delivery_interpretation': 'English-normalized ASR wording; not an acoustic prosody score',
        },
    )
    (run / 'evaluation_complete.txt').write_text('All replies have Content/Delivery and 1–5 semantic grades.\n')


def submit(run, account, partition):
    plan = validate_sources(run)
    root = Path(plan['root'])
    runtime = root / 'speech_roundtrip_runtime'
    paths = read_json(run / 'control/runtime_paths.json')
    script = Path(__file__).resolve()
    quote = shlex.quote
    inside = f'''#!/usr/bin/env bash
set -euo pipefail
export ROOT={quote(str(root))} RUN={quote(str(run))}
export PYTHONPATH={quote(str(Path(plan['speech'])/'tools/easymagpie_vllm_omni'))}:{quote(plan['speech'])}:{quote(str(PACKAGE/'src'))}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
export HF_HOME="$ROOT/speech_roundtrip_runtime/cache/huggingface" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export XDG_CACHE_HOME="$ROOT/speech_roundtrip_runtime/cache" VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false TMPDIR=/rt
export VLLM_CACHE_ROOT="$RUN/cache/vllm" TORCHINDUCTOR_CACHE_DIR="$RUN/cache/inductor" TRITON_CACHE_DIR="$RUN/cache/triton"
exec {quote(str(runtime/'envs/tts-vllm/bin/python'))} {quote(str(script))} gpu --run-dir "$RUN"
'''
    (run / 'control/gpu_inside.sh').write_text(inside)
    batch = f'''#!/usr/bin/env bash
#SBATCH --job-name=v11_cfg25_spoken_low
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --exclude=batch-block1-0062
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output={run}/logs/gpu_%j.out
#SBATCH --error={run}/logs/gpu_%j.err
set -euo pipefail
export ENROOT_CACHE_PATH={runtime}/cache/enroot ENROOT_DATA_PATH={runtime}/enroot-data ENROOT_TEMP_PATH={runtime}/tmp
NODE_TMP="/var/tmp/spoken-v11-${{SLURM_JOB_ID}}"
mkdir -p "$NODE_TMP"
srun --container-image={quote(str(runtime/'containers/nemo-speech-26.07.00.sqsh'))} --container-mounts="/lustre:/lustre,$NODE_TMP:/rt" --no-container-mount-home bash {quote(str(run/'control/gpu_inside.sh'))}
'''
    judge = f'''#!/usr/bin/env bash
#SBATCH --job-name=v11_cfg25_judges
#SBATCH --account={account}
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output={run}/logs/judges_%j.out
#SBATCH --error={run}/logs/judges_%j.err
set -euo pipefail
export PYTHONPATH={quote(str(PACKAGE/'src'))} PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec python3 {quote(paths['key_loader'])} {quote(paths['judge_python'])} {quote(str(script))} judge --run-dir {quote(str(run))}
'''
    for name, data in [('gpu.sbatch', batch), ('judges.sbatch', judge)]:
        path = run / 'control' / name
        path.write_text(data)
        subprocess.run(['bash', '-n', str(path)], check=True)
    if (run / 'control/jobs.json').exists():
        raise ValueError('Jobs already submitted; inspect them before resuming')
    gpu = subprocess.check_output(['sbatch', '--parsable', str(run / 'control/gpu.sbatch')], text=True).strip()
    write_json(run / 'control/jobs.json', {'gpu': gpu})
    judges = subprocess.check_output(
        ['sbatch', '--parsable', '--dependency=afterok:' + gpu, str(run / 'control/judges.sbatch')], text=True
    ).strip()
    write_json(run / 'control/jobs.json', {'gpu': gpu, 'judges': judges})
    print('SUBMITTED', json.dumps({'gpu': gpu, 'judges': judges}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'submit', 'gpu', 'judge'])
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--eval-root', type=Path)
    parser.add_argument('--recipe', type=Path, default=PACKAGE / 'configs/spoken_low_v11_cfg.json')
    parser.add_argument('--account', default='nemotron_speech_chat')
    parser.add_argument('--partition', default='polar4,interactive')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if args.stage == 'prepare':
        if args.eval_root is None:
            parser.error('--eval-root is required for prepare')
        prepare_run(args.eval_root, run, args.recipe)
    elif args.stage == 'submit':
        submit(run, args.account, args.partition)
    elif args.stage == 'gpu':
        run_gpu(run)
    else:
        run_judges(run)


if __name__ == '__main__':
    main()
