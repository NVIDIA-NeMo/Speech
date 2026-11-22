# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / 'tests/setup/models/prestage_asr_adapter.sh'
SMOKE = ROOT / 'tests/functional_tests/L2_ASR_Adapters.sh'
CHECKPOINT = '/home/TestData/asr/stt_en_conformer_ctc_small-1.6.0.nemo'


@pytest.fixture
def staging_env(tmp_path: Path) -> dict[str, str]:
    """Prepare a fake transfer command and a real, minimal NeMo archive."""
    archive = tmp_path / 'source.nemo'
    with tarfile.open(archive, 'w') as tar:
        for name in ('model_config.yaml', 'model_weights.ckpt'):
            data = b'fixture'
            member = tarfile.TarInfo(f'./{name}')
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    commands = tmp_path / 'bin'
    commands.mkdir()
    curl = commands / 'curl'
    curl.write_text(
        '#!/bin/bash\nset -eu\n'
        'printf "%s\\n" "$@" > "$CURL_ARGS"\n'
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = --output ]; then output=$2; shift; fi\n'
        '  shift\n'
        'done\n'
        'if [ "${FAIL_DOWNLOAD:-0}" = 1 ]; then printf partial > "$output"; exit 28; fi\n'
        'cp "$SOURCE_ARCHIVE" "$output"\n'
    )
    curl.chmod(0o755)
    return {
        **os.environ,
        'PATH': f'{commands}:{os.environ["PATH"]}',
        'SOURCE_ARCHIVE': str(archive),
        'CURL_ARGS': str(tmp_path / 'curl-args'),
    }


@pytest.mark.unit
@pytest.mark.parametrize('existing', ['missing', 'empty', 'corrupt'])
def test_stage_publishes_valid_checkpoint(existing: str, tmp_path: Path, staging_env: dict[str, str]) -> None:
    destination = tmp_path / 'models/model.nemo'
    destination.parent.mkdir()
    if existing != 'missing':
        destination.write_bytes(b'' if existing == 'empty' else b'not a model')
    result = subprocess.run(['bash', str(STAGE), str(destination)], env=staging_env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert destination.read_bytes() == Path(staging_env['SOURCE_ARCHIVE']).read_bytes()
    assert not list(destination.parent.glob('*.tmp.*'))
    args = Path(staging_env['CURL_ARGS']).read_text().splitlines()
    for flag, value in (('--connect-timeout', '15'), ('--max-time', '120'), ('--retry', '2')):
        assert args[args.index(flag) + 1] == value
    assert args[-1] == (
        'https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_ctc_small/'
        'versions/1.6.0/files/stt_en_conformer_ctc_small.nemo'
    )


@pytest.mark.unit
@pytest.mark.parametrize('failure', ['transfer', 'invalid_archive', 'missing_weights'])
def test_stage_does_not_publish_failed_download(failure: str, tmp_path: Path, staging_env: dict[str, str]) -> None:
    destination = tmp_path / 'models/model.nemo'
    if failure == 'transfer':
        staging_env['FAIL_DOWNLOAD'] = '1'
    elif failure == 'invalid_archive':
        Path(staging_env['SOURCE_ARCHIVE']).write_bytes(b'not an archive')
    else:
        with tarfile.open(staging_env['SOURCE_ARCHIVE'], 'w') as tar:
            tar.addfile(tarfile.TarInfo('./model_config.yaml'))
    result = subprocess.run(['bash', str(STAGE), str(destination)], env=staging_env, capture_output=True, text=True)
    assert result.returncode != 0
    assert Path(staging_env["CURL_ARGS"]).is_file()
    assert not destination.exists()
    assert not list(destination.parent.glob('*.tmp.*'))


@pytest.mark.unit
def test_stage_reuses_checkpoint_without_network(tmp_path: Path, staging_env: dict[str, str]) -> None:
    destination = tmp_path / 'model.nemo'
    destination.write_bytes(Path(staging_env['SOURCE_ARCHIVE']).read_bytes())
    staging_env['FAIL_DOWNLOAD'] = '1'
    result = subprocess.run(['bash', str(STAGE), str(destination)], env=staging_env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not Path(staging_env['CURL_ARGS']).exists()
    assert destination.read_bytes() == Path(staging_env['SOURCE_ARCHIVE']).read_bytes()


@pytest.mark.unit
def test_both_adapter_variants_restore_staged_checkpoint(tmp_path: Path) -> None:
    coverage = tmp_path / 'coverage'
    calls = tmp_path / 'calls.jsonl'
    coverage.write_text(
        f'#!{sys.executable}\nimport json, os, sys\n'
        'with open(os.environ["COVERAGE_CALLS"], "a") as stream:\n'
        '    stream.write(json.dumps(sys.argv[1:]) + "\\n")\n'
    )
    coverage.chmod(0o755)
    result = subprocess.run(
        ['bash', str(SMOKE)],
        env={**os.environ, 'PATH': f'{tmp_path}:{os.environ["PATH"]}', 'COVERAGE_CALLS': str(calls)},
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    invocations = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(invocations) == 2
    for args in invocations:
        assert 'model.pretrained_model=null' in args
        assert f'model.nemo_model={CHECKPOINT}' in args
        assert 'trainer.accelerator=gpu' in args
        assert 'trainer.max_steps=5' in args
    assert 'model.adapter.adapter_name=an4' in invocations[0]
    assert 'model.adapter.adapter_type=tiny_attn' in invocations[1]


@pytest.mark.unit
def test_staging_runs_on_test_runner_before_training_container() -> None:
    action = yaml.safe_load((ROOT / '.github/actions/test-template/action.yml').read_text())
    steps = action['runs']['steps']
    stage = next(step for step in steps if step['name'] == 'Pre-stage ASR adapter checkpoint')
    assert stage['if'] == "${{ inputs.test_dir == 'functional_tests' && inputs.script == 'L2_ASR_Adapters' }}"
    assert steps.index(stage) < next(i for i, step in enumerate(steps) if step['name'] == 'Start container')
    assert 'docker run --rm' in stage['run']
    assert '"$TEST_DATA_PATH:/home/TestData"' in stage['run']
    assert 'bash tests/setup/models/prestage_asr_adapter.sh' in stage['run']
    assert CHECKPOINT in STAGE.read_text()


@pytest.mark.unit
@pytest.mark.parametrize('exit_code', [0, 28])
def test_action_propagates_preparation_result(exit_code: int, tmp_path: Path) -> None:
    action = yaml.safe_load((ROOT / '.github/actions/test-template/action.yml').read_text())
    stage = next(step for step in action['runs']['steps'] if step['name'] == 'Pre-stage ASR adapter checkpoint')
    docker = tmp_path / 'docker'
    calls = tmp_path / 'docker-args.json'
    docker.write_text(
        f'#!{sys.executable}\nimport json, os, sys\n'
        'with open(os.environ["DOCKER_CALLS"], "w") as stream:\n'
        '    json.dump(sys.argv[1:], stream)\n'
        'sys.exit(int(os.environ["PREPARATION_EXIT"]))\n'
    )
    docker.chmod(0o755)
    result = subprocess.run(
        ['bash', '-e', '-o', 'pipefail', '-c', stage['run']],
        env={
            **os.environ,
            'PATH': f'{tmp_path}:{os.environ["PATH"]}',
            'TEST_IMAGE': 'test-image',
            'TEST_DATA_PATH': str(tmp_path / 'test data'),
            'TEST_CHECKOUT': 'run-id/check-out/NeMo',
            'DOCKER_CALLS': str(calls),
            'PREPARATION_EXIT': str(exit_code),
        },
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == exit_code, result.stderr
    assert json.loads(calls.read_text()) == [
        'run',
        '--rm',
        '--volume',
        f'{ROOT}/run-id/check-out/NeMo:/workspace',
        '--volume',
        f'{tmp_path}/test data:/home/TestData',
        '--workdir',
        '/workspace',
        'test-image',
        'bash',
        'tests/setup/models/prestage_asr_adapter.sh',
    ]
