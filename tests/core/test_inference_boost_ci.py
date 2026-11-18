# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
PREFIX = "L2_Speech_Transcription_Speech_to_Text_Inference_Boost_GT"
CASES = [
    ("", "buffered_rnnt", "stt_en_fastconformer_transducer_large", "rnnt", []),
    ("_TDT", "buffered_rnnt", "nvidia/stt_en_fastconformer_tdt_large", "tdt", []),
    ("_Cache_Aware", "cache_aware_rnnt", "nvidia/nemotron-speech-streaming-en-0.6b", "ca_rnnt", ["default"]),
    (
        "_Cache_Aware_Case_Insensitive",
        "cache_aware_rnnt",
        "nvidia/nemotron-speech-streaming-en-0.6b",
        "ca_rnnt",
        ["case_insensitive"],
    ),
]


@pytest.mark.unit
def test_boost_cases_have_independent_required_jobs():
    workflow = yaml.safe_load((ROOT / ".github/workflows/cicd-main-speech.yml").read_text())
    job = workflow["jobs"]["e2e-tests"]
    rows = [row for row in job["strategy"]["matrix"]["include"] if row["script"].startswith(PREFIX)]
    assert sorted(row["script"] for row in rows) == sorted(PREFIX + case[0] for case in CASES)
    assert job["strategy"]["fail-fast"] is False
    for row in rows:
        assert row["runner"] == "${{ inputs.runner }}"
        assert not row.get("is-optional", False)
        assert row.get("timeout", 10) == 10
    action = next(step for step in job["steps"] if step.get("uses") == "./.github/actions/test-template")
    assert action["with"]["script"] == "${{ matrix.script }}"
    assert action["with"]["timeout"] == "${{ matrix.timeout || 10 }}"


@pytest.mark.unit
@pytest.mark.parametrize("suffix,config,model,output,modes", CASES)
@pytest.mark.parametrize("exit_code", [0, 17])
def test_boost_job_runs_one_case_and_propagates_failure(
    tmp_path, monkeypatch, suffix, config, model, output, modes, exit_code
):
    record = tmp_path / "commands.jsonl"
    coverage = tmp_path / "coverage"
    coverage.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['BOOST_COMMAND_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "sys.exit(int(os.environ['BOOST_EXIT_CODE']))\n"
    )
    coverage.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("BOOST_COMMAND_LOG", str(record))
    monkeypatch.setenv("BOOST_EXIT_CODE", str(exit_code))
    result = subprocess.run(
        ["bash", str(ROOT / "tests/functional_tests" / f"{PREFIX}{suffix}.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == exit_code, result.stderr
    expected = [
        "run",
        "-a",
        "--data-file=/workspace/.coverage",
        "--source=/workspace/nemo",
        "examples/asr/asr_streaming_inference/asr_streaming_infer.py",
        "--config-path=../conf/asr_streaming_inference/",
        f"--config-name={config}.yaml",
        "audio_file=/home/TestData/asr/canary/dev-other-wav-10-boost-gt.json",
        f"output_filename=/tmp/stt_inference_boost_gt_res_{output}.json",
        f"asr.model_name={model}",
    ]
    if modes:
        expected += [
            f"asr.per_stream_biasing_defaults.boosting_model_cfg.bpe_mode={modes[0]}",
            "asr.per_stream_biasing_defaults.boosting_model_alpha=2.0",
        ]
    expected += [
        "streaming.batch_size=5",
        "lang=en",
        "enable_itn=False",
        "enable_nmt=False",
        "asr_output_granularity=segment",
    ]
    assert [json.loads(line) for line in record.read_text().splitlines()] == [expected]
