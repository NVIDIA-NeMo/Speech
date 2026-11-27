# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check a tiny Transcribe checkpoint through the vLLM SpeechLM plugin."""

import json
import os
from pathlib import Path

import pytest
import soundfile as sf
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

GREEDY = SamplingParams(temperature=0, max_tokens=6, ignore_eos=True, skip_special_tokens=False)


@pytest.fixture(scope="module")
def work_dir():
    return Path(os.environ["NEMOTRON35_SERVING_DIR"])


@pytest.fixture(scope="module")
def model_config(work_dir):
    return json.loads((work_dir / "fixture/model/config.json").read_text())


@pytest.fixture(scope="module")
def clips(work_dir):
    short = sf.read(work_dir / "fixture/short.wav", dtype="float32")
    long = sf.read(work_dir / "fixture/long.wav", dtype="float32")
    assert short[1] == long[1] == 16000
    assert len(short[0]) < 30 * 16000 < len(long[0])
    return short, long


@pytest.fixture(scope="module")
def engine(work_dir, clips):
    speculative_config = None
    if os.environ.get("NEMOTRON35_MTP") == "1":
        speculative_config = {"method": "mtp", "num_speculative_tokens": 2}
    return LLM(
        model=str(work_dir / "fixture/model"),
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        max_num_seqs=4,
        limit_mm_per_prompt={"audio": {"count": 2, "length": len(clips[1][0])}},
        gpu_memory_utilization=0.1,
        kv_cache_memory_bytes=64 * 1024**2,
        mm_processor_cache_gb=0,
        enable_prefix_caching=True,
        async_scheduling=False,
        disable_log_stats=False,
        speculative_config=speculative_config,
    )


@pytest.fixture(scope="module")
def requests(work_dir, model_config, clips):
    tokenizer = AutoTokenizer.from_pretrained(work_dir / "fixture/model", local_files_only=True)
    audio = model_config["audio_locator_tag"]
    short, long = clips

    def request(text, *clips):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
        if clips:
            return {"prompt": prompt, "multi_modal_data": {"audio": list(clips)}}
        return {"prompt": prompt}

    # Fill a hybrid attention/Mamba cache page on the two-audio request.
    names = "Names: " + "Rose Princess. " * 64
    return {
        "short-audio": request(f"Transcribe the audio. {audio}", short),
        "long-contextual-audio": request(f"Transcribe the audio. Names: Rose Princess. {audio}", long),
        "two-audio-clips": request(f"{names}Transcribe the audio. {audio} again {audio}", short, long),
        "text-only": request("Names: Rose Princess."),
    }


@pytest.fixture(scope="module")
def cold_batch(engine, requests, work_dir):
    outputs = engine.generate(list(requests.values()), GREEDY, use_tqdm=False)
    results = dict(zip(requests, outputs, strict=True))
    if os.environ.get("NEMOTRON35_MTP") != "1":
        (work_dir / "standard-tokens.json").write_text(json.dumps(_tokens_by_request(results)))
    return results


@pytest.mark.parametrize(
    ("case", "audio_tokens"),
    [
        ("short-audio", 136),
        ("long-contextual-audio", 376),
        ("two-audio-clips", 512),
        ("text-only", 0),
    ],
    ids=["short-audio", "long-contextual-audio", "two-audio-clips", "text-only"],
)
def test_each_request_reserves_audio_tokens_and_completes(cold_batch, model_config, case, audio_tokens):
    output = cold_batch[case]
    assert output.prompt_token_ids.count(model_config["audio_token_id"]) == audio_tokens
    assert output.finished
    assert len(output.outputs) == 1
    completion = output.outputs[0]
    assert completion.finish_reason == "length"
    assert len(completion.token_ids) == 6
    assert all(0 <= token < model_config["llm_config"]["vocab_size"] for token in completion.token_ids)


def test_replaying_requests_hits_prefix_cache_and_keeps_tokens(engine, requests, cold_batch):
    hits_before = _metric_count(engine, "vllm:prefix_cache_hits")
    reversed_names = list(reversed(requests))
    replay = engine.generate([requests[name] for name in reversed_names], GREEDY, use_tqdm=False)
    assert _metric_count(engine, "vllm:prefix_cache_hits") > hits_before
    assert _tokens_by_request(dict(zip(reversed_names, replay, strict=True))) == _tokens_by_request(cold_batch)


def test_mtp_drafts_tokens(engine, cold_batch):
    assert _metric_count(engine, "vllm:spec_decode_num_draft_tokens") > 0


def test_mtp_matches_standard_greedy_tokens(work_dir, cold_batch):
    standard_tokens = json.loads((work_dir / "standard-tokens.json").read_text())
    assert _tokens_by_request(cold_batch) == standard_tokens


def _tokens_by_request(outputs):
    return {name: list(output.outputs[0].token_ids) for name, output in outputs.items()}


def _metric_count(engine, name):
    return sum(metric.value for metric in engine.get_metrics() if metric.name == name)
