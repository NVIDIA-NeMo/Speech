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
"""Run a small Nemotron 3.5 Transcribe export through the vLLM SpeechLM plugin."""

import argparse
import json
from pathlib import Path

import soundfile as sf
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def check_inference(fixture: Path, result: Path, use_mtp: bool) -> None:
    model = fixture / "model"
    config = json.loads((model / "config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    audio_token = config["audio_locator_tag"]
    audio_token_id = config["audio_token_id"]
    base_vocab_size = config["llm_config"]["vocab_size"]
    short = sf.read(fixture / "short.wav", dtype="float32")
    long = sf.read(fixture / "long.wav", dtype="float32")
    assert short[1] == long[1] == 16000
    assert len(short[0]) < 30 * 16000 < len(long[0])

    def request(text, audio=()):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt, "multi_modal_data": {"audio": list(audio)}} if audio else {"prompt": prompt}

    # Give the two-audio request enough context to fill a hybrid attention/Mamba cache page.
    names = "Names: " + "Rose Princess. " * 64
    requests = [
        request(f"Transcribe the audio. {audio_token}", [short]),
        request(f"Transcribe the audio. Names: Rose Princess. {audio_token}", [long]),
        request(f"{names}Transcribe the audio. {audio_token} again {audio_token}", [short, long]),
        request("Names: Rose Princess."),
    ]
    llm = LLM(
        model=str(model),
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        max_num_seqs=len(requests),
        limit_mm_per_prompt={"audio": {"count": 2, "length": len(long[0])}},
        gpu_memory_utilization=0.1,
        kv_cache_memory_bytes=64 * 1024**2,
        mm_processor_cache_gb=0,
        enable_prefix_caching=True,
        async_scheduling=False,
        disable_log_stats=False,
        speculative_config={"method": "mtp", "num_speculative_tokens": 2} if use_mtp else None,
    )
    params = SamplingParams(temperature=0, max_tokens=6, ignore_eos=True, skip_special_tokens=False)
    # The frontend's 10 ms hop and 8x feature stacking set the online PEE output lengths.
    short_count, long_count = [(len(waveform) // 160 + 7) // 8 for waveform, _ in (short, long)]
    expected_counts = [short_count, long_count, short_count + long_count, 0]
    outputs = llm.generate(requests, params, use_tqdm=False)
    assert [output.prompt_token_ids.count(audio_token_id) for output in outputs] == expected_counts
    tokens = _completion_tokens(outputs, params.max_tokens, base_vocab_size)

    hits = _count_metric(llm, "vllm:prefix_cache_hits")
    replay = llm.generate(requests[::-1], params, use_tqdm=False)
    assert _completion_tokens(replay, params.max_tokens, base_vocab_size)[::-1] == tokens
    assert _count_metric(llm, "vllm:prefix_cache_hits") > hits, "Second batch did not reuse cached prefixes"
    if use_mtp:
        assert _count_metric(llm, "vllm:spec_decode_num_draft_tokens") > 0, "MTP did not draft tokens"
        assert json.loads(result.read_text()) == tokens, "Two-token MTP changed greedy decoding"
    else:
        result.write_text(json.dumps(tokens))
    print(f"Nemotron 3.5 vLLM inference passed (MTP={use_mtp}, audio token counts={expected_counts})")


def _completion_tokens(outputs, expected_length: int, vocab_size: int) -> list[list[int]]:
    tokens = []
    for request in outputs:
        assert request.finished and len(request.outputs) == 1
        completion = request.outputs[0]
        assert completion.finish_reason == "length"
        assert len(completion.token_ids) == expected_length
        assert all(0 <= token < vocab_size for token in completion.token_ids)
        tokens.append(list(completion.token_ids))
    return tokens


def _count_metric(llm: LLM, name: str) -> int:
    return sum(metric.value for metric in llm.get_metrics() if metric.name == name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--mtp", action="store_true")
    args = parser.parse_args()
    check_inference(args.fixture, args.result, args.mtp)
