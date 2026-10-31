# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Matched offline ASR evaluation and an optional eight-candidate learning-signal pilot.

Uses the training recipe's text-before-audio prompt, stopping token and decoding
limits. MER is Chinese characters plus English words; WER uses the ordinary ASR
reward normalizer. No references are included in generation requests.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import soundfile as sf
from nemo_rl.environments.asr_reward import normalize_asr_tokens
from rapidfuzz.distance import Levenshtein
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from nemo.collections.speechlm2.parts.metrics.mer import mer_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", choices=["mer", "wer"], default="mer")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Also generate eight candidates at temperature 0.8",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=17278)
    parser.add_argument("--max-model-len", type=int, default=62464)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=True,
        gpu_memory_utilization=0.82,
        max_model_len=args.max_model_len,
        max_num_seqs=32,
        max_num_batched_tokens=8192,
        skip_mm_profiling=True,
        limit_mm_per_prompt={"audio": {"count": 1, "length": 57566936}},
        enable_prefix_caching=False,
        seed=42,
    )
    greedy = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        stop_token_ids=[11],
        top_p=1.0,
        top_k=-1,
        skip_special_tokens=True,
    )
    sampled = SamplingParams(
        n=8,
        temperature=0.8,
        max_tokens=args.max_tokens,
        stop_token_ids=[11],
        top_p=1.0,
        top_k=-1,
        seed=42,
        skip_special_tokens=True,
    )
    tokenize = (lambda text: mer_tokens(text).split()) if args.metric == "mer" else normalize_asr_tokens
    total_ref = total_error = oracle_error = 0
    nonconstant = better = 0
    start = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset : offset + args.batch_size]
            requests = []
            for row in batch:
                audio, sr = sf.read(row["audio_filepath"], dtype="float32", always_2d=True)
                if sr != 16000 or not len(audio):
                    raise ValueError("Evaluation requires nonempty 16-kHz audio")
                prompt = row.get("prompt") or row["context"]
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt + "\n<|audio|>"}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                requests.append(
                    {
                        "prompt": rendered,
                        "multi_modal_data": {"audio": (audio.mean(axis=1), sr)},
                    }
                )
            outputs = llm.generate(requests, greedy)
            candidates = llm.generate(requests, sampled) if args.pilot else [None] * len(batch)
            for row, result, candidate in zip(batch, outputs, candidates):
                reference = row.get("text", row.get("expected_answer"))
                ref = tokenize(reference)
                if not ref:
                    raise ValueError("Scoring requires nonempty normalized references")
                pred = result.outputs[0]
                errors = Levenshtein.distance(ref, tokenize(pred.text))
                record = {
                    "id": row["id"],
                    "reference": reference,
                    "pred_text": pred.text,
                    "errors": errors,
                    "reference_tokens": len(ref),
                    "finish_reason": pred.finish_reason,
                    "generated_tokens": len(pred.token_ids),
                }
                total_ref += len(ref)
                total_error += errors
                if candidate is not None:
                    alternatives = []
                    for item in candidate.outputs:
                        err = Levenshtein.distance(ref, tokenize(item.text))
                        alternatives.append(
                            {
                                "text": item.text,
                                "errors": err,
                                "reward": 1 - err / len(ref),
                                "finish_reason": item.finish_reason,
                            }
                        )
                    values = [item["errors"] for item in alternatives]
                    nonconstant += len(set(values)) > 1
                    better += min(values) < errors
                    oracle_error += min(values)
                    record["candidates"] = alternatives
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
    summary = {
        "model": args.model,
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "rows": len(rows),
        "metric": args.metric,
        "error_rate": total_error / total_ref,
        "errors": total_error,
        "reference_tokens": total_ref,
        "elapsed_seconds": time.monotonic() - start,
        "decoding": {
            "temperature": 0,
            "top_p": 1,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "stop_token_ids": [11],
            "seed": 42,
        },
        "pilot": args.pilot,
    }
    if args.pilot:
        summary.update(
            pilot_generations=8,
            pilot_temperature=0.8,
            groups_with_reward_spread=nonconstant,
            groups_with_better_than_greedy=better,
            oracle_error_rate=oracle_error / total_ref,
        )
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
