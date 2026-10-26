# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
"""Offline evaluation of a SALM checkpoint with vLLM on any speech->text task.

Generalizes vllm_asr_eval.py to tasks whose prompt varies per example (spoken QA,
intent classification with a label list, translation with a target language):

* the prompt for a row is ``--prompt`` if given, else the row's ``context``
  field -- the same field ``lhotse_as_conversation`` reads at training time, so
  the evaluation prompt equals the training prompt by construction;
* the engine is loaded once and several manifests can be decoded in one run
  (``--manifest a.json b.json --out a.jsonl b.jsonl``);
* scoring is delegated to task_metrics.py (WER, BLEU/chrF, accuracy, SQuAD F1).

Runs in the vLLM environment. Thinking is disabled in the chat template. For checkpoints with multi-token-prediction
heads, speculative decoding is on by default; on hybrid (state-space) backbones keep ``--mamba-cache-mode all`` so it
stays exact.
"""
import argparse
import fcntl
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import task_metrics  # noqa: E402

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--manifest", required=True, nargs="+")
    p.add_argument("--out", required=True, nargs="+")
    p.add_argument("--task", required=True, choices=["asr", "bleu", "cls", "squad", "mer", "pcc", "eer", "multilabel"])
    p.add_argument("--prompt", default=None, help="fixed prompt; default: each row's `context` field")
    p.add_argument(
        "--prompts-file",
        default=None,
        help="json list of prompts: decode the (single) manifest once per prompt with one engine, writing "
        "<out>.p<k>.jsonl; used by prompt_search.py",
    )
    p.add_argument(
        "--prompt-template",
        default=None,
        help="per-row prompt as a str.format template over manifest fields, e.g. "
        "'Passage: {passage}' (for prompt sweeps on tasks with per-example inputs)",
    )
    p.add_argument("--system-prompt", default=None)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument(
        "--max-tokens-per-sec",
        type=float,
        default=None,
        help="cap each request at min(--max-tokens, rate*duration+32): a guard against greedy "
        "repetition loops on long-output (ASR) tasks. Applied identically to every system compared.",
    )
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("VLLM_GPU_UTIL", 0.85)),
        help="default 0.85, or $VLLM_GPU_UTIL (0.33 lets an eval share one large GPU with a small training run)",
    )
    p.add_argument("--spec-k", type=int, default=2)
    p.add_argument(
        "--mamba-cache-mode",
        default="all",
        choices=["none", "align", "all"],
        help="'all' is required for exact greedy output with spec-k>=2 (docs/fine-tuning-guide.md, section 9)",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--normalizer", default="simple")
    p.add_argument("--bleu-tokenize", default="13a")
    p.add_argument("--labels", default=None, help="class label file (cls task)")
    p.add_argument(
        "--score-labels",
        nargs=2,
        default=None,
        metavar=("NEG", "POS"),
        help="also store label_score = log P(POS) - log P(NEG), the exact sequence log-probabilities of the two "
        "label continuations of the prompt -- for EER-style metrics",
    )
    p.add_argument(
        "--reuse",
        action="store_true",
        help="skip an output whose .summary.json fingerprint (model, manifest, prompt, decoder and scoring settings, "
        "code revision) matches; refuse if it exists with a different fingerprint",
    )
    return p.parse_args()


def load_manifest(path, limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def load_audio(row, target_sr=16000):
    offset = float(row.get("offset", 0.0) or 0.0)
    duration = row.get("duration", None)
    with sf.SoundFile(row["audio_filepath"]) as fh:
        sr = fh.samplerate
        if offset:
            fh.seek(int(round(offset * sr)))
        frames = -1 if duration is None else int(round(float(duration) * sr))
        audio = fh.read(frames=frames, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def eval_fingerprint(args, manifest, prompt, template):
    """Everything a cached result depends on: model, data, prompt, decoder settings, scoring and code."""
    from fingerprint import checkpoint_digest, code_revision, file_digest

    return {
        "model": checkpoint_digest(args.model) if Path(args.model).is_dir() else args.model,
        "manifest": file_digest(manifest),
        "prompt": prompt,
        "prompt_template": template,
        "system_prompt": args.system_prompt,
        "decoder": {
            "max_tokens": args.max_tokens,
            "max_tokens_per_sec": args.max_tokens_per_sec,
            "max_model_len": args.max_model_len,
            "spec_k": args.spec_k,
            "mamba_cache_mode": args.mamba_cache_mode,
            "limit": args.limit,
        },
        "scoring": {
            "task": args.task,
            "normalizer": args.normalizer,
            "bleu_tokenize": args.bleu_tokenize,
            "labels": file_digest(args.labels) if args.labels else None,
            "score_labels": args.score_labels,
        },
        "code": code_revision(),
    }


def main():
    args = parse_args()
    assert len(args.manifest) == len(args.out), "--manifest and --out must pair up"
    task_metrics.check_dependencies(args.task)
    if args.score_labels:
        # Exact label scores need exact prompt_logprobs. With prefix caching (and the hybrid-cache mode that
        # speculative decoding requires), requests sharing a cached prompt+audio prefix get wrong prompt_logprobs on
        # hybrid backbones, so label scoring runs on a plain engine. Generation is unaffected, just slower.
        if args.spec_k > 0:
            print("[eval] --score-labels: disabling speculative decoding and prefix caching for exact scores")
        args.spec_k = 0
    from fingerprint import differences

    jobs = list(zip(args.manifest, args.out, [args.prompt] * len(args.manifest)))
    if args.prompts_file:
        assert len(args.manifest) == 1, "--prompts-file takes a single manifest"
        with open(args.prompts_file) as f:
            cands = json.load(f)
        jobs = [(args.manifest[0], f"{args.out[0]}.p{k}.jsonl", c) for k, c in enumerate(cands)]
    planned = []
    for manifest, out, fixed_prompt in jobs:
        # a --prompts-file candidate with {field} placeholders is a per-row template
        template = fixed_prompt if (args.prompts_file and "{" in fixed_prompt) else args.prompt_template
        fp = eval_fingerprint(args, manifest, fixed_prompt, template)
        summary_path = Path(f"{out}.summary.json")
        if args.reuse and summary_path.exists():
            cached = json.loads(summary_path.read_text())
            diff = differences(cached.get("fingerprint") or {}, fp)
            if diff:
                raise SystemExit(
                    f"{summary_path} was produced with different inputs (differs in: {', '.join(diff)}). Use a new "
                    f"output name or work directory, or delete it to recompute."
                )
            print(f"[eval] RESULT {out}: {args.task} score={cached['score']:.4f}  (cached, fingerprint matches)")
            continue
        planned.append((manifest, out, fixed_prompt, template, fp))
    if not planned:
        return

    from transformers import AutoTokenizer
    from vllm import SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    def render(prompt):
        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        content = prompt if "<|audio|>" in prompt else f"{prompt} <|audio|>"
        messages.append({"role": "user", "content": content})
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )

    llm_kwargs = dict(
        model=args.model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"audio": 1},
        enforce_eager=False,
    )
    if args.score_labels:
        llm_kwargs["enable_prefix_caching"] = False
    if args.spec_k > 0:
        llm_kwargs["speculative_config"] = {
            "method": "mtp",
            "model": args.model,
            "num_speculative_tokens": args.spec_k,
        }
        llm_kwargs["mamba_cache_mode"] = args.mamba_cache_mode
    llm = start_engine(**llm_kwargs)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    labels = task_metrics.load_labels(args.labels)

    for manifest, out, fixed_prompt, template, fp in planned:
        rows = load_manifest(manifest, args.limit)
        if template is not None:
            prompts = [render(template.replace("\\n", "\n").format(**r)) for r in rows]
        else:
            prompts = [render(fixed_prompt if fixed_prompt is not None else r.get("context", "")) for r in rows]
        print(f"[eval] {len(rows)} rows from {manifest}; first prompt: {prompts[0]!r}", flush=True)
        audios = [load_audio(r) for r in rows]
        reqs = [{"prompt": p, "multi_modal_data": {"audio": (a, 16000)}} for p, a in zip(prompts, audios)]
        total_audio = sum(float(r.get("duration") or 0.0) for r in rows)
        t0 = time.perf_counter()
        if args.max_tokens_per_sec:
            per = [sampling.clone() for _ in rows]
            for sp, r in zip(per, rows):
                sp.max_tokens = min(
                    args.max_tokens, int(args.max_tokens_per_sec * float(r.get("duration") or 30)) + 32
                )
            outputs = llm.generate(reqs, per)
        else:
            outputs = llm.generate(reqs, sampling)
        label_scores = (
            score_label_pair(llm, tokenizer, prompts, audios, args.score_labels) if args.score_labels else None
        )
        elapsed = time.perf_counter() - t0

        scored = []
        for i, (r, o) in enumerate(zip(rows, outputs)):
            hyp = _THINK_RE.sub("", o.outputs[0].text).replace("<think>", "").replace("</think>", "").strip()
            rec = {k: v for k, v in r.items() if k not in ("audio_filepath",)}
            rec["pred_text"] = hyp
            if label_scores is not None:
                rec["label_score"] = label_scores[i]
            scored.append(rec)
        # Write hypotheses before scoring: a metric bug must not cost a decode.
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fo:
            for rec in scored:
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
        res = task_metrics.score(
            args.task, scored, normalizer=args.normalizer, bleu_tokenize=args.bleu_tokenize, labels=labels
        )
        summary = {
            "manifest": manifest,
            "model": args.model,
            "task": args.task,
            "prompt": template or fixed_prompt,
            "spec_k": args.spec_k,
            "mamba_cache_mode": args.mamba_cache_mode,
            "max_tokens_per_sec": args.max_tokens_per_sec,
            "num_rows": len(rows),
            "rtfx": total_audio / elapsed if elapsed else None,
            **res,
            "fingerprint": fp,
        }
        with open(str(out_path) + ".summary.json", "w") as fo:
            json.dump(summary, fo, indent=2, ensure_ascii=False)
        brief = {
            k: v
            for k, v in summary.items()
            if k not in ("confusion", "per_class_recall", "by_normalizer", "fingerprint")
        }
        print(
            f"[eval] RESULT {out}: {args.task} score={res['score']:.4f}  "
            f"{json.dumps(brief, ensure_ascii=False)[:600]}",
            flush=True,
        )


def label_logprob_spans(tokenizer, prompt, label, end_token="<|im_end|>"):
    """The scoring text (prompt + label + end-of-turn) and how many trailing tokens belong to the label."""
    text = prompt + label + end_token
    n_prompt = len(tokenizer.encode(prompt, add_special_tokens=False))
    n_total = len(tokenizer.encode(text, add_special_tokens=False))
    if n_total <= n_prompt:
        raise ValueError(f"label {label!r} adds no tokens after the prompt")
    return text, n_total - n_prompt


def sum_label_logprob(prompt_logprobs, token_ids, k):
    """Exact log-probability of the last `k` prompt tokens; fails if vLLM did not return one of them."""
    total = 0.0
    for i in range(len(token_ids) - k, len(token_ids)):
        entry = prompt_logprobs[i] if prompt_logprobs is not None else None
        if not entry or token_ids[i] not in entry:
            raise RuntimeError(f"no log-probability returned for label token at position {i}")
        total += entry[token_ids[i]].logprob
    return total


def score_label_pair(llm, tokenizer, prompts, audios, score_labels):
    """label_score = log P(POS) - log P(NEG): exact sequence log-probabilities of the two label continuations.

    Each label is appended to the rendered prompt (followed by the end-of-turn token) and scored with
    ``prompt_logprobs``, so the score covers every token of the label and never depends on a top-k cut-off.
    """
    from vllm import SamplingParams

    neg, pos = score_labels
    reqs, spans = [], []
    for p, a in zip(prompts, audios):
        for lab in (neg, pos):
            text, k = label_logprob_spans(tokenizer, p, lab)
            reqs.append({"prompt": text, "multi_modal_data": {"audio": (a, 16000)}})
            spans.append(k)
    outs = llm.generate(reqs, SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0))
    lp = [sum_label_logprob(o.prompt_logprobs, o.prompt_token_ids, k) for o, k in zip(outs, spans)]
    return [lp[2 * i + 1] - lp[2 * i] for i in range(len(prompts))]


def start_engine(**llm_kwargs):
    """Construct a vLLM engine while holding a start-up lock.

    vLLM sizes its KV cache by profiling memory during start-up. Two engines starting at once on a shared GPU count
    each other's allocations and can end up with a negative KV-cache budget; serializing only the start-up avoids
    that, while decoding still runs concurrently. The lock lives in $SALM_FT_WORK (or /tmp).
    """
    from vllm import LLM

    lock = Path(os.environ.get("SALM_FT_WORK", "/tmp")) / ".vllm_start.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            return LLM(**llm_kwargs)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
