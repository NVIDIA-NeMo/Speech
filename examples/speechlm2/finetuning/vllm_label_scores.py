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
"""Exact per-label log-probabilities from a SALM checkpoint (closed-set tasks).

For every manifest row (optionally split into fixed windows) and every label, the
rendered prompt is followed by the label text and `<|im_end|>`, and vLLM returns
`prompt_logprobs`; the score of a label is the summed log-probability of its
tokens. Unlike reading the generated answer, this gives a full score vector per
example, which enables window voting, calibration and ensembling (GTZAN).

Output JSONL: one row per (example, window) with `scores: {label: logprob}`.

Usage: vllm_label_scores.py --model M --manifest m.json --labels labels.txt --out s.jsonl
       [--windows 10] (seconds; the full clip is always scored as window "full") [--reuse]

`<out>.fingerprint.json` records the model, manifest, labels, windows and code revision. With --reuse an existing
output is kept only if that fingerprint matches; a mismatch is an error.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fingerprint import FingerprintMismatch, checkpoint_digest, code_revision, differences, file_digest  # noqa: E402
from vllm_task_eval import load_audio, load_manifest, start_engine  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--windows", type=float, default=None, help="also score non-overlapping windows of this length")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.40)
    p.add_argument("--reuse", action="store_true", help="keep an existing output with a matching fingerprint")
    a = p.parse_args()
    fp = {
        "model": checkpoint_digest(a.model) if Path(a.model).is_dir() else a.model,
        "manifest": file_digest(a.manifest),
        "labels": file_digest(a.labels),
        "windows": a.windows,
        "code": code_revision(),
    }
    fp_path = Path(a.out + ".fingerprint.json")
    if a.reuse and Path(a.out).exists():
        diff = differences(json.loads(fp_path.read_text()) if fp_path.exists() else {}, fp)
        if diff:
            raise FingerprintMismatch(f"{a.out} was produced with different inputs (differs in: {', '.join(diff)})")
        print(f"[scores] {a.out} exists with a matching fingerprint, reusing")
        return
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    with open(a.labels) as f:
        labels = [label.strip() for label in f if label.strip()]
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    llm = start_engine(
        model=a.model,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=128,
        gpu_memory_utilization=a.gpu_memory_utilization,
        limit_mm_per_prompt={"audio": 1},
        # Every label of a row shares the same prompt+audio prefix; with prefix caching, prompt_logprobs of requests
        # that hit a cached prefix are not exact on hybrid backbones, so caching is off for scoring.
        enable_prefix_caching=False,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)

    rows = load_manifest(a.manifest)
    reqs, meta = [], []
    for r in rows:
        prefix = tok.apply_chat_template(
            [{"role": "user", "content": f"{r['context']} <|audio|>"}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        n_prefix = len(tok.encode(prefix, add_special_tokens=False))
        audio = load_audio(r)
        wins = [("full", audio)]
        if a.windows:
            w = int(a.windows * 16000)
            wins += [(f"w{i}", audio[i * w : (i + 1) * w]) for i in range(len(audio) // w)]
        for wname, wav in wins:
            for lab in labels:
                text = prefix + lab + "<|im_end|>"
                n_total = len(tok.encode(text, add_special_tokens=False))
                reqs.append({"prompt": text, "multi_modal_data": {"audio": (wav, 16000)}})
                meta.append((r["id"], r["text"], wname, lab, n_total - n_prefix))
    print(f"[scores] {len(rows)} rows -> {len(reqs)} scoring requests", flush=True)
    outs = llm.generate(reqs, sp)
    agg = {}
    for (rid, gold, wname, lab, k), o in zip(meta, outs):
        plp = o.prompt_logprobs
        toks = o.prompt_token_ids
        # the label (+ <|im_end|>) are the last k prompt tokens
        lp = sum(plp[i][toks[i]].logprob for i in range(len(toks) - k, len(toks)))
        agg.setdefault((rid, wname), {"id": rid, "text": gold, "window": wname, "scores": {}})["scores"][lab] = lp
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for v in agg.values():
            f.write(json.dumps(v) + "\n")
    fp_path.write_text(json.dumps(fp, indent=1, sort_keys=True))
    print(f"[scores] wrote {len(agg)} (row, window) score vectors to {a.out}", flush=True)


if __name__ == "__main__":
    main()
