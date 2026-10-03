# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Automatic prompt search for a SALM checkpoint on a dev set (never on test).

Each round: (1) an open LLM (vLLM, default gpt-oss-120b) proposes `--candidates` prompts from the task description,
the best prompt so far, and a sample of that prompt's dev errors; (2) all candidates are decoded on the dev subset
with one SALM engine (`vllm_task_eval.py --prompts-file`); (3) the best one seeds the next round.

    python prompt_search.py --model <salm> --dev dev.json --task-args "--task cls --labels labels.txt" \\
        --description "Emotion of the speaker, one of six labels" --seed-prompt "What emotion ...?" --out-dir ps/

Writes <out-dir>/search.json (every prompt tried, its dev score, the round) and prints the ranking. Higher-is-better
is inferred from the metric (WER/MER/EER lower).
"""
import argparse
import json
import random
import re
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOWER = {"wer", "mer", "eer"}

PROPOSE = """You are optimizing the instruction given to a speech-language model together with an audio clip.
Task: {description}
Current best instruction (dev score {score:.4f}, {direction} is better):
{best}

Examples of the model's current mistakes on dev (reference -> model output):
{errors}

Write {n} new, diverse candidate instructions that could fix these mistakes. Consider: naming the language and script
of the expected output, naming the domain and its vocabulary, stating the exact output format (e.g. the allowed labels,
'answer with one word', 'digits for numbers'), and what to avoid (translating, explaining, transliterating). Keep each
under 60 words. {fields} If the current instruction contains the answer's label list or format, keep that information.
Return a JSON array of strings only."""


def run(argv):
    """Run a command given as an argument vector (no shell: paths and prompts are passed verbatim)."""
    print("$", shlex.join(argv), flush=True)
    subprocess.run(argv, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="SALM checkpoint (HF export dir)")
    p.add_argument("--dev", required=True)
    p.add_argument("--task-args", required=True, help="vllm_task_eval.py task flags, e.g. '--task asr'")
    p.add_argument("--description", required=True)
    p.add_argument("--seed-prompt", action="append", required=True, help="one or more starting prompts")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument(
        "--fields-note",
        default="",
        help="describe per-row placeholders to the proposer, e.g. 'Include {context} verbatim (label list).'",
    )
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--limit", type=int, default=500, help="dev rows per decode")
    p.add_argument("--proposer", default="openai/gpt-oss-120b")
    p.add_argument("--python-vllm", default=sys.executable)
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tried = {}

    def evaluate(prompts, tag):
        prompts = [x for x in dict.fromkeys(prompts) if x not in tried]
        if not prompts:
            return
        pf = out / f"{tag}_prompts.json"
        pf.write_text(json.dumps(prompts, ensure_ascii=False, indent=1))
        run(
            [a.python_vllm, str(HERE / "vllm_task_eval.py"), "--model", a.model, "--manifest", a.dev]
            + ["--out", str(out / tag), "--prompts-file", str(pf), "--limit", str(a.limit)]
            + shlex.split(a.task_args)  # the only free-form part: decoder/task flags
        )
        for k, pr in enumerate(prompts):
            s = json.load(open(f"{out / tag}.p{k}.jsonl.summary.json"))
            tried[pr] = dict(score=s["score"], metric=s.get("metric", ""), round=tag, hyps=f"{out / tag}.p{k}.jsonl")

    evaluate(a.seed_prompt, "r0")
    lower = next(iter(tried.values()))["metric"] in LOWER
    best = lambda: (min if lower else max)(tried, key=lambda k: tried[k]["score"])  # noqa: E731
    for r in range(1, a.rounds + 1):
        b = best()
        rows = [json.loads(line) for line in open(tried[b]["hyps"])]
        wrong = [
            x for x in rows if str(x.get("pred_text", "")).strip().lower() != str(x.get("text", "")).strip().lower()
        ]
        random.Random(r).shuffle(wrong)
        errors = "\n".join(f"- {str(x.get('text'))[:120]!r} -> {str(x.get('pred_text'))[:120]!r}" for x in wrong[:15])
        req = out / f"r{r}_request.json"
        req.write_text(
            json.dumps(
                PROPOSE.format(
                    description=a.description,
                    best=b,
                    score=tried[b]["score"],
                    errors=errors,
                    direction="lower" if lower else "higher",
                    n=a.candidates,
                    fields=a.fields_note,
                )
            )
        )
        cand_file = out / f"r{r}_candidates.json"
        run([a.python_vllm, str(HERE / "prompt_search.py"), "--propose", str(req), str(cand_file), a.proposer])
        evaluate(json.load(open(cand_file)), f"r{r}")
        print(f"[search] round {r}: best {tried[best()]['score']:.4f}  {best()!r}", flush=True)
    ranking = sorted(tried.items(), key=lambda kv: kv[1]["score"], reverse=not lower)
    (out / "search.json").write_text(
        json.dumps([dict(prompt=k, **v) for k, v in ranking], indent=1, ensure_ascii=False)
    )
    for k, v in ranking:
        print(f"{v['score']:.4f}  [{v['round']}]  {k}")


def propose(req_file, out_file, model):
    """Worker: one proposer-LLM call (separate process so its GPU memory is freed before decoding)."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, max_model_len=8192, gpu_memory_utilization=0.40)
    msg = [
        {"role": "system", "content": "Reasoning: medium\nOutput only JSON."},
        {"role": "user", "content": json.load(open(req_file))},
    ]
    txt = llm.chat([msg], SamplingParams(temperature=0.8, max_tokens=4000))[0].outputs[0].text
    m = re.search(r"\[.*\]", txt.split("assistantfinal")[-1], re.S)
    cands = [c.strip() for c in json.loads(m.group(0)) if isinstance(c, str) and c.strip()] if m else []
    Path(out_file).write_text(json.dumps(cands, ensure_ascii=False, indent=1))
    print(f"[propose] {len(cands)} candidates")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--propose":
        propose(*sys.argv[2:5])
    else:
        main()
