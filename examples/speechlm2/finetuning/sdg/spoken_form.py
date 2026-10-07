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
"""Add a `spoken` field to synthetic text: how the written medical string is said aloud (for the TTS input).

The label stays the written `text` (what the benchmark's references look like); the TTS reads `spoken`, so
abbreviations and doses are pronounced the way doctors say them ('HbA1c' -> 'H B A one C', '500mg' -> 'five hundred
milligram', '1-0-1' -> 'one zero one'). Also drops rows equal to a test reference (--exclude). Uses vLLM.
"""
import argparse
import json
import re

PROMPT = (
    "For each medical string below (as written by a doctor in India), write exactly how an Indian doctor would say it "
    "aloud: expand numbers, doses and units into words ('500mg' -> 'five hundred milligram', '1-0-1' -> 'one zero "
    "one', '13 3 23' -> 'thirteen three twenty three'), spell out abbreviations the way they are pronounced ('HbA1c' "
    "-> "
    "'H B A one C', 'CBC' -> 'C B C', 'BID' -> 'B I D', 'SOS' -> 'S O S', 'c/o' -> 'complaints of', 'ln' -> 'lymph "
    "node', "
    "'2d echo' -> 'two D echo'), keep brand and drug names as words, and drop speaker labels like 'Patient:'. Return "
    "a "
    "JSON array of strings with exactly one spoken form per input, in the same order.\n\n"
)


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def clean(t):
    t = t.replace("‑", "-").replace("–", "-").replace("—", "-").replace("’", "'")
    t = re.sub(r"^\s*(Doctor|Patient|Dr\.?|Pt\.?)\s*:\s*", "", t, flags=re.I)
    t = re.sub(r"\s*\((e\.g\.|i\.e\.)[^)]*\)", "", t)
    return re.sub(r"\s+", " ", t).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--texts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--exclude", default=None, help="test manifest: drop rows whose normalized text equals a reference")
    p.add_argument("--model", default="openai/gpt-oss-120b")
    p.add_argument("--batch", type=int, default=25)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.40)
    a = p.parse_args()
    from vllm import LLM, SamplingParams

    with open(a.texts) as f:
        rows = [json.loads(line) for line in f]
    for r in rows:
        r["text"] = clean(r["text"])
    refs = set()
    if a.exclude:
        with open(a.exclude) as f:
            refs = {norm(json.loads(line)["text"]) for line in f}
    n0 = len(rows)
    rows = [r for r in rows if r["text"] and norm(r["text"]) not in refs]
    print(f"[spoken] {n0} rows, {n0 - len(rows)} dropped (empty or equal to a test reference)")
    chunks = [rows[i : i + a.batch] for i in range(0, len(rows), a.batch)]
    llm = LLM(model=a.model, max_model_len=8192, gpu_memory_utilization=a.gpu_memory_utilization, max_num_seqs=256)
    msgs = [
        [
            {"role": "system", "content": "Reasoning: low\nOutput only JSON."},
            {"role": "user", "content": PROMPT + json.dumps([r["text"] for r in c], ensure_ascii=False)},
        ]
        for c in chunks
    ]
    outs = llm.chat(msgs, SamplingParams(temperature=0.2, max_tokens=6000))
    kept = 0
    with open(a.out, "w") as f:
        for c, o in zip(chunks, outs):
            txt = o.outputs[0].text.split("assistantfinal")[-1]
            m = re.search(r"\[.*\]", txt, re.S)
            try:
                spoken = json.loads(m.group(0)) if m else []
            except json.JSONDecodeError:
                spoken = []
            if len(spoken) != len(c):
                continue  # misaligned answer: drop the chunk rather than risk label/audio mismatch
            for r, s in zip(c, spoken):
                if isinstance(s, str) and s.strip():
                    f.write(json.dumps({**r, "spoken": s.strip()}, ensure_ascii=False) + "\n")
                    kept += 1
    print(f"[spoken] wrote {a.out}: {kept} rows")


if __name__ == "__main__":
    main()
