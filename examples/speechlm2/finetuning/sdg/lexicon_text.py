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
"""Vocabulary-coverage text from a large name list, e.g. the 1mg Indian medicine list (karthikqnq/1mgdataset, MIT).

Rows: product names (one per brand family first, then random extras), title-cased as pharmacy catalogs write them,
plus LLM prescription sentences around sampled brands (vLLM, gpt-oss-120b). 5% of families (and every 20th sentence
request) are held out for dev. Output rows {text, style, concept, split}.
"""
import argparse
import csv
import json
import random
import re

RX = (
    "Write {n} different sentences an Indian doctor would dictate in a prescription or case note, each using one or "
    "two of these medicines by brand name with strength, form, dose schedule and duration where natural: {brands}. "
    "Vary length from 5 to 30 words. Return a JSON array of strings only."
)


def title(name):
    # catalog style: capitalize words, keep strengths/units as written (500mg), keep hyphenated parts
    return " ".join(w[:1].upper() + w[1:] if w[:1].isalpha() else w for w in name.split())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", required=True, help="CSV with a `name` column (e.g. medicine_dataset.csv)")
    p.add_argument("--out", required=True)
    p.add_argument("--names", type=int, default=50000)
    p.add_argument("--rx-requests", type=int, default=800)
    p.add_argument("--model", default="openai/gpt-oss-120b")
    a = p.parse_args()
    csv.field_size_limit(10**9)
    rng = random.Random(4)
    names = sorted(
        {
            re.sub(r"\s+", " ", r["name"]).strip()
            for r in csv.DictReader(open(a.list, errors="replace"))
            if r.get("name")
        }
    )
    fam = {}
    for n in names:
        fam.setdefault(re.split(r"[\s-]", n)[0], []).append(n)
    fams = sorted(fam)
    rng.shuffle(fams)
    dev_fams = set(fams[: len(fams) // 20])
    picked = [rng.choice(fam[f]) for f in fams]  # one per family first: breadth of vocabulary
    extra = rng.sample(names, max(0, a.names - len(picked)))
    rows = []
    for n in (picked + extra)[: a.names]:
        f = re.split(r"[\s-]", n)[0]
        rows.append(
            dict(text=title(n), style="narration_entity", concept="drugs", split="dev" if f in dev_fams else "train")
        )
    from vllm import LLM, SamplingParams

    reqs = []
    for i in range(a.rx_requests):
        sp = "dev" if i % 20 == 0 else "train"
        pool = [title(rng.choice(fam[f])) for f in rng.sample(fams, 40) if (f in dev_fams) == (sp == "dev")][:6]
        if pool:
            reqs.append((sp, RX.format(n=15, brands="; ".join(pool))))
    llm = LLM(model=a.model, max_model_len=8192, gpu_memory_utilization=0.40, max_num_seqs=256)
    msgs = [
        [{"role": "system", "content": "Reasoning: low\nOutput only JSON."}, {"role": "user", "content": q}]
        for _, q in reqs
    ]
    for (sp, _), o in zip(reqs, llm.chat(msgs, SamplingParams(temperature=1.0, top_p=0.95, max_tokens=3000, seed=4))):
        m = re.search(r"\[.*\]", o.outputs[0].text.split("assistantfinal")[-1], re.S)
        try:
            items = json.loads(m.group(0)) if m else []
        except json.JSONDecodeError:
            items = []
        rows += [
            dict(text=t.strip(), style="narration_sentence", concept="sentence", split=sp)
            for t in items
            if isinstance(t, str) and t.strip()
        ]
    seen, n = set(), 0
    with open(a.out, "w") as f:
        for r in rows:
            if r["text"].lower() not in seen:
                seen.add(r["text"].lower())
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {a.out}: {n} rows ({len(fams)} families, {len(dev_fams)} held out)")


if __name__ == "__main__":
    main()
