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
"""Catalog-grounded synthetic medical text: product names, salts, manufacturers, prescriptions. Run with vLLM.

Source: PharmaLens Indian medicine catalog (sinhal/indian-pharma-dataset-2026-augast, MIT; 195,605 products).
Writes rows {text, style, concept, split} where split is by product family (`_clean_product`): 5% of families are
held out for dev. Rows:
  - product names as catalogued ("Rabekind-DSR Capsule", "Plavix Tablet")
  - salt compositions ("Domperidone 30mg + Rabeprazole 20mg")
  - manufacturer names ("Mankind Pharma Ltd")
  - prescription sentences written by gpt-oss-120b around sampled real brands (the LLM sees only catalog entries)
"""
import argparse
import json
import random
import re

RX = (
    "Write {n} different sentences an Indian doctor would dictate in a prescription or case note, each using one or "
    "two of these real medicines by brand name (with strength, form, dose schedule and duration where natural), "
    "together with findings, investigations or advice: {brands}. Vary length from 5 to 30 words and style (terse "
    "notes, full sentences, advice to the patient). Return a JSON array of strings only."
)


def clean_salt(s):
    s = re.sub(r"\s*\(([^)]*)\)", r" \1", s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--products", type=int, default=24000)
    p.add_argument("--salts", type=int, default=5000)
    p.add_argument("--manufacturers", type=int, default=3000)
    p.add_argument("--rx-requests", type=int, default=800)
    p.add_argument("--model", default="openai/gpt-oss-120b")
    a = p.parse_args()
    rng = random.Random(0)
    cat = [r for r in json.load(open(a.catalog)) if r.get("product_name")]
    fams = sorted({r.get("_clean_product") or r["product_name"].lower() for r in cat})
    rng.shuffle(fams)
    dev_fams = set(fams[: len(fams) // 20])
    split = lambda r: "dev" if (r.get("_clean_product") or r["product_name"].lower()) in dev_fams else "train"  # noqa
    rows = []
    for r in rng.sample(cat, min(a.products, len(cat))):
        rows.append(dict(text=r["product_name"].strip(), style="narration_entity", concept="drugs", split=split(r)))
    salts = {}
    for r in cat:
        if r.get("salt_composition"):
            salts.setdefault(clean_salt(r["salt_composition"]), split(r))
    for s in rng.sample(sorted(salts), min(a.salts, len(salts))):
        rows.append(dict(text=s, style="narration_entity", concept="drugs", split=salts[s]))
    mans = sorted({r["manufacturer"].strip() for r in cat if r.get("manufacturer")})
    for i, m in enumerate(rng.sample(mans, min(a.manufacturers, len(mans)))):
        rows.append(
            dict(text=m, style="narration_entity", concept="misc_medical", split="dev" if i % 20 == 0 else "train")
        )

    from vllm import LLM, SamplingParams

    reqs = []
    for i in range(a.rx_requests):
        pool = [r for r in rng.sample(cat, 12)]
        sp = "dev" if i % 20 == 0 else "train"
        pool = [r for r in pool if split(r) == sp] or pool[:1]
        brands = "; ".join(f"{r['product_name']} ({clean_salt(r.get('salt_composition') or '')})" for r in pool[:6])
        reqs.append((sp, RX.format(n=15, brands=brands)))
    llm = LLM(model=a.model, max_model_len=8192, gpu_memory_utilization=0.40, max_num_seqs=256)
    msgs = [
        [{"role": "system", "content": "Reasoning: low\nOutput only JSON."}, {"role": "user", "content": q}]
        for _, q in reqs
    ]
    outs = llm.chat(msgs, SamplingParams(temperature=1.0, top_p=0.95, max_tokens=3000, seed=2))
    for (sp, _), o in zip(reqs, outs):
        m = re.search(r"\[.*\]", o.outputs[0].text.split("assistantfinal")[-1], re.S)
        try:
            items = json.loads(m.group(0)) if m else []
        except json.JSONDecodeError:
            items = []
        for t in items:
            if isinstance(t, str) and t.strip():
                rows.append(dict(text=t.strip(), style="narration_sentence", concept="sentence", split=sp))
    seen, n = set(), 0
    with open(a.out, "w") as f:
        for r in rows:
            k = r["text"].lower()
            if k not in seen:
                seen.add(k)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {a.out}: {n} rows")


if __name__ == "__main__":
    main()
