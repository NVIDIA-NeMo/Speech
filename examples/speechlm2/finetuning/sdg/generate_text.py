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
"""Synthetic medical text for Indian-context medical ASR, generated with an open LLM (gpt-oss-120b) via vLLM.

No benchmark data is shown to the LLM: prompts are built only from a list of specialties and utterance styles.
Output: jsonl rows {"text", "style", "specialty", "concept"}; exact duplicates removed.
"""
import argparse
import json
import random
import re

SPECIALTIES = [
    "general medicine",
    "diabetology and endocrinology",
    "cardiology",
    "paediatrics",
    "orthopaedics",
    "dermatology",
    "gastroenterology",
    "ENT",
    "ophthalmology",
    "obstetrics and gynaecology",
    "psychiatry",
    "pulmonology",
    "nephrology and urology",
    "neurology",
    "oncology",
    "dentistry",
    "infectious diseases",
    "rheumatology",
]

ENTITY_TASKS = {
    "drugs": "{n} different medicines as an Indian doctor writes them on a prescription for {sp}: mostly Indian brand "
    "names with strength and form (for example '<Brand> 500mg Tablet', '<Brand> D 30mg by 20mg Capsule', '<Brand> "
    "Syrup "
    "5ml', '<Brand> Cream'), some generic names alone, and a few drug classes",
    "clinical_findings": "{n} clinical findings, symptoms or diagnoses as written in Indian case notes for {sp}, in "
    "the terse style doctors use (lowercase or mixed case, abbreviations such as 'ln', 'c/o', 'h/o', 'b/l', 'rt', "
    "'lt', "
    "anatomical detail, e.g. 'left inguinal ln', 'mild burning micturition since 2 days')",
    "diagnostics": "{n} lab tests, scans and investigations as Indian doctors write them for {sp} (e.g. '2d echo', "
    "'HbA1c', 'CBC with ESR', 'USG abdomen and pelvis', 'serum creatinine', 'MRI LS spine')",
    "advices": "{n} short patient advices as Indian doctors write them for {sp} (diet, rest, follow-up, lifestyle, "
    "e.g. 'avoid oily and spicy food', 'review after 1 week with reports', 'drink plenty of fluids')",
    "misc_medical": "{n} other short medical strings seen in Indian prescriptions for {sp}: Indian pharmaceutical "
    "company names (e.g. '... Private Limited', '... Pharmaceuticals'), dosage schedules ('1-0-1 after food', 'twice "
    "a "
    "day for 5 days'), and medical devices or supplies",
}

SENTENCE_TASK = (
    "{n} different sentences an Indian doctor speaks aloud while dictating a prescription or case notes in {sp}. Each "
    "sentence mixes clinical findings, Indian brand-name medicines with strength and schedule (e.g. 'Tab <Brand> "
    "650mg SOS for fever', 'Take <Brand> once in a day after food'), investigations and advice. Vary length from 6 to "
    "35 words. Write numbers as a doctor writes them (500 mg, 1-0-1, 13 3 23)."
)
CONVO_TASK = (
    "{n} short turns from doctor-patient consultations in India in {sp}, spoken in natural Indian English, each 8-40 "
    "words, from either the doctor or the patient. Include fillers and backchannels as spoken ('okay okay', 'hmm', "
    "'so', 'actually'), symptoms, durations, and medicine names with doses where natural."
)

SYSTEM = "Reasoning: low\nYou generate realistic, diverse training text for a speech recognizer. Output only JSON."


def build_requests(n_rounds, per_request, seed):
    rng = random.Random(seed)
    reqs = []
    for _ in range(n_rounds):
        for sp in SPECIALTIES:
            for concept, tpl in ENTITY_TASKS.items():
                reqs.append(("narration_entity", sp, concept, tpl.format(n=per_request, sp=sp)))
            reqs.append(("narration_sentence", sp, "sentence", SENTENCE_TASK.format(n=per_request, sp=sp)))
            reqs.append(("conversation", sp, "conversation", CONVO_TASK.format(n=per_request, sp=sp)))
    rng.shuffle(reqs)
    return reqs


def parse_list(text):
    text = text.split("assistantfinal")[-1]
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [str(x).strip() for x in items if isinstance(x, (str, int, float)) and str(x).strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openai/gpt-oss-120b")
    p.add_argument("--out", required=True)
    p.add_argument("--rounds", type=int, default=10, help="passes over specialties x styles")
    p.add_argument("--per-request", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    a = p.parse_args()
    from vllm import LLM, SamplingParams

    reqs = build_requests(a.rounds, a.per_request, a.seed)
    llm = LLM(model=a.model, max_model_len=8192, gpu_memory_utilization=a.gpu_memory_utilization, max_num_seqs=256)
    msgs = [
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Write {task}. Return a JSON array of strings, nothing else."},
        ]
        for _, _, _, task in reqs
    ]
    sp = SamplingParams(temperature=a.temperature, top_p=0.95, max_tokens=4096, seed=a.seed)
    outs = llm.chat(msgs, sp)
    seen, n = set(), 0
    with open(a.out, "w") as f:
        for (style, spec, concept, _), o in zip(reqs, outs):
            for t in parse_list(o.outputs[0].text):
                key = t.lower()
                if key in seen or len(t) > 400:
                    continue
                seen.add(key)
                f.write(json.dumps({"text": t, "style": style, "specialty": spec, "concept": concept}) + "\n")
                n += 1
    print(f"wrote {a.out}: {n} unique texts from {len(reqs)} requests")


if __name__ == "__main__":
    main()
