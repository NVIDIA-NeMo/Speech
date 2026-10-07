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
"""Score Eka Care Medical ASR hypotheses with Eka's own KARMA toolkit (github.com/eka-care/KARMA-OpenMedEvalKit).

Input: a vllm_task_eval.py output jsonl (rows with id, text, pred_text) and the Eka manifest (for medical_entities).
Reports WER/CER under three text treatments, because the published vendor table does not state which one it used:
  raw     - KARMA ASRMetrics on the raw strings (case and punctuation count)
  norm    - KARMA general_text_processor defaults (lowercase, punctuation removed; digits kept) on both sides
  norm_n2w- as norm, plus digits -> words (num2text)
and KARMA's semantic metrics (semWER, semCER, kwWER = keyword WER over the annotated medical entities).
Writes <hyps>.eka.json.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.environ.get("KARMA_SRC", "."))  # a checkout of KARMA-OpenMedEvalKit
from karma.metrics.asr.asr_metrics import ASRMetrics  # noqa: E402
from karma.metrics.asr.asr_semantic_metrics import ASRSemanticMetrics  # noqa: E402
from karma.processors.general_text_processor import GeneralTextProcessor  # noqa: E402

_THINK = re.compile(r"<think>.*?</think>", re.S)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hyps", required=True)
    p.add_argument("--manifest", required=True, help="Eka manifest from prepare_benchmarks.py eka")
    p.add_argument("--by", default="recording_context", help="also report per value of this manifest field")
    p.add_argument("--subset", default=None, help="tune | heldout: restrict to that side of the speaker split")
    p.add_argument("--split", default=None, help="JSON {id: tune|heldout} for --subset")
    a = p.parse_args()
    with open(a.manifest) as f:
        man = {r["id"]: r for r in map(json.loads, f)}
    if a.subset:
        with open(a.split) as f:
            split = json.load(f)
        man = {k: v for k, v in man.items() if split[k] == a.subset}
    with open(a.hyps) as f:
        hyp = [json.loads(line) for line in f]
    rows = [(man[h["id"]], _THINK.sub("", h.get("pred_text") or "").strip()) for h in hyp if h["id"] in man]
    assert len(rows) == len(man), f"{len(rows)} hypotheses for {len(man)} references"

    def score(sub):
        refs = [m["text"] for m, _ in sub]
        preds = [h for _, h in sub]
        out = {}
        for name, proc in (
            ("raw", None),
            ("norm", GeneralTextProcessor()),
            ("norm_n2w", GeneralTextProcessor(use_num2text=True)),
        ):
            r, h = (refs, preds) if proc is None else (proc.process(list(refs)), proc.process(list(preds)))
            res = ASRMetrics().evaluate(h, r)
            out[f"wer_{name}"], out[f"cer_{name}"] = res.wer, res.cer
        sem = ASRSemanticMetrics("asr_semantic_metric").evaluate(
            preds, refs, language="en", entities=[m["medical_entities"] for m, _ in sub]
        )
        out.update(semwer=sem.semantic_wer, semcer=sem.semantic_cer, kwwer=sem.entity_wer, n=len(sub))
        return out

    res = {"all": score(rows)}
    if a.by:
        for v in sorted({m.get(a.by) for m, _ in rows}):
            res[f"{a.by}={v}"] = score([x for x in rows if x[0].get(a.by) == v])
    with open(a.hyps + (f".{a.subset}" if a.subset else "") + ".eka.json", "w") as f:
        json.dump(res, f, indent=1)
    for k, v in res.items():
        print(
            f"{k:40s} n={v['n']:5d}  WER raw {100 * v['wer_raw']:5.2f}  norm {100 * v['wer_norm']:5.2f}  "
            f"n2w {100 * v['wer_norm_n2w']:5.2f}  | CER norm {100 * v['cer_norm']:5.2f}  "
            f"semWER {100 * v['semwer']:5.2f}  kwWER {100 * (v['kwwer'] or 0):5.2f}"
        )


if __name__ == "__main__":
    main()
