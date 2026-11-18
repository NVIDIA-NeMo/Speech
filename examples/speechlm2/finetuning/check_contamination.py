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
"""Check training text against a test set for contamination before training on it.

    python check_contamination.py --test test.json --train a.json b.jsonl [--ngram 8] [--drop-out clean.json]

Reports, per training file: test references that appear verbatim (after lowercasing and stripping punctuation), the
share of test references with at least one shared n-gram, and, if the test manifest has an `entities` or
`medical_entities` field, the share of test entities that occur somewhere in the training text (domain vocabulary
overlap: expected for a good synthetic corpus and not by itself contamination, but worth reporting). With
--drop-out, writes the concatenated training rows minus any row whose text equals a test reference or shares an
n-gram with one.
"""
import argparse
import ast
import json
import re


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(s).lower())).strip()


def ngrams(words, n):
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def entities(row):
    raw = row.get("entities") or row.get("medical_entities")
    if not raw:
        return []
    try:
        items = ast.literal_eval(raw) if isinstance(raw, str) else raw
    except (ValueError, SyntaxError):
        return []
    return [norm(x[0]) for x in items if isinstance(x, (list, tuple)) and x and norm(x[0])]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", required=True)
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--ngram", type=int, default=8)
    p.add_argument("--drop-out", default=None)
    a = p.parse_args()
    with open(a.test) as f:
        test = [json.loads(line) for line in f if line.strip()]
    refs = {norm(r["text"]) for r in test if norm(r["text"])}
    ref_ngrams = {}
    for r in refs:
        for g in ngrams(r.split(), a.ngram):
            ref_ngrams.setdefault(g, set()).add(r)
    ents = {e for r in test for e in entities(r)}
    kept = []
    for path in a.train:
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        texts = [norm(r["text"]) for r in rows]
        exact = sum(t in refs for t in texts)
        hit_refs, bad = set(), 0
        blob = " " + " ".join(texts) + " "
        for r, t in zip(rows, texts):
            shared = [ref_ngrams[g] for g in ngrams(t.split(), a.ngram) if g in ref_ngrams]
            for s in shared:
                hit_refs |= s
            if t in refs or shared:
                bad += 1
            else:
                kept.append(r)
        ent_cov = sum(f" {e} " in blob for e in ents) / max(1, len(ents)) if ents else None
        print(
            f"{path}: {len(rows)} rows | exact test matches {exact} | rows sharing a {a.ngram}-gram with test {bad} | "
            f"test refs touched {len(hit_refs)}/{len(refs)} ({100 * len(hit_refs) / max(1, len(refs)):.2f}%)"
            + (f" | test entities seen in training text {100 * ent_cov:.1f}% of {len(ents)}" if ents else "")
        )
    if a.drop_out:
        with open(a.drop_out, "w") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {a.drop_out}: {len(kept)} rows")


if __name__ == "__main__":
    main()
