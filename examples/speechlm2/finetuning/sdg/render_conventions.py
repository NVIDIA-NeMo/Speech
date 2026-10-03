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
"""Convention-conditioned training rows: one clip, several written forms, each stated in the prompt.

Each synthetic clip (same audio) gets --variants rows. Each row picks a written *style*: numbers as digits or words;
units joined, spaced or spelled out; company suffixes in full or abbreviated; brand hyphens kept or dropped; strength
separators "/" or "by"; casing; punctuation. It renders the target text in that style and states the style in the
prompt. The TTS read a spoken form, so every written variant matches the audio. A share of rows gets a "possible terms"
list: catalog names rendered in the same style, near-miss distractors from the catalog, and sometimes not the right
term. That teaches the model to copy a spelling only when the audio supports it.

At test time, state the target benchmark's convention with the same wording, via `describe(style)`.

Optional term lists come from a product catalog JSON (list of {"product_name", "manufacturer", ...}), e.g. the
PharmaLens Indian formulary (sinhal/indian-pharma-dataset-2026-augast, MIT). Requires num2words and rapidfuzz.
"""
import argparse
import json
import random
import re

from num2words import num2words
from rapidfuzz import fuzz, process

UNITS = {
    "mg": "milligram",
    "mcg": "microgram",
    "ml": "millilitre",
    "gm": "gram",
    "g": "gram",
    "kg": "kilogram",
    "iu": "international units",
    "%": "percent",
}
UNIT_RX = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|mcg|ml|gm|g|kg|iu|%)(?![a-z])(/[a-z]+)?", re.I)
NUM_RX = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")
BASE = "Transcribe this Indian English medical audio verbatim in the Latin alphabet."
P_MED = (
    "The speaker is an Indian doctor speaking English. Transcribe verbatim in English, in the Latin alphabet, "
    "including drug brand names, dosages and clinical findings."
)


def n2w(x):
    try:
        return num2words(float(x)) if "." in x else num2words(int(x))
    except (ValueError, OverflowError):
        return x


def render(text, st):
    t = text
    if st["company"] == "full":
        t = re.sub(r"\bPvt\.?(?=\s|$)", "Private", t)
        t = re.sub(r"\bLtd\.?(?=\s|$|,)", "Limited", t)
    else:
        t = re.sub(r"\bPrivate(?=\s+Limited)", "Pvt", t)
        t = re.sub(r"\bLimited\b", "Ltd", t)
    if st["slash"] == "by":
        t = re.sub(r"(?<=\w)\s*/\s*(?=\d)", " by ", t)

    def unit(m):
        num, u, per = m.group(1), m.group(2), m.group(3) or ""
        if st["units"] == "words":
            per_w = f" per {per[1:]}" if per else ""
            return f"{n2w(num) if st['numbers'] == 'words' else num} {UNITS[u.lower()]}{per_w}"
        num = n2w(num) if st["numbers"] == "words" else num
        # "joined" = drug strengths attached (500mg); lab concentrations keep the space (6.35 mg/dL)
        joined = st["units"] == "joined" and st["numbers"] == "digits" and not per
        return f"{num}{u}{per}" if joined else f"{num} {u}{per}"

    t = UNIT_RX.sub(unit, t)
    if st["numbers"] == "words":
        t = NUM_RX.sub(lambda m: n2w(m.group(1)), t)
    if st["hyphens"] == "space":
        t = re.sub(r"(?<=[A-Za-z0-9])-(?=[A-Za-z0-9])", " ", t)
    if st["punct"] == "none":
        t = re.sub(r"(?<!\d)[.,]|[.,](?!\d)|[;:!?\"()]", " ", t)  # keep decimal points / digit grouping
    if st["case"] == "lower":
        t = t.lower()
    elif st["case"] == "title":
        t = " ".join(w[:1].upper() + w[1:] for w in t.split(" "))
    return re.sub(r"\s+", " ", t).strip()


def describe(st):
    parts = [BASE]
    parts.append("Write numbers as digits." if st["numbers"] == "digits" else "Write numbers in words.")
    parts.append(
        {
            "joined": "Attach units to drug strengths without a space (500mg, 5ml), but keep a space before "
            "lab units (6.35 mg/dL).",
            "spaced": "Put a space between a number and its unit (500 mg, 5 ml).",
            "words": "Spell units out in words (milligram, millilitre).",
        }[st["units"]]
    )
    parts.append(
        "Write company suffixes in full (Private Limited)."
        if st["company"] == "full"
        else "Abbreviate company suffixes (Pvt Ltd)."
    )
    parts.append(
        "Keep hyphens in brand names (Piopar-MF)."
        if st["hyphens"] == "keep"
        else "Write brand name parts as separate words, without hyphens."
    )
    parts.append(
        "Use a slash between combined strengths (15mg/500mg)."
        if st["slash"] == "slash"
        else "Say 'by' between combined strengths (15mg by 500mg)."
    )
    parts.append(
        {"keep": "Use normal capitalization.", "lower": "Use lowercase only.", "title": "Capitalize every word."}[
            st["case"]
        ]
    )
    parts.append("Use normal punctuation." if st["punct"] == "keep" else "Do not use punctuation.")
    return " ".join(parts)


def sample_style(rng):
    return dict(
        numbers=rng.choice(["digits"] * 3 + ["words"]),
        units=rng.choice(["joined", "joined", "spaced", "words"]),
        company=rng.choice(["full", "abbr"]),
        hyphens=rng.choice(["keep", "keep", "space"]),
        slash=rng.choice(["slash", "slash", "by"]),
        case=rng.choice(["keep", "lower", "title"]),
        punct=rng.choice(["keep", "none"]),
    )


class Catalog:
    """Near-miss lookup over catalog brand families (first word of the product name)."""

    STOP = {"tablet", "capsule", "syrup", "injection", "cream", "gel", "drops", "suspension", "the", "and", "with"}

    def __init__(self, path):
        cat = [r for r in json.load(open(path)) if r.get("product_name")]
        self.by_family = {}
        for r in cat:
            fam = re.split(r"[\s-]", r["product_name"].strip())[0].lower()
            if len(fam) >= 4 and fam not in self.STOP:
                self.by_family.setdefault(fam, []).append(r["product_name"].strip())
        self.families = sorted(self.by_family)
        self.makers = sorted({r["manufacturer"].strip() for r in cat if r.get("manufacturer")})

    def near(self, word, k, rng, exclude=()):
        hits = process.extract(word.lower(), self.families, scorer=fuzz.ratio, limit=k + 3, score_cutoff=60)
        out = []
        for fam, _, _ in hits:
            names = [n for n in self.by_family[fam] if n not in exclude]
            if names:
                out.append(rng.choice(names))
        return out[:k]

    def gold_terms(self, text):
        """Catalog-looking terms in a text: words that are catalog families, plus the whole text if short."""
        words = re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", text)
        return [w for w in words if w.lower() in self.by_family][:3]


def bias_list(text, st, cat, rng, is_entity):
    gold = [text] if is_entity and len(text.split()) <= 6 else cat.gold_terms(text)
    if not gold:
        return None
    anchor = re.split(r"[\s-]", gold[0])[0]
    distract = cat.near(anchor, rng.randint(2, 10), rng, exclude=set(gold))
    if rng.random() < 0.3:
        distract += [rng.choice(cat.makers)]
    keep_gold = rng.random() < 0.75
    terms = (gold if keep_gold else []) + distract
    if not terms:
        return None
    rng.shuffle(terms)
    return "; ".join(dict.fromkeys(render(x, st) for x in terms))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", default=None, help="catalog JSON for term lists (omit to disable them)")
    p.add_argument("--inputs", nargs="+", required=True, help="synthetic-speech manifests (text = written form)")
    p.add_argument("--out", required=True)
    p.add_argument("--variants", type=int, default=2)
    p.add_argument("--plain-frac", type=float, default=0.2)
    p.add_argument("--bias-frac", type=float, default=0.5)
    a = p.parse_args()
    rng = random.Random(3)
    cat = Catalog(a.catalog) if a.catalog else None
    out = []
    for path in a.inputs:
        for r in map(json.loads, open(path)):
            is_entity = r.get("style") == "narration_entity"
            for v in range(a.variants):
                row = {k: r[k] for k in ("audio_filepath", "duration", "style", "concept", "voice") if k in r}
                row["id"] = f"{r['id']}_v{v}"
                if rng.random() < a.plain_frac:
                    row.update(text=r["text"], context=P_MED)
                else:
                    st = sample_style(rng)
                    ctx = describe(st)
                    if cat is not None and rng.random() < a.bias_frac:
                        bl = bias_list(r["text"], st, cat, rng, is_entity)
                        if bl:
                            ctx += f" Possible terms (may be incomplete or wrong): {bl}."
                    row.update(text=render(r["text"], st), context=ctx, fmt=st)
                out.append(row)
    rng.shuffle(out)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {a.out}: {len(out)} rows, {sum(float(r['duration']) for r in out) / 3600:.1f} h (audio reused)")


if __name__ == "__main__":
    main()
