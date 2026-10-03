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
"""Task metrics for speech-LLM benchmarks: WER, BLEU/chrF, label accuracy, SQuAD EM/F1.

Kept separate from the decoder so a saved hypothesis file can be re-scored
without a GPU:

    python task_metrics.py --task bleu --hyps eval/x.jsonl [--bleu-tokenize ja-mecab]

Every hypothesis file is JSONL with at least ``text`` (reference) and
``pred_text`` (hypothesis); tasks may read extra fields (``answers`` for SQuAD,
``labels`` for classification).
"""
import argparse
import collections
import json
import re
import string

# ---------------------------------------------------------------- WER


def wer_normalizers():
    from whisper_normalizer.basic import BasicTextNormalizer
    from whisper_normalizer.english import EnglishTextNormalizer

    _punct = re.compile(r"[^\w\s']", re.UNICODE)
    _ws = re.compile(r"\s+")

    def simple(text: str) -> str:
        # lowercase, strip punctuation, collapse whitespace; leaves numbers and non-English tokens alone
        return _ws.sub(" ", _punct.sub(" ", text.lower())).strip()

    return {"simple": simple, "english": EnglishTextNormalizer(), "basic": BasicTextNormalizer(), "none": lambda x: x}


def edit_counts(refs, hyps):
    """Corpus-level insertions, deletions and substitutions between token sequences (kaldialign)."""
    import kaldialign

    tot = {"ins": 0, "del": 0, "sub": 0, "ref_len": 0}
    for ref, hyp in zip(refs, hyps):
        d = kaldialign.edit_distance(ref, hyp)
        for k in tot:
            tot[k] += d[k]
    tot["err_rate"] = (tot["ins"] + tot["del"] + tot["sub"]) / tot["ref_len"]
    return tot


def score_wer(rows, normalizer="simple"):
    norms = wer_normalizers()
    out = {}
    for name, fn in norms.items():
        refs = [fn(r["text"]) for r in rows]
        hyps = [fn(r["pred_text"]) for r in rows]
        keep = [i for i, r in enumerate(refs) if r.strip()]
        refs = [refs[i] for i in keep]
        hyps = [hyps[i] for i in keep]
        if not refs:
            raise ValueError(f"every reference is empty after the {name!r} normalizer; WER is undefined")
        m = edit_counts([r.split() for r in refs], [h.split() for h in hyps])
        cm = edit_counts([list(" ".join(r.split())) for r in refs], [list(" ".join(h.split())) for h in hyps])
        n = m["ref_len"]
        out[name] = {
            "wer": m["err_rate"],
            "cer": cm["err_rate"],
            "ins": m["ins"] / n,
            "del": m["del"] / n,
            "sub": m["sub"] / n,
        }
    head = out[normalizer]
    return {"metric": "wer", "score": head["wer"], "normalizer": normalizer, "by_normalizer": out}


def wer_by_duration(rows, normalizer="simple", edges=(0, 3, 10, 30)):
    """WER per segment-duration bucket: where the errors of a long-form or conversational test set live."""
    out = {}
    for lo, hi in zip(edges, (*edges[1:], float("inf"))):
        sub = [r for r in rows if lo <= float(r.get("duration") or 0) < hi]
        if sub:
            res = score_wer(sub, normalizer)["by_normalizer"][normalizer]
            res["words"] = sum(len(wer_normalizers()[normalizer](r["text"]).split()) for r in sub)
            out[f"{lo}-{hi:g} s"] = dict(res, segments=len(sub))
    return out


# ---------------------------------------------------------------- BLEU


def score_bleu(rows, tokenize="13a"):
    import sacrebleu

    hyps = [r["pred_text"] for r in rows]
    refs = [r["text"] for r in rows]
    bleu, chrf = sacrebleu.BLEU(tokenize=tokenize), sacrebleu.CHRF()
    b = bleu.corpus_score(hyps, [refs])
    c = chrf.corpus_score(hyps, [refs])
    return {
        "metric": "bleu",
        "score": b.score,
        "bleu_signature": str(bleu.get_signature()),
        "bleu_str": str(b),
        "chrf": c.score,
    }


# ---------------------------------------------------------------- classification


def _norm_label(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    s = s.strip().strip(string.punctuation + " \n\t")
    return re.sub(r"\s+", " ", s)


# Surface variants a model may use for a label. Mapping them is deliberately generous to
# the base model: "angry" for the label "anger" is a correct answer in the wrong form.
LABEL_SYNONYMS = {
    "angry": "anger",
    "anger": "angry",
    "happiness": "happy",
    "joy": "happy",
    "sadness": "sad",
    "fearful": "fear",
    "scared": "fear",
    "afraid": "fear",
    "disgusted": "disgust",
    "calm": "neutral",
}


def map_to_label(pred: str, labels):
    """Map free-form model output onto the closed label set.

    In order: exact match (``_`` and spaces equivalent); a synonym of a label; the
    label whose name occurs in the output (longest first, so ``calendar_query`` beats
    ``calendar``); an output equal to the part after the first ``_`` of exactly one label
    (``maths`` -> ``qa_maths``). Anything else is ``<invalid>`` and counts as wrong,
    reported separately so a format problem is not mistaken for a recognition problem.
    """
    norm = lambda x: _norm_label(x).replace("_", " ")  # noqa: E731
    p = norm(pred)
    lab = {norm(x): x for x in labels}
    if p in lab:
        return lab[p], True
    if p in LABEL_SYNONYMS and LABEL_SYNONYMS[p] in lab:
        return lab[LABEL_SYNONYMS[p]], False
    for k in sorted(lab, key=len, reverse=True):
        if re.search(rf"(?<![\w]){re.escape(k)}(?![\w])", p):
            return lab[k], False
    for w in p.split():
        if w in LABEL_SYNONYMS and LABEL_SYNONYMS[w] in lab:
            return lab[LABEL_SYNONYMS[w]], False
    suffix = [v for k, v in lab.items() if " " in k and k.split(" ", 1)[1] == p]
    if len(suffix) == 1:
        return suffix[0], False
    return "<invalid>", False


def score_cls(rows, labels=None):
    if labels is None:
        labels = sorted({r["text"] for r in rows})
    correct, exact, invalid = 0, 0, 0
    per_class = collections.defaultdict(lambda: [0, 0])
    confusion = collections.defaultdict(collections.Counter)
    for r in rows:
        pred, is_exact = map_to_label(r["pred_text"], labels)
        r["pred_label"] = pred
        exact += is_exact
        invalid += pred == "<invalid>"
        ok = pred == r["text"]
        correct += ok
        per_class[r["text"]][0] += ok
        per_class[r["text"]][1] += 1
        confusion[r["text"]][pred] += 1
    n = len(rows)
    recalls = {k: c / t for k, (c, t) in per_class.items()}
    return {
        "metric": "accuracy",
        "score": correct / n,
        "unweighted_accuracy": sum(recalls.values()) / len(recalls),
        "exact_format_rate": exact / n,
        "invalid_rate": invalid / n,
        "per_class_recall": recalls,
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


# ---------------------------------------------------------------- SQuAD


def _squad_norm(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(pred, gold):
    p, g = _squad_norm(pred).split(), _squad_norm(gold).split()
    if not p or not g:
        return float(p == g)
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


NO_ANSWER = "unanswerable"


def score_squad(rows):
    """Official SQuAD v2 EM/F1. Unanswerable questions have gold ``[""]``; the model
    signals no-answer by emitting ``unanswerable`` (mapped to the empty string)."""
    em = f1 = 0.0
    split = {"has_ans": [0, 0.0, 0.0], "no_ans": [0, 0.0, 0.0]}
    for r in rows:
        golds = r.get("answers") or [""]
        pred = r["pred_text"].strip()
        if _squad_norm(pred) == NO_ANSWER:
            pred = ""
        e = max(float(_squad_norm(pred) == _squad_norm(g)) for g in golds)
        f = max(_f1(pred, g) for g in golds)
        em += e
        f1 += f
        k = "has_ans" if any(g for g in golds) else "no_ans"
        split[k][0] += 1
        split[k][1] += e
        split[k][2] += f
    n = len(rows)
    return {
        "metric": "squad_f1",
        "score": 100 * f1 / n,
        "exact_match": 100 * em / n,
        "by_type": {
            k: {"n": c, "em": 100 * e / max(c, 1), "f1": 100 * f / max(c, 1)} for k, (c, e, f) in split.items()
        },
    }


# ---------------------------------------------------------------- mixed error rate (code-switching)

_CJK = re.compile(r"([㐀-䶿一-鿿豈-﫿぀-ヿ])")
_MER_PUNCT = re.compile(r"[^\w\s']|_", re.UNICODE)


def mer_tokens(text: str) -> str:
    """Characters for CJK, words for everything else (the ASCEND / SEAME convention)."""
    t = _MER_PUNCT.sub(" ", text.lower())
    t = _CJK.sub(r" \1 ", t)
    return " ".join(t.split())


def score_mer(rows):
    refs = [mer_tokens(r["text"]) for r in rows]
    hyps = [mer_tokens(r["pred_text"]) for r in rows]
    keep = [i for i, x in enumerate(refs) if x]
    refs, hyps = [refs[i] for i in keep], [hyps[i] for i in keep]
    m = edit_counts([r.split() for r in refs], [h.split() for h in hyps])
    n = m["ref_len"]
    return {"metric": "mer", "score": m["err_rate"], "ins": m["ins"] / n, "del": m["del"] / n, "sub": m["sub"] / n}


# ---------------------------------------------------------------- Pearson correlation of predicted scores

PCC_KEYS = ("accuracy", "fluency", "prosodic", "total")
_PCC_RE = {k: re.compile(rf"{k[:6]}\w*\D{{0,12}}?(\d+(?:\.\d+)?)", re.I) for k in PCC_KEYS}


def parse_scores(text):
    out = {}
    for k, rx in _PCC_RE.items():
        m = rx.search(text)
        if m:
            out[k] = float(m.group(1))
    return out


def score_pcc(rows, fallback=5.0):
    """Pearson r per aspect (speechocean762 protocol). Unparseable aspects get the mid-scale
    value `fallback` and are counted in `invalid_rate`."""
    import numpy as np

    if len(rows) < 2:
        raise ValueError("Pearson correlation needs at least two rows")
    res, invalid = {}, 0
    preds = [parse_scores(r["pred_text"]) for r in rows]
    invalid = sum(len(p) < len(PCC_KEYS) for p in preds)
    for k in PCC_KEYS:
        g = np.array([r["scores"][k] for r in rows], float)
        h = np.array([p.get(k, fallback) for p in preds], float)
        res[k] = float(np.corrcoef(g, h)[0, 1]) if h.std() > 0 else 0.0
        res[k + "_mse"] = float(((g - h) ** 2).mean())
    return {
        "metric": "pcc_total",
        "score": res["total"],
        "pcc": {k: res[k] for k in PCC_KEYS},
        "mse": {k: res[k + "_mse"] for k in PCC_KEYS},
        "invalid_rate": invalid / len(rows),
    }


# ---------------------------------------------------------------- equal error rate (anti-spoofing)


def eer(scores, is_target):
    """EER for a detection score where higher = more likely `target` (here: spoof)."""
    import numpy as np

    s, y = np.asarray(scores, float), np.asarray(is_target, bool)
    if s.size == 0:
        raise ValueError("EER needs at least one scored example")
    if y.all() or not y.any():
        raise ValueError("EER needs both target and non-target examples (only one class is present)")
    if not np.isfinite(s).all():
        raise ValueError("EER scores must be finite")
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    P, N = y.sum(), (~y).sum()
    tp, fp = np.cumsum(y), np.cumsum(~y)
    # evaluate only at distinct thresholds (ties must move together)
    last = np.r_[s[1:] != s[:-1], True]
    tp, fp = np.r_[0, tp[last]], np.r_[0, fp[last]]
    fnr, fpr = 1 - tp / P, fp / N
    i = np.argmin(np.abs(fnr - fpr))
    return float((fnr[i] + fpr[i]) / 2)


def score_eer(rows):
    """Headline: EER of the model's spoof score, the exact sequence log-probability
    log P("spoof") - log P("bonafide") given the prompt (vllm_task_eval.py --score-labels).
    Accuracy of the generated label is reported alongside."""
    missing = sum("label_score" not in r for r in rows)
    if missing:
        raise ValueError(f"{missing} of {len(rows)} rows have no label_score; decode with --score-labels")
    y = [r["text"] == "spoof" for r in rows]
    e = eer([r["label_score"] for r in rows], y)
    c = score_cls(rows, labels=["bonafide", "spoof"])
    return {
        "metric": "eer",
        "score": e,
        "accuracy": c["score"],
        "balanced_accuracy": c["unweighted_accuracy"],
        "invalid_rate": c["invalid_rate"],
        "confusion": c["confusion"],
    }


# ---------------------------------------------------------------- multi-label (stuttering events)

ML_EVENTS = {
    "prolongation": [r"prolong"],
    "block": [r"\bblock"],
    "sound repetition": [r"sound[\s_-]*rep"],
    "word repetition": [r"word[\s_-]*rep"],
    "interjection": [r"interject"],
}


def parse_events(text):
    t = text.lower()
    return {e for e, pats in ML_EVENTS.items() if any(re.search(p, t) for p in pats)}


def score_multilabel(rows):
    tp = collections.Counter()
    fp = collections.Counter()
    fn = collections.Counter()
    for r in rows:
        g, h = set(r.get("events") or []), parse_events(r["pred_text"])
        for e in ML_EVENTS:
            tp[e] += e in g and e in h
            fp[e] += e not in g and e in h
            fn[e] += e in g and e not in h
    f1 = {e: (2 * tp[e] / (2 * tp[e] + fp[e] + fn[e]) if tp[e] + fp[e] + fn[e] else 0.0) for e in ML_EVENTS}
    T, F, N = sum(tp.values()), sum(fp.values()), sum(fn.values())
    return {
        "metric": "macro_f1",
        "score": sum(f1.values()) / len(f1),
        "per_class_f1": f1,
        "micro_f1": 2 * T / (2 * T + F + N) if T + F + N else 0.0,
        "support": {e: tp[e] + fn[e] for e in ML_EVENTS},
    }


# ---------------------------------------------------------------- dispatch


TASK_DEPENDENCIES = {
    "asr": ["kaldialign", "whisper_normalizer"],
    "mer": ["kaldialign"],
    "bleu": ["sacrebleu"],
    "pcc": ["numpy"],
    "eer": ["numpy"],
}


def check_dependencies(task):
    """Fail before decoding, not after, if the metric for `task` can't be computed in this environment."""
    import importlib.util

    missing = [m for m in TASK_DEPENDENCIES.get(task, []) if importlib.util.find_spec(m) is None]
    if missing:
        raise SystemExit(
            f"scoring task {task!r} needs {', '.join(missing)}, which isn't installed in this environment "
            f"(pip install {' '.join(m.replace('_', '-') for m in missing)})"
        )


def score(task, rows, **kw):
    if not rows:
        raise ValueError("no rows to score")
    if task == "asr":
        return score_wer(rows, normalizer=kw.get("normalizer") or "simple")
    if task == "bleu":
        return score_bleu(rows, tokenize=kw.get("bleu_tokenize") or "13a")
    if task == "cls":
        return score_cls(rows, labels=kw.get("labels"))
    if task == "squad":
        return score_squad(rows)
    if task == "mer":
        return score_mer(rows)
    if task == "pcc":
        return score_pcc(rows)
    if task == "eer":
        return score_eer(rows)
    if task == "multilabel":
        return score_multilabel(rows)
    raise ValueError(task)


def load_labels(path):
    if not path:
        return None
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["asr", "bleu", "cls", "squad", "mer", "pcc", "eer", "multilabel"])
    p.add_argument("--hyps", required=True)
    p.add_argument("--normalizer", default=None)
    p.add_argument("--bleu-tokenize", default=None)
    p.add_argument("--labels", default=None, help="file with one class label per line")
    p.add_argument("--by-duration", action="store_true", help="asr: WER per segment-duration bucket")
    a = p.parse_args()
    rows = [json.loads(line) for line in open(a.hyps) if line.strip()]
    if a.by_duration:
        for k, v in wer_by_duration(rows, a.normalizer or "simple").items():
            print(
                f"{k:>9}: WER {100 * v['wer']:5.1f}%  (ins {100 * v['ins']:4.1f}, del {100 * v['del']:4.1f}, "
                f"sub {100 * v['sub']:4.1f})  {v['segments']:5d} segments, {v['words']:6d} words"
            )
        return
    res = score(a.task, rows, normalizer=a.normalizer, bleu_tokenize=a.bleu_tokenize, labels=load_labels(a.labels))
    res.pop("confusion", None)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
