# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Render cpWER results, and stamp the settings that produced them.

Two jobs, both about making a number interpretable rather than merely correct.

**Scale is split by artifact, never by flag.** ``CpWER.compute()`` and the per-row manifest fields
stay fractions; everything human-facing -- console, log, metrics dict -- is percent, and says so.
A single key name meaning 0.2983 in one file and 29.83 in another is the most reliable way to
produce a false alarm, so the conversion happens in exactly one place: here.

**The settings are stamped beside the number.** Ten axes can be combined freely, so a bare cpWER is
ambiguous. :func:`axis_fingerprint` writes them out literally and enumerates how they differ from a
named reference configuration -- text, not a hash, so a stamp recorded months ago stays readable and
an axis added later reads as "at its default then" instead of invalidating the record.
"""

from typing import Optional

from nemo.collections.speechlm2.parts.metrics.cpwer_scoring import (
    AXIS_FIELDS,
    CpWERScoringConfig,
    resolve_normalizer,
    resolve_placements,
)

__all__ = ["REFERENCE_AXES", "axis_fingerprint", "cpwer_metrics_dict", "format_cpwer_report"]

#: The axis values that reproduce the CHiME-8 / NeMo-Skills reference scorer. Documentation, not a
#: preset: nothing selects these as a group, and a caller reaches them by setting fields.
#: ``cpwer_ceiling_source`` is absent because that scorer has no ceiling, so it cannot have an
#: opinion -- which is why the verdict below reads "9 of 9" and never "10 of 10".
REFERENCE_AXES = {
    "cpwer_normalizer": "chime8",
    "cpwer_tag_syntax_ref": "spk",
    "cpwer_tag_syntax_hyp": "canonical",
    "cpwer_tag_case_sensitive": False,
    "cpwer_untagged_speaker_ref": None,
    "cpwer_untagged_speaker_hyp": 0,
    "cpwer_keep_empty_streams": False,
    "cpwer_drop_tag_residue": False,
    "cpwer_speaker_order": "first_seen",
}


def cpwer_metrics_dict(corpus: dict, subsets: dict, cfg: CpWERScoringConfig, *, scale: str = "percent") -> dict:
    """Assemble the metrics mapping a run writes out.

    Args:
        corpus: the corpus ``CpWER.compute()`` output, in fractions.
        subsets: ``{subset name: compute() output}``, in fractions.
        cfg: the settings that produced them, stamped into the result.
        scale: ``'percent'`` (default, rounded to 2dp) or ``'fraction'``.

    Returns:
        dict: the headline rates at ``scale``, the integer totals, the always-present counters, a
        per-subset block, and an unrounded ``*_fraction`` copy of every rate so a re-pool never has
        to undo rounding.
    """
    if scale not in ("percent", "fraction"):
        raise ValueError(f"Unknown scale={scale!r}. Accepted: 'percent', 'fraction'")
    out = {"scale": scale}
    out.update(_rates(corpus, "corpus", scale))
    for key in ("errors", "ref_words", "insertions", "deletions", "substitutions"):
        out[f"cpwer_{key}"] = corpus.get(f"cpwer_{key}_corpus", 0)
    for key in ("sessions", "admitted", "abstained", "zero_ref_words", "empty_hyp", "untagged_hyp_sessions"):
        out[f"cpwer_{key}"] = corpus.get(f"cpwer_{key}_corpus", 0)
    out["cpwer_rows_without_subset"] = corpus.get("cpwer_rows_without_subset", 0)

    # Stratification by REFERENCE speaker count -- orthogonal to the subset breakdown, and the
    # cross-arm anchor recorded in every historical run log.
    strata = {}
    for key, value in corpus.items():
        if key.endswith("spk") and key.startswith("cpwer_corpus_"):
            strata[key[len("cpwer_corpus_") :]] = round(value * 100, 2) if scale == "percent" else value
    if strata:
        out["cpwer_by_num_speakers"] = strata

    per_subset = {}
    for name, block in subsets.items():
        entry = _rates(block, name, scale)
        entry["cpwer_sessions"] = block.get(f"cpwer_sessions_{name}", 0)
        # Per-subset error counts, which the reference scorer omits. Costs nothing, and without them
        # a merged result cannot be re-pooled from the summary alone.
        for key in ("errors", "ref_words"):
            entry[f"cpwer_{key}"] = block.get(f"cpwer_{key}_{name}", 0)
        per_subset[name] = entry
    if per_subset:
        out["cpwer_per_subset"] = per_subset
        micros = [v["cpwer"] for v in per_subset.values() if v.get("cpwer") is not None]
        if micros:
            # An unweighted mean over subsets -- a third average, distinct from both the corpus micro
            # and the session macro. Named so it cannot be mistaken for either.
            out["cpwer_subset_macro"] = round(sum(micros) / len(micros), 2 if scale == "percent" else 6)

    out["cpwer_axes"] = axis_fingerprint(cfg)
    return out


def format_cpwer_report(metrics: dict, cfg: CpWERScoringConfig, *, wer: Optional[float] = None) -> str:
    """The human-facing block, as printed to a console and appended to a log.

    Args:
        metrics: the output of :func:`cpwer_metrics_dict`.
        cfg: the settings, for the stamp.
        wer: speaker-agnostic WER as a fraction, if computed.

    Returns:
        str: a multi-line block. Percent throughout, with the unit on every line.
    """
    fp = metrics["cpwer_axes"]
    lines = [f"cpwer_axes: {fp['resolved']}"]
    lines.append(f"reference-comparable: {fp['verdict']}")
    for deviation in fp["deviations"]:
        lines.append(f"    differs: {deviation}")
    if wer is not None:
        lines.append(f"WER: {wer:.2%} [normalizer={cfg.use_normalizer}]")
    lines.append(f"cpWER (micro): {_pct(metrics.get('cpwer'))}")
    lines.append(f"cpWER (macro): {_pct(metrics.get('cpwer_macro'))}")
    if "cpwer_subset_macro" in metrics:
        lines.append(f"cpWER (subset-macro): {_pct(metrics['cpwer_subset_macro'])}")
    if metrics.get("cpwer_notag_ceiling") is not None:
        lines.append(
            f"cpWER no-tag ceiling: {_pct(metrics['cpwer_notag_ceiling'])} "
            f"(a word-perfect but unattributed hypothesis) [source={cfg.cpwer_ceiling_source}]"
        )
    for label, value in sorted(metrics.get("cpwer_by_num_speakers", {}).items()):
        # Keep the literal words "reference speakers": a run script greps the log for them, and a
        # more compact label would silently drop these lines from its filtered output.
        lines.append(f"  {label:>5s} reference speakers: {_pct(value)}")
    for name, block in sorted(metrics.get("cpwer_per_subset", {}).items()):
        lines.append(f"  {name:<32s} {_pct(block.get('cpwer'))}  n={block.get('cpwer_sessions', 0)}")
    # Always printed, even at zero: a counter that appears only when non-zero cannot be told from
    # one that was never computed, and these are exactly the ones that explain a surprising number.
    lines.append(
        "  sessions {cpwer_sessions}  admitted {cpwer_admitted}  abstained {cpwer_abstained}  "
        "zero-ref-words {cpwer_zero_ref_words}  empty-hyp {cpwer_empty_hyp}  "
        "untagged-hyp {cpwer_untagged_hyp_sessions}  no-subset {cpwer_rows_without_subset}".format(**metrics)
    )
    return "\n".join(lines)


def axis_fingerprint(cfg: CpWERScoringConfig) -> dict:
    """Stamp the resolved axes, and say how they differ from the reference configuration.

    Literal text rather than a hash, deliberately. A hash is unreadable once recorded, and adding a
    defaulted axis later would change every previously-recorded stamp even though nothing behaved
    differently. Text degrades gracefully: an axis missing from an old stamp simply reads as having
    been at its default then.

    Never raises and never gates anything. Its job is to describe, so that a number carrying an
    unusual combination is obviously unusual rather than silently wrong.

    Args:
        cfg: the resolved settings.

    Returns:
        dict: ``resolved`` (sorted ``key=value`` text over all ten axes), ``deviations`` (one
        readable phrase per axis differing from the reference), ``verdict`` (``"yes (9 of 9 ...)"``
        or ``"no"``), and ``non_axis`` for the settings that change a number without being axes (the role
        placements only when set).
    """
    resolved = {name: getattr(cfg, name) for name in AXIS_FIELDS}
    resolved["cpwer_normalizer"] = resolve_normalizer(cfg)

    deviations = []
    for name, expected in REFERENCE_AXES.items():
        actual = resolved[name]
        if actual != expected:
            deviations.append(f"{name}={actual!r} -> reference uses {expected!r}")

    # Capabilities the reference scorer has no counterpart for. A non-default value there is not a
    # deviation on an axis -- it is a configuration that scorer cannot express at all, so a
    # comparison is not meaningful regardless of how the ten axes are set.
    incomparable = []
    if resolve_placements(cfg) != ("prefix", "prefix"):
        incomparable.append(f"{_placement_setting(cfg)} has no reference equivalent")
    if cfg.cpwer_max_speakers is not None:
        incomparable.append(f"cpwer_max_speakers={cfg.cpwer_max_speakers!r} has no reference equivalent")

    matched = len(REFERENCE_AXES) - len(deviations)
    if incomparable:
        verdict = "no (" + "; ".join(incomparable) + ")"
    elif deviations:
        verdict = f"no ({matched} of {len(REFERENCE_AXES)} comparable axes match)"
    else:
        verdict = f"yes ({matched} of {len(REFERENCE_AXES)} comparable axes; cpwer_ceiling_source excluded)"

    non_axis = f"placement={cfg.cpwer_placement!r},max_speakers={cfg.cpwer_max_speakers!r}"
    # The role placements only when set, so that a config without them keeps the stamp it always had.
    for role in ("ref", "hyp"):
        value = getattr(cfg, f"cpwer_placement_{role}", None)
        if value is not None:
            non_axis += f",placement_{role}={value!r}"

    return {
        "resolved": ",".join(f"{k}={resolved[k]!r}" for k in sorted(resolved)),
        "deviations": deviations + incomparable,
        "verdict": verdict,
        "non_axis": non_axis,
    }


def _rates(block: dict, name: str, scale: str) -> dict:
    """Headline rates for one accumulator, at `scale`, plus their unrounded originals."""
    out = {}
    for key, suffix in (("cpwer", ""), ("cpwer_macro", "_macro"), ("cpwer_notag_ceiling", "_notag_ceiling")):
        raw = block.get(f"cpwer{suffix}_{name}")
        if raw is None:
            continue
        out[key] = round(raw * 100, 2) if scale == "percent" else raw
        out[f"{key}_fraction"] = raw
    return out


def _pct(value) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def _placement_setting(cfg) -> str:
    """The placement as the verdict names it: ``cpwer_placement`` alone, or both roles once either is set."""
    if getattr(cfg, "cpwer_placement_ref", None) is None and getattr(cfg, "cpwer_placement_hyp", None) is None:
        return f"cpwer_placement={cfg.cpwer_placement!r}"
    ref, hyp = resolve_placements(cfg)
    return f"cpwer_placement_ref={ref!r}, cpwer_placement_hyp={hyp!r}"
