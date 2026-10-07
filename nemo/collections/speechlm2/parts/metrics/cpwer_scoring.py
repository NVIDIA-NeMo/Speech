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
"""Score a prediction manifest offline, without a model or a GPU.

The settings an inference script and an offline scorer share live here once, as a dataclass both
compose as a mixin. Duplicating them and testing that the copies agree detects drift; sharing one
definition prevents it.

Composition, not inheritance: an eval config *has* scoring settings, it is not a kind of cpWER
config, and a second metric must be able to contribute its own fields without displacing these.
Whoever adds one should write a self-contained all-defaulted dataclass with flat ``<metric>_*``
names and its own ``AXIS_FIELDS`` / ``NON_AXIS_FIELDS`` pair -- never extend this one.
"""

import re
from dataclasses import dataclass
from typing import Optional

from nemo.collections.speechlm2.parts.metrics.cpwer import CpWER

__all__ = [
    "AXIS_FIELDS",
    "NON_AXIS_FIELDS",
    "CpWERScoringConfig",
    "join_reference_manifest",
    "resolve_normalizer",
    "resolve_placements",
    "resolve_text_fields",
    "score_rows",
]

#: The axes a comparability verdict ranges over. Exists so the fingerprint, the verdict and the
#: inventory test all derive from one list instead of three hand-maintained ones: a field in neither
#: this nor :data:`NON_AXIS_FIELDS` is one the fingerprint would silently omit.
AXIS_FIELDS = (
    "cpwer_normalizer",
    "cpwer_tag_syntax_ref",
    "cpwer_tag_syntax_hyp",
    "cpwer_tag_case_sensitive",
    "cpwer_untagged_speaker_ref",
    "cpwer_untagged_speaker_hyp",
    "cpwer_keep_empty_streams",
    "cpwer_drop_tag_residue",
    "cpwer_speaker_order",
    "cpwer_ceiling_source",
)

#: Shared, but not axes: bookkeeping, capabilities no reference scorer has, and the WER path.
NON_AXIS_FIELDS = (
    "compute_cpwer",
    "cpwer_placement",
    "cpwer_max_speakers",
    "cpwer_report_notag_ceiling",
    "use_normalizer",
    "normalizer_language",
    "subset_field",
    "cpwer_placement_ref",
    "cpwer_placement_hyp",
)

# Cut ids carry a `-<offset>-<duration>` suffix that a source manifest's id does not. `{6,}` rather
# than `{6}`: a recording past 99999.99s produces a 7-digit field, and a fixed width would silently
# fail to join exactly the longest recordings.
_ID_SUFFIX = re.compile(r"-\d{6,}-\d{6,}$")

# Markers a streaming decoder interleaves with its text. Not words; scoring them would inflate both
# sides. Only ever applied to a legacy field, never to `pred_text_raw`, which is marker-free.
_CONTENT_MARKERS = re.compile(r"\[(BLANK|WRITE)\]")


@dataclass
class CpWERScoringConfig:
    """Every field an inference script and an offline scorer share.

    Composed as a mixin by both entry-point configs, so the two cannot drift. Flat field names, not
    a nested object: nesting would rename every override string in every existing shell script.
    """

    # --- non-axis: what to score, and how to report it ---
    compute_cpwer: bool = True
    use_normalizer: Optional[str] = "whisper"  # WER path; cpWER falls back to it
    normalizer_language: str = "en"

    # --- the ten axes; every default is today's behaviour, byte for byte ---
    cpwer_normalizer: Optional[str] = None  # None -> inherit use_normalizer
    cpwer_tag_syntax_ref: str = "spk"
    cpwer_tag_syntax_hyp: str = "spk"
    cpwer_tag_case_sensitive: bool = True
    cpwer_untagged_speaker_ref: Optional[int] = 0
    cpwer_untagged_speaker_hyp: Optional[int] = 0
    cpwer_keep_empty_streams: bool = True
    cpwer_drop_tag_residue: bool = True
    cpwer_speaker_order: str = "index"
    cpwer_ceiling_source: str = "strip_tags"

    # --- non-axis: unchanged in name, position and default ---
    cpwer_placement: str = "prefix"
    cpwer_max_speakers: Optional[int] = None
    cpwer_report_notag_ceiling: bool = True
    subset_field: str = "subset_for_metrics"

    # --- non-axis: the placement of each role. None follows cpwer_placement. Set them when the roles differ, as for
    # a suffix-placement model scored against references written in prefix form ---
    cpwer_placement_ref: Optional[str] = None
    cpwer_placement_hyp: Optional[str] = None

    def effective_normalizer(self) -> Optional[str]:
        """The normalizer cpWER will actually use, resolving the inherit-from-WER default."""
        return resolve_normalizer(self)

    def validate(self) -> None:
        """Check every string axis against its vocabulary.

        Vocabulary only -- no combination is rejected. Any mix of axes is legal and is simply
        reported as what it is; refusing combinations would be guessing at intent.

        Raises:
            ValueError: naming the field and the accepted values.
        """
        for name, accepted in (
            ("cpwer_placement", ("prefix", "suffix")),
            ("cpwer_placement_ref", (None, "prefix", "suffix")),
            ("cpwer_placement_hyp", (None, "prefix", "suffix")),
            ("cpwer_speaker_order", ("index", "first_seen")),
            ("cpwer_ceiling_source", ("strip_tags", "streams")),
        ):
            value = getattr(self, name)
            if value not in accepted:
                raise ValueError(f"Unknown {name}={value!r}. Accepted: {', '.join(map(repr, accepted))}")


def resolve_normalizer(cfg) -> Optional[str]:
    """The normalizer cpWER will actually use, resolving the inherit-from-WER default.

    A free function rather than only a method because ``@hydra_runner`` hands every entry point a
    ``DictConfig``, on which dataclass methods are not bound -- ``cfg.effective_normalizer()`` raises
    ``ConfigAttributeError`` there. Reading fields works on either, so every internal caller uses
    this and the surface accepts "any object carrying the config's fields", as documented.

    Args:
        cfg: a :class:`CpWERScoringConfig`, or any object with its fields (e.g. a ``DictConfig``).

    Returns:
        Optional[str]: ``cpwer_normalizer`` when set, otherwise ``use_normalizer``.
    """
    return cfg.cpwer_normalizer if cfg.cpwer_normalizer is not None else cfg.use_normalizer


def resolve_placements(cfg) -> tuple:
    """The tag placements cpWER will actually parse the reference and the hypothesis with.

    A free function for the reason :func:`resolve_normalizer` is one: it must also work on a ``DictConfig``. An object
    without the role fields reads as having them unset, as a stamp recorded before they existed does.

    Args:
        cfg: a :class:`CpWERScoringConfig`, or any object with its fields (e.g. a ``DictConfig``).

    Returns:
        tuple: ``(reference placement, hypothesis placement)``: ``cpwer_placement_ref`` and ``cpwer_placement_hyp``
        where set, otherwise ``cpwer_placement``.
    """
    ref = getattr(cfg, "cpwer_placement_ref", None)
    hyp = getattr(cfg, "cpwer_placement_hyp", None)
    return (cfg.cpwer_placement if ref is None else ref, cfg.cpwer_placement if hyp is None else hyp)


def score_rows(rows: list, cfg: CpWERScoringConfig, *, reference_field=None, hypothesis_field=None) -> tuple:
    """Score a manifest offline.

    Args:
        rows: manifest records, each carrying raw reference and hypothesis text.
        cfg: resolved scoring settings.
        reference_field: force a reference field instead of trying the resolution chain.
        hypothesis_field: force a hypothesis field.

    Returns:
        tuple: ``(per_row, corpus, subsets)`` -- one metric dict per row, the corpus ``compute()``
        output in fractions, and ``{subset name: compute() output}``.

    Two accumulators run rather than one, because a subset named ``_all_`` would double-count into
    the cross-dataset total. Their global micros are asserted equal.
    """
    cfg.validate()
    corpus_metric = _build_metric(cfg)
    subset_metric = _build_metric(cfg)
    per_row, subsets_seen, rows_without_subset = [], set(), 0

    for row in rows:
        ref_raw, hyp_raw = resolve_text_fields(row, reference_field=reference_field, hypothesis_field=hypothesis_field)
        result = corpus_metric.score_session(ref_raw, hyp_raw)
        corpus_metric.update("corpus", [ref_raw], [hyp_raw])
        subset = _subset_of(row, cfg.subset_field)
        if subset:
            subset_metric.update(subset, [ref_raw], [hyp_raw])
            subsets_seen.add(subset)
        else:
            rows_without_subset += 1
        per_row.append(_row_metrics(result))

    corpus = corpus_metric.compute()
    subset_out = subset_metric.compute()
    corpus["cpwer_rows_without_subset"] = rows_without_subset
    subsets = {
        name: {k: v for k, v in subset_out.items() if k.endswith(f"_{name}") or k == "cpwer"}
        for name in sorted(subsets_seen)
    }
    return per_row, corpus, subsets


def resolve_text_fields(row: dict, *, reference_field=None, hypothesis_field=None) -> tuple:
    """Find the raw reference and hypothesis on a manifest row.

    Prefers the verbatim fields, falling back to the spellings other scorers use. Text that was
    already normalized before it was written is refused rather than scored: re-normalizing an
    already-normalized string is not the same as normalizing the original, so the number would be
    wrong in a way nothing downstream could detect.

    Args:
        row: one manifest record.
        reference_field: force a field name, bypassing the chain.
        hypothesis_field: force a field name.

    Returns:
        tuple: ``(reference, hypothesis)``, raw.

    Raises:
        KeyError: naming what is missing and what would have to be supplied instead.
    """
    ref = _first_present(row, reference_field, ("text_raw", "expected_answer"))
    hyp = _first_present(row, hypothesis_field, ("pred_text_raw", "predicted_answer", "generation"))
    if ref is None:
        raise KeyError(
            "No raw reference on this row. Tried text_raw / expected_answer. A pre-split manifest "
            "has only `text`, which was normalized AND tag-stripped before it was written and "
            "cannot be re-scored: supply a reference manifest, or force a field explicitly."
        )
    if hyp is None:
        raise KeyError("No raw hypothesis on this row. Tried pred_text_raw / predicted_answer / generation.")
    # Only a legacy field can carry markers; `pred_text_raw` is marker-free by construction.
    if hypothesis_field != "pred_text_raw" and "pred_text_raw" not in row:
        hyp = _CONTENT_MARKERS.sub(" ", hyp)
    return ref, hyp


def join_reference_manifest(rows: list, ref_rows: list, *, on: str = "id") -> list:
    """Attach raw references from a source manifest to prediction rows.

    Joins on ``sample_id`` when both sides have it, else on the cut id with its
    ``-<offset>-<duration>`` suffix stripped -- a prediction row's id carries that suffix and the
    source manifest's does not, so a raw join matches nothing.

    Args:
        rows: prediction rows.
        ref_rows: source manifest rows carrying the raw reference.
        on: field to join on when the suffix-stripped id is not needed.

    Returns:
        list: ``rows``, each with the reference row's fields merged in under any key it lacks.

    Raises:
        KeyError: on a row that matches nothing, rather than joining partially and scoring a subset
            of the corpus as though it were all of it.
    """
    index = {}
    for r in ref_rows:
        index[_join_key(r, on)] = r
    joined, missing = [], 0
    for row in rows:
        match = index.get(_join_key(row, on))
        if match is None:
            missing += 1
            continue
        joined.append({**{k: v for k, v in match.items() if k not in row}, **row})
    if missing:
        raise KeyError(f"{missing}/{len(rows)} prediction rows matched no reference row on {on!r}")
    return joined


def _build_metric(cfg: CpWERScoringConfig) -> CpWER:
    return CpWER.from_config(cfg)


def _row_metrics(result) -> dict:
    """One row's metric dict.

    An abstained row OMITS the five count keys and keeps ``cpwer: null`` as its marker, so a reader
    can tell "not scored" from "scored, zero errors" by key presence. Writing zeros for both would
    make an unscorable corpus report as a perfect one.
    """
    out = {"cpwer": result.cpwer, "cpwer_ref_speakers": list(result.ref_speakers)}
    if result.abstained:
        return out
    out.update(
        {
            "cpwer_errors": result.errors,
            "cpwer_ref_words": result.ref_words,
            "cpwer_insertions": result.ins,
            "cpwer_deletions": result.dels,
            "cpwer_substitutions": result.subs,
            "num_ref_speakers": result.num_ref_speakers,
            "num_hyp_speakers": result.num_hyp_speakers,
            "notag_ceiling": result.notag_ceiling,
        }
    )
    return out


def _first_present(row: dict, forced, chain) -> Optional[str]:
    """First field in ``chain`` the row actually carries.

    Tests presence, NOT truthiness: an empty hypothesis is a real result -- the model emitted
    nothing, which scores as all deletions -- and must not fall through to the next candidate or
    be reported as a missing field. Measured on a real 2912-row manifest: 2 rows.
    """
    if forced is not None:
        return row.get(forced)
    for key in chain:
        if row.get(key) is not None:
            return row[key]
    return None


def _join_key(row: dict, on: str) -> str:
    if row.get("sample_id"):
        return str(row["sample_id"])
    return _ID_SUFFIX.sub("", str(row.get(on, "")))


def _subset_of(row: dict, field: str) -> Optional[str]:
    """The subset label for one row, from the row itself or from its nested input manifest.

    Checked in both places because the two manifest shapes this scores differ: a row written here
    nests the whole input record under ``custom``, while another scorer's output carries the field
    at the top level. Looking in one place only would silently bucket nothing.
    """
    value = row.get(field)
    if value:
        return str(value)
    nested = row.get("custom")
    if isinstance(nested, dict) and nested.get(field):
        return str(nested[field])
    return None
