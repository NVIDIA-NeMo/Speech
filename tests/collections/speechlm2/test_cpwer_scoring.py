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
"""Pin offline scoring: the shared config, field resolution, the join, and the two row shapes."""

from dataclasses import fields

import pytest

from nemo.collections.speechlm2.parts.metrics.cpwer_scoring import (
    AXIS_FIELDS,
    NON_AXIS_FIELDS,
    CpWERScoringConfig,
    join_reference_manifest,
    resolve_text_fields,
    score_rows,
)


def _row(ref, hyp, **extra):
    return {"id": "cut-000000-001000", "text_raw": ref, "pred_text_raw": hyp, **extra}


# --------------------------------------------------------------------------------------------
# the shared config
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_field_inventory_is_exhaustive():
    """A field in neither list is one the fingerprint would silently omit.

    This is what makes adding a field without classifying it fail loudly, rather than producing a
    run that claims comparability while differing in something nobody recorded.
    """
    declared = {f.name for f in fields(CpWERScoringConfig)}
    assert declared == set(AXIS_FIELDS) | set(NON_AXIS_FIELDS)
    assert len(AXIS_FIELDS) == 10
    assert not set(AXIS_FIELDS) & set(NON_AXIS_FIELDS)


@pytest.mark.unit
def test_every_axis_defaults_to_todays_behaviour():
    cfg = CpWERScoringConfig()
    assert cfg.cpwer_tag_syntax_ref == cfg.cpwer_tag_syntax_hyp == "spk"
    assert cfg.cpwer_tag_case_sensitive is True
    assert cfg.cpwer_untagged_speaker_ref == cfg.cpwer_untagged_speaker_hyp == 0
    assert cfg.cpwer_keep_empty_streams is True
    assert cfg.cpwer_drop_tag_residue is True
    assert cfg.cpwer_speaker_order == "index"
    assert cfg.cpwer_ceiling_source == "strip_tags"
    assert cfg.cpwer_normalizer is None  # inherits use_normalizer


@pytest.mark.unit
def test_effective_normalizer_inherits_then_overrides():
    assert CpWERScoringConfig(use_normalizer="whisper").effective_normalizer() == "whisper"
    assert CpWERScoringConfig(use_normalizer="whisper", cpwer_normalizer="chime8").effective_normalizer() == "chime8"


@pytest.mark.unit
@pytest.mark.parametrize(
    "field,bad",
    [("cpwer_placement", "Prefix"), ("cpwer_speaker_order", "first-seen"), ("cpwer_ceiling_source", "tags")],
)
def test_validate_rejects_a_typo_instead_of_taking_a_default_branch(field, bad):
    cfg = CpWERScoringConfig(**{field: bad})
    with pytest.raises(ValueError, match=field):
        cfg.validate()


@pytest.mark.unit
def test_validate_accepts_any_axis_combination():
    """No combination is refused -- refusing one would be guessing at intent."""
    CpWERScoringConfig(
        cpwer_placement="suffix", cpwer_normalizer="chime8", cpwer_max_speakers=4, cpwer_speaker_order="first_seen"
    ).validate()


# --------------------------------------------------------------------------------------------
# the placement of each role
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_role_placements_are_non_axis_settings_that_default_to_unset():
    """Classified, so the inventory holds, and unset by default, so no default changes."""
    assert {"cpwer_placement_ref", "cpwer_placement_hyp"} <= set(NON_AXIS_FIELDS)
    cfg = CpWERScoringConfig()
    assert (cfg.cpwer_placement, cfg.cpwer_placement_ref, cfg.cpwer_placement_hyp) == ("prefix", None, None)
    cfg.validate()


@pytest.mark.unit
@pytest.mark.parametrize(
    "fields,expected",
    [
        ({}, ("prefix", "prefix")),
        ({"cpwer_placement": "suffix"}, ("suffix", "suffix")),
        ({"cpwer_placement_hyp": "suffix"}, ("prefix", "suffix")),
        ({"cpwer_placement": "suffix", "cpwer_placement_ref": "prefix"}, ("prefix", "suffix")),
        (
            {"cpwer_placement": "suffix", "cpwer_placement_ref": "prefix", "cpwer_placement_hyp": "prefix"},
            ("prefix", "prefix"),
        ),
    ],
)
def test_each_role_resolves_to_its_own_placement_or_the_shared_one(fields, expected):
    from nemo.collections.speechlm2.parts.metrics import resolve_placements

    cfg = CpWERScoringConfig(**fields)
    cfg.validate()
    assert resolve_placements(cfg) == expected


@pytest.mark.unit
def test_an_object_without_the_role_fields_resolves_them_as_unset():
    """As a stamp written before they existed reads: at their default then."""
    from types import SimpleNamespace

    from nemo.collections.speechlm2.parts.metrics import resolve_placements

    assert resolve_placements(SimpleNamespace(cpwer_placement="suffix")) == ("suffix", "suffix")


@pytest.mark.unit
@pytest.mark.parametrize("field", ["cpwer_placement_ref", "cpwer_placement_hyp"])
@pytest.mark.parametrize("bad", ["postfix", "Prefix", ""])
def test_validate_rejects_an_unknown_role_placement(field, bad):
    cfg = CpWERScoringConfig(**{field: bad})
    with pytest.raises(ValueError, match=f"Unknown {field}={bad!r}"):
        cfg.validate()


@pytest.mark.unit
def test_score_rows_parses_each_role_with_its_own_placement():
    """Speakers A, B, A, prefix reference, word-perfect suffix hypothesis: 0 errors per role, 8 of 10 shared."""
    row = _row(
        "<spk:0> so the budget is fine <spk:1> yeah <spk:0> but the schedule slipped",
        "so the budget is fine <spk:0> yeah <spk:1> but the schedule slipped <spk:0>",
    )
    roles = CpWERScoringConfig(cpwer_placement_ref="prefix", cpwer_placement_hyp="suffix")
    per_row, corpus, _ = score_rows([row], roles)
    assert (per_row[0]["cpwer"], corpus["cpwer_errors_corpus"], corpus["cpwer_ref_words_corpus"]) == (0.0, 0, 10)
    per_row, corpus, _ = score_rows([row], CpWERScoringConfig(cpwer_placement="suffix"))
    assert (per_row[0]["cpwer"], corpus["cpwer_errors_corpus"], corpus["cpwer_ref_words_corpus"]) == (0.8, 8, 10)


# --------------------------------------------------------------------------------------------
# field resolution
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_raw_fields_are_preferred():
    ref, hyp = resolve_text_fields(_row("<spk:0> a", "<spk:0> b", text="normalized", pred_text="normalized"))
    assert (ref, hyp) == ("<spk:0> a", "<spk:0> b")


@pytest.mark.unit
def test_falls_back_to_another_scorers_spellings():
    ref, hyp = resolve_text_fields({"expected_answer": "<spk:0> a", "predicted_answer": "<spk:0> b"})
    assert (ref, hyp) == ("<spk:0> a", "<spk:0> b")


@pytest.mark.unit
def test_empty_hypothesis_is_a_result_not_a_missing_field():
    """The model emitted nothing -- that scores as all deletions and must not raise.

    Measured on a real 2912-row manifest: 2 rows.
    """
    ref, hyp = resolve_text_fields(_row("<spk:0> a b", ""))
    assert (ref, hyp) == ("<spk:0> a b", "")


@pytest.mark.unit
def test_pre_normalized_only_row_is_refused_with_a_useful_message():
    """`text` was normalized AND tag-stripped before it was written; re-normalizing is not the same
    as normalizing the original, so scoring it would be wrong undetectably."""
    with pytest.raises(KeyError, match="raw reference"):
        resolve_text_fields({"text": "hello there", "pred_text": "hello there"})


@pytest.mark.unit
@pytest.mark.parametrize(
    "row,forced",
    [
        ({"text_raw": "<spk:0> a", "pred_text_annotated": "<spk:0> a [BLANK] b"}, "pred_text_annotated"),
        ({"expected_answer": "<spk:0> a", "generation": "<spk:0> a [BLANK] b"}, None),
    ],
)
def test_content_markers_are_stripped_from_legacy_hypothesis_fields(row, forced):
    """A streaming decoder interleaves `[BLANK]`/`[WRITE]`; they are not words.

    Without this every headline number shifts, because the markers survive tag splitting and are
    only deleted later by a bracket-stripping normalizer -- by which point they have already merged
    or displaced real tokens.
    """
    _, hyp = resolve_text_fields(row, hypothesis_field=forced)
    assert "[BLANK]" not in hyp


@pytest.mark.unit
def test_markers_are_left_alone_when_the_raw_field_is_present():
    """`pred_text_raw` is marker-free by construction, so anything marker-shaped in it is text."""
    _, hyp = resolve_text_fields({"text_raw": "<spk:0> a", "pred_text_raw": "<spk:0> a [BLANK] b"})
    assert hyp == "<spk:0> a [BLANK] b"


# --------------------------------------------------------------------------------------------
# the two row shapes
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_admitted_row_carries_all_five_counts():
    per_row, _, _ = score_rows([_row("<spk:0> hello there", "<spk:0> hello world")], CpWERScoringConfig())
    row = per_row[0]
    assert row["cpwer_errors"] == row["cpwer_insertions"] + row["cpwer_deletions"] + row["cpwer_substitutions"]
    assert isinstance(row["cpwer_ref_words"], int)


@pytest.mark.unit
def test_abstained_row_omits_the_counts_and_keeps_cpwer_null():
    """Key ABSENCE is the marker, so "not scored" is distinguishable from "scored, zero errors".

    Writing zeros for both would make an unscorable corpus report as a perfect one.
    """
    cfg = CpWERScoringConfig(cpwer_untagged_speaker_ref=None)
    per_row, corpus, _ = score_rows([_row("no tags at all", "<spk:0> hello")], cfg)
    row = per_row[0]
    assert row["cpwer"] is None
    with pytest.raises(KeyError):
        row["cpwer_errors"]
    assert corpus["cpwer_abstained_corpus"] == 1
    assert corpus["cpwer_admitted_corpus"] == 0


@pytest.mark.unit
def test_counters_are_emitted_even_at_zero():
    """A counter that appears only when non-zero cannot be told from one never computed."""
    _, corpus, _ = score_rows([_row("<spk:0> a", "<spk:0> a")], CpWERScoringConfig())
    for key in ("abstained", "admitted", "zero_ref_words", "empty_hyp", "untagged_hyp_sessions", "sessions"):
        assert f"cpwer_{key}_corpus" in corpus


@pytest.mark.unit
def test_integer_totals_repool_to_the_micro():
    rows = [_row("<spk:0> a b c", "<spk:0> a x c"), _row("<spk:0> d e", "<spk:0> d e")]
    _, corpus, _ = score_rows(rows, CpWERScoringConfig())
    assert corpus["cpwer_errors_corpus"] / corpus["cpwer_ref_words_corpus"] == pytest.approx(corpus["cpwer_corpus"])


# --------------------------------------------------------------------------------------------
# subsets and the join
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_subset_is_found_at_top_level_or_nested_under_custom():
    """Two manifest shapes: ours nests the input record, another scorer's is flat."""
    flat = _row("<spk:0> a", "<spk:0> a", subset_for_metrics="ami")
    nested = _row("<spk:0> a", "<spk:0> a", custom={"subset_for_metrics": "ami"})
    for rows in ([flat], [nested]):
        _, corpus, subsets = score_rows(rows, CpWERScoringConfig())
        assert list(subsets) == ["ami"]
        assert corpus["cpwer_rows_without_subset"] == 0


@pytest.mark.unit
def test_rows_without_a_subset_are_counted_not_dropped():
    _, corpus, subsets = score_rows([_row("<spk:0> a", "<spk:0> a")], CpWERScoringConfig())
    assert subsets == {}
    assert corpus["cpwer_rows_without_subset"] == 1


@pytest.mark.unit
def test_join_strips_the_offset_duration_suffix():
    preds = [{"id": "AMI_x-000000-001234", "pred_text_raw": "<spk:0> a"}]
    refs = [{"id": "AMI_x", "text_raw": "<spk:0> a"}]
    assert join_reference_manifest(preds, refs)[0]["text_raw"] == "<spk:0> a"


@pytest.mark.unit
def test_join_prefers_sample_id_when_present():
    preds = [{"id": "whatever", "sample_id": "S1", "pred_text_raw": "<spk:0> a"}]
    refs = [{"id": "other", "sample_id": "S1", "text_raw": "<spk:0> a"}]
    assert join_reference_manifest(preds, refs)[0]["text_raw"] == "<spk:0> a"


@pytest.mark.unit
def test_join_raises_rather_than_scoring_a_subset_as_the_whole_corpus():
    preds = [{"id": "A-000000-001000"}, {"id": "B-000000-001000"}]
    refs = [{"id": "A", "text_raw": "<spk:0> a"}]
    with pytest.raises(KeyError, match="matched no reference row"):
        join_reference_manifest(preds, refs)
