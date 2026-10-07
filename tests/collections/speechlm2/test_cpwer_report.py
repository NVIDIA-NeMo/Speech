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
"""Pin the scale convention and the comparability stamp."""

import pytest

from nemo.collections.speechlm2.parts.metrics.cpwer_report import (
    REFERENCE_AXES,
    axis_fingerprint,
    cpwer_metrics_dict,
    format_cpwer_report,
)
from nemo.collections.speechlm2.parts.metrics.cpwer_scoring import AXIS_FIELDS, CpWERScoringConfig, score_rows


def _scored(cfg, rows=None):
    rows = rows or [
        {"text_raw": "<spk:0> hello there", "pred_text_raw": "<spk:0> hello world", "subset_for_metrics": "a"},
        {"text_raw": "<spk:0> good day", "pred_text_raw": "<spk:0> good day", "subset_for_metrics": "b"},
    ]
    _, corpus, subsets = score_rows(rows, cfg)
    return cpwer_metrics_dict(corpus, subsets, cfg)


# --------------------------------------------------------------------------------------------
# scale
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_metrics_are_percent_and_say_so():
    """The same key meaning 0.2983 in one file and 29.83 in another is the classic false alarm."""
    m = _scored(CpWERScoringConfig())
    assert m["scale"] == "percent"
    assert m["cpwer"] == pytest.approx(m["cpwer_fraction"] * 100, abs=0.01)
    assert m["cpwer"] > 1.0  # a percent, not a fraction


@pytest.mark.unit
def test_unrounded_fractions_are_kept_beside_the_rounded_percents():
    """A re-pool must never have to undo 2dp rounding."""
    m = _scored(CpWERScoringConfig())
    assert m["cpwer_fraction"] == pytest.approx(m["cpwer_errors"] / m["cpwer_ref_words"])


@pytest.mark.unit
def test_fraction_scale_is_available_and_unrounded():
    cfg = CpWERScoringConfig()
    _, corpus, subsets = score_rows([{"text_raw": "<spk:0> a b c", "pred_text_raw": "<spk:0> a x c"}], cfg)
    m = cpwer_metrics_dict(corpus, subsets, cfg, scale="fraction")
    assert m["scale"] == "fraction"
    assert m["cpwer"] == m["cpwer_fraction"] < 1.0


@pytest.mark.unit
def test_unknown_scale_raises():
    cfg = CpWERScoringConfig()
    _, corpus, subsets = score_rows([{"text_raw": "<spk:0> a", "pred_text_raw": "<spk:0> a"}], cfg)
    with pytest.raises(ValueError, match="Unknown scale"):
        cpwer_metrics_dict(corpus, subsets, cfg, scale="percentage")


@pytest.mark.unit
def test_integer_totals_survive_into_the_metrics_dict():
    m = _scored(CpWERScoringConfig())
    assert m["cpwer_errors"] == m["cpwer_insertions"] + m["cpwer_deletions"] + m["cpwer_substitutions"]


# --------------------------------------------------------------------------------------------
# the fingerprint
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_fingerprint_covers_every_axis():
    """An axis missing from the stamp is one a reader cannot check."""
    fp = axis_fingerprint(CpWERScoringConfig())
    for name in AXIS_FIELDS:
        assert name in fp["resolved"]


@pytest.mark.unit
def test_fingerprint_is_text_not_a_hash():
    """Readable months later, and an axis added afterwards does not invalidate an old stamp."""
    fp = axis_fingerprint(CpWERScoringConfig())
    assert "cpwer_speaker_order='index'" in fp["resolved"]


@pytest.mark.unit
def test_fingerprint_resolves_the_inherited_normalizer():
    """The stamp records what was USED, not the None that means "inherit"."""
    fp = axis_fingerprint(CpWERScoringConfig(use_normalizer="chime8", cpwer_normalizer=None))
    assert "cpwer_normalizer='chime8'" in fp["resolved"]


@pytest.mark.unit
def test_defaults_are_not_reference_comparable_and_say_why():
    fp = axis_fingerprint(CpWERScoringConfig())
    assert fp["verdict"].startswith("no")
    assert any("cpwer_normalizer" in d for d in fp["deviations"])


@pytest.mark.unit
def test_the_reference_corner_is_comparable():
    fp = axis_fingerprint(CpWERScoringConfig(**REFERENCE_AXES))
    assert fp["verdict"].startswith("yes")
    assert fp["deviations"] == []
    # Nine, never ten: the reference scorer has no ceiling, so it cannot have an opinion on that one.
    assert "9 of 9" in fp["verdict"]


@pytest.mark.unit
def test_ceiling_source_never_affects_the_verdict():
    for source in ("strip_tags", "streams"):
        fp = axis_fingerprint(CpWERScoringConfig(**REFERENCE_AXES, cpwer_ceiling_source=source))
        assert fp["verdict"].startswith("yes")


@pytest.mark.unit
@pytest.mark.parametrize("override", [{"cpwer_placement": "suffix"}, {"cpwer_max_speakers": 4}])
def test_capabilities_with_no_reference_equivalent_force_not_comparable(override):
    """Not a deviation on an axis -- a configuration that scorer cannot express at all."""
    fp = axis_fingerprint(CpWERScoringConfig(**REFERENCE_AXES, **override))
    assert fp["verdict"].startswith("no")
    assert any("no reference equivalent" in d for d in fp["deviations"])


@pytest.mark.unit
def test_the_default_stamp_is_unchanged():
    """A config that sets no role placement stamps exactly what it did before they existed."""
    fp = axis_fingerprint(CpWERScoringConfig())
    assert fp["verdict"] == "no (2 of 9 comparable axes match)"
    assert fp["non_axis"] == "placement='prefix',max_speakers=None"
    assert len(fp["deviations"]) == 7 and not any("placement" in d for d in fp["deviations"])


@pytest.mark.unit
def test_the_shared_placement_alone_stamps_as_before():
    fp = axis_fingerprint(CpWERScoringConfig(**REFERENCE_AXES, cpwer_placement="suffix"))
    assert fp["deviations"] == ["cpwer_placement='suffix' has no reference equivalent"]
    assert fp["verdict"] == "no (cpwer_placement='suffix' has no reference equivalent)"
    assert fp["non_axis"] == "placement='suffix',max_speakers=None"


@pytest.mark.unit
def test_mixed_role_placements_have_no_reference_equivalent():
    cfg = CpWERScoringConfig(**REFERENCE_AXES, cpwer_placement_ref="prefix", cpwer_placement_hyp="suffix")
    fp = axis_fingerprint(cfg)
    reason = "cpwer_placement_ref='prefix', cpwer_placement_hyp='suffix' has no reference equivalent"
    assert fp["deviations"] == [reason]
    assert fp["verdict"] == f"no ({reason})"
    assert fp["non_axis"] == "placement='prefix',max_speakers=None,placement_ref='prefix',placement_hyp='suffix'"
    assert f"    differs: {reason}" in format_cpwer_report(_scored(cfg), cfg).splitlines()


@pytest.mark.unit
@pytest.mark.parametrize(
    "roles,non_axis,verdict",
    [
        (
            {"cpwer_placement_hyp": "suffix"},
            "placement='prefix',max_speakers=None,placement_hyp='suffix'",
            "no (cpwer_placement_ref='prefix', cpwer_placement_hyp='suffix' has no reference equivalent)",
        ),
        (
            {"cpwer_placement_ref": "suffix"},
            "placement='prefix',max_speakers=None,placement_ref='suffix'",
            "no (cpwer_placement_ref='suffix', cpwer_placement_hyp='prefix' has no reference equivalent)",
        ),
        # Set, but prefix for both roles, which is what the reference scorer does.
        (
            {"cpwer_placement": "suffix", "cpwer_placement_ref": "prefix", "cpwer_placement_hyp": "prefix"},
            "placement='suffix',max_speakers=None,placement_ref='prefix',placement_hyp='prefix'",
            "yes (9 of 9 comparable axes; cpwer_ceiling_source excluded)",
        ),
    ],
    ids=["hyp_only", "ref_only", "both_prefix"],
)
def test_a_role_placement_is_stamped_when_set(roles, non_axis, verdict):
    fp = axis_fingerprint(CpWERScoringConfig(**REFERENCE_AXES, **roles))
    assert (fp["non_axis"], fp["verdict"]) == (non_axis, verdict)


@pytest.mark.unit
def test_fingerprint_never_raises_on_an_unusual_combination():
    """It describes; it does not gate. Gating here would block a legitimate experiment."""
    axis_fingerprint(CpWERScoringConfig(cpwer_placement="suffix", cpwer_normalizer="chime8", cpwer_max_speakers=2))


# --------------------------------------------------------------------------------------------
# the rendered block
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_report_names_the_unit_on_every_rate_line():
    text = format_cpwer_report(_scored(CpWERScoringConfig()), CpWERScoringConfig(), wer=0.25)
    for line in text.splitlines():
        if line.startswith(("WER:", "cpWER (")):
            assert "%" in line


@pytest.mark.unit
def test_report_prints_counters_even_when_zero():
    text = format_cpwer_report(_scored(CpWERScoringConfig()), CpWERScoringConfig())
    assert "abstained 0" in text and "zero-ref-words 0" in text


@pytest.mark.unit
def test_report_carries_the_stamp_and_the_verdict():
    cfg = CpWERScoringConfig()
    text = format_cpwer_report(_scored(cfg), cfg)
    assert "cpwer_axes:" in text
    assert "reference-comparable:" in text


@pytest.mark.unit
def test_subset_macro_is_named_so_it_cannot_be_mistaken_for_the_others():
    """A third average: unweighted over subsets, distinct from corpus micro and session macro.

    Uses deliberately unbalanced subsets, because on balanced ones the two coincide and the test
    would pass without demonstrating anything.
    """
    rows = [
        # `big` has many words and one error; `small` has few words and one error, so weighting by
        # words (micro) and weighting by subset (subset-macro) must disagree.
        {
            "text_raw": "<spk:0> a b c d e f g h",
            "pred_text_raw": "<spk:0> a b c d e f g X",
            "subset_for_metrics": "big",
        },
        {"text_raw": "<spk:0> p q", "pred_text_raw": "<spk:0> p X", "subset_for_metrics": "small"},
    ]
    m = _scored(CpWERScoringConfig(), rows=rows)
    assert m["cpwer_per_subset"]["big"]["cpwer"] == pytest.approx(12.5)  # 1/8
    assert m["cpwer_per_subset"]["small"]["cpwer"] == pytest.approx(50.0)  # 1/2
    assert m["cpwer"] == pytest.approx(20.0)  # micro: 2 errors / 10 words
    assert m["cpwer_subset_macro"] == pytest.approx(31.25)  # mean(12.5, 50.0)
