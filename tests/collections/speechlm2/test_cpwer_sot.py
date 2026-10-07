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
"""SOT -> per-speaker parsing and cpWER scoring."""

import pytest

from nemo.collections.asr.metrics.cpwer import calculate_session_cpWER, calculate_session_cpWER_detail
from nemo.collections.asr.parts.utils.sot_speaker_alignment import remove_speaker_tags, sot_to_speaker_texts
from nemo.collections.speechlm2.parts.metrics import CpWER


@pytest.mark.unit
@pytest.mark.parametrize(
    "text, expected",
    [
        ("<spk:0> hello there <spk:1> hi <spk:0> how are you", {0: "hello there how are you", 1: "hi"}),
        # words before the first tag are KEPT: parse_speaker_tokens drops them, which would make
        # them phantom deletions against the reference.
        ("um <spk:0> hello", {0: "um hello"}),
        # untagged text must not raise -- strip_speaker_tags does, which would crash the control arm.
        ("no tags at all here", {0: "no tags at all here"}),
        # a tag with no words still registers the speaker
        ("<spk:0> hi <spk:1>", {0: "hi", 1: ""}),
        # malformed residue is dropped, not scored as the words "spk"/"0"
        ("<spk:0 hello <spk:1> hi", {0: "hello", 1: "hi"}),
        # the default bucket is only created if a word actually needs it
        ("<spk:2> only speaker two", {2: "only speaker two"}),
        ("", {}),
        (None, {}),
    ],
)
def test_sot_to_speaker_texts(text, expected):
    assert sot_to_speaker_texts(text) == expected


@pytest.mark.unit
def test_sot_to_speaker_texts_max_speakers_folds_rather_than_drops():
    assert sot_to_speaker_texts("<spk:0> a <spk:5> b", max_speakers=3) == {0: "a", 3: "b"}


@pytest.mark.unit
@pytest.mark.parametrize(
    "text, expected",
    [("<spk:0> hello <spk:1> world", "hello world"), ("plain text", "plain text"), ("<spk:0>", ""), (None, "")],
)
def test_remove_speaker_tags(text, expected):
    assert remove_speaker_tags(text) == expected


@pytest.mark.unit
def test_cpwer_detail_matches_the_public_api():
    hyp = ["hey how are you we that's nice", "i'm good yes hi is your sister"]
    ref = ["hi how are you well that's nice", "i'm good yeah how is your sister"]
    rate, min_perm, ref_trans = calculate_session_cpWER(hyp, ref)
    detail = calculate_session_cpWER_detail(hyp, ref)
    assert detail.cpwer == rate
    assert detail.min_perm_hyp_trans == min_perm
    assert detail.ref_trans == ref_trans
    # the counts the public API cannot give, needed for a corpus micro-average
    assert detail.errors == 4 and detail.ref_words == 14
    assert detail.ins + detail.dels + detail.subs == detail.errors


@pytest.mark.unit
def test_cpwer_detail_marks_padded_reference_slots():
    detail = calculate_session_cpWER_detail(["a b c"], ["a b c", "d e f g"])
    assert detail.assignment == [0, -1], "the unmatched reference speaker should be marked as padded"
    assert detail.dels == 4 and detail.ref_words == 7


@pytest.mark.unit
def test_cpwer_is_permutation_invariant():
    metric = CpWER(normalize=False, verbose=False)
    ref = "<spk:0> hello there <spk:1> general kenobi"
    assert metric.score_session(ref, ref).cpwer == 0.0
    swapped = "<spk:1> hello there <spk:0> general kenobi"
    assert metric.score_session(ref, swapped).cpwer == 0.0


@pytest.mark.unit
def test_cpwer_charges_an_untagged_hypothesis_and_matches_its_ceiling():
    """The control arm emits no tags. cpWER must charge it, not silently score 0."""
    metric = CpWER(normalize=False, verbose=False)
    ref = "<spk:0> hello there <spk:1> general kenobi"
    result = metric.score_session(ref, "hello there general kenobi")
    assert result.cpwer == 1.0, "a word-perfect but unattributed hypothesis must still be charged"
    assert result.num_hyp_speakers == 1 and result.num_ref_speakers == 2
    # word-perfect-but-tagless is exactly the no-tag ceiling, so the two are directly comparable
    assert result.notag_ceiling == pytest.approx(result.cpwer)


@pytest.mark.unit
def test_speakers_are_split_before_normalization():
    """Normalizers strip <...> spans, so normalizing first collapses every session to one speaker."""

    def bracket_stripping_normalizer(text):
        import re

        return re.sub(r"[<\[][^>\]]*[>\]]", "", text).lower().strip()

    metric = CpWER(normalize=True, normalizer=bracket_stripping_normalizer, verbose=False)
    ref = "<spk:0> HELLO THERE <spk:1> GENERAL KENOBI"
    result = metric.score_session(ref, ref)
    assert result.num_ref_speakers == 2, "tags were consumed by the normalizer before splitting"
    assert result.cpwer == 0.0


@pytest.mark.unit
def test_empty_speaker_buckets_survive_normalization():
    """A speaker whose only word is a filler must remain a speaker.

    22/600 references in the multi-speaker debug set have exactly this shape. If the emptied bucket
    is dropped, a reference speaker vanishes and the permutation search silently misaligns.
    """

    def filler_dropping_normalizer(text):
        return " ".join(w for w in text.lower().split() if w not in {"hmm", "uh", "um"})

    metric = CpWER(normalize=True, normalizer=filler_dropping_normalizer, verbose=False)
    result = metric.score_session("<spk:0> hello world <spk:1> hmm", "<spk:0> hello world <spk:1> hmm")
    assert result.num_ref_speakers == 2, "the filler-only speaker was dropped"
    assert result.cpwer == 0.0


@pytest.mark.unit
def test_empty_reference_is_excluded_not_infinite():
    """One inf session would poison every corpus aggregate."""
    metric = CpWER(normalize=False, verbose=False)
    result = metric.score_session("", "<spk:0> hello")
    assert result.cpwer is None
    assert result.abstained, "a reference with no streams is not scored at all"
    metric.update("val", ["", "<spk:0> a b"], ["<spk:0> hello", "<spk:0> a b"])
    out = metric.compute()
    assert out["cpwer_val"] == 0.0, "the empty-reference session should not affect the aggregate"
    # Renamed from `cpwer_skipped_empty_ref_val` and widened: it now covers every abstain reason,
    # not only an empty reference.
    assert out["cpwer_abstained_val"] == 1
    assert out["cpwer_admitted_val"] == 1
    assert out["cpwer_sessions_val"] == 2


@pytest.mark.unit
def test_abstained_is_distinct_from_zero_reference_words():
    """Two different things that both leave `cpwer` None, counted separately.

    An abstain is not scored at all. A reference that parses into streams but normalizes to zero
    words IS scored -- its errors pool into the micro numerator against a zero denominator, which
    is what lets micro exceed 100%.
    """
    metric = CpWER(normalize=True, normalizer=lambda s: "", verbose=False)
    metric.update("val", ["<spk:0> hello"], ["<spk:0> world"])
    out = metric.compute()
    assert out["cpwer_abstained_val"] == 0
    assert out["cpwer_zero_ref_words_val"] == 1
    assert out["cpwer_admitted_val"] == 1


@pytest.mark.unit
def test_micro_aggregate_is_error_weighted():
    metric = CpWER(normalize=False, verbose=False)
    # 1 error / 10 words, then 0 errors / 2 words -> micro 1/12, macro (0.1 + 0)/2 = 0.05
    metric.update(
        "val",
        ["<spk:0> " + " ".join(f"w{i}" for i in range(10)), "<spk:0> a b"],
        ["<spk:0> " + " ".join(["x"] + [f"w{i}" for i in range(1, 10)]), "<spk:0> a b"],
    )
    out = metric.compute()
    assert out["cpwer_val"] == pytest.approx(1 / 12)
    assert out["cpwer_macro_val"] == pytest.approx(0.05)


# --------------------------------------------------------------------------- #
# Deferred-identity (suffix) tag placement
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_suffix_placement_matches_prefix_on_the_same_content():
    """`a b <spk:0> c <spk:1>` must bucket identically to `<spk:0> a b <spk:1> c`.

    The two target formats differ only in WHERE the identity is emitted; scoring must not see a
    difference, or an arm comparison would measure the parser rather than the model.
    """
    prefix = sot_to_speaker_texts("<spk:0> how are you <spk:1> i am fine")
    suffix = sot_to_speaker_texts("how are you <spk:0> i am fine <spk:1>", placement='suffix')
    assert prefix == suffix == {0: "how are you", 1: "i am fine"}


@pytest.mark.unit
def test_suffix_placement_drops_the_structural_switch_marker():
    """`<spk_switch>` opens a run but is not a word; scoring it would inflate the reference."""
    got = sot_to_speaker_texts("<spk_switch> how are you <spk:0> <spk_switch> i am fine <spk:1>", placement='suffix')
    assert got == {0: "how are you", 1: "i am fine"}
    assert remove_speaker_tags("<spk_switch> how are you <spk:0>") == "how are you"


@pytest.mark.unit
@pytest.mark.parametrize("marker", ["<|turn_start|>", "<|turn_end|>"])
def test_turn_markers_are_not_words(marker):
    """The turn-start token `<|turn_start|>` (and any `<|turn_x|>`) is a structural marker too."""
    got = sot_to_speaker_texts(f"{marker} how are you <spk:0> {marker} i am fine <spk:1>", placement='suffix')
    assert got == {0: "how are you", 1: "i am fine"}
    assert sot_to_speaker_texts(f"{marker}<spk:0> how are you {marker}<spk:1> i am fine") == got
    assert remove_speaker_tags(f"{marker} how are you <spk:0>") == "how are you"


@pytest.mark.unit
def test_the_turn_start_token_makes_no_phantom_speaker():
    """In front of the first tag, `<|turn_start|>` was a word of the default speaker 0: a third, phantom speaker."""
    ref = "<spk:2> hi there <spk:3> yes"
    hyp = "<|turn_start|><spk:2> hi there <|turn_start|><spk:3> yes"
    assert sot_to_speaker_texts(hyp) == {2: "hi there", 3: "yes"}
    result = CpWER(normalize=False, verbose=False).score_session(ref, hyp)
    assert (result.cpwer, result.num_hyp_speakers) == (0.0, 2)  # it was (2/3, 3): two inserted words


@pytest.mark.unit
@pytest.mark.parametrize("marker", ["<spk_switch>", "<|turn_start|>"])
def test_a_marker_glued_to_a_word_is_dropped(marker):
    """A marker with no space before it was scored as part of the word."""
    text = f"<spk:0> i am fine,{marker} <spk:1> yes{marker}<spk:0> ok"
    assert sot_to_speaker_texts(text) == {0: "i am fine, ok", 1: "yes"}
    assert remove_speaker_tags(text) == "i am fine, yes ok"
    # Glued on both sides, it still separates two words, as in the speaker-agnostic text.
    assert sot_to_speaker_texts(f"<spk:0> yes{marker}ok") == {0: "yes ok"}
    assert remove_speaker_tags(f"<spk:0> yes{marker}ok") == "yes ok"


@pytest.mark.unit
@pytest.mark.parametrize(
    "text,placement",
    [
        ("<spk:0> so you need it <spk:1> i <spk:0> very complex <spk:2> yeah", "prefix"),
        ("so you need it <spk:0> i <spk:1> very complex <spk:0> yeah <spk:2>", "suffix"),
        (
            "<spk_switch> so you need it <spk:0> <spk_switch> i <spk:1> "
            "<spk_switch> very complex <spk:0> <spk_switch> yeah <spk:2>",
            "suffix",
        ),
    ],
    ids=["prefix", "suffix", "suffix_spk_switch"],
)
def test_the_manifest_formats_parse_as_before(text, placement):
    """The three formats of the manifests in use, each read with its own placement, give the same streams."""
    assert remove_speaker_tags(text) == "so you need it i very complex yeah"
    assert sot_to_speaker_texts(text, placement=placement) == {0: "so you need it very complex", 1: "i", 2: "yeah"}


@pytest.mark.unit
def test_the_structural_pattern():
    from nemo.collections.asr.parts.utils.sot_speaker_alignment import SPEAKER_STRUCTURAL_PATTERN

    for marker in ("<spk_switch>", "<|turn_start|>", "<|turn_end|>"):
        assert SPEAKER_STRUCTURAL_PATTERN.fullmatch(marker), marker
    for other in ("<spk:0>", "<|im_start|>", "<|write|>", "<blank>", "<turn_start>", "<|turn_start_2|>", "<|Turn|>"):
        assert not SPEAKER_STRUCTURAL_PATTERN.search(other), other


@pytest.mark.unit
def test_suffix_placement_orphans_an_unclosed_trailing_run():
    """A run the model never closed falls to `default_speaker`, it does not inherit the previous.

    This is the failure mode suffix placement adds: in prefix form a missing tag means the words
    continue the previous speaker, here they are attributed to whoever `default_speaker` names.
    """
    got = sot_to_speaker_texts("how are you <spk:0> i am fine", placement='suffix', default_speaker=0)
    assert got == {0: "how are you i am fine"}
    assert sot_to_speaker_texts("how are you <spk:1> i am fine", placement='suffix', default_speaker=None) == {
        1: "how are you"
    }


@pytest.mark.unit
def test_zero_ref_words_is_scored_as_zero():
    """GATE 2: a reference that parses into streams but normalizes to no words scores 0.0.

    Not None, and not an abstain. The row IS scored: its errors pool into the micro numerator
    against a zero denominator contribution -- which is what lets micro exceed 100% -- and a literal
    0.0 joins the macro list, pulling the macro down. Both are deliberate, so that this agrees with
    the reference scorer rather than being independently defensible.

    Distinct from GATE 1 (the reference parses to zero streams), which is not scored at all.
    """
    # The real case: a reference of pure filler. The normalizer erases "mm" and keeps the
    # hypothesis, so the reference has no words while the hypothesis has two.
    metric = CpWER(normalize=True, verbose=False)

    result = metric.score_session("<spk:0> mm", "<spk:0> hello world")
    assert result.cpwer == 0.0, "gate 2 scores 0.0, not None"
    assert not result.abstained, "gate 2 is scored; only gate 1 abstains"
    assert result.ref_words == 0
    assert result.errors == 2, "the hypothesis words are real insertions"

    # One clean session plus one gate-2 session.
    metric.update("val", ["<spk:0> a b", "<spk:0> mm"], ["<spk:0> a b", "<spk:0> hello world"])
    out = metric.compute()
    assert out["cpwer_zero_ref_words_val"] == 1
    assert out["cpwer_abstained_val"] == 0
    assert out["cpwer_admitted_val"] == 2
    # Micro: 2 errors over 2 reference words -- the gate-2 row contributed errors but no denominator,
    # which is exactly how micro can exceed 100%.
    assert out["cpwer_errors_val"] == 2 and out["cpwer_ref_words_val"] == 2
    assert out["cpwer_val"] == 1.0
    # Macro: mean(0.0 from the clean session, 0.0 from the gate-2 row) -- the spurious 0.0 is in.
    assert out["cpwer_macro_val"] == 0.0
    assert len(out) and out["cpwer_sessions_val"] == 2


@pytest.mark.unit
def test_gate_two_row_shape_stays_distinct_from_an_abstain():
    """Same "no rate", two situations -- a reader must be able to tell them apart by key presence."""
    from nemo.collections.speechlm2.parts.metrics.cpwer_scoring import CpWERScoringConfig, score_rows

    counts = ("cpwer_errors", "cpwer_ref_words", "cpwer_insertions", "cpwer_deletions", "cpwer_substitutions")

    # Gate 2: parses into one stream that normalizes empty -> scored, all five counts, cpwer 0.0.
    gate2 = score_rows(
        [{"text_raw": "<spk:0> mm", "pred_text_raw": "<spk:0> hello"}],
        CpWERScoringConfig(use_normalizer="none", cpwer_normalizer="none"),
    )[0][0]

    # Gate 1: no tag at all, under a config that discards untagged reference text -> zero streams.
    gate1 = score_rows(
        [{"text_raw": "no tags here", "pred_text_raw": "<spk:0> hello"}],
        CpWERScoringConfig(cpwer_untagged_speaker_ref=None),
    )[0][0]

    assert gate1["cpwer"] is None and all(k not in gate1 for k in counts)
    assert gate2["cpwer"] is not None and all(k in gate2 for k in counts)
