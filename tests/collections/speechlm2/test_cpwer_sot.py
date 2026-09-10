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
    assert metric.score_session("", "<spk:0> hello").cpwer is None
    metric.update("val", ["", "<spk:0> a b"], ["<spk:0> hello", "<spk:0> a b"])
    out = metric.compute()
    assert out["cpwer_val"] == 0.0, "the empty-reference session should not affect the aggregate"
    assert out["cpwer_skipped_empty_ref_val"] == 1


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
