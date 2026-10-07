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
"""Pin the scoring axes on ``sot_to_speaker_texts``.

Each keyword-only argument selects a behaviour some other scorer has, so that a number can be
reproduced without adopting a foreign pipeline wholesale. Two properties are pinned here:

* **Every default reproduces the pre-axis behaviour.** Verified separately by a 64,000-case
  differential against the previous implementation across 16 option combinations; the cases below
  are the readable subset that would localise a break.
* **Each axis moves exactly one thing.** An axis that silently changed a second behaviour would be
  indistinguishable from a scoring bug at the far end of the pipeline.
"""

import pytest

from nemo.collections.asr.parts.utils.sot_speaker_alignment import sot_to_speaker_texts as parse

# --------------------------------------------------------------------------------------------
# defaults: the pre-axis behaviour
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "text,expected",
    [
        ("<spk:0> hello there <spk:1> hi", {0: "hello there", 1: "hi"}),
        ("<spk:0>a<spk:1>b", {0: "a", 1: "b"}),  # no whitespace around tags
        ("hello there <spk:1> hi", {0: "hello there", 1: "hi"}),  # pre-tag run -> default bucket
        ("no tags at all", {0: "no tags at all"}),
        ("", {}),
        ("<spk:0> a <spk:1> b <spk:0> c", {0: "a c", 1: "b"}),  # same speaker resumes
        ("<spk:2> hi", {2: "hi"}),  # default bucket NOT invented when nothing precedes
    ],
)
def test_defaults_match_the_pre_axis_parser(text, expected):
    assert parse(text) == expected


@pytest.mark.unit
def test_empty_bucket_is_kept_by_default():
    """A tag with no words still creates a speaker: bucket count comes from TAGS, not from text."""
    assert parse("<spk:0> <spk:1> hi") == {0: "", 1: "hi"}
    assert parse("<spk:0> <spk:1> hi", keep_empty=False) == {1: "hi"}


# --------------------------------------------------------------------------------------------
# tag_syntax
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "syntax,text,expected",
    [
        ("spk", "<spk:0> a <spk:1> b", {0: "a", 1: "b"}),
        ("bracket", "[s0] a [s1] b", {0: "a", 1: "b"}),
        ("spk+bracket", "<spk:0> a [s1] b", {0: "a", 1: "b"}),
        ("canonical", "speaker 1: a <spk:2> b", {1: "a", 2: "b"}),
        ("canonical", "speaker_3 - a", {3: "a"}),
    ],
)
def test_tag_syntax_selects_the_pattern(syntax, text, expected):
    assert parse(text, tag_syntax=syntax) == expected


@pytest.mark.unit
def test_bracket_tags_are_invisible_to_the_default_syntax():
    """Not a bug: `[s0]` is only a tag when the axis says so, so the whole text is one speaker."""
    assert parse("[s0] a [s1] b") == {0: "[s0] a [s1] b"}


@pytest.mark.unit
@pytest.mark.parametrize(
    "alias,canonical",
    [
        ("<spk:*>", "spk"),
        ("<spk:n>", "spk"),
        ("spk_tag", "spk"),
        ("[s*]", "bracket"),
        ("s_bracket", "bracket"),
        ("[s0]", "bracket"),
        ("both", "spk+bracket"),
        ("all", "spk+bracket"),
        ("SPK", "spk"),
        (" both ", "spk+bracket"),
    ],
)
def test_tag_syntax_aliases_from_other_scorers(alias, canonical):
    text = "<spk:0> a [s1] b"
    assert parse(text, tag_syntax=alias) == parse(text, tag_syntax=canonical)


@pytest.mark.unit
def test_unknown_tag_syntax_raises():
    """Falling back silently would score an entire corpus as one speaker."""
    with pytest.raises(ValueError, match="Unknown tag_syntax"):
        parse("<spk:0> a", tag_syntax="spk_bracket_thing")


@pytest.mark.unit
def test_canonical_false_positive_on_prose_is_intentional():
    """`speaker 3-4 was chosen` reads as a tag. Inherited to stay in agreement with that scorer."""
    assert parse("and then speaker 3-4 was chosen", tag_syntax="canonical") == {
        0: "and then",
        3: "4 was chosen",
    }
    # ...and does NOT fire under the default syntax.
    assert parse("and then speaker 3-4 was chosen") == {0: "and then speaker 3-4 was chosen"}


# --------------------------------------------------------------------------------------------
# case_sensitive
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_uppercase_tags_are_text_by_default_and_tags_when_asked():
    # The whole turn collapses into one bucket carrying the literal tag text -- worse than
    # abstaining, which is why the axis exists.
    assert parse("<SPK:0> a <SPK:1> b") == {0: "<SPK:0> a <SPK:1> b"}
    assert parse("<SPK:0> a <SPK:1> b", case_sensitive=False) == {0: "a", 1: "b"}


@pytest.mark.unit
def test_case_insensitive_does_not_change_lowercase_input():
    text = "<spk:0> a <spk:1> b"
    assert parse(text, case_sensitive=False) == parse(text)


# --------------------------------------------------------------------------------------------
# drop_tag_residue
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("placement", ["prefix", "suffix"])
def test_residue_axis_applies_to_both_placements(placement):
    """The filter lives in one place now; under the old two-loop body it was easy to fix only one.

    A residue filter that worked in prefix but not suffix would be a silent no-op for
    deferred-identity targets.
    """
    text = "<spk:0> alpha <spk:0 bravo" if placement == "prefix" else "alpha <spk:0 bravo <spk:0>"
    dropped = parse(text, placement=placement, drop_tag_residue=True)
    kept = parse(text, placement=placement, drop_tag_residue=False)
    assert "<spk:0" not in " ".join(dropped.values())
    assert "<spk:0" in " ".join(kept.values())


@pytest.mark.unit
def test_structural_markers_are_stripped_on_every_axis_setting():
    """`<spk_switch>` is not a word -- scoring it would inflate the reference word count."""
    for residue in (True, False):
        for placement in ("prefix", "suffix"):
            out = parse(
                "<spk:0> alpha <spk_switch> bravo",
                placement=placement,
                drop_tag_residue=residue,
            )
            assert "<spk_switch>" not in " ".join(out.values())


@pytest.mark.unit
@pytest.mark.parametrize("marker", ["<|turn_start|>", "<|turn_end|>"])
def test_turn_markers_are_stripped_on_every_axis_setting(marker):
    """`<|turn_start|>`, the turn-start token, is not a word either, nor is any `<|turn_x|>`."""
    for residue in (True, False):
        for placement in ("prefix", "suffix"):
            for syntax in ("spk", "canonical"):
                out = parse(
                    f"{marker}<spk:0> alpha {marker} bravo",
                    placement=placement,
                    drop_tag_residue=residue,
                    tag_syntax=syntax,
                )
                assert out == {0: "alpha bravo"}, out


# --------------------------------------------------------------------------------------------
# speaker_order
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_speaker_order_changes_order_only_not_content():
    text = "<spk:2> a <spk:0> b <spk:11> c"
    by_index = parse(text)
    by_seen = parse(text, speaker_order="first_seen")
    assert list(by_index) == [0, 2, 11]  # numeric, so 11 sorts after 2
    assert list(by_seen) == [2, 0, 11]
    assert by_index == by_seen  # same mapping, different iteration order
    assert all(isinstance(k, int) for k in by_seen)


# --------------------------------------------------------------------------------------------
# axes that already existed, re-pinned because the body was rewritten under them
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_suffix_orphan_trailing_run_falls_to_the_default_bucket():
    """Words after the last closing tag were never attributed; they do NOT inherit the previous."""
    assert parse("a <spk:1> b", placement="suffix") == {0: "b", 1: "a"}


@pytest.mark.unit
def test_untagged_speaker_none_discards_the_pre_tag_run():
    assert parse("hello <spk:1> hi", default_speaker=None) == {1: "hi"}
    assert parse("no tags at all", default_speaker=None) == {}


@pytest.mark.unit
def test_max_speakers_folds_and_yields_n_plus_one_buckets():
    assert parse("<spk:0> a <spk:1> b <spk:5> c", max_speakers=2) == {0: "a", 1: "b", 2: "c"}
