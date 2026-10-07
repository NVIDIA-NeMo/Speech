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

"""SOT ``<spk:N>`` tag emission in the interleaved per-chunk targets."""

import re
import warnings
from dataclasses import dataclass

import pytest
from omegaconf import OmegaConf

from nemo.collections.asr.parts.utils.sot_speaker_alignment import sot_to_speaker_texts
from nemo.collections.speechlm2.data.streaming_stt_dataset import (
    get_llm_messages_for_batch,
    get_llm_messages_for_sample,
)
from nemo.collections.speechlm2.parts.alignments import WordAlignment

TAG = re.compile(r"<spk:(\d+)>")
# The structural markers, as `SPEAKER_STRUCTURAL_PATTERN` defines them; spelled out so that a test fails on code that
# has no such pattern rather than at import.
MARKER = re.compile(r"<spk_[a-z]+>|<\|turn_[a-z]+\|>")


def _align(spec, step=0.2):
    """``spec`` is [(word, speaker), ...] laid out on a regular grid."""
    return [WordAlignment(w, i * step, i * step + step * 0.9, speaker=s) for i, (w, s) in enumerate(spec)]


def _turns(alignments, transcript, **kwargs):
    """Non-blank assistant contents; ``kwargs`` go to :func:`_messages`."""
    messages = _messages(alignments, transcript, **kwargs)
    return [m["content"] for m in messages if m["role"] == "assistant" and m["content"] != "<blank>"]


def _messages(
    alignments,
    transcript,
    *,
    chunk_size=2,
    write_token="",
    prepend=False,
    template="<spk:{i}>",
    num_delay_frames=0,
    audio_duration_secs=None,
    **kwargs,
):
    return get_llm_messages_for_sample(
        system_role="system",
        system_prompt="Transcribe.",
        audio_tag="<audio>",
        blank_token="<blank>",
        chunk_size=chunk_size,
        num_delay_frames=num_delay_frames,
        audio_duration_secs=(len(alignments) * 0.2 + 0.5 if audio_duration_secs is None else audio_duration_secs),
        frame_length_in_secs=0.08,
        alignments=alignments,
        transcript=transcript,
        words_per_group=1,
        prepend_write_token=prepend,
        write_token=write_token,
        speaker_token_template=template,
        **kwargs,
    )


def _speaker_texts(text, placement="prefix", switch="<spk_switch>"):
    """Per-speaker word lists, as cpWER sees them."""
    text = text.replace("<|write|>", " ").replace(switch, " ")
    return {k: v.split() for k, v in sot_to_speaker_texts(text, placement=placement).items()}


class TestSpeakerTagEmission:
    @pytest.mark.unit
    def test_tag_emitted_only_on_speaker_change(self):
        al = _align([("a", 0), ("b", 0), ("c", 1), ("d", 1)])
        out = " ".join(_turns(al, "<spk:0> a b <spk:1> c d"))
        assert TAG.findall(out) == ["0", "1"], "one tag per change, not per word"

    @pytest.mark.unit
    def test_emitted_tag_sequence_matches_the_transcript(self):
        transcript = "<spk:0> a <spk:1> b <spk:0> c d <spk:2> e"
        al = _align([("a", 0), ("b", 1), ("c", 0), ("d", 0), ("e", 2)])
        assert TAG.findall(" ".join(_turns(al, transcript))) == TAG.findall(transcript)

    @pytest.mark.unit
    def test_write_token_stays_outermost(self):
        # Q11: `prepend_write_token` exists so the LM's first output token is a binary blank/write
        # decision. A tag outside it would make that distribution multi-modal.
        al = _align([("a", 0), ("b", 1)])
        out = _turns(al, "<spk:0> a <spk:1> b", write_token="<|write|>", prepend=True)
        tagged = [c for c in out if "<spk:" in c]
        assert tagged, "expected at least one tagged turn"
        assert all(c.startswith("<|write|><spk:") for c in tagged)

    @pytest.mark.unit
    def test_no_double_space_after_an_injected_tag(self):
        al = _align([("alpha", 0), ("beta", 1)])
        for content in _turns(al, "<spk:0> alpha <spk:1> beta"):
            assert "  " not in content

    @pytest.mark.unit
    def test_template_none_disables_tagging(self):
        al = _align([("a", 0), ("b", 1)])
        out = " ".join(_turns(al, "<spk:0> a <spk:1> b", template=None))
        assert "<spk:" not in out

    @pytest.mark.unit
    def test_words_without_speaker_are_untagged(self):
        # Single-speaker manifests carry no `speaker_ids`; the path must stay inert.
        al = [WordAlignment("a", 0.0, 0.2), WordAlignment("b", 0.3, 0.5)]
        assert "<spk:" not in " ".join(_turns(al, "a b"))

    @pytest.mark.unit
    def test_state_follows_the_last_word_of_a_group(self):
        # A group's transcript slice can already carry a mid-group tag, so the "last emitted
        # speaker" is the group's LAST word, not its first. Tracking the first would re-emit a
        # redundant tag (or drop a needed one) on the following group.
        transcript = "<spk:0> a <spk:1> b c"
        al = _align([("a", 0), ("b", 1), ("c", 1)])
        out = " ".join(_turns(al, transcript, chunk_size=40))  # force one big group
        assert TAG.findall(out) == ["0", "1"]


# Multi-speaker transcripts in the manifests' prefix format, with the speaker of every word.
_SOT_CASES = {
    "two_runs": ("<spk:0> a b <spk:1> c d", [("a", 0), ("b", 0), ("c", 1), ("d", 1)]),
    "back_and_forth": (
        "<spk:0> a <spk:1> b <spk:0> c d <spk:2> e",
        [("a", 0), ("b", 1), ("c", 0), ("d", 0), ("e", 2)],
    ),
    "one_word_runs": (
        "<spk:1> yes <spk:0> no <spk:1> yes <spk:2> maybe",
        [("yes", 1), ("no", 0), ("yes", 1), ("maybe", 2)],
    ),
    "punctuation": (
        "<spk:0> Hello, there. <spk:1> Hi! <spk:0> OK.",
        [("hello", 0), ("there", 0), ("hi", 1), ("ok", 0)],
    ),
    "single_speaker": ("<spk:0> just one voice here", [("just", 0), ("one", 0), ("voice", 0), ("here", 0)]),
}


class TestSuffixPlacementAndSwitchToken:
    """PF-6: suffix placement, the flush turn and the switch token build targets from ``speaker_ids``."""

    @pytest.mark.unit
    @pytest.mark.parametrize("chunk_size", [2, 5, 40])
    @pytest.mark.parametrize("switch", [None, "<spk_switch>", "<|turn_start|>"])
    @pytest.mark.parametrize("case", sorted(_SOT_CASES))
    def test_suffix_placement_attributes_words(self, case, switch, chunk_size):
        # A chunk that spans a speaker change used to carry the NEXT run's prefix tag from the
        # transcript slice where the suffix target needs the closing tag of the run it ends, so
        # the words before it went to the wrong speaker.
        transcript, spec = _SOT_CASES[case]
        al = _align(spec)
        out = " ".join(
            _turns(al, transcript, chunk_size=chunk_size, speaker_tag_placement="suffix", turn_start_token=switch)
        )
        assert _speaker_texts(out, placement="suffix") == _speaker_texts(transcript)
        # Every run is closed exactly once, by its own speaker, and opened by one turn-start token.
        runs = [s for i, (_, s) in enumerate(spec) if i + 1 == len(spec) or spec[i + 1][1] != s]
        assert TAG.findall(out) == [str(s) for s in runs]
        assert MARKER.findall(out) == ([switch] * len(runs) if switch else [])

    @pytest.mark.unit
    def test_suffix_placement_one_chunk_exact(self):
        al = _align(_SOT_CASES["two_runs"][1])
        transcript = _SOT_CASES["two_runs"][0]
        assert _turns(al, transcript, chunk_size=40, speaker_tag_placement="suffix") == [" a b <spk:0> c d <spk:1>"]
        assert _turns(
            al, transcript, chunk_size=40, speaker_tag_placement="suffix", turn_start_token="<spk_switch>"
        ) == ["<spk_switch> a b <spk:0> <spk_switch> c d <spk:1>"]

    @pytest.mark.unit
    def test_suffix_placement_keeps_non_speaker_markup(self):
        # Only speaker tags are rebuilt; other markup between the runs stays in the target.
        al = _align([("a", 0), ("b", 1)])
        out = _turns(al, "<spk:0> a <laugh> <spk:1> b", chunk_size=40, speaker_tag_placement="suffix")
        assert out == [" a <laugh> <spk:0> b <spk:1>"]

    @pytest.mark.unit
    @pytest.mark.parametrize("placement", ["prefix", "suffix"])
    def test_flush_turn_keeps_leading_tag(self, placement):
        # `a b` are emitted in the last chunk; the delay pushes `c` (speaker 1) and `d` (speaker 2)
        # past the last boundary, into the flush turn. That turn opens speaker 1's run, so under
        # prefix it must start with <spk:1>; without it `c` was attributed to speaker 0.
        transcript = "<spk:0> a b <spk:1> c <spk:2> d"
        al = [
            WordAlignment("a", 0.00, 0.06, speaker=0),
            WordAlignment("b", 0.08, 0.14, speaker=0),
            WordAlignment("c", 0.16, 0.22, speaker=1),
            WordAlignment("d", 0.24, 0.30, speaker=2),
        ]
        messages = _messages(
            al,
            transcript,
            chunk_size=2,
            num_delay_frames=2,
            audio_duration_secs=0.32,
            write_token="<|write|>",
            prepend=True,
            speaker_tag_placement=placement,
            use_flush_token=True,
            flush_token="<|flush|>",
        )
        flush_at = [m["content"] for m in messages].index("<|flush|>")
        before = [m["content"] for m in messages[:flush_at] if m["role"] == "assistant" and m["content"] != "<blank>"]
        flush_turn = messages[flush_at + 1]["content"]
        assert before == (["<|write|><spk:0> a b"] if placement == "prefix" else ["<|write|> a b <spk:0>"])
        if placement == "prefix":
            assert flush_turn == "<|write|><spk:1> c <spk:2> d"
        else:
            assert flush_turn == "<|write|> c <spk:1> d <spk:2>"
        out = " ".join(before + [flush_turn])
        assert _speaker_texts(out, placement=placement) == _speaker_texts(transcript)

    @pytest.mark.unit
    @pytest.mark.parametrize("chunk_size", [2, 5, 40])
    @pytest.mark.parametrize("case", sorted(_SOT_CASES))
    def test_switch_token_under_prefix(self, case, chunk_size):
        # The switch token used to be read by the suffix branch only, so under prefix it was
        # silently dropped. It now sits directly in front of every identity tag.
        transcript, spec = _SOT_CASES[case]
        al = _align(spec)
        plain = _turns(al, transcript, chunk_size=chunk_size)
        switched = _turns(al, transcript, chunk_size=chunk_size, turn_start_token="<spk_switch>")
        out = " ".join(switched)
        assert out.count("<spk_switch>") == len(TAG.findall(out)) > 0
        assert re.findall(r"<spk_switch>(<spk:\d+>)", out) == re.findall(r"<spk:\d+>", out)
        # Removing the switch token gives back the plain prefix target, word for word.
        assert [c.replace("<spk_switch>", "") for c in switched] == plain

    @pytest.mark.unit
    def test_switch_token_under_prefix_exact(self):
        al = _align(_SOT_CASES["two_runs"][1])
        transcript = _SOT_CASES["two_runs"][0]
        kwargs = dict(turn_start_token="<spk_switch>", write_token="<|write|>", prepend=True)
        assert _turns(al, transcript, chunk_size=40, **kwargs) == [
            "<|write|><spk_switch><spk:0> a b <spk_switch><spk:1> c d"
        ]
        assert _turns(al, transcript, chunk_size=2, **kwargs) == [
            "<|write|><spk_switch><spk:0> a",
            "<|write|> b",
            "<|write|><spk_switch><spk:1> c",
            "<|write|> d",
        ]


# Prefix placement without flush or switch: the exact targets the builder produced before PF-6.
# Any change here changes the supervision of every prefix recipe.
_PREFIX_PINS = [
    ("two_runs", dict(chunk_size=40), ["<spk:0> a b <spk:1> c d"]),
    ("two_runs", dict(chunk_size=2), ["<spk:0> a", " b", "<spk:1> c", " d"]),
    ("two_runs", dict(chunk_size=5), ["<spk:0> a b", "<spk:1> c d"]),
    ("back_and_forth", dict(chunk_size=40), ["<spk:0> a <spk:1> b <spk:0> c d <spk:2> e"]),
    ("back_and_forth", dict(chunk_size=5, num_delay_frames=3), ["<spk:0> a <spk:1> b", "<spk:0> c d", "<spk:2> e"]),
    ("one_word_runs", dict(chunk_size=2), ["<spk:1> yes", "<spk:0> no", "<spk:1> yes", "<spk:2> maybe"]),
    ("punctuation", dict(chunk_size=40), ["<spk:0> Hello, there. <spk:1> Hi! <spk:0> OK."]),
    (
        "punctuation",
        dict(chunk_size=2, num_delay_frames=6),
        ["<spk:0> Hello,", " there.", "<spk:1> Hi!", "<spk:0> OK."],
    ),
    ("single_speaker", dict(chunk_size=2), ["<spk:0> just", " one", " voice", " here"]),
    # residual fold: the delay pushes the last words past the final boundary
    (
        "back_and_forth",
        dict(chunk_size=2, num_delay_frames=3, audio_duration_secs=0.6),
        ["<spk:0> a", "<spk:1> b <spk:0> c d <spk:2> e"],
    ),
    # dynamic chunking
    ("back_and_forth", dict(chunk_size=0), ["<spk:0> a", "<spk:1> b", "<spk:0> c", " d", "<spk:2> e"]),
    # write token outermost
    (
        "two_runs",
        dict(chunk_size=5, write_token="<|write|>", prepend=True),
        ["<|write|><spk:0> a b", "<|write|><spk:1> c d"],
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize("case,kwargs,expected", _PREFIX_PINS)
def test_prefix_placement_is_unchanged(case, kwargs, expected):
    transcript, spec = _SOT_CASES[case]
    al = _align(spec)
    assert _turns(al, transcript, **kwargs) == expected
    assert _turns(al, transcript, turn_start_token=None, use_flush_token=False, **kwargs) == expected


# The four combinations of placement and turn-start token for `a b` of speaker 0 and `c d` of speaker 1, as in the
# table of the docs (all in one chunk), and the two with the token in chunks of one word.
_TURN_START_PINS = [
    ("prefix", None, 40, ["<spk:0> a b <spk:1> c d"]),
    ("prefix", "<|turn_start|>", 40, ["<|turn_start|><spk:0> a b <|turn_start|><spk:1> c d"]),
    ("suffix", None, 40, [" a b <spk:0> c d <spk:1>"]),
    ("suffix", "<|turn_start|>", 40, ["<|turn_start|> a b <spk:0> <|turn_start|> c d <spk:1>"]),
    ("prefix", "<|turn_start|>", 2, ["<|turn_start|><spk:0> a", " b", "<|turn_start|><spk:1> c", " d"]),
    ("suffix", "<|turn_start|>", 2, ["<|turn_start|> a", " b <spk:0>", "<|turn_start|> c", " d <spk:1>"]),
]


class TestTurnStartToken:
    """``turn_start_token``, formerly ``speaker_switch_token``, opens every speaker run under either placement."""

    @pytest.mark.unit
    @pytest.mark.parametrize("placement,token,chunk_size,expected", _TURN_START_PINS)
    def test_every_combination_of_placement_and_token(self, placement, token, chunk_size, expected):
        transcript, spec = _SOT_CASES["two_runs"]
        got = _turns(
            _align(spec), transcript, chunk_size=chunk_size, speaker_tag_placement=placement, turn_start_token=token
        )
        assert got == expected

    @pytest.mark.unit
    @pytest.mark.parametrize("placement", ["prefix", "suffix"])
    def test_the_deprecated_keyword_still_sets_the_token(self, placement):
        transcript, spec = _SOT_CASES["back_and_forth"]
        al = _align(spec)
        expected = _turns(al, transcript, speaker_tag_placement=placement, turn_start_token="<spk_switch>")
        with pytest.warns(
            FutureWarning, match="speaker_switch_token is deprecated: it was renamed to turn_start_token"
        ):
            got = _turns(al, transcript, speaker_tag_placement=placement, speaker_switch_token="<spk_switch>")
        assert got == expected
        assert " ".join(got).count("<spk_switch>") == 4  # a | b | c d | e

    @pytest.mark.unit
    def test_the_batch_builder_resolves_the_deprecated_keyword_once(self):
        transcript, spec = _SOT_CASES["two_runs"]
        kwargs = dict(
            system_role="system",
            system_prompt=["Transcribe."] * 2,
            audio_tag="<audio>",
            blank_token="<blank>",
            chunk_size=40,
            num_delay_frames=0,
            audio_durations_secs=[1.3, 1.3],
            frame_length_in_secs=0.08,
            alignments=[_align(spec)] * 2,
            transcripts=[transcript] * 2,
            speaker_token_template="<spk:{i}>",
        )
        expected = get_llm_messages_for_batch(**kwargs, turn_start_token="<spk_switch>")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = get_llm_messages_for_batch(**kwargs, speaker_switch_token="<spk_switch>")
        assert got == expected
        assert [issubclass(w.category, FutureWarning) for w in caught] == [True]
        assert "<spk_switch><spk:0> a b <spk_switch><spk:1> c d" in [m["content"] for m in got[1]]

    @pytest.mark.unit
    def test_both_keywords(self):
        transcript, spec = _SOT_CASES["two_runs"]
        al = _align(spec)
        expected = _turns(al, transcript, turn_start_token="<|turn_start|>")
        with pytest.warns(FutureWarning):
            assert _turns(
                al, transcript, turn_start_token="<|turn_start|>", speaker_switch_token="<|turn_start|>"
            ) == (expected)
        with pytest.raises(ValueError, match=r"turn_start_token='<\|turn_start\|>' and speaker_switch_token="):
            _turns(al, transcript, turn_start_token="<|turn_start|>", speaker_switch_token="<spk_switch>")

    @pytest.mark.unit
    @pytest.mark.parametrize("unset", [None, ""])
    def test_an_unset_keyword_is_ignored(self, unset):
        transcript, spec = _SOT_CASES["two_runs"]
        al = _align(spec)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            got = _turns(al, transcript, turn_start_token="<|turn_start|>", speaker_switch_token=unset)
            assert _turns(al, transcript, turn_start_token=unset) == _turns(al, transcript)
        assert got == _turns(al, transcript, turn_start_token="<|turn_start|>")

    @pytest.mark.unit
    @pytest.mark.parametrize("placement", ["postfix", "Suffix", "PREFIX", ""])
    def test_an_unknown_placement_is_rejected(self, placement):
        # Any value but 'suffix' used to train prefix targets silently.
        transcript, spec = _SOT_CASES["two_runs"]
        with pytest.raises(ValueError, match=r"speaker_tag_placement=.* use 'prefix' .* or 'suffix'"):
            _turns(_align(spec), transcript, speaker_tag_placement=placement)

    @pytest.mark.unit
    @pytest.mark.parametrize("chunk_size", [2, 40])
    @pytest.mark.parametrize("marker", ["<spk_switch>", "<|turn_start|>"])
    @pytest.mark.parametrize(
        "placement,token",
        [("suffix", None), ("suffix", "<|turn_start|>"), ("prefix", "<|turn_start|>"), ("suffix", "<spk_switch>")],
    )
    def test_markers_in_the_manifest_text_stay_out_of_the_targets(self, placement, token, marker, chunk_size):
        """A transcript that carries markers of its own, such as the suffix-format text written for ``<spk_switch>``,
        gives the targets of the clean prefix-format transcript: the configured token alone, where it belongs. A
        marker that was not the configured token used to leak into the targets."""
        clean, spec = _SOT_CASES["two_runs"]
        marked = f"{marker} a b <spk:0> {marker} c d <spk:1>"
        kwargs = dict(chunk_size=chunk_size, speaker_tag_placement=placement)
        if token:
            kwargs["turn_start_token"] = token
        assert _turns(_align(spec), marked, **kwargs) == _turns(_align(spec), clean, **kwargs)

    @pytest.mark.unit
    @pytest.mark.parametrize("placement", ["prefix", "suffix"])
    @pytest.mark.parametrize("token,marker", [("<|turn_start|>", "<spk_switch>"), ("<spk_switch>", "<|turn_start|>")])
    def test_another_marker_stays_out_under_the_deprecated_keyword(self, token, marker, placement):
        """As above, with a marker other than the configured token, set through the deprecated keyword, which the
        builder took before the rename too: the marker used to leak into the targets."""
        _, spec = _SOT_CASES["two_runs"]
        marked = f"{marker} a b <spk:0> {marker} c d <spk:1>"
        expected = {
            "prefix": f"{token}<spk:0> a b {token}<spk:1> c d",
            "suffix": f"{token} a b <spk:0> {token} c d <spk:1>",
        }[placement]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = _turns(
                _align(spec), marked, chunk_size=40, speaker_tag_placement=placement, speaker_switch_token=token
            )
        assert got == [expected]
        assert [issubclass(w.category, FutureWarning) for w in caught] == [True]


_DATA = {"sample_rate": 16000, "frame_length_in_secs": 0.08, "chunk_size": 14}
# (keys, the resolved token, whether the deprecated key warns): `null` and "" count as unset under either key.
_ALIASES = {
    "unset": ({}, None, False),
    "new_only": ({"turn_start_token": "<|turn_start|>"}, "<|turn_start|>", False),
    "old_only": ({"speaker_switch_token": "<spk_switch>"}, "<spk_switch>", True),
    "both_equal": ({"turn_start_token": "<spk_switch>", "speaker_switch_token": "<spk_switch>"}, "<spk_switch>", True),
    "old_null": ({"turn_start_token": "<|turn_start|>", "speaker_switch_token": None}, "<|turn_start|>", False),
    "old_empty": ({"turn_start_token": "<|turn_start|>", "speaker_switch_token": ""}, "<|turn_start|>", False),
    "new_null": ({"turn_start_token": None, "speaker_switch_token": "<spk_switch>"}, "<spk_switch>", True),
    "new_empty": ({"turn_start_token": "", "speaker_switch_token": "<spk_switch>"}, "<spk_switch>", True),
    "both_empty": ({"turn_start_token": "", "speaker_switch_token": ""}, None, False),
}


def _data_config(cls=None, **keys):
    from nemo.collections.speechlm2.data.streaming_stt_dataset import StreamingSTTDataConfig
    from nemo.collections.speechlm2.parts.utils import to_dataclass

    return to_dataclass(cls or StreamingSTTDataConfig, keys.pop("raw", None) or {**_DATA, **keys})


class TestTurnStartTokenDataConfig:
    """``turn_start_token`` replaces ``speaker_switch_token``, which is still read, in the dataset config."""

    @pytest.mark.unit
    @pytest.mark.parametrize("case", list(_ALIASES))
    def test_alias_rules(self, case):
        keys, resolved, warns = _ALIASES[case]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = _data_config(**keys)
        deprecations = [str(w.message) for w in caught if issubclass(w.category, FutureWarning)]
        assert cfg.turn_start_token == resolved
        assert cfg.speaker_switch_token is None
        assert deprecations == (
            [
                "speaker_switch_token is deprecated: it was renamed to turn_start_token. "
                f"Using {resolved!r} as turn_start_token."
            ]
            if warns
            else []
        )

    @pytest.mark.unit
    def test_the_deprecation_reaches_the_nemo_log(self, monkeypatch):
        # NeMo logs every warning and drops those whose category has an 'ignore' filter, whatever the filter's
        # message. Lightning, which nemo.collections.speechlm2 imports, registers one for FutureWarning (for FSDP
        # messages only), so a plain FutureWarning never reached the log.
        from nemo.utils import logging as nemo_logging

        logged = []
        monkeypatch.setattr(nemo_logging, "warning", lambda msg, *args, **kwargs: logged.append(msg % args))
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*FSDP.state_dict_type.*")
            warnings.showwarning = nemo_logging._showwarning
            cfg = _data_config(speaker_switch_token="<spk_switch>")
        assert cfg.turn_start_token == "<spk_switch>"
        assert [m for m in logged if "speaker_switch_token is deprecated" in m], logged

    @pytest.mark.unit
    def test_different_values_under_both_keys_are_an_error(self):
        message = (
            r"turn_start_token='<\|turn_start\|>' and speaker_switch_token='<spk_switch>' are both set and differ"
        )
        with pytest.raises(ValueError, match=message):
            _data_config(turn_start_token="<|turn_start|>", speaker_switch_token="<spk_switch>")

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "model_key,data_key", [("switch_token", "speaker_switch_token"), ("turn_start_token",) * 2]
    )
    def test_a_key_interpolated_from_the_model_key(self, model_key, data_key):
        root = OmegaConf.create(
            {
                "model": {"speaker_tokens": {model_key: "<|turn_start|>"}},
                "data": {"dataset": {**_DATA, data_key: f"${{model.speaker_tokens.{model_key}}}"}},
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            assert _data_config(raw=root.data.dataset).turn_start_token == "<|turn_start|>"

    @pytest.mark.unit
    def test_a_subclass_reads_the_deprecated_key(self):
        from nemo.collections.speechlm2.data.streaming_stt_dataset import StreamingSTTDataConfig

        @dataclass
        class Sub(StreamingSTTDataConfig):
            extra: int = 0

        with pytest.warns(FutureWarning):
            cfg = _data_config(Sub, speaker_switch_token="<spk_switch>")
        assert (cfg.turn_start_token, cfg.speaker_switch_token) == ("<spk_switch>", None)

    @pytest.mark.unit
    @pytest.mark.parametrize("placement", ["postfix", "Prefix", "pre"])
    def test_an_unknown_placement_is_rejected(self, placement):
        with pytest.raises(ValueError, match=r"speaker_tag_placement=.* use 'prefix' .* or 'suffix'"):
            _data_config(speaker_tag_placement=placement)

    @pytest.mark.unit
    @pytest.mark.parametrize("placement", ["prefix", "suffix"])
    def test_prefix_and_suffix_are_accepted(self, placement):
        assert _data_config(speaker_tag_placement=placement).speaker_tag_placement == placement


@pytest.fixture(scope="module")
def tokenizer():
    from nemo.collections.common.tokenizers import AutoTokenizer

    tok = AutoTokenizer("Qwen/Qwen3-1.7B", use_fast=True)
    markers = ["<|turn_start|>", "<spk_switch>", "<turn>", "x<|turn_start|>"]
    tok.add_special_tokens({"additional_special_tokens": ["<blank>"] + [f"<spk:{i}>" for i in range(4)] + markers})
    return tok


def _dataset(tokenizer, multispeaker=True, **keys):
    from nemo.collections.speechlm2.data.streaming_stt_dataset import StreamingSTTDataset

    cfg = {**_DATA, "blank_token": "<blank>", "words_per_group": 1, **keys}
    if multispeaker:
        cfg["multispeaker_cfg"] = {"enable": True, "num_speakers": 4}
    return StreamingSTTDataset(cfg=OmegaConf.create(cfg), tokenizer=tokenizer)


class TestTurnStartTokenDataset:
    """The dataset checks the turn-start token and writes it into the targets."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "keys", [{"turn_start_token": "<|turn_start|>"}, {"speaker_switch_token": "<spk_switch>"}], ids=["new", "old"]
    )
    def test_a_registered_marker_is_accepted(self, tokenizer, keys):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            dataset = _dataset(tokenizer, **keys)
        assert dataset.cfg.turn_start_token == next(iter(keys.values()))

    @pytest.mark.unit
    @pytest.mark.parametrize("token", ["<turn>", "x<|turn_start|>"], ids=["no_marker", "holds_a_marker"])
    def test_a_token_that_is_not_a_marker_is_rejected(self, tokenizer, token):
        # One token, but scoring would count it, or what it holds besides a marker, as a word.
        with pytest.raises(ValueError, match=re.escape(f"turn_start_token={token!r} must match")):
            _dataset(tokenizer, turn_start_token=token)

    @pytest.mark.unit
    def test_an_unregistered_token_is_rejected(self, tokenizer):
        with pytest.raises(ValueError, match=r"turn_start_token '<\|turn_end\|>' tokenizes into \d+ tokens"):
            _dataset(tokenizer, turn_start_token="<|turn_end|>")

    @pytest.mark.unit
    def test_without_multispeaker_the_token_is_not_used_or_checked(self, tokenizer):
        assert _dataset(tokenizer, multispeaker=False, turn_start_token="<|turn_end|>")._multispeaker_enabled is False

    @pytest.mark.unit
    def test_an_unknown_placement_is_rejected(self, tokenizer):
        with pytest.raises(ValueError, match="speaker_tag_placement='postfix'"):
            _dataset(tokenizer, speaker_tag_placement="postfix")

    @pytest.mark.unit
    @pytest.mark.parametrize("placement", ["prefix", "suffix"])
    @pytest.mark.parametrize("keys", [{"turn_start_token": "<|turn_start|>"}, {}], ids=["token", "no_token"])
    def test_the_targets_open_every_run_with_the_token(self, tokenizer, placement, keys):
        from lhotse import CutSet, MonoCut, SupervisionSegment
        from lhotse.testing.dummies import dummy_recording

        from nemo.collections.speechlm2.data.streaming_stt_dataset import IGNORE_INDEX

        words = [("a", 0.0, 0.2, 0), ("b", 0.3, 0.5, 0), ("c", 0.8, 1.0, 1), ("d", 1.1, 1.3, 1), ("e", 1.5, 1.7, 0)]
        recording = dummy_recording(0, duration=2.0, with_data=True)
        supervision = SupervisionSegment(
            id="sup-0", recording_id=recording.id, start=0.0, duration=2.0, text="<spk:0> a b <spk:1> c d <spk:0> e"
        )
        custom = {
            "alignments": [{"text": w, "start_time": start, "end_time": end} for w, start, end, _ in words],
            "speaker_ids": [speaker for *_, speaker in words],
        }
        cut = MonoCut(
            id="cut-0",
            start=0.0,
            duration=2.0,
            channel=0,
            recording=recording,
            supervisions=[supervision],
            custom=custom,
        )
        batch = _dataset(tokenizer, speaker_tag_placement=placement, **keys)[CutSet([cut])]
        targets = [t for t in batch.target_tokens[0].tolist() if t != IGNORE_INDEX]
        ids = tokenizer.tokenizer.convert_tokens_to_ids
        tags = [ids(f"<spk:{i}>") for i in range(4)]
        assert targets.count(ids("<|turn_start|>")) == (3 if keys else 0)  # a b | c d | e
        assert [t for t in targets if t in tags] == [tags[0], tags[1], tags[0]]
