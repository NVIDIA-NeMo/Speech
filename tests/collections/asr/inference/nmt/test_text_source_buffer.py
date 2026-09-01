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

import pytest

from nemo.collections.asr.inference.nmt.text_source_buffer import TextMTSourceBuffer, last_text_boundary


@pytest.mark.parametrize("text", ["Dr. Smith continues", "value 3.14 remains", "the U.S. economy"])
def test_non_terminal_period_does_not_close_source(text):
    assert last_text_boundary(text) == -1


def test_stable_punctuation_closes_source_and_retains_suffix():
    buffer = TextMTSourceBuffer()
    decision = buffer.update("The first sentence. The next", acoustic_eou=False, elapsed_ms=480)

    assert decision.is_final
    assert decision.source == "The first sentence."
    assert decision.retained_suffix == "The next"
    assert decision.boundary_reason == "punctuation"


def test_acoustic_eou_does_not_finalize_mt_source():
    buffer = TextMTSourceBuffer()
    first = buffer.update("A short fragment", acoustic_eou=True, elapsed_ms=480)
    second = buffer.update("continues here", acoustic_eou=False, elapsed_ms=480)

    assert not first.is_final
    assert not second.is_final
    assert second.source == "A short fragment continues here"


def test_cumulative_asr_updates_append_only_unseen_text():
    buffer = TextMTSourceBuffer()
    buffer.update("the language model", acoustic_eou=False, elapsed_ms=480)
    decision = buffer.update("the language models improve", acoustic_eou=False, elapsed_ms=480)

    assert decision.source == "the language models improve"


def test_earlier_punctuation_revision_does_not_duplicate_source():
    buffer = TextMTSourceBuffer()
    first = buffer.update("Earlier text. Active source", acoustic_eou=False, elapsed_ms=480)
    second = buffer.update("Earlier text Active source", acoustic_eou=True, elapsed_ms=480)

    assert first.source == "Earlier text."
    assert first.retained_suffix == "Active source"
    assert second.source == "Active source"
    assert not second.is_final


def test_deferred_boundary_is_reassembled_for_retry():
    buffer = TextMTSourceBuffer()
    decision = buffer.update("Closed unit. New suffix", acoustic_eou=False, elapsed_ms=480)
    buffer.defer_boundary(decision.source, decision.retained_suffix)

    assert buffer.active_source == "Closed unit. New suffix"


def test_source_unit_limit_is_a_safety_boundary():
    buffer = TextMTSourceBuffer(max_source_units=3)
    decision = buffer.update("one two three four", acoustic_eou=False, elapsed_ms=480)

    assert decision.is_final
    assert decision.source == "one two three"
    assert decision.retained_suffix == "four"
    assert decision.boundary_reason == "max_source_units"


def test_duration_limit_is_a_safety_boundary():
    buffer = TextMTSourceBuffer(max_duration_ms=900)
    buffer.update("unfinished", acoustic_eou=False, elapsed_ms=480)
    decision = buffer.update("unfinished source", acoustic_eou=False, elapsed_ms=480)

    assert decision.is_final
    assert decision.source == "unfinished source"
    assert decision.boundary_reason == "max_duration"


def test_stream_end_flushes_retained_source_once():
    buffer = TextMTSourceBuffer()
    buffer.update("retained ending", acoustic_eou=False, elapsed_ms=480)

    decision = buffer.flush()
    assert decision.is_final
    assert decision.source == "retained ending"
    assert decision.boundary_reason == "stream_end"
    assert not buffer.flush().is_final
