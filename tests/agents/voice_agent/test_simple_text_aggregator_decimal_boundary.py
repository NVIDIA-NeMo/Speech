# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import importlib
import sys
import types

import pytest

# `nemo.agents.voice_agent` depends on the optional `pipecat-ai`/`loguru` stack
# (see examples/voice_agent/environment.yaml), which isn't part of this repo's
# test environment. `simple_text_aggregator.py`'s module-level `find_last_period_index`
# / `has_partial_decimal` helpers under test have no real dependency on either
# package, so stub just enough of both to import the real module and exercise
# its actual functions directly, rather than re-implementing or AST-parsing them.


def _install_stub_modules():
    if "loguru" not in sys.modules:
        loguru_stub = types.ModuleType("loguru")
        loguru_stub.logger = types.SimpleNamespace(
            debug=lambda *a, **k: None,
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
        )
        sys.modules["loguru"] = loguru_stub

    if "pipecat" not in sys.modules:
        sys.modules["pipecat"] = types.ModuleType("pipecat")
    if "pipecat.utils" not in sys.modules:
        sys.modules["pipecat.utils"] = types.ModuleType("pipecat.utils")
    if "pipecat.utils.string" not in sys.modules:
        mod = types.ModuleType("pipecat.utils.string")
        mod.match_endofsentence = lambda text: None
        sys.modules["pipecat.utils.string"] = mod
    if "pipecat.utils.text" not in sys.modules:
        sys.modules["pipecat.utils.text"] = types.ModuleType("pipecat.utils.text")
    if "pipecat.utils.text.base_text_aggregator" not in sys.modules:
        mod = types.ModuleType("pipecat.utils.text.base_text_aggregator")

        class Aggregation:
            def __init__(self, text, type):
                self.text = text
                self.type = type

        class AggregationType:
            SENTENCE = "sentence"

        mod.Aggregation = Aggregation
        mod.AggregationType = AggregationType
        sys.modules["pipecat.utils.text.base_text_aggregator"] = mod
    if "pipecat.utils.text.simple_text_aggregator" not in sys.modules:
        mod = types.ModuleType("pipecat.utils.text.simple_text_aggregator")

        class SimpleTextAggregator:
            def __init__(self, **kwargs):
                self._text = ""

        mod.SimpleTextAggregator = SimpleTextAggregator
        sys.modules["pipecat.utils.text.simple_text_aggregator"] = mod


_install_stub_modules()
find_last_period_index = importlib.import_module(
    "nemo.agents.voice_agent.pipecat.utils.text.simple_text_aggregator"
).find_last_period_index


@pytest.mark.parametrize(
    "text",
    [
        "It costs $3.14.",
        "Your total is $19.99.",
        "The average temperature is 72.5.",
    ],
)
def test_sentence_ending_in_decimal_number_is_recognized_as_complete(text):
    """`find_last_period_index`'s own `has_partial_decimal` docstring gives
    "It costs $3.14." as an example of a sentence that is "clearly ... complete"
    and must NOT be treated as a partial decimal. But the decimal/bullet/
    abbreviation heuristics only run when the buffered text contains exactly one
    period (`if num_periods == 1:`); a second period anywhere in the buffer (here,
    the decimal point itself) skips them entirely, falls through to the generic
    "digit before the period -> partial decimal" rule, and misclassifies the
    sentence-ending period as part of an in-progress decimal -- so the aggregator
    never flushes this sentence to TTS at its natural boundary.
    """
    idx = find_last_period_index(text)
    assert idx == len(text) - 1, (
        f"find_last_period_index({text!r}) == {idx}, expected {len(text) - 1} "
        "(the trailing period) -- a sentence ending in a decimal number must be "
        "recognized as complete."
    )


def test_first_of_two_sentences_ending_in_decimal_is_split_off():
    text = "That will be 3.14. Thank you!"
    idx = find_last_period_index(text)
    expected = text.index(". Thank you!")
    assert idx == expected, (
        f"find_last_period_index({text!r}) == {idx}, expected {expected} -- the "
        "first sentence should be split off at its own decimal-terminated period, "
        "not merged with the next sentence."
    )


@pytest.mark.parametrize(
    "text",
    [
        "It is 3.",  # a bare trailing digit-period is genuinely ambiguous
        "It costs $3.",  # ("3." could still grow into "3.5") -- must stay -1
        "First, let's begin. The meeting is at 3.",
    ],
)
def test_ambiguous_trailing_digit_period_is_still_rejected(text):
    """The fix must not treat every digit-before-period as complete -- an
    in-progress decimal (or a bare trailing number that could still grow one)
    must still return -1, exactly as before."""
    assert find_last_period_index(text) == -1
