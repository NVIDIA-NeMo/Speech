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

from nemo.collections.asr.inference.nmt.prompts import RivaV2TranslatorPromptTemplate


def test_riva_prompt_preserves_aligned_context_as_separate_turns():
    messages = RivaV2TranslatorPromptTemplate.messages(
        "English",
        "German",
        "Current source",
        "Aktuelle",
        ["First source", "Second source"],
        ["Erste Ausgabe", "Zweite Ausgabe"],
    )

    assert messages == [
        {"role": "system", "content": "en-de"},
        {"role": "user", "content": "First source"},
        {"role": "assistant", "content": "Erste Ausgabe"},
        {"role": "user", "content": "Second source"},
        {"role": "assistant", "content": "Zweite Ausgabe"},
        {"role": "user", "content": "Current source"},
        {"role": "assistant", "content": "Aktuelle"},
    ]


def test_riva_prompt_rejects_unaligned_context_turns():
    try:
        RivaV2TranslatorPromptTemplate.messages(
            "English",
            "German",
            "Current source",
            "",
            ["First source", "Second source"],
            ["Erste Ausgabe"],
        )
    except ValueError as error:
        assert "same number of turns" in str(error)
    else:
        raise AssertionError("Expected unaligned Riva context to be rejected")


def test_riva_prompt_expands_simulstream_language_codes():
    messages = RivaV2TranslatorPromptTemplate.messages("en", "zh", "Current source", "")

    assert messages[0] == {"role": "system", "content": "en-zh"}


def test_riva_prompt_expands_extended_model_card_language_names():
    messages = RivaV2TranslatorPromptTemplate.messages(
        "English",
        "Brazilian Portuguese",
        "Current source",
        "",
    )

    assert messages[0] == {"role": "system", "content": "en-pt-br"}
