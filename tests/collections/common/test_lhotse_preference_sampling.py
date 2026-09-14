# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
from lhotse import CutSet, SupervisionSegment
from lhotse.testing.dummies import dummy_cut
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse import cutset as cutset_module
from nemo.collections.common.data.lhotse.cutset import sample_preference_to_conversation
from nemo.collections.common.data.lhotse.text_adapters import AudioTurn, TextTurn


PREFERENCE_INSTRUCTIONS = [
    {
        "prompt": "Translate the audio.",
        "target": "Bonjour le monde.",
        "tags": {"type": "translation", "target_lang": "fr"},
    },
    {
        "prompt": "Summarize the audio.",
        "target": "A short summary.",
        "tags": {"type": "summary", "target_lang": "en"},
    },
]


def _make_cut(unique_id=0, custom=None):
    cut = dummy_cut(
        unique_id,
        duration=1.0,
        supervisions=[
            SupervisionSegment(
                id=f"sup-{unique_id}",
                recording_id=f"dummy-recording-{unique_id:04d}",
                start=0.0,
                duration=1.0,
                text="Original transcript.",
                language="en",
            )
        ],
    )
    cut.custom = custom if custom is not None else {"preference_instructions": PREFERENCE_INSTRUCTIONS}
    return cut


@pytest.mark.unit
def test_preference_sampling_is_deterministic_and_builds_matching_turns():
    weights = {"translation": 1.0, "summary": 3.0}

    first = sample_preference_to_conversation(
        _make_cut(unique_id=7),
        audio_locator_tag="<audio>",
        token_equivalent_duration=0.25,
        weights=weights,
        seed=123,
    )
    second = sample_preference_to_conversation(
        _make_cut(unique_id=7),
        audio_locator_tag="<audio>",
        token_equivalent_duration=0.25,
        weights=weights,
        seed=123,
    )

    assert first.custom["_pref_type"] == second.custom["_pref_type"]
    assert [turn.to_dict() for turn in first.turns] == [turn.to_dict() for turn in second.turns]

    selected = {instruction["tags"]["type"]: instruction for instruction in PREFERENCE_INSTRUCTIONS}[
        first.custom["_pref_type"]
    ]
    assert [type(turn) for turn in first.turns] == [TextTurn, AudioTurn, TextTurn]
    assert [turn.role for turn in first.turns] == ["user", "user", "assistant"]
    assert first.turns[0].value == selected["prompt"]
    assert first.turns[1].audio_locator_tag == "<audio>"
    assert first.turns[1].text == selected["target"]
    assert first.turns[2].value == selected["target"]
    assert first.token_equivalent_duration == 0.25
    assert first.turns[1].cut.supervisions[0].text == selected["target"]
    assert first.turns[1].cut.supervisions[0].language == selected["tags"]["target_lang"]


@pytest.mark.unit
def test_preference_sampling_excludes_zero_weight_and_empty_targets():
    cut = _make_cut(
        custom={
            "preference_instructions": PREFERENCE_INSTRUCTIONS
            + [
                {
                    "prompt": "Invalid instruction.",
                    "target": "",
                    "tags": {"type": "invalid", "target_lang": "en"},
                }
            ]
        }
    )

    conversation = sample_preference_to_conversation(
        cut,
        audio_locator_tag="<audio>",
        token_equivalent_duration=0.25,
        weights=OmegaConf.create({"translation": 0.0, "summary": 1.0, "invalid": 100.0}),
    )

    assert conversation.custom["_pref_type"] == "summary"
    assert conversation.turns[-1].value == "A short summary."


@pytest.mark.unit
def test_preference_sampling_falls_back_to_configured_transcript():
    cut = _make_cut(
        custom={
            "normalized_text": "Fallback transcript.",
            "source_lang": "de",
        }
    )

    conversation = sample_preference_to_conversation(
        cut,
        audio_locator_tag="<audio>",
        token_equivalent_duration=0.25,
        fallback_text_field="normalized_text",
    )

    assert conversation.custom["_pref_type"] == "fallback"
    assert [type(turn) for turn in conversation.turns] == [AudioTurn, TextTurn]
    assert conversation.turns[0].text == "Fallback transcript."
    assert conversation.turns[1].value == "Fallback transcript."
    assert conversation.turns[0].cut.supervisions[0].text == "Fallback transcript."
    assert conversation.turns[0].cut.supervisions[0].language == "de"


@pytest.mark.unit
def test_lhotse_as_conversation_routes_preference_sampling_config(monkeypatch):
    cut = _make_cut(unique_id=11)
    monkeypatch.setattr(
        cutset_module,
        "read_cutset_from_config",
        lambda config: (CutSet.from_cuts([cut]), False),
    )
    config = OmegaConf.create(
        {
            "audio_locator_tag": "<audio>",
            "token_equivalent_duration": 0.5,
            "preference_sampling": {
                "weights": {"translation": 1.0, "summary": 0.0},
                "seed": 17,
                "fallback_text_field": "normalized_text",
            },
        }
    )

    cuts, is_tarred = cutset_module.read_lhotse_as_conversation(config)
    (conversation,) = list(cuts)

    assert is_tarred is False
    assert conversation.custom["_pref_type"] == "translation"
    assert conversation.turns[0].value == "Translate the audio."
    assert conversation.turns[-1].value == "Bonjour le monde."
    assert conversation.token_equivalent_duration == 0.5
