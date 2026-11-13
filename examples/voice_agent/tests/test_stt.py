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

import asyncio
from unittest.mock import MagicMock

import pytest

from nemo.agents.voice_agent.pipecat.services.nemo import stt

EOU_MODEL = "nvidia/parakeet_realtime_eou_120m-v1"


def _construct(monkeypatch, **kwargs):
    loader = MagicMock()
    monkeypatch.setattr(stt, "NemoStreamingASRService", loader)

    async def construct():
        return stt.NemoSTTService(device="cpu", **kwargs)

    return asyncio.run(construct()), loader


@pytest.mark.unit
def test_default_stt_model_is_eou_model(monkeypatch):
    service, loader = _construct(monkeypatch)
    loader.assert_called_once_with(EOU_MODEL, [70, 1], device="cpu", decoder_type="rnnt", frame_len_in_secs=0.08)
    assert service._has_turn_taking is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs, expected_model, expected_turn_taking",
    [
        ({"has_turn_taking": False}, EOU_MODEL, False),
        ({"has_turn_taking": True}, EOU_MODEL, True),
        ({"model": "some/other-model"}, "some/other-model", False),
    ],
)
def test_stt_model_and_turn_taking_overrides(monkeypatch, kwargs, expected_model, expected_turn_taking):
    service, loader = _construct(monkeypatch, **kwargs)
    assert loader.call_args.args[0] == expected_model
    assert service._has_turn_taking is expected_turn_taking
