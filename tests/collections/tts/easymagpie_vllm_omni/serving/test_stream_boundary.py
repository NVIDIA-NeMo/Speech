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
"""Regression tests for standard conditional streaming prompt boundaries."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration
from easymagpie_vllm_omni.scheduler import EasyMagpieARAsyncScheduler
from torch import nn
from vllm.v1.request import RequestStatus
from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler
from vllm_omni.engine.serialization import deserialize_additional_information, serialize_additional_information


@pytest.mark.parametrize(("stage", "resumable"), [(0, True), (1, True), (0, False), (1, False)])
def test_initial_stream_boundary_is_only_set_for_stage_zero(
    monkeypatch: pytest.MonkeyPatch,
    stage: int,
    resumable: bool,
) -> None:
    """Record the initial conditioning boundary only for a resumable talker.

    Args:
        monkeypatch: Fixture replacing base scheduler admission.
        stage: Pipeline stage receiving the request.
        resumable: Whether further streaming updates are expected.
    """
    scheduler, request = _stream_request(stage=stage, resumable=resumable)
    parent = Mock()
    monkeypatch.setattr(OmniARAsyncScheduler, "add_request", parent)

    scheduler.add_request(request)

    parent.assert_called_once_with(request)
    info = deserialize_additional_information(request.additional_information)
    assert info == (
        {"speaker_id": "demon", "_stream_prompt_len": 71} if stage == 0 and resumable else {"speaker_id": "demon"}
    )


@pytest.mark.parametrize("resumable", [True, False])
def test_stream_updates_preserve_the_initial_boundary(
    monkeypatch: pytest.MonkeyPatch,
    resumable: bool,
) -> None:
    """Keep the original boundary when continuation or final updates arrive.

    Args:
        monkeypatch: Fixture replacing base scheduler admission.
        resumable: Whether the incoming update continues the text stream.
    """
    scheduler, request = _stream_request(stage=0, resumable=resumable)
    initial_info = serialize_additional_information({"speaker_id": "demon", "_stream_prompt_len": 71})
    session = SimpleNamespace(status=RequestStatus.RUNNING, additional_information=initial_info)
    scheduler.requests[request.request_id] = session
    request.num_prompt_tokens = 73
    update_info = request.additional_information
    parent = Mock()
    monkeypatch.setattr(OmniARAsyncScheduler, "add_request", parent)

    scheduler.add_request(request)

    parent.assert_called_once_with(request)
    assert session.additional_information is initial_info
    assert request.additional_information is update_info


def test_initial_stream_rejects_a_caller_supplied_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject caller metadata that could bypass initial conditioning.

    Args:
        monkeypatch: Fixture replacing base scheduler admission.
    """
    scheduler, request = _stream_request(stage=0, resumable=True)
    request.additional_information = serialize_additional_information({"_stream_prompt_len": 0})
    parent = Mock()
    monkeypatch.setattr(OmniARAsyncScheduler, "add_request", parent)

    with pytest.raises(ValueError, match="Streaming prompt boundary is reserved"):
        scheduler.add_request(request)

    parent.assert_not_called()


@pytest.mark.parametrize(("offset", "span"), [(0, 71), (70, 1), (71, 1), (72, 1)])
def test_admitted_stream_uses_decode_after_initial_prefill(
    monkeypatch: pytest.MonkeyPatch,
    offset: int,
    span: int,
) -> None:
    """Exercise real prefill slicing using metadata from standard admission.

    Args:
        monkeypatch: Fixture replacing base admission and acoustic decode.
        offset: Number of prompt rows already computed by the runtime.
        span: Number of rows in the scheduled chunk.
    """
    scheduler, request = _stream_request(stage=0, resumable=True)
    monkeypatch.setattr(OmniARAsyncScheduler, "add_request", Mock())
    scheduler.add_request(request)
    info = deserialize_additional_information(request.additional_information)
    info.update(
        text_prefill_num=0, _omni_is_prefill=True, _omni_num_computed_tokens=offset, _omni_prompt_len=offset + span
    )
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.has_phoneme = False
    model._dec_text_mask = torch.ones(80, dtype=torch.long)
    model._dec_audio_valid = torch.ones(80, dtype=torch.bool)
    monkeypatch.setattr(model, "_maybe_set_code_sampling_params", Mock())
    monkeypatch.setattr(model, "_build_prefill_embeds", Mock(return_value=torch.zeros(71, 2)))
    monkeypatch.setattr(model, "_build_text_prefill_embeds", Mock(return_value=None))
    monkeypatch.setattr(model, "_is_user_audio_prefill_chunk", Mock(return_value=False))
    decode = Mock(return_value="decoded")
    monkeypatch.setattr(model, "_preprocess_decode", decode)

    result = model.preprocess(torch.zeros(span, dtype=torch.long), None, **info)

    if offset < 71:
        assert result[1].shape == (span, 2)
        decode.assert_not_called()
    else:
        assert result == "decoded"
        decode.assert_called_once()


def _stream_request(
    stage: int,
    resumable: bool,
) -> tuple[EasyMagpieARAsyncScheduler, SimpleNamespace]:
    """Build a scheduler shell and serialized request without starting engines.

    Args:
        stage: Pipeline stage receiving the request.
        resumable: Whether additional streaming updates are expected.

    Returns:
        Scheduler and request with a 71-row initial prompt.
    """
    scheduler = object.__new__(EasyMagpieARAsyncScheduler)
    scheduler.requests = {}
    scheduler.vllm_config = SimpleNamespace(model_config=SimpleNamespace(stage_id=stage))
    request = SimpleNamespace(
        request_id="stream",
        resumable=resumable,
        num_prompt_tokens=71,
        additional_information=serialize_additional_information({"speaker_id": "demon"}),
    )
    return scheduler, request
