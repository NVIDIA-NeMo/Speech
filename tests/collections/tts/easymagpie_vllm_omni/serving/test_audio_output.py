# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from __future__ import annotations

from collections import UserDict
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from easymagpie_vllm_omni.audio_output import extract_audio_from_stage_output


def test_extract_audio_unwraps_list_and_mapping_payload():
    payload = UserDict(
        {
            "audio": torch.tensor([0.25, -0.5], dtype=torch.float32),
            "sr": torch.tensor(22050, dtype=torch.int32),
        }
    )
    completion = SimpleNamespace(multimodal_output=payload)
    request_output = SimpleNamespace(outputs=[completion])
    stage_output = SimpleNamespace(request_output=request_output)

    waveform, sample_rate = extract_audio_from_stage_output([stage_output])

    torch.testing.assert_close(torch.from_numpy(waveform), payload["audio"])
    assert sample_rate == 22050
