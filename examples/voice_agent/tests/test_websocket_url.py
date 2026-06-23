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

import sys
from pathlib import Path

import pytest

voice_agent_server_path = Path(__file__).resolve().parents[1] / "server"
sys.path.insert(0, str(voice_agent_server_path))

from websocket_url import build_websocket_url


@pytest.mark.unit
def test_build_websocket_url_uses_configured_host():
    assert build_websocket_url("voice-agent.example", 8765) == "ws://voice-agent.example:8765"


@pytest.mark.unit
def test_build_websocket_url_does_not_accept_request_host():
    forged_request_host = "evil.example"

    ws_url = build_websocket_url("voice-agent.example", 8765)

    assert forged_request_host not in ws_url
    assert ws_url == "ws://voice-agent.example:8765"


@pytest.mark.unit
@pytest.mark.parametrize("scheme", ["http", "https", ""])
def test_build_websocket_url_rejects_non_websocket_schemes(scheme):
    with pytest.raises(ValueError):
        build_websocket_url("voice-agent.example", 8765, scheme=scheme)
