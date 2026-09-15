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

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.unit
def test_voice_agent_import_warns_once():
    result = subprocess.run(
        [
            sys.executable,
            "-E",
            "-c",
            "import nemo.agents.voice_agent; import nemo.agents.voice_agent",
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stderr.count("FutureWarning:") == 1
    assert "nemo.agents.voice_agent is deprecated" in result.stderr
    assert "https://github.com/NVIDIA-NeMo/labs-Voice-Agent" in result.stderr
    assert "<string>:1: FutureWarning:" in result.stderr


@pytest.mark.unit
@pytest.mark.parametrize(
    "args",
    [
        ["-c", "import nemo.agents"],
        ["-W", "ignore::FutureWarning", "-c", "import nemo.agents.voice_agent"],
    ],
)
def test_voice_agent_warning_is_scoped_and_filterable(args):
    result = subprocess.run(
        [sys.executable, "-E", *args],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stderr == ""
