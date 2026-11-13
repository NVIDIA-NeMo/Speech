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
import subprocess
import sys

import pytest


@pytest.mark.unit
def test_tts_collection_imports_without_speechlm2_dependencies():
    # The `asr` and `tts` extras do not install `peft`, and importing any `nemo.collections.speechlm2` module
    # imports all SpeechLM2 models, which need it. Hide `peft` as such an install would.
    script = (
        "import sys; sys.modules['peft'] = None; import nemo.collections.tts; "
        "loaded = [n for n in sys.modules if n.startswith('nemo.collections.speechlm2')]; assert not loaded, loaded"
    )
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
