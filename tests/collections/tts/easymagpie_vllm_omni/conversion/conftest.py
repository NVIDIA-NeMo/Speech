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
"""Path setup for Speech-side EasyMagpie checkpoint conversion tests."""
from __future__ import annotations

import sys
from pathlib import Path

EASYMAGPIE_ROOT = Path(__file__).resolve().parents[5] / "tools" / "easymagpie_vllm_omni"
SCRIPTS_DIR = EASYMAGPIE_ROOT / "scripts"

# Conversion uses Speech and pure PyTorch helpers, never the vLLM runtime.
sys.path.insert(0, str(EASYMAGPIE_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))
