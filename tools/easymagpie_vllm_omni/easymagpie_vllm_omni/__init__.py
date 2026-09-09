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
"""EasyMagpieTTS model and pipeline definitions for vLLM-Omni."""

from easymagpie_vllm_omni.config import EASYMAGPIE_SMALLMAMBA, EasyMagpieOmniArch

__all__ = ["EASYMAGPIE_SMALLMAMBA", "EasyMagpieOmniArch"]


def __getattr__(name: str):
    # Lazily expose pipeline configs without importing vllm_omni (a heavy
    # dependency) at package import time.
    if name == "EASYMAGPIE_PIPELINE":
        from easymagpie_vllm_omni.pipeline import EASYMAGPIE_PIPELINE

        return EASYMAGPIE_PIPELINE
    if name == "EASYMAGPIE_LM_PIPELINE":
        from easymagpie_vllm_omni.pipeline import EASYMAGPIE_LM_PIPELINE

        return EASYMAGPIE_LM_PIPELINE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
