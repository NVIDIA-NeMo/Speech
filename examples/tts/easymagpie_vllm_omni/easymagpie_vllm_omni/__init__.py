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
"""EasyMagpieTTS model definition for vLLM-Omni.

This package provides an inference-only re-implementation of EasyMagpieTTS
(decoder-only, Nemotron-H hybrid-Mamba backbone + autoregressive local
transformer over the stacked audio codebooks) that plugs into the vLLM-Omni
serving stack via the standard ``preprocess`` / ``postprocess`` /
``make_omni_output`` hooks.

The companion ``vllm_plugin_easymagpie_omni`` package registers the model with
vLLM's ``ModelRegistry`` through the ``vllm.general_plugins`` entry point.
"""

from easymagpie_vllm_omni.config import EASYMAGPIE_SMALLMAMBA, EasyMagpieOmniArch

__all__ = ["EASYMAGPIE_SMALLMAMBA", "EasyMagpieOmniArch"]
