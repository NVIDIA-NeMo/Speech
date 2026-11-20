# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Native Torch implementations of the EasyMagpie spectral codec."""

from easymagpie_vllm_omni.codec.config import EasyMagpieCodecConfig
from easymagpie_vllm_omni.codec.encoder import EasyMagpieCodecEncoder, EasyMagpieCodecEncoderOutput
from easymagpie_vllm_omni.codec.encoder_config import EasyMagpieCodecEncoderConfig
from easymagpie_vllm_omni.codec.reference_speaker_encoder import EasyMagpieReferenceSpeakerEncoder

__all__ = [
    "EasyMagpieCodecConfig",
    "EasyMagpieCodecEncoder",
    "EasyMagpieCodecEncoderConfig",
    "EasyMagpieCodecEncoderOutput",
    "EasyMagpieReferenceSpeakerEncoder",
]
