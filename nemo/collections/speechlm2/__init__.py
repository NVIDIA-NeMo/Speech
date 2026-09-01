# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import os


# The vLLM plugin only needs ``nemo.collections.speechlm2.vllm``. Importing the
# training API here pulls in optional export/training dependencies (for example
# ONNX and Lightning) that intentionally are not part of a lean serving image.
# The dedicated serving launcher opts into this narrow package initialization.
_VLLM_ONLY = os.getenv("NEMO_SPEECHLM2_VLLM_ONLY") == "1"

if not _VLLM_ONLY:
    from .data import DataModule, DuplexEARTTSDataset, DuplexS2SDataset, DuplexSTTDataset, SALMDataset
    from .models import (
        SALM,
        DuplexEARTTS,
        DuplexS2SModel,
        DuplexS2SSpeechDecoderModel,
        DuplexSTTModel,
        NemotronVoiceChat,
        SALMAutomodel,
        SALMWithAsrDecoder,
    )

__all__ = [
    'DataModule',
    'DuplexS2SDataset',
    'DuplexSTTDataset',
    'DuplexEARTTSDataset',
    'SALMDataset',
    'DuplexEARTTS',
    'DuplexS2SModel',
    'DuplexS2SSpeechDecoderModel',
    'DuplexSTTModel',
    'SALM',
    'SALMAutomodel',
    'SALMWithAsrDecoder',
    'NemotronVoiceChat',
]
