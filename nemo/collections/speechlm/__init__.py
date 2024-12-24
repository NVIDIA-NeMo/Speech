# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.speechlm.models import HFAutoModelForSpeechSeq2Seq
from nemo.utils import logging

__all__ = [
    "HFAutoModelForSpeechSeq2Seq",
]

try:
    import nemo_run as run

    from nemo.collections.llm.recipes import adam
    from nemo.collections.speechlm.api import finetune, generate, pretrain, train, validate

    __all__.extend(
        [
            "train",
            "pretrain",
            "validate",
            "finetune",
            "generate",
        ]
    )
except ImportError as error:
    logging.warning(f"Failed to import nemo.collections.speechlm.[api, recipes]: {error}")
