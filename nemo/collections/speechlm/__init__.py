# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.speechlm.recipes.pipeline import speech_to_text_llm_train, speech_to_text_llm_validate
from nemo.utils import logging

__all__ = [
    "speech_to_text_llm_train",
    "speech_to_text_llm_validate",
]

try:
    import nemo_run as run  # noqa: F401

    from nemo.collections.speechlm.api import finetune, generate, pretrain, train, validate  # noqa: F401
    from nemo.collections.speechlm.recipes.optim import adam  # noqa

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
