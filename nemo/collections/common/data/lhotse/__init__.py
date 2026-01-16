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

import inspect

from lhotse.dataset.sampling.base import Sampler

from nemo.collections.common.data.lhotse.cutset import read_cutset_from_config
from nemo.collections.common.data.lhotse.dataloader import (
    LhotseDataLoadingConfig,
    get_lhotse_dataloader_from_config,
    get_lhotse_sampler_from_config,
)
from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoIterator, LazyNeMoTarredIterator
from nemo.collections.common.data.lhotse.text_adapters import (
    NeMoMultimodalConversation,
    NeMoSFTExample,
    SourceTargetTextExample,
    TextExample,
)
