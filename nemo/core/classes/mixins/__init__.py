# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo.core.classes.mixins.access_mixins import AccessMixin, set_access_cfg
from nemo.core.classes.mixins.adapter_mixin_strategies import (
    ResidualAddAdapterStrategy,
    ResidualAddAdapterStrategyConfig,
    ReturnResultAdapterStrategy,
    ReturnResultAdapterStrategyConfig,
)
from nemo.core.classes.mixins.adapter_mixins import (
    AdapterModelPTMixin,
    AdapterModuleMixin,
    get_registered_adapter,
    register_adapter,
)
from nemo.core.classes.mixins.hf_io_mixin import HuggingFaceFileIO
