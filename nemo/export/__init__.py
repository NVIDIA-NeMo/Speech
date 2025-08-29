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

# WAR for trtllm and lightning conflict
try:
    from nemo.lightning import io

    __all__ = ["io"]
except (ImportError, ModuleNotFoundError):
    pass

import warnings

warnings.warn(
    "The 'nemo.export' is deprecated and will be removed in NeMo FW 25.09 container release. "
    "For evaluation functionality, please use the new Eval repository: https://github.com/NVIDIA-NeMo/Export-Deploy",
    DeprecationWarning,
    stacklevel=2,
)
