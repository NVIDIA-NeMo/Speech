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


import warnings

from nemo.deploy.deploy_base import DeployBase  # noqa: F401
from nemo.deploy.deploy_pytriton import DeployPyTriton  # noqa: F401
from nemo.deploy.triton_deployable import ITritonDeployable  # noqa: F401

warnings.warn(
    "The 'nemo.deploy' is deprecated and will be removed in NeMo FW 25.09 container release. "
    "For evaluation functionality, please use the new Eval repository: https://github.com/NVIDIA-NeMo/Export-Deploy",
    DeprecationWarning,
    stacklevel=2,
)
