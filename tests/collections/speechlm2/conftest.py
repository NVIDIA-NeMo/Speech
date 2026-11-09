# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import pytest
import torch


@pytest.fixture(scope="module")
def cuda_default_device():
    """
    Use CUDA as torch's default device for one test module, then restore the previous default.

    ``torch.set_default_device`` is process-wide. Calling it at import time leaks into every other module collected
    in the same pytest session (pytest imports all modules before running any test), which makes device-agnostic tests
    elsewhere run on CUDA and fail with device mismatches. Opt a module in with
    ``pytestmark = pytest.mark.usefixtures("cuda_default_device")``, and request this fixture explicitly from
    module-scoped fixtures that build models or tensors, so that they are created on CUDA too.
    """
    if not torch.cuda.is_available():
        yield
        return
    previous = torch.get_default_device()
    torch.set_default_device("cuda")
    try:
        yield
    finally:
        torch.set_default_device(previous)
