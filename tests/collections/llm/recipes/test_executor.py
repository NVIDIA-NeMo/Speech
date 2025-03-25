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
from unittest.mock import patch

import pytest

from nemo.collections.llm.recipes.run.executor import torchrun


@patch('torch.cuda.is_available')
@patch('torch.cuda.device_count')
def test_torchrun_with_explicit_devices(mock_device_count, mock_cuda_available):
    """Test torchrun factory with automatic device detection"""
    # Mock CUDA being available with 2 devices
    mock_cuda_available.return_value = True
    mock_device_count.return_value = 2

    config = torchrun(devices=2)

    assert config.ntasks_per_node == 2
    assert config.launcher == 'torchrun'
    assert config.env_vars == {'TORCH_NCCL_AVOID_RECORD_STREAMS': '1'}


@patch('torch.cuda.is_available')
def test_torchrun_raises_error_without_cuda(mock_cuda_available):
    """Test torchrun factory raises error when CUDA is not available and devices not specified"""
    mock_cuda_available.return_value = False

    with pytest.raises(RuntimeError) as exc_info:
        torchrun(devices=None)

    assert "Cannot infer the 'ntasks_per_node' parameter" in str(exc_info.value)
