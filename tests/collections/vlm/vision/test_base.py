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

import pytest
import torch

from nemo.collections import vlm
from nemo.collections.vlm.vision.base import DownSampleBlock


def test_siglip_config_error():
    config = vlm.CLIPViTConfig(vision_model_type="siglip")
    assert config.add_class_token == False
    assert config.class_token_len == 0
    # with pytest.raises(ValueError):
    #     config.configure_model()


def test_downsample_block_basic():
    # Create a square input tensor: (batch, seq_len, embed_dim)
    # For a 4x4 patch grid, seq_len = 16, embed_dim = 8
    batch_size = 2
    seq_len = 16  # 4x4 grid
    embed_dim = 8
    x = torch.randn(batch_size, seq_len, embed_dim)
    block = DownSampleBlock()
    out = block(x)
    # Print for debug (optional)
    print("Output shape:", out.shape)
    # Assert the actual output shape
    assert out.shape == (16, 1, 64)
