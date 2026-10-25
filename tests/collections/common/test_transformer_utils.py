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
import torch.nn as nn

from nemo.collections.common.parts.transformer_utils import mask_padded_tokens


class TestMaskPaddedTokens:
    """Test suite for mask_padded_tokens function."""

    @pytest.mark.unit
    def test_mask_padded_tokens_jit_script_compilation(self):
        """Test that mask_padded_tokens works correctly with TorchScript compilation.

        This test ensures type hints are properly defined.
        """

        class SimpleModule(nn.Module):
            """Module wrapper for testing mask_padded_tokens with TorchScript."""

            def __init__(self, pad_id: int = 0):
                super().__init__()
                self.pad = pad_id

            def forward(self, tokens: torch.Tensor) -> torch.Tensor:
                mask = mask_padded_tokens(tokens, self.pad)
                return mask.float()

        module = SimpleModule(pad_id=0)
        scripted_module = torch.jit.script(module)

        assert isinstance(scripted_module, torch.jit.ScriptModule), "Failed to compile with TorchScript"
