# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo.collections.tts.modules.magpietts_modules import FeatureMasking


def _set_seed():
    torch.manual_seed(42)


class TestFeatureMasking:

    @pytest.mark.unit
    @pytest.mark.parametrize('batch_size', [2, 5])
    @pytest.mark.parametrize('time', [5, 10])
    @pytest.mark.parametrize('hidden_size', [128, 256])
    def test_feature_masking_infer(self, batch_size, time, hidden_size):
        _set_seed()
        feature_masking = FeatureMasking(hidden_size=hidden_size)
        masked_emb = feature_masking.masked_emb[0, 0]
        inputs = torch.randn(size=(batch_size, time, hidden_size), dtype=torch.float32)
        mask = torch.randint(low=0, high=2, size=(batch_size, time), dtype=torch.bool)
        masked_tensor = feature_masking.infer(inputs=inputs, mask=mask)
        for i in range(batch_size):
            for j in range(time):
                output_val = masked_tensor[i, j]
                if mask[i, j]:
                    torch.testing.assert_close(actual=output_val, expected=masked_emb)
                else:
                    torch.testing.assert_close(actual=output_val, expected=inputs[i, j])
