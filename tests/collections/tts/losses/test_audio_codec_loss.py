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

from nemo.collections.tts.losses.audio_codec_loss import MaskedMAELoss, MaskedMSELoss


class TestAudioCodecLoss:
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_masked_loss_l1(self):
        loss_fn = MaskedMAELoss()
        target = torch.tensor([[[1.0], [2.0], [0.0]], [[3.0], [0.0], [0.0]]]).transpose(1, 2)
        predicted = torch.tensor([[[0.5], [1.0], [0.0]], [[4.5], [0.0], [0.0]]]).transpose(1, 2)
        target_len = torch.tensor([2, 1])

        loss = loss_fn(predicted=predicted, target=target, target_len=target_len)

        assert loss == 1.125

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_masked_loss_l2(self):
        loss_fn = MaskedMSELoss()
        target = torch.tensor([[[1.0], [2.0], [4.0]], [[3.0], [0.0], [0.0]]]).transpose(1, 2)
        predicted = torch.tensor([[[0.5], [1.0], [4.0]], [[4.5], [0.0], [0.0]]]).transpose(1, 2)
        target_len = torch.tensor([3, 1])

        loss = loss_fn(predicted=predicted, target=target, target_len=target_len)

        assert loss == (4 / 3)
