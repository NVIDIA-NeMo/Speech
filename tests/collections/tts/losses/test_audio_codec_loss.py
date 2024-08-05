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
from torchmetrics import ScaleInvariantSignalDistortionRatio

from nemo.collections.common.parts.utils import mask_sequence_tensor
from nemo.collections.tts.losses.audio_codec_loss import MaskedMAELoss, MaskedMSELoss, SISDRLoss


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

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_si_sdr_loss(self):
        loss_fn = SISDRLoss()
        sdr_fn = ScaleInvariantSignalDistortionRatio(zero_mean=True)

        num_samples = 1000
        torch.manual_seed(100)
        target = torch.rand([1, num_samples])
        predicted = torch.rand([1, num_samples])
        target_len = torch.tensor([num_samples, num_samples])

        torch_si_sdr = sdr_fn(preds=predicted, target=target)
        loss = loss_fn(audio_real=target, audio_gen=predicted, audio_len=target_len)
        si_sdr = -loss

        torch.testing.assert_close(actual=si_sdr, expected=torch_si_sdr)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_si_sdr_loss_batch(self):
        loss_fn = SISDRLoss()
        si_sdr_fn = ScaleInvariantSignalDistortionRatio(zero_mean=True)

        batch_size = 3
        num_samples = 1000
        torch.manual_seed(100)
        target = torch.rand([batch_size, num_samples])
        predicted = torch.rand([batch_size, num_samples])

        target_len = torch.tensor([500, 250, 900])
        target = mask_sequence_tensor(target, lengths=target_len)
        predicted = mask_sequence_tensor(predicted, lengths=target_len)

        torch_si_sdr = 0.0
        for i in range(batch_size):
            torch_si_sdr += si_sdr_fn(preds=predicted[i, : target_len[i]], target=target[i, : target_len[i]])
        torch_si_sdr /= batch_size

        loss = loss_fn(audio_real=target, audio_gen=predicted, audio_len=target_len)
        si_sdr = -loss

        torch.testing.assert_close(actual=si_sdr, expected=torch_si_sdr)
