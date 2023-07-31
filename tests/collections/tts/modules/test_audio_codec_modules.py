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

from nemo.collections.tts.modules.audio_codec_modules import (
    Conv1dNorm,
    ConvTranspose1dNorm,
    get_down_sample_padding,
    get_up_sample_padding,
)


class TestAudioCodecModules:
    def setup_class(self):
        self.in_channels = 8
        self.out_channels = 16
        self.batch_size = 2
        self.len1 = 4
        self.len2 = 8
        self.max_len = 10
        self.kernel_size = 3

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_conv1d(self):
        inputs = torch.rand([self.batch_size, self.in_channels, self.max_len])
        lengths = torch.tensor([self.len1, self.len2], dtype=torch.int32)

        conv = Conv1dNorm(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size)
        out = conv(inputs, lengths)

        assert out.shape == (self.batch_size, self.out_channels, self.max_len)
        assert torch.all(out[0, :, : self.len1] != 0.0)
        assert torch.all(out[0, :, self.len1 :] == 0.0)
        assert torch.all(out[1, :, : self.len2] != 0.0)
        assert torch.all(out[1, :, self.len2 :] == 0.0)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_conv1d_downsample(self):
        stride = 2
        out_len = self.max_len // stride
        out_len_1 = self.len1 // stride
        out_len_2 = self.len2 // stride
        inputs = torch.rand([self.batch_size, self.in_channels, self.max_len])
        lengths = torch.tensor([out_len_1, out_len_2], dtype=torch.int32)

        padding = get_down_sample_padding(kernel_size=self.kernel_size, stride=stride)
        conv = Conv1dNorm(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=padding,
        )
        out = conv(inputs, lengths)

        assert out.shape == (self.batch_size, self.out_channels, out_len)
        assert torch.all(out[0, :, :out_len_1] != 0.0)
        assert torch.all(out[0, :, out_len_1:] == 0.0)
        assert torch.all(out[1, :, :out_len_2] != 0.0)
        assert torch.all(out[1, :, out_len_2:] == 0.0)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_conv1d_transpose_upsample(self):
        stride = 2
        out_len = self.max_len * stride
        out_len_1 = self.len1 * stride
        out_len_2 = self.len2 * stride
        inputs = torch.rand([self.batch_size, self.in_channels, self.max_len])
        lengths = torch.tensor([out_len_1, out_len_2], dtype=torch.int32)

        conv = ConvTranspose1dNorm(
            in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=stride
        )
        out = conv(inputs, lengths)

        assert out.shape == (self.batch_size, self.out_channels, out_len)
        assert torch.all(out[0, :, :out_len_1] != 0.0)
        assert torch.all(out[0, :, out_len_1:] == 0.0)
        assert torch.all(out[1, :, :out_len_2] != 0.0)
        assert torch.all(out[1, :, out_len_2:] == 0.0)
