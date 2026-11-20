# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo.collections.audio.parts.submodules.multichannel import (
    ChannelAttentionPool,
    ChannelAugment,
    ChannelAveragePool,
    TransformAttendConcatenate,
    TransformAverageConcatenate,
)


class TestChannelAugment:
    @pytest.mark.unit
    @pytest.mark.parametrize('num_channels', [1, 2, 6])
    def test_channel_selection(self, num_channels):
        """Test getting a fixed number of channels without randomization.
        The first few channels will always be selected.
        """
        num_examples = 100
        batch_size = 4
        num_samples = 100

        uut = ChannelAugment(permute_channels=False, num_channels_min=1, num_channels_max=num_channels)

        for n in range(num_examples):
            input = torch.rand(batch_size, num_channels, num_samples)
            output = uut(input=input)

            num_channels_out = output.size(-2)

            assert torch.allclose(
                output, input[:, :num_channels_out, :]
            ), f'Failed for num_channels_out {num_channels_out}, example {n}'


class TestTAC:
    @pytest.mark.unit
    @pytest.mark.parametrize('num_channels', [1, 2, 6])
    def test_average(self, num_channels):
        """Test transform-average-concatenate."""
        num_examples = 10
        batch_size = 4
        in_features = 128
        out_features = 96
        num_frames = 20

        uut = TransformAverageConcatenate(in_features=in_features, out_features=out_features)

        for n in range(num_examples):
            input = torch.rand(batch_size, num_channels, in_features, num_frames)
            output = uut(input=input)

            # Dimensions must match
            assert output.shape == (
                batch_size,
                num_channels,
                out_features,
                num_frames,
            ), f'Example {n}: output shape {output.shape} not matching the expected ({batch_size}, {num_channels}, {out_features}, {num_frames})'

            # Second half of features must be the same for all channels (concatenated average)
            if num_channels > 1:
                # reference
                avg_ref = output[:, 0, out_features // 2 :, :]
                for m in range(1, num_channels):
                    assert torch.allclose(
                        output[:, m, out_features // 2 :, :], avg_ref
                    ), f'Example {n}: average not matching'

    @pytest.mark.unit
    @pytest.mark.parametrize('num_channels', [1, 2, 6])
    def test_attend(self, num_channels):
        """Test transform-attend-concatenate.
        Second half of features is different across channels, since we're using attention, so
        we check only for shape.
        """
        num_examples = 10
        batch_size = 4
        in_features = 128
        out_features = 96
        num_frames = 20

        uut = TransformAttendConcatenate(in_features=in_features, out_features=out_features)

        for n in range(num_examples):
            input = torch.rand(batch_size, num_channels, in_features, num_frames)
            output = uut(input=input)

            # Dimensions must match
            assert output.shape == (
                batch_size,
                num_channels,
                out_features,
                num_frames,
            ), f'Example {n}: output shape {output.shape} not matching the expected ({batch_size}, {num_channels}, {out_features}, {num_frames})'


class TestChannelPool:
    @pytest.mark.unit
    @pytest.mark.parametrize('num_channels', [1, 2, 6])
    def test_average(self, num_channels):
        """Test average channel pooling."""
        num_examples = 10
        batch_size = 4
        in_features = 128
        num_frames = 20

        uut = ChannelAveragePool()

        for n in range(num_examples):
            input = torch.rand(batch_size, num_channels, in_features, num_frames)
            output = uut(input=input)

            # Dimensions must match
            assert torch.allclose(
                output, torch.mean(input, dim=1)
            ), f'Example {n}: output not matching the expected average'

    @pytest.mark.unit
    @pytest.mark.parametrize('num_channels', [2, 6])
    def test_attention(self, num_channels):
        """Test attention for channel pooling."""
        num_examples = 10
        batch_size = 4
        in_features = 128
        num_frames = 20

        uut = ChannelAttentionPool(in_features=in_features)

        for n in range(num_examples):
            input = torch.rand(batch_size, num_channels, in_features, num_frames)
            output = uut(input=input)

            # Dimensions must match
            assert output.shape == (
                batch_size,
                in_features,
                num_frames,
            ), f'Example {n}: output shape {output.shape} not matching the expected ({batch_size}, {in_features}, {num_frames})'
