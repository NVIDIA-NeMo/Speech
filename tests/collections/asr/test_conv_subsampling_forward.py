# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Regression tests for ConvSubsampling forward paths (issue #16223).

These tests verify that the three code paths broken by the
MaskedConvSequential refactor (#13827) work on CPU without downloads.
"""

import pytest
import torch

from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling


BATCH, FEAT_IN, TIME = 2, 80, 100
FEAT_OUT, CONV_CHANNELS = 128, 64


def _make_input():
    x = torch.randn(BATCH, TIME, FEAT_IN)
    lengths = torch.tensor([TIME, TIME - 10])
    return x, lengths


class TestConvSubsamplingForwardPaths:
    """Regression tests for #16223: broken forward paths after MaskedConvSequential refactor."""

    @pytest.mark.unit
    @pytest.mark.parametrize("subsampling", ["striding_conv1d", "dw_striding_conv1d"])
    def test_conv1d_forward(self, subsampling):
        """Conv1d subsampling types must not raise TypeError from missing lengths."""
        module = ConvSubsampling(subsampling, 4, FEAT_IN, FEAT_OUT, CONV_CHANNELS)
        x, lengths = _make_input()
        out, out_lengths = module(x, lengths)
        assert out.shape[0] == BATCH
        assert out.shape[2] == FEAT_OUT
        assert (out_lengths <= TIME).all()

    @pytest.mark.unit
    def test_vggnet_forward(self):
        """VGGNet subsampling must handle MaxPool2d int attributes."""
        module = ConvSubsampling("vggnet", 4, FEAT_IN, FEAT_OUT, CONV_CHANNELS)
        x, lengths = _make_input()
        out, out_lengths = module(x, lengths)
        assert out.shape[0] == BATCH
        assert out.shape[2] == FEAT_OUT
        assert (out_lengths <= TIME).all()

    @pytest.mark.unit
    @pytest.mark.parametrize("subsampling", ["striding", "dw_striding"])
    def test_chunking_minus_one(self, subsampling):
        """chunking_factor=-1 must pass lengths to MaskedConvSequential."""
        module = ConvSubsampling(
            subsampling, 4, FEAT_IN, FEAT_OUT, CONV_CHANNELS, subsampling_conv_chunking_factor=-1
        )
        x, lengths = _make_input()
        out, out_lengths = module(x, lengths)
        assert out.shape[0] == BATCH
        assert out.shape[2] == FEAT_OUT
        assert (out_lengths <= TIME).all()

    @pytest.mark.unit
    @pytest.mark.parametrize("subsampling", ["striding", "dw_striding"])
    def test_default_chunking_unchanged(self, subsampling):
        """Default chunking_factor=1 still works (regression guard)."""
        module = ConvSubsampling(subsampling, 4, FEAT_IN, FEAT_OUT, CONV_CHANNELS)
        x, lengths = _make_input()
        out, out_lengths = module(x, lengths)
        assert out.shape[0] == BATCH
        assert out.shape[2] == FEAT_OUT
        assert (out_lengths <= TIME).all()
