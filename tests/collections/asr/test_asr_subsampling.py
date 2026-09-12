# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling, calc_length


class TestASRSubsamplingConvChunking:
    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_forward(self):
        asr_model = ASRModel.from_pretrained("stt_en_fastconformer_ctc_large")
        asr_model = asr_model.eval()
        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0

        len = 512

        input_signal_batch1 = torch.randn(size=(1, len), device=asr_model.device)
        length_batch1 = torch.randint(low=321, high=500, size=[1], device=asr_model.device)

        input_signal_batch4 = torch.randn(size=(4, len), device=asr_model.device)
        length_batch4 = torch.randint(low=321, high=500, size=[4], device=asr_model.device)

        with torch.inference_mode():
            # regular inference
            logprobs_batch1_nosplit, _, _ = asr_model.forward(
                input_signal=input_signal_batch1, input_signal_length=length_batch1
            )
            logprobs_batch4_nosplit, _, _ = asr_model.forward(
                input_signal=input_signal_batch4, input_signal_length=length_batch4
            )

            # force chunking to 2
            asr_model.change_subsampling_conv_chunking_factor(subsampling_conv_chunking_factor=2)

            # chunked inference by channels as batch is 1
            logprobs_batch1_split, _, _ = asr_model.forward(
                input_signal=input_signal_batch1, input_signal_length=length_batch1
            )
            # chunked inference by batch as it is 4 [> 1]
            logprobs_batch4_split, _, _ = asr_model.forward(
                input_signal=input_signal_batch4, input_signal_length=length_batch4
            )

        diff = torch.mean(torch.abs(logprobs_batch1_split - logprobs_batch1_nosplit))
        assert diff <= 0.2
        diff = torch.mean(torch.abs(logprobs_batch4_split - logprobs_batch4_nosplit))
        assert diff <= 0.2


class TestConvSubsamplingForwardPaths:
    """CPU tests for ConvSubsampling paths that the chunking/splitting tests do not cover.

    Covers `subsampling_conv_chunking_factor=-1` (chunking disabled), the 1-D conv
    stacks, and ceil-mode length bookkeeping for vggnet pooling.
    """

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    @pytest.mark.parametrize("subsampling", ["striding", "dw_striding", "vggnet"])
    @pytest.mark.parametrize("subsampling_factor", [4, 8])
    def test_no_chunking_forward_matches_chunked_path(self, subsampling, subsampling_factor):
        """With chunking disabled (-1), forward must run and match the default path."""
        torch.manual_seed(0)
        no_chunk = ConvSubsampling(
            subsampling=subsampling,
            subsampling_factor=subsampling_factor,
            feat_in=80,
            feat_out=64,
            conv_channels=32,
            subsampling_conv_chunking_factor=-1,
        ).eval()
        default = ConvSubsampling(
            subsampling=subsampling,
            subsampling_factor=subsampling_factor,
            feat_in=80,
            feat_out=64,
            conv_channels=32,
        ).eval()
        default.load_state_dict(no_chunk.state_dict())

        x = torch.randn(2, 101, 80)
        lengths = torch.tensor([101, 97])

        with torch.inference_mode():
            out, out_lengths = no_chunk(x, lengths)
            ref_out, ref_lengths = default(x, lengths)

        assert out.shape == ref_out.shape
        assert torch.allclose(out, ref_out)
        assert out_lengths.tolist() == ref_lengths.tolist()
        assert bool((out_lengths <= out.shape[1]).all())

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    @pytest.mark.parametrize("subsampling", ["striding_conv1d", "dw_striding_conv1d"])
    @pytest.mark.parametrize("subsampling_factor", [2, 4, 8])
    def test_conv1d_stacks_forward(self, subsampling, subsampling_factor):
        """The 1-D conv stacks must run with the default config and report calc_length lengths."""
        module = ConvSubsampling(
            subsampling=subsampling,
            subsampling_factor=subsampling_factor,
            feat_in=80,
            feat_out=64,
            conv_channels=32,
        ).eval()

        x = torch.randn(2, 101, 80)
        lengths = torch.tensor([101, 97])

        with torch.inference_mode():
            out, out_lengths = module(x, lengths)

        sampling_num = module._sampling_num
        expected = calc_length(
            lengths=lengths.to(dtype=torch.float),
            all_paddings=module._left_padding + module._right_padding,
            kernel_size=module._kernel_size,
            stride=module._stride,
            ceil_mode=module._ceil_mode,
            repeat_num=sampling_num,
        )
        assert out.shape[0] == 2
        assert out.shape[2] == 64
        assert out_lengths.tolist() == expected.tolist()

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    @pytest.mark.parametrize("input_length", [101, 97])
    def test_vggnet_lengths_use_ceil_mode(self, input_length):
        """vggnet pooling runs with ceil_mode=True; reported lengths must match it."""
        module = ConvSubsampling(
            subsampling='vggnet', subsampling_factor=4, feat_in=80, feat_out=64, conv_channels=32
        ).eval()

        x = torch.randn(1, input_length, 80)
        lengths = torch.tensor([input_length])

        with torch.inference_mode():
            out, out_lengths = module(x, lengths)

        expected = calc_length(
            lengths=lengths.to(dtype=torch.float),
            all_paddings=0,
            kernel_size=2,
            stride=2,
            ceil_mode=True,
            repeat_num=2,
        )
        assert out.shape[1] == expected.item()
        assert out_lengths.tolist() == expected.tolist()
