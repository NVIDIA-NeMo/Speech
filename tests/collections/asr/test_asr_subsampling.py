# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import math

import pytest
import torch

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.submodules import subsampling as subsampling_module
from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling


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


def _build_conv_subsampling(feat_in=16, conv_channels=8, factor=4):
    """A tiny dw_striding ConvSubsampling for unit-testing the 32-bit chunking logic."""
    return ConvSubsampling(
        subsampling="dw_striding",
        subsampling_factor=factor,
        feat_in=feat_in,
        feat_out=32,
        conv_channels=conv_channels,
        subsampling_conv_chunking_factor=1,
    ).eval()


def _install_split_spy(monkeypatch, sub):
    """Record the batch size of every conv_split_by_batch call; returns the list."""
    calls = []
    original_split = sub.conv_split_by_batch

    def spy_split(inp, lens):
        calls.append(int(inp.shape[0]))
        return original_split(inp, lens)

    monkeypatch.setattr(sub, "conv_split_by_batch", spy_split)
    return calls


class TestConvSubsampling32BitIndexing:
    """Unit tests for the exact 32-bit element-limit guard and auto chunking factor.

    These run on small synthetic inputs with the limit lowered via monkeypatch, so they
    exercise the splitting logic without allocating multi-GB tensors.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize("shape", [(1, 7, 16), (3, 50, 16), (5, 123, 16)])
    def test_first_conv_output_numel_matches_real_conv(self, shape):
        # The estimate must equal the actual element count of the first conv's output,
        # which is the largest activation the 32-bit limit is checked against.
        sub = _build_conv_subsampling()
        x = torch.randn(*shape)
        real_numel = sub.conv[0](x.unsqueeze(1)).numel()  # run only the first Conv2d
        assert sub._first_conv_output_numel(x) == real_numel

    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_guard_splits_at_exact_limit(self, monkeypatch, batch_size):
        # At output == limit the split must trigger. The previous '>' guard let a tensor of
        # exactly INT_MAX elements (the value that trips canUse32BitIndexMath) through unsplit.
        sub = _build_conv_subsampling()
        x = torch.randn(batch_size, 50, 16)
        lengths = torch.full((batch_size,), 50, dtype=torch.long)

        # Reference with the real (large) limit: no splitting happens.
        ref, ref_len = sub(x.clone(), lengths.clone())

        split_calls = _install_split_spy(monkeypatch, sub)
        monkeypatch.setattr(subsampling_module, "_MAX_CONV_NUMEL_32BIT", sub._first_conv_output_numel(x))

        out, out_len = sub(x.clone(), lengths.clone())

        assert split_calls, "the guard did not split when the first-conv output equals the 32-bit limit"
        # Splitting (by batch, or by channel when batch_size == 1) must not change the result.
        assert torch.allclose(out, ref, atol=1e-5)
        assert torch.equal(out_len, ref_len)

    @pytest.mark.unit
    def test_guard_does_not_split_below_limit(self, monkeypatch):
        # One element below the limit must not split: no needless chunking.
        sub = _build_conv_subsampling()
        x = torch.randn(4, 50, 16)
        lengths = torch.full((4,), 50, dtype=torch.long)

        split_calls = _install_split_spy(monkeypatch, sub)
        monkeypatch.setattr(subsampling_module, "_MAX_CONV_NUMEL_32BIT", sub._first_conv_output_numel(x) + 1)

        sub(x.clone(), lengths.clone())
        assert not split_calls

    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size", [4, 8, 16])
    def test_auto_chunking_keeps_each_chunk_below_limit(self, monkeypatch, batch_size):
        # The auto chunking factor must split into chunks whose first-conv output is strictly
        # below the limit (the previous float formula could pick a fractional factor or leave a
        # chunk sitting exactly at the limit).
        sub = _build_conv_subsampling()
        x = torch.randn(batch_size, 40, 16)
        lengths = torch.full((batch_size,), 40, dtype=torch.long)
        limit = sub._first_conv_output_numel(x) // 3 + 1  # forces a multi-way split
        monkeypatch.setattr(subsampling_module, "_MAX_CONV_NUMEL_32BIT", limit)

        # Record the batch size of each chunk actually fed to the conv stack.
        chunk_batches = []
        original_forward = sub.conv.forward

        def recording_forward(inp, lens):
            chunk_batches.append(int(inp.shape[0]))
            return original_forward(inp, lens)

        monkeypatch.setattr(sub.conv, "forward", recording_forward)

        sub(x.clone(), lengths.clone())

        assert len(chunk_batches) > 1, "expected the input to be split into multiple chunks"
        for chunk_batch in chunk_batches:
            assert sub._first_conv_output_numel(x[:chunk_batch]) < limit
