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
import pytest
import torch
from nemo.collections.asr.models import ASRModel


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


class TestStreamingDropExtraPreEncoded:
    """``ConvSubsampling.get_streaming_drop_size`` must match what the encoder actually
    produces from a ``cache_size``-long input segment.

    Regression test for the streaming/full-pass mismatch reported in
    https://github.com/NVIDIA-NeMo/NeMo/issues/15482 — the old formula
    ``1 + (cache_size - 1) // subsampling_factor`` diverges from the true convolutional
    recurrence for arbitrary ``pre_encode_cache_size``.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "subsampling,subsampling_factor",
        [
            ("striding", 4),
            ("striding", 8),
            ("dw_striding", 4),
            ("dw_striding", 8),
        ],
    )
    @pytest.mark.parametrize("cache_size", [1, 4, 8, 9, 11, 16, 32])
    def test_drop_size_matches_forward(self, subsampling, subsampling_factor, cache_size):
        """For a causal conv subsampling, the number of output frames the actual
        ``forward`` returns from a ``cache_size``-long input must equal
        ``get_streaming_drop_size(cache_size)``.
        """
        from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling

        feat_in = 80
        sub = ConvSubsampling(
            subsampling=subsampling,
            subsampling_factor=subsampling_factor,
            feat_in=feat_in,
            feat_out=16,
            conv_channels=16,
            subsampling_conv_chunking_factor=1,
            is_causal=True,
        )
        sub.eval()
        x = torch.zeros(1, cache_size, feat_in)
        lengths = torch.tensor([cache_size], dtype=torch.int64)
        with torch.no_grad():
            _, out_lengths = sub(x, lengths)
        expected = int(out_lengths[0].item())
        assert sub.get_streaming_drop_size(cache_size) == expected

    @pytest.mark.unit
    def test_drop_size_zero_for_empty_cache(self):
        from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling, StackingSubsampling

        sub = ConvSubsampling(
            subsampling="striding",
            subsampling_factor=8,
            feat_in=80,
            feat_out=16,
            conv_channels=16,
            subsampling_conv_chunking_factor=1,
            is_causal=True,
        )
        assert sub.get_streaming_drop_size(0) == 0

        stack = StackingSubsampling(subsampling_factor=4, feat_in=80, feat_out=16)
        assert stack.get_streaming_drop_size(0) == 0

    @pytest.mark.unit
    def test_drop_size_legacy_formula_diverges_for_non_default_cache(self):
        """Document the bug being fixed: at the issue-reported case ``cache_size=11``
        with ``subsampling_factor=8``, the old formula returns 2 but the true value is 3.
        """
        from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling

        sub = ConvSubsampling(
            subsampling="striding",
            subsampling_factor=8,
            feat_in=80,
            feat_out=16,
            conv_channels=16,
            subsampling_conv_chunking_factor=1,
            is_causal=True,
        )
        cache_size = 11
        legacy = 1 + (cache_size - 1) // 8
        assert legacy == 2  # old, wrong
        assert sub.get_streaming_drop_size(cache_size) == 3  # new, matches the forward pass

    @pytest.mark.unit
    def test_stacking_drop_size(self):
        from nemo.collections.asr.parts.submodules.subsampling import StackingSubsampling

        stack = StackingSubsampling(subsampling_factor=4, feat_in=80, feat_out=16)
        # StackingSubsampling.get_streaming_cache_size() returns 0 by default, but the
        # helper should still answer sensibly for any positive cache_size.
        assert stack.get_streaming_drop_size(4) == 1
        assert stack.get_streaming_drop_size(7) == 1
        assert stack.get_streaming_drop_size(8) == 2
