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
from nemo.collections.asr.parts.submodules import subsampling as subsampling_module
from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling, SubsamplingReductionModule


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
        "cache_size,expected_drop",
        [(11, 1), ([0, 11], 2), (0, 0), ([0, 0], 0)],
        ids=["scalar", "list", "empty_scalar", "empty_list"],
    )
    def test_legacy_pre_encode_drop_size(self, cache_size: int | list[int], expected_drop: int) -> None:
        from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder

        class LegacyPreEncode(torch.nn.Module):
            def get_streaming_cache_size(self) -> int | list[int]:
                return cache_size

        encoder = ConformerEncoder(
            feat_in=80,
            n_layers=1,
            d_model=16,
            subsampling_factor=8,
            n_heads=4,
        )
        encoder.pre_encode = LegacyPreEncode()
        encoder.setup_streaming_params()

        assert encoder.streaming_cfg.pre_encode_cache_size == cache_size
        assert encoder.streaming_cfg.drop_extra_pre_encoded == expected_drop

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
    """Guard and auto-chunking tests, run on small inputs with the limit lowered via monkeypatch."""

    @pytest.mark.unit
    @pytest.mark.parametrize("shape", [(1, 7, 16), (3, 50, 16), (5, 123, 16)])
    def test_first_conv_output_numel_matches_real_conv(self, shape):
        sub = _build_conv_subsampling()
        x = torch.randn(*shape)
        real_numel = sub.conv[0](x.unsqueeze(1)).numel()  # only the first Conv2d
        assert sub._first_conv_output_numel(x) == real_numel

    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_guard_splits_at_exact_limit(self, monkeypatch, batch_size):
        # '>=' must split at output == limit; the old '>' let an INT_MAX tensor through.
        sub = _build_conv_subsampling()
        x = torch.randn(batch_size, 50, 16)
        lengths = torch.full((batch_size,), 50, dtype=torch.long)

        ref, ref_len = sub(x.clone(), lengths.clone())

        split_calls = _install_split_spy(monkeypatch, sub)
        monkeypatch.setattr(subsampling_module, "_MAX_CONV_NUMEL_32BIT", sub._first_conv_output_numel(x))

        out, out_len = sub(x.clone(), lengths.clone())

        assert split_calls, "the guard did not split when the first-conv output equals the limit"
        assert torch.allclose(out, ref, atol=1e-5)
        assert torch.equal(out_len, ref_len)

    @pytest.mark.unit
    def test_guard_does_not_split_below_limit(self, monkeypatch):
        # One element below the limit must not split.
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
        # Each chunk's first-conv output must end up strictly below the limit.
        sub = _build_conv_subsampling()
        x = torch.randn(batch_size, 40, 16)
        lengths = torch.full((batch_size,), 40, dtype=torch.long)
        limit = sub._first_conv_output_numel(x) // 3 + 1  # forces a multi-way split
        monkeypatch.setattr(subsampling_module, "_MAX_CONV_NUMEL_32BIT", limit)

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

    @pytest.mark.unit
    def test_split_by_batch_uses_size_one_when_factor_exceeds_batch(self, monkeypatch):
        # b=3 makes cf round up to 4, so b // cf == 0; must use single-sample batches instead
        # of the channel fallback (which misreads the batch as channels and errors out).
        sub = _build_conv_subsampling()
        x = torch.randn(3, 50, 16)
        lengths = torch.full((3,), 50, dtype=torch.long)

        ref, ref_len = sub(x.clone(), lengths.clone())

        # Limit above one sample but below the whole batch, so cf == 4 > b == 3 yet a sample fits.
        limit = sub._first_conv_output_numel(x) // 2
        assert sub._first_conv_output_numel(x[:1]) < limit
        monkeypatch.setattr(subsampling_module, "_MAX_CONV_NUMEL_32BIT", limit)

        channel_calls = []
        original_channel = sub.conv_split_by_channel

        def spy_channel(inp):
            channel_calls.append(int(inp.shape[0]))
            return original_channel(inp)

        monkeypatch.setattr(sub, "conv_split_by_channel", spy_channel)

        chunk_batches = []
        original_forward = sub.conv.forward

        def recording_forward(inp, lens):
            chunk_batches.append(int(inp.shape[0]))
            return original_forward(inp, lens)

        monkeypatch.setattr(sub.conv, "forward", recording_forward)

        out, out_len = sub(x.clone(), lengths.clone())

        assert not channel_calls, "should split by batch (size 1), not fall back to channel splitting"
        assert chunk_batches == [1, 1, 1], f"expected three single-sample chunks, got {chunk_batches}"
        assert torch.allclose(out, ref, atol=1e-5)
        assert torch.equal(out_len, ref_len)


class TestSubsamplingReductionModulePooling:
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    @pytest.mark.parametrize("reduction_factor", [2, 4, 8])
    def test_pooling_lengths_match_output(self, reduction_factor):
        """Pool-based reduction applies a single MaxPool1d(kernel_size=reduction_factor),
        so the returned lengths must match that one pooling step."""
        module = SubsamplingReductionModule(reduction='pooling', d_model=8, reduction_factor=reduction_factor)

        x = torch.randn(2, 100, 8)
        lengths = torch.tensor([100, 90])

        out, out_lengths = module(x, lengths)

        assert out.shape == (2, out.shape[1], 8)
        assert out_lengths[0].item() == out.shape[1]
        expected = torch.div(lengths - reduction_factor, reduction_factor, rounding_mode='floor') + 1
        assert out_lengths.tolist() == expected.tolist()
