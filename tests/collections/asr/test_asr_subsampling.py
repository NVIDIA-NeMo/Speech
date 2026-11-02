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
import os
import tempfile

import onnx
import pytest
import torch
from omegaconf import ListConfig

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.submodules import subsampling as subsampling_module
from nemo.collections.asr.parts.submodules.subsampling import (
    ConvSubsampling,
    SubsamplingReductionModule,
    _layer_padding,
)


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


def _build_strided_conv_subsampling(**overrides):
    """A tiny striding ConvSubsampling with per-stage time strides and a frequency pool."""
    options = dict(
        subsampling="striding",
        subsampling_factor=6,
        feat_in=8,
        feat_out=16,
        conv_channels=[2, 3, 3],
        conv_strides=[1, 3, 2],
        freq_pools=[2, 1, 1],
        out_bias=False,
    )
    options.update(overrides)
    return ConvSubsampling(**options).eval()


class TestConvSubsamplingStrides:
    @pytest.mark.unit
    def test_lengths_shapes_and_padding_invariance(self):
        torch.manual_seed(0)
        module = _build_strided_conv_subsampling()
        features = torch.randn(2, 50, 8)
        lengths = torch.tensor([50, 31])
        with torch.no_grad():
            out, out_lengths = module(x=features, lengths=lengths)
            alone, alone_lengths = module(x=features[1:, :31], lengths=lengths[1:])
        # time strides 1, 3, 2 with kernel 3 and padding 1: ceil(ceil(T / 3) / 2)
        assert out.shape == (2, 9, 16)
        assert out_lengths.tolist() == [9, 6] and alone_lengths.tolist() == [6]
        assert module.out.bias is None
        # the frequency axis is halved once by the pool, so the projection sees 3 channels x 4 bins
        assert module.out.weight.shape == (16, 12)
        torch.testing.assert_close(out[1, :6], alone[0, :6], atol=1e-5, rtol=1e-5)

    @pytest.mark.unit
    def test_channels_as_omegaconf_list(self):
        module = _build_strided_conv_subsampling(conv_channels=ListConfig([2, 3, 3]), freq_pools=ListConfig([2, 1, 1]))
        assert module.out.weight.shape == (16, 12)

    @pytest.mark.unit
    def test_validation(self):
        with pytest.raises(ValueError, match="product of conv_strides"):
            ConvSubsampling("striding", 4, 8, 16, [2, 2], conv_strides=[1, 3])
        with pytest.raises(ValueError, match="only supported with subsampling='striding'"):
            ConvSubsampling("dw_striding", 6, 8, 16, [2, 2, 2], conv_strides=[1, 3, 2])
        with pytest.raises(ValueError, match="one entry per stride"):
            ConvSubsampling("striding", 6, 8, 16, [2, 2], conv_strides=[1, 3, 2])
        with pytest.raises(ValueError, match="needs conv_strides"):
            ConvSubsampling("striding", 4, 8, 16, [2, 3])
        with pytest.raises(ValueError, match="conv_strides must be positive integers"):
            ConvSubsampling("striding", 6, 8, 16, 4, conv_strides=[-1, -3, 2])
        with pytest.raises(ValueError, match="freq_pools must be positive integers"):
            ConvSubsampling("striding", 6, 8, 16, 4, conv_strides=[1, 3, 2], freq_pools=[0, 1, 1])

    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size", [4, 1])
    def test_batch_chunking_matches_unchunked(self, batch_size):
        torch.manual_seed(0)
        module = _build_strided_conv_subsampling()
        features = torch.randn(batch_size, 50, 8)
        lengths = torch.tensor([50, 31, 50, 12][:batch_size])
        with torch.no_grad():
            reference, reference_lengths = module(x=features, lengths=lengths)
            # a factor above 1 always splits by batch; a batch of one falls back to the unsplit convolution
            module.change_subsampling_conv_chunking_factor(2)
            out, out_lengths = module(x=features, lengths=lengths)
        torch.testing.assert_close(out, reference, atol=1e-5, rtol=1e-5)
        assert out_lengths.tolist() == reference_lengths.tolist()

    @pytest.mark.unit
    def test_export_to_onnx(self):
        torch.manual_seed(0)
        module = _build_strided_conv_subsampling()
        features = torch.randn(2, 50, 8)
        lengths = torch.tensor([50, 31])
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'subsampling.onnx')
            torch.onnx.export(
                module,
                (features, lengths),
                filename,
                input_names=['features', 'lengths'],
                output_names=['outputs', 'out_lengths'],
                dynamic_axes={
                    'features': {0: 'batch', 1: 'time'},
                    'lengths': {0: 'batch'},
                    'outputs': {0: 'batch', 1: 'time'},
                    'out_lengths': {0: 'batch'},
                },
                opset_version=17,
                dynamo=False,
            )
            onnx_model = onnx.load(filename)
            onnx.checker.check_model(onnx_model, full_check=True)  # throws when failed
            assert [node.name for node in onnx_model.graph.input] == ['features', 'lengths']
            try:
                import onnxruntime
            except ImportError:
                return
            session = onnxruntime.InferenceSession(filename, providers=['CPUExecutionProvider'])
            outputs, out_lengths = session.run(None, {'features': features.numpy(), 'lengths': lengths.numpy()})
        with torch.no_grad():
            reference, reference_lengths = module(x=features, lengths=lengths)
        torch.testing.assert_close(torch.from_numpy(outputs), reference, atol=1e-5, rtol=1e-5)
        assert out_lengths.tolist() == reference_lengths.tolist()

    @pytest.mark.unit
    def test_production_configuration(self):
        # 80 mel features, channels 32, 64, 64, time strides 1, 3, 2 and one frequency pool of 2
        module = _build_strided_conv_subsampling(feat_in=80, feat_out=256, conv_channels=[32, 64, 64])
        features = torch.randn(3, 101, 80)
        lengths = torch.tensor([101, 67, 13])
        with torch.no_grad():
            out, out_lengths = module(x=features, lengths=lengths)
        assert out.shape == (3, 17, 256) and out_lengths.tolist() == [17, 12, 3]
        assert module.out.weight.shape == (256, 64 * 40)
        # the 32-bit indexing guard counts the first convolution's output elements,
        # which in this configuration is the largest activation of the stack
        first_conv_numel = module.conv[0](features.unsqueeze(1)).numel()
        assert module._first_conv_output_numel(features) == first_conv_numel
        assert module._max_conv_numel(features) == first_conv_numel

    @pytest.mark.unit
    def test_max_conv_numel_counts_every_stage(self):
        # channels 2 then 8 with time strides 2, 1 and no pooling: the second convolution's output
        # is four times the first one's, so the first-conv estimate alone would under-count the stack
        module = _build_strided_conv_subsampling(
            subsampling_factor=2, conv_channels=[2, 8], conv_strides=[2, 1], freq_pools=[1, 1]
        )
        features = torch.randn(4, 50, 8)
        activation_numels = [features.numel()]
        x = features.unsqueeze(1)
        with torch.no_grad():
            for layer in module.conv:
                x = layer(x)
                if isinstance(layer, torch.nn.Conv2d):
                    activation_numels.append(x.numel())
        assert activation_numels == [1600, 1600, 6400]
        assert module._max_conv_numel(features) == max(activation_numels)
        assert module._first_conv_output_numel(features) == 1600
        # a frequency pool after the first stage halves the second convolution's map
        pooled = _build_strided_conv_subsampling(
            subsampling_factor=2, conv_channels=[2, 8], conv_strides=[2, 1], freq_pools=[2, 1]
        )
        assert pooled._max_conv_numel(features) == 3200

    @pytest.mark.unit
    def test_guard_splits_when_a_later_convolution_is_largest(self, monkeypatch):
        torch.manual_seed(0)
        module = _build_strided_conv_subsampling(
            subsampling_factor=2, conv_channels=[2, 8], conv_strides=[2, 1], freq_pools=[1, 1]
        )
        features = torch.randn(4, 50, 8)
        lengths = torch.tensor([50, 31, 50, 12])
        with torch.no_grad():
            reference, reference_lengths = module(x=features, lengths=lengths)
        # the limit sits at the second convolution's output, above the input and the first output
        limit = module._max_conv_numel(features)
        assert limit > max(features.numel(), module._first_conv_output_numel(features))
        monkeypatch.setattr(subsampling_module, "_MAX_CONV_NUMEL_32BIT", limit)
        split_calls = _install_split_spy(monkeypatch, module)
        chunk_batches = []
        original_forward = module.conv.forward

        def recording_forward(inp, lens):
            chunk_batches.append(int(inp.shape[0]))
            return original_forward(inp, lens)

        monkeypatch.setattr(module.conv, "forward", recording_forward)
        with torch.no_grad():
            out, out_lengths = module(x=features, lengths=lengths)
        assert split_calls == [4] and chunk_batches == [2, 2]
        for chunk_batch in chunk_batches:
            assert module._max_conv_numel(features[:chunk_batch]) < limit
        torch.testing.assert_close(out, reference, atol=1e-5, rtol=1e-5)
        assert out_lengths.tolist() == reference_lengths.tolist()
        # with a frequency pool after the first stage the stack stays below the same limit: no split
        pooled = _build_strided_conv_subsampling(
            subsampling_factor=2, conv_channels=[2, 8], conv_strides=[2, 1], freq_pools=[2, 1]
        )
        pooled_split_calls = _install_split_spy(monkeypatch, pooled)
        with torch.no_grad():
            pooled(x=features, lengths=lengths)
        assert not pooled_split_calls

    @pytest.mark.unit
    def test_defaults_unchanged(self):
        module = ConvSubsampling("striding", 4, 8, 16, 4)
        assert module.out.bias is not None and module.out.weight.shape == (16, 8)
        features = torch.randn(1, 20, 8)
        out, out_lengths = module(x=features, lengths=torch.tensor([20]))
        assert out.shape == (1, 5, 16) and out_lengths.tolist() == [5]

    @pytest.mark.unit
    def test_reset_parameters_without_output_bias(self):
        # reset_parameters re-initializes the dw_striding stages and the projection; no bias to draw without out_bias
        module = ConvSubsampling("dw_striding", 4, 8, 16, 4, out_bias=False)
        assert module.out.bias is None
        module.reset_parameters()
        with_bias = ConvSubsampling("dw_striding", 4, 8, 16, 4)
        torch.nn.init.zeros_(with_bias.out.bias)
        with_bias.reset_parameters()
        assert with_bias.out.bias.abs().sum() > 0

    @pytest.mark.unit
    def test_layer_padding_of_pooling_layers(self):
        assert _layer_padding(torch.nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))) == (0, 0)
        assert _layer_padding(torch.nn.Conv2d(1, 1, kernel_size=3, padding=1)) == (1, 1)
