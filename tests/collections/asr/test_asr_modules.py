# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import copy
import random

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.asr import modules
from nemo.collections.asr.parts.packed_sequence import (
    PackedEncoderActivations,
    pack_encoder_output,
    split_encoder_output,
)
from nemo.collections.asr.parts.preprocessing.features import (
    FilterbankFeatures,
    normalize_batch,
    normalize_packed_batch,
)
from nemo.collections.asr.parts.submodules.subsampling import FeatureStacking
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.core.utils import numba_utils
from nemo.core.utils.numba_utils import __NUMBA_MINIMUM_VERSION__
from nemo.utils import config_utils, logging


class TestASRModulesBasicTests:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("exact_pad", "preemph", "normalize", "pad_to", "frame_splicing"),
        [
            (False, 0.97, "per_feature", 0, 1),
            (False, None, "all_features", 16, 1),
            (True, 0.97, None, 0, 1),
            (False, 0.97, "per_feature", 16, 2),
        ],
    )
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
        ],
    )
    def test_AudioToMelSpectrogramPreprocessor_packed_waveform_matches_padded(
        self, exact_pad, preemph, normalize, pad_to, frame_splicing, device
    ):
        preprocessor = (
            modules.AudioToMelSpectrogramPreprocessor(
                normalize=normalize,
                dither=0,
                pad_to=pad_to,
                exact_pad=exact_pad,
                preemph=preemph,
                frame_splicing=frame_splicing,
            )
            .eval()
            .to(device)
        )
        lengths = torch.tensor([4096, 2500, 701], dtype=torch.long, device=device)
        torch.manual_seed(7)
        audios = torch.randn(3, int(lengths.max()), device=device)
        for row, length in zip(audios, lengths):
            row[int(length) :] = 0.0
        packed_audio_samples = torch.cat([row[: int(length)] for row, length in zip(audios, lengths)])
        audio_cu_seqlens = torch.cat([lengths.new_zeros(1), lengths.cumsum(dim=0, dtype=torch.long)])

        expected, expected_lens = preprocessor(input_signal=audios, length=lengths)
        actual = preprocessor.forward_packed(
            input_signal=packed_audio_samples,
            length=lengths,
            input_signal_cu_seqlens=audio_cu_seqlens,
        )

        assert isinstance(actual, PackedEncoderActivations)
        assert torch.equal(actual.lengths, expected_lens)
        assert actual.padded_length == expected.shape[2]
        assert actual.data.shape == (int(expected_lens.sum()), expected.shape[1])
        for row, (features, valid_length) in enumerate(zip(split_encoder_output(actual), expected_lens.tolist())):
            torch.testing.assert_close(
                features,
                expected[row, :, :valid_length].transpose(0, 1),
                rtol=1e-5,
                atol=2e-6,
            )

    @pytest.mark.unit
    def test_AudioToMelSpectrogramPreprocessor_packed_waveform_uses_one_vectorized_stft(self, monkeypatch):
        preprocessor = modules.AudioToMelSpectrogramPreprocessor(dither=0, pad_to=16).eval()
        lengths = torch.tensor([4096, 2500, 701], dtype=torch.long)
        padded = torch.randn(3, int(lengths.max()))
        packed = torch.cat([row[: int(length)] for row, length in zip(padded, lengths)])
        cu_seqlens = torch.cat([lengths.new_zeros(1), lengths.cumsum(dim=0)])
        calls = 0
        stft_input_shape = None
        original_stft = preprocessor.featurizer.stft

        def count_stft(*args, **kwargs):
            nonlocal calls, stft_input_shape
            calls += 1
            stft_input_shape = args[0].shape
            return original_stft(*args, **kwargs)

        def reject_dense_frontend(*args, **kwargs):
            raise AssertionError("packed waveform preprocessing must not call the dense frontend")

        monkeypatch.setattr(preprocessor.featurizer, "stft", count_stft)
        monkeypatch.setattr(preprocessor, "forward", reject_dense_frontend)
        actual = preprocessor.forward_packed(packed, lengths, cu_seqlens)

        assert calls == 1
        assert stft_input_shape[0] == 1
        assert stft_input_shape[1] <= packed.numel() + lengths.numel() * (
            preprocessor.featurizer.n_fft + preprocessor.featurizer.hop_length
        )
        assert actual.total_tokens == int(actual.lengths.sum())

    @pytest.mark.unit
    def test_AudioToMelSpectrogramPreprocessor_packed_waveform_isolates_boundaries(self):
        preprocessor = modules.AudioToMelSpectrogramPreprocessor(
            normalize="per_feature", dither=0, pad_to=0, preemph=0.97
        ).eval()
        first = torch.zeros(1600)
        first[-1] = 1.0
        second = torch.zeros(1000)
        second[0] = -1.0
        lengths = torch.tensor([first.numel(), second.numel()], dtype=torch.long)
        packed = torch.cat([first, second])
        cu_seqlens = torch.tensor([0, first.numel(), packed.numel()], dtype=torch.long)

        actual = preprocessor.forward_packed(
            input_signal=packed,
            length=lengths,
            input_signal_cu_seqlens=cu_seqlens,
        )

        for row, (features, waveform) in enumerate(zip(split_encoder_output(actual), (first, second))):
            expected, expected_lens = preprocessor(input_signal=waveform.unsqueeze(0), length=lengths[row : row + 1])
            assert actual.lengths[row] == expected_lens[0]
            valid_length = int(expected_lens[0])
            torch.testing.assert_close(features, expected[0, :, :valid_length].transpose(0, 1), rtol=1e-5, atol=2e-6)

    @pytest.mark.unit
    @pytest.mark.parametrize("exact_pad", [False, True])
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
        ],
    )
    def test_AudioToMelSpectrogramPreprocessor_packed_waveform_preserves_zero_length_rows(self, exact_pad, device):
        preprocessor = (
            modules.AudioToMelSpectrogramPreprocessor(dither=0, pad_to=0, exact_pad=exact_pad).eval().to(device)
        )
        lengths = torch.tensor([0, 1600, 2400], dtype=torch.long, device=device)
        audios = torch.randn(3, 2400, device=device)
        audios[0].zero_()
        audios[1, 1600:].zero_()
        packed = torch.cat([audios[1, :1600], audios[2]])
        cu_seqlens = torch.tensor([0, 0, 1600, 4000], dtype=torch.long, device=device)

        expected, expected_lens = preprocessor(input_signal=audios, length=lengths)
        actual = preprocessor.forward_packed(packed, lengths, cu_seqlens)

        assert torch.equal(actual.lengths, expected_lens)
        for row, features in enumerate(split_encoder_output(actual)):
            valid_length = int(expected_lens[row])
            torch.testing.assert_close(features, expected[row, :, :valid_length].transpose(0, 1), rtol=1e-5, atol=3e-6)

    @pytest.mark.unit
    def test_AudioToMelSpectrogramPreprocessor_packed_dither_is_seed_deterministic(self):
        preprocessor = modules.AudioToMelSpectrogramPreprocessor(
            normalize="per_feature", dither=1e-5, pad_to=0
        ).train()
        lengths = torch.tensor([1600, 900], dtype=torch.long)
        packed = torch.linspace(-0.5, 0.5, int(lengths.sum()))
        cu_seqlens = torch.tensor([0, 1600, 2500], dtype=torch.long)
        padded = torch.zeros(2, 1600)
        padded[0] = packed[:1600]
        padded[1, :900] = packed[1600:]

        torch.manual_seed(11)
        expected, expected_lens = preprocessor(input_signal=padded.clone(), length=lengths)
        torch.manual_seed(11)
        first = preprocessor.forward_packed(packed.clone(), lengths, cu_seqlens)
        torch.manual_seed(11)
        second = preprocessor.forward_packed(packed.clone(), lengths, cu_seqlens)

        assert torch.equal(first.lengths, expected_lens)
        assert torch.equal(first.lengths, second.lengths)
        assert torch.equal(first.data, second.data)
        for row, (features, valid_length) in enumerate(zip(split_encoder_output(first), expected_lens.tolist())):
            # Padded batches consume dither RNG for padding positions, so later rows
            # are tolerance-equivalent rather than bitwise-identical.
            torch.testing.assert_close(
                features,
                expected[row, :, :valid_length].transpose(0, 1),
                rtol=1e-3,
                atol=5e-4,
            )

    @pytest.mark.unit
    def test_AudioToMelSpectrogramPreprocessor_packed_waveform_validates_metadata(self):
        preprocessor = modules.AudioToMelSpectrogramPreprocessor(dither=0, pad_to=0)
        packed = torch.zeros(6)
        lengths = torch.tensor([4, 2], dtype=torch.long)

        with pytest.raises(ValueError, match="must equal length"):
            preprocessor.forward_packed(packed, lengths, torch.tensor([0, 3, 6]))
        with pytest.raises(ValueError, match="ends at 6"):
            preprocessor.forward_packed(packed[:5], lengths, torch.tensor([0, 4, 6]))
        with pytest.raises(TypeError, match="integer dtype"):
            preprocessor.forward_packed(packed, lengths, torch.tensor([0.0, 4.0, 6.0]))

    @pytest.mark.unit
    def test_AudioToMelSpectrogramPreprocessor_packed_padding_value_matches_dense_feature_stacking(self):
        preprocessor = modules.AudioToMelSpectrogramPreprocessor(
            features=8, normalize="per_feature", dither=0, pad_to=16, pad_value=-3.0
        ).eval()
        lengths = torch.tensor([4096, 2500, 701])
        audios = torch.randn(3, int(lengths.max()))
        audios.masked_fill_(torch.arange(audios.shape[1])[None, :] >= lengths[:, None], 0.0)
        packed_audio = torch.cat([row[: int(length)] for row, length in zip(audios, lengths)])
        audio_cu_seqlens = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
        stacking = FeatureStacking(subsampling_factor=4, feat_in=8, feat_out=16).eval()

        dense_mels, mel_lengths = preprocessor(input_signal=audios, length=lengths)
        packed_mels = preprocessor.forward_packed(packed_audio, lengths, audio_cu_seqlens)
        dense_stacked, output_lengths = stacking(dense_mels, mel_lengths)
        expected = pack_encoder_output(dense_stacked, output_lengths)
        actual = stacking.forward_packed(packed_mels)

        assert packed_mels.padding_value == -3.0
        torch.testing.assert_close(actual.data, expected.data, rtol=1e-5, atol=2e-6)

    @pytest.mark.unit
    @pytest.mark.parametrize("linear_spec", [False, True])
    def test_FilterbankFeatures_packed_all_empty_has_the_right_width(self, linear_spec):
        featurizer = FilterbankFeatures(nfilt=23, frame_splicing=2, dither=0, pad_to=0).eval()
        packed = featurizer.forward_packed(
            torch.empty(0), torch.tensor([0, 0]), torch.tensor([0, 0, 0]), linear_spec=linear_spec
        )

        expected_width = featurizer.n_fft // 2 + 1 if linear_spec else 46
        assert packed.data.shape == (0, expected_width)
        assert packed.lengths.tolist() == [0, 0]
        assert packed.cu_seqlens.tolist() == [0, 0, 0]

    @pytest.mark.unit
    def test_FilterbankFeatures_forward_packed_honors_use_grads(self):
        lengths = torch.tensor([1600, 900])
        cu_seqlens = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
        samples = torch.randn(int(lengths.sum()), requires_grad=True)
        frozen = FilterbankFeatures(nfilt=8, normalize=None, dither=0, pad_to=0, use_grads=False).eval()
        differentiable = FilterbankFeatures(nfilt=8, normalize=None, dither=0, pad_to=0, use_grads=True).eval()

        assert not frozen.forward_packed(samples, lengths, cu_seqlens).data.requires_grad
        output = differentiable.forward_packed(samples, lengths, cu_seqlens)
        output.data.sum().backward()
        assert samples.grad is not None
        assert torch.isfinite(samples.grad).all()

    @pytest.mark.unit
    def test_FilterbankFeatures_packed_linear_spectrum_matches_nondefault_dense_stft(self):
        featurizer = FilterbankFeatures(
            n_window_size=400,
            n_window_stride=137,
            n_fft=512,
            nfilt=8,
            normalize=None,
            dither=0,
            pad_to=0,
        ).eval()
        lengths = torch.tensor([4097, 2503])
        audios = torch.randn(2, int(lengths.max()))
        audios.masked_fill_(torch.arange(audios.shape[1])[None, :] >= lengths[:, None], 0.0)
        packed_audio = torch.cat([row[: int(length)] for row, length in zip(audios, lengths)])
        cu_seqlens = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])

        expected, expected_lengths = featurizer(audios, lengths, linear_spec=True)
        actual = featurizer.forward_packed(packed_audio, lengths, cu_seqlens, linear_spec=True)
        expected = pack_encoder_output(expected.transpose(1, 2), expected_lengths)

        torch.testing.assert_close(actual.data, expected.data, rtol=1e-5, atol=3e-6)

    @pytest.mark.unit
    def test_FilterbankFeatures_packed_fixed_normalization_matches_dense(self):
        normalize = {
            "fixed_mean": [value for row in ([0.1] * 8, [-0.2] * 8) for value in row],
            "fixed_std": [value for row in ([1.5] * 8, [0.75] * 8) for value in row],
        }
        featurizer = FilterbankFeatures(nfilt=8, normalize=normalize, dither=0, pad_to=0).eval()
        lengths = torch.tensor([4096, 2500])
        audios = torch.randn(2, int(lengths.max()))
        audios.masked_fill_(torch.arange(audios.shape[1])[None, :] >= lengths[:, None], 0.0)
        packed_audio = torch.cat([row[: int(length)] for row, length in zip(audios, lengths)])
        cu_seqlens = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])

        expected, expected_lengths = featurizer(audios, lengths)
        actual = featurizer.forward_packed(packed_audio, lengths, cu_seqlens)
        expected = pack_encoder_output(expected.transpose(1, 2), expected_lengths)

        torch.testing.assert_close(actual.data, expected.data, rtol=1e-5, atol=3e-6)

    @pytest.mark.unit
    @pytest.mark.parametrize("use_vectorized_code", [False, True])
    def test_SpectrogramAugmentation_forward_packed_matches_dense_vectorized_masks(self, use_vectorized_code):
        augment = modules.SpectrogramAugmentation(
            freq_masks=2,
            time_masks=2,
            freq_width=3,
            time_width=4,
            rect_masks=1,
            rect_time=3,
            rect_freq=2,
            rng=random.Random(13),
            mask_value=-2.0,
            use_vectorized_spec_augment=use_vectorized_code,
        ).train()
        lengths = torch.tensor([12, 7, 0])
        features = torch.randn(3, 8, 12)
        features.masked_fill_(torch.arange(12)[None, None, :] >= lengths[:, None, None], 0.0)
        packed = pack_encoder_output(features.transpose(1, 2), lengths)
        packed_augment = copy.deepcopy(augment)
        augment.spec_augment.use_vectorized_code = True

        torch.manual_seed(29)
        expected = augment(input_spec=features.clone(), length=lengths)
        torch.manual_seed(29)
        actual = packed_augment.forward_packed(packed)
        expected = pack_encoder_output(expected.transpose(1, 2), lengths)

        torch.testing.assert_close(actual.data, expected.data, rtol=0.0, atol=0.0)

    @pytest.mark.unit
    def test_SpectrogramAugmentation_packed_cutout_preserves_frontend_padded_rng_range(self):
        preprocessor = modules.AudioToMelSpectrogramPreprocessor(
            features=8, normalize=None, dither=0, pad_to=16
        ).eval()
        lengths = torch.tensor([1600, 900])
        audios = torch.randn(2, int(lengths.max()))
        audios.masked_fill_(torch.arange(audios.shape[1])[None, :] >= lengths[:, None], 0.0)
        packed_audio = torch.cat([row[: int(length)] for row, length in zip(audios, lengths)])
        cu_seqlens = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
        dense_mels, mel_lengths = preprocessor(input_signal=audios, length=lengths)
        packed_mels = preprocessor.forward_packed(packed_audio, lengths, cu_seqlens)
        augment = modules.SpectrogramAugmentation(
            freq_masks=0,
            time_masks=0,
            rect_masks=2,
            rect_time=5,
            rect_freq=2,
            rng=random.Random(31),
        )
        packed_augment = copy.deepcopy(augment)

        expected = augment(input_spec=dense_mels.clone(), length=mel_lengths)
        actual = packed_augment.forward_packed(packed_mels)
        expected = pack_encoder_output(expected.transpose(1, 2), mel_lengths)

        assert packed_mels.padded_length == dense_mels.shape[2]
        assert torch.equal(actual.data == 0.0, expected.data == 0.0)
        torch.testing.assert_close(actual.data, expected.data, rtol=1e-5, atol=2e-6)

    @pytest.mark.unit
    def test_SpectrogramAugmentation_packed_cutout_accepts_all_empty_batch(self):
        augment = modules.SpectrogramAugmentation(rect_masks=1, rect_time=5, rect_freq=2)
        packed = PackedEncoderActivations(
            torch.empty(0, 8),
            torch.tensor([0, 0]),
            torch.tensor([0, 0, 0], dtype=torch.int32),
            0,
        )

        assert augment.forward_packed(packed) is packed

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "normalize_type",
        [
            None,
            "per_feature",
            "all_features",
            {
                "fixed_mean": [value for row in ([0.1] * 4, [-0.2] * 4) for value in row],
                "fixed_std": [value for row in ([1.5] * 4, [0.75] * 4) for value in row],
            },
        ],
    )
    def test_normalize_packed_batch_preserves_dense_partial_stack_padding(self, normalize_type):
        lengths = torch.tensor([7, 4])
        features = torch.randn(2, 4, 7)
        features.masked_fill_(torch.arange(7)[None, None, :] >= lengths[:, None, None], -3.0)
        packed = PackedEncoderActivations(
            pack_encoder_output(features.transpose(1, 2), lengths).data,
            lengths,
            torch.tensor([0, 7, 11], dtype=torch.int32),
            7,
            padding_value=-3.0,
            padded_length=7,
        )
        stacking = FeatureStacking(subsampling_factor=3, feat_in=4, feat_out=6).eval()

        if normalize_type is None:
            dense_normalized = features
            packed_normalized = packed
        else:
            dense_normalized, _, _ = normalize_batch(features, lengths, normalize_type)
            packed_normalized = normalize_packed_batch(packed, normalize_type)
        expected, output_lengths = stacking(dense_normalized, lengths)
        expected = pack_encoder_output(expected, output_lengths)
        actual = stacking.forward_packed(packed_normalized)

        torch.testing.assert_close(actual.data, expected.data, rtol=1e-5, atol=2e-6)

    @pytest.mark.unit
    def test_FilterbankFeatures_packed_narrowband_uses_configured_rng(self):
        featurizer = FilterbankFeatures(
            nfilt=8,
            normalize=None,
            dither=0,
            pad_to=0,
            rng=random.Random(17),
            nb_augmentation_prob=0.5,
            nb_max_freq=2000,
        ).train()
        lengths = torch.tensor([4096, 2500, 1600])
        audios = torch.randn(3, int(lengths.max()))
        audios.masked_fill_(torch.arange(audios.shape[1])[None, :] >= lengths[:, None], 0.0)
        packed_audio = torch.cat([row[: int(length)] for row, length in zip(audios, lengths)])
        cu_seqlens = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
        state = featurizer._rng.getstate()

        expected, expected_lengths = featurizer(audios, lengths)
        featurizer._rng.setstate(state)
        actual = featurizer.forward_packed(packed_audio, lengths, cu_seqlens)
        expected = pack_encoder_output(expected.transpose(1, 2), expected_lengths)

        torch.testing.assert_close(actual.data, expected.data, rtol=1e-5, atol=2e-6)

    @pytest.mark.unit
    def test_AudioToMelSpectrogramPreprocessor_config(self):
        # Test that dataclass matches signature of module
        result = config_utils.assert_dataclass_signature_match(
            modules.AudioToMelSpectrogramPreprocessor,
            modules.audio_preprocessing.AudioToMelSpectrogramPreprocessorConfig,
        )
        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None

    @pytest.mark.unit
    def test_AudioToMelSpectrogramPreprocessor_batch(self):
        # Test 1 that should test the pure stft implementation as much as possible
        instance1 = modules.AudioToMelSpectrogramPreprocessor(normalize="per_feature", dither=0, pad_to=0)

        # Ensure that the two functions behave similarily
        for _ in range(10):
            input_signal, length = instance1.input_example(4, 512, 321)

            with torch.no_grad():
                # batch size 1
                res_instance, length_instance = [], []
                for i in range(input_signal.size(0)):
                    res_ins, length_ins = instance1(input_signal=input_signal[i : i + 1], length=length[i : i + 1])
                    res_instance.append(res_ins)
                    length_instance.append(length_ins)

                res_instance = torch.cat(res_instance, 0)
                length_instance = torch.cat(length_instance, 0)

                # batch size 4
                res_batch, length_batch = instance1(input_signal=input_signal, length=length)

            assert res_instance.shape == res_batch.shape
            assert length_instance.shape == length_batch.shape
            diff = torch.mean(torch.abs(res_instance - res_batch))
            assert diff <= 1e-3
            diff = torch.max(torch.abs(res_instance - res_batch))
            assert diff <= 1e-3

    @pytest.mark.run_only_on('GPU')
    def test_AudioToMelSpectrogramPreprocessor_gpu(self):
        instance0 = modules.AudioToMelSpectrogramPreprocessor().to("cuda")
        input_signal, length = instance0.input_example()

        with torch.no_grad():
            processed_signal, _ = instance0(input_signal=input_signal, length=length)

        assert processed_signal.device == input_signal.device

    @pytest.mark.unit
    def test_SpectrogramAugmentationr_legacy(self):
        # Make sure constructor works
        instance1 = modules.SpectrogramAugmentation(
            freq_masks=10, time_masks=3, rect_masks=3, use_numba_spec_augment=False, use_vectorized_spec_augment=False
        )
        assert isinstance(instance1, modules.SpectrogramAugmentation)

        # Make sure forward doesn't throw with expected input
        instance0 = modules.AudioToMelSpectrogramPreprocessor(dither=0)
        input_signal, length = instance0.input_example(4, 512, 321)
        res0 = instance0(input_signal=input_signal, length=length)
        res = instance1(input_spec=res0[0], length=length)

        assert res.shape == res0[0].shape

    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    def test_SpectrogramAugmentationr_vectorized(self):
        # Make sure constructor works
        instance1 = modules.SpectrogramAugmentation(
            freq_masks=10, time_masks=3, rect_masks=3, use_numba_spec_augment=False, use_vectorized_spec_augment=True
        )
        assert isinstance(instance1, modules.SpectrogramAugmentation)

        # Make sure forward doesn't throw with expected input
        instance0 = modules.AudioToMelSpectrogramPreprocessor(dither=0)
        input_signal, length = instance0.input_example(4, 512, 321)
        res0 = instance0(input_signal=input_signal, length=length)
        res = instance1(input_spec=res0[0], length=length)

        assert res.shape == res0[0].shape

    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    def test_SpectrogramAugmentationr_numba_kernel(self, caplog):
        numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        logging._logger.propagate = True
        original_verbosity = logging.get_verbosity()
        logging.set_verbosity(logging.DEBUG)
        caplog.set_level(logging.DEBUG)

        # Make sure constructor works
        instance1 = modules.SpectrogramAugmentation(
            freq_masks=10, time_masks=3, rect_masks=3, use_numba_spec_augment=True, use_vectorized_spec_augment=False
        )
        assert isinstance(instance1, modules.SpectrogramAugmentation)

        # Make sure forward doesn't throw with expected input
        instance0 = modules.AudioToMelSpectrogramPreprocessor(dither=0)
        input_signal, length = instance0.input_example(8, 512, 321)
        res0 = instance0(input_signal=input_signal, length=length)
        res = instance1(input_spec=res0[0], length=length)

        assert res.shape == res0[0].shape

        # check tha numba kernel debug message indicates that it is available for use
        assert """Numba SpecAugment kernel is available""" in caplog.text

        logging._logger.propagate = False
        logging.set_verbosity(original_verbosity)

    @pytest.mark.unit
    def test_SpectrogramAugmentationr_config(self):
        # Test that dataclass matches signature of module
        result = config_utils.assert_dataclass_signature_match(
            modules.SpectrogramAugmentation,
            modules.audio_preprocessing.SpectrogramAugmentationConfig,
        )
        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None

    @pytest.mark.unit
    def test_CropOrPadSpectrogramAugmentation(self):
        # Make sure constructor works
        audio_length = 128
        instance1 = modules.CropOrPadSpectrogramAugmentation(audio_length=audio_length)
        assert isinstance(instance1, modules.CropOrPadSpectrogramAugmentation)

        # Make sure forward doesn't throw with expected input
        instance0 = modules.AudioToMelSpectrogramPreprocessor(dither=0)
        input_signal, length = instance0.input_example(4, 512, 321)
        res0 = instance0(input_signal=input_signal, length=length)
        res, new_length = instance1(input_signal=res0[0], length=length)

        assert res.shape == torch.Size([4, 64, audio_length])
        assert all(new_length == torch.tensor([128] * 4))

    @pytest.mark.unit
    def test_CropOrPadSpectrogramAugmentation_config(self):
        # Test that dataclass matches signature of module
        result = config_utils.assert_dataclass_signature_match(
            modules.CropOrPadSpectrogramAugmentation,
            modules.audio_preprocessing.CropOrPadSpectrogramAugmentationConfig,
        )
        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None

    @pytest.mark.unit
    def test_MaskedPatchAugmentation(self):
        # Make sure constructor works
        audio_length = 128
        instance1 = modules.MaskedPatchAugmentation(patch_size=16, mask_patches=0.5, freq_masks=2, freq_width=10)
        assert isinstance(instance1, modules.MaskedPatchAugmentation)

        # Make sure forward doesn't throw with expected input
        instance0 = modules.AudioToMelSpectrogramPreprocessor(dither=0)
        input_signal, length = instance0.input_example(4, 512, 321)
        res0 = instance0(input_signal=input_signal, length=length)
        res = instance1(input_spec=res0[0], length=length)

        assert res.shape == res0[0].shape

    @pytest.mark.unit
    def test_MaskedPatchAugmentation_config(self):
        # Test that dataclass matches signature of module
        result = config_utils.assert_dataclass_signature_match(
            modules.MaskedPatchAugmentation,
            modules.audio_preprocessing.MaskedPatchAugmentationConfig,
        )
        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None

    @pytest.mark.unit
    def test_RNNTDecoder(self):
        vocab = list(range(10))
        vocab = [str(x) for x in vocab]
        vocab_size = len(vocab)

        pred_config = OmegaConf.create(
            {
                '_target_': 'nemo.collections.asr.modules.RNNTDecoder',
                'prednet': {
                    'pred_hidden': 32,
                    'pred_rnn_layers': 1,
                },
                'vocab_size': vocab_size,
                'blank_as_pad': True,
            }
        )

        prednet = modules.RNNTDecoder.from_config_dict(pred_config)

        # num params
        pred_hidden = pred_config.prednet.pred_hidden
        embed = (vocab_size + 1) * pred_hidden  # embedding with blank
        rnn = (
            2 * 4 * (pred_hidden * pred_hidden + pred_hidden)
        )  # (ih + hh) * (ifco gates) * (indim * hiddendim + bias)
        assert prednet.num_weights == (embed + rnn)

        # State initialization
        x_ = torch.zeros(4, dtype=torch.float32)
        states = prednet.initialize_state(x_)

        for state_i in states:
            assert state_i.dtype == x_.dtype
            assert state_i.device == x_.device
            assert state_i.shape[1] == len(x_)

        # Blank hypotheses test
        blank = vocab_size
        hyp = Hypothesis(score=0.0, y_sequence=[blank])
        cache = {}
        pred, states, _ = prednet.score_hypothesis(hyp, cache)

        assert pred.shape == torch.Size([1, 1, pred_hidden])
        assert len(states) == 2
        for state_i in states:
            assert state_i.dtype == pred.dtype
            assert state_i.device == pred.device
            assert state_i.shape[1] == len(pred)

        # Blank stateless predict
        g, states = prednet.predict(y=None, state=None, add_sos=False, batch_size=1)

        assert g.shape == torch.Size([1, 1, pred_hidden])
        assert len(states) == 2
        for state_i in states:
            assert state_i.dtype == g.dtype
            assert state_i.device == g.device
            assert state_i.shape[1] == len(g)

        # Blank stateful predict
        g, states2 = prednet.predict(y=None, state=states, add_sos=False, batch_size=1)

        assert g.shape == torch.Size([1, 1, pred_hidden])
        assert len(states2) == 2
        for state_i, state_j in zip(states, states2):
            assert (state_i - state_j).square().sum().sqrt() > 0.0

        # Predict with token and state
        token = torch.full([1, 1], fill_value=0, dtype=torch.long)
        g, states = prednet.predict(y=token, state=states2, add_sos=False, batch_size=None)

        assert g.shape == torch.Size([1, 1, pred_hidden])
        assert len(states) == 2

        # Predict with blank token and no state
        token = torch.full([1, 1], fill_value=blank, dtype=torch.long)
        g, states = prednet.predict(y=token, state=None, add_sos=False, batch_size=None)

        assert g.shape == torch.Size([1, 1, pred_hidden])
        assert len(states) == 2

    @pytest.mark.unit
    def test_RNNTJoint(self):
        vocab = list(range(10))
        vocab = [str(x) for x in vocab]
        vocab_size = len(vocab)

        batchsize = 4
        encoder_hidden = 64
        pred_hidden = 32
        joint_hidden = 16

        joint_cfg = OmegaConf.create(
            {
                '_target_': 'nemo.collections.asr.modules.RNNTJoint',
                'num_classes': vocab_size,
                'vocabulary': vocab,
                'jointnet': {
                    'encoder_hidden': encoder_hidden,
                    'pred_hidden': pred_hidden,
                    'joint_hidden': joint_hidden,
                    'activation': 'relu',
                },
            }
        )

        jointnet = modules.RNNTJoint.from_config_dict(joint_cfg)

        enc = torch.zeros(batchsize, encoder_hidden, 48)  # [B, D1, T]
        dec = torch.zeros(batchsize, pred_hidden, 24)  # [B, D2, U]

        # forward call test
        out = jointnet(encoder_outputs=enc, decoder_outputs=dec)
        assert out.shape == torch.Size([batchsize, 48, 24, vocab_size + 1])  # [B, T, U, V + 1]

        # joint() step test
        enc2 = enc.transpose(1, 2)  # [B, T, D1]
        dec2 = dec.transpose(1, 2)  # [B, U, D2]
        out2 = jointnet.joint(enc2, dec2)  # [B, T, U, V + 1]
        assert (out - out2).abs().sum() <= 1e-5

        # assert vocab size
        assert jointnet.num_classes_with_blank == vocab_size + 1

    @pytest.mark.unit
    def test_HATJoint(self):
        vocab = list(range(10))
        vocab = [str(x) for x in vocab]
        vocab_size = len(vocab)

        batchsize = 4
        encoder_hidden = 64
        pred_hidden = 32
        joint_hidden = 16

        joint_cfg = OmegaConf.create(
            {
                '_target_': 'nemo.collections.asr.modules.HATJoint',
                'num_classes': vocab_size,
                'vocabulary': vocab,
                'jointnet': {
                    'encoder_hidden': encoder_hidden,
                    'pred_hidden': pred_hidden,
                    'joint_hidden': joint_hidden,
                    'activation': 'relu',
                },
            }
        )

        jointnet = modules.HATJoint.from_config_dict(joint_cfg)

        enc = torch.zeros(batchsize, encoder_hidden, 48)  # [B, D1, T]
        dec = torch.zeros(batchsize, pred_hidden, 24)  # [B, D2, U]

        # forward call test
        out = jointnet(encoder_outputs=enc, decoder_outputs=dec)
        assert out.shape == torch.Size([batchsize, 48, 24, vocab_size + 1])  # [B, T, U, V + 1]

        # joint() step test
        enc2 = enc.transpose(1, 2)  # [B, T, D1]
        dec2 = dec.transpose(1, 2)  # [B, U, D2]
        out2 = jointnet.joint(enc2, dec2)  # [B, T, U, V + 1]
        assert (out - out2).abs().sum() <= 1e-5

        # joint() step test for internal LM subtraction
        jointnet.return_hat_ilm = True
        hat_output = jointnet.joint(enc2, dec2)  # HATJointOutput dataclass
        out3, ilm = hat_output.hat_logprobs, hat_output.ilm_logprobs  # [B, T, U, V + 1] and [B, 1, U, V]
        assert (out - out3).abs().sum() <= 1e-5
        assert ilm.shape == torch.Size([batchsize, 1, 24, vocab_size])  # [B, 1, U, V] without blank simbol

        # assert vocab size
        assert jointnet.num_classes_with_blank == vocab_size + 1
