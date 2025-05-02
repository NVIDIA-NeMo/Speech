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

import importlib
from typing import Optional

import numpy as np
import pytest
import torch

from nemo.collections.audio.modules.features import SpectrogramToMultichannelFeatures
from nemo.collections.audio.modules.masking import (
    MaskBasedDereverbWPE,
    MaskEstimatorFlexChannels,
    MaskEstimatorGSS,
    MaskReferenceChannel,
)
from nemo.collections.audio.modules.ssl_pretrain_masking import SSLPretrainWithMaskedPatch
from nemo.collections.audio.modules.transforms import AudioToSpectrogram
from nemo.collections.audio.parts.submodules.multichannel import WPEFilter
from nemo.collections.audio.parts.utils.audio import convmtx_mc_numpy
from nemo.utils import logging

try:
    importlib.import_module('torchaudio')

    HAVE_TORCHAUDIO = True
except ModuleNotFoundError:
    HAVE_TORCHAUDIO = False


class TestSpectrogramToMultichannelFeatures:
    @pytest.mark.unit
    @pytest.mark.parametrize('fft_length', [128])
    @pytest.mark.parametrize('num_channels', [1, 3])
    @pytest.mark.parametrize('mag_reduction', [None, 'rms', 'abs_mean', 'mean_abs'])
    @pytest.mark.parametrize('mag_power', [None, 2])
    @pytest.mark.parametrize('mag_normalization', [None, 'mean', 'mean_var'])
    def test_magnitude(
        self,
        fft_length: int,
        num_channels: int,
        mag_reduction: Optional[str],
        mag_power: Optional[float],
        mag_normalization: Optional[str],
    ):
        """Test calculation of spatial features for multi-channel audio."""
        atol = 5e-5
        batch_size = 8
        num_samples = fft_length * 50
        num_examples = 10
        random_seed = 42

        _rng = np.random.default_rng(seed=random_seed)

        hop_length = fft_length // 4
        audio2spec = AudioToSpectrogram(fft_length=fft_length, hop_length=hop_length)

        spec2feat = SpectrogramToMultichannelFeatures(
            num_subbands=audio2spec.num_subbands,
            mag_reduction=mag_reduction,
            mag_power=mag_power,
            mag_normalization=mag_normalization,
            use_ipd=False,
        )

        for n in range(num_examples):
            x = _rng.normal(size=(batch_size, num_channels, num_samples))

            # convert to spectrogram
            spec, spec_len = audio2spec(input=torch.Tensor(x), input_length=torch.Tensor([num_samples] * batch_size))

            # UUT output
            feat, _ = spec2feat(input=spec, input_length=spec_len)
            feat_np = feat.cpu().detach().numpy()

            # Golden output
            spec_np = spec.cpu().detach().numpy()
            if mag_reduction is None:
                feat_golden = np.abs(spec_np)
            elif mag_reduction == 'rms':
                feat_golden = np.sqrt(np.mean(np.abs(spec_np) ** 2, axis=1, keepdims=True))
            elif mag_reduction == 'mean_abs':
                feat_golden = np.mean(np.abs(spec_np), axis=1, keepdims=True)
            elif mag_reduction == 'abs_mean':
                feat_golden = np.abs(np.mean(spec_np, axis=1, keepdims=True))
            else:
                raise NotImplementedError(f'Magnitude reduction {mag_reduction} not implemented')

            if mag_power is not None:
                feat_golden = np.power(feat_golden, mag_power)

            if mag_normalization == 'mean':
                feat_golden = feat_golden - np.mean(feat_golden, axis=(1, 3), keepdims=True)
            elif mag_normalization == 'mean_var':
                feat_golden = feat_golden - np.mean(feat_golden, axis=(1, 3), keepdims=True)
                feat_golden = feat_golden / np.sqrt(np.mean(feat_golden**2, axis=(1, 3), keepdims=True))

            # Compare shape
            assert feat_np.shape == feat_golden.shape, f'Feature shape not matching for example {n}'

            # Compare values
            assert np.allclose(feat_np, feat_golden, atol=atol), f'Features not matching for example {n}'

    @pytest.mark.unit
    @pytest.mark.parametrize('fft_length', [128])
    @pytest.mark.parametrize('num_channels', [1, 3])
    @pytest.mark.parametrize('ipd_normalization', [None, 'mean', 'mean_var'])
    @pytest.mark.parametrize('use_input_length', [True, False])
    def test_ipd(self, fft_length: int, num_channels: int, ipd_normalization: Optional[str], use_input_length: bool):
        """Test calculation of IPD spatial features for multi-channel audio."""
        atol = 5e-5
        batch_size = 8
        num_samples = fft_length * 50
        num_examples = 10
        random_seed = 42

        _rng = np.random.default_rng(seed=random_seed)

        hop_length = fft_length // 4
        audio2spec = AudioToSpectrogram(fft_length=fft_length, hop_length=hop_length)

        spec2feat = SpectrogramToMultichannelFeatures(
            num_subbands=audio2spec.num_subbands,
            mag_reduction='rms',
            use_ipd=True,
            mag_normalization=None,
            ipd_normalization=ipd_normalization,
        )

        for n in range(num_examples):
            x = _rng.normal(size=(batch_size, num_channels, num_samples))

            spec, spec_len = audio2spec(input=torch.Tensor(x), input_length=torch.Tensor([num_samples] * batch_size))

            # UUT output
            feat, _ = spec2feat(input=spec, input_length=spec_len if use_input_length else None)
            feat_np = feat.cpu().detach().numpy()
            ipd = feat_np[..., audio2spec.num_subbands :, :]

            # Golden output
            spec_np = spec.cpu().detach().numpy()
            spec_mean = np.mean(spec_np, axis=1, keepdims=True)
            ipd_golden = np.angle(spec_np) - np.angle(spec_mean)
            ipd_golden = np.remainder(ipd_golden + np.pi, 2 * np.pi) - np.pi

            if ipd_normalization == 'mean':
                ipd_golden = ipd_golden - np.mean(ipd_golden, axis=(1, 3), keepdims=True)
            elif ipd_normalization == 'mean_var':
                ipd_golden = ipd_golden - np.mean(ipd_golden, axis=(1, 3), keepdims=True)
                ipd_golden = ipd_golden / np.sqrt(
                    np.maximum(np.mean(ipd_golden**2, axis=(1, 3), keepdims=True), spec2feat.eps)
                )

            # Compare shape
            assert ipd.shape == ipd_golden.shape, f'Feature shape not matching for example {n}'

            # Compare values
            assert np.allclose(ipd, ipd_golden, atol=atol), f'Features not matching for example {n}'

    @pytest.mark.unit
    @pytest.mark.parametrize('use_ipd', [False, True])
    def test_num_channels(self, use_ipd: bool):
        """Test num channels property."""
        uut = SpectrogramToMultichannelFeatures(num_subbands=32, use_ipd=use_ipd)
        with pytest.raises(ValueError):
            # num_input_channels is not set
            uut.num_channels

        for num_channels in [1, 2, 3, 4]:
            # num_input_channels is set
            uut = SpectrogramToMultichannelFeatures(num_subbands=32, num_input_channels=num_channels, use_ipd=use_ipd)
            assert uut.num_channels == num_channels

        for num_channels in [1, 2, 3, 4]:
            # num_input_channels is set, but magnitude will be reduced
            uut = SpectrogramToMultichannelFeatures(
                num_subbands=32, num_input_channels=num_channels, use_ipd=use_ipd, mag_reduction='rms'
            )
            if use_ipd:
                assert uut.num_channels == num_channels
            else:
                assert uut.num_channels == 1

    @pytest.mark.unit
    @pytest.mark.parametrize('use_ipd', [False, True])
    def test_num_features(self, use_ipd: bool):
        """Test num features property."""
        for num_subbands in [5, 10]:
            uut = SpectrogramToMultichannelFeatures(num_subbands=num_subbands, use_ipd=use_ipd)
            assert uut.num_features == 2 * num_subbands if use_ipd else num_subbands

    @pytest.mark.unit
    def test_unsupported_norm(self):
        """Test initialization with unsupported normalization."""
        # test magnitude normalization
        with pytest.raises(NotImplementedError):
            SpectrogramToMultichannelFeatures(
                num_subbands=32,
                mag_reduction='rms',
                use_ipd=False,
                mag_normalization='not-implemented',
            )
        # test phase normalization
        with pytest.raises(NotImplementedError):
            SpectrogramToMultichannelFeatures(
                num_subbands=32,
                use_ipd=True,
                ipd_normalization='not-implemented',
            )
        # test magnitude reduction
        uut = SpectrogramToMultichannelFeatures(
            num_subbands=32,
            mag_reduction='not-implemented',
        )
        input = torch.randn(1, 3, 100, 100)
        with pytest.raises(ValueError):
            uut(input=input, input_length=torch.Tensor([100]))


class TestMaskBasedProcessor:
    @pytest.mark.unit
    @pytest.mark.skipif(not HAVE_TORCHAUDIO, reason="Modules in this test require torchaudio")
    @pytest.mark.parametrize('fft_length', [256])
    @pytest.mark.parametrize('num_channels', [1, 4])
    @pytest.mark.parametrize('num_masks', [1, 2])
    def test_mask_reference_channel(self, fft_length: int, num_channels: int, num_masks: int):
        """Test masking of the reference channel."""
        if num_channels == 1:
            # Only one channel available
            ref_channels = [0]
        else:
            # Use first or last channel for MC signals
            ref_channels = [0, num_channels - 1]

        atol = 1e-6
        batch_size = 8
        num_samples = fft_length * 50
        num_examples = 10
        random_seed = 42

        _rng = np.random.default_rng(seed=random_seed)

        hop_length = fft_length // 4
        audio2spec = AudioToSpectrogram(fft_length=fft_length, hop_length=hop_length)

        for ref_channel in ref_channels:

            mask_processor = MaskReferenceChannel(ref_channel=ref_channel)

            for n in range(num_examples):
                x = _rng.normal(size=(batch_size, num_channels, num_samples))

                spec, spec_len = audio2spec(
                    input=torch.Tensor(x), input_length=torch.Tensor([num_samples] * batch_size)
                )

                # Randomly-generated mask
                mask = _rng.uniform(
                    low=0.0, high=1.0, size=(batch_size, num_masks, audio2spec.num_subbands, spec.shape[-1])
                )

                # UUT output
                out, _ = mask_processor(input=spec, input_length=spec_len, mask=torch.tensor(mask))
                out_np = out.cpu().detach().numpy()

                # Golden output
                spec_np = spec.cpu().detach().numpy()
                out_golden = np.zeros_like(mask, dtype=spec_np.dtype)
                for m in range(num_masks):
                    out_golden[:, m, ...] = spec_np[:, ref_channel, ...] * mask[:, m, ...]

                # Compare shape
                assert out_np.shape == out_golden.shape, f'Output shape not matching for example {n}'

                # Compare values
                assert np.allclose(out_np, out_golden, atol=atol), f'Output not matching for example {n}'


class TestMaskBasedDereverb:
    @pytest.mark.unit
    @pytest.mark.parametrize('num_channels', [1, 3])
    @pytest.mark.parametrize('filter_length', [10])
    @pytest.mark.parametrize('delay', [0, 5])
    def test_wpe_convtensor(self, num_channels: int, filter_length: int, delay: int):
        """Test construction of convolutional tensor in WPE. Compare against
        reference implementation convmtx_mc.
        """
        atol = 1e-6
        random_seed = 42
        num_examples = 10
        batch_size = 8
        num_subbands = 15
        num_frames = 21

        _rng = np.random.default_rng(seed=random_seed)
        input_size = (batch_size, num_channels, num_subbands, num_frames)

        for n in range(num_examples):
            X = _rng.normal(size=input_size) + 1j * _rng.normal(size=input_size)

            # Reference
            tilde_X_ref = np.zeros((batch_size, num_subbands, num_frames, num_channels * filter_length), dtype=X.dtype)
            for b in range(batch_size):
                for f in range(num_subbands):
                    tilde_X_ref[b, f, :, :] = convmtx_mc_numpy(
                        X[b, :, f, :].transpose(), filter_length=filter_length, delay=delay
                    )

            # UUT
            tilde_X_uut = WPEFilter.convtensor(torch.tensor(X), filter_length=filter_length, delay=delay)

            # UUT has vectors arranged in a tensor shape with permuted columns
            # Reorganize to match the shape and column permutation
            tilde_X_uut = WPEFilter.permute_convtensor(tilde_X_uut)
            tilde_X_uut = tilde_X_uut.cpu().detach().numpy()

            assert np.allclose(tilde_X_uut, tilde_X_ref, atol=atol), f'Example {n}: comparison failed'

    @pytest.mark.unit
    @pytest.mark.parametrize('num_channels', [1, 3])
    @pytest.mark.parametrize('filter_length', [10])
    @pytest.mark.parametrize('delay', [0, 5])
    def test_wpe_filter(self, num_channels: int, filter_length: int, delay: int):
        """Test estimation of correlation matrices, filter and filtering."""
        atol = 1e-6
        random_seed = 42
        num_examples = 10
        batch_size = 4
        num_subbands = 15
        num_frames = 50

        wpe_filter = WPEFilter(filter_length=filter_length, prediction_delay=delay, diag_reg=None)

        _rng = np.random.default_rng(seed=random_seed)
        input_size = (batch_size, num_channels, num_subbands, num_frames)

        for n in range(num_examples):
            X = torch.tensor(_rng.normal(size=input_size) + 1j * _rng.normal(size=input_size))
            weight = torch.tensor(_rng.uniform(size=(batch_size, num_subbands, num_frames)))

            # Create convtensor (B, C, F, N, filter_length)
            tilde_X = wpe_filter.convtensor(X, filter_length=filter_length, delay=delay)

            # Test 1:
            # estimate_correlation

            # Reference
            # move channels to back
            X_golden = X.permute(0, 2, 3, 1)
            # move channels to back and reshape to (B, F, N, C*filter_length)
            tilde_X_golden = tilde_X.permute(0, 2, 3, 1, 4).reshape(
                batch_size, num_subbands, num_frames, num_channels * filter_length
            )
            # (B, F, C * filter_length, C * filter_length)
            Q_golden = torch.matmul(tilde_X_golden.transpose(-1, -2).conj(), weight[..., None] * tilde_X_golden)
            # (B, F, C * filter_length, C)
            R_golden = torch.matmul(tilde_X_golden.transpose(-1, -2).conj(), weight[..., None] * X_golden)

            # UUT
            Q_uut, R_uut = wpe_filter.estimate_correlations(input=X, weight=weight, tilde_input=tilde_X)
            # Flatten (B, F, C, filter_length, C, filter_length) into (B, F, C*filter_length, C*filter_length)
            Q_uut_flattened = Q_uut.flatten(start_dim=-2, end_dim=-1).flatten(start_dim=-3, end_dim=-2)
            # Flatten (B, F, C, filter_length, C, filter_length) into (B, F, C*filter_length, C*filter_length)
            R_uut_flattened = R_uut.flatten(start_dim=-3, end_dim=-2)

            assert torch.allclose(Q_uut_flattened, Q_golden, atol=atol), f'Example {n}: comparison failed for Q'
            assert torch.allclose(R_uut_flattened, R_golden, atol=atol), f'Example {n}: comparison failed for R'

            # Test 2:
            # estimate_filter

            # Reference
            G_golden = torch.linalg.solve(Q_golden, R_golden)

            # UUT
            G_uut = wpe_filter.estimate_filter(Q_uut, R_uut)
            # Flatten and move output channels to back
            G_uut_flattened = G_uut.reshape(batch_size, num_channels, num_subbands, -1).permute(0, 2, 3, 1)

            assert torch.allclose(G_uut_flattened, G_golden, atol=atol), f'Example {n}: comparison failed for G'

            # Test 3:
            # apply_filter

            # Reference
            U_golden = torch.matmul(tilde_X_golden, G_golden)

            # UUT
            U_uut = wpe_filter.apply_filter(filter=G_uut, tilde_input=tilde_X)
            U_uut_ref = U_uut.permute(0, 2, 3, 1)

            assert torch.allclose(
                U_uut_ref, U_golden, atol=atol
            ), f'Example {n}: comparison failed for undesired output U'

    @pytest.mark.unit
    @pytest.mark.parametrize('num_channels', [3])
    @pytest.mark.parametrize('filter_length', [5])
    @pytest.mark.parametrize('delay', [0, 2])
    def test_mask_based_dereverb_init(self, num_channels: int, filter_length: int, delay: int):
        """Test that dereverb can be initialized and can process audio."""
        num_examples = 10
        batch_size = 8
        num_subbands = 15
        num_frames = 21
        num_iterations = 2

        input_size = (batch_size, num_subbands, num_frames, num_channels)

        dereverb = MaskBasedDereverbWPE(
            filter_length=filter_length, prediction_delay=delay, num_iterations=num_iterations
        )

        for n in range(num_examples):
            # multi-channel input
            x = torch.randn(input_size) + 1j * torch.randn(input_size)
            # random input_length
            x_length = torch.randint(1, num_frames, (batch_size,))
            # multi-channel mask
            mask = torch.rand(input_size)

            # UUT
            y, y_length = dereverb(input=x, input_length=x_length, mask=mask)

            assert y.shape == x.shape, 'Output shape not matching, example {n}'
            assert torch.equal(y_length, x_length), 'Length not matching, example {n}'


class TestMaskEstimator:
    @pytest.mark.unit
    @pytest.mark.skipif(not HAVE_TORCHAUDIO, reason="Modules in this test require torchaudio")
    @pytest.mark.parametrize('channel_reduction_position', [0, 1, -1])
    @pytest.mark.parametrize('channel_reduction_type', ['average', 'attention'])
    @pytest.mark.parametrize('channel_block_type', ['transform_average_concatenate', 'transform_attend_concatenate'])
    def test_flex_channels(
        self, channel_reduction_position: int, channel_reduction_type: str, channel_block_type: str
    ):
        """Test initialization of the mask estimator and make sure it can process input tensor."""
        # Model parameters
        num_subbands_tests = [32, 65]
        num_outputs_tests = [1, 2]
        num_blocks_tests = [1, 5]

        # Input configuration
        num_channels_tests = [1, 4]
        batch_size = 4
        num_frames = 50

        for num_subbands in num_subbands_tests:
            for num_outputs in num_outputs_tests:
                for num_blocks in num_blocks_tests:
                    logging.debug(
                        'Instantiate with num_subbands=%d, num_outputs=%d, num_blocks=%d',
                        num_subbands,
                        num_outputs,
                        num_blocks,
                    )

                    # Instantiate
                    uut = MaskEstimatorFlexChannels(
                        num_outputs=num_outputs,
                        num_subbands=num_subbands,
                        num_blocks=num_blocks,
                        channel_reduction_position=channel_reduction_position,
                        channel_reduction_type=channel_reduction_type,
                        channel_block_type=channel_block_type,
                    )

                    # Process different channel configurations
                    for num_channels in num_channels_tests:
                        logging.debug('Process num_channels=%d', num_channels)
                        input_size = (batch_size, num_channels, num_subbands, num_frames)

                        # multi-channel input
                        spec = torch.randn(input_size, dtype=torch.cfloat)
                        spec_length = torch.randint(1, num_frames, (batch_size,))

                        # UUT
                        mask, mask_length = uut(input=spec, input_length=spec_length)

                        # Check output dimensions match
                        expected_mask_shape = (batch_size, num_outputs, num_subbands, num_frames)
                        assert (
                            mask.shape == expected_mask_shape
                        ), f'Output shape mismatch: expected {expected_mask_shape}, got {mask.shape}'

                        # Check output lengths match
                        assert torch.all(
                            mask_length == spec_length
                        ), f'Output length mismatch: expected {spec_length}, got {mask_length}'

    @pytest.mark.unit
    @pytest.mark.parametrize('num_channels', [1, 4])
    @pytest.mark.parametrize('num_subbands', [32, 65])
    @pytest.mark.parametrize('num_outputs', [2, 3])
    @pytest.mark.parametrize('batch_size', [1, 4])
    def test_gss(self, num_channels: int, num_subbands: int, num_outputs: int, batch_size: int):
        """Test initialization of the GSS mask estimator and make sure it can process an input tensor.
        This tests initialization and the output shape. It does not test correctness of the output.
        """
        # Test vector length
        num_frames = 50

        # Instantiate UUT
        uut = MaskEstimatorGSS()

        # Process the current configuration
        logging.debug('Process num_channels=%d', num_channels)
        input_size = (batch_size, num_channels, num_subbands, num_frames)
        logging.debug('Input size: %s', input_size)

        # multi-channel input
        mixture_spec = torch.randn(input_size, dtype=torch.cfloat)
        source_activity = torch.randn(batch_size, num_outputs, num_frames) > 0

        # UUT
        mask = uut(input=mixture_spec, activity=source_activity)

        # Check output dimensions match
        expected_mask_shape = (batch_size, num_outputs, num_subbands, num_frames)
        assert (
            mask.shape == expected_mask_shape
        ), f'Output shape mismatch: expected {expected_mask_shape}, got {mask.shape}'


class TestSSLPretrainMaskingWithPatch:
    @pytest.mark.unit
    @pytest.mark.parametrize('patch_size', [1, 5, 10])
    @pytest.mark.parametrize('mask_fraction', [0.5, 1.0])
    @pytest.mark.parametrize('training', [True, False])
    def test_masking(self, patch_size: int, mask_fraction: float, training: bool):
        """Test SSL pretrain masking."""
        num_subbands = 32
        num_frames = 5000
        num_channels = 1
        batch_size = 8
        abs_tol = 1e-2

        # Instantiate
        uut = SSLPretrainWithMaskedPatch(patch_size=patch_size, mask_fraction=mask_fraction)

        # Set training mode
        if training:
            uut.train()
        else:
            uut.eval()

        # Generate random input spec and length
        rng = torch.Generator()
        rng.manual_seed(0)
        input_spec = torch.randn(batch_size, num_channels, num_subbands, num_frames, dtype=torch.cfloat, generator=rng)
        input_length = torch.randint(num_frames // 2, num_frames, (batch_size,), generator=rng)
        for b in range(batch_size):
            input_spec[b, :, :, input_length[b] :] = 0.0

        # Apply masking
        masked_spec = uut(input_spec=input_spec, length=input_length)

        # Check output dimensions match
        assert masked_spec.shape == input_spec.shape

        # Check output values are masked for each example in the batch
        for b in range(batch_size):
            # Estimate mask fraction
            est_mask_fraction = torch.sum(masked_spec[b, :, :, : input_length[b]].abs() == 0.0) / (
                num_channels * num_subbands * input_length[b]
            )

            # Check if the estimated mask fraction is close to the expected mask fraction
            assert (
                abs(est_mask_fraction - mask_fraction) < abs_tol
            ), f'Example {b}: est_mask_fraction = {est_mask_fraction}, mask_fraction = {mask_fraction}'

    @pytest.mark.unit
    def test_unsupported_initialization(self):
        """Test SSL pretrain masking."""
        with pytest.raises(ValueError):
            SSLPretrainWithMaskedPatch(patch_size=0)
        with pytest.raises(ValueError):
            SSLPretrainWithMaskedPatch(patch_size=-1)
        with pytest.raises(ValueError):
            SSLPretrainWithMaskedPatch(mask_fraction=1.1)
        with pytest.raises(ValueError):
            SSLPretrainWithMaskedPatch(mask_fraction=-0.1)
