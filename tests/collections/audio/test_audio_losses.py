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

import os

import numpy as np
import pytest
import torch

from nemo.collections.audio.losses.audio import (
    MAELoss,
    MSELoss,
    SDRLoss,
    calculate_mae_batch,
    calculate_mean,
    calculate_mse_batch,
    calculate_sdr_batch,
    convolution_invariant_target,
    scale_invariant_target,
)
from nemo.collections.audio.losses.maxine import CombinedLoss
from nemo.collections.audio.parts.utils.audio import (
    calculate_sdr_numpy,
    convolution_invariant_target_numpy,
    scale_invariant_target_numpy,
)


class TestAudioLosses:
    @pytest.mark.unit
    @pytest.mark.parametrize('num_channels', [1, 4])
    @pytest.mark.parametrize('use_mask', [True, False])
    @pytest.mark.parametrize('use_input_length', [True, False])
    def test_calculate_mean(self, num_channels: int, use_mask: bool, use_input_length: bool):
        """Test mean calculation"""
        batch_size = 8
        num_samples = 50
        num_batches = 10
        random_seed = 42
        eps = 1e-10
        atol = 1e-6

        rng = torch.Generator()
        rng.manual_seed(random_seed)

        for n in range(num_batches):

            input_signal = torch.randn(size=(batch_size, num_channels, num_samples), generator=rng)
            # Random input length
            input_length = torch.randint(low=1, high=num_samples, size=(batch_size,), generator=rng)
            # Corresponding mask
            mask = torch.zeros(size=(batch_size, 1, num_samples))
            for i in range(batch_size):
                mask[i, :, : input_length[i]] = 1.0

            if use_mask and use_input_length:
                with pytest.raises(RuntimeError):
                    calculate_mean(input_signal, mask=mask, input_length=input_length, eps=eps)
                # Done with this test
                continue

            if use_mask:
                uut = calculate_mean(input_signal, mask=mask, eps=eps)
            elif use_input_length:
                uut = calculate_mean(input_signal, input_length=input_length, eps=eps)
            else:
                uut = calculate_mean(input_signal, eps=eps)

            # Calculate mean manually
            for b in range(batch_size):
                for c in range(num_channels):
                    golden = torch.mean(
                        input_signal[b, c, : input_length[b] if use_input_length or use_mask else num_samples]
                    )
                    assert torch.allclose(
                        uut[b, c], golden, atol=atol
                    ), f"Mean not matching for example {n}, channel {c}"

    @pytest.mark.unit
    def test_calculate_sdr_scale_and_convolution_invariant(self):
        """Test SDR calculation with scale and conovolution invariant options."""
        estimate = torch.randn(size=(1, 1, 100))
        target = torch.randn(size=(1, 1, 100))

        with pytest.raises(ValueError):
            # using both scale and convolution invariant is not allowed
            calculate_sdr_batch(estimate=estimate, target=target, scale_invariant=True, convolution_invariant=True)

    @pytest.mark.unit
    def test_calculate_mse_input_and_mask(self):
        """Test MSE calculation with simultaneous input length and mask."""
        estimate = torch.randn(size=(1, 1, 100))
        target = torch.randn(size=(1, 1, 100))
        input_length = torch.tensor([100])
        mask = torch.ones(size=(1, 1, 100))

        with pytest.raises(RuntimeError):
            # using both input_length and mask is not allowed
            calculate_mse_batch(estimate=estimate, target=target, input_length=input_length, mask=mask)

    @pytest.mark.unit
    def test_calculate_mse_invalid_dimensions(self):
        """Test MSE calculation with unsupported dimensions."""
        estimate = torch.randn(size=(1, 1, 100, 10))
        target = torch.randn(size=(1, 1, 100))

        with pytest.raises(AssertionError):
            # mismatched dimensions are not allowed
            calculate_mse_batch(estimate=estimate, target=target)

        estimate = torch.randn(size=(1, 1, 100, 10, 20))
        target = torch.randn(size=estimate.shape)

        with pytest.raises(RuntimeError):
            # dimensions larger than four are not allowed
            calculate_mse_batch(estimate=estimate, target=target)

    @pytest.mark.unit
    def test_calculate_mae_input_and_mask(self):
        """Test MAE calculation with simultaneous input length and mask."""
        estimate = torch.randn(size=(1, 1, 100))
        target = torch.randn(size=(1, 1, 100))
        input_length = torch.tensor([100])
        mask = torch.ones(size=(1, 1, 100))

        with pytest.raises(RuntimeError):
            # using both input_length and mask is not allowed
            calculate_mae_batch(estimate=estimate, target=target, input_length=input_length, mask=mask)

    @pytest.mark.unit
    def test_calculate_mae_invalid_dimensions(self):
        """Test MAE calculation with unsupported dimensions."""
        estimate = torch.randn(size=(1, 1, 100, 10))
        target = torch.randn(size=(1, 1, 100))

        with pytest.raises(AssertionError):
            # mismatched dimensions are not allowed
            calculate_mae_batch(estimate=estimate, target=target)

        estimate = torch.randn(size=(1, 1, 100, 10, 20))
        target = torch.randn(size=estimate.shape)

        with pytest.raises(RuntimeError):
            # dimensions larger than four are not allowed
            calculate_mae_batch(estimate=estimate, target=target)

    @pytest.mark.unit
    @pytest.mark.parametrize('num_channels', [1, 4])
    def test_sdr(self, num_channels: int):
        """Test SDR calculation"""
        test_eps = [0, 1e-16, 1e-1]
        batch_size = 8
        num_samples = 50
        num_batches = 10
        random_seed = 42
        atol = 1e-6

        _rng = np.random.default_rng(seed=random_seed)

        for remove_mean in [True, False]:
            for eps in test_eps:

                sdr_loss = SDRLoss(eps=eps, remove_mean=remove_mean)

                for n in range(num_batches):

                    # Generate random signal
                    target = _rng.normal(size=(batch_size, num_channels, num_samples))
                    # Random noise + scaling
                    noise = _rng.uniform(low=0.01, high=1) * _rng.normal(size=(batch_size, num_channels, num_samples))
                    # Estimate
                    estimate = target + noise

                    # DC bias for both
                    target += _rng.uniform(low=-1, high=1)
                    estimate += _rng.uniform(low=-1, high=1)

                    # Tensors for testing the loss
                    tensor_estimate = torch.tensor(estimate)
                    tensor_target = torch.tensor(target)

                    # Reference SDR
                    golden_sdr = np.zeros((batch_size, num_channels))
                    for b in range(batch_size):
                        for m in range(num_channels):
                            golden_sdr[b, m] = calculate_sdr_numpy(
                                estimate=estimate[b, m, :],
                                target=target[b, m, :],
                                remove_mean=remove_mean,
                                eps=eps,
                            )

                    # Calculate SDR in torch
                    uut_sdr = calculate_sdr_batch(
                        estimate=tensor_estimate,
                        target=tensor_target,
                        remove_mean=remove_mean,
                        eps=eps,
                    )

                    # Calculate SDR loss
                    uut_sdr_loss = sdr_loss(estimate=tensor_estimate, target=tensor_target)

                    # Compare torch SDR vs numpy
                    assert np.allclose(
                        uut_sdr.cpu().detach().numpy(), golden_sdr, atol=atol
                    ), f'SDR not matching for example {n}, eps={eps}, remove_mean={remove_mean}'

                    # Compare SDR loss vs average of torch SDR
                    assert np.isclose(
                        uut_sdr_loss, -uut_sdr.mean(), atol=atol
                    ), f'SDRLoss not matching for example {n}, eps={eps}, remove_mean={remove_mean}'

    @pytest.mark.unit
    @pytest.mark.parametrize('num_channels', [1, 4])
    def test_sdr_weighted(self, num_channels: int):
        """Test SDR calculation with weighting for channels"""
        batch_size = 8
        num_samples = 50
        num_batches = 10
        random_seed = 42
        atol = 1e-6

        _rng = np.random.default_rng(seed=random_seed)

        channel_weight = _rng.uniform(low=0.01, high=1.0, size=num_channels)
        channel_weight = channel_weight / np.sum(channel_weight)
        sdr_loss = SDRLoss(weight=channel_weight)

        for n in range(num_batches):

            # Generate random signal
            target = _rng.normal(size=(batch_size, num_channels, num_samples))
            # Random noise + scaling
            noise = _rng.uniform(low=0.001, high=10) * _rng.normal(size=target.shape)
            # Estimate
            estimate = target + noise

            # Tensors for testing the loss
            tensor_estimate = torch.tensor(estimate)
            tensor_target = torch.tensor(target)

            # Reference SDR
            golden_sdr = 0
            for b in range(batch_size):
                sdr = [
                    calculate_sdr_numpy(estimate=estimate[b, m, :], target=target[b, m, :])
                    for m in range(num_channels)
                ]
                # weighted sum
                sdr = np.sum(np.array(sdr) * channel_weight)
                golden_sdr += sdr
            golden_sdr /= batch_size  # average over batch

            # Calculate SDR
            uut_sdr_loss = sdr_loss(estimate=tensor_estimate, target=tensor_target)

            # Compare
            assert np.allclose(
                uut_sdr_loss.cpu().detach().numpy(), -golden_sdr, atol=atol
            ), f'SDRLoss not matching for example {n}'

    @pytest.mark.unit
    @pytest.mark.parametrize('num_channels', [1, 4])
    def test_sdr_input_length(self, num_channels):
        """Test SDR calculation with input length."""
        batch_size = 8
        max_num_samples = 50
        num_batches = 10
        random_seed = 42
        atol = 1e-6

        _rng = np.random.default_rng(seed=random_seed)

        sdr_loss = SDRLoss()

        for n in range(num_batches):

            # Generate random signal
            target = _rng.normal(size=(batch_size, num_channels, max_num_samples))
            # Random noise + scaling
            noise = _rng.uniform(low=0.001, high=10) * _rng.normal(size=target.shape)
            # Estimate
            estimate = target + noise

            # Limit calculation to random input_length samples
            input_length = _rng.integers(low=1, high=max_num_samples, size=batch_size)

            # Tensors for testing the loss
            tensor_estimate = torch.tensor(estimate)
            tensor_target = torch.tensor(target)
            tensor_input_length = torch.tensor(input_length)

            # Reference SDR
            golden_sdr = 0
            for b, b_len in enumerate(input_length):
                sdr = [
                    calculate_sdr_numpy(estimate=estimate[b, m, :b_len], target=target[b, m, :b_len])
                    for m in range(num_channels)
                ]
                sdr = np.mean(np.array(sdr))
                golden_sdr += sdr
            golden_sdr /= batch_size  # average over batch

            # Calculate SDR
            uut_sdr_loss = sdr_loss(estimate=tensor_estimate, target=tensor_target, input_length=tensor_input_length)

            # Compare
            assert np.allclose(
                uut_sdr_loss.cpu().detach().numpy(), -golden_sdr, atol=atol
            ), f'SDRLoss not matching for example {n}'

    @pytest.mark.unit
    @pytest.mark.parametrize('num_channels', [1, 4])
    def test_sdr_scale_invariant(self, num_channels: int):
        """Test SDR calculation with scale invariant option."""
        batch_size = 8
        max_num_samples = 50
        num_batches = 10
        random_seed = 42
        atol = 1e-6

        _rng = np.random.default_rng(seed=random_seed)

        sdr_loss = SDRLoss(scale_invariant=True)

        for n in range(num_batches):

            # Generate random signal
            target = _rng.normal(size=(batch_size, num_channels, max_num_samples))
            # Random noise + scaling
            noise = _rng.uniform(low=0.001, high=10) * _rng.normal(size=target.shape)
            # Estimate
            estimate = target + noise

            # Limit calculation to random input_length samples
            input_length = _rng.integers(low=1, high=max_num_samples, size=batch_size)

            # Tensors for testing the loss
            tensor_estimate = torch.tensor(estimate)
            tensor_target = torch.tensor(target)
            tensor_input_length = torch.tensor(input_length)

            # Reference SDR
            golden_sdr = 0
            for b, b_len in enumerate(input_length):
                sdr = [
                    calculate_sdr_numpy(
                        estimate=estimate[b, m, :b_len], target=target[b, m, :b_len], scale_invariant=True
                    )
                    for m in range(num_channels)
                ]
                sdr = np.mean(np.array(sdr))
                golden_sdr += sdr
            golden_sdr /= batch_size  # average over batch

            # Calculate SDR loss
            uut_sdr_loss = sdr_loss(estimate=tensor_estimate, target=tensor_target, input_length=tensor_input_length)

            # Compare
            assert np.allclose(
                uut_sdr_loss.cpu().detach().numpy(), -golden_sdr, atol=atol
            ), f'SDRLoss not matching for example {n}'

    @pytest.mark.unit
    @pytest.mark.parametrize('num_channels', [1, 4])
    def test_sdr_binary_mask(self, num_channels):
        """Test SDR calculation with temporal mask."""
        batch_size = 8
        max_num_samples = 50
        num_batches = 10
        random_seed = 42
        atol = 1e-6

        _rng = np.random.default_rng(seed=random_seed)

        sdr_loss = SDRLoss()

        for n in range(num_batches):

            # Generate random signal
            target = _rng.normal(size=(batch_size, num_channels, max_num_samples))
            # Random noise + scaling
            noise = _rng.uniform(low=0.001, high=10) * _rng.normal(size=target.shape)
            # Estimate
            estimate = target + noise

            # Limit calculation to masked samples
            mask = _rng.integers(low=0, high=2, size=(batch_size, num_channels, max_num_samples))

            # Tensors for testing the loss
            tensor_estimate = torch.tensor(estimate)
            tensor_target = torch.tensor(target)
            tensor_mask = torch.tensor(mask)

            # Reference SDR
            golden_sdr = 0
            for b in range(batch_size):
                sdr = [
                    calculate_sdr_numpy(
                        estimate=estimate[b, m, mask[b, m, :] > 0], target=target[b, m, mask[b, m, :] > 0]
                    )
                    for m in range(num_channels)
                ]
                sdr = np.mean(np.array(sdr))
                golden_sdr += sdr
            golden_sdr /= batch_size  # average over batch

            # Calculate SDR loss
            uut_sdr_loss = sdr_loss(estimate=tensor_estimate, target=tensor_target, mask=tensor_mask)

            # Compare
            assert np.allclose(
                uut_sdr_loss.cpu().detach().numpy(), -golden_sdr, atol=atol
            ), f'SDRLoss not matching for example {n}'

    @pytest.mark.unit
    @pytest.mark.parametrize('num_channels', [1])
    @pytest.mark.parametrize('sdr_max', [10, 0])
    def test_sdr_max(self, num_channels: int, sdr_max: float):
        """Test SDR calculation with soft max threshold."""
        batch_size = 8
        max_num_samples = 50
        num_batches = 10
        random_seed = 42
        atol = 1e-6

        _rng = np.random.default_rng(seed=random_seed)

        sdr_loss = SDRLoss(sdr_max=sdr_max)

        for n in range(num_batches):

            # Generate random signal
            target = _rng.normal(size=(batch_size, num_channels, max_num_samples))
            # Random noise + scaling
            noise = _rng.uniform(low=0.001, high=10) * _rng.normal(size=target.shape)
            # Estimate
            estimate = target + noise

            # Limit calculation to random input_length samples
            input_length = _rng.integers(low=1, high=max_num_samples, size=batch_size)

            # Tensors for testing the loss
            tensor_estimate = torch.tensor(estimate)
            tensor_target = torch.tensor(target)
            tensor_input_length = torch.tensor(input_length)

            # Reference SDR
            golden_sdr = 0
            for b, b_len in enumerate(input_length):
                sdr = [
                    calculate_sdr_numpy(estimate=estimate[b, m, :b_len], target=target[b, m, :b_len], sdr_max=sdr_max)
                    for m in range(num_channels)
                ]
                sdr = np.mean(np.array(sdr))
                golden_sdr += sdr
            golden_sdr /= batch_size  # average over batch

            # Calculate SDR loss
            uut_sdr_loss = sdr_loss(estimate=tensor_estimate, target=tensor_target, input_length=tensor_input_length)

            # Compare
            assert np.allclose(
                uut_sdr_loss.cpu().detach().numpy(), -golden_sdr, atol=atol
            ), f'SDRLoss not matching for example {n}'

    @pytest.mark.unit
    @pytest.mark.parametrize('filter_length', [1, 32])
    @pytest.mark.parametrize('num_channels', [1, 4])
    @pytest.mark.parametrize('use_mask', [True, False])
    @pytest.mark.parametrize('use_input_length', [True, False])
    def test_target_calculation(self, num_channels: int, filter_length: int, use_mask: bool, use_input_length: bool):
        """Test target calculation with scale and convolution invariance."""
        batch_size = 8
        max_num_samples = 50
        num_batches = 10
        random_seed = 42
        atol = 1e-6

        _rng = np.random.default_rng(seed=random_seed)

        for n in range(num_batches):

            # Generate random signal
            target = _rng.normal(size=(batch_size, num_channels, max_num_samples))
            # Random noise + scaling
            noise = _rng.uniform(low=0.001, high=10) * _rng.normal(size=target.shape)
            # Estimate
            estimate = target + noise

            # Limit calculation to random input_length samples
            input_length = _rng.integers(low=filter_length, high=max_num_samples, size=batch_size)

            # Corresponding mask
            mask = torch.zeros(size=(batch_size, 1, max_num_samples))
            for i in range(batch_size):
                mask[i, :, : input_length[i]] = 1.0

            if use_mask and use_input_length:
                with pytest.raises(RuntimeError):
                    scale_invariant_target(
                        estimate=torch.tensor(estimate),
                        target=torch.tensor(target),
                        input_length=torch.tensor(input_length),
                        mask=mask,
                    )

                with pytest.raises(RuntimeError):
                    convolution_invariant_target(
                        estimate=torch.tensor(estimate),
                        target=torch.tensor(target),
                        input_length=torch.tensor(input_length),
                        mask=mask,
                        filter_length=filter_length,
                    )

                # Done with this test
                continue

            # UUT
            si_target = scale_invariant_target(
                estimate=torch.tensor(estimate),
                target=torch.tensor(target),
                input_length=torch.tensor(input_length) if use_input_length else None,
                mask=mask if use_mask else None,
            )
            ci_target = convolution_invariant_target(
                estimate=torch.tensor(estimate),
                target=torch.tensor(target),
                input_length=torch.tensor(input_length) if use_input_length else None,
                mask=mask if use_mask else None,
                filter_length=filter_length,
            )

            if filter_length == 1:
                assert torch.allclose(ci_target, si_target), f'SI and CI should match for filter_length=1'

            # Compare against numpy
            for b in range(batch_size):
                # valid length for the current example
                b_len = input_length[b] if use_input_length or use_mask else max_num_samples

                # calculate reference target for each channel
                for m in range(num_channels):
                    # Scale invariant reference
                    si_target_ref = scale_invariant_target_numpy(
                        estimate=estimate[b, m, :b_len], target=target[b, m, :b_len]
                    )

                    assert np.allclose(
                        si_target[b, m, :b_len].cpu().detach().numpy(), si_target_ref, atol=atol
                    ), f'SI not matching for example {n}, channel {m}'

                    # Convolution invariant reference
                    ci_target_ref = convolution_invariant_target_numpy(
                        estimate=estimate[b, m, :b_len], target=target[b, m, :b_len], filter_length=filter_length
                    )

                    assert np.allclose(
                        ci_target[b, m, :b_len].cpu().detach().numpy(), ci_target_ref, atol=atol
                    ), f'CI not matching for example {n}, channel {m}'

    @pytest.mark.unit
    @pytest.mark.parametrize('filter_length', [1, 32])
    @pytest.mark.parametrize('num_channels', [1, 4])
    def test_sdr_convolution_invariant(self, num_channels: int, filter_length: int):
        """Test SDR calculation with convolution invariant option."""
        batch_size = 8
        max_num_samples = 50
        num_batches = 10
        random_seed = 42
        atol = 1e-6

        _rng = np.random.default_rng(seed=random_seed)

        sdr_loss = SDRLoss(convolution_invariant=True, convolution_filter_length=filter_length)

        for n in range(num_batches):

            # Generate random signal
            target = _rng.normal(size=(batch_size, num_channels, max_num_samples))
            # Random noise + scaling
            noise = _rng.uniform(low=0.001, high=10) * _rng.normal(size=target.shape)
            # Estimate
            estimate = target + noise

            # Limit calculation to random input_length samples
            input_length = _rng.integers(low=filter_length, high=max_num_samples, size=batch_size)

            # Calculate SDR loss
            uut_sdr_loss = sdr_loss(
                estimate=torch.tensor(estimate), target=torch.tensor(target), input_length=torch.tensor(input_length)
            )

            # Reference SDR
            golden_sdr = 0
            for b, b_len in enumerate(input_length):
                sdr = [
                    calculate_sdr_numpy(
                        estimate=estimate[b, m, :b_len],
                        target=target[b, m, :b_len],
                        convolution_invariant=True,
                        convolution_filter_length=filter_length,
                    )
                    for m in range(num_channels)
                ]
                sdr = np.mean(np.array(sdr))
                golden_sdr += sdr
            golden_sdr /= batch_size  # average over batch

            # Compare
            assert np.allclose(
                uut_sdr_loss.cpu().detach().numpy(), -golden_sdr, atol=atol
            ), f'SDRLoss not matching for example {n}'

    @pytest.mark.unit
    def test_sdr_scale_and_convolution_invariant(self):
        """Test SDR calculation with scale and conovolution invariant options."""
        with pytest.raises(ValueError):
            # using both scale and convolution invariant is not allowed
            SDRLoss(scale_invariant=True, convolution_invariant=True)

    @pytest.mark.unit
    def test_sdr_length_and_mask(self):
        """Test SDR calculation with simultaneous input length and mask."""

        estimate = torch.randn(size=(1, 1, 100))
        target = torch.randn(size=(1, 1, 100))
        input_length = torch.tensor([100])
        mask = torch.ones(size=(1, 1, 100))

        sdr_loss = SDRLoss(scale_invariant=False, convolution_invariant=False)

        with pytest.raises(RuntimeError):
            # using both input_length and mask is not allowed
            sdr_loss(estimate=estimate, target=target, input_length=input_length, mask=mask)

    @pytest.mark.unit
    def test_sdr_invalid_weight(self):
        """Test SDR with invalid weights."""
        with pytest.raises(ValueError):
            # negative weights are not allowed
            SDRLoss(weight=[-1, 1])

        with pytest.raises(ValueError):
            # weights should sum to 1
            SDRLoss(weight=[0.1, 0.1])

    @pytest.mark.unit
    def test_sdr_invalid_reduction(self):
        """Test SDR with invalid reduction."""
        with pytest.raises(ValueError):
            SDRLoss(reduction='not-mean')

    @pytest.mark.unit
    @pytest.mark.parametrize('num_channels', [1, 4])
    @pytest.mark.parametrize('ndim', [3, 4])
    def test_mse(self, num_channels: int, ndim: int):
        """Test MSE calculation"""
        batch_size = 8
        num_samples = 50
        num_features = 123
        num_batches = 10
        random_seed = 42
        atol = 1e-6

        signal_shape = (
            (batch_size, num_channels, num_features, num_samples)
            if ndim == 4
            else (batch_size, num_channels, num_samples)
        )

        reduction_dim = (-2, -1) if ndim == 4 else -1

        mse_loss = MSELoss(ndim=ndim)

        _rng = np.random.default_rng(seed=random_seed)

        for n in range(num_batches):

            # Generate random signal
            target = _rng.normal(size=signal_shape)
            # Random noise + scaling
            noise = _rng.uniform(low=0.01, high=1) * _rng.normal(size=signal_shape)
            # Estimate
            estimate = target + noise

            # DC bias for both
            target += _rng.uniform(low=-1, high=1)
            estimate += _rng.uniform(low=-1, high=1)

            # Tensors for testing the loss
            tensor_estimate = torch.tensor(estimate)
            tensor_target = torch.tensor(target)

            # Reference MSE
            golden_mse = np.zeros((batch_size, num_channels))
            for b in range(batch_size):
                for m in range(num_channels):
                    err = estimate[b, m, :] - target[b, m, :]
                    golden_mse[b, m] = np.mean(np.abs(err) ** 2, axis=reduction_dim)

            # Calculate MSE in torch
            uut_mse = calculate_mse_batch(estimate=tensor_estimate, target=tensor_target)

            # Calculate MSE loss
            uut_mse_loss = mse_loss(estimate=tensor_estimate, target=tensor_target)

            # Compare torch SDR vs numpy
            assert np.allclose(
                uut_mse.cpu().detach().numpy(), golden_mse, atol=atol
            ), f'MSE not matching for example {n}'

            # Compare SDR loss vs average of torch SDR
            assert np.isclose(uut_mse_loss, uut_mse.mean(), atol=atol), f'MSELoss not matching for example {n}'

    @pytest.mark.unit
    @pytest.mark.parametrize('num_channels', [1, 4])
    @pytest.mark.parametrize('ndim', [3, 4])
    def test_mse_weighted(self, num_channels: int, ndim: int):
        """Test MSE calculation with weighting for channels"""
        batch_size = 8
        num_samples = 50
        num_features = 123
        num_batches = 10
        random_seed = 42
        atol = 1e-6

        signal_shape = (
            (batch_size, num_channels, num_features, num_samples)
            if ndim == 4
            else (batch_size, num_channels, num_samples)
        )

        reduction_dim = (-2, -1) if ndim == 4 else -1

        _rng = np.random.default_rng(seed=random_seed)

        channel_weight = _rng.uniform(low=0.01, high=1.0, size=num_channels)
        channel_weight = channel_weight / np.sum(channel_weight)
        mse_loss = MSELoss(weight=channel_weight, ndim=ndim)

        for n in range(num_batches):

            # Generate random signal
            target = _rng.normal(size=signal_shape)
            # Random noise + scaling
            noise = _rng.uniform(low=0.001, high=10) * _rng.normal(size=target.shape)
            # Estimate
            estimate = target + noise

            # Tensors for testing the loss
            tensor_estimate = torch.tensor(estimate)
            tensor_target = torch.tensor(target)

            # Reference MSE
            golden_mse = 0
            for b in range(batch_size):
                mse = [
                    np.mean(np.abs(estimate[b, m, :] - target[b, m, :]) ** 2, axis=reduction_dim)
                    for m in range(num_channels)
                ]
                # weighted sum
                mse = np.sum(np.array(mse) * channel_weight)
                golden_mse += mse
            golden_mse /= batch_size  # average over batch

            # Calculate MSE loss
            uut_mse_loss = mse_loss(estimate=tensor_estimate, target=tensor_target)

            # Compare
            assert np.allclose(
                uut_mse_loss.cpu().detach().numpy(), golden_mse, atol=atol
            ), f'MSELoss not matching for example {n}'

    @pytest.mark.unit
    @pytest.mark.parametrize('num_channels', [1, 4])
    @pytest.mark.parametrize('ndim', [3, 4])
    def test_mse_input_length(self, num_channels: int, ndim: int):
        """Test MSE calculation with input length."""
        batch_size = 8
        max_num_samples = 50
        num_features = 123
        num_batches = 10
        random_seed = 42
        atol = 1e-6

        signal_shape = (
            (batch_size, num_channels, num_features, max_num_samples)
            if ndim == 4
            else (batch_size, num_channels, max_num_samples)
        )

        reduction_dim = (-2, -1) if ndim == 4 else -1

        _rng = np.random.default_rng(seed=random_seed)

        mse_loss = MSELoss(ndim=ndim)

        for n in range(num_batches):

            # Generate random signal
            target = _rng.normal(size=signal_shape)
            # Random noise + scaling
            noise = _rng.uniform(low=0.001, high=10) * _rng.normal(size=target.shape)
            # Estimate
            estimate = target + noise

            # Limit calculation to random input_length samples
            input_length = _rng.integers(low=1, high=max_num_samples, size=batch_size)

            # Tensors for testing the loss
            tensor_estimate = torch.tensor(estimate)
            tensor_target = torch.tensor(target)
            tensor_input_length = torch.tensor(input_length)

            # Reference MSE
            golden_mse = 0
            for b, b_len in enumerate(input_length):
                mse = [
                    np.mean(np.abs(estimate[b, m, ..., :b_len] - target[b, m, ..., :b_len]) ** 2, axis=reduction_dim)
                    for m in range(num_channels)
                ]
                mse = np.mean(np.array(mse))
                golden_mse += mse
            golden_mse /= batch_size  # average over batch

            # Calculate MSE
            uut_mse_loss = mse_loss(estimate=tensor_estimate, target=tensor_target, input_length=tensor_input_length)

            # Compare
            assert np.allclose(
                uut_mse_loss.cpu().detach().numpy(), golden_mse, atol=atol
            ), f'MSELoss not matching for example {n}'

    @pytest.mark.unit
    def test_mse_invalid_weight(self):
        """Test MSE with unsupported weights."""
        with pytest.raises(ValueError):
            # negative weights are not allowed
            MSELoss(weight=[-1, 1])

        with pytest.raises(ValueError):
            # weights should sum to 1
            MSELoss(weight=[0.1, 0.1])

    @pytest.mark.unit
    def test_mse_invalid_reduction(self):
        """Test MSE with unsupported reduction."""
        with pytest.raises(ValueError):
            # unsupported reduction
            MSELoss(reduction='not-mean')

    @pytest.mark.unit
    def test_mse_invalid_ndim(self):
        """Test MSE with unsupported dimensions."""
        with pytest.raises(ValueError):
            # supports dimensions 3, 4
            MSELoss(ndim=2)

        with pytest.raises(ValueError):
            # supports dimensions 3, 4
            MSELoss(ndim=5)

    @pytest.mark.unit
    @pytest.mark.parametrize('num_channels', [1, 4])
    @pytest.mark.parametrize('ndim', [3, 4])
    def test_mae(self, num_channels: int, ndim: int):
        """Test MAE calculation"""
        batch_size = 8
        num_samples = 50
        num_features = 123
        num_batches = 10
        random_seed = 42
        atol = 1e-6

        signal_shape = (
            (batch_size, num_channels, num_features, num_samples)
            if ndim == 4
            else (batch_size, num_channels, num_samples)
        )

        reduction_dim = (-2, -1) if ndim == 4 else -1

        mae_loss = MAELoss(ndim=ndim)

        _rng = np.random.default_rng(seed=random_seed)

        for n in range(num_batches):

            # Generate random signal
            target = _rng.normal(size=signal_shape)
            # Random noise + scaling
            noise = _rng.uniform(low=0.01, high=1) * _rng.normal(size=signal_shape)
            # Estimate
            estimate = target + noise

            # DC bias for both
            target += _rng.uniform(low=-1, high=1)
            estimate += _rng.uniform(low=-1, high=1)

            # Tensors for testing the loss
            tensor_estimate = torch.tensor(estimate)
            tensor_target = torch.tensor(target)

            # Reference MSE
            golden_mae = np.zeros((batch_size, num_channels))
            for b in range(batch_size):
                for m in range(num_channels):
                    err = estimate[b, m, :] - target[b, m, :]
                    golden_mae[b, m] = np.mean(np.abs(err), axis=reduction_dim)

            # Calculate MSE loss
            uut_mae_loss = mae_loss(estimate=tensor_estimate, target=tensor_target)

            # Compare
            assert np.allclose(
                uut_mae_loss.cpu().detach().numpy(), golden_mae.mean(), atol=atol
            ), f'MAE not matching for example {n}'

    @pytest.mark.unit
    @pytest.mark.parametrize('num_channels', [1, 4])
    @pytest.mark.parametrize('ndim', [3, 4])
    def test_mae_weighted(self, num_channels: int, ndim: int):
        """Test MAE calculation with weighting for channels"""
        batch_size = 8
        num_samples = 50
        num_features = 123
        num_batches = 10
        random_seed = 42
        atol = 1e-6

        signal_shape = (
            (batch_size, num_channels, num_features, num_samples)
            if ndim == 4
            else (batch_size, num_channels, num_samples)
        )

        reduction_dim = (-2, -1) if ndim == 4 else -1

        _rng = np.random.default_rng(seed=random_seed)

        channel_weight = _rng.uniform(low=0.01, high=1.0, size=num_channels)
        channel_weight = channel_weight / np.sum(channel_weight)
        mae_loss = MAELoss(weight=channel_weight, ndim=ndim)

        for n in range(num_batches):

            # Generate random signal
            target = _rng.normal(size=signal_shape)
            # Random noise + scaling
            noise = _rng.uniform(low=0.001, high=10) * _rng.normal(size=target.shape)
            # Estimate
            estimate = target + noise

            # Tensors for testing the loss
            tensor_estimate = torch.tensor(estimate)
            tensor_target = torch.tensor(target)

            # Reference MAE
            golden_mae = 0
            for b in range(batch_size):
                mae = [
                    np.mean(np.abs(estimate[b, m, :] - target[b, m, :]), axis=reduction_dim)
                    for m in range(num_channels)
                ]
                # weighted sum
                mae = np.sum(np.array(mae) * channel_weight)
                golden_mae += mae
            golden_mae /= batch_size  # average over batch

            # Calculate MAE loss
            uut_mae_loss = mae_loss(estimate=tensor_estimate, target=tensor_target)

            # Compare
            assert np.allclose(
                uut_mae_loss.cpu().detach().numpy(), golden_mae, atol=atol
            ), f'MAELoss not matching for example {n}'

    @pytest.mark.unit
    @pytest.mark.parametrize('num_channels', [1, 4])
    @pytest.mark.parametrize('ndim', [3, 4])
    def test_mae_input_length(self, num_channels: int, ndim: int):
        """Test MAE calculation with input length."""
        batch_size = 8
        max_num_samples = 50
        num_features = 123
        num_batches = 10
        random_seed = 42
        atol = 1e-6

        signal_shape = (
            (batch_size, num_channels, num_features, max_num_samples)
            if ndim == 4
            else (batch_size, num_channels, max_num_samples)
        )

        reduction_dim = (-2, -1) if ndim == 4 else -1

        _rng = np.random.default_rng(seed=random_seed)

        mae_loss = MAELoss(ndim=ndim)

        for n in range(num_batches):

            # Generate random signal
            target = _rng.normal(size=signal_shape)
            # Random noise + scaling
            noise = _rng.uniform(low=0.001, high=10) * _rng.normal(size=target.shape)
            # Estimate
            estimate = target + noise

            # Limit calculation to random input_length samples
            input_length = _rng.integers(low=1, high=max_num_samples, size=batch_size)

            # Tensors for testing the loss
            tensor_estimate = torch.tensor(estimate)
            tensor_target = torch.tensor(target)
            tensor_input_length = torch.tensor(input_length)

            # Reference MSE
            golden_mae = 0
            for b, b_len in enumerate(input_length):
                mae = [
                    np.mean(np.abs(estimate[b, m, ..., :b_len] - target[b, m, ..., :b_len]), axis=reduction_dim)
                    for m in range(num_channels)
                ]
                mae = np.mean(np.array(mae))
                golden_mae += mae
            golden_mae /= batch_size  # average over batch

            # Calculate MSE
            uut_mae_loss = mae_loss(estimate=tensor_estimate, target=tensor_target, input_length=tensor_input_length)

            # Compare
            assert np.allclose(
                uut_mae_loss.cpu().detach().numpy(), golden_mae, atol=atol
            ), f'MAELoss not matching for example {n}'

    @pytest.mark.unit
    def test_mae_invalid_weight(self):
        """Test MAE with invalid weights."""
        with pytest.raises(ValueError):
            # negative weights are not allowed
            MAELoss(weight=[-1, 1])

        with pytest.raises(ValueError):
            # weights should sum to 1
            MAELoss(weight=[0.1, 0.1])

    @pytest.mark.unit
    def test_mae_invalid_reduction(self):
        """Test MAE with invalid reduction."""
        with pytest.raises(ValueError):
            # unsupported reduction
            MAELoss(reduction='not-mean')

    @pytest.mark.unit
    def test_mae_invalid_ndim(self):
        """Test MAE with invalid dimensions."""
        with pytest.raises(ValueError):
            # supports dimensions 3, 4
            MAELoss(ndim=2)

        with pytest.raises(ValueError):
            # supports dimensions 3, 4
            MAELoss(ndim=5)

    @pytest.mark.unit
    def test_maxine_combined_loss(self, test_data_dir):
        INPUT_LOCATION = os.path.join(test_data_dir, 'audio', 'maxine', 'input.bin')
        ATOL = 1e-2

        GOLDEN_VALUES = [
            ((1, 0, 0, True, True), 80.0),
            ((0, 1, 0, True, True), 1.9749),
            ((0, 0, 1, True, True), 19.2192),
            ((1, 1, 1, True, True), 101.1941),
        ]
        batch_size = 16

        for value in GOLDEN_VALUES:
            config, golden = value
            sisnr_wt, asr_wt, spec_wt, use_asr, use_spec = config

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            loss_instance = CombinedLoss(
                sample_rate=16000,
                fft_length=1920,
                hop_length=480,
                num_mels=320,
                sisnr_loss_weight=sisnr_wt,
                asr_loss_weight=asr_wt,
                spectral_loss_weight=spec_wt,
                use_asr_loss=use_asr,
                use_mel_spec=use_spec,
            ).to(device)

            input_data = torch.tensor(np.fromfile(INPUT_LOCATION, np.float32))
            input_data = input_data.repeat(batch_size).reshape((batch_size, 1, -1))
            input_data = input_data.to(device)

            estimate = torch.tensor(np.zeros(input_data.shape, np.float32)).reshape((batch_size, 1, -1)).to(device)

            loss = loss_instance.forward(estimate=estimate, target=input_data).cpu()

            assert np.allclose(loss, golden, atol=ATOL)
