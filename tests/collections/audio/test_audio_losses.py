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

import numpy as np
import pytest
import torch

from nemo.collections.audio.losses.audio import (
    MAELoss,
    MSELoss,
    SDRLoss,
    calculate_mse_batch,
    calculate_sdr_batch,
    convolution_invariant_target,
    scale_invariant_target,
)
from nemo.collections.audio.parts.utils.audio import (
    calculate_sdr_numpy,
    convolution_invariant_target_numpy,
    scale_invariant_target_numpy,
)


class TestAudioLosses:
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
    def test_target_calculation(self, num_channels: int, filter_length: int):
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

            # UUT
            si_target = scale_invariant_target(
                estimate=torch.tensor(estimate),
                target=torch.tensor(target),
                input_length=torch.tensor(input_length),
                mask=None,
            )
            ci_target = convolution_invariant_target(
                estimate=torch.tensor(estimate),
                target=torch.tensor(target),
                input_length=torch.tensor(input_length),
                mask=None,
                filter_length=filter_length,
            )

            if filter_length == 1:
                assert torch.allclose(ci_target, si_target), f'SI and CI should match for filter_length=1'

            # Compare against numpy
            for b, b_len in enumerate(input_length):
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
