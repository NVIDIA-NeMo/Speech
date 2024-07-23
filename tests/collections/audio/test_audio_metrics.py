# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
from torchmetrics.audio.snr import SignalNoiseRatio

from nemo.collections.audio.metrics.audio import AudioMetricWrapper
from nemo.collections.audio.metrics.squim import SquimMOSMetric, SquimObjectiveMetric

try:
    import torchaudio

    HAVE_TORCHAUDIO = True
except ModuleNotFoundError:
    HAVE_TORCHAUDIO = False


class TestAudioMetricWrapper:
    def test_metric_full_batch(self):
        """Test metric on batches where all examples have equal length."""
        ref_metric = SignalNoiseRatio()
        wrapped_metric = AudioMetricWrapper(metric=SignalNoiseRatio())

        num_resets = 5
        num_batches = 10
        batch_size = 8
        num_channels = 2
        num_samples = 200

        batch_shape = (batch_size, num_channels, num_samples)

        for nr in range(num_resets):
            for nb in range(num_batches):
                target = torch.rand(*batch_shape)
                preds = target + torch.rand(1) * torch.rand(*batch_shape)

                # test forward for a single batch
                batch_value_wrapped = wrapped_metric(preds=preds, target=target)
                batch_value_ref = ref_metric(preds=preds, target=target)

                assert torch.allclose(
                    batch_value_wrapped, batch_value_ref
                ), f'Metric forward not matching for batch {nb}, reset {nr}'

            # test compute (over num_batches)
            assert torch.allclose(
                wrapped_metric.compute(), ref_metric.compute()
            ), f'Metric compute not matching for batch {nb}, reset {nr}'

            ref_metric.reset()
            wrapped_metric.reset()

    def test_input_length(self):
        """Test metric on batches where examples have different length."""
        ref_metric = SignalNoiseRatio()
        wrapped_metric = AudioMetricWrapper(metric=SignalNoiseRatio())

        num_resets = 5
        num_batches = 10
        batch_size = 8
        num_channels = 2
        num_samples = 200

        batch_shape = (batch_size, num_channels, num_samples)

        for nr in range(num_resets):
            for nb in range(num_batches):
                target = torch.rand(*batch_shape)
                preds = target + torch.rand(1) * torch.rand(*batch_shape)

                input_length = torch.randint(low=num_samples // 2, high=num_samples, size=(batch_size,))

                # test forward for a single batch
                batch_value_wrapped = wrapped_metric(preds=preds, target=target, input_length=input_length)

                # compute reference value, assuming batch reduction using averaging
                batch_value_ref = 0
                for b_idx, b_len in enumerate(input_length):
                    batch_value_ref += ref_metric(preds=preds[b_idx, ..., :b_len], target=target[b_idx, ..., :b_len])
                batch_value_ref /= batch_size  # average

                assert torch.allclose(
                    batch_value_wrapped, batch_value_ref
                ), f'Metric forward not matching for batch {nb}, reset {nr}'

            # test compute (over num_batches)
            assert torch.allclose(
                wrapped_metric.compute(), ref_metric.compute()
            ), f'Metric compute not matching for batch {nb}, reset {nr}'

            ref_metric.reset()
            wrapped_metric.reset()

    @pytest.mark.unit
    @pytest.mark.parametrize('channel', [0, 1])
    def test_channel(self, channel):
        """Test metric on a single channel from a batch."""
        ref_metric = SignalNoiseRatio()
        # select only a single channel
        wrapped_metric = AudioMetricWrapper(metric=SignalNoiseRatio(), channel=channel)

        num_resets = 5
        num_batches = 10
        batch_size = 8
        num_channels = 2
        num_samples = 200

        batch_shape = (batch_size, num_channels, num_samples)

        for nr in range(num_resets):
            for nb in range(num_batches):
                target = torch.rand(*batch_shape)
                preds = target + torch.rand(1) * torch.rand(*batch_shape)

                # varying length
                input_length = torch.randint(low=num_samples // 2, high=num_samples, size=(batch_size,))

                # test forward for a single batch
                batch_value_wrapped = wrapped_metric(preds=preds, target=target, input_length=input_length)

                # compute reference value, assuming batch reduction using averaging
                batch_value_ref = 0
                for b_idx, b_len in enumerate(input_length):
                    batch_value_ref += ref_metric(
                        preds=preds[b_idx, channel, :b_len], target=target[b_idx, channel, :b_len]
                    )
                batch_value_ref /= batch_size  # average

                assert torch.allclose(
                    batch_value_wrapped, batch_value_ref
                ), f'Metric forward not matching for batch {nb}, reset {nr}'

            # test compute (over num_batches)
            assert torch.allclose(
                wrapped_metric.compute(), ref_metric.compute()
            ), f'Metric compute not matching for batch {nb}, reset {nr}'

            ref_metric.reset()
            wrapped_metric.reset()


class TestSquimMetrics:
    @pytest.mark.unit
    @pytest.mark.parametrize('fs', [16000, 24000])
    def test_squim_mos(self, fs: int):
        """Test Squim MOS metric"""
        if HAVE_TORCHAUDIO:
            # Setup
            num_batches = 4
            batch_size = 4
            atol = 1e-6

            # UUT
            squim_mos_metric = SquimMOSMetric(fs=fs)

            # Helper function
            resampler = torchaudio.transforms.Resample(
                orig_freq=fs,
                new_freq=16000,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method='sinc_interp_kaiser',
                beta=14.769656459379492,
            )
            squim_mos_model = torchaudio.pipelines.SQUIM_SUBJECTIVE.get_model()

            def calculate_squim_mos(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                if fs != 16000:
                    preds = resampler(preds)
                    target = resampler(target)

                # Calculate MOS
                mos_batch = squim_mos_model(preds, target)
                return mos_batch

            # Test
            mos_sum = torch.tensor(0.0)

            for n in range(num_batches):
                preds = torch.randn(batch_size, fs)
                target = torch.randn(batch_size, fs)

                # UUT forward
                squim_mos_metric.update(preds=preds, target=target)

                # Golden
                mos_golden = calculate_squim_mos(preds=preds, target=target)
                # Accumulate
                mos_sum += mos_golden.sum()

            # Check the final value of the metric
            mos_golden_final = mos_sum / (num_batches * batch_size)
            assert torch.allclose(squim_mos_metric.compute(), mos_golden_final, atol=atol), f'Comparison failed'

        else:
            with pytest.raises(ModuleNotFoundError):
                SquimMOSMetric(fs=fs)

    @pytest.mark.unit
    @pytest.mark.parametrize('metric', ['stoi', 'pesq', 'si_sdr'])
    @pytest.mark.parametrize('fs', [16000, 24000])
    def test_squim_objective(self, metric: str, fs: int):
        """Test Squim objective metric"""
        if HAVE_TORCHAUDIO:
            # Setup
            num_batches = 4
            batch_size = 4
            atol = 1e-6

            # UUT
            squim_objective_metric = SquimObjectiveMetric(fs=fs, metric=metric)

            # Helper function
            resampler = torchaudio.transforms.Resample(
                orig_freq=fs,
                new_freq=16000,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method='sinc_interp_kaiser',
                beta=14.769656459379492,
            )
            squim_objective_model = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model()

            def calculate_squim_objective(preds: torch.Tensor) -> torch.Tensor:
                if fs != 16000:
                    preds = resampler(preds)

                # Calculate metric
                stoi_batch, pesq_batch, si_sdr_batch = squim_objective_model(preds)

                if metric == 'stoi':
                    return stoi_batch
                elif metric == 'pesq':
                    return pesq_batch
                elif metric == 'si_sdr':
                    return si_sdr_batch
                else:
                    raise ValueError(f'Unknown metric {metric}')

            # Test
            metric_sum = torch.tensor(0.0)

            for n in range(num_batches):
                preds = torch.randn(batch_size, fs)

                # UUT forward
                squim_objective_metric.update(preds=preds, target=None)

                # Golden
                metric_golden = calculate_squim_objective(preds=preds)
                # Accumulate
                metric_sum += metric_golden.sum()

            # Check the final value of the metric
            metric_golden_final = metric_sum / (num_batches * batch_size)
            assert torch.allclose(
                squim_objective_metric.compute(), metric_golden_final, atol=atol
            ), f'Comparison failed'

        else:
            with pytest.raises(ModuleNotFoundError):
                SquimObjectiveMetric(fs=fs, metric=metric)
