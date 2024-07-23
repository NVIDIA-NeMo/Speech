# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any

import torch
from torchmetrics import Metric
from nemo.utils import logging

try:
    import torchaudio

    HAVE_TORCHAUDIO = True
except ModuleNotFoundError:
    HAVE_TORCHAUDIO = False


class SquimMOSMetric(Metric):
    """A metric calculating the average Torchaudio Squim MOS.

    Args:
        fs: sampling rate of the input signals
    """

    sample_rate: int = 16000  # sample rate of the model
    mos_sum: torch.Tensor
    num_examples: torch.Tensor
    higher_is_better: bool = True

    def __init__(self, fs: int, **kwargs: Any):
        super().__init__(**kwargs)

        if not HAVE_TORCHAUDIO:
            raise ModuleNotFoundError(f"{self.__class__.__name__} metric needs `torchaudio`.")

        if fs != self.sample_rate:
            # Resampler: kaiser_best
            self._squim_mos_metric_resampler = torchaudio.transforms.Resample(
                orig_freq=fs,
                new_freq=self.sample_rate,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method='sinc_interp_kaiser',
                beta=14.769656459379492,
            )
            logging.warning('Input signals will be resampled from fs=%d to %d Hz', fs, self.sample_rate)
        self.fs = fs

        # MOS model
        self._squim_mos_metric_model = torchaudio.pipelines.SQUIM_SUBJECTIVE.get_model()

        self.add_state('mos_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('num_examples', default=torch.tensor(0), dist_reduce_fx='sum')
        logging.debug('Setup metric %s with input fs=%s', self.__class__.__name__, self.fs)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update the metric by calculating the MOS score for the current batch.

        Args:
            preds: tensor with predictions, shape (B, T)
            target: tensor with target signals, shape (B, T). Target can be a non-matching reference.
        """
        if self.fs != self.sample_rate:
            preds = self._squim_mos_metric_resampler(preds)
            target = self._squim_mos_metric_resampler(target)

        if preds.ndim == 1:
            # Unsqueeze batch dimension
            preds = preds.unsqueeze(0)
            target = target.unsqueeze(0)
        elif preds.ndim > 2:
            raise ValueError(f'Expected 1D or 2D signals, got {preds.ndim}D signals')

        mos_batch = self._squim_mos_metric_model(preds, target)

        self.mos_sum += mos_batch.sum()
        self.num_examples += mos_batch.numel()

    def compute(self) -> torch.Tensor:
        """Compute the underlying metric."""
        return self.mos_sum / self.num_examples

    def state_dict(self, *args, **kwargs):
        """Do not save the MOS model and resampler in the state dict."""
        state_dict = super().state_dict(*args, **kwargs)
        # Do not include resampler or mos_model in the state dict
        remove_keys = [
            key
            for key in state_dict.keys()
            if '_squim_mos_metric_resampler' in key or '_squim_mos_metric_model' in key
        ]
        for key in remove_keys:
            del state_dict[key]
        return state_dict


class SquimObjectiveMetric(Metric):
    """A metric calculating the average Torchaudio Squim objective metric.

    Args:
        fs: sampling rate of the input signals
        metric: the objective metric to calculate. One of 'stoi', 'pesq', 'si_sdr'
    """

    sample_rate: int = 16000  # sample rate of the model
    metric_sum: torch.Tensor
    num_examples: torch.Tensor
    higher_is_better: bool = True

    def __init__(self, fs: int, metric: str, **kwargs: Any):
        super().__init__(**kwargs)

        if not HAVE_TORCHAUDIO:
            raise ModuleNotFoundError(f"{self.__class__.__name__} needs `torchaudio`.")

        if fs != self.sample_rate:
            # Resampler: kaiser_best
            self._squim_objective_metric_resampler = torchaudio.transforms.Resample(
                orig_freq=fs,
                new_freq=self.sample_rate,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method='sinc_interp_kaiser',
                beta=14.769656459379492,
            )
            logging.warning('Input signals will be resampled from fs=%d to %d Hz', fs, self.sample_rate)
        self.fs = fs

        if metric not in ['stoi', 'pesq', 'si_sdr']:
            raise ValueError(f'Unsupported metric {metric}. Supported metrics are "stoi", "pesq", "si_sdr".')

        self.metric = metric

        # Objective model
        self._squim_objective_metric_model = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model()

        self.add_state('metric_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('num_examples', default=torch.tensor(0), dist_reduce_fx='sum')
        logging.debug('Setup %s with metric=%s, input fs=%s', self.__class__.__name__, self.metric, self.fs)

    def update(self, preds: torch.Tensor, target: Any = None) -> None:
        """Update the metric by calculating the selected metric score for the current batch.

        Args:
            preds: tensor with predictions, shape (B, T)
            target: None, not used. Keeping for interfacfe compatibility with other metrics.
        """
        if self.fs != self.sample_rate:
            preds = self._squim_objective_metric_resampler(preds)

        if preds.ndim == 1:
            # Unsqueeze batch dimension
            preds = preds.unsqueeze(0)
        elif preds.ndim > 2:
            raise ValueError(f'Expected 1D or 2D signals, got {preds.ndim}D signals')

        stoi_batch, pesq_batch, si_sdr_batch = self._squim_objective_metric_model(preds)

        if self.metric == 'stoi':
            metric_batch = stoi_batch
        elif self.metric == 'pesq':
            metric_batch = pesq_batch
        elif self.metric == 'si_sdr':
            metric_batch = si_sdr_batch
        else:
            raise ValueError(f'Unknown metric {self.metric}')

        self.metric_sum += metric_batch.sum()
        self.num_examples += metric_batch.numel()

    def compute(self) -> torch.Tensor:
        """Compute the underlying metric."""
        return self.metric_sum / self.num_examples

    def state_dict(self, *args, **kwargs):
        """Do not save the MOS model and resampler in the state dict."""
        state_dict = super().state_dict(*args, **kwargs)
        # Do not include resampler or mos_model in the state dict
        remove_keys = [
            key
            for key in state_dict.keys()
            if '_squim_objective_metric_resampler' in key or '_squim_objective_metric_model' in key
        ]
        for key in remove_keys:
            del state_dict[key]
        return state_dict
