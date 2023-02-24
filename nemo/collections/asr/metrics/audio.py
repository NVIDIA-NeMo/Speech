# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Callable, Iterable, List, Optional, Tuple

import torch
from torchmetrics import Metric
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.pit import PermutationInvariantTraining
from torchmetrics.audio.sdr import ScaleInvariantSignalDistortionRatio, SignalDistortionRatio
from torchmetrics.audio.snr import ScaleInvariantSignalNoiseRatio, SignalNoiseRatio
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

from nemo.utils import logging

__all__ = ['AudioMetricWrapper']

__VERIFIED_METRICS__ = [
    PermutationInvariantTraining,
    ScaleInvariantSignalDistortionRatio,
    SignalDistortionRatio,
    ScaleInvariantSignalNoiseRatio,
    SignalNoiseRatio,
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
]


class AudioMetricWrapper(Metric):
    """A wrapper around an audio metric enabling selection of a specific channel
    and handling of examples in a batch with varying valid input length.

    Note:
        This class assumes that the underlying metric uses averaging to calculate the
        value over a batch. This assumption is only used by `forward` and does not
        impact other methods, such as `update` and `compute`.

    Args:
        metric: base metric that should be wrapped. It is assumed that calculation
                of the metric over a batch is done by averaging.
        channel: Optional, for selecting a channel from `preds` and `target` signals.
                 If None, all channels are used.
        metric_using_batch_averaging: Optional, used to denote that the base metric
                                      is using averaging to calculate the metric value
                                      for a batch.
    """

    full_state_update: bool = False

    def __init__(
        self, metric: Metric, channel: Optional[int] = None, metric_using_batch_averaging: Optional[bool] = None
    ):
        super().__init__()
        if not isinstance(metric, Metric):
            raise ValueError(f"Expected argument `metric` to be an instance of `torchmetrics.Metric` but got {metric}")

        if not metric_using_batch_averaging and type(metric) not in __VERIFIED_METRICS__:
            raise ValueError(
                f'Metric {metric} is not in verified metrics. {self.__class__.__name__} assumes reduction over batch is calculated using averaging. \n'
                'This should not affect the final results, but values for a single batch obtained using `forward` may be inaccurate if using `input_length`. \n'
                'To suppress this message, please confirm the used metric is using batch averaging and set "metric_using_batch_averaging = True"'
            )

        self._metric = metric
        self._channel = channel
        logging.debug('Setup metric %s, channel %s', metric, str(channel))

    def _select_channel(self, preds: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select a single channel from input signals.

        Args:
            preds: tensor with shape (B, C, T)
            target: tensor with shape (B, C, T)

        Returns:
            Original tensors if self.channel is None, shape (B, C, T).
            A single channel from input tensors if self.channel is set, shape (B, T)
        """
        if self._channel is None:
            return preds, target
        else:
            return preds[:, self._channel, ...], target[:, self._channel, ...]

    @staticmethod
    def _trim_inputs(
        preds: torch.Tensor, target: torch.Tensor, input_length: torch.Tensor
    ) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        """Trim input tensors to input_length samples.

        Args:
            preds: tensor with shape (B, C, T)
            target: tensor with shape (B, C, T)

        Returns:
            An iterable with tuples of (preds, target) with
            the correct length.
        """
        # Each example has a different length
        for b_idx, b_len in enumerate(input_length):
            b_preds = preds[b_idx, ..., :b_len]
            b_target = target[b_idx, ..., :b_len]

            yield b_preds, b_target

    @staticmethod
    def _batch_reduction(batch_values: List[torch.Tensor]) -> torch.Tensor:
        """Reduce metric values for each example in a batch to a single
        value for the whole batch.

        Args:
            batch_values: list of metric values for each example in a batch

        Returns:
            Average metric value over the batch.
        """
        return sum(batch_values) / len(batch_values)

    def update(self, preds: torch.Tensor, target: torch.Tensor, input_length: Optional[torch.Tensor] = None) -> None:
        """Update the underlying metric by taking into account channel selector and input length.

        Args:
            preds: tensor with predictions, shape (B, C, T)
            target: tensor with target signals, shape (B, C, T)
            input_length: Optional, input tensor with length (in samples) of each signal in the batch, shape (B,).
                          If not provided, it is assumed that all samples are valid.
        """
        preds, target = self._select_channel(preds=preds, target=target)

        if input_length is None:
            self._metric.update(preds=preds, target=target)
        else:
            # Each example in this batch has a different length
            for b_preds, b_target in self._trim_inputs(preds=preds, target=target, input_length=input_length):
                self._metric.update(preds=b_preds, target=b_target)

    def compute(self) -> torch.Tensor:
        """Compute the underlying metric.
        """
        return self._metric.compute()

    def forward(
        self, preds: torch.Tensor, target: torch.Tensor, input_length: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Call underlying forward method to add the batch statistics to the accumulated metric state
        and return the result for the current batch.

        Args:
            preds: tensor with predictions, shape (B, C, T)
            target: tensor with target signals, shape (B, C, T)
            input_length: Optional, input tensor with length (in samples) of each signal in the batch, shape (B,).
                          If not provided, it is assumed that all samples are valid.

        Returns:
            Underlying metric averaged on the current batch.
        """
        preds, target = self._select_channel(preds=preds, target=target)

        if input_length is None:
            return self._metric(preds=preds, target=target)
        else:
            # Each example in this batch has a different length
            batch_values = []
            for b_preds, b_target in self._trim_inputs(preds=preds, target=target, input_length=input_length):
                batch_values.append(self._metric(preds=b_preds, target=b_target))
            # Average over the batch
            return self._batch_reduction(batch_values)

    def reset(self) -> None:
        """Reset the underlying metric.
        """
        self._metric.reset()

    def __repr__(self) -> str:
        """Return string representation of the object.
        """
        _op_metric = f"(metric: {repr(self._metric)}, channel: {self._channel})"
        repr_str = self.__class__.__name__ + _op_metric

        return repr_str

    def _wrap_compute(self, compute: Callable) -> Callable:
        """Overwrite to do nothing, as in CompositionalMetric.
        """
        return compute
