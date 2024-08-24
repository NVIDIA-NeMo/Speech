# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from abc import ABC, abstractmethod
from typing import Tuple

import einops
import torch

from nemo.collections.common.parts.utils import mask_sequence_tensor
from nemo.utils import logging


class ConditionalFlow(ABC):
    """
    Abstract class for different conditional flow-matching (CFM) classes

    Time horizon is [time_min, time_max (should be 1)]

    every path is "conditioned" on endpoints of the path
    endpoints are just our paired data samples
    subclasses need to implement mean, std, and vector_field

    """

    def __init__(self, time_min: float = 1e-8, time_max: float = 1.0):
        self.time_min = time_min
        self.time_max = time_max

    @abstractmethod
    def mean(self, *, time: torch.Tensor, x_start: torch.Tensor, x_end: torch.Tensor) -> torch.Tensor:
        """
        Return the mean of p_t(x | x_start, x_end) at time t
        """
        pass

    @abstractmethod
    def std(self, *, time: torch.Tensor, x_start: torch.Tensor, x_end: torch.Tensor) -> torch.Tensor:
        """
        Return the standard deviation of p_t(x | x_start, x_end) at time t
        """
        pass

    @abstractmethod
    def vector_field(
        self, *, time: torch.Tensor, x_start: torch.Tensor, x_end: torch.Tensor, point: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the conditional vector field v_t( point | x_start, x_end)
        """
        pass

    @staticmethod
    def _broadcast_time(time: torch.Tensor, n_dim: int) -> torch.Tensor:
        """
        Broadcast time tensor to the desired number of dimensions
        """
        if time.ndim == 1:
            target_shape = ' '.join(['B'] + ['1'] * (n_dim - 1))
            time = einops.rearrange(time, f'B -> {target_shape}')

        return time

    def generate_time(self, batch_size: int) -> torch.Tensor:
        """
        Randomly sample a batchsize of time_steps from U[0~1]
        """
        return torch.clamp(torch.rand((batch_size,)), self.time_min, self.time_max)

    def sample(self, *, time: torch.Tensor, x_start: torch.Tensor, x_end: torch.Tensor) -> torch.Tensor:
        """
        Generate a sample from p_t(x | x_start, x_end) at time t.
        Note that this implementation assumes all path marginals are normally distributed.
        """
        time = self._broadcast_time(time, n_dim=x_start.ndim)

        mean = self.mean(time=time, x_start=x_start, x_end=x_end)
        std = self.std(time=time, x_start=x_start, x_end=x_end)
        return mean + std * torch.randn_like(mean)

    def flow(
        self, *, time: torch.Tensor, x_start: torch.Tensor, x_end: torch.Tensor, point: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the conditional flow phi_t( point | x_start, x_end).
        This is an affine flow.
        """
        mean = self.mean(time=time, x_start=x_start, x_end=x_end)
        std = self.std(time=time, x_start=x_start, x_end=x_end)
        return mean + std * (point - x_start)


class OptimalTransportFlow(ConditionalFlow):
    """The OT-CFM model from [Lipman et at, 2023]

    Every conditional path the following holds:
    p_0 = N(x_start, sigma_start)
    p_1 = N(x_end, sigma_end),

    mean(x, t) = (time_max - t) * x_start + t * x_end
        (linear interpolation between x_start and x_end)

    std(x, t) = (time_max - t) * sigma_start + t * sigma_end

    Every conditional path is optimal transport map from p_0(x_start, x_end) to p_1(x_start, x_end)
    Marginal path is not guaranteed to be an optimal transport map from p_0 to p_1

    To get the OT-CFM model from [Lipman et at, 2023] just pass zeroes for x_start
    To get the I-CFM model, set sigma_min=sigma_max
    To get the rectified flow model, set sigma_min=sigma_max=0

    Args:
        time_min: minimum time value used in the process
        time_max: maximum time value used in the process
        sigma_start: the standard deviation of the initial distribution
        sigma_end: the standard deviation of the target distribution
    """

    def __init__(
        self, time_min: float = 1e-8, time_max: float = 1.0, sigma_start: float = 1.0, sigma_end: float = 1e-4
    ):
        super().__init__(time_min=time_min, time_max=time_max)
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\ttime_min:       %s', self.time_min)
        logging.debug('\ttime_max:       %s', self.time_max)
        logging.debug('\tsgima_start:    %s', self.sigma_start)
        logging.debug('\tsigma_end:      %s', self.sigma_end)

    def mean(self, *, x_start: torch.Tensor, x_end: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return (self.time_max - time) * x_start + time * x_end

    def std(self, *, x_start: torch.Tensor, x_end: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return (self.time_max - time) * self.sigma_start + time * self.sigma_end

    def vector_field(
        self,
        *,
        x_start: torch.Tensor,
        x_end: torch.Tensor,
        time: torch.Tensor,
        point: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        time = self._broadcast_time(time, n_dim=x_start.ndim)

        if self.sigma_start == self.sigma_end:
            return x_end - x_start

        num = self.sigma_end * (point - x_start) - self.sigma_start * (point - x_end)
        denom = (1 - time) * self.sigma_start + time * self.sigma_end
        return num / (denom + eps)


class ConditionalFlowMatchingSampler(ABC):
    """
    Abstract class for different sampler to solve the ODE in CFM

    Args:
        estimator: the NN-based conditional vector field estimator
        num_steps: How many time steps to iterate in the process
        time_min: minimum time value used in the process
        time_max: maximum time value used in the process

    """

    def __init__(
        self,
        estimator: torch.nn.Module,
        num_steps: int = 5,
        time_min: float = 1e-8,
        time_max: float = 1.0,
    ):
        self.estimator = estimator
        self.num_steps = num_steps
        self.time_min = time_min
        self.time_max = time_max

    @property
    def time_step(self):
        return (self.time_max - self.time_min) / self.num_steps

    @abstractmethod
    def forward(
        self, state: torch.Tensor, estimator_condition: torch.Tensor, state_length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class ConditionalFlowMatchingEulerSampler(ConditionalFlowMatchingSampler):
    """
    The Euler Sampler for solving the ODE in CFM on a uniform time grid
    """

    def __init__(
        self,
        estimator: torch.nn.Module,
        num_steps: int = 5,
        time_min: float = 1e-8,
        time_max: float = 1.0,
    ):
        super().__init__(
            estimator=estimator,
            num_steps=num_steps,
            time_min=time_min,
            time_max=time_max,
        )
        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tnum_steps:      %s', self.num_steps)
        logging.debug('\ttime_min:       %s', self.time_min)
        logging.debug('\ttime_max:       %s', self.time_max)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @torch.inference_mode()
    def forward(
        self, state: torch.Tensor, estimator_condition: torch.Tensor, state_length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        time_steps = torch.linspace(self.time_min, self.time_max, self.num_steps + 1)

        if state_length is not None:
            state = mask_sequence_tensor(state, state_length)

        for t in time_steps:
            time = t * torch.ones(state.shape[0], device=state.device)

            if estimator_condition is None:
                estimator_input = state
            else:
                estimator_input = torch.cat([state, estimator_condition], dim=1)

            vector_field, _ = self.estimator(input=estimator_input, input_length=state_length, condition=time)

            state = state + vector_field * self.time_step

            if state_length is not None:
                state = mask_sequence_tensor(state, state_length)

        return state, state_length
