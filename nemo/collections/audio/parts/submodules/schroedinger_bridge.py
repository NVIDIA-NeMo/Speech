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
import math
from abc import ABC, abstractmethod
from typing import Optional

import torch

from nemo.collections.common.parts.utils import mask_sequence_tensor
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types import LengthsType, NeuralType, SpectrogramType
from nemo.utils import logging


class SBNoiseSchedule(NeuralModule, ABC):
    """Noise schedule for the Schrödinger Bridge

    Args:
        time_min: minimum time for the process
        time_max: maximum time for the process
        num_steps: number of steps for the process
        eps: small regularization

    References:
        Schrödinger Bridge for Generative Speech Enhancement, https://arxiv.org/abs/2407.16074
    """

    def __init__(
        self,
        time_min: float = 0.0,
        time_max: float = 1.0,
        num_steps: int = 100,
        eps: float = 1e-8,
    ):
        super().__init__()

        # min and max time
        if time_min < 0:
            raise ValueError(f'time_min should be non-negative, current value {time_min}')

        if time_max <= time_min:
            raise ValueError(f'time_max should be larger than time_min, current max {time_max} and min {time_min}')

        self.time_min = time_min
        self.time_max = time_max

        if num_steps <= 0:
            raise ValueError(f'Expected num_steps > 0, got {num_steps}')

        self.num_steps = num_steps

        if eps <= 0:
            raise ValueError(f'Expected eps > 0, got {eps}')

        self.eps = eps

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\ttime_min:  %s', self.time_min)
        logging.debug('\ttime_max:  %s', self.time_max)
        logging.debug('\tnum_steps: %s', self.num_steps)
        logging.debug('\teps:       %s', self.eps)

    @property
    def dt(self) -> float:
        """Time step for the process."""
        return self.time_max / self.num_steps

    @property
    def time_delta(self) -> float:
        """Time range for the process."""
        return self.time_max - self.time_min

    def generate_time(self, size: int, device: torch.device) -> torch.Tensor:
        """Generate random time steps in the valid range."""
        time = torch.rand(size, device=device) * self.time_delta + self.time_min
        return time

    @property
    def alpha_t_max(self):
        """Return alpha_t at t_max."""
        t_max = torch.tensor([self.time_max], device=alpha.device)
        return self.alpha(t_max)

    @property
    def sigma_t_max(self):
        """Return sigma_t at t_max."""
        t_max = torch.tensor([self.time_max], device=alpha.device)
        return self.sigma(t_max)

    @abstractmethod
    def f(self, time: torch.Tensor) -> torch.Tensor:
        """Drift scaling f(t).

        Args:
            time: tensor with time steps

        Returns:
            Tensor the same size as time, representing drift scaling.
        """
        pass

    @abstractmethod
    def g(self, time: torch.Tensor) -> torch.Tensor:
        """Diffusion scaling g(t).

        Args:
            time: tensor with time steps

        Returns:
            Tensor the same size as time, representing diffusion scaling.
        """
        pass

    @abstractmethod
    def alpha(self, time: torch.Tensor) -> torch.Tensor:
        """Return alpha for SB noise schedule.

            alpha_t = exp( int_0^s f(s) ds  )

        Args:
            time: tensor with time steps

        Returns:
            Tensor the same size as time, representing alpha for each time.
        """
        pass

    def alpha_bar_from_alpha(self, alpha: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Return alpha_bar for SB.

            alpha_bar = alpha_t / alpha_t_max

        Args:
            alpha: tensor with alpha values

        Returns:
            Tensors the same size as alpha, representing alpha_bar and alpha_t_max.
        """
        alpha_t_max = self.alpha(torch.tensor([self.time_max], device=alpha.device))
        alpha_bar = alpha / (alpha_t_max + self.eps)
        return alpha_bar, alpha_t_max

    def get_alphas(self, time: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """Return alpha, alpha_bar and alpha_t_max for SB.

        Args:
            time: tensor with time steps

        Returns:
            Tuple of tensors with alpha, alpha_bar and alpha_t_max.
        """
        alpha = self.alpha(time)
        alpha_bar, alpha_t_max = self.alpha_bar_from_alpha(alpha)
        return alpha, alpha_bar, alpha_t_max

    @abstractmethod
    def sigma(self, time: torch.Tensor) -> torch.Tensor:
        """Return sigma_t for SB.

            sigma_t^2 = int_0^s g^2(s) / alpha_s^2 ds

        Args:
            time: tensor with time steps

        Returns:
            Tensor the same size as time, representing sigma for each time.
        """
        pass

    def sigma_bar_from_sigma(self, sigma: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Return sigma_bar_t for SB.

            sigma_bar_t^2 = sigma_t_max^2 - sigma_t^2

        Args:
            sigma: tensor with sigma values

        Returns:
            Tensors the same size as sigma, representing sigma_bar and sigma_t_max.
        """
        sigma_t_max = self.sigma(torch.tensor([self.time_max], device=sigma.device))
        sigma_bar_sq = sigma_t_max**2 - sigma**2
        return torch.sqrt(sigma_bar_sq + self.eps), sigma_t_max

    def get_sigmas(self, time: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """Return sigma, sigma_bar and sigma_t_max for SB.

        Args:
            time: tensor with time steps

        Returns:
            Tuple of tensors with sigma, sigma_bar and sigma_t_max.
        """
        sigma = self.sigma(time)
        sigma_bar, sigma_t_max = self.sigma_bar_from_sigma(sigma)
        return sigma, sigma_bar, sigma_t_max

    @abstractmethod
    def copy(self):
        """Return a copy of the noise schedule."""
        pass

    def __repr__(self):
        desc = f'{self.__class__.__name__}(time_min={self.time_min}, time_max={self.time_max}, num_steps={self.num_steps})'
        desc += f'\n\tdt:         {self.dt}'
        desc += f'\n\ttime_delta: {self.time_delta}'
        return desc


class SBNoiseScheduleVE(SBNoiseSchedule):
    """Variance exploding noise schedule for the Schrödinger Bridge.

    Args:
        k: defines the base for the exponential diffusion coefficient
        c: scaling for the diffusion coefficient
        time_min: minimum time for the process
        time_max: maximum time for the process
        num_steps: number of steps for the process
        eps: small regularization

    References:
        Schrödinger Bridge for Generative Speech Enhancement, https://arxiv.org/abs/2407.16074
    """

    def __init__(
        self,
        k: float,
        c: float,
        time_min: float = 0.0,
        time_max: float = 1.0,
        num_steps: int = 100,
        eps: float = 1e-8,
    ):
        super().__init__(time_min=time_min, time_max=time_max, num_steps=num_steps, eps=eps)

        # Shape parameters
        if k <= 1:
            raise ValueError(f'Expected k > 1, got {k}')

        if c <= 0:
            raise ValueError(f'Expected c > 0, got {c}')

        self.c = c
        self.k = k

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tk:         %s', self.k)
        logging.debug('\tc:         %s', self.c)
        logging.debug('\ttime_min:  %s', self.time_min)
        logging.debug('\ttime_max:  %s', self.time_max)
        logging.debug('\tnum_steps: %s', self.num_steps)
        logging.debug('\teps:       %s', self.eps)

    def f(self, time: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(time)

    def g(self, time: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.c) * self.k**self.time

    def alpha(self, time: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(time)

    def sigma(self, time: torch.Tensor) -> torch.Tensor:
        sigma_sq = self.c * (self.k ** (2 * time) - 1) / (2 * math.log(self.k) + self.eps)
        return torch.sqrt(sigma_sq)

    def copy(self):
        return SBNoiseScheduleVE(
            k=self.k,
            c=self.c,
            time_min=self.time_min,
            time_max=self.time_max,
            num_steps=self.num_steps,
            eps=self.eps,
        )

    def __repr__(self):
        desc = super().__repr__()
        desc += f'\n\tk: {self.k}'
        desc += f'\n\tc: {self.c}'
        return desc


class SBNoiseScheduleVP(SBNoiseSchedule):
    """Variance preserving noise schedule for the Schrödinger Bridge.

    Args:
        beta_0: defines the lower bound for diffusion coefficient
        beta_1: defines upper bound for diffusion coefficient
        c: scaling for the diffusion coefficient
        time_min: minimum time for the process
        time_max: maximum time for the process
        num_steps: number of steps for the process
        eps: small regularization
    """

    def __init__(
        self,
        beta_0: float,
        beta_1: float,
        c: float = 1.0,
        time_min: float = 0.0,
        time_max: float = 1.0,
        num_steps: int = 100,
        eps: float = 1e-8,
    ):
        super().__init__(time_min=time_min, time_max=time_max, num_steps=num_steps, eps=eps)

        # Shape parameters
        if beta_0 < 0:
            raise ValueError(f'Expected beta_0 >= 0, got {beta_0}')

        if beta_1 < 0:
            raise ValueError(f'Expected beta_1 >= 0, got {beta_1}')

        if beta_0 >= beta_1:
            raise ValueError(f'Expected beta_0 < beta_1, got beta_0={beta_0} and beta_1={beta_1}')

        if c <= 0:
            raise ValueError(f'Expected c > 0, got {c}')

        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.c = c

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tbeta_0:    %s', self.beta_0)
        logging.debug('\tbeta_1:    %s', self.beta_1)
        logging.debug('\tc:         %s', self.c)
        logging.debug('\ttime_min:  %s', self.time_min)
        logging.debug('\ttime_max:  %s', self.time_max)
        logging.debug('\tnum_steps: %s', self.num_steps)
        logging.debug('\teps:       %s', self.eps)

    def f(self, time: torch.Tensor) -> torch.Tensor:
        return -0.5 * (self.beta_0 + time * (self.beta_1 - self.beta_0))

    def g(self, time: torch.Tensor) -> torch.Tensor:
        g_sq = self.c * (self.beta_0 + time * (self.beta_1 - self.beta_0))
        return torch.sqrt(g_sq)

    def alpha(self, time: torch.Tensor) -> torch.Tensor:
        tmp = self.beta_0 * time + (self.beta_1 - self.beta_0) / 2 * time**2
        return torch.exp(-0.5 * tmp)

    def sigma(self, time: torch.Tensor) -> torch.Tensor:
        sigma_sq = self.beta_0 * time + (self.beta_1 - self.beta_0) / 2 * time**2
        sigma_sq = torch.exp(sigma_sq) - 1
        sigma_sq = self.c * sigma_sq
        return torch.sqrt(sigma_sq)

    def copy(self):
        return SBNoiseScheduleVP(
            beta_0=self.beta_0,
            beta_1=self.beta_1,
            c=self.c,
            time_min=self.time_min,
            time_max=self.time_max,
            num_steps=self.num_steps,
            eps=self.eps,
        )

    def __repr__(self):
        desc = super().__repr__()
        desc += f'\n\tbeta_0: {self.beta_0}'
        desc += f'\n\tbeta_1: {self.beta_1}'
        desc += f'\n\tc:      {self.c}'
        return desc


class SBSampler(NeuralModule):
    """Schrödinger Bridge sampler.

    Args:
        noise_schedule: noise schedule for the bridge
        estimator: neural estimator
        estimator_output: defines the output of the estimator, e.g., data_prediction
        estimator_time: time for conditioning the estimator, e.g., 'current'
                        or 'previous'. Default is 'previous'.
        process: defines the process, e.g., sde or ode
        time_max: maximum time for the process
        time_min: minimum time for the process
        num_steps: number of steps for the process
        eps: small regularization to prevent division by zero

    References:
        Schrödinger Bridge for Generative Speech Enhancement, https://arxiv.org/abs/2407.16074
        Schrodinger Bridges Beat Diffusion Models on Text-to-Speech Synthesis, https://arxiv.org/abs/2312.03491
    """

    def __init__(
        self,
        noise_schedule: SBNoiseSchedule,
        estimator: NeuralModule,  # neural estimator
        estimator_output: str,
        estimator_time: str = 'previous',  # time for the estimator
        process: str = 'sde',
        time_max: Optional[float] = None,
        time_min: Optional[float] = None,
        num_steps: int = 50,
        eps: float = 1e-8,
    ):
        super().__init__()
        # Create a copy of the noise schedule
        self.noise_schedule = noise_schedule.copy()

        # Update sampling parameters
        if time_max is not None:
            self.noise_schedule.time_max = time_max
            logging.info('noise_schedule.time_max set to: %s', self.noise_schedule.time_max)

        if time_min is not None:
            self.noise_schedule.time_min = time_min
            logging.info('noise_schedule.time_min set to: %s', self.noise_schedule.time_min)

        self.noise_schedule.num_steps = num_steps
        logging.info('noise_schedule.num_steps set to: %s', self.noise_schedule.num_steps)

        # Estimator
        self.estimator = estimator
        self.estimator_output = estimator_output
        self.estimator_time = estimator_time

        # Sampling process
        self.process = process

        # Small regularization
        if eps <= 0:
            raise ValueError(f'Expected eps > 0, got {eps}')
        self.eps = eps

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\testimator_output: %s', self.estimator_output)
        logging.debug('\testimator_time:   %s', self.estimator_time)
        logging.debug('\tprocess:          %s', self.process)
        logging.debug('\ttime_min:         %s', self.time_min)
        logging.debug('\ttime_max:         %s', self.time_max)
        logging.debug('\tnum_steps:        %s', self.num_steps)
        logging.debug('\teps:              %s', self.eps)

    @property
    def time_max(self):
        return self.noise_schedule.time_max

    @time_max.setter
    def time_max(self, value: float):
        self.noise_schedule.time_max = value
        logging.debug('noise_schedule.time_max set to: %s', self.noise_schedule.time_max)

    @property
    def time_min(self):
        return self.noise_schedule.time_min

    @time_min.setter
    def time_min(self, value: float):
        self.noise_schedule.time_min = value
        logging.debug('noise_schedule.time_min set to: %s', self.noise_schedule.time_min)

    @property
    def num_steps(self):
        return self.noise_schedule.num_steps

    @num_steps.setter
    def num_steps(self, value: int):
        self.noise_schedule.num_steps = value
        logging.debug('noise_schedule.num_steps set to: %s', self.noise_schedule.num_steps)

    @property
    def process(self):
        return self._process

    @process.setter
    def process(self, value: str):
        if value not in ['sde', 'ode']:
            raise ValueError(f'Unexpected process: {value}')
        self._process = value
        logging.info('process set to: %s', self._process)

    @property
    def estimator_time(self):
        return self._estimator_time

    @estimator_time.setter
    def estimator_time(self, value: str):
        if value not in ['current', 'previous']:
            raise ValueError(f'Unexpected estimator time: {value}')
        self._estimator_time = value
        logging.info('estimator time set to: %s', self._estimator_time)

    @typecheck(
        input_types={
            "prior_mean": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "estimator_condition": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType(), optional=True),
            "state_length": NeuralType(tuple('B'), LengthsType(), optional=True),
        },
        output_types={
            "sample": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "state_length": NeuralType(tuple('B'), LengthsType(), optional=True),
        },
    )
    @torch.inference_mode()
    def forward(
        self, prior_mean: torch.Tensor, estimator_condition: torch.Tensor, state_length: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Takes prior mean and generates a sample."""
        # SB starts from the prior mean
        state = prior_mean

        if state_length is not None:
            state = mask_sequence_tensor(state, state_length)

        # Time steps for sampling
        time_steps = torch.linspace(self.time_max, self.time_min, self.num_steps + 1, device=state.device)

        # Initial values
        time_prev = time_steps[0] * torch.ones(state.shape[0], device=state.device)
        alpha_prev, _, alpha_t_max = self.noise_schedule.get_alphas(time_prev)
        sigma_prev, sigma_bar_prev, sigma_t_max = self.noise_schedule.get_sigmas(time_prev)

        # Sampling
        # Sample at the initial time step (`self.time_max`) is exactly the prior_mean.
        # We do not need to estimate it, but we need to pass it to the next time step.
        # We iterate through the following time steps to generate the sample at the final time (`self.time_min`).
        for t in time_steps[1:]:

            # Prepare time steps for the whole batch
            time = t * torch.ones(state.shape[0], device=state.device)

            # Prepare input for estimator, concatenate conditioning along the channel dimension
            estimator_input = state if estimator_condition is None else torch.cat([state, estimator_condition], dim=1)
            estimator_time = time if self.estimator_time == 'current' else time_prev

            # Estimator
            if self.estimator_output == 'data_prediction':
                current_estimate, _ = self.estimator(
                    input=estimator_input, input_length=state_length, condition=estimator_time
                )
            else:
                raise NotImplementedError(f'Unexpected estimator output: {self.estimator_output}')

            # Get noise schedule for current time
            alpha_t, alpha_bar_t, _ = self.noise_schedule.get_alphas(time)
            sigma_t, sigma_bar_t, _ = self.noise_schedule.get_sigmas(time)

            if self.process == 'sde':
                # Calculate scaling for the first-order discretization from the paper
                weight_prev = alpha_t * sigma_t**2 / (alpha_prev * sigma_prev**2 + self.eps)
                tmp = 1 - sigma_t**2 / (sigma_prev**2 + self.eps)
                weight_estimate = alpha_t * tmp
                weight_z = alpha_t * sigma_t * torch.sqrt(tmp)

                # View as [B, C, D, T]
                weight_prev = weight_prev.view(-1, 1, 1, 1)
                weight_estimate = weight_estimate.view(-1, 1, 1, 1)
                weight_z = weight_z.view(-1, 1, 1, 1)

                # Random sample
                z_norm = torch.randn_like(state)

                # Update state: weighted sum of previous state, current estimate and noise
                state = weight_prev * state + weight_estimate * current_estimate + weight_z * z_norm
            elif self.process == 'ode':
                # Calculate scaling for the first-order discretization from the paper
                weight_prev = alpha_t * sigma_t * sigma_bar_t / (alpha_prev * sigma_prev * sigma_bar_prev + self.eps)
                weight_estimate = (
                    alpha_t
                    / (sigma_t_max**2 + self.eps)
                    * (sigma_bar_t**2 - sigma_bar_prev * sigma_t * sigma_bar_t / (sigma_prev + self.eps))
                )
                weight_prior_mean = (
                    alpha_t
                    / (alpha_t_max * sigma_t_max**2 + self.eps)
                    * (sigma_t**2 - sigma_prev * sigma_t * sigma_bar_t / (sigma_bar_prev + self.eps))
                )

                # View as [B, C, D, T]
                weight_prev = weight_prev.view(-1, 1, 1, 1)
                weight_estimate = weight_estimate.view(-1, 1, 1, 1)
                weight_prior_mean = weight_prior_mean.view(-1, 1, 1, 1)

                # Update state: weighted sum of previous state, current estimate and prior
                state = weight_prev * state + weight_estimate * current_estimate + weight_prior_mean * prior_mean
            else:
                raise RuntimeError(f'Unexpected process: {self.process}')

            # Save previous values
            time_prev = time
            alpha_prev = alpha_t
            sigma_prev = sigma_t
            sigma_bar_prev = sigma_bar_t

        # Final output
        if state_length is not None:
            state = mask_sequence_tensor(state, state_length)

        return state, state_length
