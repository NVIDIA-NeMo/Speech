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
from typing import Optional, Tuple, Type

import numpy as np
import torch

from nemo.collections.common.parts.utils import mask_sequence_tensor
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types import FloatType, LengthsType, NeuralType, SpectrogramType, VoidType
from nemo.utils import logging


class StochasticDifferentialEquation(NeuralModule, ABC):
    """Base class for stochastic differential equations."""

    def __init__(self, time_min: float, time_max: float, num_steps: int):
        super().__init__()

        # min and max time
        if time_min <= 0:
            raise ValueError(f'time_min should be positive, current value {time_min}')

        if time_max <= time_min:
            raise ValueError(f'time_max should be larger than time_min, current max {time_max} and min {time_min}')

        self.time_min = time_min
        self.time_max = time_max

        # number of steps
        if num_steps <= 0:
            raise ValueError(f'num_steps needs to be positive: current value {num_steps}')

        self.num_steps = num_steps

    @property
    def dt(self) -> float:
        """Time step for this SDE.
        This denotes the step size between `0` and `self.time_max` when using `self.num_steps`.
        """
        return self.time_max / self.num_steps

    @property
    def time_delta(self) -> float:
        """Time range for this SDE."""
        return self.time_max - self.time_min

    def generate_time(self, size: int, device: torch.device) -> torch.Tensor:
        """Generate random time steps in the valid range.

        Time steps are generated between `self.time_min` and `self.time_max`.

        Args:
            size: number of samples
            device: device to use

        Returns:
            A tensor of floats with shape (size,)
        """
        time = torch.rand(size, device=device) * self.time_delta + self.time_min
        return time

    @abstractmethod
    def coefficients(self, state: torch.Tensor, time: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: tensor of shape (B, C, D, T)
            time: tensor of shape (B,)

        Returns:
            Tuple with drift and diffusion coefficients.
        """
        pass

    @typecheck(
        input_types={
            "prior_mean": NeuralType(('B', 'C', 'D', 'T'), VoidType()),
        },
        output_types={
            "sample": NeuralType(('B', 'C', 'D', 'T'), VoidType()),
        },
    )
    @abstractmethod
    def prior_sampling(self, prior_mean: torch.Tensor) -> torch.Tensor:
        """Generate a sample from the prior distribution p_T.

        Args:
            prior_mean: Mean of the prior distribution

        Returns:
            A sample from the prior distribution.
        """
        pass

    def discretize(
        self, *, state: torch.Tensor, time: torch.Tensor, state_length: Optional[torch.Tensor] = None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Assume we have the following SDE:

            dx = drift(x, t) * dt + diffusion(x, t) * dwt

        where `wt` is the standard Wiener process.

        We assume the following discretization:

            new_state = current_state + total_drift + total_diffusion * z_norm

        where `z_norm` is sampled from normal distribution with zero mean and unit variance.

        Args:
            state: current state of the process, shape (B, C, D, T)
            time: current time of the process, shape (B,)
            state_length: length of the valid time steps for each example in the batch, shape (B,)
            **kwargs: other parameters

        Returns:
            Drift and diffusion.
        """
        # Get coefficients
        drift_coefficient, diffusion_coefficient = self.coefficients(
            state=state, time=time, state_length=state_length, **kwargs
        )

        # Discretized drift
        drift = drift_coefficient * self.dt

        # Note:
        # Scale with sqrt(dt) because z_norm is sampled from a normal distribution with zero mean and
        # unit variance and dwt is normally distributed with zero mean and variance dt
        diffusion = diffusion_coefficient * np.sqrt(self.dt)

        return drift, diffusion

    @abstractmethod
    def copy(self):
        """Create a copy of this SDE."""
        pass

    def __repr__(self):
        desc = f'{self.__class__.__name__}(time_min={self.time_min}, time_max={self.time_max}, num_steps={self.num_steps})'
        desc += f'\n\tdt:         {self.dt}'
        desc += f'\n\ttime_delta: {self.time_delta}'
        return desc


class OrnsteinUhlenbeckVarianceExplodingSDE(StochasticDifferentialEquation):
    """This class implements the Ornstein-Uhlenbeck SDE with variance exploding noise schedule.

    The SDE is given by:

        dx = theta * (y - x) dt + g(t) dw

    where `theta` is the stiffness parameter and `g(t)` is the diffusion coefficient:

        g(t) = std_min * (std_max/std_min)^t * sqrt(2 * log(std_max/std_min))

    References:
        Richter et al., Speech Enhancement and Dereverberation with Diffusion-based Generative Models, Tr. ASLP 2023
    """

    def __init__(
        self,
        stiffness: float,
        std_min: float,
        std_max: float,
        num_steps: int = 100,
        time_min: float = 3e-2,
        time_max: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__(time_min=time_min, time_max=time_max, num_steps=num_steps)

        # Small regularization
        if eps <= 0:
            raise ValueError(f'eps should be positive, current value {eps}')
        self.eps = eps

        # stifness
        self.stiffness = stiffness

        # noise schedule
        if std_min <= 0:
            raise ValueError(f'std_min should be positive, current value {std_min}')

        if std_max <= std_min:
            raise ValueError(f'std_max should be larger than std_min, current max {std_max} and min {std_min}')

        self.std_min = std_min
        self.std_max = std_max

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tstiffness:     %s', self.stiffness)
        logging.debug('\tstd_min:       %s', self.std_min)
        logging.debug('\tstd_max:       %s', self.std_max)
        logging.debug('\tnum_steps:     %s', self.num_steps)
        logging.debug('\ttime_min:      %s', self.time_min)
        logging.debug('\ttime_max:      %s', self.time_max)
        logging.debug('\teps:           %s', self.eps)

    @property
    def std_ratio(self) -> float:
        return self.std_max / (self.std_min + self.eps)

    @property
    def log_std_ratio(self) -> float:
        return np.log(self.std_ratio + self.eps)

    @typecheck(
        input_types={
            "state": NeuralType(('B', 'C', 'D', 'T'), VoidType()),
            "prior_mean": NeuralType(('B', 'C', 'D', 'T'), VoidType()),
            "time": NeuralType(tuple('B'), FloatType()),
        },
        output_types={
            "mean": NeuralType(('B', 'C', 'D', 'T'), FloatType()),
        },
    )
    def perturb_kernel_mean(self, state: torch.Tensor, prior_mean: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Return the mean of the perturbation kernel for this SDE.

        Args:
            state: current state of the process, shape (B, C, D, T)
            prior_mean: mean of the prior distribution
            time: current time of the process, shape (B,)

        Returns:
            A tensor of shape (B, C, D, T)
        """
        # exponential weighting
        weight = torch.exp(-self.stiffness * time)

        # view as [B, C, D, T]
        weight = weight.view(-1, 1, 1, 1)

        # closed-form mean
        mean = weight * state + (1 - weight) * prior_mean

        return mean

    @typecheck(
        input_types={
            "time": NeuralType(tuple('B'), FloatType()),
        },
        output_types={
            "std": NeuralType(tuple('B'), FloatType()),
        },
    )
    def perturb_kernel_std(self, time: torch.Tensor) -> torch.Tensor:
        """Return the standard deviation of the perturbation kernel for this SDE.

        Note that the standard deviation depends on the time and the noise schedule,
        which is parametrized using `self.stiffness`, `self.std_min` and `self.std_max`.

        Args:
            time: current time of the process, shape (B,)

        Returns:
            A tensor of shape (B,)
        """
        var = (self.std_min**2) * self.log_std_ratio
        var *= torch.pow(self.std_ratio, 2 * time) - torch.exp(-2 * self.stiffness * time)
        var /= self.stiffness + self.log_std_ratio
        std = torch.sqrt(var)
        return std

    @typecheck(
        input_types={
            "state": NeuralType(('B', 'C', 'D', 'T'), VoidType()),
            "prior_mean": NeuralType(('B', 'C', 'D', 'T'), VoidType()),
            "time": NeuralType(tuple('B'), FloatType()),
        },
        output_types={
            "mean": NeuralType(('B', 'C', 'D', 'T'), FloatType()),
            "std": NeuralType(('B', 'C', 'D', 'T'), FloatType()),
        },
    )
    def perturb_kernel_params(self, state: torch.Tensor, prior_mean: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Return the mean and standard deviation of the perturbation kernel for this SDE.

        Args:
            state: current state of the process, shape (B, C, D, T)
            prior_mean: mean of the prior distribution
            time: current time of the process, shape (B,)
        """
        assert torch.all(time <= self.time_max)
        assert torch.all(time >= self.time_min)

        # compute the mean
        mean = self.perturb_kernel_mean(state=state, prior_mean=prior_mean, time=time)

        # compute the standard deviation
        std = self.perturb_kernel_std(time=time)
        # view as [B, C, D, T]
        std = std.view(-1, 1, 1, 1)

        return mean, std

    @typecheck(
        input_types={
            "state": NeuralType(('B', 'C', 'D', 'T'), VoidType()),
            "time": NeuralType(tuple('B'), VoidType()),
            "prior_mean": NeuralType(('B', 'C', 'D', 'T'), VoidType()),
            "state_length": NeuralType(tuple('B'), LengthsType(), optional=True),
        },
        output_types={
            "drift_coefficient": NeuralType(('B', 'C', 'D', 'T'), FloatType()),
            "diffusion_coefficient": NeuralType(('B', 'C', 'D', 'T'), FloatType()),
        },
    )
    def coefficients(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        prior_mean: torch.Tensor,
        state_length: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute drift and diffusion coefficients for this SDE.

        Args:
            state: current state of the process, shape (B, C, D, T)
            time: current time of the process, shape (B,)
            prior_mean: mean of the prior distribution
            state_length: length of the valid time steps for each example in the batch

        Returns:
            Drift and diffusion coefficients.
        """
        # Drift coefficient
        drift_coefficient = self.stiffness * (prior_mean - state)

        # Diffusion coefficient
        diffusion_coefficient = self.std_min * torch.pow(self.std_ratio, time) * np.sqrt(2 * self.log_std_ratio)
        # View in the same shape as the state
        diffusion_coefficient = diffusion_coefficient.view(-1, *([1] * (state.dim() - 1)))

        if state_length is not None:
            drift_coefficient = mask_sequence_tensor(drift_coefficient, state_length)
            diffusion_coefficient = mask_sequence_tensor(diffusion_coefficient, state_length)

        return drift_coefficient, diffusion_coefficient

    def prior_sampling(self, prior_mean: torch.Tensor) -> torch.Tensor:
        """Generate a sample from the prior distribution p_T.

        Args:
            prior_mean: Mean of the prior distribution
        """
        # Final time step for all samples in the batch
        time = self.time_max * torch.ones(prior_mean.shape[0], device=prior_mean.device)

        # Compute the std of the prior distribution
        std = self.perturb_kernel_std(time=time)

        # view as [B, C, D, T]
        std = std.view(-1, 1, 1, 1)

        # Generate a sample from a normal distribution centered at prior_mean
        sample = prior_mean + torch.randn_like(prior_mean) * std

        return sample

    def copy(self):
        return OrnsteinUhlenbeckVarianceExplodingSDE(
            stiffness=self.stiffness,
            std_min=self.std_min,
            std_max=self.std_max,
            num_steps=self.num_steps,
            time_min=self.time_min,
            time_max=self.time_max,
            eps=self.eps,
        )

    def __repr__(self):
        desc = f'{self.__class__.__name__}(stiffness={self.stiffness}, std_min={self.std_min}, std_max={self.std_max}, num_steps={self.num_steps}, time_min={self.time_min}, time_max={self.time_max}, eps={self.eps})'
        desc += f'\n\tdt:         {self.dt}'
        desc += f'\n\ttime_delta: {self.time_delta}'
        desc += f'\n\tstd_ratio:  {self.std_ratio}'
        desc += f'\n\tlog_std_ratio:  {self.log_std_ratio}'

        return desc


class ReverseStochasticDifferentialEquation(StochasticDifferentialEquation):
    def __init__(self, *, sde: Type[StochasticDifferentialEquation], score_estimator: Type[NeuralModule]):
        """Use the forward SDE and a score estimator to define the reverse SDE.

        Args:
            sde: forward SDE
            score_estimator: neural score estimator
        """
        super().__init__(time_min=sde.time_min, time_max=sde.time_max, num_steps=sde.num_steps)
        self.score_estimator = score_estimator
        self.forward_sde = sde

        logging.debug('Initialized %s', self.__class__.__name__)

    def coefficients(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        score_condition: Optional[torch.Tensor] = None,
        state_length: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute drift and diffusion coefficients for the reverse SDE.

        Args:
            state: current state of the process, shape (B, C, D, T)
            time: current time of the process, shape (B,)
        """
        raise NotImplementedError('Coefficients not necessary for the reverse SDE.')

    def prior_sampling(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        """Prior sampling is not necessary for the reverse SDE."""
        raise NotImplementedError('Prior sampling not necessary for the reverse SDE.')

    def discretize(
        self,
        *,
        state: torch.Tensor,
        time: torch.Tensor,
        score_condition: Optional[torch.Tensor] = None,
        state_length: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Discretize the reverse SDE.

        Args:
            state: current state of the process, shape (B, C, D, T)
            time: current time of the process, shape (B,)
            score_condition: condition for the score estimator
            state_length: length of the valid time steps for each example in the batch
            **kwargs: other parameters for discretization of the forward SDE
        """
        # Drift and diffusion from the forward SDE
        forward_drift, forward_diffusion = self.forward_sde.discretize(state=state, time=time, **kwargs)

        # For input for the score estimator:
        # - if no condition is provided, use the state
        # - if a condition is provided, concatenate the state and the condition along the channel dimension
        score_input = state if score_condition is None else torch.cat([state, score_condition], dim=1)

        # Estimate score
        score, _ = self.score_estimator(input=score_input, input_length=state_length, condition=time)

        # Adjust drift
        drift = forward_drift - forward_diffusion.pow(2) * score

        # Adjust diffusion
        diffusion = forward_diffusion

        if state_length is not None:
            drift = mask_sequence_tensor(drift, state_length)
            diffusion = mask_sequence_tensor(diffusion, state_length)

        return drift, diffusion

    def copy(self):
        return ReverseStochasticDifferentialEquation(sde=self.forward_sde.copy(), score_estimator=self.score_estimator)

    def __repr__(self):
        desc = f'{self.__class__.__name__}(sde={self.forward_sde}, score_estimator={self.score_estimator})'
        return desc


class PredictorCorrectorSampler(NeuralModule):
    """Predictor-Corrector sampler for the reverse SDE.

    Args:
        sde: forward SDE
        score_estimator: neural score estimator
        predictor: predictor for the reverse process
        corrector: corrector for the reverse process
        num_steps: number of time steps for the reverse process
        num_corrector_steps: number of corrector steps
        time_max: maximum time
        time_min: minimum time
        snr: SNR for Annealed Langevin Dynamics
        output_type: type of the output ('state' for the final state, or 'mean' for the mean of the final state)

    References:
        - Song et al., Score-based generative modeling through stochastic differential equations, 2021
    """

    def __init__(
        self,
        sde,
        score_estimator,
        predictor: str = 'reverse_diffusion',
        corrector: str = 'annealed_langevin_dynamics',
        num_steps: int = 50,
        num_corrector_steps: int = 1,
        time_max: Optional[float] = None,
        time_min: Optional[float] = None,
        snr: float = 0.5,
        output_type: str = 'mean',
    ):
        super().__init__()
        # Create a copy of SDE
        self.sde = sde.copy()

        # Update SDE parameters for sampling
        if time_max is not None:
            self.sde.time_max = time_max
            logging.info('sde.time_max set to: %s', self.sde.time_max)

        if time_min is not None:
            self.sde.time_min = time_min
            logging.info('sde.time_min set to: %s', self.sde.time_min)

        self.sde.num_steps = num_steps
        logging.info('sde.num_steps set to: %s', self.sde.num_steps)

        # Update local values
        self.time_max = self.sde.time_max
        self.time_min = self.sde.time_min
        self.num_steps = self.sde.num_steps

        # Predictor setup
        if predictor == 'reverse_diffusion':
            self.predictor = ReverseDiffusionPredictor(sde=self.sde, score_estimator=score_estimator)
        else:
            raise RuntimeError(f'Unexpected predictor: {predictor}')

        # Corrector setup
        if corrector == 'annealed_langevin_dynamics':
            self.corrector = AnnealedLangevinDynamics(
                sde=self.sde, score_estimator=score_estimator, snr=snr, num_steps=num_corrector_steps
            )
        else:
            raise RuntimeError(f'Unexpected corrector: {corrector}')

        if output_type not in ['mean', 'state']:
            raise ValueError(f'Unexpected output type: {output_type}')
        self.output_type = output_type

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tpredictor:           %s', predictor)
        logging.debug('\tcorrector:           %s', corrector)
        logging.debug('\tnum_steps:           %s', self.num_steps)
        logging.debug('\ttime_min:            %s', self.time_min)
        logging.debug('\ttime_max:            %s', self.time_max)
        logging.debug('\tnum_corrector_steps: %s', num_corrector_steps)
        logging.debug('\tsnr:                 %s', snr)
        logging.debug('\toutput_type:         %s', self.output_type)

    @typecheck(
        input_types={
            "prior_mean": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "score_condition": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType(), optional=True),
            "state_length": NeuralType(tuple('B'), LengthsType(), optional=True),
        },
        output_types={
            "sample": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "state_length": NeuralType(tuple('B'), LengthsType(), optional=True),
        },
    )
    @torch.inference_mode()
    def forward(
        self, prior_mean: torch.Tensor, score_condition: torch.Tensor, state_length: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Takes prior (noisy) mean and generates a sample by solving the reverse SDE.

        Args:
            prior_mean: mean for the prior distribution, e.g., noisy observation
            score_condition: conditioning for the score estimator
            state_length: length of the valid time steps for each example in the batch

        Returns:
            Generated `sample` and the corresponding `sample_length`.
        """
        # Sample from the prior distribution
        state = self.sde.prior_sampling(prior_mean=prior_mean)

        if state_length is not None:
            state = mask_sequence_tensor(state, state_length)

        # Time steps for evaluation
        time_steps = torch.linspace(self.time_max, self.time_min, self.num_steps, device=state.device)

        # Sampling
        for t in time_steps:
            # time steps for the whole batch
            time = t * torch.ones(state.shape[0], device=state.device)

            # corrector step
            state, _ = self.corrector(
                state=state, time=time, score_condition=score_condition, state_length=state_length
            )

            # predictor step
            state, state_mean = self.predictor(
                state=state,
                time=time,
                score_condition=score_condition,
                prior_mean=prior_mean,
                state_length=state_length,
            )

        # Final output
        if self.output_type == 'state':
            sample = state
        elif self.output_type == 'mean':
            sample = state_mean
        else:
            raise RuntimeError(f'Unexpected output type: {self.output_type}')

        if state_length is not None:
            sample = mask_sequence_tensor(sample, state_length)

        return sample, state_length


class Predictor(torch.nn.Module, ABC):
    """Predictor for the reverse process.

    Args:
        sde: forward SDE
        score_estimator: neural score estimator
    """

    def __init__(self, sde, score_estimator):
        super().__init__()
        self.reverse_sde = ReverseStochasticDifferentialEquation(sde=sde, score_estimator=score_estimator)

    @abstractmethod
    @torch.inference_mode()
    def forward(
        self,
        *,
        state: torch.Tensor,
        time: torch.Tensor,
        score_condition: Optional[torch.Tensor] = None,
        state_length: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Predict the next state of the reverse process.

        Args:
            state: current state of the process, shape (B, C, D, T)
            time: current time of the process, shape (B,)
            score_condition: conditioning for the score estimator
            state_length: length of the valid time steps for each example in the batch

        Returns:
            New state and mean.
        """
        pass


class ReverseDiffusionPredictor(Predictor):
    """Predict the next state of the reverse process using the reverse diffusion process.

    Args:
        sde: forward SDE
        score_estimator: neural score estimator
    """

    def __init__(self, sde, score_estimator):
        super().__init__(sde=sde, score_estimator=score_estimator)

    @torch.inference_mode()
    def forward(self, *, state, time, score_condition=None, state_length=None, **kwargs):
        """Predict the next state of the reverse process using the reverse diffusion process.

        Args:
            state: current state of the process, shape (B, C, D, T)
            time: current time of the process, shape (B,)
            score_condition: conditioning for the score estimator
            state_length: length of the valid time steps for each example in the batch

        Returns:
            New state and mean of the diffusion process.
        """
        drift, diffusion = self.reverse_sde.discretize(
            state=state, time=time, score_condition=score_condition, state_length=state_length, **kwargs
        )

        # Generate a random sample from a standard normal distribution
        z_norm = torch.randn_like(state)

        # Compute the mean of the next state
        mean = state - drift

        # Compute new state by sampling
        new_state = mean + diffusion * z_norm

        if state_length is not None:
            new_state = mask_sequence_tensor(new_state, state_length)
            mean = mask_sequence_tensor(mean, state_length)

        return new_state, mean


class Corrector(NeuralModule, ABC):
    """Corrector for the reverse process.

    Args:
        sde: forward SDE
        score_estimator: neural score estimator
        snr: SNR for Annealed Langevin Dynamics
        num_steps: number of steps for the corrector
    """

    def __init__(
        self,
        sde: Type[StochasticDifferentialEquation],
        score_estimator: Type[NeuralModule],
        snr: float,
        num_steps: int,
    ):
        super().__init__()
        self.sde = sde
        self.score_estimator = score_estimator
        self.snr = snr
        self.num_steps = num_steps

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tsnr:             %s', snr)
        logging.debug('\tnum_steps:       %s', num_steps)

    @abstractmethod
    @typecheck(
        input_types={
            "state": NeuralType(('B', 'C', 'D', 'T'), VoidType()),
            "time": NeuralType(tuple('B'), FloatType()),
            "score_condition": NeuralType(('B', 'C', 'D', 'T'), VoidType(), optional=True),
            "state_length": NeuralType(tuple('B'), LengthsType(), optional=True),
        },
        output_types={
            "state": NeuralType(('B', 'C', 'D', 'T'), VoidType()),
        },
    )
    @torch.inference_mode()
    def forward(self, state, time, score_condition=None, state_length=None):
        """
        Args:
            state: current state of the process, shape (B, C, D, T)
            time: current time of the process, shape (B,)
            score_condition: conditioning for the score estimator
            state_length: length of the valid time steps for each example in the batch

        Returns:
            New state and mean.
        """
        pass


class AnnealedLangevinDynamics(Corrector):
    """Annealed Langevin Dynamics for the reverse process.

    References:
        - Song et al., Score-based generative modeling through stochastic differential equations, 2021
    """

    def __init__(self, sde, **kwargs):
        if not isinstance(sde, OrnsteinUhlenbeckVarianceExplodingSDE):
            raise ValueError(f'Expected an instance of OrnsteinUhlenbeckVarianceExplodingSDE, got {type(sde)}')
        super().__init__(sde=sde, **kwargs)

    @torch.inference_mode()
    def forward(self, state, time, score_condition=None, state_length=None):
        """Correct the state using Annealed Langevin Dynamics.

        Args:
            state: current state of the process, shape (B, C, D, T)
            time: current time of the process, shape (B,)
            score_condition: conditioning for the score estimator
            state_length: length of the valid time steps for each example in the batch

        Returns:
            New state and mean of the diffusion process.

        References:
            Alg. 4 in http://arxiv.org/abs/2011.13456
        """
        # Compute the standard deviation of the diffusion process
        std = self.sde.perturb_kernel_std(time=time)
        # View as [B, 1, 1, 1]
        std = std.view(-1, *([1] * (state.dim() - 1)))

        for i in range(self.num_steps):
            # prepare input for the score estimator, concatenate conditioning along the channel dimension
            score_input = state if score_condition is None else torch.cat([state, score_condition], dim=1)

            # calculate the score
            score, _ = self.score_estimator(input=score_input, input_length=state_length, condition=time)

            # generate a sample from a standard normal distribution
            z_norm = torch.randn_like(state)

            # compute the step size
            # note: this is slightly different than in the paper, where std = ||z_norm||_2 / ||score||_2
            step_size = 2 * (self.snr * std).pow(2)

            # update the mean
            mean = state + step_size * score

            # update the state
            state = mean + z_norm * torch.sqrt(step_size * 2)

        if state_length is not None:
            state = mask_sequence_tensor(state, state_length)
            mean = mask_sequence_tensor(mean, state_length)

        return state, mean
