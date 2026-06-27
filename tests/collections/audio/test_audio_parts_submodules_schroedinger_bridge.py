# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from dataclasses import dataclass

import pytest
import torch

from nemo.collections.audio.parts.submodules.schroedinger_bridge import (
    SBNoiseSchedule,
    SBNoiseScheduleVE,
    SBNoiseScheduleVP,
    SBSampler,
)

NUM_STEPS = [1, 5, 10, 20, 100]


class SlightlyIncreasingSigmaNoiseSchedule(SBNoiseSchedule):
    def __init__(
        self,
        sigma_base: float = 1.0,
        delta: float = 1e-6,
        time_min: float = 1.0 - 1e-6,
        time_max: float = 1.0,
        num_steps: int = 1,
        eps: float = 1e-8,
    ):
        super().__init__(time_min=time_min, time_max=time_max, num_steps=num_steps, eps=eps)
        self.sigma_base = sigma_base
        self.delta = delta

    def f(self, time: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(time)

    def g(self, time: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(time)

    def alpha(self, time: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(time)

    def sigma(self, time: torch.Tensor) -> torch.Tensor:
        sigma_base = torch.full_like(time, self.sigma_base)
        delta = torch.full_like(time, self.delta)
        return torch.where(time < self.time_max, sigma_base + delta, sigma_base)

    def copy(self):
        return SlightlyIncreasingSigmaNoiseSchedule(
            sigma_base=self.sigma_base,
            delta=self.delta,
            time_min=self.time_min,
            time_max=self.time_max,
            num_steps=self.num_steps,
            eps=self.eps,
        )


@pytest.mark.unit
def test_sb_sampler_sde_clamps_negative_tmp_before_sqrt():
    class IdentityEstimator(torch.nn.Module):
        def forward(self, input, input_length, condition):
            return input, input_length

    noise_schedule = SlightlyIncreasingSigmaNoiseSchedule()
    sampler = SBSampler(
        noise_schedule=noise_schedule,
        estimator=IdentityEstimator(),
        estimator_output='data_prediction',
        process='sde',
        num_steps=1,
    )

    init_state = torch.ones(1, 1, 1, 2)

    time_prev = torch.tensor([sampler.time_max], device=init_state.device)
    time = torch.tensor([sampler.time_min], device=init_state.device)
    sigma_prev, _, _ = sampler.noise_schedule.get_sigmas(time_prev)
    sigma_t, _, _ = sampler.noise_schedule.get_sigmas(time)
    raw_tmp = 1 - sigma_t**2 / (sigma_prev**2 + sampler.eps)

    assert raw_tmp.item() < 0

    sample, sample_length = sampler.forward(prior_mean=init_state, estimator_condition=None, state_length=None)

    expected_scale = sigma_t**2 / (sigma_prev**2 + sampler.eps)
    expected = expected_scale.view(-1, 1, 1, 1) * init_state

    assert sample_length is None
    assert torch.isfinite(sample).all()
    torch.testing.assert_close(sample, expected)


@pytest.mark.parametrize("num_steps", NUM_STEPS)
@pytest.mark.parametrize("process", ["sde", "ode"])
@pytest.mark.parametrize("noise_schedule_type", ["ve", "vp"])
def test_sb_sampler_nfe(num_steps, process, noise_schedule_type):
    """
    For this specific solver the number of steps should be equal to the number of function (estimator) evaluations
    """
    if noise_schedule_type == "ve":
        noise_schedule = SBNoiseScheduleVE(k=2.0, c=0.5, num_steps=num_steps)
    elif noise_schedule_type == "vp":
        noise_schedule = SBNoiseScheduleVP(beta_0=0.1, beta_1=1.0, c=0.5, num_steps=num_steps)
    else:
        raise ValueError(f"Invalid noise schedule type: {noise_schedule_type}")

    class IdentityEstimator(torch.nn.Module):
        def forward(self, input, input_length, condition):
            return input, input_length

    @dataclass
    class ForwardCounterHook:
        counter: int = 0

        def __call__(self, *args, **kwargs):
            self.counter += 1

    estimator = IdentityEstimator()
    counter_hook = ForwardCounterHook()
    estimator.register_forward_hook(counter_hook)

    sampler = SBSampler(
        noise_schedule=noise_schedule,
        estimator=estimator,
        estimator_output='data_prediction',
        process=process,
        num_steps=num_steps,
    )

    b, c, d, l = 2, 3, 4, 5
    lengths = [5, 3]
    init_state = torch.randn(b, c, d, l)
    init_state_length = torch.LongTensor(lengths)

    sampler.forward(prior_mean=init_state, estimator_condition=None, state_length=init_state_length)

    assert counter_hook.counter == sampler.num_steps
