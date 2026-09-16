# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from nemo.collections.audio.parts.submodules.schroedinger_bridge import SBNoiseScheduleVE, SBNoiseScheduleVP, SBSampler

NUM_STEPS = [1, 5, 10, 20, 100]


@pytest.mark.parametrize("noise_schedule_type", ["ve", "vp"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")),
    ],
)
def test_sb_noise_schedule_max_properties(noise_schedule_type, device):
    if noise_schedule_type == "ve":
        noise_schedule = SBNoiseScheduleVE(k=2.0, c=0.5)
    else:
        noise_schedule = SBNoiseScheduleVP(beta_0=0.1, beta_1=1.0, c=0.5)

    noise_schedule = noise_schedule.to(device)
    time_max = torch.tensor([noise_schedule.time_max], device=device)
    torch.testing.assert_close(noise_schedule.alpha_t_max, noise_schedule.alpha(time_max))
    torch.testing.assert_close(noise_schedule.sigma_t_max, noise_schedule.sigma(time_max))


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
