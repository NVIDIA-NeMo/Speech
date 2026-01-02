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

from nemo.collections.audio.parts.submodules.flow import ConditionalFlowMatchingEulerSampler, OptimalTransportFlow

NUM_STEPS = [1, 5, 10, 20, 100]
TIMES_MIN = [0.0, 1e-8, 1e-2, 0.1, 0.25, 0.4]
TIMES_MAX = [0.5, 0.7, 0.99, 1.0 - 1e-8, 1.0]


@pytest.mark.parametrize("num_steps", NUM_STEPS)
@pytest.mark.parametrize("estimator_target", ['conditional_vector_field', 'data'])
def test_euler_sampler_nfe(num_steps, estimator_target):
    """
    For this specific solver the number of steps should be equal to the number of function (estimator) evaluations
    """

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

    flow = OptimalTransportFlow()
    sampler = ConditionalFlowMatchingEulerSampler(
        estimator=estimator, num_steps=num_steps, estimator_target=estimator_target, flow=flow
    )

    b, c, d, l = 2, 3, 4, 5
    lengths = [5, 3]
    init_state = torch.randn(b, c, d, l)
    init_state_length = torch.LongTensor(lengths)

    sampler.forward(state=init_state, estimator_condition=None, state_length=init_state_length)

    assert counter_hook.counter == sampler.num_steps


@pytest.mark.parametrize('time_min', TIMES_MIN)
@pytest.mark.parametrize('time_max', TIMES_MAX)
def test_time_generation_bounds_optimal_transport(time_min, time_max):
    """
    This test uses a flow with certain time_min and time_max parameters to generate timepoints and checks if timepoints belong in [time_min, time_max] interval.
    """
    rng = torch.Generator(device='cpu')
    rng.manual_seed(0)

    flow = OptimalTransportFlow(time_min=time_min, time_max=time_max)
    time = flow.generate_time(batch_size=1_000, rng=rng)

    assert torch.all(time >= time_min).item()
    assert torch.all(time <= time_max).item()


@pytest.mark.parametrize('time_min', TIMES_MIN)
@pytest.mark.parametrize('time_max', TIMES_MAX)
def test_time_generation_bounds_optimal_transport_negative_examples(time_min, time_max):
    """
    This test uses a flow with certain time_min and time_max parameters, widens them, generates timepoints and checks if timepoints belong in [time_min, time_max] interval.
    Since we widen the interval when initializing the flow, we expect that after taking enough samples some of them will be outside intended interval.
    """
    rng = torch.Generator(device='cpu')
    rng.manual_seed(0)

    flow = OptimalTransportFlow(time_min=time_min - 0.1, time_max=time_max + 0.1)
    time = flow.generate_time(batch_size=1_000, rng=rng)
    assert not torch.all(time >= time_min).item()
    assert not torch.all(time <= time_max).item()
