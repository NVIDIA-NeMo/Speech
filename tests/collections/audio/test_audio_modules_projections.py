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

import pytest
import torch

from nemo.collections.audio.modules.projections import MixtureConsistencyProjection


class TestMixtureConsistencyProjection:
    @pytest.mark.unit
    @pytest.mark.parametrize('weighting', [None, 'power'])
    @pytest.mark.parametrize('num_sources', [1, 3])
    def test_mixture_consistency(self, weighting: str, num_sources: int):
        batch_size = 4
        num_subbands = 33
        num_samples = 100
        num_examples = 8
        atol = 1e-5

        rng = torch.Generator()
        rng.manual_seed(42)

        # create projection
        uut = MixtureConsistencyProjection(weighting=weighting)

        for n in range(num_examples):
            # single-channel mixture
            mixture = torch.randn(batch_size, 1, num_subbands, num_samples, generator=rng, dtype=torch.cfloat)
            # source estimates
            estimate = torch.randn(
                batch_size, num_sources, num_subbands, num_samples, generator=rng, dtype=torch.cfloat
            )

            # project
            uut_projected = uut(mixture=mixture, estimate=estimate)

            # estimated mixture
            estimated_mixture = torch.sum(estimate, dim=1, keepdim=True)

            if weighting is None:
                weight = 1 / num_sources
            elif weighting == 'power':
                weight = estimate.abs().pow(2)
                weight = weight / (weight.sum(dim=1, keepdim=True) + uut.eps)
            else:
                raise ValueError(f'Weighting {weighting} not implemented')
            correction = weight * (mixture - estimated_mixture)
            ref_projected = estimate + correction

            # check consistency
            assert torch.allclose(uut_projected, ref_projected, atol=atol)

    @pytest.mark.unit
    def test_unsupported_weighting(self):
        # Initialize with unsupported weighting
        with pytest.raises(NotImplementedError):
            MixtureConsistencyProjection(weighting='not-implemented')

        # Initialize with None and change later
        uut = MixtureConsistencyProjection(weighting=None)
        uut.weighting = 'not-implemented'
        with pytest.raises(NotImplementedError):
            uut(
                mixture=torch.randn(1, 1, 1, 1, dtype=torch.cfloat),
                estimate=torch.randn(1, 1, 1, 1, dtype=torch.cfloat),
            )

    @pytest.mark.unit
    def test_unsupported_inputs(self):
        # Multi-channel mixtures are not supported
        uut = MixtureConsistencyProjection(weighting=None)
        with pytest.raises(ValueError):
            uut(
                mixture=torch.randn(1, 2, 1, 1, dtype=torch.cfloat),
                estimate=torch.randn(1, 2, 1, 1, dtype=torch.cfloat),
            )

        # Consistency projection is applied in the time-frequency domain
        # It is expected that the mixture has a single channel, and shape (B, 1, F, N)
        with pytest.raises(TypeError):
            uut(mixture=torch.randn(1, 2, 1), estimate=torch.randn(1, 2, 1))
        # It is expected that the estimate has shape (B, num_sources, F, N)
        with pytest.raises(TypeError):
            uut(mixture=torch.randn(1, 1, 1, 1, dtype=torch.cfloat), estimate=torch.randn(1, 2, 1))
