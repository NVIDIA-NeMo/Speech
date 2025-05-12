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

import einops
import pytest
import torch

from nemo.collections.audio.parts.submodules.conformer import SpectrogramConformer
from nemo.collections.audio.parts.submodules.ncsnpp import (
    NoiseConditionalScoreNetworkPlusPlus,
    SpectrogramNoiseConditionalScoreNetworkPlusPlus,
)
from nemo.collections.audio.parts.submodules.transformerunet import SpectrogramTransformerUNet, TransformerUNet


@pytest.fixture(params=[True, False], ids=["conditioned_on_time", "not_conditioned_on_time"])
def ncsnpp(request):
    return NoiseConditionalScoreNetworkPlusPlus(
        in_channels=2, out_channels=1, num_resolutions=2, channels=(16, 16, 16), conditioned_on_time=request.param
    )


@pytest.fixture(params=[True, False], ids=["conditioned_on_time", "not_conditioned_on_time"])
def transformerunet(request):
    dim = 16
    return TransformerUNet(
        dim=dim, depth=2, heads=4, ff_mult=2, adaptive_rmsnorm=request.param, adaptive_rmsnorm_cond_dim_in=dim
    )


@pytest.fixture(params=[True, False], ids=["conditioned_on_time", "not_conditioned_on_time"])
def spectrogram_ncsnpp(request):
    return SpectrogramNoiseConditionalScoreNetworkPlusPlus(
        in_channels=2, out_channels=1, num_resolutions=2, channels=(16, 16, 16), conditioned_on_time=request.param
    )


@pytest.fixture(params=[True, False], ids=["conditioned_on_time", "not_conditioned_on_time"])
def spectrogram_transformerunet(request):
    return SpectrogramTransformerUNet(
        in_channels=2, out_channels=1, freq_dim=16, dim=16, depth=2, heads=4, ff_mult=2, adaptive_rmsnorm=request.param
    )


@pytest.fixture()
def spectrogram_conformer():
    return SpectrogramConformer(
        in_channels=2,
        out_channels=1,
        feat_in=16,
        feat_out=16,
        n_layers=2,
        d_model=16,
        conv_kernel_size=3,
        subsampling_factor=1,
    )


@pytest.fixture()
def mock_input_3d():
    batch_size = 3
    input_dim = 16
    time_steps = 20
    with torch.random.fork_rng():
        torch.random.manual_seed(0)
        input_ = torch.randn(batch_size, input_dim, time_steps)
        input_length = torch.randint(low=1, high=time_steps, size=(batch_size,))
        condition = torch.ones(batch_size).float()
    return input_, input_length, condition


@pytest.fixture()
def mock_input_4d():
    batch_size = 3
    channels = 2
    input_dim = 16
    time_steps = 20
    with torch.random.fork_rng():
        torch.random.manual_seed(0)
        input_ = torch.randn(batch_size, channels, input_dim, time_steps)
        input_length = torch.randint(low=1, high=time_steps, size=(batch_size,))
        condition = torch.ones(batch_size).float()
    return input_, input_length, condition


def test_ncsnpp_forward(ncsnpp, mock_input_4d):
    input_, input_length, condition = mock_input_4d
    batch_size, _, input_dim, time_steps = input_.shape

    output, output_length = ncsnpp(
        input=input_, input_length=input_length, condition=condition if ncsnpp.conditioned_on_time else None
    )
    assert output.shape[0] == batch_size
    assert output.shape[1] == ncsnpp.out_channels
    assert output.shape[2] == input_dim
    assert output.shape[3] == time_steps
    assert torch.all(output_length == input_length)


def test_transformerunet_forward(transformerunet, mock_input_3d):
    input_, _, _ = mock_input_3d
    input_ = einops.rearrange(input_, "B D T -> B T D")

    batch_size, *_ = input_.shape

    if transformerunet.adaptive_rmsnorm:
        adaptive_rmsnorm_cond = torch.ones((batch_size, transformerunet.adaptive_rmsnorm_cond_dim_in)).float()
    else:
        adaptive_rmsnorm_cond = None

    output = transformerunet(x=input_, adaptive_rmsnorm_cond=adaptive_rmsnorm_cond)
    assert output.shape == input_.shape


def test_spectrogram_ncsnpp_forward(spectrogram_ncsnpp, mock_input_4d):
    input_, input_length, condition = mock_input_4d
    input_ = torch.view_as_complex(torch.stack([input_, input_], dim=-1)).contiguous()

    batch_size, _, input_dim, time_steps = input_.shape

    output, output_length = spectrogram_ncsnpp(
        input=input_,
        input_length=input_length,
        condition=condition if spectrogram_ncsnpp.ncsnpp.conditioned_on_time else None,
    )
    assert output.shape[0] == batch_size
    assert output.shape[1] == spectrogram_ncsnpp.out_channels
    assert output.shape[2] == input_dim
    assert output.shape[3] == time_steps
    assert torch.all(output_length == input_length)


def test_spectrogram_transformerunet_forward(spectrogram_transformerunet, mock_input_4d):
    input_, input_length, condition = mock_input_4d
    input_ = torch.view_as_complex(torch.stack([input_, input_], dim=-1)).contiguous()

    batch_size, _, input_dim, time_steps = input_.shape

    output, output_length = spectrogram_transformerunet(
        input=input_,
        input_length=input_length,
        condition=condition if hasattr(spectrogram_transformerunet, 'sinu_pos_emb') else None,
    )
    assert output.shape[0] == batch_size
    assert output.shape[1] == spectrogram_transformerunet.out_channels
    assert output.shape[2] == input_dim
    assert output.shape[3] == time_steps
    assert torch.all(output_length == input_length)


def test_spectrogram_conformer_forward(spectrogram_conformer, mock_input_4d):
    input_, input_length, condition = mock_input_4d
    input_ = torch.view_as_complex(torch.stack([input_, input_], dim=-1)).contiguous()

    batch_size, _, input_dim, time_steps = input_.shape

    output, output_length = spectrogram_conformer(input=input_, input_length=input_length)
    assert output.shape[0] == batch_size
    assert output.shape[1] == spectrogram_conformer.out_channels
    assert output.shape[2] == input_dim
    assert output.shape[3] == time_steps
    assert torch.all(output_length == input_length)
