# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
# SPDX-License-Identifier: LicenseRef-Apache2
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

import os
from contextlib import contextmanager
from datetime import timedelta
from typing import Generator

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from nemo.collections.llm.gpt.model.hyena import HyenaTestConfig
from nemo.collections.llm.gpt.model.megatron.hyena.engine import (
    adjust_filter_shape_for_broadcast,
    fftconv_func,
    parallel_fir,
    parallel_iir,
    prefill_via_modal_fft,
    step_fir,
    step_iir,
)
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_config import HyenaConfig
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_utils import (
    ParallelCausalDepthwiseConv1d,
    ParallelHyenaOperator,
    ParallelShortHyenaOperator,
)


@contextmanager
def simple_parallel_state():
    """
    Context manager to set up and tear down a simple model parallel state for testing.

    This function initializes the distributed process group and model parallel state,
    sets up environment variables, and seeds the model parallel RNG. It ensures
    proper cleanup after use.
    """
    try:
        # Clean up any existing state
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()

        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")

        # Initialize process group
        if not dist.is_initialized():
            timeout_timedelta = timedelta(seconds=1800)
            if torch.cuda.is_available():
                dist.init_process_group(backend="nccl", timeout=timeout_timedelta)
            else:
                dist.init_process_group(backend="gloo", timeout=timeout_timedelta)

        # Initialize parallel state
        parallel_state.initialize_model_parallel()

        # Initialize the model parallel RNG
        model_parallel_cuda_manual_seed(42)

        yield

    finally:
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.fixture
def hyena_config() -> HyenaConfig:
    """
    Pytest fixture to provide a default HyenaConfig instance.
    """
    return HyenaConfig()


@pytest.fixture
def test_config() -> HyenaTestConfig:
    """Create a test config based on the parametrized dtype and config type"""
    config = HyenaTestConfig(num_layers=2, hidden_size=864, num_attention_heads=1)
    return config


class TestParallelHyenaOperator:
    """
    Test suite for the ParallelHyenaOperator class.
    """

    @pytest.fixture
    def operator(
        self,
        test_config: HyenaTestConfig,
        hyena_config: HyenaConfig,
    ) -> Generator[ParallelHyenaOperator, None, None]:
        """
        Pytest fixture to create a ParallelHyenaOperator instance within a simple parallel state.
        """
        with simple_parallel_state():
            yield ParallelHyenaOperator(
                hidden_size=test_config.hidden_size,
                transformer_config=test_config,
                hyena_config=hyena_config,
                max_sequence_length=1024,
                operator_type="hyena_medium_conv",
                init_method="small_init",
            )

    @pytest.mark.run_only_on('GPU')
    def test_initialization(self, operator: ParallelHyenaOperator):
        """
        Test that the ParallelHyenaOperator is initialized with correct attributes and parameter count.
        """
        assert operator.hidden_size == 864
        assert operator.operator_type == "hyena_medium_conv"
        assert isinstance(operator.conv_bias, torch.nn.Parameter)
        num_weights = sum(p.numel() for p in operator.parameters())
        assert num_weights == 111456

    @pytest.mark.run_only_on('GPU')
    def test_gpu_forward(self, operator: ParallelHyenaOperator, test_config: HyenaTestConfig):
        """
        Test the forward pass of ParallelHyenaOperator on GPU.
        """
        device = torch.device("cuda")
        operator = operator.to(device)
        batch_size = 2
        seq_len = operator.L  # operator.L maps to max_sequence_length
        g = operator.num_groups
        dg = operator.group_dim
        dtype = test_config.params_dtype

        x1 = torch.ones((batch_size, g * dg, seq_len), device=device, dtype=dtype)
        x2 = torch.ones((batch_size, g * dg, seq_len), device=device, dtype=dtype)
        v = torch.ones((batch_size, g * dg, seq_len), device=device, dtype=dtype)

        output = operator(x1, x2, v)
        assert output.shape[0] == batch_size
        assert output.shape[1] == operator.hidden_size
        assert output.shape[2] == seq_len


class TestParallelShortHyenaOperator:
    """
    Test suite for the ParallelShortHyenaOperator class.
    """

    @pytest.fixture
    def operator(
        self,
        test_config: HyenaTestConfig,
        hyena_config: HyenaConfig,
    ) -> Generator[ParallelShortHyenaOperator, None, None]:
        """
        Pytest fixture to create a ParallelShortHyenaOperator instance within a simple parallel state.
        """
        with simple_parallel_state():
            yield ParallelShortHyenaOperator(
                hidden_size=test_config.hidden_size,
                transformer_config=test_config,
                hyena_config=hyena_config,
                init_method="small_init",
                short_conv_class=ParallelCausalDepthwiseConv1d,
                use_fast_causal_conv=False,
                local_init=False,
                use_conv_bias=False,
            )

    @pytest.mark.run_only_on('GPU')
    def test_initialization(self, operator: ParallelShortHyenaOperator):
        """
        Test that the ParallelShortHyenaOperator is initialized with correct attributes and parameter count.
        """
        assert operator.hidden_size == 864
        assert operator.pregate
        assert operator.postgate
        num_weights = sum(p.numel() for p in operator.parameters())
        assert num_weights == 6048

    @pytest.mark.run_only_on('GPU')
    def test_gpu_forward(self, operator: ParallelShortHyenaOperator, test_config: HyenaTestConfig):
        """
        Test the forward pass of ParallelShortHyenaOperator on GPU.
        """
        device = torch.device("cuda")
        operator = operator.to(device)
        batch_size = 2
        seq_len = 1024
        g = operator.num_groups
        dg = operator.group_dim
        dtype = test_config.params_dtype

        x1 = torch.ones((batch_size, g * dg, seq_len), device=device, dtype=dtype)
        x2 = torch.ones((batch_size, g * dg, seq_len), device=device, dtype=dtype)
        v = torch.ones((batch_size, g * dg, seq_len), device=device, dtype=dtype)

        output = operator(x1, x2, v)
        assert output.shape[0] == batch_size
        assert output.shape[1] == operator.hidden_size
        assert output.shape[2] == seq_len

    @pytest.mark.run_only_on('GPU')
    def test_fast_causal_conv_short_conv_len_validation(self, test_config: HyenaTestConfig, hyena_config: HyenaConfig):
        """
        Test that ParallelShortHyenaOperator raises an assertion error when use_fast_causal_conv=True
        and hyena_short_conv_len > 4, which is not supported.
        """
        # Create a config with hyena_short_conv_len > 4
        hyena_config.hyena_short_conv_len = 5

        # Ensure transformer_config.use_subquadratic_ops is False
        test_config.use_subquadratic_ops = False

        with simple_parallel_state():
            with pytest.raises(AssertionError, match="fast_conv_mixer requires hyena_short_conv_len <= 4"):
                ParallelShortHyenaOperator(
                    hidden_size=test_config.hidden_size,
                    transformer_config=test_config,
                    hyena_config=hyena_config,
                    init_method="small_init",
                    short_conv_class=ParallelCausalDepthwiseConv1d,
                    use_fast_causal_conv=True,  # This should trigger the assertion
                    local_init=False,
                    use_conv_bias=False,
                )


class TestParallelShortHyenaOperatorWithConvBias:
    """
    Test suite for the ParallelShortHyenaOperator class with convolution bias enabled.
    """

    @pytest.fixture
    def operator(
        self,
        test_config: HyenaTestConfig,
        hyena_config: HyenaConfig,
    ) -> Generator[ParallelShortHyenaOperator, None, None]:
        """
        Pytest fixture to create a ParallelShortHyenaOperator instance with conv bias within a simple parallel state.
        """
        with simple_parallel_state():
            yield ParallelShortHyenaOperator(
                hidden_size=test_config.hidden_size,
                transformer_config=test_config,
                hyena_config=hyena_config,
                init_method="small_init",
                short_conv_class=ParallelCausalDepthwiseConv1d,
                use_fast_causal_conv=False,
                local_init=False,
                use_conv_bias=True,
            )

    @pytest.mark.run_only_on('GPU')
    def test_initialization(self, operator: ParallelShortHyenaOperator):
        """
        Test that the ParallelShortHyenaOperator (with conv bias) is initialized with correct attributes and parameter count.
        """
        assert operator.hidden_size == 864
        assert operator.pregate
        assert operator.postgate
        num_weights = sum(p.numel() for p in operator.parameters())
        assert num_weights == 6912

    @pytest.mark.run_only_on('GPU')
    def test_gpu_forward(self, operator: ParallelShortHyenaOperator, test_config: HyenaTestConfig):
        """
        Test the forward pass of ParallelShortHyenaOperator (with conv bias) on GPU.
        """
        device = torch.device("cuda")
        operator = operator.to(device)
        batch_size = 2
        seq_len = 1024
        g = operator.num_groups
        dg = operator.group_dim
        dtype = test_config.params_dtype

        x1 = torch.ones((batch_size, g * dg, seq_len), device=device, dtype=dtype)
        x2 = torch.ones((batch_size, g * dg, seq_len), device=device, dtype=dtype)
        v = torch.ones((batch_size, g * dg, seq_len), device=device, dtype=dtype)

        output = operator(x1, x2, v)
        assert output.shape[0] == batch_size
        assert output.shape[1] == operator.hidden_size
        assert output.shape[2] == seq_len


class TestParallelCausalDepthwiseConv1d:
    """
    Test suite for the ParallelCausalDepthwiseConv1d class.
    """

    @pytest.fixture
    def operator(
        self,
        test_config: HyenaTestConfig,
        hyena_config: HyenaConfig,
    ) -> Generator[ParallelCausalDepthwiseConv1d, None, None]:
        """
        Pytest fixture to create a ParallelCausalDepthwiseConv1d instance within a simple parallel state.
        """
        with simple_parallel_state():
            yield ParallelCausalDepthwiseConv1d(
                d_model=test_config.hidden_size,
                transformer_config=test_config,
                hyena_config=hyena_config,
                kernel_size=hyena_config.short_conv_L,
                init_method=test_config.init_method,
                bias=hyena_config.conv_proj_bias,
                use_fast_causal_conv=hyena_config.fast_conv_proj,
            )

    @pytest.mark.run_only_on('GPU')
    def test_initialization(self, operator: ParallelCausalDepthwiseConv1d):
        """
        Test that the ParallelCausalDepthwiseConv1d is initialized with correct attributes and parameter count.
        """
        assert operator.d_model == 864
        assert operator.kernel_size == 3
        assert operator.use_conv_bias
        num_weights = sum(p.numel() for p in operator.parameters())
        assert num_weights == 2592

    @pytest.mark.run_only_on('GPU')
    def test_gpu_forward(self, operator: ParallelCausalDepthwiseConv1d, test_config: HyenaTestConfig):
        """
        Test the forward pass of ParallelCausalDepthwiseConv1d on GPU.
        """
        device = torch.device("cuda")
        operator = operator.to(device)
        batch_size = 2
        d_model = operator.d_model
        seq_len = 1024
        dtype = test_config.params_dtype

        x1 = torch.ones((batch_size, d_model, seq_len), device=device, dtype=dtype)
        output = operator(x1, False)

        assert output.shape[0] == batch_size
        assert output.shape[1] == d_model
        assert output.shape[2] == seq_len


@pytest.fixture
def setup_tensors():
    """Setup common test tensors."""
    torch.manual_seed(42)
    return {
        "u": torch.randn(2, 4, 8),
        "k": torch.randn(4, 8),
        "D": torch.randn(4),
        "bias": torch.randn(4),
        "weight": torch.randn(4, 1, 3),
        "weight_long": torch.randn(4, 1, 128),
    }


@pytest.mark.run_only_on('GPU')
def test_adjust_filter_shape_for_broadcast():
    """Test filter shape adjustment for broadcasting."""
    # Case: u: [B, D, L], h: [D, L]
    u = torch.randn(2, 4, 8)
    h = torch.randn(4, 8)
    adjusted_h = adjust_filter_shape_for_broadcast(u, h)
    assert adjusted_h.shape == (1, 4, 8)

    # Case: u: [B, D1, D2, L], h: [D, L] -> should become [1, 1, D, L]
    u = torch.randn(2, 3, 4, 8)
    h = torch.randn(4, 8)
    adjusted_h = adjust_filter_shape_for_broadcast(u, h)
    assert adjusted_h.shape == (1, 1, 4, 8)


@pytest.mark.run_only_on('GPU')
def test_fftconv_func(setup_tensors):
    """Test FFT convolution."""
    u = setup_tensors["u"]
    k = setup_tensors["k"]
    D = setup_tensors["D"]

    result = fftconv_func(u=u, k=k, D=D)

    assert result.shape == u.shape
    assert not torch.isnan(result).any()


@pytest.mark.run_only_on('GPU')
def test_parallel_fir_short_filter(setup_tensors):
    """Test parallel FIR with short filter (conv1d path)."""
    u = torch.randn(2, 8, 4)  # B L D
    weight = setup_tensors["weight"]
    bias = setup_tensors["bias"]

    z, fir_state = parallel_fir(
        u=u,
        weight=weight,
        bias=bias,
        L=8,
        gated_bias=False,
        fir_length=3,
        compute_state=True,
    )

    assert z.shape == (2, 4, 8)
    # fir_state should be last fir_length-1 elements, so L-(fir_length-1) = 8-2 = 6, but we want last 2
    assert fir_state is not None
    assert fir_state.shape == (2, 4, 2)  # fir_length - 1


@pytest.mark.run_only_on('GPU')
def test_parallel_fir_long_filter(setup_tensors):
    """Test parallel FIR with long filter (FFT path)."""
    u = torch.randn(2, 8, 4)
    weight = setup_tensors["weight_long"]
    bias = setup_tensors["bias"]

    z, fir_state = parallel_fir(
        u=u,
        weight=weight,
        bias=bias,
        L=8,
        gated_bias=False,
        fir_length=128,
        compute_state=True,
    )

    assert z.shape == (2, 4, 8)
    # For long filter, fir_state is last fir_length-1 elements, but L=8 < fir_length-1=127
    # So it should be the last L elements = 8
    assert fir_state is not None
    assert fir_state.shape == (2, 4, 8)


@pytest.mark.run_only_on('GPU')
def test_parallel_fir_gated_bias(setup_tensors):
    """Test parallel FIR with gated bias."""
    u = torch.randn(2, 8, 4)
    weight = setup_tensors["weight"]
    bias = setup_tensors["bias"]

    z_gated, _ = parallel_fir(
        u=u,
        weight=weight,
        bias=bias,
        L=8,
        gated_bias=True,
        fir_length=3,
        compute_state=False,
    )

    z_ungated, _ = parallel_fir(
        u=u,
        weight=weight,
        bias=bias,
        L=8,
        gated_bias=False,
        fir_length=3,
        compute_state=False,
    )

    assert not torch.allclose(z_gated, z_ungated)


@pytest.mark.run_only_on('GPU')
def test_parallel_iir():
    """Test parallel IIR."""
    hidden_size = 4
    L = 8
    z_pre = torch.randn(2, 12, L)  # B, 3*hidden_size, L
    h = torch.randn(hidden_size, L)  # D, L
    D = torch.randn(hidden_size)
    poles = torch.randn(hidden_size, 2, 1)  # D, state_dim, 1
    t = torch.arange(L).float()

    y, iir_state = parallel_iir(
        z_pre=z_pre,
        h=h,
        D=D,
        L=L,
        poles=poles,
        t=t,
        hidden_size=hidden_size,
        compute_state=True,
    )

    assert y.shape == (2, L, hidden_size)  # B L D
    assert iir_state is not None
    assert iir_state.shape == (2, hidden_size, 2)  # B D state_dim


@pytest.mark.run_only_on('GPU')
def test_step_fir():
    """Test step FIR."""
    u = torch.randn(2, 4)
    fir_state = torch.randn(2, 4, 3)  # B D cache_size
    weight = torch.randn(4, 1, 4)  # D 1 filter_len
    bias = torch.randn(4)

    y, new_state = step_fir(
        u=u,
        fir_state=fir_state,
        weight=weight,
        bias=bias,
        gated_bias=False,
        flip_filter=False,
    )

    assert y.shape == (2, 4)
    # State gets updated: if cache_size < filter_length-1, append; otherwise roll
    # cache_size=3, filter_length=4, so 3 < 4-1=3 is False, so we roll
    assert new_state.shape == (2, 4, 3)


@pytest.mark.run_only_on('GPU')
def test_step_fir_flip_filter():
    """Test step FIR with flipped filter."""
    u = torch.randn(2, 4)
    fir_state = torch.randn(2, 4, 3)
    weight = torch.randn(4, 1, 4)

    y_normal, _ = step_fir(u=u, fir_state=fir_state, weight=weight, flip_filter=False)
    y_flipped, _ = step_fir(u=u, fir_state=fir_state, weight=weight, flip_filter=True)

    assert not torch.allclose(y_normal, y_flipped)


@pytest.mark.run_only_on('GPU')
def test_step_iir():
    """Test step IIR."""
    x2 = torch.randn(2, 4)  # B D
    x1 = torch.randn(2, 4)  # B D
    v = torch.randn(2, 4)  # B D
    D = torch.randn(4)  # D
    residues = torch.randn(4, 4)  # D state_dim (needs to match iir_state last dim)
    poles = torch.randn(4, 1)  # D 1
    iir_state = torch.randn(2, 4, 4)  # B D state_dim

    y, new_state = step_iir(
        x2=x2,
        x1=x1,
        v=v,
        D=D,
        residues=residues,
        poles=poles,
        iir_state=iir_state,
    )

    assert y.shape == (2, 4)
    assert new_state.shape == (2, 4, 4)


@pytest.mark.run_only_on('GPU')
def test_prefill_via_modal_fft():
    """Test prefill via modal FFT."""
    x1v = torch.randn(2, 4, 8)
    L = 8
    poles = torch.randn(4, 2, 1)  # D state_dim 1
    t = torch.arange(L).float()
    X_s = torch.fft.fft(x1v.to(torch.float32), n=2 * L)

    state = prefill_via_modal_fft(x1v=x1v, L=L, poles=poles, t=t, X_s=X_s)

    assert state.shape == (2, 4, 2)
