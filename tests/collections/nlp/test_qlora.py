import pytest
import torch
from torch import nn

from nemo.collections.nlp.modules.common.megatron.adapters.qlora import NF4LayerNormLinearWrapper, NF4LinearWrapper

ao = pytest.importorskip("torchao.dtypes.nf4tensor", reason="torchao is not installed, skipping qlora tests")


@pytest.fixture
def input_tensor():
    return torch.randn([8, 4096], dtype=torch.bfloat16, device='cuda') / 10


@pytest.fixture
def original_weight():
    return torch.randn([1024, 4096], dtype=torch.bfloat16) / 10


@pytest.fixture
def norm_weight():
    return torch.randn([4096], dtype=torch.bfloat16, device='cuda') / 100


@pytest.fixture
def norm_bias():
    return torch.randn([4096], dtype=torch.bfloat16, device='cuda') / 100


@pytest.fixture
def ao_nf4_weight(original_weight):
    return ao.NF4Tensor.from_tensor(original_weight.cuda(), 64, 256)


@torch.no_grad()
def test_nf4_linear(input_tensor, original_weight, ao_nf4_weight):

    nemo_nf4_linear = NF4LinearWrapper(original_weight)
    assert nemo_nf4_linear.weight.is_nf4_quantized
    nemo_output, _ = nemo_nf4_linear(input_tensor)

    ao_output = ao.linear_nf4(input_tensor, ao_nf4_weight)

    assert torch.allclose(nemo_output, ao_output, atol=1e-2)


# @torch.no_grad()
def test_nf4_layernorm_linear(input_tensor, original_weight, norm_weight, norm_bias, ao_nf4_weight):
    ln = nn.LayerNorm(input_tensor.size(-1))
    ln.weight = nn.Parameter(norm_weight)
    ln.bias = nn.Parameter(norm_bias)

    nemo_nf4_layernorm_linear = NF4LayerNormLinearWrapper(original_weight, norm_weight, norm_bias, "LayerNorm", False)
    assert nemo_nf4_layernorm_linear.weight.is_nf4_quantized
    (nemo_output, nemo_norm_output), _ = nemo_nf4_layernorm_linear(input_tensor)

    ao_norm_output = ln(input_tensor)
    ao_output = ao.linear_nf4(ln(input_tensor), ao_nf4_weight)
    assert torch.allclose(nemo_norm_output, ao_norm_output, atol=1e-2)
    assert torch.allclose(nemo_output, ao_output, atol=1e-2)


@torch.no_grad()
def test_nf4_rmsnorm_linear(input_tensor, original_weight, norm_weight, norm_bias, ao_nf4_weight):
    from nemo.utils.export_utils import TorchRMSNorm

    rms_norm = TorchRMSNorm(norm_weight)

    nemo_nf4_layernorm_linear = NF4LayerNormLinearWrapper(original_weight, norm_weight, None, "RMSNorm", False)
    assert nemo_nf4_layernorm_linear.weight.is_nf4_quantized
    (nemo_output, nemo_norm_output), _ = nemo_nf4_layernorm_linear(input_tensor)

    ao_norm_output = rms_norm(input_tensor)
    ao_output = ao.linear_nf4(ao_norm_output, ao_nf4_weight)

    assert torch.allclose(nemo_norm_output, ao_norm_output, atol=1e-2)
    assert torch.allclose(nemo_output, ao_output, atol=1e-2)
