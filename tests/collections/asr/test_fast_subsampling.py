# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
import torch

from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling
from nemo.core.utils.optional_libs import TRITON_AVAILABLE

CUDA_TRITON_AVAILABLE = TRITON_AVAILABLE and torch.cuda.is_available()

FEAT_IN = 80
FEAT_OUT = 512
CONV_CHANNELS = 256

# The kernels run conv0's GEMM in TF32, 10 mantissa bits against fp32's 23, while the reference
# stays fp32. Parity is therefore bounded by TF32 rounding, not by fp32 noise. Worst measured is
# 2.1e-4, at factor 4, which has the fewest layers after the fused head to average the error out.
FP32_PARITY_ATOL = 5e-4


@pytest.fixture(scope="session", autouse=True)
def one_config_per_kernel():
    """Autotuning 180 configs per process dominates the runtime and none of it tests behaviour.

    Set NEMO_TRITON_AUTOTUNE=1 to sweep them all, which is worth doing when a kernel changes.
    """
    if not CUDA_TRITON_AVAILABLE or os.environ.get("NEMO_TRITON_AUTOTUNE"):
        yield
        return
    from triton.runtime.autotuner import Autotuner

    from nemo.collections.asr.parts.triton import depthwise_conv, subsampling

    pinned = [
        obj for module in (subsampling, depthwise_conv) for obj in vars(module).values() if isinstance(obj, Autotuner)
    ]
    full = [kernel.configs for kernel in pinned]
    for kernel in pinned:
        kernel.configs = kernel.configs[:1]
    yield
    for kernel, configs in zip(pinned, full):
        kernel.configs = configs


@pytest.fixture
def fp32_reference():
    """Keep the PyTorch reference in fp32 so it stays ground truth for the fused path."""
    cudnn, matmul = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cudnn.allow_tf32 = cudnn
    torch.backends.cuda.matmul.allow_tf32 = matmul


def _build(factor=8, causal=False, use_triton=None, device="cuda", dtype=torch.float32, activation=None):
    torch.manual_seed(0)
    module = ConvSubsampling(
        subsampling='dw_striding',
        subsampling_factor=factor,
        feat_in=FEAT_IN,
        feat_out=FEAT_OUT,
        conv_channels=CONV_CHANNELS,
        activation=torch.nn.ReLU() if activation is None else activation,
        is_causal=causal,
        use_triton=use_triton,
    )
    return module.to(device=device, dtype=dtype)


def _inputs(batch=4, time=1032, ragged=False, device="cuda", dtype=torch.float32, seed=1):
    torch.manual_seed(seed)
    x = torch.randn(batch, time, FEAT_IN, device=device, dtype=dtype)
    if ragged:
        lengths = torch.randint(time // 2, time + 1, (batch,), device=device, dtype=torch.int64)
        lengths[0] = time  # keep the padded extent pinned to `time`
    else:
        lengths = torch.full((batch,), time, device=device, dtype=torch.int64)
    return x, lengths


def _run(module, x, lengths, use_triton, grad=None):
    """Forward (and optionally backward) with the fast path forced on or off."""
    module.conv.fuse_triton = use_triton
    module.zero_grad(set_to_none=True)
    out, out_lengths = module(x, lengths)
    grads = None
    if grad is not None:
        out.backward(grad)
        grads = [p.grad.detach().clone() for p in module.parameters()]
    return out.detach(), out_lengths, grads


def _max_rel(actual, expected):
    return max(((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item() for a, b in zip(actual, expected))


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.parametrize("factor", [4, 8, 16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("ragged", [False, True])
def test_triton_subsampling_matches_pytorch(fp32_reference, factor, causal, ragged):
    """The fused path reproduces the PyTorch stack for every supported subsampling factor.

    Both paths share one module, so the weights are the same objects and only arithmetic differs.
    """
    module = _build(factor=factor, causal=causal)
    assert module.conv.fuse_triton, "dw_striding with factor >= 4 must be eligible"
    x, lengths = _inputs(ragged=ragged)

    reference, reference_lengths, _ = _run(module, x, lengths, use_triton=False)
    fused, fused_lengths, _ = _run(module, x, lengths, use_triton=True)

    assert torch.equal(fused_lengths, reference_lengths)
    torch.testing.assert_close(fused, reference, rtol=0, atol=FP32_PARITY_ATOL)


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.parametrize(
    "factor, use_triton, eligible",
    [
        (8, None, True),  # eligible stack, no explicit choice
        (8, False, False),  # the config switch overrides an eligible stack
        (2, None, False),  # stops after [conv, act], so there is no depthwise to absorb
    ],
)
def test_fast_path_eligibility(factor, use_triton, eligible):
    assert bool(_build(factor=factor, use_triton=use_triton).conv.fuse_triton) is eligible


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
def test_fast_path_requires_relu():
    """The kernels apply ReLU, so any other activation keeps the whole stack on PyTorch."""
    assert not _build(activation=torch.nn.GELU()).conv.fuse_triton


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_triton_subsampling_half_precision(fp32_reference, dtype):
    """Half precision agrees to within the dtype's own resolution."""
    module = _build(dtype=dtype)
    x, lengths = _inputs(ragged=True, dtype=dtype)

    reference, reference_lengths, _ = _run(module, x, lengths, use_triton=False)
    fused, fused_lengths, _ = _run(module, x, lengths, use_triton=True)

    assert torch.equal(fused_lengths, reference_lengths)
    torch.testing.assert_close(fused.float(), reference.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_triton_subsampling_under_autocast(fp32_reference, dtype):
    """Parameters stay fp32 and the autocast region alone picks the compute dtype."""
    module = _build()
    x, lengths = _inputs(batch=2, time=520, ragged=True)
    assert next(module.parameters()).dtype == torch.float32

    with torch.autocast("cuda", dtype=dtype):
        reference, reference_lengths, _ = _run(module, x, lengths, use_triton=False)
        fused, fused_lengths, _ = _run(module, x, lengths, use_triton=True)

    assert fused.dtype == reference.dtype == dtype
    assert torch.equal(fused_lengths, reference_lengths)
    torch.testing.assert_close(fused.float(), reference.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
def test_triton_subsampling_with_utterances_shorter_than_a_tile(fp32_reference):
    """Utterances that end inside the first output tile.

    Skipped tiles must still store zeros: the following pointwise contracts over every time
    position for its weight gradient.
    """
    module = _build()
    x, _ = _inputs(batch=3, time=520)
    lengths = torch.tensor([520, 16, 1], device=x.device, dtype=torch.int64)

    with torch.no_grad():
        shape = module(x, lengths)[0].shape
    torch.manual_seed(7)
    grad = torch.randn(shape, device=x.device) * 0.01

    reference, reference_lengths, reference_grads = _run(module, x, lengths, False, grad=grad)
    fused, fused_lengths, fused_grads = _run(module, x, lengths, True, grad=grad)
    _, _, fused_again = _run(module, x, lengths, True, grad=grad)

    assert torch.equal(fused_lengths, reference_lengths)
    torch.testing.assert_close(fused, reference, rtol=0, atol=FP32_PARITY_ATOL)
    control = _max_rel(fused_again, fused_grads)
    cross = _max_rel(fused_grads, reference_grads)
    assert cross <= max(10 * control, 5e-2), f"cross-path {cross:.2e} exceeds nondeterminism {control:.2e}"


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.parametrize("causal", [False, True])
def test_triton_subsampling_backward_matches_pytorch(fp32_reference, causal):
    """Gradients agree to within the fused path's own run-to-run nondeterminism.

    ``tl.atomic_add`` ordering is not reproducible, so the control run below measures that spread
    and the cross-path difference must not be materially larger.
    """
    module = _build(causal=causal)
    x, lengths = _inputs(batch=2, time=520, ragged=True)

    with torch.no_grad():
        shape = module(x, lengths)[0].shape
    torch.manual_seed(7)
    grad = torch.randn(shape, device=x.device) * 0.01

    _, _, reference_grads = _run(module, x, lengths, False, grad=grad)
    _, _, fused_grads = _run(module, x, lengths, True, grad=grad)
    _, _, fused_again = _run(module, x, lengths, True, grad=grad)

    control = _max_rel(fused_again, fused_grads)
    cross = _max_rel(fused_grads, reference_grads)
    assert cross <= max(10 * control, 5e-2), f"cross-path {cross:.2e} exceeds nondeterminism {control:.2e}"


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
def test_input_that_requires_a_gradient_takes_the_pytorch_path():
    """The fused kernel returns no input gradient, so an input that needs one bypasses it.

    SSL pretraining with a trainable mask embedding in the spectrogram is the caller that does.
    """
    module = _build()
    assert module.conv.fuse_triton
    x, lengths = _inputs(batch=2, time=520)
    x.requires_grad_()

    module(x, lengths)[0].sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
def test_triton_subsampling_is_invariant_to_padding_and_batch_size(fp32_reference):
    """Valid frames do not depend on how much padding shares the batch, nor on batch size.

    Checked on the module output, since the fused path has no per-layer convolution modules.
    """
    module = _build()
    module.conv.fuse_triton = True
    x, lengths = _inputs(batch=4, time=520, ragged=True)

    full, full_lengths = module(x, lengths)

    padded = torch.cat([x, torch.randn(x.shape[0], 256, FEAT_IN, device=x.device)], dim=1)
    padded_out, padded_lengths = module(padded, lengths)
    assert torch.equal(padded_lengths, full_lengths)
    for index, valid in enumerate(full_lengths.tolist()):
        torch.testing.assert_close(padded_out[index, :valid], full[index, :valid], rtol=0, atol=1e-5)

    single_out, single_lengths = module(x[:1], lengths[:1])
    torch.testing.assert_close(single_out[0, : single_lengths[0]], full[0, : full_lengths[0]], rtol=0, atol=1e-5)


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
def test_triton_subsampling_on_streaming_chunk_shapes(fp32_reference):
    """Cache-aware streaming prepends ``subsampling_factor + 1`` real frames to every chunk."""
    module = _build(causal=True)
    cache_frames = module.get_streaming_cache_size()[1]
    chunk = module.subsampling_factor * 8
    x, lengths = _inputs(batch=3, time=chunk + cache_frames)

    reference, reference_lengths, _ = _run(module, x, lengths, use_triton=False)
    fused, fused_lengths, _ = _run(module, x, lengths, use_triton=True)

    assert torch.equal(fused_lengths, reference_lengths)
    torch.testing.assert_close(fused, reference, rtol=0, atol=FP32_PARITY_ATOL)


@pytest.mark.unit
def test_falls_back_when_triton_is_unavailable(monkeypatch):
    """Without Triton the module still builds and runs, on the PyTorch path."""
    monkeypatch.setattr("nemo.collections.asr.parts.submodules.subsampling.TRITON_AVAILABLE", False)
    module = _build(device="cpu")
    assert not module.conv.fuse_triton
    x, lengths = _inputs(batch=2, time=264, device="cpu")
    out, out_lengths = module(x, lengths)
    assert out.shape[0] == 2 and out.shape[2] == FEAT_OUT
    assert out_lengths.shape == (2,)


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.parametrize("dynamo", [False, True])
def test_export_takes_the_pytorch_path(tmp_path, dynamo):
    """Both exporters trace on tensors a Triton launch cannot read."""
    onnx = pytest.importorskip("onnx")
    module = _build().eval()
    assert module.conv.fuse_triton
    x, lengths = _inputs(batch=2, time=264)
    path = str(tmp_path / "subsampling.onnx")

    torch.onnx.export(module, (x, lengths), path, dynamo=dynamo)

    # The five convolutions of the PyTorch stack; the fused path launches kernels instead.
    convolutions = [node for node in onnx.load(path).graph.node if node.op_type == "Conv"]
    assert len(convolutions) == 5


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
def test_backward_survives_callers_editing_the_returned_lengths():
    """Callers own the lengths and edit them in place, e.g. speechlm2's duplex model."""
    module = _build()
    x, lengths = _inputs(batch=7)
    out, out_lengths = module(x, lengths)

    torch.clamp_(out_lengths, max=64)
    out_lengths[0] = 5

    out.float().pow(2).mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
def test_deterministic_algorithms_take_the_pytorch_path(monkeypatch):
    """The weight gradients accumulate through atomics, whose summation order varies per run."""
    module = _build()
    assert module.conv.fuse_triton
    x, lengths = _inputs(batch=2, time=520)

    def unreachable(*args, **kwargs):
        raise AssertionError("the fused path ran under deterministic algorithms")

    monkeypatch.setattr("nemo.collections.asr.parts.submodules.subsampling.fused_conv_relu_dw", unreachable)
    torch.use_deterministic_algorithms(True)
    try:
        out, _ = module.conv(x, lengths)
    finally:
        torch.use_deterministic_algorithms(False)

    assert torch.isfinite(out).all()


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
def test_cpu_input_uses_the_pytorch_path():
    """Eligibility is a property of the config; a CPU tensor still takes the PyTorch path."""
    module = _build(device="cpu")
    assert module.conv.fuse_triton
    x, lengths = _inputs(batch=2, time=264, device="cpu")
    out, _ = module(x, lengths)
    assert torch.isfinite(out).all()


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
def test_state_dict_is_identical_with_and_without_triton():
    """Checkpoints must load either way; the fast path adds no persistent state."""
    with_triton = _build(use_triton=True)
    without_triton = _build(use_triton=False)

    assert list(with_triton.state_dict().keys()) == list(without_triton.state_dict().keys())

    without_triton.load_state_dict(with_triton.state_dict())
    with_triton.load_state_dict(without_triton.state_dict())
