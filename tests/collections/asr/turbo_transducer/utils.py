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

"""
Shared inputs and references for the tests of the fused Joint kernels.

The kernels are compiled for each combination of `tl.constexpr` arguments (hidden dim, vocab size, blank id,
precision flags, pruning range) and tensor dtypes; one compilation takes 0.3-0.6 seconds without the cache.
To keep the test suite fast, the tests share the dimensions and the precision settings defined here,
and deviate from them only when the dimension itself is the subject of the test.
"""

from dataclasses import dataclass, field

import pytest
import torch
import torch.nn.functional as F

from nemo.collections.asr.parts.turbo_transducer.rnnt_consistency import get_rnnt_mask, kl_loss_torch
from nemo.collections.asr.parts.turbo_transducer.rnnt_logprobs import rnnt_logprobs_torch
from nemo.core.utils.optional_libs import TRITON_AVAILABLE

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.turbo_transducer.utils_triton import BF16_ATOMICS_AVAILABLE

# With the tile size 16 used in tests (see `conftest.py`): 2 full hidden tiles and a partial one
HIDDEN_DIM = 40
# 2 full vocab tiles and a partial one; the chunked backward pads the vocab to 64 => one more fully padded tile
VOCAB_SIZE = 35
BLANK_ID = VOCAB_SIZE - 1
# The lattice [3, 31, 8] has 744 rows: 47 row tiles with a partial last one, 3 chunks of the chunked backward
# (256 + 256 + 232 rows) if the chunk size is minimal. Lengths include a single frame and an empty target
SRC_LENGTHS = (31, 1, 20)
TGT_LENGTHS = (7, 0, 3)
# Backward pruning range: 3 * 31 * 3 = 279 rows => 2 chunks of the chunked backward with a partial last one
PRUNING_RANGE_SIZE = 3

# Lattices (src_lengths, tgt_lengths) that differ in the properties used by Triton to specialize integer arguments
# (equal to 1, divisible by 16) for T, U+1 and the number of the lattice rows.
# U+1 is in (4, 8] for all of them: the block size of the pruning window selection is the next power of 2 of U+1
LATTICES_WITH_DIFFERENT_SPECIALIZATION = [
    (SRC_LENGTHS, TGT_LENGTHS),  # T=31, U+1=8, 744 rows
    ((16,), (7,)),  # T=16, 128 rows
    ((32, 5), (7, 3)),  # T=32, 512 rows
    ((1,), (7,)),  # T=1
    ((5, 4), (4, 4)),  # U+1=5
    ((9, 3, 1, 1), (5, 0, 3, 2)),  # U+1=6, 216 rows
]

# Dimensions that differ from the default ones by the alignment to the tiles and by the position of the blank:
# (hidden_dim, vocab_size, blank_id)
DIMS_DEFAULT = (HIDDEN_DIM, VOCAB_SIZE, BLANK_ID)
DIMS_TILE_ALIGNED_BLANK_FIRST = (32, 64, 0)
DIMS_BELOW_TILE_BLANK_MIDDLE = (4, 5, 2)

# Kernel arguments for the exact computations: IEEE FP32 matmul, gradients accumulated in FP32
FULL_PRECISION = dict(use_high_precision_mm=True, input_grad_full_precision=True, weight_grad_full_precision=True)

# Max error of the exact FP32 computations, relative to the largest magnitude of the reference
FULL_PRECISION_TOLERANCE = 1e-5


@dataclass
class JointInputs:
    """Inputs of the fused Joint kernels; for the consistency loss - the double batch [teacher; student]"""

    joint_hidden: torch.Tensor  # [B, T, U+1, hidden_dim]
    weight: torch.Tensor  # [vocab_size, hidden_dim]
    bias: torch.Tensor  # [vocab_size]
    targets: torch.Tensor  # [B, U]
    src_lengths: torch.Tensor  # [B]
    tgt_lengths: torch.Tensor  # [B]
    blank_id: int
    lattice_shape: tuple[int, ...] = field(init=False)  # [B, T, U+1]

    def __post_init__(self):
        self.lattice_shape = tuple(self.joint_hidden.shape[:-1])

    def leaves(self, dtype: torch.dtype | None = None) -> list[torch.Tensor]:
        """Copies of (joint_hidden, weight, bias) that require gradient, optionally converted to `dtype`"""
        return [
            (value.detach().clone() if dtype is None else value.detach().to(dtype)).requires_grad_()
            for value in (self.joint_hidden, self.weight, self.bias)
        ]

    def kernel_kwargs(self, leaves: list[torch.Tensor]) -> dict:
        """Tensor arguments of the fused Joint functions with (joint_hidden, weight, bias) taken from `leaves`"""
        return dict(
            joint_hidden=leaves[0],
            targets=self.targets,
            tgt_lengths=self.tgt_lengths,
            src_lengths=self.src_lengths,
            weight=leaves[1],
            bias=leaves[2],
            blank_id=self.blank_id,
        )

    def random_grads(self, num_outputs: int = 2, seed: int = 1) -> list[torch.Tensor]:
        """
        Random upstream gradients for the outputs: target and blank log-probs [B, T, U+1],
        then (if requested) the consistency loss [B // 2, T, U+1]
        """
        generator = torch.Generator(device=self.joint_hidden.device).manual_seed(seed)
        shapes = [self.lattice_shape, self.lattice_shape, (self.lattice_shape[0] // 2, *self.lattice_shape[1:])]
        return [
            torch.randn(shape, device=self.joint_hidden.device, generator=generator) for shape in shapes[:num_outputs]
        ]


def make_joint_inputs(
    hidden_dtype: torch.dtype = torch.float32,
    param_dtype: torch.dtype = torch.float32,
    dims: tuple[int, int, int] = DIMS_DEFAULT,
    src_lengths: tuple[int, ...] = SRC_LENGTHS,
    tgt_lengths: tuple[int, ...] = TGT_LENGTHS,
    double_batch: bool = False,
    seed: int = 0,
    device: torch.device | str = "cuda",
) -> JointInputs:
    """
    Random inputs with logits of the order of 1, so that the distributions over the vocabulary are neither flat
    nor degenerate. With `double_batch`, the batch is duplicated as required by the consistency loss:
    the second (student) half has perturbed hidden states, identical targets and lengths.
    """
    hidden_dim, vocab_size, blank_id = dims
    batch_size, max_src_length, max_tgt_length = len(src_lengths), max(src_lengths), max(tgt_lengths)
    generator = torch.Generator(device=device).manual_seed(seed)
    joint_hidden = 0.5 * torch.randn(
        batch_size, max_src_length, max_tgt_length + 1, hidden_dim, device=device, generator=generator
    )
    weight = 0.3 * torch.randn(vocab_size, hidden_dim, device=device, generator=generator)
    bias = 0.3 * torch.randn(vocab_size, device=device, generator=generator)
    # at least one column: the kernels and the reference accept targets wider than the lattice
    targets = torch.randint(
        0, vocab_size - 1, (batch_size, max(max_tgt_length, 1)), device=device, generator=generator
    )
    targets += targets >= blank_id  # non-blank labels
    src_lengths = torch.tensor(src_lengths, device=device)
    tgt_lengths = torch.tensor(tgt_lengths, device=device)
    if double_batch:
        student_hidden = joint_hidden + 0.15 * torch.randn(joint_hidden.shape, device=device, generator=generator)
        joint_hidden = torch.cat([joint_hidden, student_hidden])
        targets, src_lengths, tgt_lengths = targets.repeat(2, 1), src_lengths.repeat(2), tgt_lengths.repeat(2)
    return JointInputs(
        joint_hidden=joint_hidden.to(hidden_dtype),
        weight=weight.to(param_dtype),
        bias=bias.to(param_dtype),
        targets=targets,
        src_lengths=src_lengths,
        tgt_lengths=tgt_lengths,
        blank_id=blank_id,
    )


def reference_outputs(
    inputs: JointInputs,
    leaves: list[torch.Tensor],
    weight_matmul_dtype: torch.dtype | None = None,
    consistency_symmetrical: bool | None = None,
) -> list[torch.Tensor]:
    """
    Reference target/blank log-probs (and the consistency loss if `consistency_symmetrical` is not None)
    from the materialized logits, computed in the dtype of `leaves` (FP64 for the exact reference).

    `weight_matmul_dtype` emulates the matmul operand of the kernel for the master weights of a higher precision:
    the weight is rounded to this dtype with a straight-through gradient.
    """
    joint_hidden, weight, bias = leaves
    if weight_matmul_dtype is not None:
        weight = weight + (weight.to(weight_matmul_dtype).to(weight.dtype) - weight).detach()
    logits = F.linear(joint_hidden, weight, bias)
    target_logprobs, blank_logprobs = rnnt_logprobs_torch(
        logits=logits,
        targets=inputs.targets,
        blank_id=inputs.blank_id,
        src_lengths=inputs.src_lengths,
        tgt_lengths=inputs.tgt_lengths,
    )
    if consistency_symmetrical is None:
        return [target_logprobs, blank_logprobs]
    half_batch_size = logits.shape[0] // 2
    _, blank_mask = get_rnnt_mask(
        batch_size=half_batch_size,
        src_length_max=logits.shape[1],
        tgt_length_max_plus_1=logits.shape[2],
        device=logits.device,
        src_lengths=inputs.src_lengths[:half_batch_size],
        tgt_lengths=inputs.tgt_lengths[:half_batch_size],
    )
    consistency_kl = kl_loss_torch(
        teacher_logprobs=F.log_softmax(logits[:half_batch_size], dim=-1),
        student_logprobs=F.log_softmax(logits[half_batch_size:], dim=-1),
        mask=blank_mask,
        symmetrical=consistency_symmetrical,
    )
    return [target_logprobs, blank_logprobs, consistency_kl]


def reference_pruning_windows(
    grad_target: torch.Tensor,
    grad_blank: torch.Tensor,
    src_lengths: torch.Tensor,
    tgt_lengths: torch.Tensor,
    range_size: int,
    paired: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference selection of the backward pruning windows: for each frame, the first window of `range_size`
    target positions (clipped to the valid positions) maximizing abs(target grad) + abs(blank grad).
    With `paired`, one window is selected for a (teacher, student) pair of frames of the double batch
    by the summed importance.

    Returns:
        starts of the windows [B, T] ([B // 2, T] if paired; zeros for padded frames)
        and the mask of the lattice positions inside the windows [B, T, U+1] (for the whole double batch if paired)
    """
    batch_size, max_src_length, max_tgt_length_plus_1 = grad_blank.shape
    tgt_positions = torch.arange(max_tgt_length_plus_1, device=grad_blank.device)
    target_valid = tgt_positions[None, :] < tgt_lengths[:, None]
    blank_valid = tgt_positions[None, :] <= tgt_lengths[:, None]
    importance = grad_target.abs() * target_valid[:, None, :] + grad_blank.abs() * blank_valid[:, None, :]
    if paired:
        batch_size //= 2
        importance = importance[:batch_size] + importance[batch_size:]
    starts = torch.zeros((batch_size, max_src_length), dtype=torch.int32, device=grad_blank.device)
    mask = torch.zeros((batch_size, max_src_length, max_tgt_length_plus_1), dtype=torch.bool, device=grad_blank.device)
    for batch_index in range(batch_size):
        tgt_length = int(tgt_lengths[batch_index])
        window_size = min(range_size, tgt_length + 1)
        for src_index in range(int(src_lengths[batch_index])):
            windows = importance[batch_index, src_index, : tgt_length + 1].unfold(0, window_size, 1)
            start = int(windows.sum(-1).argmax())  # the first maximum
            starts[batch_index, src_index] = start
            mask[batch_index, src_index, start : start + window_size] = True
    return starts, mask.repeat(2, 1, 1) if paired else mask


def random_pruning_range_starts(inputs: JointInputs, range_size: int, paired: bool = False) -> torch.Tensor:
    """
    Random starts of the pruning ranges inside the lattice, int32 [B, T] ([B // 2, T] if paired: ranges shared by
    the halves of the double batch). The kernels do not require connected ranges.
    """
    batch_size, max_src_length, max_tgt_length_plus_1 = inputs.lattice_shape
    num_utterances = batch_size // 2 if paired else batch_size
    generator = torch.Generator(device=inputs.joint_hidden.device).manual_seed(7)
    starts = torch.randint(
        0, max_tgt_length_plus_1, (num_utterances, max_src_length), device=generator.device, generator=generator
    )
    max_starts = (inputs.tgt_lengths[:num_utterances] + 1 - range_size).clamp(min=0)
    return starts.minimum(max_starts[:, None]).to(torch.int32)


def gather_pruning_ranges(joint_hidden: torch.Tensor, range_starts: torch.Tensor, range_size: int) -> torch.Tensor:
    """
    Joint hidden of the pruned lattice [B, T, range_size, hidden_dim]: the rows of the pruning ranges of the dense
    joint hidden [B, T, U+1, hidden_dim]. The gradient for the dense tensor is zero outside the ranges.
    """
    batch_size, max_src_length = joint_hidden.shape[:2]
    starts = range_starts.long().repeat(batch_size // range_starts.shape[0], 1)
    return joint_hidden[
        torch.arange(batch_size, device=starts.device)[:, None, None],
        torch.arange(max_src_length, device=starts.device)[None, :, None],
        starts[:, :, None] + torch.arange(range_size, device=starts.device),
    ]


def reference_pruned_lattice_outputs(
    inputs: JointInputs,
    leaves: list[torch.Tensor],
    range_starts: torch.Tensor,
    range_size: int,
    weight_matmul_dtype: torch.dtype | None = None,
    consistency_symmetrical: bool | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """
    `reference_outputs` restricted to the pruning ranges: log-probabilities [B, T, U+1] are `-inf` outside the ranges
    (and at the invalid positions inside them), the consistency loss is gathered for the positions of the ranges
    [B // 2, T, range_size] (zero at the invalid positions). The gradient of these outputs for the leaves is zero
    outside the ranges.

    Returns:
        the outputs and the mask of the ranges [B, T, U+1] (shared by the halves of the double batch)
    """
    batch_size, max_src_length, max_tgt_length_plus_1 = inputs.lattice_shape
    device = range_starts.device
    starts = range_starts.long().repeat(batch_size // range_starts.shape[0], 1)[:, :, None]
    tgt_positions = torch.arange(max_tgt_length_plus_1, device=device)
    range_mask = (tgt_positions >= starts) & (tgt_positions < starts + range_size)
    src_valid = torch.arange(max_src_length, device=device)[None, :, None] < inputs.src_lengths[:, None, None]
    blank_valid = range_mask & src_valid & (tgt_positions <= inputs.tgt_lengths[:, None, None])
    target_valid = blank_valid & (tgt_positions < inputs.tgt_lengths[:, None, None])
    outputs = reference_outputs(
        inputs, leaves, weight_matmul_dtype=weight_matmul_dtype, consistency_symmetrical=consistency_symmetrical
    )
    pruned_outputs = [
        outputs[0].masked_fill(~target_valid, float("-inf")),
        outputs[1].masked_fill(~blank_valid, float("-inf")),
    ]
    if consistency_symmetrical is not None:
        half_batch_size = batch_size // 2
        target_indices = starts[:half_batch_size] + torch.arange(range_size, device=device)
        pruned_outputs.append((outputs[2] * blank_valid[:half_batch_size]).gather(2, target_indices))
    return pruned_outputs, range_mask


def random_grads_like(outputs: list[torch.Tensor], seed: int = 1) -> list[torch.Tensor]:
    """Random upstream gradients of the shapes of the outputs"""
    generator = torch.Generator(device=outputs[0].device).manual_seed(seed)
    return [torch.randn(output.shape, device=output.device, generator=generator) for output in outputs]


def max_relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """
    Max absolute error relative to the largest magnitude of the reference (at least 1).

    An elementwise relative tolerance is not applicable: elements of the gradients are sums of terms of both signs,
    so the rounding error of an element is determined by the magnitude of the terms, not of the element itself.
    """
    expected = expected.double()
    return ((actual.double() - expected).abs().max() / expected.abs().max().clamp_min(1.0)).item()


def assert_close_to_reference(
    actual: list[torch.Tensor], expected: list[torch.Tensor], tolerance: float | list[float], names: list[str]
):
    """Check `max_relative_error` for each pair of tensors; `tolerance` is common or per-tensor"""
    tolerances = tolerance if isinstance(tolerance, (list, tuple)) else [tolerance] * len(names)
    assert len(actual) == len(expected) == len(names) == len(tolerances)
    for actual_tensor, expected_tensor, max_error, name in zip(actual, expected, tolerances, names):
        assert actual_tensor.shape == expected_tensor.shape, f"{name}: shape mismatch"
        error = max_relative_error(actual_tensor, expected_tensor)
        assert error <= max_error, f"{name}: relative error {error:.3g} > {max_error:.3g}"


def assert_pruned_outputs_close_to_reference(
    actual: list[torch.Tensor], expected: list[torch.Tensor], tolerance: float | list[float], names: list[str]
):
    """
    `assert_close_to_reference` for the outputs of the pruned lattice: the positions with zero probability (`-inf`)
    must be the same, the other values are compared
    """
    for actual_tensor, expected_tensor, name in zip(actual, expected, names):
        assert actual_tensor.shape == expected_tensor.shape, f"{name}: shape mismatch"
        assert torch.equal(torch.isneginf(actual_tensor), torch.isneginf(expected_tensor)), f"{name}: -inf mismatch"
    assert_close_to_reference(
        [tensor.nan_to_num(neginf=0.0) for tensor in actual],
        [tensor.nan_to_num(neginf=0.0) for tensor in expected],
        tolerance,
        names,
    )


def count_compiled_kernels(*modules) -> dict[str, int]:
    """
    Number of the compiled variants for each Triton kernel (`*_kernel`) of the modules.
    A kernel is compiled for each new combination of dtypes, `tl.constexpr` and specialized arguments.
    """
    counts = {}
    for module in modules:
        for name, kernel in vars(module).items():
            if not name.endswith("_kernel"):
                continue
            while not hasattr(kernel, "device_caches"):  # unwrap the autotuner
                kernel = kernel.fn
            counts[name] = sum(len(device_cache[0]) for device_cache in kernel.device_caches.values())
    return counts


def set_tensor_descriptors_allowed(monkeypatch: pytest.MonkeyPatch, module, allowed: bool) -> list[bool]:
    """
    Select how the Triton kernels of the module load / store the tiles: with tensor descriptors (`allowed`:
    the test is skipped if the GPU or Triton does not support them), or with pointers and explicit masks.
    Returns the list with `TENSOR_DESCRIPTORS_ALLOWED` argument of each kernel launched by the module.
    """
    if allowed and not module.can_use_tensor_descriptors(torch.zeros(4, 8, device="cuda")):
        pytest.skip("Tensor descriptors require compute capability 9.0+ and Triton 3.4+")
    if not allowed:
        monkeypatch.setattr(module, "can_use_tensor_descriptors", lambda *matrices: False)
    launched_with_descriptors_allowed = []
    launch_kernel_on_device = module.launch_kernel_on_device

    def record_launch(kernel, grid, device_tensor, **kernel_arguments):
        if "TENSOR_DESCRIPTORS_ALLOWED" in kernel_arguments:
            launched_with_descriptors_allowed.append(kernel_arguments["TENSOR_DESCRIPTORS_ALLOWED"])
        launch_kernel_on_device(kernel, grid, device_tensor, **kernel_arguments)

    monkeypatch.setattr(module, "launch_kernel_on_device", record_launch)
    return launched_with_descriptors_allowed


def skip_if_bf16_atomics_unsupported(backward_mode: str | None, kernel_kwargs: dict | None = None):
    """
    Skip a test of the Triton kernels with BF16 joint hidden, if the gradients are accumulated with BF16 atomics
    (atomic backward without `input_grad_full_precision`), which are not supported by Triton < 3.4
    """
    uses_bf16_atomics = backward_mode == "atomic" and not (kernel_kwargs or {}).get("input_grad_full_precision")
    if uses_bf16_atomics and TRITON_AVAILABLE and not BF16_ATOMICS_AVAILABLE:
        pytest.skip("BF16 atomics require Triton 3.4+")
