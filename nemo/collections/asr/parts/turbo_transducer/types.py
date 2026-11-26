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

from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple, Optional

import torch
from omegaconf import OmegaConf

from nemo.collections.asr.parts.turbo_transducer.backend import TurboTransducerBackend
from nemo.utils.enum import PrettyStrEnum


class TransducerLossInputs(NamedTuple):
    """
    Per-position quantities extracted from the Joint output, sufficient to compute the transducer losses.

    All tensors here are small (no vocabulary dimension), which allows the loss functions to operate
    without materializing (or keeping gradients for) the full `[B, T, U+1, V]` Joint output.

    Fields:
        target_logprobs: log-probabilities for target labels, `[B, T, U+1]`. Zero for non-existent targets.
            With the pruned lattice (pruning ranges applied in forward): `-inf` outside the pruning ranges.
        blank_logprobs: log-probabilities for the blank label, `[B, T, U+1]`.
            With the pruned lattice: `-inf` outside the pruning ranges.
        consistency_kl: optional per-position consistency (KL) loss, `[B // 2, T, U+1]`, computed between
            the first (teacher) and the second (student) half of a double batch. `None` if not requested.
            With the pruned lattice: only for the positions of the pruning ranges, `[B // 2, T, range_size]`.
    """

    target_logprobs: torch.Tensor
    blank_logprobs: torch.Tensor
    consistency_kl: Optional[torch.Tensor] = None


class JointLossInputs(NamedTuple):
    """
    Output of `Joint.forward_for_loss`: the fields of `TransducerLossInputs` and the values reported by the Joint.

    Fields:
        target_logprobs, blank_logprobs, consistency_kl: see `TransducerLossInputs`
        full_lattice_loss: optional RNN-T loss of the full (not pruned) lattice for each utterance, `[B]`,
            without gradient. Computed by the first pass of the two-pass pruning (no dropout, no FastEmit scaling)
            and intended for logging; `None` otherwise.
    """

    target_logprobs: torch.Tensor
    blank_logprobs: torch.Tensor
    consistency_kl: Optional[torch.Tensor] = None
    full_lattice_loss: Optional[torch.Tensor] = None


class TrainingKernelType(PrettyStrEnum):
    """
    Optimized (training-only) Joint kernel.

    - FLASH: packs valid lattice positions and recomputes the Joint in row tiles during backward.
    - FUSED_VOCAB: the vocabulary projection is fused with log-softmax, the Joint hidden `[B, T, U+1, D]` is
      materialized, but the logits `[B, T, U+1, V]` are not.
    """

    FLASH = "flash"
    FUSED_VOCAB = "fused_vocab"


class FusedVocabBackwardMode(PrettyStrEnum):
    """
    Backward algorithm for the Triton `fused_vocab` Joint.

    - CHUNKED (default): processes the flattened batch in row chunks: a Triton kernel recomputes logits and writes
      grad logits for the chunk into a bounded buffer, then two GEMMs compute grad_joint_hidden and accumulate
      grad_weight. Avoids large-scale atomics; memory overhead is bounded by the chunk buffer (independent of
      B, T, U).
    - ATOMIC (experimental): single fused kernel; recomputes logits per (flattened batch, vocab) tile and accumulates
      grad_joint_hidden / grad_weight / grad_bias with atomics. No buffer for the grad logits. Native bf16 atomics
      require sm_90+ (on older GPUs they are emulated with compare-and-swap loops) and Triton 3.4+.
      Slower than CHUNKED on the GPUs where both modes were measured (RTX A6000, GB300).
    """

    ATOMIC = "atomic"
    CHUNKED = "chunked"


class TurboJointPruningType(PrettyStrEnum):
    """
    Pruning of the Joint lattice (training only): the Joint is evaluated only for a window of `range_size`
    contiguous target positions per frame.

    - BACKWARD: the forward pass is computed for the full lattice (the loss value is exact), the windows are
      selected by the gradient of the loss, and the gradient of the Joint is computed only inside the windows
      (approximated gradient). Makes the backward pass faster.
    - TWO_PASS: the windows are selected by a forward pass for the full lattice without gradient (no dropout),
      then the Joint is evaluated only inside the windows, with gradient. The loss is the loss of the pruned
      lattice (not less than the exact one), and its gradient is exact. The activations with gradient are stored
      only for the windows, which saves memory.
    """

    BACKWARD = "backward"
    TWO_PASS = "two_pass"


@dataclass
class TurboJointPruningConfig:
    """
    Pruning for the Joint turbo kernel, see `TurboJointPruningType`.

    Fields:
        type: pruning type. None disables pruning (other fields are ignored).
        range_size: number of target positions retained per frame; required if `type` is set.
            At least 1 for `backward` and at least 2 for `two_pass` (a window of a single position can't emit).
        start_step: the first optimizer step with pruning; before it the full lattice is used.
            None or a non-positive value enables pruning from the first step.
    """

    type: Optional[str] = None
    range_size: Optional[int] = None
    start_step: Optional[int] = None

    @property
    def pruning_type(self) -> Optional[TurboJointPruningType]:
        return TurboJointPruningType(self.type) if self.type is not None else None

    @property
    def enabled(self) -> bool:
        return self.type is not None

    def validate(self):
        """Check the pruning type and the range size (strict: invalid values are rejected, not ignored)"""
        pruning_type = self.pruning_type
        if pruning_type is None:
            return
        min_range_size = 2 if pruning_type is TurboJointPruningType.TWO_PASS else 1
        if self.range_size is None or self.range_size < min_range_size:
            raise ValueError(
                f"Pruning `{pruning_type}` requires `range_size` >= {min_range_size}, got {self.range_size}"
            )

    def is_active(self, current_step: Optional[int]) -> bool:
        """Whether pruning should be applied at the given optimizer step"""
        if not self.enabled:
            return False
        if self.start_step is None or self.start_step <= 0:
            return True
        if current_step is None:
            raise ValueError(f"Pruning is scheduled to start at step {self.start_step}: the current step is required")
        return current_step >= self.start_step


@dataclass
class TrainingKernelConfig:
    """Select the training joint. Only options for the selected implementation are accepted."""

    type: Optional[str] = None

    @property
    def kernel_type(self) -> Optional[TrainingKernelType]:
        return TrainingKernelType(self.type) if self.type is not None else None

    @staticmethod
    def from_config(config: Optional[Mapping[str, Any]]) -> "TrainingKernelConfig":
        if config is not None and not isinstance(config, Mapping):
            raise TypeError(f"`training_kernel` must be a mapping or None, got {type(config).__name__}")
        if config is None or config.get("type") is None:
            return TrainingKernelConfig()
        kind = TrainingKernelType(config["type"])
        schema = FlashKernelConfig if kind is TrainingKernelType.FLASH else FusedVocabKernelConfig
        options = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(schema), config))
        options.validate()
        return options


@dataclass
class FlashKernelConfig(TrainingKernelConfig):
    """Maximum number of valid alignment positions per recomputed tile."""

    max_joint_rows: int = 200_000

    def validate(self):
        if self.max_joint_rows < 1:
            raise ValueError("max_joint_rows must be positive")


@dataclass
class FusedVocabKernelConfig(TrainingKernelConfig):
    """Vocabulary projection, gradient accumulation and optional lattice pruning."""

    backend: Optional[str] = TurboTransducerBackend.TRITON.value
    use_high_precision_mm: bool = False
    mm_full_precision: bool = False
    input_grad_full_precision: bool = False
    weight_grad_full_precision: bool = False
    backward_mode: Optional[str] = FusedVocabBackwardMode.CHUNKED.value
    pruning: Optional[TurboJointPruningConfig] = None

    def validate(self):
        check_turbo_joint_backend(self.backend)
        if self.backward_mode is not None:
            FusedVocabBackwardMode(self.backward_mode)
        if self.pruning is not None:
            self.pruning.validate()

    @property
    def uses_pruning(self) -> bool:
        return self.pruning is not None and self.pruning.enabled

    def get_active_pruning(self, current_step: Optional[int]) -> Optional[TurboJointPruningConfig]:
        """Pruning config to apply at the given optimizer step, None if pruning is not applied"""
        if self.pruning is None or not self.pruning.is_active(current_step):
            return None
        return self.pruning


def check_turbo_joint_backend(backend: Optional[str | TurboTransducerBackend]):
    """
    Check the backend of the Joint turbo kernel: only Triton is implemented, None selects it.

    Raises:
        NotImplementedError: for the cuTile backend (planned, not implemented yet)
        ValueError: for any other backend
    """
    if backend is None:
        return
    backend = TurboTransducerBackend(backend)  # ValueError for an unknown name
    if backend is TurboTransducerBackend.CUTILE:
        raise NotImplementedError(
            f"cuTile backend (`{backend}`) is not implemented yet for the Joint turbo kernel, "
            f"use `{TurboTransducerBackend.TRITON}`"
        )
    if backend is not TurboTransducerBackend.TRITON:
        raise ValueError(
            f"Backend `{backend}` is not supported by the Joint turbo kernel, use `{TurboTransducerBackend.TRITON}`"
        )
