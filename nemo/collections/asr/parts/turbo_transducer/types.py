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
        blank_logprobs: log-probabilities for the blank label, `[B, T, U+1]`.
        consistency_kl: optional per-position consistency (KL) loss, `[B // 2, T, U+1]`, computed between
            the first (teacher) and the second (student) half of a double batch. `None` if not requested.
    """

    target_logprobs: torch.Tensor
    blank_logprobs: torch.Tensor
    consistency_kl: Optional[torch.Tensor] = None


class TurboJointKernelType(PrettyStrEnum):
    """
    Optimized (training-only) Joint kernel.

    - FUSED_VOCAB: the vocabulary projection is fused with log-softmax, the Joint hidden `[B, T, U+1, D]` is
      materialized, but the logits `[B, T, U+1, V]` are not.
    """

    FUSED_VOCAB = "fused_vocab"


@dataclass
class TurboJointBackwardPruningConfig:
    """
    Backward pruning for the Joint turbo kernel: the gradient is computed only for `range_size` contiguous target
    positions per frame. The forward pass (the loss value) is not changed, the gradient is approximated.

    Fields:
        range_size: number of target positions retained per frame. None or a non-positive value disables pruning.
        start_step: the first optimizer step with pruning; before it the full gradient is computed.
            None or a non-positive value enables pruning from the first step.
    """

    range_size: Optional[int] = None
    start_step: Optional[int] = None

    @property
    def enabled(self) -> bool:
        return self.range_size is not None and self.range_size > 0

    def is_active(self, current_step: Optional[int]) -> bool:
        """Whether pruning should be applied at the given optimizer step"""
        if not self.enabled:
            return False
        if self.start_step is None or self.start_step <= 0:
            return True
        if current_step is None:
            raise ValueError(
                f"Backward pruning is scheduled to start at step {self.start_step}: the current step is required"
            )
        return current_step >= self.start_step


@dataclass
class TurboJointKernelConfig:
    """
    Configuration of the Joint turbo kernel (`RNNTJoint(turbo_kernel=...)`).

    Fields:
        type: kernel type, see `TurboJointKernelType`. None disables the turbo kernel (other fields are ignored).
        backend: `triton` or `cutile`; None selects the available backend automatically (Triton is preferred).
        use_high_precision_mm: use FP32 matmul operands with IEEE input precision (Triton).
        mm_full_precision: use FP32 matmul operands for BF16 activations (by default BF16 operands are used).
        input_grad_full_precision: accumulate the gradient for the Joint hidden in FP32.
        weight_grad_full_precision: accumulate the gradient for the vocabulary projection weight in FP32.
        backward_pruning: optional backward pruning, see `TurboJointBackwardPruningConfig`. None disables pruning.
    """

    type: Optional[str] = None
    backend: Optional[str] = TurboTransducerBackend.TRITON.value
    use_high_precision_mm: bool = False
    mm_full_precision: bool = False
    input_grad_full_precision: bool = False
    weight_grad_full_precision: bool = False
    backward_pruning: Optional[TurboJointBackwardPruningConfig] = None

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]]) -> "TurboJointKernelConfig":
        """
        Construct the validated config from a mapping (dict or DictConfig).
        Unknown keys and values of wrong types are rejected; missing keys get the defaults.
        """
        if config is None:
            return cls()
        if not isinstance(config, Mapping) and not OmegaConf.is_dict(config):
            raise TypeError(f"`turbo_kernel` must be a mapping or None, got {type(config).__name__}")
        kernel_config: TurboJointKernelConfig = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(cls), config))
        if kernel_config.type is not None:
            TurboJointKernelType(kernel_config.type)  # validate
            if kernel_config.backend is not None and TurboTransducerBackend(kernel_config.backend) not in {
                TurboTransducerBackend.TRITON,
                TurboTransducerBackend.CUTILE,
            }:
                raise ValueError(
                    f"Backend `{kernel_config.backend}` is not supported by the Joint turbo kernel, "
                    f"use `{TurboTransducerBackend.TRITON}` or `{TurboTransducerBackend.CUTILE}`"
                )
        return kernel_config

    @property
    def kernel_type(self) -> Optional[TurboJointKernelType]:
        return TurboJointKernelType(self.type) if self.type is not None else None

    @property
    def uses_backward_pruning(self) -> bool:
        return self.backward_pruning is not None and self.backward_pruning.enabled

    def get_backward_pruning_range_size(self, current_step: Optional[int]) -> Optional[int]:
        """Pruning range size to use at the given optimizer step, None if pruning is not applied"""
        if self.backward_pruning is None or not self.backward_pruning.is_active(current_step):
            return None
        return self.backward_pruning.range_size
