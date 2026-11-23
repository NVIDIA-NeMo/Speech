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

from typing import ContextManager

import torch
import triton
import triton.language as tl


def cuda_device_guard(tensor: torch.Tensor) -> ContextManager:
    """
    Context manager that makes the CUDA device of the tensor current.

    Triton compiles and launches kernels for the current CUDA device (`torch.cuda.current_device()`),
    which can differ from the device of the tensors (e.g., for a model moved to `cuda:1` manually).
    All kernel launches should be wrapped with this guard.
    """
    return torch.cuda.device(tensor.device)


@triton.jit
def log_add_exp(log_probs_1, log_probs_2):
    """
    Numerically stable log(exp(log_probs_1) + exp(log_probs_2)).
    If both arguments are `-inf` (zero probability), the result is `-inf`. NaN arguments are propagated.
    """
    max_score = tl.maximum(log_probs_1, log_probs_2)
    log_sum = max_score + tl.log(tl.exp(log_probs_1 - max_score) + tl.exp(log_probs_2 - max_score))
    # NB: `-inf - (-inf)` is NaN, the case of two `-inf` arguments is handled separately.
    # Both arguments are checked explicitly (not the maximum): `tl.maximum` ignores NaN, but NaN must be propagated
    both_neg_inf = (log_probs_1 == -float("inf")) & (log_probs_2 == -float("inf"))
    return tl.where(both_neg_inf, -float("inf"), log_sum)
