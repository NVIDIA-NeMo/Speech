# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Bound temporary accelerator memory while collecting optimizer checkpoints."""

from contextlib import contextmanager
from types import MethodType

import torch
from torch.distributed.tensor import DTensor


def _fused_adam_cpu_state_dict(optimizer):
    # Keep the standard optimizer metadata/hooks, but transfer each converted
    # state before converting the next one. TE otherwise accumulates all FP32
    # moments on the accelerator before DCP can apply cpu_offload.
    state_dict = torch.optim.Optimizer.state_dict(optimizer)
    parameters = {
        saved: parameter
        for saved_group, group in zip(state_dict["param_groups"], optimizer.param_groups)
        for saved, parameter in zip(saved_group["params"], group["params"])
    }
    for key, state in state_dict["state"].items():
        if key not in parameters:
            continue
        parameter = parameters[key]
        converted = {}
        for name in state:
            value = optimizer.get_unscaled_state(parameter, name)
            if isinstance(parameter, DTensor):
                value = DTensor.from_local(
                    value,
                    device_mesh=parameter.device_mesh,
                    placements=parameter.placements,
                    shape=parameter.size(),
                    stride=parameter.stride(),
                )
            converted[name] = value.cpu()
            del value
        state_dict["state"][key] = converted
    return state_dict


@contextmanager
def _cpu_optimizer_checkpoint(optimizer):
    """Collect TE FusedAdam's canonical state on CPU without retaining FP32 GPU copies.

    The live low-precision state is unchanged. The temporary instance method is
    restored even if checkpoint collection raises; other optimizers are untouched.
    """
    if not callable(getattr(optimizer, "get_unscaled_state", None)):
        yield
        return
    try:
        from transformer_engine.pytorch.optimizers import FusedAdam
    except ImportError:
        yield
        return

    if (
        not isinstance(optimizer, FusedAdam)
        or type(optimizer).state_dict is not FusedAdam.state_dict
        or "state_dict" in optimizer.__dict__
    ):
        yield
        return
    optimizer.state_dict = MethodType(_fused_adam_cpu_state_dict, optimizer)
    try:
        yield
    finally:
        del optimizer.state_dict
