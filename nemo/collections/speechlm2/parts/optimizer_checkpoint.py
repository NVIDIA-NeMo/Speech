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

"""Bound checkpoint memory and restore compact optimizer state in place."""

from contextlib import contextmanager
from copy import copy
from types import MethodType

import torch
from torch.distributed.tensor import DTensor


class _FusedAdamRestoreOptimizer(torch.optim.Optimizer):
    """Expose live compact buffers to DCP and apply only loaded group metadata.

    PyTorch's state-dict utilities retain responsibility for parameter names,
    groups, and DTensor sharding. The adapter never calls TE's bulk serializer
    or loader, both of which would create additional parameter-state copies.
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        groups = [dict(group, step=group.get("step", 0)) for group in optimizer.param_groups]
        super().__init__(groups, copy(optimizer.defaults))
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if not optimizer.state[parameter]:
                    optimizer.initialize_state(parameter, store_param_remainders=(parameter.dtype == torch.bfloat16))
        self.state = optimizer.state

    def step(self, closure=None):
        """A checkpoint adapter performs no training updates."""

    def state_dict(self):
        """Return references to live compact optimizer buffers."""
        return super().state_dict()

    def load_state_dict(self, state_dict):
        """DCP already filled the live buffers; only metadata remains to apply."""
        parameter_ids = {key for group in state_dict["param_groups"] for key in group["params"]}
        metadata = dict(state_dict)
        metadata["state"] = {key: value for key, value in state_dict["state"].items() if key not in parameter_ids}
        super().load_state_dict(metadata)
        self.optimizer.param_groups = self.param_groups
        self.optimizer.state.update(self.state)
        self.state = self.optimizer.state


def _fused_adam_restore_optimizer(optimizer):
    """Return an in-place DCP adapter for BF16-moment FusedAdam state.

    Other moment dtypes can require per-tensor scaling; their existing restore
    path is retained. Custom serialization and load hooks retain their own semantics.
    """
    if not callable(getattr(optimizer, "get_unscaled_state", None)):
        return None
    try:
        from transformer_engine.pytorch.optimizers import FusedAdam
    except ImportError:
        return None
    if (
        type(optimizer) is not FusedAdam  # pylint: disable=unidiomatic-typecheck
        or "state_dict" in optimizer.__dict__
        or "load_state_dict" in optimizer.__dict__
        or any(
            getattr(optimizer, name, {})
            for name in (
                "_optimizer_state_dict_pre_hooks",
                "_optimizer_state_dict_post_hooks",
                "_optimizer_load_state_dict_pre_hooks",
                "_optimizer_load_state_dict_post_hooks",
            )
        )
        or not optimizer.master_weights
        or not optimizer.store_param_remainders
        or optimizer.name_to_dtype_map["exp_avg"] != torch.bfloat16
        or optimizer.name_to_dtype_map["exp_avg_sq"] != torch.bfloat16
        or any(
            parameter.dtype not in (torch.bfloat16, torch.float32)
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
    ):
        return None
    return _FusedAdamRestoreOptimizer(optimizer)


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
