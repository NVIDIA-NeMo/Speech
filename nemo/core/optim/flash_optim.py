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

from typing import Any

import torch

from nemo.utils import logging


def patch_flashoptim_uneven_shard_support(optimizer) -> None:
    """Patch flashoptim to handle FSDP2 unevenly-sharded parameters in DCP.

    FlashOptim <= 0.1.3 raises ``ValueError`` when saving optimizer state for
    parameters whose shard dimension is not evenly divisible by the FSDP mesh
    size. The root cause is that ``DTensor.from_local()`` is called without an
    explicit ``shape``, so it infers ``global = local * mesh_size`` which is
    wrong for padded (uneven) shards.

    This patch replaces ``_wrap_state_as_dtensor`` on the optimizer class so
    that ``shape=param.shape`` and ``stride=param.stride()`` are always passed,
    which is correct for both even and uneven shards.
    """

    klass = type(optimizer)
    if not hasattr(klass, "_wrap_state_as_dtensor"):
        return
    if getattr(klass, "_nemo_patched_uneven_shard", False):
        return

    @staticmethod
    def _fixed_wrap_state_as_dtensor(state: dict[str, Any], param: torch.Tensor) -> None:  # noqa: UP006
        if not hasattr(param, "device_mesh"):
            return

        from torch.distributed.tensor import DTensor

        mesh = param.device_mesh
        placements = param.placements

        for key, val in state.items():
            if isinstance(val, torch.Tensor) and not isinstance(val, DTensor) and val.dim() > 0:
                state[key] = DTensor.from_local(
                    val,
                    mesh,
                    placements,
                    shape=param.shape,
                    stride=param.stride(),
                )

    klass._wrap_state_as_dtensor = _fixed_wrap_state_as_dtensor
    klass._nemo_patched_uneven_shard = True
    logging.info("Patched flashoptim %s to support unevenly-sharded FSDP2 parameters in DCP.", klass.__name__)
