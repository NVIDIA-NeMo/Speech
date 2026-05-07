# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Context-Parallelism (CP) helpers for SALMAutomodel.

Read CP-related metadata out of the device mesh that ``AutomodelParallelStrategy``
hands to the LightningModule. ``get_cp_mesh`` returns ``(None, 1, 0)`` when CP
is inactive so callers can short-circuit any CP-specific work without an extra
``hasattr``/``mesh_dim_names`` dance.
"""
from __future__ import annotations

from typing import Optional

import torch.distributed as dist


def get_cp_mesh(device_mesh) -> tuple[Optional[object], int, int]:
    """Return ``(cp_mesh, cp_size, cp_rank)`` or ``(None, 1, 0)`` when CP is inactive."""
    if device_mesh is None:
        return None, 1, 0
    names = device_mesh.mesh_dim_names or ()
    if "cp" not in names or device_mesh["cp"].size() <= 1:
        return None, 1, 0
    cp_mesh = device_mesh["cp"]
    cp_rank = dist.get_rank(group=cp_mesh.get_group())
    return cp_mesh, cp_mesh.size(), cp_rank
