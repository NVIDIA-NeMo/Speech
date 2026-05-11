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

from __future__ import annotations

import os
from datetime import timedelta
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning.pytorch.strategies.model_parallel import ModelParallelStrategy
from typing_extensions import override


def setup_distributed(
    tp_size: int = 1,
    pp_size: int = 1,
    cp_size: int = 1,
    ep_size: int = 1,
    dp_size: int | None = None,
    dp_replicate_size: int | None = None,
    distributed_config=None,
    moe_config=None,
    activation_checkpointing_llm: bool = False,
    activation_checkpointing_perception: bool = False,
    backend: str = "nccl",
) -> AutomodelParallelStrategy:
    """Initialize torch.distributed, set CUDA device, and create a device mesh.

    This is a convenience function for inference scripts that need distributed
    model-parallel loading without a Lightning Trainer.

    Returns an :class:`AutomodelParallelStrategy` with ``device_mesh``,
    ``distributed_config``, ``moe_config``, and ``moe_mesh`` ready to use.
    """
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    strategy = AutomodelParallelStrategy(
        tp_size=tp_size,
        pp_size=pp_size,
        cp_size=cp_size,
        ep_size=ep_size,
        dp_size=dp_size,
        dp_replicate_size=dp_replicate_size,
        distributed_config=distributed_config,
        moe_config=moe_config,
        activation_checkpointing_llm=activation_checkpointing_llm,
        activation_checkpointing_perception=activation_checkpointing_perception,
    )
    strategy.create_device_mesh()
    return strategy


class AutomodelParallelStrategy(ModelParallelStrategy):
    """A Lightning strategy using nemo_automodel for device mesh creation,
    supporting extended parallelism: FSDP2, TP, PP, CP, EP, and HSDP.

    This is a drop-in replacement for ``ModelParallelStrategy`` that delegates
    device mesh creation to
    ``nemo_automodel.components.distributed.mesh_utils.create_device_mesh``.

    The resulting device mesh has dimensions ``(pp, dp_replicate, dp_shard, cp, tp)``
    with flattened submeshes ``dp``, ``dp_shard_cp``, and ``dp_cp``.

    Models using this strategy access the mesh in ``configure_model()`` via
    ``self.device_mesh`` and can use dimension names like ``"tp"``, ``"dp"``,
    ``"dp_shard"``, ``"cp"``, ``"pp"``, etc.

    Args:
        dp_size: Data parallel size. If None, inferred from world_size and
            other parallelism sizes.
        dp_replicate_size: HSDP replication group size. If None, defaults to 1.
        tp_size: Tensor parallel size.
        pp_size: Pipeline parallel size.
        cp_size: Context parallel size.
        ep_size: Expert parallel size (for MoE models).
        distributed_config: An ``FSDP2Config`` (or ``MegatronFSDPConfig``/``DDPConfig``)
            from nemo_automodel. If None, a default ``FSDP2Config()`` is created.
        moe_config: An ``MoEParallelizerConfig`` from nemo_automodel. Optional.
        activation_checkpointing_llm: Enable activation checkpointing for LLM
            transformer blocks. When True, this single knob covers both paths:
            FSDP2 AC (by forcing ``FSDP2Config.activation_checkpointing=True``)
            and the EP/MoE parallelizer AC (``MoEParallelizerConfig`` has no
            such field; the EP parallelizer reads it as a separate runtime arg).
        activation_checkpointing_perception: Enable activation checkpointing
            for the perception encoder's transformer layers (applied with
            ``checkpoint_wrapper`` before FSDP2 sharding).
        save_distributed_checkpoint: If True, each rank saves its shard of weights
            and optimizer states. If False, full state is assembled on rank 0.
        process_group_backend: Distributed backend (e.g. ``"nccl"``).
        timeout: Process group initialization timeout.
    """

    def __init__(
        self,
        dp_size: Optional[int] = None,
        dp_replicate_size: Optional[int] = None,
        tp_size: int = 1,
        pp_size: int = 1,
        cp_size: int = 1,
        ep_size: int = 1,
        distributed_config=None,
        moe_config=None,
        activation_checkpointing_llm: bool = False,
        activation_checkpointing_perception: bool = False,
        save_distributed_checkpoint: bool = True,
        process_group_backend: Optional[str] = None,
        timeout: Optional[timedelta] = default_pg_timeout,
    ) -> None:
        super().__init__(
            # These are unused because we override setup_environment(),
            # but the base class requires them.
            data_parallel_size=1,
            tensor_parallel_size=1,
            save_distributed_checkpoint=save_distributed_checkpoint,
            process_group_backend=process_group_backend,
            timeout=timeout,
        )
        self._dp_size = dp_size
        self._dp_replicate_size = dp_replicate_size
        self._tp_size = tp_size
        self._pp_size = pp_size
        self._cp_size = cp_size
        self._ep_size = ep_size
        self._distributed_config = distributed_config
        self._moe_config = moe_config
        self._activation_checkpointing_llm = activation_checkpointing_llm
        self._activation_checkpointing_perception = activation_checkpointing_perception
        self._moe_mesh = None

    @property
    def moe_mesh(self):
        """The MoE device mesh, or None if expert parallelism is not used."""
        return self._moe_mesh

    @property
    def distributed_config(self):
        """The nemo_automodel distributed configuration."""
        return self._distributed_config

    @property
    def moe_config(self):
        """The nemo_automodel MoE configuration."""
        return self._moe_config

    @property
    def activation_checkpointing_llm(self) -> bool:
        """Whether activation checkpointing is enabled for the LLM.

        Covers both FSDP2 AC and EP/MoE AC paths.
        """
        return self._activation_checkpointing_llm

    @property
    def activation_checkpointing_perception(self) -> bool:
        """Whether activation checkpointing is enabled for the perception encoder."""
        return self._activation_checkpointing_perception

    def create_device_mesh(self):
        """Create the device mesh from the configured parallelism sizes.

        Requires ``torch.distributed`` to already be initialized.  This is
        called automatically by :meth:`setup_environment`, but can also be
        called standalone (e.g. in checkpoint-conversion scripts) after
        manual ``dist.init_process_group``.

        Returns:
            Tuple of ``(device_mesh, moe_mesh)``.
        """
        from nemo_automodel.components.distributed.config import FSDP2Config
        from nemo_automodel.components.distributed.mesh_utils import create_device_mesh

        if self._distributed_config is None:
            self._distributed_config = FSDP2Config()

        self._device_mesh, self._moe_mesh = create_device_mesh(
            self._distributed_config,
            dp_size=self._dp_size,
            dp_replicate_size=self._dp_replicate_size,
            tp_size=self._tp_size,
            pp_size=self._pp_size,
            cp_size=self._cp_size,
            ep_size=self._ep_size,
            world_size=dist.get_world_size(),
        )
        return self._device_mesh, self._moe_mesh

    @override
    def setup_environment(self) -> None:
        # Initialize accelerator device and distributed process group.
        self._setup_distributed()

        self.create_device_mesh()

        # Make device mesh accessible to the LightningModule via self.device_mesh
        assert self.lightning_module is not None
        self.lightning_module._device_mesh = self._device_mesh
        self.lightning_module._moe_mesh = self._moe_mesh

    @property
    @override
    def distributed_sampler_kwargs(self) -> Dict[str, Any]:
        if self._device_mesh is None:
            raise RuntimeError("Accessing distributed_sampler_kwargs before setup_environment() is not allowed.")
        # automodel's flattened "dp" submesh covers dp_replicate * dp_shard
        dp_mesh = self._device_mesh["dp"]
        return {"num_replicas": dp_mesh.size(), "rank": dp_mesh.get_local_rank()}
