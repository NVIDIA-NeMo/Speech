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

import logging
import os
import socket
from typing import Any, Callable, Dict, Optional

import torch
from lightning_fabric.plugins import CheckpointIO
from lightning_fabric.utilities.types import _PATH
from megatron.core.dist_checkpointing.tensor_aware_state_dict import MCoreTensorAwareStateDict
from nvidia_resiliency_ext.checkpointing.local.base_state_dict import TensorAwareStateDict
from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.base_manager import BaseCheckpointManager
from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager import LocalCheckpointManager
from nvidia_resiliency_ext.checkpointing.local.replication.strategies import LazyCliqueReplicationStrategy
from nvidia_resiliency_ext.fault_tolerance.dict_utils import dict_list_map_inplace
from nvidia_resiliency_ext.ptl_resiliency.local_checkpoint_callback import (
    HierarchicalCheckpointIO,
    LocalCheckpointCallback,
)

from nemo.lightning.pytorch.trainer import Trainer
from nemo.utils.callbacks.dist_ckpt_io import AsyncCompatibleCheckpointIO, AsyncFinalizableCheckpointIO

logger = logging.getLogger(__name__)


class MCoreHierarchicalCheckpointIO(HierarchicalCheckpointIO, AsyncCompatibleCheckpointIO):
    """HierarchicalCheckpointIO implementation compatible with MCore distributed checkpointing.

    Args:
        wrapped_checkpoint_io (CheckpointIO): previously used checkpoint_io (for global checkpoints).
        local_ckpt_manager (BaseCheckpointManager): local checkpoint manager used to store the local checkpoints
        get_global_ckpt_iteration_fn (Callable[[_PATH], int]): a function that retrieves the iteration
            of a global checkpoint that will be compared with local checkpoint iteration
            in order to decide which to resume from.
        async_save (bool, optional): enables asynchronous save. Passed down to the local checkpoint
            manager unless overriden with `local_ckpt_options` in `_save_local_checkpoint`.
            If True, MCoreHierarchicalCheckpointIO must be wrapped with `AsyncFinalizableCheckpointIO` wrapper
        local_ckpt_algo (str, optional): local checkpoint save algorithm. See MCoreTensorAwareStateDict for details.
            By default, uses a fully parallel save and load algorithm ('fully_parallel`).
        parallelization_group (ProcessGroup, optional): save/load parallelization group
        allow_cache (bool, optional): if True, subsequent checkpoint saves will reuse
            the cached parallelization metadata.
    """

    def __init__(
        self,
        wrapped_checkpoint_io: CheckpointIO,
        local_ckpt_manager: BaseCheckpointManager,
        get_global_ckpt_iteration_fn: Callable[[_PATH], int],
        async_save: bool = False,
        local_ckpt_algo: str = "fully_parallel",
        parallelization_group: Optional[torch.distributed.ProcessGroup] = None,
        allow_cache: bool = False,
    ):
        super().__init__(wrapped_checkpoint_io, local_ckpt_manager, get_global_ckpt_iteration_fn, async_save)
        self.local_ckpt_algo = local_ckpt_algo
        self.parallelization_group = parallelization_group
        self.cached_metadata = None
        self.allow_cache = allow_cache

    def to_tensor_aware_state_dict(self, checkpoint: Dict[str, Any]) -> TensorAwareStateDict:
        """Specialized implementation using MCoreTensorAwareStateDict.

        Wraps the state dict in MCoreTensorAwareStateDict and makes sure
        that "common" state dict doesn't have any CUDA tensors.
        """
        state_dict_for_save, _ = MCoreTensorAwareStateDict.from_state_dict(
            checkpoint,
            algo=self.local_ckpt_algo,
            parallelization_group=self.parallelization_group,
            cached_metadata=self.cached_metadata,
        )

        def to_cpu(x):
            if isinstance(x, torch.Tensor) and x.device.type != "cpu":
                logger.debug("Moving CUDA tensor to CPU")
                x = x.to("cpu", non_blocking=True)
            return x

        dict_list_map_inplace(to_cpu, state_dict_for_save.common)
        if self.allow_cache:
            self.cached_metadata = None
        return state_dict_for_save

    def from_tensor_aware_state_dict(
        self, tensor_aware_checkpoint: TensorAwareStateDict, sharded_state_dict=None, strict=None
    ):
        """Unwraps MCoreTensorAwareStateDict to a plain state dict."""
        assert isinstance(
            tensor_aware_checkpoint, MCoreTensorAwareStateDict
        ), f"Unexpected tensor aware state dict type: {type(tensor_aware_checkpoint)}"
        if strict is not None:
            logger.warning("MCoreTensorAwareStateDict does not yet support the 'strict' argument.")

        return tensor_aware_checkpoint.to_state_dict(
            sharded_state_dict,
            algo=self.local_ckpt_algo,
            parallelization_group=self.parallelization_group,
        )


def update_trainer_local_checkpoint_io(
    trainer: Trainer,
    local_checkpoint_base_dir: str,
    get_global_ckpt_iteration_fn: Callable[[_PATH], int],
    **kwargs,
) -> None:
    """Update the Trainer with the corresponding MCoreHierarchicalCheckpointIO if local checkpointing is used.

    Args:
        trainer (nl.Trainer): Trainer object to drive training loop.
        local_checkpoint_base_dir (str): Root directory under which to save local checkpoints.
        get_global_ckpt_iteration_fn (Callable): a function that retrieves the iteration of a global checkpoint
            that will be compared with local checkpoint iteration in order to decide which to resume from.
        **kwargs (dict): Additional kwargs passed to initialize MCoreHierarchicalCheckpointIO.

    Note:
        Async saving of local checkpoints is inferred based on what was configured on the strategy, if available.

    """
    callbacks = trainer.callbacks
    use_local_ckpt = any(isinstance(cb, LocalCheckpointCallback) for cb in callbacks)
    if not use_local_ckpt:
        return

    checkpoint_io = trainer.strategy.checkpoint_io
    # Infer async save setting for local checkpoint
    # based on whether async saving is configured for saving the global checkpoints
    async_save = getattr(trainer.strategy, "async_save", False)
    if async_save:
        # Access inner checkpoint IO
        assert isinstance(checkpoint_io, AsyncFinalizableCheckpointIO), type(checkpoint_io)
        checkpoint_io = checkpoint_io.checkpoint_io

    if trainer.num_nodes > 1:
        repl_strategy = LazyCliqueReplicationStrategy()
    else:
        # No replication for single node
        repl_strategy = None

    local_ckpt_manager = LocalCheckpointManager(
        os.path.join(local_checkpoint_base_dir, "local_ckpt", socket.gethostname()),
        repl_strategy=repl_strategy,
    )
    hierarchical_checkpointing_io = MCoreHierarchicalCheckpointIO(
        checkpoint_io,
        local_ckpt_manager,
        get_global_ckpt_iteration_fn,
        async_save=async_save,
        **kwargs,
    )

    if async_save:
        hierarchical_checkpointing_io = AsyncFinalizableCheckpointIO(hierarchical_checkpointing_io)

    trainer.strategy.checkpoint_io = hierarchical_checkpointing_io
