# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, cast

import pytorch_lightning as pl
import torch
from lightning_fabric.plugins import ClusterEnvironment
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedBase, ShardedObject, ShardedTensor
from megatron.core.dist_checkpointing.strategies.torch import sharded_tensor_to_torch_sharded_tensor
from megatron.core.transformer.utils import _get_extra_state_offsets
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loops.progress import _BatchProgress
from pytorch_lightning.plugins.io.wrapper import _WrappingCheckpointIO
from torch.distributed._sharded_tensor import ShardedTensor as TorchShardedTensor
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh
from typing_extensions import override

from nemo.lightning import _strategy_lib
from nemo.lightning.io.pl import MegatronCheckpointIO
from nemo.lightning.pytorch.callbacks import MegatronProgressBar, ProgressPrinter
from nemo.utils.callbacks.dist_ckpt_io import AsyncFinalizableCheckpointIO


@dataclass(kw_only=True)
class RestoreConfig:
    path: str
    adapter_path: Optional[str] = None
    load_model_state: bool = True
    load_optim_state: bool = False
    # eg tokenizer, etc.
    load_artifacts: bool = True


class _MegatronBatchProgress(_BatchProgress):
    @override
    def load_state_dict(self, state_dict: dict) -> None:
        ## in megatron, we want to start the batch progress over when
        ## restoring from a checkpoint
        return


def setup_parallel_ranks(strategy: pl.strategies.Strategy):
    from megatron.core.model_parallel_config import ModelParallelConfig

    env = cast(ClusterEnvironment, strategy.cluster_environment)
    parallelism = getattr(strategy, "parallelism", ModelParallelConfig())
    _strategy_lib.init_parallel_ranks(env.world_size(), env.global_rank(), env.local_rank(), parallelism)


def init_model_parallel(pl_module: pl.LightningModule):
    from megatron.core import parallel_state

    from nemo.utils import AppState

    if not parallel_state.model_parallel_is_initialized():
        app_state = AppState()

        if app_state.model_parallel_size is not None:
            _strategy_lib.init_model_parallel(pl_module)


def setup_data_sampler(trainer: pl.Trainer):
    datamodule = getattr(trainer, "datamodule", None)
    if datamodule is not None:
        if hasattr(trainer.strategy, "data_sampler") and trainer.strategy.data_sampler is not None:
            datamodule.data_sampler = trainer.strategy.data_sampler
        elif hasattr(datamodule, "data_sampler"):
            trainer.strategy.data_sampler = datamodule.data_sampler

    if trainer.strategy.data_sampler is not None:
        trainer.strategy.data_sampler.setup(trainer.strategy.cluster_environment.global_rank())
        trainer.strategy.data_sampler.connect(trainer)

    if hasattr(datamodule, "reconfigure_limit_batches"):
        datamodule.reconfigure_limit_batches()


def fix_progress_bar(trainer: pl.Trainer, replace_progress_bar: bool = True, progress_interval: int = 1) -> None:
    callbacks: List[pl.Callback] = cast(List[pl.Callback], getattr(trainer, "callbacks"))
    contains_megatron_progress, contains_progress = False, False
    for callback in callbacks:
        if isinstance(callback, MegatronProgressBar):
            contains_megatron_progress = True
        if callback.__class__ == TQDMProgressBar:
            contains_progress = True
    if not contains_megatron_progress and contains_progress:
        for i, callback in enumerate(callbacks):
            if isinstance(callback, TQDMProgressBar):
                if replace_progress_bar:
                    printer = ProgressPrinter(log_interval=progress_interval)
                    printer._trainer = trainer
                    if not trainer.is_global_zero:
                        printer.disable()
                    callbacks[i] = printer
                else:
                    callback.__class__ = MegatronProgressBar
                break


def ckpt_to_dir(filepath: Union[str, Path]) -> Path:
    """PTL considers checkpoints as .ckpt files.
    This method removes the extension and returns a path
    to be used as a directory for distributed checkpoints.
    """
    filepath = Path(filepath)

    if filepath.suffix == ".ckpt":
        return filepath.with_name(filepath.stem)

    return filepath


def create_checkpoint_io(**kwargs):
    checkpoint_io = MegatronCheckpointIO(**kwargs)
    if kwargs.get("async_save", False):
        checkpoint_io = AsyncFinalizableCheckpointIO(checkpoint_io)

    return checkpoint_io


def mcore_to_pyt_sharded_state_dict(
    checkpoint: Dict[str, List[torch.Tensor]],
    sharded_state_dict: Dict[str, Union[List[ShardedTensor], ShardedObject]],
    dtensor: bool = False,
    device_mesh: DeviceMesh = None,
) -> Dict[str, Union[TorchShardedTensor, io.BytesIO]]:
    def _mcore_to_pyt_dtensor(
        tens: List[torch.Tensor],
        sh_tens: List[ShardedTensor],
        device_mesh: DeviceMesh,
    ) -> DTensor:
        assert len(tens) == 1 and len(sh_tens) == 1

        dten = DTensor.from_local(
            tens[0],
            device_mesh,
            (
                Replicate(),
                Shard(dim=0),
            ),  # hardcoded for HSDP
        )
        return dten

    def _mcore_to_pyt_sharded_tensor(tens: List[torch.Tensor], sh_tens: List[ShardedTensor]) -> TorchShardedTensor:
        for ten, sh_ten in zip(tens, sh_tens):
            # remove prepend axes and put in loaded tensor
            sh_ten.global_shape = sh_ten.global_shape[sh_ten.prepend_axis_num :]
            sh_ten.global_offset = sh_ten.global_offset[sh_ten.prepend_axis_num :]
            sh_ten.axis_fragmentations = sh_ten.axis_fragmentations[sh_ten.prepend_axis_num :]
            sh_ten.prepend_axis_num = 0
            sh_ten.data = ten
            sh_ten.validate_metadata_integrity()

        return sharded_tensor_to_torch_sharded_tensor(sh_tens)

    def _convert(checkpoint, sharded_state_dict, k, device_mesh=None):
        assert k in sharded_state_dict, f"{k} not in sharded_state_dict"

        if isinstance(sharded_state_dict[k], Dict):
            for kk in sharded_state_dict[k]:
                _convert(checkpoint[k], sharded_state_dict[k], kk, device_mesh=device_mesh)
        elif isinstance(sharded_state_dict[k], ShardedObject):
            """Do nothing. checkpoint[k] contains loaded io.BytesIO already."""
        elif isinstance(sharded_state_dict[k], List):  # list of ShardedTensor
            if dtensor:
                checkpoint[k] = _mcore_to_pyt_dtensor(checkpoint[k], sharded_state_dict[k], device_mesh=device_mesh)
            else:
                checkpoint[k] = _mcore_to_pyt_sharded_tensor(checkpoint[k], sharded_state_dict[k])

    for k in checkpoint:
        _convert(checkpoint, sharded_state_dict, k, device_mesh=device_mesh)

    return checkpoint


def pyt_to_mcore_state_dict(
    state_dict: Dict[str, Any], prefix: str = "", device_mesh: DeviceMesh = None
) -> Dict[str, List[ShardedBase]]:
    def _dtensor_to_mcore_sharded_tensor(
        key: str,
        dten: DTensor,
        prepend_offsets: Iterable[Tuple[int, int, int]] = (),
        prefix: str = "",
        allow_shape_mismatch: bool = False,
        device_mesh: DeviceMesh = None,
    ) -> List[ShardedTensor]:
        prepend_axis_num = len(prepend_offsets)

        assert device_mesh is not None
        assert isinstance(dten, DTensor), dten

        ten = dten.to_local()
        global_shape = dten.shape

        rank_offsets = []
        replica_id = 0
        axis = list(range(len(global_shape)))
        axis_fragm = [global_shape[i] // ten.shape[i] for i in axis]

        for i, placement in enumerate(dten.placements):
            if isinstance(placement, Shard):
                ax = placement.dim
                rank_offsets.append((ax + prepend_axis_num, dten.device_mesh.get_local_rank(i), axis_fragm[ax]))
            elif placement.is_replicate():
                replica_id = device_mesh.get_local_rank(i)

        local_shard = ShardedTensor.from_rank_offsets(
            f"{prefix}{key}",
            ten,
            *prepend_offsets,  # prepend layer shards
            *rank_offsets,
            replica_id=replica_id,
            prepend_axis_num=prepend_axis_num,
            allow_shape_mismatch=allow_shape_mismatch,
        )
        return [local_shard]

    def _torch_to_mcore_sharded_tensor(
        key: str,
        sh_ten: TorchShardedTensor,
        prepend_offsets: Iterable[Tuple[int, int, int]] = (),
        prefix: str = "",
        allow_shape_mismatch: bool = False,
    ) -> List[ShardedTensor]:
        prepend_axis_num = len(prepend_offsets)

        assert isinstance(sh_ten, TorchShardedTensor), sh_ten
        sharded_meta = sh_ten.metadata()
        local_shards = sh_ten.local_shards()

        # DEBUG
        assert all([ls.metadata.placement == local_shards[0].metadata.placement for ls in local_shards]), [
            ls.meta.placement for ls in local_shards
        ]

        global_shape = sharded_meta.size

        axis = list(range(len(global_shape)))
        axis_fragm = [global_shape[i] // local_shards[0].metadata.shard_sizes[i] for i in axis]
        rank_offsets = []

        for i in range(len(local_shards)):
            local_shard = local_shards[i]
            ten, meta = local_shard.tensor, local_shard.metadata

            for j in range(len(axis)):
                axis_rank_offset = meta.shard_offsets[j] // meta.shard_sizes[j]
                rank_offsets.append((axis[j] + prepend_axis_num, axis_rank_offset, axis_fragm[j]))

            local_shards[i] = ShardedTensor.from_rank_offsets(
                f"{prefix}{key}",
                ten,
                *prepend_offsets,  # prepend layer shards
                *rank_offsets,
                replica_id=0,
                prepend_axis_num=prepend_axis_num,
                allow_shape_mismatch=allow_shape_mismatch,
            )

        return local_shards

    def _torch_to_mcore_sharded_object(
        key: str,
        obj: io.BytesIO,
        sharded_offsets: Iterable[Tuple[int, int, int]] = (),
        prefix: str = "",
    ) -> ShardedObject:
        replica_id = (
            0,
            0,
            parallel_state.get_data_parallel_rank(with_context_parallel=True),
        )

        return ShardedObject(f"{prefix}{key}", obj, *_get_extra_state_offsets(sharded_offsets), replica_id)

    def _convert(state_dict, k, sh_key, v, prepend_offsets, prefix="", allow_shape_mismatch=False, device_mesh=None):
        if isinstance(v, Dict):
            for kk, vv in v.items():
                _convert(
                    v,
                    kk,
                    sh_key,
                    vv,
                    prepend_offsets,
                    prefix=f"{prefix}{kk}.",
                    allow_shape_mismatch=allow_shape_mismatch,
                    device_mesh=device_mesh,
                )
        elif isinstance(v, DTensor):
            state_dict[k] = _dtensor_to_mcore_sharded_tensor(
                sh_key,
                v,
                prepend_offsets,
                prefix=prefix,
                allow_shape_mismatch=allow_shape_mismatch,
                device_mesh=device_mesh,
            )
        elif isinstance(v, TorchShardedTensor):
            state_dict[k] = _torch_to_mcore_sharded_tensor(
                sh_key, v, prepend_offsets, prefix=prefix, allow_shape_mismatch=allow_shape_mismatch
            )
        elif isinstance(v, io.BytesIO):
            state_dict[k] = _torch_to_mcore_sharded_object(sh_key, v, prepend_offsets, prefix)

    num_layers = 0
    for k in state_dict:
        if k.startswith("module.decoder.layers."):
            num_layers = max(num_layers, int(k.split(".")[3]) + 1)

    for k, v in state_dict.items():
        prepend_offsets = []
        sh_key = k
        allow_shape_mismatch = k.endswith(".word_embeddings.weight")  # vocab size can be different
        if k.startswith("module.decoder.layers."):
            sh_key = k.split(".")
            global_layer_offset = int(sh_key.pop(3))
            sh_key = ".".join(sh_key)
            prepend_offsets.append((0, global_layer_offset, num_layers))

        _convert(state_dict, k, sh_key, v, prepend_offsets, prefix, allow_shape_mismatch, device_mesh)

    return state_dict
