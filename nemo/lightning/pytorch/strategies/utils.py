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

import contextlib
import io
import signal
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Set, Tuple, Union, cast

import lightning.pytorch as pl
import torch
from lightning.fabric.plugins import ClusterEnvironment
from lightning.pytorch.callbacks import TQDMProgressBar

try:
    from megatron.core import parallel_state
    from megatron.core.dist_checkpointing.mapping import ShardedBase, ShardedObject, ShardedTensor
    from megatron.core.dist_checkpointing.strategies.torch import sharded_tensor_to_torch_sharded_tensor
    from megatron.core.transformer.utils import _get_extra_state_offsets

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    ShardedObject = object
    ShardedBase = object
    ShardedTensor = object
    HAVE_MEGATRON_CORE = False


from torch import Tensor, nn
from torch.distributed._sharded_tensor import ShardedTensor as TorchShardedTensor
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh, _mesh_resources
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, SequenceParallel, parallelize_module

from nemo.lightning import _strategy_lib
from nemo.lightning.pytorch.callbacks import MegatronProgressBar, ProgressPrinter
from nemo.utils.callbacks.dist_ckpt_io import AsyncFinalizableCheckpointIO
from nemo.utils.import_utils import safe_import_from

MixedPrecisionPolicy, HAS_MIXED_PRECISION_POLICY = safe_import_from(
    "torch.distributed.fsdp", "MixedPrecisionPolicy", fallback_module="torch.distributed._composable.fsdp"
)
fully_shard, HAS_FULLY_SHARD = safe_import_from(
    "torch.distributed.fsdp", "fully_shard", fallback_module="torch.distributed._composable.fsdp"
)
CPUOffloadPolicy, HAS_CPU_OFFLOAD_POLICY = safe_import_from(
    "torch.distributed.fsdp", "CPUOffloadPolicy", fallback_module="torch.distributed._composable.fsdp"
)


@dataclass(kw_only=True)
class RestoreConfig:
    """
    Configuration for restoring model state from a checkpoint.

    Attributes:
        path (str): Path to the checkpoint directory.
        load_model_state (bool): Whether to load model weights.
        load_optim_state (bool): Whether to load optimizer state.
        load_artifacts (bool): Whether to load additional artifacts (e.g., tokenizer).
    """

    path: str
    load_model_state: bool = True
    load_optim_state: bool = False
    # eg tokenizer, etc.
    load_artifacts: bool = True


def setup_parallel_ranks(strategy: pl.strategies.Strategy):
    """
    Sets up parallel ranks for distributed training.

    Args:
        strategy (pl.strategies.Strategy): The Lightning strategy being used for training.
    """
    from megatron.core.model_parallel_config import ModelParallelConfig

    env = cast(ClusterEnvironment, strategy.cluster_environment)
    parallelism = getattr(strategy, "parallelism", ModelParallelConfig())
    _strategy_lib.init_parallel_ranks(env.world_size(), env.global_rank(), env.local_rank(), parallelism)


def init_model_parallel(pl_module: pl.LightningModule):
    """
    Initializes model parallelism for distributed training.

    Args:
        pl_module (pl.LightningModule): The PyTorch Lightning module.
    """
    from megatron.core import parallel_state

    from nemo.utils import AppState

    if not parallel_state.model_parallel_is_initialized():
        app_state = AppState()

        if app_state.model_parallel_size is not None:
            _strategy_lib.init_model_parallel(pl_module)


def setup_data_sampler(trainer: pl.Trainer):
    """
    Configures the data sampler for distributed training.

    Args:
        trainer (pl.Trainer): The PyTorch Lightning trainer instance.
    """
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
    """
    Fixes or replaces the progress bar callback in the PyTorch Lightning trainer.

    Args:
        trainer (pl.Trainer): The PyTorch Lightning trainer instance.
        replace_progress_bar (bool): Whether to replace the default progress bar.
        progress_interval (int): Interval at which to log progress.
    """
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

    Converts a checkpoint file path to a directory path by removing the `.ckpt` extension.

    Args:
        filepath (Union[str, Path]): The checkpoint file path.

    Returns:
        Path: The directory path where the checkpoint will be stored.

    """
    filepath = Path(filepath)

    if filepath.suffix == ".ckpt":
        return filepath.with_name(filepath.stem)

    return filepath


def create_checkpoint_io(wrapping_ckpt_io=None, **kwargs):
    """
    Creates a checkpoint IO handler for saving/loading checkpoints.

    Args:
        wrapping_ckpt_io: An optional wrapper for checkpoint IO.
        **kwargs: Additional arguments to configure checkpoint IO.

    Returns:
        Checkpoint IO handler instance.
    """
    if kwargs.get("model_library", None) == "huggingface":
        from nemo.lightning.io.hf import HFCheckpointIO

        checkpoint_io = HFCheckpointIO(adapter_only=kwargs.get("lora", False))
    else:
        from nemo.lightning.io.pl import MegatronCheckpointIO

        checkpoint_io = MegatronCheckpointIO(**kwargs)

    if wrapping_ckpt_io:
        checkpoint_io = wrapping_ckpt_io(checkpoint_io)
    if kwargs.get("async_save", False):
        checkpoint_io = AsyncFinalizableCheckpointIO(checkpoint_io)

    return checkpoint_io


def mcore_to_pyt_sharded_state_dict(
    checkpoint: Dict[str, List[torch.Tensor]],
    sharded_state_dict: Dict[str, Union[List[ShardedTensor], ShardedObject]],
    dtensor: bool = False,
    device_mesh: DeviceMesh = None,
) -> Dict[str, Union[TorchShardedTensor, io.BytesIO]]:
    """
    Converts a Megatron-Core sharded state dictionary into a PyTorch-compatible format.

    Args:
        checkpoint (Dict[str, List[torch.Tensor]]):
            The Megatron-Core checkpoint containing a list of tensors for each key.
        sharded_state_dict (Dict[str, Union[List[ShardedTensor], ShardedObject]]):
            The corresponding PyTorch sharded state dictionary.
        dtensor (bool, optional):
            Whether to use DTensor for the conversion. Defaults to False.
        device_mesh (DeviceMesh, optional):
            The device mesh configuration for distributed tensors.

    Returns:
        Dict[str, Union[TorchShardedTensor, io.BytesIO]]:
            A PyTorch-compatible state dictionary with properly formatted sharded tensors.
    """

    def _mcore_to_pyt_dtensor(
        tens: List[torch.Tensor],
        sh_tens: List[ShardedTensor],
        device_mesh: DeviceMesh,
    ) -> DTensor:
        """Converts a Megatron-Core tensor into a PyTorch DTensor."""
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
        """Converts a Megatron-Core tensor into a PyTorch sharded tensor."""
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
        """Recursively converts checkpoint tensors into PyTorch-compatible formats."""
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
    """
    Converts a PyTorch state dictionary into a Megatron-Core compatible format.

    Args:
        state_dict (Dict[str, Any]):
            The PyTorch state dictionary.
        prefix (str, optional):
            A prefix to prepend to all keys. Defaults to "".
        device_mesh (DeviceMesh, optional):
            The device mesh configuration for distributed tensors.

    Returns:
        Dict[str, List[ShardedBase]]:
            A Megatron-Core formatted state dictionary with properly sharded tensors.
    """

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
        """mcore helper"""
        replica_id = (
            0,
            0,
            parallel_state.get_data_parallel_rank(with_context_parallel=True),
        )

        return ShardedObject(f"{prefix}{key}", obj, *_get_extra_state_offsets(sharded_offsets), replica_id)

    def _convert(state_dict, k, sh_key, v, prepend_offsets, prefix="", allow_shape_mismatch=False, device_mesh=None):
        """mcore helper"""
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


# Taken and modified from torchtitan
# https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py
def fsdp2_strategy_parallelize(
    model,
    device_mesh: DeviceMesh = None,
    mp_policy: MixedPrecisionPolicy = None,
    use_hf_tp_plan: bool = False,
    tp_shard_plan: Optional[Dict[str, Union[RowwiseParallel, ColwiseParallel, SequenceParallel]]] = None,
    offload_policy: 'CPUOffloadPolicy' = None,
):
    """Apply parallelisms and activation checkpointing to the model.

    Args:
        model: The model to be parallelized.
        device_mesh (DeviceMesh): The device mesh for distributed training.
        mp_policy (MixedPrecisionPolicy): Mixed precision policy for model parallelism.
        tp_shard_plan (Optional[Dict[str, Union[RowwiseParallel, ColwiseParallel, SequenceParallel]]]):
            A tensor parallel sharding plan. The keys should be the module names and the values should be the
            corresponding parallel styles (e.g., RowwiseParallel, ColwiseParallel, SequenceParallel).
        offload_policy (CPUOffloadPolicy): The offload policy for FSDP. If None, it will use the default policy.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    NOTE: Currently, the user is required to manually handle precision settings such as the `mp_policy` here
    because the model parallel strategy does not respect all settings of `Fabric(precision=...)` at the moment.
    NOTE: Currently, the user should make sure that custom_tp_plan is compatible with the model architecture.
    """

    if not mp_policy:
        assert HAS_MIXED_PRECISION_POLICY is not None, "Expected to have MixedPrecisionPolicy"
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    def parallelize_helper(module, mesh, mp_policy):
        if isinstance(module, nn.ModuleList):
            for layer_id, transformer_block in enumerate(module):
                # Apply activation checkpointing
                # transformer_block = checkpoint_wrapper(transformer_block)
                # As an optimization, do not reshard after forward for the last
                # transformer block since FSDP would prefetch it immediately
                reshard_after_forward = int(layer_id) < len(module) - 1
                fully_shard(
                    transformer_block,
                    mesh=mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=reshard_after_forward,
                    offload_policy=offload_policy,
                )
                module[layer_id] = transformer_block
        else:
            for name, sub_module in module.named_children():
                parallelize_helper(sub_module, mesh, mp_policy)

    # Set FSDP sharding mesh to context parallel mesh if CP > 1, else default to the data parallel mesh.
    dp_mesh = device_mesh[
        ("dp_cp" if "dp_cp" in _mesh_resources.root_to_flatten_mapping.get(device_mesh, {}) else "data_parallel")
    ]
    tp_mesh = device_mesh["tensor_parallel"]

    if dp_mesh.size() > 1:
        assert dp_mesh.ndim == 1, "Hybrid-sharding not supported"

    # TP sharding
    if tp_mesh.size() > 1:
        if tp_shard_plan is None and use_hf_tp_plan:
            tp_shard_plan = get_hf_tp_shard_plan(model)
        parallelize_module(model, tp_mesh, tp_shard_plan)

    # FSDP sharding
    assert dp_mesh.ndim == 1, "Hybrid-sharding not supported"
    assert HAS_FULLY_SHARD is not None, "Expected to have fully_shard"

    # Find transformer layers and apply parallelisms
    parallelize_helper(model, dp_mesh, mp_policy)

    # reshard_after_forward=True based on
    # https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py#L359
    model = fully_shard(
        model, mesh=dp_mesh, mp_policy=mp_policy, reshard_after_forward=True, offload_policy=offload_policy
    )

    return model


def get_hf_tp_shard_plan(model):
    """
    Get the tensor parallel sharding plan from the model.
    """
    hf_tp_shard_plan = {}
    if hasattr(model, '_tp_plan') and model._tp_plan is not None:
        hf_tp_shard_plan.update(model._tp_plan)
    if hasattr(model.model, '_tp_plan') and model.model._tp_plan is not None:
        hf_tp_shard_plan.update({f"model.{k}": v for k, v in model.model._tp_plan.items()})

    hf_tp_shard_plan = {k: translate_to_torch_parallel_style(v) for k, v in hf_tp_shard_plan.items()}
    return hf_tp_shard_plan


@lru_cache
def translate_to_torch_parallel_style(style: str):
    """
    In model configurations, we use a neutral type (string) to specify parallel
    styles, here we translate them into torch.distributed tensor-parallel
    types.
    """
    if not isinstance(style, str):
        raise ValueError(f"Unsupported parallel style type {type(style)}, expected str")

    if style == "colwise":
        return ColwiseParallel()
    elif style == "rowwise":
        return RowwiseParallel()
    elif style == "colwise_rep":
        return ColwiseParallel(output_layouts=Replicate())
    elif style == "rowwise_rep":
        return RowwiseParallel(input_layouts=Replicate())
    elif style == "sequence_parallel":
        return SequenceParallel()
    else:
        raise ValueError(f"Unsupported parallel style value: {style}")


def to_cpu(v):
    """
    Move a tensor or distributed tensor to the CPU.

    This function takes an input tensor, which can be either a `DTensor` (distributed tensor)
    or a standard `Tensor`, and ensures that it is moved to the CPU.

    Args:
        v (DTensor | Tensor | any): The input value, which can be a `DTensor`, `Tensor`, or
                                    any other object. If `DTensor`, it checks the device and
                                    moves the tensor accordingly.

    Returns:
        Tensor | any: The corresponding CPU tensor if `v` is a `DTensor` or `Tensor`,
                    otherwise returns `v` unchanged.

    Raises:
        ValueError: If `v` is a `DTensor` but its device is neither 'cuda' nor 'cpu'.

    Example:
        >>> t = torch.tensor([1, 2, 3], device='cuda')
        >>> to_cpu(t)  # Moves tensor to CPU
        tensor([1, 2, 3])

        >>> dt = DTensor(torch.tensor([4, 5, 6], device='cuda'))
        >>> to_cpu(dt)  # Moves DTensor to CPU
        tensor([4, 5, 6])
    """
    if isinstance(v, DTensor):
        if v.device.type == "cuda":
            return v.full_tensor().cpu()
        elif v.device.type == "cpu":
            return v._local_tensor
        else:
            raise ValueError("Unknown device " + str(v.device))
    elif isinstance(v, Tensor):
        return v.cpu()
    else:
        return v


def _destroy_dist_connection() -> None:
    """Destroy process group."""
    # Don't allow Ctrl+C to interrupt this handler
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    signal.signal(signal.SIGINT, signal.SIG_DFL)


# based on https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/utils.py#L113
def create_context_parallel_ctx(
    cp_mesh: DeviceMesh,
    cp_buffers: List[torch.Tensor],
    cp_seq_dims: List[int],
    cp_no_restore_buffers: Set[torch.Tensor],
    cp_rotate_method: str,
):
    """
    Create a context parallel context.

    Args:
        cp_mesh (DeviceMesh): The device mesh for context parallel.
        cp_buffers (List[torch.Tensor]): The buffers for context parallel.
        cp_seq_dims (List[int]): The sequence dimensions for context parallel.
        cp_no_restore_buffers (Set[torch.Tensor]): The no restore buffers for context parallel.
        cp_rotate_method (str): The rotation method for context parallel, such as "allgather" or "addtoall".
    """
    from torch.distributed.tensor.experimental import context_parallel

    # TODO: uncomment this when torch.distributed.tensor.experimental._attention.set_rotate_method is available
    # from torch.distributed.tensor.experimental._attention import set_rotate_method
    # set_rotate_method(cp_rotate_method)
    return context_parallel(
        cp_mesh,
        buffers=cp_buffers,
        buffer_seq_dims=cp_seq_dims,
        no_restore_buffers=cp_no_restore_buffers,
    )


# based on https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/utils.py#L138
def get_train_context(enable_loss_parallel: bool, enable_compiled_autograd: bool):
    """
    Create a train context.

    Args:
        enable_loss_parallel (bool): Whether to enable loss parallelism.
        enable_compiled_autograd (bool): Whether to enable compiled autograd.
    """

    @contextlib.contextmanager
    def context(cp_context: Optional[Generator[None, None, None]] = None):
        with contextlib.ExitStack() as stack:
            if enable_loss_parallel:
                stack.enter_context(torch.distributed.tensor.parallel.loss_parallel())

            if enable_compiled_autograd:
                stack.enter_context(torch._dynamo.utils.maybe_enable_compiled_autograd(True))

            if cp_context is not None:
                from torch.nn.attention import SDPBackend, sdpa_kernel

                # currently we only support these two SDP backends.
                # TODO (xilunwu): support cuDNN backend
                stack.enter_context(sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]))
                stack.enter_context(cp_context)

            yield

    return context
