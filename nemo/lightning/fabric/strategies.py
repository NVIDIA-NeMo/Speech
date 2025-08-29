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

from contextlib import ExitStack, contextmanager, nullcontext
from datetime import timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ContextManager,
    Dict,
    Generator,
    Iterator,
    List,
    Literal,
    Optional,
    Union,
)

import torch
from lightning.fabric.accelerators import CPUAccelerator
from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning.fabric.plugins.precision import Precision
from lightning.fabric.strategies import DDPStrategy
from lightning.fabric.strategies.strategy import _validate_keys_for_strict_loading
from lightning.fabric.utilities.types import _PATH, _Stateful
from lightning.pytorch import LightningDataModule
from lightning.pytorch.loops.fetchers import _DataFetcher
from lightning.pytorch.plugins.io.wrapper import _WrappingCheckpointIO
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from torch import Tensor, nn
from torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks import noop_hook
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing_extensions import override

from nemo.lightning import _strategy_lib
from nemo.lightning.fabric.conversion import to_fabric
from nemo.lightning.io.pl import MegatronCheckpointIO, ckpt_to_weights_subdir
from nemo.lightning.megatron_parallel import CallbackConnector, MegatronParallel
from nemo.lightning.pytorch.strategies import MegatronStrategy
from nemo.utils.import_utils import safe_import
from nemo.utils.model_utils import unwrap_model

mto, HAVE_MODELOPT = safe_import("modelopt.torch.opt")

if TYPE_CHECKING:
    from nemo.lightning.pytorch.plugins.data_sampler import DataSampler


DDPLiteral = Literal["megatron", "pytorch"]


class FabricMegatronStrategy(DDPStrategy):
    """
    Fabric strategy for Megatron.
    """

    def __init__(
        self,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        virtual_pipeline_model_parallel_size: Optional[int] = None,
        pipeline_model_parallel_comm_backend: str = None,
        microbatch_group_size_per_vp_stage: Optional[int] = None,
        context_parallel_size: int = 1,
        sequence_parallel: bool = False,
        expert_model_parallel_size: int = 1,
        moe_extended_tp: bool = False,
        expert_tensor_parallel_size: int = None,
        encoder_tensor_model_parallel_size: Optional[int] = 0,
        encoder_pipeline_model_parallel_size: Optional[int] = 0,
        data_sampler: Optional["DataSampler"] = None,
        accelerator: Optional[Accelerator] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision: Optional[Precision] = None,
        megatron_callbacks: Optional[CallbackConnector] = None,
        ddp: Union[DDPLiteral, DistributedDataParallelConfig] = "megatron",
        process_group_backend: Optional[str] = None,
        timeout: Optional[timedelta] = default_pg_timeout,
        start_method: Literal["popen", "spawn", "fork", "forkserver"] = "popen",
        no_ddp_communication_hook: bool = True,
        output_data_idx: bool = False,
        pipeline_dtype: Optional[torch.dtype] = None,
        init_model_parallel: bool = True,
        use_tp_pp_dp_mapping: bool = False,
        num_distributed_optimizer_instances: int = 1,
        nccl_communicator_config_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision=precision,
            process_group_backend=process_group_backend,
            timeout=timeout,
            start_method=start_method,
            **kwargs,
        )
        self.megatron_callbacks = CallbackConnector()
        self.data_sampler: Optional['DataSampler'] = data_sampler
        self.tensor_model_parallel_size = tensor_model_parallel_size
        self.pipeline_model_parallel_size = pipeline_model_parallel_size
        self.pipeline_model_parallel_comm_backend = pipeline_model_parallel_comm_backend
        self.microbatch_group_size_per_vp_stage = (
            microbatch_group_size_per_vp_stage
            if microbatch_group_size_per_vp_stage is not None
            else pipeline_model_parallel_size
        )
        self.context_parallel_size = context_parallel_size
        self.expert_model_parallel_size = expert_model_parallel_size
        self.expert_tensor_parallel_size = expert_tensor_parallel_size
        self.moe_extended_tp = moe_extended_tp
        self.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
        self.sequence_parallel = sequence_parallel
        self.encoder_tensor_model_parallel_size = encoder_tensor_model_parallel_size
        self.encoder_pipeline_model_parallel_size = encoder_pipeline_model_parallel_size
        self.pipeline_dtype = pipeline_dtype
        self._init_model_parallel = init_model_parallel
        self.use_tp_pp_dp_mapping = use_tp_pp_dp_mapping
        self.num_distributed_optimizer_instances = num_distributed_optimizer_instances
        self.nccl_communicator_config_path = nccl_communicator_config_path
        self.no_ddp_communication_hook = no_ddp_communication_hook
        self.megatron_callbacks = CallbackConnector()
        if megatron_callbacks:
            self.megatron_callbacks.add(megatron_callbacks)
        self.output_data_idx = output_data_idx
        self.data_sampler: Optional["DataSampler"] = data_sampler

        # used in NVIDIA NGC PyTorch containers
        _strategy_lib.enable_nvidia_optimizations()

        self._ddp = ddp
        if ddp == "megatron":
            self.ddp_config = DistributedDataParallelConfig()
        elif isinstance(ddp, DistributedDataParallelConfig):
            self.ddp_config = ddp
        elif ddp == "pytorch":
            self.ddp_config = None
            self.no_ddp_communication_hook = False
        else:
            raise ValueError(f"Invalid DDP type: {ddp}")

    @override
    def _setup_distributed(self) -> None:
        self._set_world_ranks()

        assert self.cluster_environment is not None
        _strategy_lib.init_parallel_ranks(
            world_size=self.cluster_environment.world_size(),
            global_rank=self.cluster_environment.global_rank(),
            local_rank=self.cluster_environment.local_rank(),
            parallel_config=self.parallelism,
        )

        super()._setup_distributed()
        torch.cuda.set_device(self.cluster_environment.local_rank())

        # TODO: Fix this:
        # if self.data_config is not None:
        #     _strategy_lib.initialize_data(self.cluster_environment.global_rank(), self.data_config)
        _strategy_lib.init_model_parallel()

    def process_datamodule(self, datamodule: LightningDataModule) -> LightningDataModule:
        """
        Process the datamodule.
        """
        datamodule.setup()

        if not self.data_sampler and hasattr(datamodule, "data_sampler"):
            self.data_sampler = datamodule.data_sampler

        if self.data_sampler:
            self.data_sampler.setup(self.cluster_environment.global_rank())

        return datamodule

    @override
    def process_dataloader(self, dataloader: DataLoader) -> Iterator:
        """
        Process the dataloader. Returns an iterator.
        """
        if self.data_sampler:
            dataloader = self.data_sampler.transform_dataloader(dataloader)

        # Code taken from:
        # https://github.com/Lightning-AI/pytorch-lightning
        # /blob/6cbe9ceb560d798892bdae9186291acf9bf5d2e3/src/lightning/pytorch/loops/fit_loop.py
        # L258-L260
        output = _MegatronDataLoaderIterDataFetcher(output_data_idx=self.output_data_idx)
        output.setup(CombinedLoader(dataloader, "max_size_cycle"))
        iter(output)

        return output

    def setup_megatron_optimizer(
        self,
        model: MegatronParallel,
        optimizer_config: OptimizerConfig,
        no_weight_decay_cond: Optional[Callable] = None,
        scale_lr_cond: Optional[Callable] = None,
        lr_mult: float = 1.0,
    ) -> Optimizer:
        """
        Setup the Megatron optimizer.
        """
        if hasattr(self.precision, "convert_config"):
            optimizer_config = self.precision.convert_config(optimizer_config)

        assert optimizer_config.lr is not None, "Learning rate must be set in optimizer config"

        return _strategy_lib.setup_megatron_optimizer(
            model,
            optimizer_config,
            no_weight_decay_cond=no_weight_decay_cond,
            scale_lr_cond=scale_lr_cond,
            lr_mult=lr_mult,
        )

    @override
    def setup_optimizer(self, optimizer: Optimizer) -> Optimizer:
        """Pass the optimizer to the precision-plugin if needed & add it as callback."""
        if hasattr(self._precision, "setup_optimizer"):
            optimizer = self._precision.setup_optimizer(optimizer)

        self.megatron_callbacks.add(optimizer)

        return optimizer

    @override
    def setup_module(self, module: Module) -> MegatronParallel:
        """
        Setup the torch module. Returns a MegatronParallel object.
        """
        from megatron.core.utils import get_model_config

        _strategy_lib.set_model_parallel_attributes(module, self.parallelism)

        convert_module_fn = None
        if hasattr(self.precision, "convert_module"):
            convert_module_fn = self.precision.convert_module

        if hasattr(self.precision, "convert_config"):
            self.precision.convert_config(get_model_config(module))
            if self.ddp_config:
                self.precision.convert_config(self.ddp_config)

        # Call configure_model if it's overridden (relevant for LightningModules with lazy initialization)
        if hasattr(module, "configure_model"):
            module.configure_model()

        megatron_parallel = MegatronParallel(
            module,
            precision_plugin=self.precision,
            vp_size=self.virtual_pipeline_model_parallel_size,
            cpu=isinstance(self.accelerator, CPUAccelerator),
            ddp_config=self.ddp_config,
            convert_module_fn=convert_module_fn,
        )

        if self._init_model_parallel:
            megatron_parallel.init_model_parallel()

        if self.data_sampler:
            megatron_parallel.callbacks.add(self.data_sampler)

        if not self.ddp_config:
            from megatron.core import mpu

            from nemo.utils import AppState

            app_state = AppState()

            if app_state.model_parallel_size is not None:
                self._ddp_kwargs["process_group"] = mpu.get_data_parallel_group()

            dist_data_parallel = super().setup_module(megatron_parallel)
            if self.no_ddp_communication_hook:
                # When using custom gradient accumulation and allreduce, disable
                # DDP communication hook that works on the gradient bucket.
                # Instead, use the custom gradient function and communication hook,
                # which is defined in the master optimizer wrapper.
                dist_data_parallel.require_backward_grad_sync = False
                dist_data_parallel.register_comm_hook(None, noop_hook)

            return dist_data_parallel

        return megatron_parallel

    def module_init_context(self, empty_init: Optional[bool] = None) -> ContextManager:
        """
        Get the context manager used for initializing the module.
        """
        precision_init_ctx = self.precision.module_init_context()
        module_sharded_ctx = self.megatron_context()
        stack = ExitStack()
        if empty_init:
            # Materialization happens in `setup`. When modules get wrapped by FSDP, the sequence of operations is:
            # 1) materialize module 2) call `reset_parameters()` 3) shard the module.
            # These operations are applied to each submodule 'bottom up' in the module hierarchy.
            stack.enter_context(torch.device("meta"))
        stack.enter_context(precision_init_ctx)
        stack.enter_context(module_sharded_ctx)

        return stack

    def module_to_device(self, module: nn.Module) -> None:
        """
        Move the module to the device.
        """
        pass

    @override
    def save_checkpoint(
        self,
        path: _PATH,
        state: Dict[str, Union[Module, Optimizer, Any]],
        storage_options: Optional[Any] = None,
        filter_dict: Optional[Dict[str, Callable[[str, Any], bool]]] = None,
    ) -> None:
        """Save model, optimizer, and other state as a checkpoint file.

        Args:
            path: A path to where the file(s) should be saved
            state: A dictionary with contents to be saved. If the dict contains modules or optimizers, their
                state-dict will be retrieved and converted automatically.
            storage_options: Additional options for the ``CheckpointIO`` plugin
            filter: An optional dictionary containing filter callables that return a boolean indicating whether the
                given item should be saved (``True``) or filtered out (``False``). Each filter key should match a
                state key, where its filter will be applied to the ``state_dict`` generated.

        """
        if not storage_options:
            storage_options = {}
        storage_options['content_metadata'] = self.sharded_state_dict_metadata
        state = self._convert_stateful_objects_in_state(state, filter=(filter_dict or {}))
        self.checkpoint_io.save_checkpoint(checkpoint=state, path=path, storage_options=storage_options)

    def load_checkpoint(
        self,
        path: _PATH,
        state: Optional[Union[Module, Optimizer, Dict[str, Union[Module, Optimizer, Any]]]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load the checkpoint.
        """
        if isinstance(state, Optimizer):
            raise NotImplementedError("Optimizer loading is not supported, pass it as a dict including the model")
        unwrapped_model = unwrap_model(state["state_dict"])

        from nemo.collections.vlm.llama4.model.base import Llama4OmniBaseModel

        if HAVE_MODELOPT and isinstance(unwrapped_model, Llama4OmniBaseModel):
            # If present, first restore and modify the model according to the ModelOpt state.
            # Avoid quantizers being added to teacher model if model is a distillation model.
            core_model = unwrapped_model.language_model
            with core_model.hide_teacher_model() if hasattr(core_model, "hide_teacher_model") else nullcontext():
                mto.plugins.restore_sharded_modelopt_state(
                    [core_model], ckpt_to_weights_subdir(path, is_saving=False), prefix="module.language_model."
                )
            if mto.ModeloptStateManager.is_converted(core_model):
                print("Restored Model-Optimizer state from checkpoint.")
        torch.cuda.empty_cache()

        # After dist_checkpointing.load, sharded tensors will be replaced with tensors
        sharded_sd_metadata = self.unwrapped_checkpoint_io.load_content_metadata(path)
        sharded_state_dict = {}
        if isinstance(state, Module):
            sharded_state_dict["state_dict"] = state.sharded_state_dict(metadata=sharded_sd_metadata)
        elif strict:
            if isinstance(state['state_dict'], DistributedDataParallel):
                state["state_dict"] = state['state_dict'].module
            sharded_state_dict["state_dict"] = state["state_dict"].sharded_state_dict(metadata=sharded_sd_metadata)
            if "optimizer" in state:
                sharded_state_dict["optimizer"] = _strategy_lib.optimizer_sharded_state_dict(
                    state["state_dict"],
                    state["optimizer"],
                    is_loading=True,
                    metadata=sharded_sd_metadata,
                )
        else:
            for obj in state.items():
                if isinstance(obj, Module):
                    sharded_state_dict["state_dict"] = obj.sharded_state_dict(metadata=sharded_sd_metadata)
                elif isinstance(obj, Optimizer):
                    sharded_state_dict["optimizer"] = _strategy_lib.optimizer_sharded_state_dict(
                        obj, is_loading=True, metadata=sharded_sd_metadata
                    )

        checkpoint = self.checkpoint_io.load_checkpoint(path, sharded_state_dict=sharded_state_dict)

        if isinstance(state, Module):
            self.load_module_state_dict(module=state, state_dict=checkpoint, strict=strict)
            return {}

        _validate_keys_for_strict_loading(state.keys(), checkpoint.keys(), strict=strict)
        for name, obj in state.copy().items():
            if name not in checkpoint:
                continue
            if isinstance(obj, _Stateful):
                if isinstance(obj, Module):
                    self.load_module_state_dict(module=obj, state_dict=checkpoint.pop(name), strict=strict)
                else:
                    obj.load_state_dict(checkpoint.pop(name))
            else:
                state[name] = checkpoint.pop(name)

        return checkpoint

    @override
    def load_module_state_dict(
        self, module: Module, state_dict: Dict[str, Union[Any, Tensor]], strict: bool = True
    ) -> None:
        """
        Load the module state dict.
        """
        _strategy_lib.load_model_state_dict(module, state_dict, strict=strict)

    @property
    def sharded_state_dict_metadata(self):
        """Metadata used for sharded_state_dict generation during checkpoint save."""
        metadata = {}
        if isinstance(self.ddp_config, DistributedDataParallelConfig) and self.ddp_config.use_distributed_optimizer:
            metadata["distrib_optim_sharding_type"] = "fully_sharded_model_space"
        return metadata

    @contextmanager
    def megatron_context(self) -> Generator[None, None, None]:
        """
        Context manager for Megatron.
        """
        from megatron.core.extensions import transformer_engine as _te

        original = _te._get_extra_te_kwargs  # noqa: SLF001

        def _get_extra_te_kwargs_meta(c):
            """Forces device to meta"""
            kwargs = original(c)
            kwargs['device'] = 'meta'
            return kwargs

        _te._get_extra_te_kwargs = _get_extra_te_kwargs_meta  # noqa: SLF001

        _orig_perform_initialization = self.parallelism.perform_initialization
        _orig_use_cpu_initialization = self.parallelism.use_cpu_initialization

        self.parallelism.perform_initialization = False
        self.parallelism.use_cpu_initialization = True

        yield

        _te._get_extra_te_kwargs = original  # noqa: SLF001
        self.parallelism.perform_initialization = _orig_perform_initialization
        self.parallelism.use_cpu_initialization = _orig_use_cpu_initialization

    @property
    @override
    def checkpoint_io(self) -> CheckpointIO:
        """
        Get the checkpoint IO.
        """
        if self._checkpoint_io is None:
            self._checkpoint_io = MegatronCheckpointIO()
        elif isinstance(self._checkpoint_io, _WrappingCheckpointIO):
            self._checkpoint_io.checkpoint_io = MegatronCheckpointIO()

        return self._checkpoint_io

    @property
    def unwrapped_checkpoint_io(self) -> CheckpointIO:
        """Unwraps `checkpoint_io` from all wrappers."""
        checkpoint_io = self.checkpoint_io
        while isinstance(checkpoint_io, _WrappingCheckpointIO):
            checkpoint_io = checkpoint_io.checkpoint_io
        return checkpoint_io

    @property
    def parallelism(self):
        """
        Get the parallelism config.
        """
        from nemo.lightning.pytorch.strategies.megatron_strategy import ParallelismConfig

        return ParallelismConfig(
            tensor_model_parallel_size=self.tensor_model_parallel_size,
            pipeline_model_parallel_size=self.pipeline_model_parallel_size,
            pipeline_model_parallel_comm_backend=self.pipeline_model_parallel_comm_backend,
            virtual_pipeline_model_parallel_size=self.virtual_pipeline_model_parallel_size,
            microbatch_group_size_per_vp_stage=self.microbatch_group_size_per_vp_stage,
            context_parallel_size=self.context_parallel_size,
            sequence_parallel=self.sequence_parallel,
            expert_model_parallel_size=self.expert_model_parallel_size,
            expert_tensor_parallel_size=self.expert_tensor_parallel_size,
            moe_extended_tp=self.moe_extended_tp,
            encoder_tensor_model_parallel_size=self.encoder_tensor_model_parallel_size,
            encoder_pipeline_model_parallel_size=self.encoder_pipeline_model_parallel_size,
            pipeline_dtype=self.pipeline_dtype,
            use_tp_pp_dp_mapping=self.use_tp_pp_dp_mapping,
            num_distributed_optimizer_instances=self.num_distributed_optimizer_instances,
            nccl_communicator_config_path=self.nccl_communicator_config_path,
        )


# TODO: Fix this
class _MegatronDataLoaderIterDataFetcher(_DataFetcher):
    def __init__(self, *args: Any, output_data_idx: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.output_data_idx = output_data_idx
        self._batch: Any = None
        self._batch_idx: int = 0
        self._dataloader_idx: int = 0

    def __iter__(self) -> "_MegatronDataLoaderIterDataFetcher":
        super().__iter__()
        self.iterator_wrapper = iter(_DataFetcherWrapper(self, output_data_idx=self.output_data_idx))
        return self

    def __next__(self) -> Iterator["_DataFetcherWrapper"]:  # type: ignore[override]
        if self.done:
            raise StopIteration
        return self.iterator_wrapper

    def reset(self) -> None:
        """
        Reset the data fetcher.
        """
        super().reset()
        self._batch = None
        self._batch_idx = 0
        self._dataloader_idx = 0


class _DataFetcherWrapper(Iterator):
    def __init__(
        self,
        data_fetcher: _MegatronDataLoaderIterDataFetcher,
        output_data_idx: bool = False,
    ) -> None:
        self.data_fetcher = data_fetcher
        self.output_data_idx = output_data_idx

    @property
    def done(self) -> bool:
        """
        Check if the data fetcher is done.
        """
        return self.data_fetcher.done

    @property
    def fetched(self) -> int:
        """
        Check if the data fetcher is fetched.
        """
        return self.data_fetcher.fetched

    @property
    def length(self) -> Optional[int]:
        """
        Get the length of the data fetcher.
        """
        return self.data_fetcher.length

    @property
    def data_config(self):
        """
        Get the data config.
        """
        return self.data_fetcher.data_config

    def __next__(self):
        fetcher = self.data_fetcher
        if fetcher.done:
            raise StopIteration
        batch, batch_idx, dataloader_idx = super(_MegatronDataLoaderIterDataFetcher, fetcher).__next__()
        # save the state so the loops can access it
        fetcher._batch = batch  # noqa: SLF001
        fetcher._batch_idx = batch_idx  # noqa: SLF001
        fetcher._dataloader_idx = dataloader_idx  # noqa: SLF001

        if not self.output_data_idx:
            return batch

        return batch, batch_idx, dataloader_idx


@to_fabric.register(MegatronStrategy)
def convert_megatron_strategy(strategy: MegatronStrategy) -> FabricMegatronStrategy:
    """
    Convert the Megatron strategy to the Fabric strategy.
    """
    return FabricMegatronStrategy(
        tensor_model_parallel_size=strategy.tensor_model_parallel_size,
        pipeline_model_parallel_size=strategy.pipeline_model_parallel_size,
        pipeline_model_parallel_comm_backend=strategy.pipeline_model_parallel_comm_backend,
        virtual_pipeline_model_parallel_size=strategy.virtual_pipeline_model_parallel_size,
        microbatch_group_size_per_vp_stage=strategy.microbatch_group_size_per_vp_stage,
        context_parallel_size=strategy.context_parallel_size,
        sequence_parallel=strategy.sequence_parallel,
        expert_model_parallel_size=strategy.expert_model_parallel_size,
        expert_tensor_parallel_size=strategy.expert_tensor_parallel_size,
        moe_extended_tp=strategy.moe_extended_tp,
        encoder_tensor_model_parallel_size=strategy.encoder_tensor_model_parallel_size,
        encoder_pipeline_model_parallel_size=strategy.encoder_pipeline_model_parallel_size,
        pipeline_dtype=strategy.pipeline_dtype,
        use_tp_pp_dp_mapping=strategy.use_tp_pp_dp_mapping,
        ddp=strategy._ddp,
        process_group_backend=strategy.process_group_backend,
        timeout=strategy._timeout,
        start_method=strategy._start_method,
    )
