from contextlib import ExitStack, contextmanager
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
from lightning_fabric.accelerators import CPUAccelerator
from lightning_fabric.accelerators.accelerator import Accelerator
from lightning_fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning_fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning_fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning_fabric.plugins.precision import Precision
from lightning_fabric.strategies import DDPStrategy
from lightning_fabric.strategies.strategy import _validate_keys_for_strict_loading
from lightning_fabric.utilities.types import _PATH, _Stateful
from megatron.core.distributed import DistributedDataParallelConfig
from pytorch_lightning.loops.fetchers import _DataFetcher
from pytorch_lightning.plugins.io.wrapper import _WrappingCheckpointIO
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch import Tensor, nn
from torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks import noop_hook
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing_extensions import override

from nemo.lightning import _strategy_lib
from nemo.lightning.fabric.conversion import to_fabric
from nemo.lightning.io.pl import MegatronCheckpointIO
from nemo.lightning.megatron_parallel import CallbackConnector, MegatronParallel
from nemo.lightning.pytorch.strategies import MegatronStrategy

if TYPE_CHECKING:
    from megatron.core.model_parallel_config import ModelParallelConfig

    from nemo.lightning.pytorch.plugins.data_sampler import DataSampler


DDPLiteral = Literal["megatron", "pytorch"]


class FabricMegatronStrategy(DDPStrategy):
    def __init__(
        self,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        virtual_pipeline_model_parallel_size: Optional[int] = None,
        context_parallel_size: int = 1,
        sequence_parallel: bool = False,
        expert_model_parallel_size: int = 1,
        moe_extended_tp: bool = False,
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
        self.context_parallel_size = context_parallel_size
        self.expert_model_parallel_size = expert_model_parallel_size
        self.moe_extended_tp = moe_extended_tp
        self.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
        self.sequence_parallel = sequence_parallel
        self.pipeline_dtype = pipeline_dtype

        self.no_ddp_communication_hook = no_ddp_communication_hook
        self.megatron_callbacks = CallbackConnector()
        if megatron_callbacks:
            self.megatron_callbacks.add(megatron_callbacks)
        self.output_data_idx = output_data_idx

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

    @override
    def process_dataloader(self, dataloader: DataLoader) -> Iterator:
        loader = _strategy_lib.process_dataloader(dataloader, self.data_config)

        # Code taken from: https://github.com/Lightning-AI/pytorch-lightning/blob/6cbe9ceb560d798892bdae9186291acf9bf5d2e3/src/lightning/pytorch/loops/fit_loop.py#L258-L260
        output = _MegatronDataLoaderIterDataFetcher(self.data_config, output_data_idx=self.output_data_idx)
        output.setup(CombinedLoader(loader, "max_size_cycle"))
        iter(output)

        return output

    @override
    def setup_optimizer(self, optimizer: Optimizer) -> Optimizer:
        """Pass the optimizer to the precision-plugin if needed & add it as callback."""
        if hasattr(self._precision, "setup_optimizer"):
            optimizer = self._precision.setup_optimizer(optimizer)

        self.megatron_callbacks.add(optimizer)

        return optimizer

    @override
    def setup_module(self, module: Module) -> MegatronParallel:
        _strategy_lib.set_model_parallel_attributes(module, self.parallelism)

        # Call configure_model if it's overridden (relevant for LightningModules with lazy initialization)
        if hasattr(module, "configure_model"):
            module.configure_model()

        convert_module_fn = None
        if hasattr(self.precision, "convert_module"):
            convert_module_fn = self.precision.convert_module

        megatron_parallel = MegatronParallel(
            module,
            precision_plugin=self.precision,
            vp_size=self.virtual_pipeline_model_parallel_size,
            cpu=isinstance(self.accelerator, CPUAccelerator),
            ddp_config=self.ddp_config,
            convert_module_fn=convert_module_fn,
        )

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
        state = self._convert_stateful_objects_in_state(state, filter=(filter_dict or {}))
        self.checkpoint_io.save_checkpoint(checkpoint=state, path=path, storage_options=storage_options)

    def load_checkpoint(
        self,
        path: _PATH,
        state: Optional[Union[Module, Optimizer, Dict[str, Union[Module, Optimizer, Any]]]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        if isinstance(state, Optimizer):
            raise NotImplementedError("Optimizer loading is not supported, pass it as a dict including the model")

        torch.cuda.empty_cache()

        # After dist_checkpointing.load, sharded tensors will be replaced with tensors
        sharded_state_dict = {}
        if isinstance(state, Module):
            sharded_state_dict["state_dict"] = state.sharded_state_dict()
        elif strict:
            sharded_state_dict["state_dict"] = state["state_dict"].sharded_state_dict()
            if "optimizer" in state:
                sharded_state_dict["optimizer"] = _strategy_lib.optimizer_sharded_state_dict(
                    state["state_dict"], state["optimizer"], is_loading=True
                )
        else:
            for obj in state.items():
                if isinstance(obj, Module):
                    sharded_state_dict["state_dict"] = obj.sharded_state_dict()
                elif isinstance(obj, Optimizer):
                    sharded_state_dict["optimizer"] = _strategy_lib.optimizer_sharded_state_dict(obj, is_loading=True)

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
        _strategy_lib.load_model_state_dict(module, state_dict, strict=strict)

    @contextmanager
    def megatron_context(self) -> Generator[None, None, None]:
        def monkey_patched(config):
            return {"device": "meta"}

        from megatron.core.transformer.custom_layers import transformer_engine as _te

        original = _te._get_extra_te_kwargs  # noqa: SLF001
        _te._get_extra_te_kwargs = monkey_patched  # noqa: SLF001

        self.parallelism.perform_initialization = False
        self.parallelism.use_cpu_initialization = True

        yield

        _te._get_extra_te_kwargs = original  # noqa: SLF001

    @property
    @override
    def checkpoint_io(self) -> CheckpointIO:
        if self._checkpoint_io is None:
            self._checkpoint_io = MegatronCheckpointIO()
        elif isinstance(self._checkpoint_io, _WrappingCheckpointIO):
            self._checkpoint_io.checkpoint_io = MegatronCheckpointIO()

        return self._checkpoint_io

    @property
    def parallelism(self):
        from nemo.lightning.pytorch.strategies import ParallelismConfig

        return ParallelismConfig(
            tensor_model_parallel_size=self.tensor_model_parallel_size,
            pipeline_model_parallel_size=self.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=self.virtual_pipeline_model_parallel_size,
            context_parallel_size=self.context_parallel_size,
            sequence_parallel=self.sequence_parallel,
            expert_model_parallel_size=self.expert_model_parallel_size,
            moe_extended_tp=self.moe_extended_tp,
            pipeline_dtype=self.pipeline_dtype,
        )


# TODO: Fix this
class _MegatronDataLoaderIterDataFetcher(_DataFetcher):
    def __init__(self, data_config, *args: Any, output_data_idx: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.data_config = data_config
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
        return self.data_fetcher.done

    @property
    def fetched(self) -> int:
        return self.data_fetcher.fetched

    @property
    def length(self) -> Optional[int]:
        return self.data_fetcher.length

    @property
    def data_config(self):
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
    return FabricMegatronStrategy(
        tensor_model_parallel_size=strategy.tensor_model_parallel_size,
        pipeline_model_parallel_size=strategy.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=strategy.virtual_pipeline_model_parallel_size,
        context_parallel_size=strategy.context_parallel_size,
        sequence_parallel=strategy.sequence_parallel,
        expert_model_parallel_size=strategy.expert_model_parallel_size,
        moe_extended_tp=strategy.moe_extended_tp,
        pipeline_dtype=strategy.pipeline_dtype,
        ddp=strategy._ddp,
        process_group_backend=strategy.process_group_backend,
        timeout=strategy._timeout,
        start_method=strategy._start_method,
    )
