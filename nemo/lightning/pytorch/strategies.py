import functools
import inspect
import logging
import os
import shutil
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ContextManager, Dict, List, Literal, Mapping, Optional, TypeVar, Union, cast

import pytorch_lightning as pl
import torch
import torch.distributed
from lightning_fabric.plugins import CheckpointIO, ClusterEnvironment
from lightning_fabric.utilities.optimizer import _optimizer_to_device, _optimizers_to_device
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from pytorch_lightning.accelerators import CPUAccelerator
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loops import _AutomaticOptimization, evaluation_loop, fit_loop, prediction_loop
from pytorch_lightning.loops.fetchers import _DataLoaderIterDataFetcher
from pytorch_lightning.overrides.distributed import _sync_module_states
from pytorch_lightning.plugins.io.wrapper import _WrappingCheckpointIO
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.trainer.states import RunningStage, TrainerFn
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks import noop_hook
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from typing_extensions import override

from nemo.lightning import _strategy_lib, io
from nemo.lightning.io.pl import MegatronCheckpointIO
from nemo.lightning.megatron_parallel import CallbackConnector, MegatronParallel, _ModuleStepFunction
from nemo.lightning.pytorch.callbacks import MegatronProgressBar, ModelTransform, ProgressPrinter
from nemo.utils.callbacks.dist_ckpt_io import AsyncFinalizableCheckpointIO, AsyncFinalizerCallback

if TYPE_CHECKING:
    from nemo.lightning.pytorch.plugins.data_sampler import DataSampler

ConfigT = TypeVar("ConfigT")


DDPLiteral = Literal["megatron", "pytorch"]


@dataclass
class ParallelismConfig:
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    virtual_pipeline_model_parallel_size: int
    context_parallel_size: int
    sequence_parallel: bool
    expert_model_parallel_size: int
    moe_extended_tp: bool
    pipeline_dtype: torch.dtype


class MegatronStrategy(DDPStrategy, io.IOMixin):
    """Megatron plugin for Pytorch Lightning.

    This strategy implements model parallelism using NVIDIA's Megatron-LM framework. It supports
    various forms of parallelism including tensor model parallelism, pipeline model parallelism,
    sequence parallelism, and expert parallelism for efficient training of large language models.

    Args:
        tensor_model_parallel_size (int): Intra-layer model parallelism. Splits tensors across GPU ranks.
            Defaults to 1.
        pipeline_model_parallel_size (int): Inter-layer model parallelism. Splits transformer layers
            across GPU ranks. Defaults to 1.
        virtual_pipeline_model_parallel_size (Optional[int]): Interleaved pipeline parallelism used to
            improve performance by reducing the pipeline bubble. Defaults to None.
        context_parallel_size (int): Splits network input along sequence dimension across GPU ranks.
            Defaults to 1.
        sequence_parallel (bool): Makes tensor parallelism more memory efficient for LLMs (20B+) by
            parallelizing layer norms and dropout sequentially. Defaults to False.
        expert_model_parallel_size (int): Distributes MoE Experts across sub data parallel dimension.
            Defaults to 1.
        moe_extended_tp (bool): Alternative parallelization strategy for expert parallelism. Defaults to False.
        data_sampler (Optional['DataSampler']): Custom data sampler for distributed training. Defaults to None.
        parallel_devices (Optional[List[torch.device]]): List of devices to use for parallelism. Defaults to None.
        cluster_environment: Cluster environment for distributed training. Defaults to None.
        checkpoint_io: Checkpoint I/O handler. Defaults to None.
        find_unused_parameters (bool): Find unused parameters in DDP. Defaults to False.
        ckpt_type (TrainerCkptProtocol): Checkpoint type. Defaults to TrainerCheckpoint.
        ckpt_include_optimizer (bool): Include optimizer state in checkpoint. Defaults to True.
        ddp (Union[DDPLiteral, DistributedDataParallelConfig]): DDP configuration. Defaults to "megatron".
        lazy_init (bool): Use lazy initialization for model parallel parameters. Defaults to False.
        pipeline_dtype (Optional[torch.dtype]): Data type for pipeline parallelism. Defaults to None.
        save_ckpt_format (str): Distributed checkpoint format to use for checkpoint saving. Should be one of
            'torch_dist' or 'zarr'. Defaults to 'torch_dist'.
        ckpt_async_save (bool): Whether to save checkpoints asynchronously to reduce checkpointing overhead.
            Defaults to False.
        ckpt_torch_dist_multiproc (int): Number of extra processes per rank used during ckpt save
            with PyTorch distributed format. Defaults to None.
        ckpt_assume_constant_structure (bool): Allows caching some computation across checkpoint saves.
            Set to True only if the state dict structure doesn't change within a single job.
        ckpt_parallel_save (bool): If true, each worker will write its own part of the dist checkpoint.
            Defaults to True.
        ckpt_parallel_save_within_dp (bool): If true, save will be parallelized only within a DP group
            (whole world otherwise), which might slightly reduce the save overhead. Defaults to False.
        ckpt_parallel_load (bool): If true, each worker will load part of the dist checkpoint
            and exchange with NCCL. Might use some extra GPU memory. Defaults to False.
        ckpt_parallel_save_optim (bool): Parallel save/load of a DistributedOptimizer. 'True'
            allows performant save and reshardable checkpoints. Set to 'False' only in order to minimize
            the number of checkpoint files.
        ckpt_load_directly_on_device (bool): if True, loads the weights directly on GPU.
            Has effect only for `zarr` based checkpoints (PyT Distributed always loads on device).
            Defaults to True.
        setup_optimizers (bool): Whether to call the trainer's setup_optimizers function to perform any
            necessary conversions of optimizer parameters and move optimizer parameters to the correct device.
            Defaults to True.
        init_model_parallel (bool): Whether to initialize the model parallel groups. Defaults to True.
        replace_progress_bar (bool): Whether to replace the TQDM progress bar with a megatron-style logger
            that prints the metrics to stdout. Suitable for non-interactive settings.
        progress_interval (int): How frequently to print progress to stdout. Only used when
            replace_progress_bar is True.
        **kwargs: Additional keyword arguments.

    Note:
        This strategy is designed to work with NVIDIA's Megatron-LM framework and requires
        specific model implementations that are compatible with Megatron's parallelism techniques.
    """

    trainer: pl.Trainer

    def __init__(
        self,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        virtual_pipeline_model_parallel_size: Optional[int] = None,
        context_parallel_size: int = 1,
        sequence_parallel: bool = False,
        expert_model_parallel_size: int = 1,
        moe_extended_tp: bool = False,
        data_sampler: Optional['DataSampler'] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment=None,  # TODO: Add type-hint
        checkpoint_io=None,  # TODO: Add type-hint
        find_unused_parameters: bool = False,
        ckpt_include_optimizer: bool = True,
        ddp: Union[DDPLiteral, DistributedDataParallelConfig] = "megatron",
        lazy_init: bool = False,
        pipeline_dtype: Optional[torch.dtype] = None,
        save_ckpt_format: str = 'torch_dist',
        ckpt_async_save: bool = False,
        ckpt_torch_dist_multiproc: int = None,  ## TODO(ashors): put elsewhere?
        ckpt_assume_constant_structure: bool = False,
        ckpt_parallel_save: bool = True,
        ckpt_parallel_save_within_dp: bool = False,
        ckpt_parallel_load: bool = False,
        ckpt_parallel_save_optim: bool = True,
        ckpt_load_directly_on_device: bool = True,
        setup_optimizers: bool = True,
        init_model_parallel: bool = True,
        replace_progress_bar: bool = True,
        progress_interval: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            find_unused_parameters=find_unused_parameters,
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
        self.lazy_init = lazy_init
        self.ckpt_include_optimizer = ckpt_include_optimizer
        self.pipeline_dtype = pipeline_dtype
        self._setup_optimizers = setup_optimizers
        self._init_model_parallel = init_model_parallel
        self.log_train_loss = bool(int(os.getenv("NEMO_LOG_TRAIN_LOSS", 1)))
        self.log_memory_usage = bool(int(os.getenv("NEMO_LOG_MEMORY_USAGE", 0)))

        self.save_ckpt_format = save_ckpt_format
        self.async_save = ckpt_async_save
        self.torch_dist_multiproc = ckpt_torch_dist_multiproc
        self.assume_constant_structure = ckpt_assume_constant_structure
        self.parallel_save = ckpt_parallel_save
        self.parallel_save_within_dp = ckpt_parallel_save_within_dp
        self.parallel_load = ckpt_parallel_load
        self.parallel_save_optim = ckpt_parallel_save_optim
        self.load_directly_on_device = ckpt_load_directly_on_device

        self.replace_progress_bar = replace_progress_bar
        self.progress_interval = progress_interval

        self._ddp = ddp
        if ddp == "megatron":
            self.ddp_config = DistributedDataParallelConfig(check_for_nan_in_grad=True)
        elif isinstance(ddp, DistributedDataParallelConfig):
            self.ddp_config = ddp
        elif ddp == "pytorch":
            self.ddp_config = None
            self.no_ddp_communication_hook = False
        else:
            raise ValueError(f"Invalid DDP type: {ddp}")

        # used in NVIDIA NGC PyTorch containers
        _strategy_lib.enable_nvidia_optimizations()

    @override
    def connect(self, model: pl.LightningModule) -> None:
        super().connect(model)

        _maybe_mcore_config = _strategy_lib.set_model_parallel_attributes(model, self.parallelism)
        if _maybe_mcore_config:
            self._mcore_config = _maybe_mcore_config

        dtype_config = getattr(self._precision_plugin, 'dtype_config', None)
        if dtype_config:
            from nemo.lightning.pytorch.plugins.mixed_precision import update_config_with_dtype_overrides

            model.config = update_config_with_dtype_overrides(dtype_config, model.config)

        has_optim = getattr(model, "optim", None)
        if has_optim:
            opt_config = getattr(model.optim, "config", None)
            if isinstance(opt_config, OptimizerConfig):
                mcore_opt_config: OptimizerConfig = cast(OptimizerConfig, opt_config)
                if not self.ddp_config:
                    raise ValueError("PyTorch DDP is not enabled for mcore optimizer")
                ddp_config = cast(DistributedDataParallelConfig, self.ddp_config)

                if dtype_config:
                    model.optim.config = update_config_with_dtype_overrides(dtype_config, model.optim.config)
                    self.ddp_config = update_config_with_dtype_overrides(dtype_config, self.ddp_config)

                if mcore_opt_config.use_distributed_optimizer != ddp_config.use_distributed_optimizer:
                    from nemo.utils import logging

                    logging.info("Fixing mis-match between ddp-config & mcore-optimizer config")
                    ddp_config.use_distributed_optimizer = mcore_opt_config.use_distributed_optimizer

    @override
    def setup(self, trainer: pl.Trainer) -> None:
        assert self.accelerator is not None
        self.accelerator.setup(trainer)
        self.trainer = trainer

        # move the model to the correct device
        # self.model_to_device()

        # skip wrapping the model if we are not fitting as no gradients need to be exchanged
        trainer_fn = trainer.state.fn

        if trainer_fn == TrainerFn.FITTING and self._layer_sync:
            assert self.model is not None
            self.model = self._layer_sync.apply(self.model)

        datamodule = getattr(trainer, "datamodule", None)
        if not self.data_sampler and hasattr(datamodule, "data_sampler"):
            self.data_sampler = datamodule.data_sampler
            self.data_sampler.setup(self.cluster_environment.global_rank())
            if hasattr(datamodule, "reconfigure_limit_batches"):
                datamodule.reconfigure_limit_batches()

        if self.data_sampler:
            self.data_sampler.connect(trainer)

        self._fix_progress_bar(trainer)
        self.setup_megatron_parallel(trainer)
        self.setup_precision_plugin()

        if getattr(self.lightning_module, "model_transform", None):
            # Ensure the ModelTransform callback is pass to the trainer.
            # Callback.setup() is called before the current Strategy.setup(), so we can
            # only perform a check here; adding the callback here would not be sufficient
            if not any(isinstance(cb, ModelTransform) for cb in trainer.callbacks):
                raise ValueError(
                    "You specified a model_transform function in the model, but no"
                    "ModelTransform callback was found in the trainer. "
                    "Please initialize the trainer with "
                    "`trainer = Trainer(..., callbacks=[ModelTransform()])`"
                )

        if trainer.num_sanity_val_steps > 1 and self.pipeline_model_parallel_size > 1:
            # TODO: log here
            trainer.num_sanity_val_steps = 0

        for loop in [fit_loop, evaluation_loop, prediction_loop]:
            loop._select_data_fetcher = _data_fetcher_wrapper(loop._select_data_fetcher)  # noqa: SLF001

        if trainer_fn == TrainerFn.FITTING:
            # TODO: Make sure we don't always wrap the model in data-parallel
            # See: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/parts/nlp_overrides.py#L215-L217

            # do not wrap with DDP if not fitting as there's no gradients to reduce
            self.configure_ddp()

            trainer.fit_loop.epoch_loop.automatic_optimization = _MegatronAutomaticOptimization(trainer)

            import torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook as post_localSGD

            if isinstance(self._ddp_comm_state, post_localSGD.PostLocalSGDState):
                self._enable_model_averaging()
        else:
            # we need to manually synchronize the module's states since we aren't using the DDP wrapper
            assert self.model is not None
            _sync_module_states(self.model)

        ## add AsyncFinalizerCallback if using async
        if self.async_save:
            have_async_callback = False
            for callback in self.trainer.callbacks:
                if isinstance(callback, AsyncFinalizerCallback):
                    have_async_callback = True
                    break
            if not have_async_callback:
                self.trainer.callbacks.append(AsyncFinalizerCallback())

    @override
    def setup_distributed(self) -> None:
        self._setup_parallel_ranks()
        super().setup_distributed()

        from megatron.core import parallel_state

        from nemo.utils import AppState

        # init model parallel if needed
        if not parallel_state.model_parallel_is_initialized():
            app_state = AppState()

            if app_state.model_parallel_size is not None:
                _strategy_lib.init_model_parallel(self.model)

        if self.data_sampler:
            assert isinstance(self.cluster_environment, ClusterEnvironment), "Cluster environment not initialized"
            self.data_sampler.setup(self.cluster_environment.global_rank())

    @override
    def process_dataloader(self, dataloader: DataLoader) -> DataLoader:
        if self.data_sampler:
            return self.data_sampler.transform_dataloader(dataloader)

        return dataloader

    def setup_megatron_parallel(self, trainer: pl.Trainer) -> None:
        assert self.model is not None, "Model is not set"

        convert_module_fn = None
        if hasattr(self.precision_plugin, "convert_module"):
            convert_module_fn = self.precision_plugin.convert_module

        self.megatron_parallel = MegatronParallel(
            self.model,
            precision_plugin=self.precision_plugin,
            vp_size=self.virtual_pipeline_model_parallel_size,
            cpu=isinstance(trainer.accelerator, CPUAccelerator),
            ddp_config=self.ddp_config,
            convert_module_fn=convert_module_fn,
        )

        if self._init_model_parallel:
            self.init_model_parallel()

        self.megatron_parallel.trainer = trainer

        # check signature-def of self.model.configure_optimizers to check if there's an optional arg: megatron_parallel
        sig = inspect.signature(self.model.configure_optimizers)
        if "megatron_parallel" in sig.parameters:
            self.model.configure_optimizers = functools.partial(
                self.model.configure_optimizers, megatron_parallel=self.megatron_parallel
            )

        if self._setup_optimizers:
            self.setup_optimizers(trainer)

        self.model = self.megatron_parallel
        self.model.callbacks.add(getattr(trainer, "callbacks"))

        if self.data_sampler:
            self.model.callbacks.add(self.data_sampler)

        datamodule = getattr(trainer, "datamodule", None)
        if datamodule:
            self.model.callbacks.add(datamodule)

    def init_model_parallel(self):
        self.megatron_parallel.init_model_parallel()

    @override
    def configure_ddp(self) -> None:
        logging.debug(f"{self.__class__.__name__}: configuring MegatronParallel")
        self.model = self._setup_model(self.model)
        if self.ddp_config is None:
            self._register_ddp_hooks()

    @override
    def _setup_model(self, model: nn.Module) -> nn.Module:
        """Only called when we need to wrap the model for pytorch's ddp."""
        from megatron.core import parallel_state

        from nemo.utils import AppState

        app_state = AppState()
        if app_state.model_parallel_size is not None:
            self._ddp_kwargs["process_group"] = parallel_state.get_data_parallel_group()

        # Only wrap the model if we are not using Megatron's DDP
        if not self.ddp_config:
            dist_data_parallel: DistributedDataParallel = super()._setup_model(model)
            if self.no_ddp_communication_hook:
                # When using custom gradient accumulation and allreduce, disable
                # DDP communication hook that works on the gradient bucket.
                # Instead, use the custom gradient function and communication hook,
                # which is defined in the master optimizer wrapper.
                dist_data_parallel.require_backward_grad_sync = False
                dist_data_parallel.register_comm_hook(None, noop_hook)
            model = dist_data_parallel

        return model

    @override
    def setup_optimizers(self, trainer: "pl.Trainer") -> None:
        super().setup_optimizers(trainer)
        if hasattr(self.precision_plugin, "convert_optimizer"):
            _optimizers = [*self.optimizers]
            _optimizers[0] = self.precision_plugin.convert_optimizer(self.optimizers[0])
            self.optimizers = _optimizers

        _optimizers_to_device(self.optimizers, self.root_device)

    def _setup_parallel_ranks(self) -> None:
        self.set_world_ranks()
        env = cast(ClusterEnvironment, self.cluster_environment)

        _strategy_lib.init_parallel_ranks(env.world_size(), env.global_rank(), env.local_rank(), self.parallelism)

    @override
    def training_step(self, dataloader_iter, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        assert self.lightning_module is not None
        assert self.model is not None
        kwargs = self._update_step_kwargs(dataloader_iter, kwargs, "training")

        with self.precision_plugin.train_step_context():  # TODO: Do we need this?
            # Set grad to zero.
            for model_chunk in self.model:
                model_chunk.zero_grad_buffer()
            for opt in self.optimizers:
                opt.zero_grad()

            out = self.model(dataloader_iter, forward_only=False, *args, **kwargs)

            self.lightning_module.log(
                'global_step',
                self.trainer.global_step,
                prog_bar=True,
                batch_size=1,
            )

            self.lightning_module.log(
                'step',
                self.trainer.global_step,
            )

            if self.log_memory_usage:
                max_memory_reserved = torch.cuda.max_memory_reserved()
                memory_allocated = torch.cuda.memory_allocated()
                self.lightning_module.log(
                    "peak_memory_usage",
                    max_memory_reserved,
                    prog_bar=True,
                    batch_size=1,
                )
                self.lightning_module.log(
                    "memory_allocated",
                    memory_allocated,
                    prog_bar=True,
                    batch_size=1,
                )

            if self.log_train_loss:
                # p2p now, broadcast later at ckpt. only with pp, some ranks will log 0.0
                # WHICH IS OK because we broadcast later at checkpoint time
                _strategy_lib._sync_from_last_pipeline_stage(out, broadcast=False)
                self.lightning_module.log('reduced_train_loss', out, prog_bar=True, batch_size=1, sync_dist=False)

            return out

    @override
    def validation_step(self, dataloader_iter, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        assert self.lightning_module is not None
        assert self.model is not None
        kwargs = self._update_step_kwargs(dataloader_iter, kwargs, "validation")

        with self.precision_plugin.val_step_context():  # TODO: Do we need this?
            out = self.model(dataloader_iter, forward_only=True, *args, **kwargs)

            from megatron.core import parallel_state

            pp_size = parallel_state.get_pipeline_model_parallel_world_size()
            if pp_size > 1:
                # ranks that are not final pp stage have 0 for loss, and out will be mean-reduced over pp
                # groups (due to sync_dist), which divides val_loss by pp_size. so we multiply by pp_size to cancel out
                self.lightning_module.log(
                    'val_loss',
                    out * pp_size,
                    prog_bar=True,
                    sync_dist=True,
                    sync_dist_group=parallel_state.get_pipeline_model_parallel_group(),
                    on_epoch=True,
                )
            else:
                self.lightning_module.log('val_loss', out, prog_bar=True, on_epoch=True)

            return out

    @override
    def test_step(self, dataloader_iter, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        assert self.lightning_module is not None
        assert self.model is not None
        kwargs = self._update_step_kwargs(dataloader_iter, kwargs, "test")

        with self.precision_plugin.test_step_context():  # TODO: Do we need this?
            return self.model(dataloader_iter, forward_only=True, *args, **kwargs)

    @override
    def predict_step(self, dataloader_iter, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        assert self.lightning_module is not None
        assert self.model is not None
        kwargs = self._update_step_kwargs(dataloader_iter, kwargs, "predict")

        with self.precision_plugin.predict_step_context():  # TODO: Do we need this?
            return self.model(dataloader_iter, forward_only=True, *args, **kwargs)

    @override
    def teardown(self) -> None:
        super().teardown()

    @override
    def model_sharded_context(self) -> ContextManager:
        if self.lazy_init and hasattr(self, "_mcore_config"):
            stack = ExitStack()
            stack.enter_context(_strategy_lib.megatron_lazy_init_context(self._mcore_config))
            return stack

        return super().model_sharded_context()

    def _update_step_kwargs(self, dataloader_iter, kwargs, step_name: str):
        if "data_step" not in kwargs:
            kwargs["data_step"] = self._get_data_step(step_name)
        if "forward_step" not in kwargs:
            kwargs["forward_step"] = self._get_forward_step(step_name)
        if "loss_reduction" not in kwargs:
            kwargs["loss_reduction"] = self._get_loss_reduction(step_name)
        kwargs.update(self._data_config_kwargs(dataloader_iter))

        return kwargs

    def _fix_progress_bar(self, trainer: pl.Trainer) -> None:
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
                    if self.replace_progress_bar:
                        printer = ProgressPrinter(log_interval=self.progress_interval)
                        printer._trainer = trainer
                        if not trainer.is_global_zero:
                            printer.disable()
                        callbacks[i] = printer
                    else:
                        callback.__class__ = MegatronProgressBar
                    break

    def optimizer_sharded_state_dict(self, is_loading=False):
        """
        Sharded state dictionary for an MainParamsOptimizerWrapper.
        Used to save and load the optimizer state when training with distributed_checkpoint.

        Returns
        -------
            dict: The sharded state dictionary for the optimizer
        Raises:
            ValueError: If a parameter ID does not match any model sharded parameter.
        """
        # TODO: Fix when MainParamsOptimizerWrapper is not used

        optimizer = self.lightning_module.optimizers(use_pl_optimizer=False)
        sharding_type = 'fully_sharded_model_space' if self.parallel_save_optim else 'dp_zero_gather_scatter'

        return _strategy_lib.optimizer_sharded_state_dict(
            self.megatron_parallel, optimizer, is_loading=is_loading, sharding_type=sharding_type
        )

    @override
    def save_checkpoint(
        self, checkpoint: Dict[str, Any], filepath: Union[str, Path], storage_options: Optional[Any] = None
    ) -> None:
        checkpoint["state_dict"] = OrderedDict([])  # remove device state_dict
        # retrieve `sharded_state_dict` if it has not already been configured in `on_save_checkpoint`
        if "sharded_state_dict" not in checkpoint:
            checkpoint["sharded_state_dict"] = self.megatron_parallel.sharded_state_dict()
        if self.trainer.state.fn == TrainerFn.FITTING and self.ckpt_include_optimizer:
            checkpoint["optimizer"] = [self.optimizer_sharded_state_dict()]

        self.checkpoint_io.save_checkpoint(checkpoint, filepath, storage_options=storage_options)

    @override
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """PTL method which we override to integrate distributed checkpoints for model parallel models.
        In order to load distributed checkpoints we need to provide the sharded_state_dict to
        the distributed load function. We get the sharded_state_dict from self.lightning_module
        which makes it convenient to have the loading logic happen at the strategy level.
        """
        torch.cuda.empty_cache()

        # After dist_checkpointing.load, sharded tensors will be replaced with tensors
        sharded_state_dict = {}
        sharded_state_dict["state_dict"] = self.megatron_parallel.sharded_state_dict()

        if self.ckpt_include_optimizer and self.trainer.state.fn == TrainerFn.FITTING:
            if self.lightning_module.optimizers(use_pl_optimizer=False):
                sharded_state_dict["optimizer"] = [self.optimizer_sharded_state_dict(is_loading=True)]

        checkpoint = self.checkpoint_io.load_checkpoint(checkpoint_path, sharded_state_dict=sharded_state_dict)

        return checkpoint

    @override
    def load_optimizer_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        if not self.ckpt_include_optimizer:
            return

        optimizer_states = checkpoint["optimizer"]
        for optimizer, opt_state in zip(self.optimizers, optimizer_states):
            optimizer.load_state_dict(opt_state)
            _optimizer_to_device(optimizer, self.root_device)

    def remove_checkpoint(self, filepath: Union[str, Path]) -> None:
        if self.is_global_zero:
            shutil.rmtree(ckpt_to_dir(filepath))

    def load_model_state_dict(self, checkpoint: Mapping[str, Any], strict: bool = True) -> None:
        assert self.megatron_parallel is not None

        _strategy_lib.load_model_state_dict(self.megatron_parallel, checkpoint, strict=strict)
        for opt in self.optimizers:
            opt.reload_model_params()

    @property
    @override
    def checkpoint_io(self) -> CheckpointIO:
        if self._checkpoint_io is None:
            self._checkpoint_io = MegatronCheckpointIO(
                save_ckpt_format=self.save_ckpt_format,
                async_save=self.async_save,
                torch_dist_multiproc=self.torch_dist_multiproc,
                assume_constant_structure=self.assume_constant_structure,
                parallel_save=self.parallel_save,
                parallel_save_within_dp=self.parallel_save_within_dp,
                parallel_load=self.parallel_load,
                load_directly_on_device=self.load_directly_on_device,
            )
            if self.async_save:
                self._checkpoint_io = AsyncFinalizableCheckpointIO(self._checkpoint_io)
        elif isinstance(self._checkpoint_io, _WrappingCheckpointIO):
            self._checkpoint_io.checkpoint_io = MegatronCheckpointIO()

        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, io: CheckpointIO) -> None:
        self._checkpoint_io = io

    @property
    def current_epoch_step(self) -> int:
        """
        Get the value of step within an epoch.
        """
        return max(
            self.trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.optimizer.step.current.completed,
            self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.current.completed,
        )

    def _get_data_step(self, step_type: str) -> Optional[_ModuleStepFunction]:
        for fn_name in [f"{step_type}_data_step", "data_step"]:
            if hasattr(self.lightning_module, fn_name):
                return _ModuleStepFunction(fn_name)

        return None

    def _get_forward_step(self, step_type: str) -> Optional[_ModuleStepFunction]:
        from megatron.core import parallel_state

        if parallel_state.is_pipeline_last_stage():
            if not hasattr(self.lightning_module, f"{step_type}_step"):
                raise ValueError(f"LightningModule does not have {step_type}_step method")

            return _ModuleStepFunction(f"{step_type}_step", includes_self=True)

        for fn_name in [f"{step_type}_forward_step", "forward_step"]:
            if hasattr(self.lightning_module, fn_name):
                return _ModuleStepFunction(fn_name, includes_self=True)

        return None

    def _get_loss_reduction(self, step_type: str) -> Optional[_ModuleStepFunction]:
        for fn_name in [f"{step_type}_loss_reduction", "loss_reduction"]:
            if hasattr(self.lightning_module, fn_name):
                return _ModuleStepFunction(fn_name, is_property=True)

        return None

    def _data_config_kwargs(self, dataloader_iter) -> Dict[str, Any]:
        if not hasattr(dataloader_iter, "data_config") and self.data_sampler:
            if hasattr(self.data_sampler, "megatron_data_kwargs"):
                return self.data_sampler.megatron_data_kwargs

        return {}

    @property
    def distributed_sampler_kwargs(self) -> Dict[str, Any]:
        from nemo.utils import AppState

        app_state = AppState()
        if app_state.model_parallel_size is not None:
            # When using model parallel, data parallel groups are non-trivial and they
            # correspond to the logical GPUs. This means that the GPUs that form a
            # single logical GPU all need to get the same batch of data.
            distributed_sampler_kwargs = dict(
                num_replicas=app_state.data_parallel_size, rank=app_state.data_parallel_rank
            )
            return distributed_sampler_kwargs

        else:
            return super().distributed_sampler_kwargs

    @property
    def restore_checkpoint_after_setup(self) -> bool:
        """Needs to be True for distributed checkpointing because
        we require the model to have configured the optimizer before
        deserializing the checkpoint.
        """
        return True

    @property
    def parallelism(self) -> ParallelismConfig:
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

    @contextmanager
    @override
    def tensor_init_context(self, empty_init: Optional[bool] = None):
        # Materializaton happens in `setup()`
        # @akoumparouli: using Parent's tensor_init_context causes mcore
        # parameters to be initialized on GPU instead of (assumed) CPU.
        yield


def ckpt_to_dir(filepath: Union[str, Path]) -> Path:
    """PTL considers checkpoints as .ckpt files.
    This method removes the extension and returns a path
    to be used as a directory for distributed checkpoints.
    """
    filepath = Path(filepath)

    if filepath.suffix == ".ckpt":
        return filepath.with_name(filepath.stem)

    return filepath


def _data_fetcher_wrapper(fn):
    @functools.wraps(fn)
    def wrapped(trainer: pl.Trainer, stage: RunningStage):
        if isinstance(trainer.strategy, MegatronStrategy):
            return _DataLoaderIterDataFetcher()

    return wrapped


class _MegatronAutomaticOptimization(_AutomaticOptimization):
    """
    Custom loop for automatic optimization, tailored to work with a specific training_step
    implementation that involves custom data preparation, forward pass, and loss reduction steps.
    """

    def __init__(self, trainer: "pl.Trainer") -> None:
        super().__init__(trainer)
        self._skip_backward = True  # megatron will do the backward pass
