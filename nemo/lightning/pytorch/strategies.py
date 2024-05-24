import functools
import logging
import shutil
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, ContextManager, Dict, List, Mapping, Optional, TypeVar, Union, cast

import pytorch_lightning as pl
import torch
import torch.distributed
from lightning_fabric.plugins import CheckpointIO, ClusterEnvironment
from lightning_fabric.utilities.optimizer import _optimizers_to_device
from pytorch_lightning.accelerators import CPUAccelerator
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loops import _AutomaticOptimization, evaluation_loop, fit_loop, prediction_loop
from pytorch_lightning.loops.fetchers import _DataLoaderIterDataFetcher
from pytorch_lightning.overrides.distributed import _sync_module_states
from pytorch_lightning.plugins.io.wrapper import _WrappingCheckpointIO
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.trainer.states import RunningStage, TrainerFn
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks import noop_hook
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from typing_extensions import override

from nemo.lightning import _strategy_lib, io
from nemo.lightning.io.pl import MegatronCheckpointIO, TrainerCheckpoint, TrainerCkptProtocol
from nemo.lightning.megatron_parallel import CallbackConnector, MegatronParallel, _ModuleStepFunction
from nemo.lightning.pytorch.callbacks import MegatronProgressBar

if TYPE_CHECKING:
    from nemo.lightning.pytorch.plugins.data_sampler import DataSampler

ConfigT = TypeVar("ConfigT")


class MegatronStrategy(DDPStrategy, io.IOMixin):
    """Megatron plugin for Pytorch Lightning.

    Args:
        no_ddp_communication_hook: Disable DDP communication hook when using AMP-O2
        with FP32 gradient accumulation.
    """

    trainer: pl.Trainer

    def __init__(
        self,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        virtual_pipeline_model_parallel_size: Optional[int] = None,
        sequence_parallel: bool = False,
        data_sampler: Optional['DataSampler'] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment=None,  # TODO: Add type-hint
        checkpoint_io=None,  # TODO: Add type-hint
        no_ddp_communication_hook: bool = True,
        find_unused_parameters: bool = False,
        enable_nemo_ckpt_io: bool = True,
        ckpt_type: TrainerCkptProtocol = TrainerCheckpoint,
        ckpt_include_optimizer: bool = False,
        lazy_init: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            parallel_devices,
            cluster_environment,
            checkpoint_io,
            find_unused_parameters=find_unused_parameters,
            **kwargs,
        )
        self.no_ddp_communication_hook = no_ddp_communication_hook
        self.megatron_callbacks = CallbackConnector()
        self.data_sampler: Optional['DataSampler'] = data_sampler
        self.tensor_model_parallel_size = tensor_model_parallel_size
        self.pipeline_model_parallel_size = pipeline_model_parallel_size
        self.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
        self.sequence_parallel = sequence_parallel
        self.enable_nemo_ckpt_io = enable_nemo_ckpt_io
        self.ckpt_type = ckpt_type
        self.lazy_init = lazy_init
        self.ckpt_include_optimizer = ckpt_include_optimizer

        # used in NVIDIA NGC PyTorch containers
        _strategy_lib.enable_nvidia_optimizations()

    @override
    def connect(self, model: pl.LightningModule) -> None:
        super().connect(model)

        # Right now mcore sub-classes ModelParellelConfig, we should remove that
        # Given Lightning's structure it would be better if parallelism is a different object
        # Since then it can be passed to the Strategy

        from megatron.core.transformer.transformer_config import TransformerConfig

        has_mcore_config = isinstance(getattr(model, "config", None), TransformerConfig)
        if has_mcore_config and is_overridden("configure_model", model):
            config: TransformerConfig = model.config
            config.tensor_model_parallel_size = self.tensor_model_parallel_size
            config.pipeline_model_parallel_size = self.pipeline_model_parallel_size
            config.virtual_pipeline_model_parallel_size = self.virtual_pipeline_model_parallel_size
            config.sequence_parallel = self.sequence_parallel
            self._mcore_config = config

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

        if self.data_sampler:
            self.data_sampler.connect(trainer)

        self._fix_progress_bar(trainer)
        self.setup_megatron_parallel(trainer)
        self.setup_precision_plugin()

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

            # set up optimizers after the wrapped module has been moved to the device
            self.setup_optimizers(trainer)
            if hasattr(self.precision_plugin, "convert_optimizer"):
                _optimizers = [*self.optimizers]
                _optimizers[0] = self.precision_plugin.convert_optimizer(self.optimizers[0])
                self.optimizers = _optimizers

            _optimizers_to_device(self.optimizers, self.root_device)

            import torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook as post_localSGD

            if isinstance(self._ddp_comm_state, post_localSGD.PostLocalSGDState):
                self._enable_model_averaging()
        else:
            # we need to manually synchronize the module's states since we aren't using the DDP wrapper
            assert self.model is not None
            _sync_module_states(self.model)

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

        self.megatron_parallel = MegatronParallel(
            self.model,
            precision_plugin=self.precision_plugin,
            vp_size=self.virtual_pipeline_model_parallel_size,
            cpu=isinstance(trainer.accelerator, CPUAccelerator),
        )
        self.model = self.megatron_parallel
        self.model.trainer = trainer

        if hasattr(self.precision_plugin, "convert_module"):
            self.model = self.precision_plugin.convert_module(self.model)
        self.model.callbacks.add(getattr(trainer, "callbacks"))

        if self.data_sampler:
            self.model.callbacks.add(self.data_sampler)

        datamodule = getattr(trainer, "datamodule", None)
        if datamodule:
            self.model.callbacks.add(datamodule)

    @override
    def configure_ddp(self) -> None:
        logging.debug(f"{self.__class__.__name__}: configuring MegatronParallel")
        self.model = self._setup_model(self.model)
        self._register_ddp_hooks()

    @override
    def _setup_model(self, model: nn.Module) -> DistributedDataParallel:
        """Only called when we need to wrap the model for pytorch's ddp."""
        from megatron.core import parallel_state

        from nemo.utils import AppState

        app_state = AppState()
        if app_state.model_parallel_size is not None:
            self._ddp_kwargs["process_group"] = parallel_state.get_data_parallel_group()

        dist_data_parallel: DistributedDataParallel = super()._setup_model(model)
        if self.no_ddp_communication_hook:
            # When using custom gradient accumulation and allreduce, disable
            # DDP communication hook that works on the gradient bucket.
            # Instead, use the custom gradient function and communication hook,
            # which is defined in the master optimizer wrapper.
            dist_data_parallel.require_backward_grad_sync = False
            dist_data_parallel.register_comm_hook(None, noop_hook)

        return dist_data_parallel

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
            return self.model(dataloader_iter, *args, **kwargs)

    @override
    def validation_step(self, dataloader_iter, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        assert self.lightning_module is not None
        assert self.model is not None
        kwargs = self._update_step_kwargs(dataloader_iter, kwargs, "validation")

        with self.precision_plugin.val_step_context():  # TODO: Do we need this?
            return self.model(dataloader_iter, *args, **kwargs)

    @override
    def test_step(self, dataloader_iter, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        assert self.lightning_module is not None
        assert self.model is not None
        kwargs = self._update_step_kwargs(dataloader_iter, kwargs, "test")

        with self.precision_plugin.test_step_context():  # TODO: Do we need this?
            return self.model(dataloader_iter, *args, **kwargs)

    @override
    def predict_step(self, dataloader_iter, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        assert self.lightning_module is not None
        assert self.model is not None
        kwargs = self._update_step_kwargs(dataloader_iter, kwargs, "predict")

        with self.precision_plugin.predict_step_context():  # TODO: Do we need this?
            return self.model(dataloader_iter, *args, **kwargs)

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
            for callback in callbacks:
                if isinstance(callback, TQDMProgressBar):
                    callback.__class__ = MegatronProgressBar
                    break

    def optimizer_sharded_state_dict(self):
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

        return _strategy_lib.optimizer_sharded_state_dict(self.megatron_parallel, optimizer)

    @override
    def save_checkpoint(
        self, checkpoint: Dict[str, Any], filepath: Union[str, Path], storage_options: Optional[Any] = None
    ) -> None:
        checkpoint["state_dict"] = OrderedDict([])  # remove device state_dict
        checkpoint["sharded_state_dict"] = self.megatron_parallel.sharded_state_dict()
        if self.trainer.state.fn == TrainerFn.FITTING:
            checkpoint["optimizer_states"] = [self.optimizer_sharded_state_dict()]

        self.checkpoint_io.save_checkpoint(checkpoint, filepath, storage_options=storage_options)
        if self.enable_nemo_ckpt_io and self.is_global_zero and self.ckpt_type:
            self.ckpt_type.from_strategy(self).io_dump(ckpt_to_dir(filepath))

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
                sharded_state_dict["optimizer_states"] = [self.optimizer_sharded_state_dict()]

        checkpoint = self.checkpoint_io.load_checkpoint(checkpoint_path, sharded_state_dict=sharded_state_dict)

        return checkpoint

    def remove_checkpoint(self, filepath: Union[str, Path]) -> None:
        if self.is_global_zero:
            shutil.rmtree(ckpt_to_dir(filepath))

    def load_model_state_dict(self, checkpoint: Mapping[str, Any], strict: bool = True) -> None:
        assert self.megatron_parallel is not None
        from megatron.core import parallel_state

        for index, module in enumerate(self.megatron_parallel):
            if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                checkpoint_state_dict = checkpoint['state_dict'][f'model_{index}']
            else:
                checkpoint_state_dict = checkpoint['state_dict']
            # checkpoint_state_dict has "model." but module does not so we need to remove it when loading
            checkpoint_state_dict = {
                key.replace('model.', ''): checkpoint_state_dict.pop(key) for key in list(checkpoint_state_dict.keys())
            }
            module.load_state_dict(checkpoint_state_dict, strict=strict)

    @property
    @override
    def checkpoint_io(self) -> CheckpointIO:
        if self._checkpoint_io is None:
            self._checkpoint_io = MegatronCheckpointIO()
        elif isinstance(self._checkpoint_io, _WrappingCheckpointIO):
            self._checkpoint_io.checkpoint_io = MegatronCheckpointIO()

        return self._checkpoint_io

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
    def parallelism(self):
        from megatron.core.model_parallel_config import ModelParallelConfig

        return ModelParallelConfig(
            tensor_model_parallel_size=self.tensor_model_parallel_size,
            pipeline_model_parallel_size=self.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=self.virtual_pipeline_model_parallel_size,
        )


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
