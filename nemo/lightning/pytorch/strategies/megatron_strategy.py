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

import atexit
import functools
import inspect
import logging as _logging
import os
import shutil
from collections import OrderedDict
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ContextManager,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    TypeVar,
    Union,
    cast,
)

import lightning.pytorch as pl
import torch
import torch.distributed
from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment
from lightning.fabric.utilities.optimizer import _optimizer_to_device, _optimizers_to_device
from lightning.fabric.utilities.rank_zero import rank_zero_info
from lightning.fabric.utilities.seed import reset_seed
from lightning.pytorch.accelerators import CPUAccelerator
from lightning.pytorch.loops import _AutomaticOptimization, evaluation_loop, fit_loop, prediction_loop
from lightning.pytorch.loops.fetchers import _DataLoaderIterDataFetcher
from lightning.pytorch.plugins.io.wrapper import _WrappingCheckpointIO
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.trainer.states import RunningStage, TrainerFn
from lightning.pytorch.utilities.types import STEP_OUTPUT

try:
    from megatron.core import Timers
    from megatron.core.dist_checkpointing.validation import StrictHandling
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.optimizer import OptimizerConfig

    HAVE_MEGATRON_CORE = True
except (ImportError, ModuleNotFoundError):

    DistributedDataParallelConfig = object
    HAVE_MEGATRON_CORE = False

from torch import nn
from torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks import noop_hook
from torch.distributed.checkpoint.utils import CheckpointException
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from typing_extensions import override

from nemo.core.optim.mcore_optim import McoreDistributedOptimizer
from nemo.lightning import _strategy_lib, io
from nemo.lightning.megatron_parallel import CallbackConnector, MegatronParallel
from nemo.lightning.pytorch.callbacks import ModelTransform
from nemo.lightning.pytorch.strategies.utils import (
    RestoreConfig,
    _destroy_dist_connection,
    ckpt_to_dir,
    create_checkpoint_io,
    fix_progress_bar,
    init_model_parallel,
    setup_data_sampler,
    setup_parallel_ranks,
)
from nemo.utils import logging
from nemo.utils.callbacks.dist_ckpt_io import AsyncFinalizerCallback

if TYPE_CHECKING:
    from nemo.lightning.pytorch.plugins.data_sampler import DataSampler

_logger = _logging.getLogger(__name__)


ConfigT = TypeVar("ConfigT")


DDPLiteral = Literal["megatron", "pytorch"]
FSDPLiteral = Literal["megatron", "pytorch"]


URL = "https://docs.nvidia.com/nemo-framework/user-guide/latest/knownissues.html"
LOAD_ERROR = f"""
    (1) To resolve this issue, try to set `trainer.strategy.ckpt_load_strictness` to False. This setting enables loading older checkpoints.
    (2) For more details and troubleshooting guidance, please refer to the framework documentation: {URL}.
"""
RESHARDING_LOAD_ERROR = """
    Most likely the error is caused by the parallelism configuration mismatch.
    To resolve this issue, re-save the checkpoint with `ckpt_optim_fully_reshardable=True` flag
    in order to enable a fully-reshardable optimizer checkpoint format.
"""


@dataclass
class ParallelismConfig:
    """
    POD containing parallelism configuration.
    Parallelism configuration is passed to MegatronStrategy via constructor arguments,
    then copied to model's config during model setup.
    """

    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    virtual_pipeline_model_parallel_size: Optional[int]
    microbatch_group_size_per_vp_stage: int
    context_parallel_size: int
    sequence_parallel: bool
    expert_model_parallel_size: int
    moe_extended_tp: bool
    pipeline_dtype: torch.dtype
    encoder_tensor_model_parallel_size: int = 0
    encoder_pipeline_model_parallel_size: int = 0
    pipeline_model_parallel_comm_backend: str = None
    num_layers_in_first_pipeline_stage: Optional[int] = None
    num_layers_in_last_pipeline_stage: Optional[int] = None
    account_for_embedding_in_pipeline_split: bool = False
    account_for_loss_in_pipeline_split: bool = False
    use_te_rng_tracker: bool = False
    expert_tensor_parallel_size: int = None
    use_tp_pp_dp_mapping: bool = False
    num_distributed_optimizer_instances: int = 1
    nccl_communicator_config_path: str = None
    use_sharp: bool = False
    pipeline_model_parallel_layout: Optional[Union[str, List[List[str]]]] = None
    use_gloo_process_groups: bool = True


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
        microbatch_group_size_per_vp_stage (Optional[int]): the number of micro-batches that are executed
            at a time for a given virtual stage (both forward and backward). Defaults to None and convert
            to pipeline_parallel_size. which specifies a depth-first schedule.
        context_parallel_size (int): Splits network input along sequence dimension across GPU ranks.
            Defaults to 1.
        sequence_parallel (bool): Makes tensor parallelism more memory efficient for LLMs (20B+) by
            parallelizing layer norms and dropout sequentially. Defaults to False.
        expert_model_parallel_size (int): Distributes MoE Experts across sub data parallel dimension.
            Defaults to 1.
        expert_tensor_parallel_size (Optional[int]): Sets MoE Experts tensor parallelism size. Defaults to None.
        moe_extended_tp (bool): Alternative parallelization strategy for expert parallelism. Defaults to False.
        account_for_embedding_in_pipeline_split (bool): If set, *input* embedding layer will be treated as a standard
            transformer layer in the context of partition and placement for pipeline parallelism.
        account_for_loss_in_pipeline_split (bool): If set, loss layer will be treated as a standard transformer
            layer in the context of partition and placement for pipeline parallelism.
        data_sampler (Optional['DataSampler']): Custom data sampler for distributed training. Defaults to None.
        parallel_devices (Optional[List[torch.device]]): List of devices to use for parallelism. Defaults to None.
        cluster_environment: Cluster environment for distributed training. Defaults to None.
        checkpoint_io: Checkpoint I/O handler. Defaults to None.
        find_unused_parameters (bool): Find unused parameters in DDP. Defaults to False.
        ckpt_type (TrainerCkptProtocol): Checkpoint type. Defaults to TrainerCheckpoint.
        ckpt_load_optimizer (bool): Load optimizer state from trainer.ckpt_path. Defaults to True.
        ckpt_save_optimizer (bool): Save optimizer states in checkpoint. Defaults to True.
        ckpt_load_main_params (bool): Load main parameters from trainer.ckpt_path. Defaults to False.
        ddp (Union[DDPLiteral, DistributedDataParallelConfig]): DDP configuration. Defaults to "megatron".
        fsdp (Optional[FSDPLiteral]): Option of using torch FSDP2, select from ["megatron", "pytorch"].
            Defaults to None.
        lazy_init (bool): Use lazy initialization for model parallel parameters. Defaults to False.
        pipeline_dtype (Optional[torch.dtype]): Data type for pipeline parallelism. Defaults to None.
        save_ckpt_format (str): Distributed checkpoint format to use for checkpoint saving. Should be one of
            'torch_dist' or 'zarr'. Defaults to 'torch_dist'.
        ckpt_async_save (bool): Whether to save checkpoints asynchronously to reduce checkpointing overhead.
            Defaults to True.
        ckpt_torch_dist_multiproc (int): Number of extra processes per rank used during ckpt save
            with PyTorch distributed format. Defaults to None.
        ckpt_assume_constant_structure (bool): Allows caching some computation across checkpoint saves.
            Set to True only if the state dict structure doesn't change within a single job.
        ckpt_parallel_save (bool): If true, each worker will write its own part of the dist checkpoint.
            Defaults to True.
        ckpt_parallel_save_within_dp (bool): If true, save will be parallelized only within a DP group
            (whole world otherwise), which might slightly reduce the save overhead. Defaults to False.
        ckpt_parallel_load (bool): If true, each worker will load part of the dist checkpoint
            and exchange with NCCL. Might use some extra GPU memory. Defaults to True.
        ckpt_parallel_save_optim (bool): Parallel save/load of a DistributedOptimizer. 'True'
            allows performant save and reshardable checkpoints. Set to 'False' only in order to minimize
            the number of checkpoint files.
        ckpt_load_directly_on_device (bool): if True, loads the weights directly on GPU.
            Has effect only for `zarr` based checkpoints (PyT Distributed always loads on device).
            Defaults to True.
        ckpt_load_strictness (StrictHandling, optional): defines loading strictness.
            If not None, overwrites the `strict` flag passed to `load_checkpoint`.
            Defaults to None. For a list of supported values, refer to the Megatron Core documentation:
            https://github.com/NVIDIA/Megatron-LM/blob/d4e72c0d33edc0c53aeb624f617eb77cebce6ae9/megatron/core/dist_checkpointing/validation.py#L46
        ckpt_optim_fully_reshardable (bool, optional): switches to a fully reshardable (TP/PP/DP/EP)
            optimizer format. Defaults to False, in which case a DP-only reshardable format is used.
        distrib_optim_fully_reshardable_mem_efficient (bool, optional): minimizes CUDA and host memory
            impact when using a fully reshardable format (`ckpt_optim_fully_reshardable=True`). Enable
            only if experiencing memory problems. Defaults to False.
        setup_optimizers (bool): Whether to call the trainer's setup_optimizers function to perform any
            necessary conversions of optimizer parameters and move optimizer parameters to the correct device.
            Defaults to True.
        init_model_parallel (bool): Whether to initialize the model parallel groups. Defaults to True.
        replace_progress_bar (bool): Whether to replace the TQDM progress bar with a megatron-style logger
            that prints the metrics to stdout. Suitable for non-interactive settings.
        progress_interval (int): How frequently to print progress to stdout. Only used when
            replace_progress_bar is True.
        megatron_log_level (int): Granularity level to measure and report timing.
            0: report only iteration time and make sure timing does not introduce extra overhead.
            1: report timing for operations that are executed very limited times (basically once) during
               each iteration (such as gradient all-reduce)
            2: report timing for operations that migh be executed numerous times during each iteration.
            Note that setting the level to 1 or 2 might cause increase in iteration time.
        use_tp_pp_dp_mapping (bool): Whether to use TP-PP-DP mapping instead of TP-DP-PP mapping.
            Defaults to False.
        num_distributed_optimizer_instances (int): The number of distributed optimizer replicas across
            the data-parallel domain.
        nccl_communicator_config_path (Optional[str]): Path to the yaml file of NCCL communicator configurations.
            `min_ctas`, `max_ctas`, and `cga_cluster_size` can be set for each communicator.
        use_sharp (bool): Whether to use SHARP. Defaults to False.
        pipeline_model_parallel_layout (Optional[Union[str, List[List[str]]]]): The layout of all layers among
            different PP and VP stages.
        use_gloo_process_groups (bool): Whether to use Gloo process groups. Defaults to True.
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
        pipeline_model_parallel_comm_backend: str = None,
        num_layers_in_first_pipeline_stage: Optional[int] = None,
        num_layers_in_last_pipeline_stage: Optional[int] = None,
        virtual_pipeline_model_parallel_size: Optional[int] = None,
        microbatch_group_size_per_vp_stage: Optional[int] = None,
        context_parallel_size: int = 1,
        sequence_parallel: bool = False,
        expert_model_parallel_size: int = 1,
        moe_extended_tp: bool = False,
        expert_tensor_parallel_size: Optional[int] = None,
        encoder_tensor_model_parallel_size: Optional[int] = 0,
        encoder_pipeline_model_parallel_size: Optional[int] = 0,
        account_for_embedding_in_pipeline_split: bool = False,
        account_for_loss_in_pipeline_split: bool = False,
        data_sampler: Optional["DataSampler"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment=None,  # TODO: Add type-hint
        checkpoint_io=None,  # TODO: Add type-hint
        find_unused_parameters: bool = False,
        ckpt_load_optimizer: bool = True,
        ckpt_save_optimizer: bool = True,
        ckpt_load_main_params: bool = False,
        ddp: Union[DDPLiteral, DistributedDataParallelConfig] = "megatron",
        fsdp: Optional[FSDPLiteral] = None,
        lazy_init: bool = False,
        pipeline_dtype: Optional[torch.dtype] = None,
        use_te_rng_tracker: bool = False,
        use_sharp: bool = False,
        save_ckpt_format: str = "torch_dist",
        ckpt_async_save: bool = True,
        ckpt_torch_dist_multiproc: int = None,  ## TODO(ashors): put elsewhere?
        ckpt_assume_constant_structure: bool = False,
        ckpt_parallel_save: bool = True,
        ckpt_parallel_save_within_dp: bool = False,
        ckpt_parallel_load: bool = True,
        ckpt_parallel_save_optim: Optional[bool] = None,
        ckpt_load_directly_on_device: bool = True,
        ckpt_load_strictness: Optional['StrictHandling'] = None,
        ckpt_optim_fully_reshardable: bool = False,
        distrib_optim_fully_reshardable_mem_efficient: bool = False,
        setup_optimizers: bool = True,
        init_model_parallel: bool = True,
        replace_progress_bar: bool = True,
        progress_interval: int = 1,
        restore_config: Optional[RestoreConfig] = None,
        megatron_log_level: int = 0,
        use_tp_pp_dp_mapping: bool = False,
        num_distributed_optimizer_instances: int = 1,
        nccl_communicator_config_path: Optional[str] = None,
        pipeline_model_parallel_layout: Optional[Union[str, List[List[str]]]] = None,
        use_gloo_process_groups: bool = True,
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
        self.data_sampler: Optional["DataSampler"] = data_sampler
        self.tensor_model_parallel_size = tensor_model_parallel_size
        self.pipeline_model_parallel_size = pipeline_model_parallel_size
        self.pipeline_model_parallel_comm_backend = pipeline_model_parallel_comm_backend
        self.num_layers_in_first_pipeline_stage = num_layers_in_first_pipeline_stage
        self.num_layers_in_last_pipeline_stage = num_layers_in_last_pipeline_stage
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
        self.account_for_embedding_in_pipeline_split = account_for_embedding_in_pipeline_split
        self.account_for_loss_in_pipeline_split = account_for_loss_in_pipeline_split
        self.lazy_init = lazy_init
        self.ckpt_load_optimizer = ckpt_load_optimizer
        self.ckpt_save_optimizer = ckpt_save_optimizer
        self.ckpt_load_main_params = ckpt_load_main_params
        self.ckpt_load_strictness = ckpt_load_strictness
        self.ckpt_optim_fully_reshardable = ckpt_optim_fully_reshardable
        self.distrib_optim_fully_reshardable_mem_efficient = distrib_optim_fully_reshardable_mem_efficient
        self.use_te_rng_tracker = use_te_rng_tracker
        self.use_sharp = use_sharp
        self._pipeline_dtype = pipeline_dtype
        self._setup_optimizers = setup_optimizers
        self._init_model_parallel = init_model_parallel
        self.log_train_loss = bool(int(os.getenv("NEMO_LOG_TRAIN_LOSS", 1)))
        self.log_memory_usage = bool(int(os.getenv("NEMO_LOG_MEMORY_USAGE", 0)))
        self.use_tp_pp_dp_mapping = use_tp_pp_dp_mapping
        self.save_ckpt_format = save_ckpt_format
        self.async_save = ckpt_async_save
        self.torch_dist_multiproc = ckpt_torch_dist_multiproc
        self.assume_constant_structure = ckpt_assume_constant_structure
        self.parallel_save = ckpt_parallel_save
        self.parallel_save_within_dp = ckpt_parallel_save_within_dp
        self.parallel_load = ckpt_parallel_load
        self.parallel_save_optim = ckpt_parallel_save_optim
        self.load_directly_on_device = ckpt_load_directly_on_device
        self.num_distributed_optimizer_instances = num_distributed_optimizer_instances
        self.replace_progress_bar = replace_progress_bar
        self.progress_interval = progress_interval
        self.nccl_communicator_config_path = nccl_communicator_config_path
        self.use_gloo_process_groups = use_gloo_process_groups
        self.restore_config = restore_config
        self.timers = Timers(megatron_log_level, "minmax")  ## could also set this for optimizer if we want
        self.pipeline_model_parallel_layout = pipeline_model_parallel_layout

        self._ddp = ddp
        if ddp == "megatron":
            self.ddp_config = DistributedDataParallelConfig(check_for_nan_in_grad=True)
        elif isinstance(ddp, DistributedDataParallelConfig):
            self.ddp_config = ddp
        elif ddp == "pytorch":
            if fsdp is not None:
                raise ValueError("Please set ddp to megatron to use FSDP.")
            self.ddp_config = None
            self.no_ddp_communication_hook = False
        else:
            raise ValueError(f"Invalid DDP type: {ddp}")

        self._fsdp = None

        use_custom_fsdp = getattr(self.ddp_config, "use_custom_fsdp", False)
        use_megatron_fsdp = getattr(self.ddp_config, "use_megatron_fsdp", False)
        if fsdp is None and self.ddp_config and (use_custom_fsdp or use_megatron_fsdp):
            logging.warning(
                "FSDP option is not set but ddp_config use megatron-fsdp is set to true. "
                "Setting FSDP option to megatron"
            )
            fsdp = 'megatron'
            if use_megatron_fsdp and self.save_ckpt_format != "fsdp_dtensor":
                raise NotImplementedError(
                    f"Megatron-FSDP checkpointing is not supported with {self.save_ckpt_format}."
                )

        if fsdp == "pytorch":
            raise NotImplementedError("PyTorch FSDP2 is not supported with MegatronParallel.")
        elif fsdp == "megatron":
            self._fsdp = fsdp
            if hasattr(self.ddp_config, "use_custom_fsdp") and not use_custom_fsdp:
                self.ddp_config.use_custom_fsdp = True
                logging.warning("Setting ddp_config.use_custom_fsdp to True for MCore FSDP.")
                logging.warning(
                    "Deprecation Notice: `use_custom_fsdp` will be deprecated in M-Core 0.14. "
                    "Please use `use_megatron_fsdp` instead."
                )
            elif hasattr(self.ddp_config, "use_megatron_fsdp") and not use_megatron_fsdp:
                self.ddp_config.use_megatron_fsdp = True
                logging.warning("Setting ddp_config.use_megatron_fsdp to True for MCore FSDP.")
            logging.info("FSDP option is set to MCore. Using MCore's Custom FSDP for DP.")
        elif fsdp is not None:
            raise ValueError(f'Invalid DDP type: {fsdp}, please choose from ["megatron", "pytorch"].')

        if ddp == "megatron":
            self.ddp_config = DistributedDataParallelConfig(check_for_nan_in_grad=True)
        elif isinstance(ddp, DistributedDataParallelConfig):
            self.ddp_config = ddp
        elif ddp == "pytorch":
            if self._fsdp is not None:
                raise ValueError("Please set ddp to megatron to use FSDP.")
            self.ddp_config = None
            self.no_ddp_communication_hook = False
        else:
            raise ValueError(f"Invalid DDP type: {ddp}")

        if self.ckpt_load_optimizer and self.ckpt_load_main_params:
            raise ValueError("ckpt_load_optimizer and ckpt_load_main_params cannot be both set to True.")

        if self.parallel_save_optim is not None:
            logging.warning(
                "`ckpt_parallel_save_optim` argument is replaced with"
                " `ckpt_optim_fully_reshardable` and does not have any effect"
            )

        if isinstance(self.ddp_config, DistributedDataParallelConfig):
            self.ddp_config.num_distributed_optimizer_instances = self.num_distributed_optimizer_instances

        # used in NVIDIA NGC PyTorch containers
        _strategy_lib.enable_nvidia_optimizations()

        self.store: Optional[torch.distributed.Store] = None

    @property
    def pipeline_dtype(self):
        """ """
        if self._pipeline_dtype is None:
            dtype_config = getattr(self._precision_plugin, "dtype_config", None)
            if dtype_config is not None:
                self._pipeline_dtype = dtype_config.pipeline_dtype
        return self._pipeline_dtype

    @pipeline_dtype.setter
    def pipeline_dtype(self, value):
        self._pipeline_dtype = value

    @override
    def connect(self, model: pl.LightningModule) -> None:
        """Attaches a model to strategy."""
        super().connect(model)

        assert not 'is_hf_model' in model.__dict__, "Cannot use HFAutoModelForCausalLM with MegatronParallel"

        dtype_config = getattr(self._precision_plugin, "dtype_config", None)
        if self.pipeline_dtype is None and dtype_config:
            self.pipeline_dtype = dtype_config.pipeline_dtype

        _maybe_mcore_config = _strategy_lib.set_model_parallel_attributes(model, self.parallelism)
        if _maybe_mcore_config:
            self._mcore_config = _maybe_mcore_config

        if dtype_config:
            from nemo.lightning.pytorch.plugins.mixed_precision import update_config_with_dtype_overrides

            model.config = update_config_with_dtype_overrides(dtype_config, model.config)

        # add megatron timer to config
        if hasattr(model, "config"):
            model.config.timers = self.timers

        has_optim = getattr(model, "optim", None)
        if has_optim and self._setup_optimizers:
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
                    logging.info("Fixing mis-match between ddp-config & mcore-optimizer config")
                    ddp_config.use_distributed_optimizer = mcore_opt_config.use_distributed_optimizer

    @override
    def setup(self, trainer: pl.Trainer) -> None:
        """Setups the strategy"""
        assert self.accelerator is not None
        self.accelerator.setup(trainer)
        self.trainer = trainer

        try:
            self.model.optim.lr_scheduler.max_steps = trainer.max_steps
            logging.info(f"Copying Trainer's 'max_steps' ({trainer.max_steps}) to LR scheduler's 'max_steps'.")
        except AttributeError:
            logging.warning(
                "Could not copy Trainer's 'max_steps' to LR scheduler's 'max_steps'. "
                "If you are not using an LR scheduler, this warning can safely be ignored."
            )

        # move the model to the correct device
        # self.model_to_device()

        # skip wrapping the model if we are not fitting as no gradients need to be exchanged
        trainer_fn = trainer.state.fn

        if trainer_fn == TrainerFn.FITTING and self._layer_sync:
            assert self.model is not None
            self.model = self._layer_sync.apply(self.model)

        setup_data_sampler(self.trainer)
        fix_progress_bar(trainer, self.replace_progress_bar, self.progress_interval)

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

            def broadcast_params(module):
                """
                Modified from
                https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/distributed_data_parallel.py#L466-L483
                Syncs parameters across all DP ranks.
                """
                from megatron.core import parallel_state

                for param in module.parameters():
                    is_expert_parallel = not getattr(param, 'allreduce', True)

                    if is_expert_parallel:
                        data_parallel_group = parallel_state.get_expert_data_parallel_group()
                    else:
                        data_parallel_group = parallel_state.get_data_parallel_group(
                            with_context_parallel=True, partial_data_parallel=True
                        )
                    torch.distributed.broadcast(
                        param.data,
                        src=torch.distributed.get_global_rank(data_parallel_group, 0),
                        group=data_parallel_group,
                    )

            for model_module in self.model:
                broadcast_params(model_module)

        # add AsyncFinalizerCallback if using async
        if self.async_save:
            have_async_callback = False
            for callback in self.trainer.callbacks:
                if isinstance(callback, AsyncFinalizerCallback):
                    have_async_callback = True
                    break
            if not have_async_callback:
                self.trainer.callbacks.append(AsyncFinalizerCallback())

        # Restore model weights and optimizer states if needed
        if self.restore_config and not self.trainer.ckpt_path:
            self.selective_restore()

    @override
    def setup_distributed(self) -> None:
        """Setups dist env"""
        setup_parallel_ranks(self)

        # Implementation from superclass copied below in order to pass the store to the process group init
        reset_seed()
        self.set_world_ranks()
        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None

        if not torch.distributed.is_available():
            raise RuntimeError("torch.distributed is not available. Cannot initialize distributed process group")
        if torch.distributed.is_initialized():
            _logger.debug("torch.distributed is already initialized. Exiting early")
            return

        global_rank = self.cluster_environment.global_rank()
        world_size = self.cluster_environment.world_size()
        os.environ["MASTER_ADDR"] = self.cluster_environment.main_address
        os.environ["MASTER_PORT"] = str(self.cluster_environment.main_port)
        if timeout := os.getenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"):
            timeout = timedelta(seconds=int(timeout))
        _logger.info(f"Initializing distributed: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}")
        torch.distributed.init_process_group(
            self._process_group_backend, rank=global_rank, world_size=world_size, store=self.store, timeout=timeout
        )

        if self._process_group_backend == "nccl":
            atexit.register(_destroy_dist_connection)

        # On rank=0 let everyone know training is starting
        rank_zero_info(
            f"{'-' * 100}\n"
            f"distributed_backend={self._process_group_backend}\n"
            f"All distributed processes registered. Starting with {world_size} processes\n"
            f"{'-' * 100}\n"
        )
        init_model_parallel(self.model)

        if self.data_sampler:
            assert isinstance(self.cluster_environment, ClusterEnvironment), "Cluster environment not initialized"
            self.data_sampler.setup(self.cluster_environment.global_rank())

    @override
    def process_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """Setups dataloader"""
        if self.data_sampler:
            return self.data_sampler.transform_dataloader(dataloader)

        return dataloader

    def setup_megatron_parallel(self, trainer: pl.Trainer) -> None:
        """Configures megatron parallel"""
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
            fsdp=self._fsdp,
            convert_module_fn=convert_module_fn,
        )

        # Assign trainer to megatron_parallel before init_model_parallel as its required to check stage of trainer
        # (TESTING or not) in init_model_parallel.
        self.megatron_parallel.trainer = trainer

        if self._init_model_parallel:
            self.init_model_parallel()

        # check signature-def of self.model.configure_optimizers to check if there's an optional arg: megatron_parallel
        sig = inspect.signature(self.model.configure_optimizers)
        if "megatron_parallel" in sig.parameters:
            self.model.configure_optimizers = functools.partial(
                self.model.configure_optimizers, megatron_parallel=self.megatron_parallel
            )

        if self._setup_optimizers:
            self.setup_optimizers(trainer)

        self.model = self.megatron_parallel
        trainer_callbacks = getattr(trainer, "callbacks", None)
        if trainer_callbacks:
            self.model.callbacks.add(*trainer_callbacks)

        if self.data_sampler:
            self.model.callbacks.add(self.data_sampler)

        datamodule = getattr(trainer, "datamodule", None)
        if datamodule:
            self.model.callbacks.add(datamodule)

    def init_model_parallel(self):
        """Initializes megatron parallel"""
        self.megatron_parallel.init_model_parallel()

    @override
    def configure_ddp(self) -> None:
        """Configures ddp"""
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
        """Setups optimizers"""
        super().setup_optimizers(trainer)
        if hasattr(self.precision_plugin, "convert_optimizer"):
            _optimizers = [*self.optimizers]
            _optimizers[0] = self.precision_plugin.convert_optimizer(self.optimizers[0])
            self.optimizers = _optimizers

        _optimizers_to_device(self.optimizers, self.root_device)

    @override
    def on_validation_end(self) -> None:
        "Runs after validation loop"
        for model_chunk in self.model:
            model_chunk.train()

    @override
    def on_test_end(self) -> None:
        "Runs after test loop"
        for model_chunk in self.model:
            model_chunk.train()

    @override
    def training_step(self, dataloader_iter, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        """Runs one training step"""
        assert self.lightning_module is not None
        assert isinstance(self.model, MegatronParallel)

        with self.precision_plugin.train_step_context():  # TODO: Do we need this?
            # Set grad to zero.
            for model_chunk in self.model:
                model_chunk.zero_grad_buffer()
            for opt in self.optimizers:
                opt.zero_grad()

            out = self.model.training_step(dataloader_iter, *args, **kwargs)

            if torch.is_tensor(out):
                reduced_train_loss = out
            else:
                if not isinstance(out, dict):
                    raise ValueError(f"Expected dict or tensor for reduced_train_loss, got {type(out)}")

                if "loss" not in out:
                    raise ValueError(f"Expected 'loss' in output dict, got {out.keys()}")

                reduced_train_loss = out["loss"]
            # pl_module.log () will trigger pageable H2D Memcpy which stalls CPU. Use pin_memory=True to avoid it
            global_step = (
                self.trainer.global_step
                if (torch.is_tensor(self.trainer.global_step) and self.trainer.global_step.is_cuda)
                else torch.tensor(self.trainer.global_step, pin_memory=True).to("cuda", non_blocking=True)
            )
            self.lightning_module.log(
                "global_step",
                global_step,
                prog_bar=True,
                batch_size=1,
            )

            self.lightning_module.log(
                "step",
                global_step,
            )

            if self.log_memory_usage:
                # maximum GPU memory that has been managed by the caching allocator
                max_memory_reserved = torch.cuda.max_memory_reserved()
                max_memory_reserved = (
                    max_memory_reserved
                    if (torch.is_tensor(max_memory_reserved) and max_memory_reserved.is_cuda)
                    else torch.tensor(max_memory_reserved, pin_memory=True).to("cuda", non_blocking=True)
                )
                # maximum GPU memory that has been occupied by active tensors
                max_memory_allocated = torch.cuda.max_memory_allocated()
                max_memory_allocated = (
                    max_memory_allocated
                    if (torch.is_tensor(max_memory_allocated) and max_memory_allocated.is_cuda)
                    else torch.tensor(max_memory_allocated, pin_memory=True).to("cuda", non_blocking=True)
                )
                self.lightning_module.log(
                    "max_memory_reserved",
                    max_memory_reserved,
                    prog_bar=True,
                    batch_size=1,
                )
                self.lightning_module.log(
                    "max_memory_allocated",
                    max_memory_allocated,
                    prog_bar=True,
                    batch_size=1,
                )

            if self.log_train_loss:
                # p2p now, broadcast later at ckpt. only with pp, some ranks will log 0.0
                # WHICH IS OK because we broadcast later at checkpoint time
                _strategy_lib._sync_from_last_pipeline_stage(reduced_train_loss, broadcast=False)
                reduced_train_loss = (
                    reduced_train_loss
                    if (torch.is_tensor(reduced_train_loss) and reduced_train_loss.is_cuda)
                    else torch.tensor(reduced_train_loss, pin_memory=True).to("cuda", non_blocking=True)
                )
                self.lightning_module.log(
                    "reduced_train_loss", reduced_train_loss, prog_bar=True, batch_size=1, sync_dist=False
                )
                # Log any MoE losses.
                # @akoumparouli: disabling this as it hangs with deepseek.
                # TODO(@akoumparouli): loss_scale depends on the GBS.
            # for loss_name, loss_value in aggregate_moe_loss_stats(loss_scale=1.0).items():
            #    self.lightning_module.log(
            #    loss_name, loss_value, prog_bar=True, rank_zero_only=True, batch_size=1)

            return out

    @override
    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        closure: Callable[[], Any],
        model: Optional[Union["pl.LightningModule", nn.Module]] = None,
        **kwargs: Any,
    ) -> Any:
        """Runs one optimizer step"""
        optimizer_output = super().optimizer_step(optimizer, closure, model, **kwargs)

        if isinstance(optimizer, McoreDistributedOptimizer):
            optimizer_output, grad_norm, num_zeros_in_grad = optimizer_output
            if grad_norm is not None:
                grad_norm = (
                    grad_norm
                    if (torch.is_tensor(grad_norm) and grad_norm.is_cuda)
                    else torch.tensor(grad_norm, pin_memory=True).to("cuda", non_blocking=True)
                )
                self.lightning_module.log('grad_norm', grad_norm, batch_size=1)
            if num_zeros_in_grad is not None:
                num_zeros_in_grad = (
                    num_zeros_in_grad
                    if (torch.is_tensor(num_zeros_in_grad) and num_zeros_in_grad.is_cuda)
                    else torch.tensor(num_zeros_in_grad, pin_memory=True).to("cuda", non_blocking=True)
                )
                self.lightning_module.log('num_zeros_in_grad', num_zeros_in_grad, batch_size=1)

        return optimizer_output

    @override
    def validation_step(self, dataloader_iter, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        """Runs one validation step"""
        assert self.lightning_module is not None
        assert isinstance(self.model, MegatronParallel)

        with self.precision_plugin.val_step_context():  # TODO: Do we need this?
            out = self.model.validation_step(dataloader_iter, *args, **kwargs)

            from megatron.core import parallel_state

            pp_size = parallel_state.get_pipeline_model_parallel_world_size()
            if pp_size > 1:
                # ranks that are not final pp stage have 0 for loss, and out will be mean-reduced over pp
                # groups (due to sync_dist), which divides val_loss by pp_size. so we multiply by pp_size to cancel out
                self.lightning_module.log(
                    "val_loss",
                    out * pp_size,
                    prog_bar=True,
                    sync_dist=True,
                    sync_dist_group=parallel_state.get_pipeline_model_parallel_group(),
                    on_epoch=True,
                )
            else:
                self.lightning_module.log("val_loss", out, prog_bar=True, on_epoch=True)

            return out

    @override
    def test_step(self, dataloader_iter, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        """Runs one test step"""
        assert self.lightning_module is not None
        assert isinstance(self.model, MegatronParallel)

        with self.precision_plugin.test_step_context():  # TODO: Do we need this?
            return self.model.test_step(dataloader_iter, *args, **kwargs)

    @override
    def predict_step(self, dataloader_iter, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        """Runs one prediction step"""
        assert self.lightning_module is not None
        assert isinstance(self.model, MegatronParallel)

        with self.precision_plugin.predict_step_context():  # TODO: Do we need this?
            return self.model.predict_step(dataloader_iter, *args, **kwargs)

    @override
    def teardown(self) -> None:
        """Tearsdown the strategy"""
        if hasattr(self, "megatron_parallel"):
            self.megatron_parallel.teardown_ddp()
        super().teardown()

    @override
    def model_sharded_context(self) -> ContextManager:
        """Model sharded context"""
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

        return kwargs

    def optimizer_sharded_state_dict(self, is_loading: bool = False, metadata: Optional[dict] = None):
        """
        Sharded state dictionary for an MainParamsOptimizerWrapper.
        Used to save and load the optimizer state when training with distributed_checkpoint.

        Args:
            is_loading (bool, optional): set to True if the sharded state dict is intended
                for checkpoint loading (as opposed to saving). Defaults to False.
            metadata (dict, optional): sharded state dict metadata passed from the framework.
                Used to control the details of sharded state dict creation, in particular
                the state dict format of the DistributedOptimizer with the flag
                `distrib_optim_sharding_type`. Defaults to None (empty metadata).

        Returns
        -------
            dict: The sharded state dictionary for the optimizer
        Raises:
            ValueError: If a parameter ID does not match any model sharded parameter.
        """
        # TODO: Fix when MainParamsOptimizerWrapper is not used

        optimizer = self.lightning_module.optimizers(use_pl_optimizer=False)
        if metadata is None:
            metadata = self.sharded_state_dict_metadata
            logging.debug(
                f'No sharded_state_dict metadata passed for the optimizer,'
                f' using metadata for checkpoint save: {metadata}'
            )
        else:
            logging.debug(f'Using passed sharded_state_dict metadata in the optimizer: {metadata}')

        return _strategy_lib.optimizer_sharded_state_dict(
            self.megatron_parallel,
            optimizer,
            is_loading=is_loading,
            metadata=metadata,
        )

    def _get_fsdp_dtensor_state_dict(
        self,
        raw_state_dict,
        model_key="model",
        optimizer_key="optimizer_states",
    ):
        from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
            preprocess_state_dict_for_uneven_dtensor,
        )
        from megatron.core.transformer.fsdp_dtensor_checkpoint import (
            handle_fp8_extra_state_case,
            handle_swiglu_in_state_dict,
        )

        state_dict = raw_state_dict.copy()
        handle_fp8_extra_state_case(state_dict[model_key])
        module = self.model[0].module
        if getattr(module.config, "gated_linear_unit", False):
            model_state_dict = state_dict[model_key].copy()
            if optimizer_key in state_dict:
                optimizer_state_dict = state_dict[optimizer_key].copy()
            else:
                optimizer_state_dict = {}
            handle_swiglu_in_state_dict(module.module, model_state_dict, optimizer_state_dict)
            state_dict[model_key] = model_state_dict
            if optimizer_key in state_dict:
                state_dict[optimizer_key] = optimizer_state_dict
        preprocess_state_dict_for_uneven_dtensor(state_dict)

        return state_dict

    def _save_fsdp_dtensor_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path,
        storage_options,
    ):
        state_dict = self._get_fsdp_dtensor_state_dict(checkpoint)

        torch.distributed.checkpoint.save(
            state_dict,
            storage_writer=torch.distributed.checkpoint.FileSystemWriter(path),
        )
        self._save_fsdp_dtensor_common_state(state_dict=state_dict, ckpt_dir=path)

        if "finalize_fn" in storage_options:
            storage_options["finalize_fn"]()

    @override
    def save_checkpoint(
        self, checkpoint: Dict[str, Any], filepath: Union[str, Path], storage_options: Optional[Any] = None
    ) -> None:
        """Saves checkpoint"""
        from nemo.collections.llm.modelopt.model_utils import save_modelopt_state

        if (
            isinstance(self.ddp_config, DistributedDataParallelConfig)
            and self.ddp_config.use_distributed_optimizer
            and self.ddp_config.overlap_param_gather
        ):
            self.megatron_parallel.force_param_sync()

        checkpoint["state_dict"] = OrderedDict([])  # remove device state_dict
        # retrieve `sharded_state_dict` if it has not already been configured in `on_save_checkpoint`
        if "sharded_state_dict" not in checkpoint:
            checkpoint["sharded_state_dict"] = self.megatron_parallel.sharded_state_dict()

        if "optimizer_states" in checkpoint and self.trainer.state.fn == TrainerFn.FITTING:
            # Clear the optimizer states. This handles the case where ckpt_save_optimizer=False
            # Ideally, the optimizer state dicts should not be generated in this case
            checkpoint["optimizer_states"] = {}

            # replace unsharded optimizer_states with sharded dict.
            # note that if trainer.save_checkpoint(path, save_weights_only=True) is called,
            # the checkpoint will contain only model weights. Optimizer states will be omitted.
            if self.ckpt_save_optimizer:
                checkpoint["optimizer"] = [self.optimizer_sharded_state_dict()]

        if not storage_options:
            storage_options = {}
        storage_options['content_metadata'] = self.sharded_state_dict_metadata
        if self.save_ckpt_format == "fsdp_dtensor":
            checkpoint = checkpoint.copy()
            if "optimizer" in checkpoint:
                checkpoint["optimizer_states"] = checkpoint.pop("optimizer")[0]
            checkpoint["model"] = checkpoint.pop("sharded_state_dict")
            self._save_fsdp_dtensor_checkpoint(
                checkpoint=checkpoint,
                path=ckpt_to_dir(filepath),
                storage_options=storage_options,
            )
            checkpoint_io = None
        else:
            self.checkpoint_io.save_checkpoint(checkpoint, filepath, storage_options=storage_options)
            checkpoint_io = self.checkpoint_io

        # Save ModelOpt state too, if it exists.
        save_modelopt_state(self.megatron_parallel, filepath, checkpoint_io)

    def should_restore_optimizer_states(self, selective_restore: bool = False) -> bool:
        """Determines whether to restore optimizer states or not"""
        if selective_restore:
            return self.restore_config.load_optim_state if self.restore_config else False

        return self.ckpt_load_optimizer

    def _save_fsdp_dtensor_common_state(self, state_dict, ckpt_dir):
        state_dict = state_dict.copy()
        del state_dict["model"]
        del state_dict["optimizer_states"]
        torch.save(state_dict, os.path.join(ckpt_dir, "common.pt"))

    def _load_fsdp_dtensor_common_state(self, ckpt_dir):
        # Note: set to weights_only=True for safety reason
        # It should be switching to safetensor but NeMo is deprecated.
        # Please use Megatron-Bridge for better checkpointing support.
        return torch.load(os.path.join(ckpt_dir, "common.pt"), weights_only=True)

    def _load_fsdp_dtensor_checkpoint(self, path, sharded_state_dict, strict):
        from torch.distributed.checkpoint import default_planner

        state_dict = self._get_fsdp_dtensor_state_dict(sharded_state_dict)

        planner = default_planner.DefaultLoadPlanner(allow_partial_load=not strict)
        torch.distributed.checkpoint.load(
            state_dict,
            checkpoint_id=path,
            planner=planner,
        )
        sharded_state_dict.update(self._load_fsdp_dtensor_common_state(ckpt_dir=path))
        if "loops" in sharded_state_dict:
            sharded_state_dict["fit_loop"] = sharded_state_dict["loops"]["fit_loop"]

        return sharded_state_dict

    @override
    def load_checkpoint(self, checkpoint_path: Union[str, Path], selective_restore: bool = False) -> Dict[str, Any]:
        """PTL method which we override to integrate distributed checkpoints for model parallel models.
        In order to load distributed checkpoints we need to provide the sharded_state_dict to
        the distributed load function. We get the sharded_state_dict from self.lightning_module
        which makes it convenient to have the loading logic happen at the strategy level.
        """
        from nemo.collections.llm.modelopt.model_utils import restore_modelopt_state

        torch.cuda.empty_cache()

        restore_optimizers = self.should_restore_optimizer_states(selective_restore=selective_restore)

        # Restore ModelOpt state if it exists.
        sharded_state_context = restore_modelopt_state(self.megatron_parallel, checkpoint_path)
        if sharded_state_context is None or restore_optimizers or self.trainer.state.fn != TrainerFn.FITTING:
            sharded_state_context = nullcontext

        # After dist_checkpointing.load, sharded tensors will be replaced with tensors
        if self.save_ckpt_format == "fsdp_dtensor":
            sharded_sd_metadata = self.sharded_state_dict_metadata
        else:
            sharded_sd_metadata = self.unwrapped_checkpoint_io.load_content_metadata(checkpoint_path)

        sharded_state_dict = {}
        with sharded_state_context():
            sharded_state_dict["state_dict"] = self.megatron_parallel.sharded_state_dict(metadata=sharded_sd_metadata)

        if restore_optimizers and self.trainer.state.fn == TrainerFn.FITTING:
            if self.lightning_module.optimizers(use_pl_optimizer=False):
                sharded_state_dict["optimizer"] = [
                    self.optimizer_sharded_state_dict(is_loading=True, metadata=sharded_sd_metadata)
                ]

        strict = (
            self.lightning_module.strict_loading if self.ckpt_load_strictness is None else self.ckpt_load_strictness
        )

        try:
            if self.save_ckpt_format == "fsdp_dtensor":
                sharded_state_dict["model"] = sharded_state_dict.pop("state_dict")
                if "optimizer" in sharded_state_dict:
                    sharded_state_dict["optimizer_states"] = sharded_state_dict.pop("optimizer")[0]
                checkpoint = self._load_fsdp_dtensor_checkpoint(
                    path=ckpt_to_dir(checkpoint_path),
                    sharded_state_dict=sharded_state_dict,
                    strict=strict,
                )
            else:
                checkpoint = self.checkpoint_io.load_checkpoint(
                    checkpoint_path, sharded_state_dict=sharded_state_dict, strict=strict
                )
        except CheckpointException as e:
            if (
                "optimizer.distributed.dp_group_idx_" in str(e)
                and sharded_sd_metadata.get("distrib_optim_sharding_type") == "dp_reshardable"
            ):
                error_message = f"{e}\n{RESHARDING_LOAD_ERROR}"
            else:
                error_message = f"{e}\n{LOAD_ERROR}"
            raise RuntimeError(error_message)

        if selective_restore:
            final_checkpoint = {}
            for key in sharded_state_dict.keys():
                final_checkpoint[key] = checkpoint[key]

            return final_checkpoint

        return checkpoint

    @property
    def sharded_state_dict_metadata(self):
        """Metadata used for sharded_state_dict generation during checkpoint save."""
        metadata = {}
        metadata['singleton_local_shards'] = False
        metadata['chained_optim_avoid_prefix'] = True

        use_distributed_optimizer = (
            isinstance(self.ddp_config, DistributedDataParallelConfig) and self.ddp_config.use_distributed_optimizer
        )
        use_megatron_fsdp = getattr(self.ddp_config, "use_megatron_fsdp", False)
        if use_distributed_optimizer and use_megatron_fsdp:
            metadata["distrib_optim_sharding_type"] = "fsdp_dtensor"

        if use_distributed_optimizer and not use_megatron_fsdp:
            if self.ckpt_optim_fully_reshardable:
                metadata['distrib_optim_sharding_type'] = 'fully_reshardable'
                metadata['distrib_optim_fully_reshardable_mem_efficient'] = (
                    self.distrib_optim_fully_reshardable_mem_efficient
                )
            else:
                metadata['distrib_optim_sharding_type'] = 'dp_reshardable'
        return metadata

    def selective_restore(self) -> None:
        """Implements selective restoration of checkpoint"""
        if not self.restore_config:
            return

        logging.info(f"Doing selective restore from {self.restore_config}")

        checkpoint = self.load_checkpoint(checkpoint_path=self.restore_config.path, selective_restore=True)

        if self.restore_config.load_model_state:
            logging.info(f"Restoring model weights from {self.restore_config}")
            strict = True if self.ckpt_load_strictness is None else self.ckpt_load_strictness
            self.load_model_state_dict(checkpoint=checkpoint, strict=strict)

        if self.restore_config.load_optim_state:
            logging.info(f"Restoring optimizer states from {self.restore_config}")
            self.load_optimizer_state_dict(checkpoint=checkpoint, selective_restore=True)

        logging.info(f"Finished restoring from {self.restore_config}, cleaning up.")
        torch.cuda.empty_cache()
        # wait for all to catch up
        self.trainer.strategy.barrier("MegatronStrategy.restore_end")

    @override
    def load_optimizer_state_dict(self, checkpoint: Mapping[str, Any], selective_restore: bool = False) -> None:
        """Loads optimizer state-dict"""
        if not self.should_restore_optimizer_states(selective_restore=selective_restore):
            return

        from megatron.core import parallel_state
        from torch.distributed import DeviceMesh
        from torch.distributed._tensor import DTensor, Shard

        mesh = DeviceMesh.from_group(parallel_state.get_data_parallel_group(), "cuda")

        if self.save_ckpt_format == "fsdp_dtensor":
            assert len(self.optimizers) == 1, "FSDP DTensor format requires a single optimizer."
            self.optimizers[0].load_state_dict(checkpoint["optimizer_states"])
            return

        optimizer_states = checkpoint["optimizer"]
        for optimizer, opt_state in zip(self.optimizers, optimizer_states):
            if self._fsdp is not None:
                opt_state['fp32_from_fp16_params'] = OrderedDict()
                for opt_param in opt_state['optimizer']['state'].values():
                    if isinstance(opt_param, Dict):
                        for opt_param_state_key, opt_param_state in opt_param.items():
                            opt_param[opt_param_state_key] = DTensor.from_local(
                                opt_param_state,
                                mesh,
                                (Shard(dim=0),),
                            )

            optimizer.load_state_dict(opt_state)
            _optimizer_to_device(optimizer, self.root_device)

    def remove_checkpoint(self, filepath: Union[str, Path]) -> None:
        """Deletes checkpoint"""
        ckpt = ckpt_to_dir(filepath)
        if self.is_global_zero:
            if os.path.islink(ckpt):
                os.unlink(ckpt)
            else:
                shutil.rmtree(ckpt)

    def load_model_state_dict(self, checkpoint: Mapping[str, Any], strict: bool = True) -> None:
        """loads model state dict"""
        assert self.megatron_parallel is not None

        strict = strict if self.ckpt_load_strictness is None else self.ckpt_load_strictness
        _strategy_lib.load_model_state_dict(self.megatron_parallel, checkpoint, strict=strict)

        if not 'optimizer' in checkpoint:
            for opt in self.optimizers:
                if self.ckpt_load_main_params:
                    if "state_dict" in checkpoint:
                        opt.reload_model_params(checkpoint["state_dict"])
                    else:
                        opt.reload_model_params(checkpoint)
                else:
                    opt.reload_model_params()

    @property
    @override
    def checkpoint_io(self) -> CheckpointIO:
        """Creates & returns checkpoint io"""
        if not self._checkpoint_io:
            self._checkpoint_io = create_checkpoint_io(
                save_ckpt_format=self.save_ckpt_format,
                async_save=self.async_save,
                torch_dist_multiproc=self.torch_dist_multiproc,
                assume_constant_structure=self.assume_constant_structure,
                parallel_save=self.parallel_save,
                parallel_save_within_dp=self.parallel_save_within_dp,
                parallel_load=self.parallel_load,
                load_directly_on_device=self.load_directly_on_device,
            )

        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, io: CheckpointIO) -> None:
        """CheckpointIO setter"""
        self._checkpoint_io = io

    @property
    def unwrapped_checkpoint_io(self) -> CheckpointIO:
        """Unwraps `checkpoint_io` from all wrappers."""
        checkpoint_io = self.checkpoint_io
        while isinstance(checkpoint_io, _WrappingCheckpointIO):
            checkpoint_io = checkpoint_io.checkpoint_io
        return checkpoint_io

    @property
    def current_epoch_step(self) -> int:
        """
        Get the value of step within an epoch.
        """
        return max(
            self.trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.optimizer.step.current.completed,
            self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.current.completed,
        )

    @property
    def distributed_sampler_kwargs(self) -> Dict[str, Any]:
        """Returns dist-sampler's kwargs"""
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
        """Returns parallelism config from class attrs as a POD"""
        return ParallelismConfig(
            tensor_model_parallel_size=self.tensor_model_parallel_size,
            pipeline_model_parallel_size=self.pipeline_model_parallel_size,
            pipeline_model_parallel_comm_backend=self.pipeline_model_parallel_comm_backend,
            num_layers_in_first_pipeline_stage=self.num_layers_in_first_pipeline_stage,
            num_layers_in_last_pipeline_stage=self.num_layers_in_last_pipeline_stage,
            virtual_pipeline_model_parallel_size=self.virtual_pipeline_model_parallel_size,
            microbatch_group_size_per_vp_stage=self.microbatch_group_size_per_vp_stage,
            context_parallel_size=self.context_parallel_size,
            sequence_parallel=self.sequence_parallel,
            expert_model_parallel_size=self.expert_model_parallel_size,
            expert_tensor_parallel_size=self.expert_tensor_parallel_size,
            moe_extended_tp=self.moe_extended_tp,
            encoder_tensor_model_parallel_size=self.encoder_tensor_model_parallel_size,
            encoder_pipeline_model_parallel_size=self.encoder_pipeline_model_parallel_size,
            account_for_embedding_in_pipeline_split=self.account_for_embedding_in_pipeline_split,
            account_for_loss_in_pipeline_split=self.account_for_loss_in_pipeline_split,
            pipeline_dtype=self.pipeline_dtype,
            use_te_rng_tracker=self.use_te_rng_tracker,
            use_tp_pp_dp_mapping=self.use_tp_pp_dp_mapping,
            num_distributed_optimizer_instances=self.num_distributed_optimizer_instances,
            nccl_communicator_config_path=self.nccl_communicator_config_path,
            use_sharp=self.use_sharp,
            pipeline_model_parallel_layout=self.pipeline_model_parallel_layout,
            use_gloo_process_groups=self.use_gloo_process_groups,
        )

    @contextmanager
    @override
    def tensor_init_context(self, empty_init: Optional[bool] = None):
        """Context manager used for initialization"""
        # Materializaton happens in `setup()`
        # @akoumparouli: using Parent's tensor_init_context causes mcore
        # parameters to be initialized on GPU instead of (assumed) CPU.
        yield


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
