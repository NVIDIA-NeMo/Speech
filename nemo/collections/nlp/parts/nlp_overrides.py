# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import itertools
import os
import shutil
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Literal, Mapping, Optional, Sized, Union

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.loops.fetchers import _DataFetcher
from pytorch_lightning.overrides.base import _LightningModuleWrapperBase
from pytorch_lightning.plugins import ClusterEnvironment
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.precision import MixedPrecisionPlugin
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks import noop_hook
from torch.nn.parallel import DistributedDataParallel

from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from nemo.core.optim import MainParamsOptimizerWrapper
from nemo.utils import AppState, logging
from nemo.utils.model_utils import inject_model_parallel_rank

try:
    from apex.transformer.enums import ModelType
    from apex.transformer.pipeline_parallel.schedules.common import _calc_number_of_params
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches
    from apex.transformer.tensor_parallel.layers import set_defaults_if_not_set_tensor_model_parallel_attributes

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE = "NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE"


class NLPDDPStrategy(DDPStrategy):
    """ DDP plugin for Pytorch Lightning. Needed to customize DDP for model parallel models.

    Args:
        no_ddp_communication_hook: Disable DDP communication hook when using AMP-O2
        with FP32 gradient accumulation.
    """

    def __init__(
        self,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: ClusterEnvironment = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        no_ddp_communication_hook: bool = False,
        **kwargs: Union[Any, Dict[str, Any]],
    ) -> None:
        if not HAVE_APEX:
            raise ImportError(
                "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )

        if not HAVE_MEGATRON_CORE:
            raise ImportError(
                "megatron-core was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        super().__init__(parallel_devices, cluster_environment, checkpoint_io, **kwargs)

        self.no_ddp_communication_hook = no_ddp_communication_hook

    def setup(self, trainer: "pl.Trainer") -> None:
        """
        Override setup() of DDPStrategy to avoid _sync_module_states(self.model) during eval as it can cause PP > 1 to hang
        due to assumption in DDPStrategy class that the same model is replicated across GPUs
        """
        trainer_fn = trainer.state.fn
        if trainer_fn == TrainerFn.FITTING:
            super().setup(trainer)
        else:
            assert self.accelerator is not None
            self.accelerator.setup(trainer)

            # move the model to the correct device
            self.model_to_device()
            self.setup_precision_plugin()
            assert self.model is not None

    def setup_distributed(self, global_rank: int = None, world_size: int = None) -> None:
        # call PTL init ddp
        super().setup_distributed()

        # init model parallel if needed
        if not parallel_state.model_parallel_is_initialized():
            app_state = AppState()

            if app_state.model_parallel_size is not None:
                self.init_model_parallel(app_state.global_rank, app_state.world_size)

    def configure_ddp(self):
        """ Override LightningModule ddp if using model parallel.
            Sets find_unused_parameters to False to use activation-checkpoint-recomputation.
        """

        if (hasattr(self.model, 'megatron_amp_o2') and self.model.megatron_amp_o2) or (
            hasattr(self.model, 'with_distributed_adam') and self.model.with_distributed_adam
        ):
            # do not use DDP if using megatron amp O2 or distributed optimizer
            self._model = _LightningModuleWrapperBase(self.model)
        else:
            app_state = AppState()

            if app_state.model_parallel_size is not None:

                logging.info(f"Configuring DDP for model parallelism.")

                # With model parallelism, multiple GPUs form a large "logical GPU"
                # this means that data parallel groups span multiple GPUs
                # and are non-trivial
                # TODO: for megatron-lm self.model is a list
                # Removing self.pre_configure_ddp() as DDP's 'find_unused_parameters' now defaults
                # to False in PTL 2.0 and hence pre_configure_ddp() is removed in ddp.py
                # self.pre_configure_ddp()
                # device_ids = self.determine_ddp_device_ids()
                self._model = DistributedDataParallel(
                    _LightningModuleWrapperBase(self.model),
                    process_group=parallel_state.get_data_parallel_group(),
                    **self._ddp_kwargs,
                )

                if self.no_ddp_communication_hook:
                    # When using custom gradient accumulation and allreduce, disable
                    # DDP communication hook that works on the gradient bucket.
                    # Instead, use the custom gradient function and communication hook,
                    # which is defined in the master optimizer wrapper.
                    self._model.require_backward_grad_sync = False
                    self._model.register_comm_hook(None, noop_hook)

            else:
                super().configure_ddp()

    def init_model_parallel(self, global_rank: int, world_size: int) -> None:
        """ Initializes Megatron-LM model parallel if using model parallelism.

        Args:
            global_rank (int): the global process index.
            world_size (int): the total number of GPUs, num_nodes * num_devices
            is_slurm_managing_tasks (bool, optional): is the cluster managed by SLURM.
        """
        app_state = AppState()

        # we initialize megatron-lm model parallel and data parallel groups
        # after initializing DDP with PTL.
        if app_state.model_parallel_size is not None:
            # destroy groups in case they have already been created
            # this happens with multiple calls to trainer.test for example
            parallel_state.destroy_model_parallel()
            if torch.distributed.is_initialized():
                parallel_state.initialize_model_parallel(
                    tensor_model_parallel_size=app_state.tensor_model_parallel_size,
                    pipeline_model_parallel_size=app_state.pipeline_model_parallel_size,
                    virtual_pipeline_model_parallel_size=app_state.virtual_pipeline_model_parallel_size,
                    pipeline_model_parallel_split_rank=app_state.pipeline_model_parallel_split_rank,
                    use_fp8=app_state.use_fp8,
                )

                # assert that fake tp and pp rank match after model parallel init
                assert app_state.tensor_model_parallel_rank == parallel_state.get_tensor_model_parallel_rank()
                assert app_state.pipeline_model_parallel_rank == parallel_state.get_pipeline_model_parallel_rank()

                app_state.tensor_model_parallel_group = parallel_state.get_tensor_model_parallel_group()
                app_state.data_parallel_group = parallel_state.get_data_parallel_group()
                app_state.data_parallel_rank = parallel_state.get_data_parallel_rank()
                app_state.data_parallel_size = parallel_state.get_data_parallel_world_size()
                app_state.pipeline_model_parallel_group = parallel_state.get_pipeline_model_parallel_group()

                # create MPI process group for UCX-based communication APIs
                if app_state.init_mpi_proc_group:
                    torch.distributed.new_group(backend='mpi')

    def save_checkpoint(
        self, checkpoint: Dict[str, Any], filepath: Union[str, Path], storage_options: Optional[Any] = None
    ) -> None:
        app_state = AppState()
        # PTL override to accomodate model parallel checkpoints
        filepath = inject_model_parallel_rank(filepath)
        if self.is_global_zero or app_state.data_parallel_rank == 0:
            self.checkpoint_io.save_checkpoint(checkpoint, filepath, storage_options=storage_options)

    def load_model_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        # Release strict state dict matching when using Megatron AMP-O2 to skip matching
        # half-precision module wrapper module.
        # TODO: Refactor this to be more generic.
        model_key = None
        model_attr = None
        if hasattr(self.lightning_module, 'model'):
            model_key = 'model'
            model_attr = self.lightning_module.model
        elif hasattr(self.lightning_module, 'enc_dec_model'):
            model_key = 'enc_dec_model'
            model_attr = self.lightning_module.enc_dec_model
        if model_key is not None:
            if isinstance(model_attr, Float16Module):
                new_state_dict = {}
                for key in checkpoint['state_dict'].keys():
                    new_key = key.replace(f'{model_key}.', f'{model_key}.module.', 1)
                    new_state_dict[new_key] = checkpoint['state_dict'][key]
                checkpoint['state_dict'] = new_state_dict

        self.lightning_module.load_state_dict(checkpoint["state_dict"])

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """ PTL override to accomodate model parallel checkpoints """
        # TODO: move to CheckpointIO
        torch.cuda.empty_cache()
        checkpoint_path = inject_model_parallel_rank(checkpoint_path)
        return self.checkpoint_io.load_checkpoint(checkpoint_path)

    def remove_checkpoint(self, filepath: Union[str, Path]) -> None:
        app_state = AppState()
        # PTL override to accomodate model parallel checkpoints
        filepath = inject_model_parallel_rank(filepath)
        if self.is_global_zero or app_state.data_parallel_rank == 0:
            logging.info(f'Removing checkpoint: {filepath}')
            self.checkpoint_io.remove_checkpoint(filepath)

    @property
    def distributed_sampler_kwargs(self):
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
            return super(NLPDDPStrategy, self).distributed_sampler_kwargs


class NLPSaveRestoreConnector(SaveRestoreConnector):
    def __init__(self) -> None:
        if not HAVE_APEX:
            logging.warning(
                "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/apex\n"
                "Megatron-based models require Apex to function correctly."
            )
            # raise ImportError(
            #    "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            # )
        if not HAVE_MEGATRON_CORE:
            logging.warning(
                "megatron-core was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        super().__init__()

    def save_to(self, model, save_path: str):
        app_state = AppState()
        if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:

            dir_name = os.path.dirname(save_path)

            # first we save the weights for each model parallel rank
            if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:
                if app_state.data_parallel_rank == 0:
                    if app_state.pipeline_model_parallel_size == 1:
                        mp_model_weights = os.path.join(
                            dir_name, f'mp_rank_{app_state.tensor_model_parallel_rank:02d}_' + self.model_weights_ckpt
                        )
                    else:
                        mp_model_weights = os.path.join(
                            dir_name,
                            f'tp_rank_{app_state.tensor_model_parallel_rank:02d}_pp_rank_{app_state.pipeline_model_parallel_rank:03d}_'
                            + self.model_weights_ckpt,
                        )

                    self._save_state_dict_to_disk(model.state_dict(), mp_model_weights)

                if torch.distributed.is_initialized():
                    torch.distributed.barrier()

                # create nemo file from folder with all mp_ranks checkpoints
                if (
                    app_state.pipeline_model_parallel_rank == 0
                    and app_state.tensor_model_parallel_rank == 0
                    and app_state.data_parallel_rank == 0
                ):
                    with tempfile.TemporaryDirectory() as tmpdir:

                        if app_state.pipeline_model_parallel_size == 1:
                            # move weights to the tmpdir
                            for tp_rank in range(app_state.tensor_model_parallel_size):
                                os.makedirs(os.path.join(tmpdir, f'mp_rank_{tp_rank:02d}'))
                                mp_model_weights = os.path.join(
                                    dir_name, f'mp_rank_{tp_rank:02d}_' + self.model_weights_ckpt
                                )
                                shutil.move(
                                    mp_model_weights,
                                    os.path.join(tmpdir, f'mp_rank_{tp_rank:02d}', self.model_weights_ckpt),
                                )
                        else:
                            # move weights to the tmpdir
                            for tp_rank, pp_rank in itertools.product(
                                range(app_state.tensor_model_parallel_size),
                                range(app_state.pipeline_model_parallel_size),
                            ):
                                os.makedirs(os.path.join(tmpdir, f'tp_rank_{tp_rank:02d}_pp_rank_{pp_rank:03d}'))
                                mp_model_weights = os.path.join(
                                    dir_name, f'tp_rank_{tp_rank:02d}_pp_rank_{pp_rank:03d}_' + self.model_weights_ckpt
                                )
                                shutil.move(
                                    mp_model_weights,
                                    os.path.join(
                                        tmpdir, f'tp_rank_{tp_rank:02d}_pp_rank_{pp_rank:03d}', self.model_weights_ckpt
                                    ),
                                )

                        # create config and artifacts in tmpdir
                        config_yaml = os.path.join(tmpdir, self.model_config_yaml)
                        model.to_config_file(path2yaml_file=config_yaml)
                        if hasattr(model, 'artifacts') and model.artifacts is not None:
                            self._handle_artifacts(model, nemo_file_folder=tmpdir)
                            self._update_artifact_paths(model, path2yaml_file=config_yaml)

                        # create tar file
                        self._make_nemo_file_from_folder(save_path, tmpdir)

        else:
            return super().save_to(model, save_path)

    def modify_state_dict(self, conf, state_dict):
        if conf.get('megatron_legacy', False):
            new_state_dict = {}
            for key in state_dict.keys():
                new_key = key.replace('bert_model.language_model', 'bert_model.model.language_model')
                new_key = new_key.replace('transformer', 'encoder')
                new_key = new_key.replace('.attention.', '.self_attention.')
                new_state_dict[new_key] = state_dict[key]
            state_dict = new_state_dict

        if conf.get('megatron_amp_O2', False):
            new_state_dict = {}
            for key in state_dict.keys():
                new_key = key.replace('model.', 'model.module.', 1)
                new_state_dict[new_key] = state_dict[key]
            state_dict = new_state_dict

        return state_dict

    def restore_from(
        self,
        calling_cls,
        restore_path: str,
        override_config_path: Optional[Union[OmegaConf, str]] = None,
        map_location: Optional[torch.device] = None,
        strict: bool = True,
        return_config: bool = False,
        trainer: Trainer = None,
    ):
        """
        Restores model instance (weights and configuration) into .nemo file

        Args:
            restore_path: path to .nemo file from which model should be instantiated
            override_config_path: path to a yaml config that will override the internal
                config file or an OmegaConf / DictConfig object representing the model config.
            map_location: Optional torch.device() to map the instantiated model to a device.
                By default (None), it will select a GPU if available, falling back to CPU otherwise.
            strict: Passed to load_state_dict. By default True
            return_config: If set to true, will return just the underlying config of the restored
                model as an OmegaConf DictConfig object without instantiating the model.

        Example:
            ```
            model = nemo.collections.nlp.models.TextClassification.restore_from('asr.nemo')
            assert isinstance(model, nemo.collections.nlp.models.TextClassification)
            ```

        Returns:
            An instance of type cls or its underlying config (if return_config is set).
        """
        # Get path where the command is executed - the artifacts will be "retrieved" there
        # (original .nemo behavior)
        loaded_params = super().load_config_and_state_dict(
            calling_cls, restore_path, override_config_path, map_location, strict, return_config, trainer,
        )
        if not isinstance(loaded_params, tuple) or return_config is True:
            return loaded_params
        conf, instance, state_dict = loaded_params
        state_dict = self.modify_state_dict(conf, state_dict)
        super().load_instance_with_state_dict(instance, state_dict, strict)
        logging.info(f'Model {instance.__class__.__name__} was successfully restored from {restore_path}.')
        return instance


class PEFTSaveRestoreConnector(NLPSaveRestoreConnector):
    """
    PEFT models require the ability to load/save a small subset of the full model (once PEFT params have been infused into the base model.)
    The PEFTSaveRestoreConnector is used to allow loading and saving only the PEFT params while not saving the entire model.

    Args:
        peft_model_nemo_path: Used to provide the .nemo file corresponding to a PEFT model (which will only contain a small set of params)
        peft_model_ckpt_path: Used to provide the path to .ckpt files of a PEFT model. This is required when no .nemo is available (yet) such as during resumed training.
        peft_model_ckpt_name: The filename of the ckpt file inside the peft_model_ckpt_path folder
    If both are provided the peft_model_ckpt_path takes precedence.
    If neither are provided, PEFT params are initialized at random (not loaded from any external source).
    """

    def __init__(
        self,
        peft_model_nemo_path: Optional[str] = None,
        peft_model_ckpt_path: Optional[str] = None,
        peft_model_ckpt_name: Optional[str] = "model_weights.ckpt",
    ) -> None:
        super().__init__()
        self.peft_model_ckpt_name = peft_model_ckpt_name
        if peft_model_ckpt_path:
            # First we will try to load a adapter ckpt path
            # this is given priority over loading from nemo path to make resumption of training possible
            ckpt_name = os.path.basename(peft_model_ckpt_path)
            if not ckpt_name.strip() == '':
                # update the weights file name inside the ckpt path rank folders
                self.peft_model_ckpt_name = ckpt_name
            self.peft_model_ckpt_dir = os.path.dirname(peft_model_ckpt_path)
            assert os.path.isdir(self.peft_model_ckpt_dir)
            self.peft_model_nemo_path = None
        elif peft_model_nemo_path:
            # If resumption is not possible we will try to load a adapter nemo path
            self.peft_model_nemo_path = peft_model_nemo_path
            assert os.path.exists(self.peft_model_nemo_path)
            self.peft_model_ckpt_dir = None
        else:
            # We are not resuming training from a nemo file or a ckpt
            # We are training the adapter from randomly initialization
            self.peft_model_nemo_path = None
            self.peft_model_ckpt_dir = None

    def _load_state_dict_from_disk(self, model_weights, map_location=None):
        """
        Infuse the state_dict of the base model with PEFT params from either a peft_model_nemo_path or peft_model_ckpt_path
        """
        # first load based model weights
        base_model_state_dict = super()._load_state_dict_from_disk(model_weights, map_location)
        # Next, We want to load PEFT model's weights
        if self.peft_model_nemo_path:
            # if the PEFT weights are provided in a .nemo file
            # we need to untar the .nemo if its still tarred
            with tempfile.TemporaryDirectory() as tmpdir:
                self._unpack_nemo_file(self.peft_model_nemo_path, tmpdir)
                model_weights_path = self._inject_model_parallel_rank_for_ckpt(tmpdir, self.peft_model_ckpt_name)
                peft_state_dict = torch.load(model_weights_path, map_location)
        elif self.peft_model_ckpt_dir:
            # if the PEFT weights are provided in a ckpt path file
            # we don't need to untar
            model_weights_path = self._inject_model_parallel_rank_for_ckpt(
                self.peft_model_ckpt_dir, self.peft_model_ckpt_name
            )
            peft_state_dict = torch.load(model_weights_path, map_location)['state_dict']
        else:
            peft_state_dict = {}
        base_model_state_dict.update(peft_state_dict)  # add the PEFT state_dict into the base model's state_dict
        return base_model_state_dict

    def restore_from(
        self,
        calling_cls,
        restore_path: str,
        override_config_path: Optional[Union[OmegaConf, str]] = None,
        map_location: Optional[torch.device] = None,
        strict: bool = True,
        return_config: bool = False,
        trainer: Trainer = None,
    ):
        """
        Extends the restore_from method of the `NLPSaveRestoreConnector` so that PEFT params are inserted into the state_dict which is required when training a PEFT model from scratch.
        """
        # Get path where the command is executed - the artifacts will be "retrieved" there
        # (original .nemo behavior)
        loaded_params = super().load_config_and_state_dict(
            calling_cls, restore_path, override_config_path, map_location, strict, return_config, trainer,
        )
        if not isinstance(loaded_params, tuple) or return_config is True:
            return loaded_params
        conf, instance, state_dict = loaded_params
        state_dict = self.modify_state_dict(conf, state_dict)

        if (
            self.peft_model_nemo_path is None and self.peft_model_ckpt_dir is None
        ):  # we have this check only for training PEFT from scratch
            peft_state_dict = instance.get_peft_state_dict()
            state_dict.update(peft_state_dict)
        self.load_instance_with_state_dict(instance, state_dict, strict)
        logging.info(f'Model {instance.__class__.__name__} was successfully restored from {restore_path}.')
        return instance


class PipelineMixedPrecisionPlugin(MixedPrecisionPlugin):
    """ Overrides PTL autocasting to not wrap training/val/test_step.
        We do this because we have the megatron-core fwd/bwd functions in training_step.
        This means .backward is being called in training_step so we do not want the whole
        step wrapped in autocast.

        We instead wrap the fwd_output_and_loss_func that is passed to the megatron-core fwd/bwd functions.
    """

    def __init__(
        self,
        precision: Literal["16-mixed", "bf16-mixed"],
        device: str,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> None:
        super().__init__(precision, device, scaler=scaler)
        dtype = None
        # MixedPrecisionPlugin class in PTL >= 2.0 takes only "16-mixed" or "bf16-mixed" for precision arg
        if precision == '16-mixed':
            dtype = torch.float16
        elif precision == 'bf16-mixed':
            dtype = torch.bfloat16

        torch.set_autocast_gpu_dtype(dtype)

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """Have the PTL context manager do nothing."""
        yield


class GradScaler(torch.cuda.amp.GradScaler):
    """
    Gradient sclaer for model-parallel inf check. The inf in gradients are checked across tensor-parallel
    ranks in (1) executing optimizer step and (2) gradient scaler update.

    """

    def __init__(
        self,
        init_scale=2.0 ** 16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        enabled=True,
        hysteresis=1,
    ):
        super().__init__(
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            enabled=enabled,
        )
        self.optimizer_update_skipped: Optional[bool] = None
        self.hysteresis = hysteresis
        self._hysteresis_tracker = self.hysteresis

    def _unscale_grads_(self, optimizer, *args):
        if getattr(optimizer, "_custom_amp_unscale_grads", False):
            return optimizer.unscale_grads(*args)
        else:
            return super()._unscale_grads_(optimizer, *args)

    def _maybe_opt_step(self, optimizer, optimizer_state, *args, **kwargs):
        retval = None
        found_inf = torch.cuda.FloatTensor([sum(v.item() for v in optimizer_state["found_inf_per_device"].values())])

        # Update across all model parallel instances.
        torch.distributed.all_reduce(
            found_inf, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_model_parallel_group()
        )

        if found_inf.item() == 0:
            retval = optimizer.step(*args, **kwargs)
            self.optimizer_update_skipped = False
        else:
            self.optimizer_update_skipped = True
        return retval

    def update(self, new_scale=None):
        """
        Updates to native grad scaler update function.
        1. Check inf across model-parallel ranks.
        2. Update hysteresis tracker.
        3. Apply hysteresis to grad scale update.
        """
        if not self._enabled:
            return

        _scale, _growth_tracker = self._check_scale_growth_tracker("update")

        if new_scale is not None:
            # Accept a new user-defined scale.
            if isinstance(new_scale, float):
                self._scale.fill_(new_scale)  # type: ignore[union-attr]
            else:
                reason = "new_scale should be a float or a 1-element torch.cuda.FloatTensor with requires_grad=False."
                assert isinstance(new_scale, torch.cuda.FloatTensor), reason  # type: ignore[attr-defined]
                assert new_scale.numel() == 1, reason
                assert new_scale.requires_grad is False, reason
                self._scale.copy_(new_scale)  # type: ignore[union-attr]
        else:
            # Consume shared inf/nan data collected from optimizers to update the scale.
            # If all found_inf tensors are on the same device as self._scale, this operation is asynchronous.
            found_infs = [
                found_inf.to(device=_scale.device, non_blocking=True)
                for state in self._per_optimizer_states.values()
                for found_inf in state["found_inf_per_device"].values()
            ]

            assert len(found_infs) > 0, "No inf checks were recorded prior to update."

            found_inf_combined = found_infs[0]

            # Update across all model parallel instances.
            torch.distributed.all_reduce(
                found_inf_combined, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_model_parallel_group()
            )

            if len(found_infs) > 1:
                for i in range(1, len(found_infs)):
                    found_inf = found_infs[i]
                    # Update across all model parallel instances.
                    torch.distributed.all_reduce(
                        found_inf, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_model_parallel_group()
                    )
                    found_inf_combined += found_inf

            if found_inf_combined > 0:
                self._hysteresis_tracker -= 1
                if self._hysteresis_tracker <= 0:
                    # When hysteresis becomes zero, follow the native grad scale update rule.
                    # Increase scale and reset growth tracker
                    torch._amp_update_scale_(
                        _scale,
                        _growth_tracker,
                        found_inf_combined,
                        self._growth_factor,
                        self._backoff_factor,
                        self._growth_interval,
                    )
                else:
                    # Only reset the growth tracker when hysteresis is larger than zero
                    _growth_tracker.fill_(0.0)
            else:
                # When no inf found, follow the native grad scale update rule.
                # Increment growth_tracker, update scale when growth tracker reaches the interval, and
                # reset the hysteresis tracker.
                torch._amp_update_scale_(
                    _scale,
                    _growth_tracker,
                    found_inf_combined,
                    self._growth_factor,
                    self._backoff_factor,
                    self._growth_interval,
                )
                self._hysteresis_tracker = self.hysteresis

        # To prepare for next iteration, clear the data collected from optimizers this iteration.
        self._per_optimizer_states = defaultdict(torch.cuda.amp.grad_scaler._refresh_per_optimizer_state)

    def state_dict(self):
        """
        Add hysteresis_tracker to the native functions' state_dict
        """
        return (
            {
                "scale": self.get_scale(),
                "growth_factor": self._growth_factor,
                "backoff_factor": self._backoff_factor,
                "growth_interval": self._growth_interval,
                "_growth_tracker": self._get_growth_tracker(),
                "_hysteresis_tracker": self._hysteresis_tracker,
            }
            if self._enabled
            else {}
        )

    def load_state_dict(self, state_dict):
        """
        Load hysteresis_tracker in addition to the state dict of the native function
        """
        if not self._enabled:
            return

        if len(state_dict) == 0:
            raise RuntimeError(
                "The source state dict is empty, possibly because it was saved "
                "from a disabled instance of GradScaler."
            )

        self._init_scale = state_dict["scale"]
        if self._scale is not None:
            self._scale.fill_(state_dict["scale"])
        self._growth_factor = state_dict["growth_factor"]
        self._backoff_factor = state_dict["backoff_factor"]
        self._growth_interval = state_dict["growth_interval"]
        self._init_growth_tracker = state_dict["_growth_tracker"]
        if self._growth_tracker is not None:
            self._growth_tracker.fill_(state_dict["_growth_tracker"])
        if "_hysterisis_tracker" in state_dict:
            self._hysteresis_tracker = state_dict["_hysterisis_tracker"]
        else:
            self._hysteresis_tracker = 1


class MegatronHalfPrecisionPlugin(MixedPrecisionPlugin):
    """
    Plugin for Half (FP16 and BF16) precision training.
    This plugin assumes the use of the optimizer with master parameters (fp32).
    This plugin uses half-precision at all operators in the model so need of input precision
    at each layer operator.

    Args:
        precision: Whether to use ``torch.float16`` (``16``) or ``torch.bfloat16`` (``'bf16'``).
        device: The device for ``torch.autocast``.
        scaler: An optional :class:`torch.cuda.amp.GradScaler` to use.
    """

    def __init__(
        self, precision: Union[str, int], device: str, scaler: Optional[torch.cuda.amp.GradScaler] = None
    ) -> None:
        super().__init__(precision, device, scaler)
        dtype = None
        # MixedPrecisionPlugin class in PTL >= 2.0 takes only "16-mixed" or "bf16-mixed" for precision arg
        if precision == "16-mixed":
            dtype = torch.float16
        elif precision == "bf16-mixed":
            dtype = torch.bfloat16

        torch.set_autocast_gpu_dtype(dtype)

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        model: Union["pl.LightningModule", torch.nn.Module],
        optimizer_idx: int,
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> None:
        assert isinstance(
            optimizer, MainParamsOptimizerWrapper
        ), "MegatronHalfPrecisionPlugin supports only the optimizer with master parameters"

        if self.scaler is None:
            assert optimizer.fp32_grad_accumulation, "BF16 uses FP32 grad accumulation"
            _ = closure()
            self._after_closure(model, optimizer, optimizer_idx)
            return optimizer.step(**kwargs)

        if isinstance(optimizer, torch.optim.LBFGS):
            raise MisconfigurationException(
                f"Native AMP and the LBFGS optimizer are not compatible (optimizer {optimizer_idx})."
            )
        assert not optimizer.fp32_grad_accumulation, "FP16 uses FP16 grad accumulation"
        closure_result = closure()

        # TODO: Add an option for merged all-reduce

        # cast fp16 grads to fp32 and copy to main grads, which are used for unscale and param update
        optimizer.copy_model_grads_to_main_grads()
        # `unscale` after the closure is executed but before the `on_before_optimizer_step` hook.
        # unscale main (fp32) gradients
        self.scaler.unscale_(optimizer)
        self._after_closure(model, optimizer, optimizer_idx)
        skipped_backward = closure_result is None
        # in manual optimization, the closure does not return a value
        if not isinstance(model, pl.LightningModule) or not model.automatic_optimization or not skipped_backward:
            # note: the scaler will skip the `optimizer.step` if nonfinite gradients are found
            self.scaler.step(optimizer, **kwargs)
            self.scaler.update()

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """ No explicit precision casting. Inputs are supposed to be manually casted """
        try:
            yield
        finally:
            pass


class GlobalBatchDataFetcher(_DataFetcher):
    """ Overrides PTL DataFetcher. Used to fetch global batches."""

    def __init__(self, prefetch_batches: int = 0, store_on_device: bool = False) -> None:

        if not HAVE_APEX:
            logging.warning("Apex was not found. Using model parallel or megatron models will error out.")
        if not HAVE_MEGATRON_CORE:
            logging.warning("Megatron-core was not found. Using model parallel or megatron models will error out..")

        super().__init__(prefetch_batches=prefetch_batches, store_on_device=store_on_device)

    def _fetch_next_batch(self, iterator: Iterator) -> None:
        start_output = self.on_fetch_start()
        batch = [next(iterator) for _ in range(get_num_microbatches())]
        self.fetched += 1
        if not self.prefetch_batches and self._has_len:
            # when we don't prefetch but the dataloader is sized, we use the length for `done`
            dataloader = self.dataloader
            assert isinstance(dataloader, Sized)  # `_has_len` is True
            self.done = self.fetched >= len(dataloader)
        self.on_fetch_end(batch, start_output)
