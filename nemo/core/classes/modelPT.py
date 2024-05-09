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
from __future__ import annotations

import copy
import inspect
import os
import uuid
from abc import abstractmethod
from os import path
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import hydra
import torch

try:
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.utils import get_model_config

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities import model_summary, rank_zero_only

from nemo import package_info
from nemo.core import optim
from nemo.core.classes.common import Model
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from nemo.core.optim import McoreDistributedOptimizer, prepare_lr_scheduler
from nemo.utils import logging, model_utils
from nemo.utils.app_state import AppState
from nemo.utils.debug_hook import register_debug_hooks
from nemo.utils.exceptions import NeMoBaseException
from nemo.utils.get_rank import get_rank, is_global_rank_zero

__all__ = ['ModelPT']


# multiple interpolated values in the config
OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)


class ModelPT(LightningModule, Model):
    """
    Interface for Pytorch-lightning based NeMo models
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """
        Base class from which all NeMo models should inherit

        Args:
            cfg (DictConfig):  configuration object.
                The cfg object should have (optionally) the following sub-configs:

                * train_ds - to instantiate training dataset
                * validation_ds - to instantiate validation dataset
                * test_ds - to instantiate testing dataset
                * optim - to instantiate optimizer with learning rate scheduler

            trainer (Optional): Pytorch Lightning Trainer instance
        """
        if trainer is not None and not isinstance(trainer, Trainer):
            raise ValueError(
                f"trainer constructor argument must be either None or pytorch_lightning.Trainer. But got {type(trainer)} instead."
            )
        super().__init__()

        """
        Internal global flags that determine core functionality of ModelPT.

        _MODEL_IS_RESTORED:
            This flag determines the context of the model - whether the model is currently being
            restored or not.
            -   When set, it can be assumed that the model's will disable all automatic methods -
                setup_training_data(), setup_validation/test_data() and their multi equivalents.
            -   If a model is being restored from a archive file (tarfile), it can be assumed that
                under this context, the cwd is *inside* the tarfile itself.

        _MODEL_RESTORE_PATH:
            A string path to a a file from which the model is being restored.
            This file can either be a PyTorch Lightning Checkpoint, or a archive (tarfile) that contains
            artifact objects.
            If it is an archive file, during restoration, the cwd will be temporarily moved to inside the
            archive itself.
        """
        # set global vars in AppState
        app_state = AppState()

        # Convert config to a DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)

        # Convert config to support Hydra 1.0+ instantiation
        cfg = model_utils.maybe_update_config_version(cfg)

        if 'model' in cfg:
            raise ValueError(
                "Creating model config node is forbidden due to collision problem when loading from checkpoint."
            )

        if 'target' not in cfg:
            # This is for Jarvis service.
            OmegaConf.set_struct(cfg, False)
            cfg.target = "{0}.{1}".format(self.__class__.__module__, self.__class__.__name__)
            OmegaConf.set_struct(cfg, True)

        if 'nemo_version' not in cfg:
            with open_dict(cfg):
                cfg.nemo_version = package_info.__version__

        self._cfg = cfg

        # init mapping submodule attribute -> config_field for nested NeMo models
        self._nemo_submodule_name_to_config_field = dict()

        self.save_hyperparameters("cfg")
        self._train_dl = None
        self._validation_dl = None
        self._test_dl = None
        self._optimizer_param_groups = None
        self._optimizer = None
        self._scheduler = None
        self.set_trainer(trainer)

        self._save_restore_connector = SaveRestoreConnector()

        self._set_model_guid()

        # Set device_id in AppState
        if torch.cuda.is_available() and torch.cuda.current_device() is not None:
            app_state.device_id = torch.cuda.current_device()

        if self._cfg is not None and not self._is_model_being_restored():
            # Setup data loaders now (default) or defer setup to `self.setup()`
            # if `defer_setup` is set in the config of the corresponding dataloader.
            if (
                'train_ds' in self._cfg
                and self._cfg.train_ds is not None
                and not self._cfg.train_ds.get('defer_setup', False)
            ):
                self.setup_training_data(self._cfg.train_ds)

            if (
                'validation_ds' in self._cfg
                and self._cfg.validation_ds is not None
                and not self._cfg.validation_ds.get('defer_setup', False)
            ):
                self.setup_multiple_validation_data(val_data_config=cfg.validation_ds)

            if (
                'test_ds' in self._cfg
                and self._cfg.test_ds is not None
                and not self._cfg.test_ds.get('defer_setup', False)
            ):
                self.setup_multiple_test_data(test_data_config=cfg.test_ds)

        else:
            if 'train_ds' in self._cfg and self._cfg.train_ds is not None:
                logging.warning(
                    f"If you intend to do training or fine-tuning, please call the ModelPT.setup_training_data() method "
                    f"and provide a valid configuration file to setup the train data loader.\n"
                    f"Train config : \n{OmegaConf.to_yaml(self._cfg.train_ds)}"
                )

            if 'validation_ds' in self._cfg and self._cfg.validation_ds is not None:
                logging.warning(
                    f"If you intend to do validation, please call the ModelPT.setup_validation_data() or ModelPT.setup_multiple_validation_data() method "
                    f"and provide a valid configuration file to setup the validation data loader(s). \n"
                    f"Validation config : \n{OmegaConf.to_yaml(self._cfg.validation_ds)}"
                )
            if 'test_ds' in self._cfg and self._cfg.test_ds is not None:
                logging.warning(
                    f"Please call the ModelPT.setup_test_data() or ModelPT.setup_multiple_test_data() method "
                    f"and provide a valid configuration file to setup the test data loader(s).\n"
                    f"Test config : \n{OmegaConf.to_yaml(self._cfg.test_ds)}"
                )

        # Create list of lists for val and test outputs to support multiple dataloaders
        # Initialize an empty list as sometimes self._validation_dl can be None at this stage
        self._validation_step_outputs = None

        # Initialize an empty list as sometimes self._test_dl can be None at this stage
        self._test_step_outputs = None

        # ModelPT wrappers over subclass implementations
        self.training_step = model_utils.wrap_training_step(self.training_step)

        # Setup nsys profiling if it has been enabled in the model config
        self._setup_profiling()

        # A flag for the profile generation
        self._nsys_profile_started = False
        self._nsys_profile_complete = False
        self._memory_profile_started = False
        self._memory_profile_complete = False

    def __init_subclass__(cls) -> None:
        cls._save_restore_connector = SaveRestoreConnector()

    def on_fit_start(self) -> None:
        if self.cfg.get("dump_debug_info", False):
            register_debug_hooks(self.model, self.trainer, self.log, self.cfg.get("dump_debug_info_to_file", False))
        return super().on_fit_start()

    def register_artifact(
        self, config_path: str, src: str, verify_src_exists: bool = True,
    ):
        """ Register model artifacts with this function. These artifacts (files) will be included inside .nemo file
            when model.save_to("mymodel.nemo") is called.

            How it works:

            1. It always returns existing absolute path which can be used during Model constructor call
                EXCEPTION: src is None or "" in which case nothing will be done and src will be returned
            2. It will add (config_path, model_utils.ArtifactItem()) pair to self.artifacts

                .. code-block::

                    If "src" is local existing path:
                        then it will be returned in absolute path form.
                    elif "src" starts with "nemo_file:unique_artifact_name":
                        .nemo will be untarred to a temporary folder location and an actual existing path will be returned
                    else:
                        an error will be raised.

            WARNING: use .register_artifact calls in your models' constructors.
            The returned path is not guaranteed to exist after you have exited your model's constructor.

            Args:
                config_path (str): Artifact key. Usually corresponds to the model config.
                src (str): Path to artifact.
                verify_src_exists (bool): If set to False, then the artifact is optional and register_artifact will return None even if
                                          src is not found. Defaults to True.

            Returns:
                str: If src is not None or empty it always returns absolute path which is guaranteed to exist during model instance life
        """

        if src is None or src == "":
            return src

        if Path(src).suffix == ".nemo":
            raise NeMoBaseException(
                "Registering .nemo files as artifacts not supported. "
                "If you are trying to make a nested model, use `register_nemo_submodule`."
            )

        if not hasattr(self, 'artifacts'):
            self.artifacts = {}

        if self.artifacts is None:
            self.artifacts = {}

        if config_path in self.artifacts.keys():
            logging.warning(
                f"You tried to register an artifact under config key={config_path} but an artifact for "
                f"it has already been registered."
            )

        return self._save_restore_connector.register_artifact(self, config_path, src, verify_src_exists)

    def has_artifacts(self) -> bool:
        """Returns True if model has artifacts registered"""
        return hasattr(self, 'artifacts') and self.artifacts is not None and len(self.artifacts) > 0

    def has_native_or_submodules_artifacts(self) -> bool:
        """Returns True if it has artifacts or any of the submodules have artifacts"""
        for module in self.modules():
            if (
                isinstance(module, ModelPT)
                and hasattr(module, 'artifacts')
                and module.artifacts is not None
                and len(module.artifacts) > 0
            ):
                return True
        return False

    def has_nemo_submodules(self) -> bool:
        """Returns True if it has any registered NeMo submodules"""
        return len(self._nemo_submodule_name_to_config_field) > 0

    def register_nemo_submodule(self, name: str, config_field: str, model: "ModelPT") -> None:
        """
        Adds a NeMo model as a submodule. Submodule can be accessed via the `name` attribute on the parent NeMo model this submodule was registered on (`self`).
        In the saving process, the whole parent model (self) is held as a solid model with artifacts
        from the child submodule, the submodule config will be saved to the `config_field` of the parent model.
        This method is necessary to create a nested model, e.g.

        .. code-block:: python

            class ParentModel(ModelPT):
                def __init__(self, cfg, trainer=None):
                    super().__init__(cfg=cfg, trainer=trainer)

                    # annotate type for autocompletion and type checking (optional)
                    self.child_model: Optional[ChildModel] = None
                    if cfg.get("child_model") is not None:
                        self.register_nemo_submodule(
                            name="child_model",
                            config_field="child_model",
                            model=ChildModel(self.cfg.child_model, trainer=trainer),
                        )
                    # ... other code

        Args:
            name: name of the attribute for the submodule
            config_field: field in config, where submodule config should be saved
            model: NeMo model, instance of ModelPT
        """
        # check it is a real NeMo model
        if not isinstance(model, ModelPT):
            raise NeMoBaseException(
                f"Model is not and instance of ModelPT, so can't be registered. Got {type(model).__name__}"
            )
        # check if it is called after __init__
        if not hasattr(self, "_nemo_submodule_name_to_config_field"):
            raise NeMoBaseException(
                "You are trying to register a submodule before the model is initialized. This is not allowed. "
                "Did you forget to call `super().__init__`?"
            )
        # assign attribute to self
        setattr(self, name, model)
        # add to the submodules mapping
        self._nemo_submodule_name_to_config_field[name] = config_field

    def named_nemo_modules(
        self, prefix_name: str = "", prefix_config: str = ""
    ) -> Iterator[Tuple[str, str, "ModelPT"]]:
        """
        Returns an iterator over all NeMo submodules recursively, yielding
        tuples of (attribute path, path in config, submodule), starting from the core module

        Args:
            prefix_name: prefix for the name path
            prefix_config: prefix for the path in config

        Returns:
            Iterator over (attribute path, path in config, submodule), starting from (prefix, self)
        """
        if not hasattr(self, "_nemo_submodule_name_to_config_field"):
            raise NeMoBaseException(
                "Model is not fully initialized. Calling `named_nemo_modules` before __init__ not allowed. "
                "Did you forget to call `super().__init__`?"
            )

        yield prefix_name, prefix_config, self

        # recursive iteration over all NeMo submodules
        for name, config_field in self._nemo_submodule_name_to_config_field.items():
            attribute_path = f"{prefix_name}.{name}" if prefix_name else name
            config_path = f"{prefix_config}.{config_field}" if prefix_config else config_field
            module: ModelPT = getattr(self, name)
            for submodule_name, subconfig_path, submodule in module.named_nemo_modules(
                prefix_name=attribute_path, prefix_config=config_path
            ):
                yield submodule_name, subconfig_path, submodule

    def save_to(self, save_path: str):
        """
        Saves model instance (weights and configuration) into .nemo file
         You can use "restore_from" method to fully restore instance from .nemo file.

        .nemo file is an archive (tar.gz) with the following:
            model_config.yaml - model configuration in .yaml format. You can deserialize this into cfg argument for model's constructor
            model_wights.ckpt - model checkpoint

        Args:
            save_path: Path to .nemo file where model instance should be saved
        """

        def maybe_make_save_dir(path: 'pathlib.Path'):
            if not path.parent.exists():
                path.parent.mkdir(parents=True)

        save_path = Path(save_path).expanduser().resolve()
        app_state = AppState()
        if app_state.model_parallel_size is not None:
            if app_state.model_parallel_size > 1:
                if type(self._save_restore_connector) == SaveRestoreConnector:
                    raise ValueError(
                        'Default NeMo SaveRestoreConnector will not work in model parallel mode. You should use a '
                        'connector which supports model parallel mode, such as NLPSaveRestoreConnector in NLP. You '
                        'can also use a custom one.'
                    )
            if is_global_rank_zero():
                maybe_make_save_dir(save_path)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            # connector checks for ranks properly, no need to check here
            self._save_restore_connector.save_to(self, str(save_path))  # downstream tasks expect str, not Path
        elif is_global_rank_zero():
            maybe_make_save_dir(save_path)
            self._save_restore_connector.save_to(self, str(save_path))  # downstream tasks expect str, not Path

    @classmethod
    def restore_from(
        cls,
        restore_path: str,
        override_config_path: Optional[Union[OmegaConf, str]] = None,
        map_location: Optional[torch.device] = None,
        strict: bool = True,
        return_config: bool = False,
        save_restore_connector: SaveRestoreConnector = None,
        trainer: Optional[Trainer] = None,
    ):
        """
        Restores model instance (weights and configuration) from .nemo file.

        Args:
            restore_path: path to .nemo file from which model should be instantiated
            override_config_path: path to a yaml config that will override the internal
                config file or an OmegaConf / DictConfig object representing the model config.
            map_location: Optional torch.device() to map the instantiated model to a device.
                By default (None), it will select a GPU if available, falling back to CPU otherwise.
            strict: Passed to load_state_dict. By default True.
            return_config: If set to true, will return just the underlying config of the restored
                model as an OmegaConf DictConfig object without instantiating the model.
            trainer: Optional, a pytorch lightning Trainer object that will be forwarded to the
                instantiated model's constructor.
            save_restore_connector (SaveRestoreConnector): Can be overridden to add custom save and restore logic.

            Example:
                ```
                model = nemo.collections.asr.models.EncDecCTCModel.restore_from('asr.nemo')
                assert isinstance(model, nemo.collections.asr.models.EncDecCTCModel)
                ```

        Returns:
            An instance of type cls or its underlying config (if return_config is set).
        """

        if save_restore_connector is None:
            save_restore_connector = SaveRestoreConnector()

        if save_restore_connector.model_extracted_dir is None:
            restore_path = os.path.abspath(os.path.expanduser(restore_path))
        else:
            restore_path = os.path.abspath(os.path.expanduser(save_restore_connector.model_extracted_dir))

        if not path.exists(restore_path):
            raise FileNotFoundError(f"Can't find {restore_path}")

        app_state = AppState()
        app_state.model_restore_path = restore_path

        cls.update_save_restore_connector(save_restore_connector)
        instance = cls._save_restore_connector.restore_from(
            cls, restore_path, override_config_path, map_location, strict, return_config, trainer
        )
        if isinstance(instance, ModelPT):
            instance._save_restore_connector = save_restore_connector
        return instance

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        *args,
        map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
        hparams_file: Optional[str] = None,
        strict: bool = True,
        **kwargs,
    ):
        """
        Loads ModelPT from checkpoint, with some maintenance of restoration.
        For documentation, please refer to LightningModule.load_from_checkpoint() documentation.
        """
        checkpoint = None
        try:
            cls._set_model_restore_state(is_being_restored=True)

            checkpoint = super().load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                *args,
                map_location=map_location,
                hparams_file=hparams_file,
                strict=strict,
                **kwargs,
            )

        finally:
            cls._set_model_restore_state(is_being_restored=False)
        return checkpoint

    @abstractmethod
    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        """
        Setups data loader to be used in training

        Args:
            train_data_layer_config: training data layer parameters.
        Returns:

        """
        pass

    @abstractmethod
    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        """
        Setups data loader to be used in validation
        Args:

            val_data_layer_config: validation data layer parameters.
        Returns:

        """
        pass

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        """
        (Optionally) Setups data loader to be used in test

        Args:
            test_data_layer_config: test data layer parameters.
        Returns:

        """
        raise NotImplementedError()

    def setup_multiple_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        """
        (Optionally) Setups data loader to be used in validation, with support for multiple data loaders.

        Args:
            val_data_layer_config: validation data layer parameters.
        """
        # Set some placeholder overriden by helper method
        self._val_dl_idx = 0
        self._validation_names = None
        self._validation_dl = None  # type: torch.utils.data.DataLoader

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)

        try:
            self._multi_dataset_mode = True
            model_utils.resolve_validation_dataloaders(model=self)
        finally:
            self._multi_dataset_mode = False

        if self._validation_names is None:
            if self._validation_dl is not None and type(self._validation_dl) in [list, tuple]:
                self._validation_names = ['val_{}_'.format(idx) for idx in range(len(self._validation_dl))]

    def setup_multiple_test_data(self, test_data_config: Union[DictConfig, Dict]):
        """
        (Optionally) Setups data loader to be used in test, with support for multiple data loaders.

        Args:
            test_data_layer_config: test data layer parameters.
        """
        # Set some placeholder overriden by helper method
        self._test_dl_idx = 0
        self._test_names = None
        self._test_dl = None  # type: torch.utils.data.DataLoader

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)

        try:
            self._multi_dataset_mode = True
            model_utils.resolve_test_dataloaders(model=self)
        finally:
            self._multi_dataset_mode = False

        if self._test_names is None:
            if self._test_dl is not None and type(self._test_dl) in [list, tuple]:
                self._test_names = ['test_{}_'.format(idx) for idx in range(len(self._test_dl))]

    def setup_megatron_optimization(self, optim_config: Union[Dict[str, Any], DictConfig]):
        """
        Setup mcore optimizer config.

        Args:
            optim_config: Nemo optim args used to set up Mcore optimizer options.
        """

        config = get_model_config(self.model[0])

        megatron_optim_config = OptimizerConfig(
            fp16=config.fp16,
            bf16=config.bf16,
            params_dtype=config.params_dtype,
            lr=optim_config['lr'],
            weight_decay=optim_config['weight_decay'],
            adam_beta1=optim_config['betas'][0],
            adam_beta2=optim_config['betas'][1],
            clip_grad=self.trainer.gradient_clip_val,
            use_distributed_optimizer=self.use_mcore_dist_optim,
            overlap_grad_reduce=self.cfg.optim.get('overlap_grad_sync', False),
            overlap_param_gather=self.cfg.optim.get('overlap_param_sync', False),
        )
        return megatron_optim_config

    def setup_optimization(
        self, optim_config: Optional[Union[DictConfig, Dict]] = None, optim_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Prepares an optimizer from a string name and its optional config parameters.

        Args:
            optim_config: A dictionary containing the following keys:

                * "lr": mandatory key for learning rate. Will raise ValueError if not provided.
                * "optimizer": string name pointing to one of the available optimizers in the registry. \
                If not provided, defaults to "adam".
                * "opt_args": Optional list of strings, in the format "arg_name=arg_value". \
                The list of "arg_value" will be parsed and a dictionary of optimizer kwargs \
                will be built and supplied to instantiate the optimizer.

            optim_kwargs: A dictionary with additional kwargs for the
                optimizer. Used for non-primitive types that are not
                compatible with OmegaConf.

        """
        # Setup the optimizer parameter groups (by default use all parameters that are trainable)
        self.setup_optimizer_param_groups()

        # Make copy of the config so it can be modified later
        optim_config = self._optim_config_copy(optim_config)

        # If config is still None, return without instantiation
        if optim_config is None:
            logging.info('No optimizer config provided, therefore no optimizer was created')
            return

        # See if internal config has `optim` namespace before preservation
        if self._cfg is not None and hasattr(self._cfg, 'optim'):
            self._cfg.optim = optim_config

        # Setup optimizer and scheduler
        if optim_config is not None:
            optim_config = OmegaConf.to_container(optim_config, resolve=True)

        if self._trainer is None:
            logging.warning(f"Trainer wasn't specified in model constructor. Make sure that you really wanted it.")

        if 'sched' in optim_config and self._trainer is not None:

            if not isinstance(self._trainer.accumulate_grad_batches, int):
                raise ValueError("We do not currently support gradient acculumation that is not an integer.")
            if self.trainer.max_steps < 0:
                # Store information needed to calculate max_steps
                optim_config['sched']['t_max_epochs'] = self._trainer.max_epochs
                optim_config['sched']['t_accumulate_grad_batches'] = self._trainer.accumulate_grad_batches
                optim_config['sched']['t_limit_train_batches'] = self._trainer.limit_train_batches

                app_state = AppState()
                if app_state.data_parallel_size is not None:
                    optim_config['sched']['t_num_workers'] = app_state.data_parallel_size
                elif app_state.model_parallel_size is None:
                    optim_config['sched']['t_num_workers'] = self._trainer.num_devices * self._trainer.num_nodes
                else:
                    optim_config['sched']['t_num_workers'] = (
                        self._trainer.num_devices * self._trainer.num_nodes
                    ) / app_state.model_parallel_size
            else:
                optim_config['sched']['max_steps'] = self._trainer.max_steps

        # Force into DictConfig from nested structure
        optim_config = OmegaConf.create(optim_config)
        # Get back nested dict so we its mutable
        optim_config = OmegaConf.to_container(optim_config, resolve=True)

        # Extract scheduler config if inside optimizer config
        if 'sched' in optim_config:
            scheduler_config = optim_config.pop('sched')
        else:
            scheduler_config = None

        # Check if caller provided optimizer name, default to Adam otherwise
        optimizer_cls = optim_config.get('_target_', None)

        if optimizer_cls is None:
            # Try to get optimizer name for dynamic resolution, defaulting to Adam
            optimizer_name = optim_config.get('name', 'adam')
        else:
            if inspect.isclass(optimizer_cls):
                optimizer_name = optimizer_cls.__name__.lower()
            else:
                # resolve the class name (lowercase) from the class path if not provided
                optimizer_name = optimizer_cls.split(".")[-1].lower()

        # We are guarenteed to have lr since it is required by the argparser
        # But maybe user forgot to pass it to this function
        lr = optim_config.get('lr', None)

        # Check if caller has optimizer kwargs, default to empty dictionary
        if 'args' in optim_config:
            optimizer_args = optim_config.pop('args')
            optimizer_args = optim.parse_optimizer_args(optimizer_name, optimizer_args)
        else:
            optimizer_args = copy.deepcopy(optim_config)

            # Remove extra parameters from optimizer_args nest
            # Assume all other parameters are to be passed into optimizer constructor
            optimizer_args.pop('name', None)
            optimizer_args.pop('cls', None)
            optimizer_args.pop('lr', None)

        # Include user-provided kwargs
        if optim_kwargs is not None:
            optimizer_args.update(optim_kwargs)

        # Adaptive schedulers don't need `lr`
        if lr is not None:
            optimizer_args['lr'] = lr

        # Actually instantiate the optimizer
        if optimizer_cls is not None:
            if inspect.isclass(optimizer_cls):
                optimizer = optimizer_cls(self._optimizer_param_groups, **optimizer_args)
                logging.info("Optimizer config = %s", str(optimizer))

                self._optimizer = optimizer

            else:
                # Attempt class path resolution
                try:
                    optimizer_cls = OmegaConf.create({'_target_': optimizer_cls})
                    if lr is not None:
                        optimizer_config = {'lr': lr}
                    else:
                        optimizer_config = {}
                    optimizer_config.update(optimizer_args)

                    optimizer_instance = hydra.utils.instantiate(
                        optimizer_cls, self._optimizer_param_groups, **optimizer_config
                    )  # type: DictConfig

                    logging.info("Optimizer config = %s", str(optimizer_instance))

                    self._optimizer = optimizer_instance

                except Exception as e:
                    logging.error(
                        "Could not instantiate class path - {} with kwargs {}".format(
                            optimizer_cls, str(optimizer_config)
                        )
                    )
                    raise e

        else:
            if optimizer_name == 'mcore_distributed_optim':
                # setup megatron_optim_config and get Mcore based optimizer with the wrapper
                megatron_optim_config = self.setup_megatron_optimization(optimizer_args)
                _megatron_optimizer = get_megatron_optimizer(megatron_optim_config, self.model,)
                optimizer = McoreDistributedOptimizer(_megatron_optimizer)

            else:
                optimizer = optim.get_optimizer(optimizer_name)
                optimizer = optimizer(self._optimizer_param_groups, **optimizer_args)

                logging.info("Optimizer config = %s", str(optimizer))

            self._optimizer = optimizer

        self._scheduler = prepare_lr_scheduler(
            optimizer=self._optimizer, scheduler_config=scheduler_config, train_dataloader=self._train_dl
        )

        # Return the optimizer with/without scheduler
        # This return allows multiple optimizers or schedulers to be created
        return self._optimizer, self._scheduler

    def setup_optimizer_param_groups(self):
        """
            Used to create param groups for the optimizer.
            As an example, this can be used to specify per-layer learning rates:

            optim.SGD([
                        {'params': model.base.parameters()},
                        {'params': model.classifier.parameters(), 'lr': 1e-3}
                        ], lr=1e-2, momentum=0.9)

            See https://pytorch.org/docs/stable/optim.html for more information.
            By default, ModelPT will use self.parameters().
            Override this method to add custom param groups.
            In the config file, add 'optim_param_groups' to support different LRs
            for different components (unspecified params will use the default LR):

            model:
                optim_param_groups:
                    encoder:
                        lr: 1e-4
                        momentum: 0.8
                    decoder:
                        lr: 1e-3
                optim:
                    lr: 3e-3
                    momentum: 0.9
        """
        if not hasattr(self, "parameters"):
            self._optimizer_param_groups = None
            return

        known_groups = []
        param_groups = []
        if "optim_param_groups" in self.cfg:
            param_groups_cfg = self.cfg.optim_param_groups
            for group, group_cfg in param_groups_cfg.items():
                module = getattr(self, group, None)
                if module is None:
                    raise ValueError(f"{group} not found in model.")
                elif hasattr(module, "parameters"):
                    known_groups.append(group)
                    new_group = {"params": list(module.parameters())}
                    for k, v in group_cfg.items():
                        new_group[k] = v
                    param_groups.append(new_group)
                else:
                    raise ValueError(f"{group} does not have parameters.")

            other_params = []
            for n, p in self.named_parameters():
                is_unknown = True
                for group in known_groups:
                    if n.startswith(group):
                        is_unknown = False
                if is_unknown:
                    other_params.append(p)

            if len(other_params):
                param_groups = [{"params": other_params}] + param_groups
        else:
            param_groups = [{"params": list(self.parameters())}]

        self._optimizer_param_groups = param_groups

    def configure_optimizers(self):
        self.setup_optimization()

        if self._scheduler is None:
            return self._optimizer
        else:
            return [self._optimizer], [self._scheduler]

    def propagate_model_guid(self):
        """
        Propagates the model GUID to all submodules, recursively.
        """

        def recursively_propagate_guid(module: "NeuralModule"):
            module.model_guid = self.model_guid
            for child in module.children():
                recursively_propagate_guid(child)

        for _, _, module in self.named_nemo_modules():
            module.model_guid = self.model_guid
            recursively_propagate_guid(module)

    def setup(self, stage: Optional[str] = None):
        """Called at the beginning of fit, validate, test, or predict.
        This is called on every process when using DDP.

        Args:
            stage: fit, validate, test or predict
        """
        self.propagate_model_guid()
        if stage == 'fit':
            train_deferred_setup = (
                'train_ds' in self._cfg
                and self._cfg.train_ds is not None
                and self._cfg.train_ds.get('defer_setup', False)
            )
            if self.train_dataloader() is None and train_deferred_setup:
                self.setup_training_data(self._cfg.train_ds)

        if stage in ('fit', 'validate'):
            val_deferred_setup = (
                'validation_ds' in self._cfg
                and self._cfg.validation_ds is not None
                and self._cfg.validation_ds.get('defer_setup', False)
            )
            if self.val_dataloader() is None and val_deferred_setup:
                self.setup_multiple_validation_data(val_data_config=self._cfg.validation_ds)

        if stage == 'test':
            test_deferred_setup = (
                'test_ds' in self._cfg
                and self._cfg.test_ds is not None
                and self._cfg.test_ds.get('defer_setup', False)
            )
            if self.test_dataloader() is None and test_deferred_setup:
                self.setup_multiple_test_data(test_data_config=self._cfg.test_ds)

    def train_dataloader(self):
        if self._train_dl is not None:
            return self._train_dl

    def val_dataloader(self):
        if self._validation_dl is None:
            # None dataloader no longer supported in PTL2.0
            self._validation_dl = []

        return self._validation_dl

    def test_dataloader(self):
        if self._test_dl is None:
            # None dataloader no longer supported in PTL2.0
            self._test_dl = []

        return self._test_dl

    def on_validation_epoch_end(self) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """
        Default DataLoader for Validation set which automatically supports multiple data loaders
        via `multi_validation_epoch_end`.

        If multi dataset support is not required, override this method entirely in base class.
        In such a case, there is no need to implement `multi_validation_epoch_end` either.

        .. note::
            If more than one data loader exists, and they all provide `val_loss`,
            only the `val_loss` of the first data loader will be used by default.
            This default can be changed by passing the special key `val_dl_idx: int`
            inside the `validation_ds` config.

        Args:
            outputs: Single or nested list of tensor outputs from one or more data loaders.

        Returns:
            A dictionary containing the union of all items from individual data_loaders,
            along with merged logs from all data loaders.
        """
        # Case where we dont provide data loaders
        if self.validation_step_outputs is not None and len(self.validation_step_outputs) == 0:
            return {}

        # Case where we provide exactly 1 data loader
        if isinstance(self.validation_step_outputs[0], dict):
            output_dict = self.multi_validation_epoch_end(self.validation_step_outputs, dataloader_idx=0)

            if output_dict is not None and 'log' in output_dict:
                self.log_dict(output_dict.pop('log'), on_epoch=True)

            self.validation_step_outputs.clear()  # free memory
            return output_dict

        else:  # Case where we provide more than 1 data loader
            output_dict = {'log': {}}

            # The output is a list of list of dicts, outer list corresponds to dataloader idx
            for dataloader_idx, val_outputs in enumerate(self.validation_step_outputs):
                # Get prefix and dispatch call to multi epoch end
                dataloader_prefix = self.get_validation_dataloader_prefix(dataloader_idx)
                dataloader_logs = self.multi_validation_epoch_end(val_outputs, dataloader_idx=dataloader_idx)

                # If result was not provided, generate empty dict
                dataloader_logs = dataloader_logs or {}

                # Perform `val_loss` resolution first (if provided outside logs)
                if 'val_loss' in dataloader_logs:
                    if 'val_loss' not in output_dict and dataloader_idx == self._val_dl_idx:
                        output_dict['val_loss'] = dataloader_logs['val_loss']

                # For every item in the result dictionary
                for k, v in dataloader_logs.items():
                    # If the key is `log`
                    if k == 'log':
                        # Parse every element of the log, and attach the prefix name of the data loader
                        log_dict = {}

                        for k_log, v_log in v.items():
                            # If we are logging the metric, but dont provide it at result level,
                            # store it twice - once in log and once in result level.
                            # Also mark log with prefix name to avoid log level clash with other data loaders
                            if k_log not in output_dict['log'] and dataloader_idx == self._val_dl_idx:
                                new_k_log = k_log

                                # Also insert duplicate key with prefix for ease of comparison / avoid name clash
                                log_dict[dataloader_prefix + k_log] = v_log

                            else:
                                # Simply prepend prefix to key and save
                                new_k_log = dataloader_prefix + k_log

                            # Store log value
                            log_dict[new_k_log] = v_log

                        # Update log storage of individual data loader
                        output_logs = output_dict['log']
                        output_logs.update(log_dict)

                        # Update global log storage
                        output_dict['log'] = output_logs

                    else:
                        # If any values are stored outside 'log', simply prefix name and store
                        new_k = dataloader_prefix + k
                        output_dict[new_k] = v

                self.validation_step_outputs[dataloader_idx].clear()  # free memory

            if 'log' in output_dict:
                self.log_dict(output_dict.pop('log'), on_epoch=True)

            # return everything else
            return output_dict

    def on_test_epoch_end(self) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """
        Default DataLoader for Test set which automatically supports multiple data loaders
        via `multi_test_epoch_end`.

        If multi dataset support is not required, override this method entirely in base class.
        In such a case, there is no need to implement `multi_test_epoch_end` either.

        .. note::
            If more than one data loader exists, and they all provide `test_loss`,
            only the `test_loss` of the first data loader will be used by default.
            This default can be changed by passing the special key `test_dl_idx: int`
            inside the `test_ds` config.

        Args:
            outputs: Single or nested list of tensor outputs from one or more data loaders.

        Returns:
            A dictionary containing the union of all items from individual data_loaders,
            along with merged logs from all data loaders.
        """
        # Case where we dont provide data loaders
        if self.test_step_outputs is not None and len(self.test_step_outputs) == 0:
            return {}

        # Case where we provide exactly 1 data loader
        if isinstance(self.test_step_outputs[0], dict):
            output_dict = self.multi_test_epoch_end(self.test_step_outputs, dataloader_idx=0)

            if output_dict is not None and 'log' in output_dict:
                self.log_dict(output_dict.pop('log'), on_epoch=True)

            self.test_step_outputs.clear()  # free memory
            return output_dict

        else:  # Case where we provide more than 1 data loader
            output_dict = {'log': {}}

            # The output is a list of list of dicts, outer list corresponds to dataloader idx
            for dataloader_idx, test_outputs in enumerate(self.test_step_outputs):
                # Get prefix and dispatch call to multi epoch end
                dataloader_prefix = self.get_test_dataloader_prefix(dataloader_idx)
                dataloader_logs = self.multi_test_epoch_end(test_outputs, dataloader_idx=dataloader_idx)

                # If result was not provided, generate empty dict
                dataloader_logs = dataloader_logs or {}

                # Perform `test_loss` resolution first (if provided outside logs)
                if 'test_loss' in dataloader_logs:
                    if 'test_loss' not in output_dict and dataloader_idx == self._test_dl_idx:
                        output_dict['test_loss'] = dataloader_logs['test_loss']

                # For every item in the result dictionary
                for k, v in dataloader_logs.items():
                    # If the key is `log`
                    if k == 'log':
                        # Parse every element of the log, and attach the prefix name of the data loader
                        log_dict = {}
                        for k_log, v_log in v.items():
                            # If we are logging the loss, but dont provide it at result level,
                            # store it twice - once in log and once in result level.
                            # Also mark log with prefix name to avoid log level clash with other data loaders
                            if k_log not in output_dict['log'] and dataloader_idx == self._test_dl_idx:
                                new_k_log = k_log

                                # Also insert duplicate key with prefix for ease of comparison / avoid name clash
                                log_dict[dataloader_prefix + k_log] = v_log

                            else:
                                # Simply prepend prefix to key and save
                                new_k_log = dataloader_prefix + k_log

                            log_dict[new_k_log] = v_log

                        # Update log storage of individual data loader
                        output_logs = output_dict.get('log', {})
                        output_logs.update(log_dict)

                        # Update global log storage
                        output_dict['log'] = output_logs

                    else:
                        # If any values are stored outside 'log', simply prefix name and store
                        new_k = dataloader_prefix + k
                        output_dict[new_k] = v
                self.test_step_outputs[dataloader_idx].clear()  # free memory

            if 'log' in output_dict:
                self.log_dict(output_dict.pop('log'), on_epoch=True)

            # return everything else
            return output_dict

    def multi_validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]], dataloader_idx: int = 0
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """
        Adds support for multiple validation datasets. Should be overriden by subclass,
        so as to obtain appropriate logs for each of the dataloaders.

        Args:
            outputs: Same as that provided by LightningModule.on_validation_epoch_end()
                for a single dataloader.
            dataloader_idx: int representing the index of the dataloader.

        Returns:
            A dictionary of values, optionally containing a sub-dict `log`,
            such that the values in the log will be pre-pended by the dataloader prefix.
        """
        logging.warning(
            "Multi data loader support has been enabled, but "
            "`multi_validation_epoch_end(outputs, dataloader_idx) has not been implemented.\n"
            "If you require multi data loader support for validation sets, please override this method.\n"
            "If you do not require multi data loader support, please instead override "
            "`on_validation_epoch_end(outputs)."
        )

    def multi_test_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]], dataloader_idx: int = 0
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """
        Adds support for multiple test datasets. Should be overriden by subclass,
        so as to obtain appropriate logs for each of the dataloaders.

        Args:
            outputs: Same as that provided by LightningModule.on_validation_epoch_end()
                for a single dataloader.
            dataloader_idx: int representing the index of the dataloader.

        Returns:
            A dictionary of values, optionally containing a sub-dict `log`,
            such that the values in the log will be pre-pended by the dataloader prefix.
        """
        logging.warning(
            "Multi data loader support has been enabled, but "
            "`multi_test_epoch_end(outputs, dataloader_idx) has not been implemented.\n"
            "If you require multi data loader support for validation sets, please override this method.\n"
            "If you do not require multi data loader support, please instead override "
            "`on_test_epoch_end(outputs)."
        )

    def get_validation_dataloader_prefix(self, dataloader_idx: int = 0) -> str:
        """
        Get the name of one or more data loaders, which will be prepended to all logs.

        Args:
            dataloader_idx: Index of the data loader.

        Returns:
            str name of the data loader at index provided.
        """
        return self._validation_names[dataloader_idx]

    def get_test_dataloader_prefix(self, dataloader_idx: int = 0) -> str:
        """
        Get the name of one or more data loaders, which will be prepended to all logs.

        Args:
            dataloader_idx: Index of the data loader.

        Returns:
            str name of the data loader at index provided.
        """
        return self._test_names[dataloader_idx]

    def load_part_of_state_dict(self, state_dict, include, exclude, load_from_string=None):

        excluded_param_names = []
        # create dict
        dict_to_load = {}
        for k, v in state_dict.items():
            should_add = False
            # if any string in include is present, should add
            for p in include:
                if p in k:
                    should_add = True
                    break
            # except for if any string from exclude is present
            for e in exclude:
                if e in k:
                    excluded_param_names.append(k)
                    should_add = False
                    break
            if should_add:
                dict_to_load[k] = v

        # Restore checkpoint part into current model
        self.load_state_dict(dict_to_load, strict=False)
        if load_from_string is not None:
            logging.info(f'Model checkpoint partially restored from {load_from_string}')
            if len(excluded_param_names) > 0:
                logging.info(
                    f'The following parameters were excluded when loading from {load_from_string} : {excluded_param_names}'
                )
                logging.info(f'Make sure that this is what you wanted!')
        else:
            if len(excluded_param_names) > 0:
                logging.info(
                    f'The following parameters were excluded when loading checkpoint : {excluded_param_names}'
                )

    @rank_zero_only
    def maybe_init_from_pretrained_checkpoint(self, cfg: OmegaConf, map_location: str = 'cpu'):
        """
        Initializes a given model with the parameters obtained via specific config arguments.
        The state dict of the provided model will be updated with `strict=False` setting so as to prevent
        requirement of exact model parameters matching.

        Initializations:
            init_from_nemo_model: Str path to a .nemo model in order to load state_dict from single nemo file;
            if loading from multiple files, pass in a dict where the values have the following fields:

                path: Str path to .nemo model

                include: Optional list of strings, at least one of which needs to be contained in parameter name
                to be loaded from this .nemo file. Default: everything is included.

                exclude: Optional list of strings, which can be used to exclude any parameter containing one of
                these strings from being loaded from this .nemo file. Default: nothing is excluded.

                hydra usage example:

                init_from_nemo_model:
                    model0:
                        path:<path/to/model1>
                        include:["encoder"]
                    model1:
                        path:<path/to/model2>
                        include:["decoder"]
                        exclude:["embed"]

            init_from_pretrained_model: Str name of a pretrained model checkpoint (obtained via cloud).
                The model will be downloaded (or a cached copy will be used), instantiated and then
                its state dict will be extracted. If loading from multiple models, you can pass in a dict
                with the same format as for init_from_nemo_model, except with "name" instead of "path"

            init_from_ptl_ckpt: Str name of a Pytorch Lightning checkpoint file. It will be loaded and
                the state dict will extracted. If loading from multiple files, you can pass in a dict
                with the same format as for init_from_nemo_model.

        Args:
            cfg: The config used to instantiate the model. It need only contain one of the above keys.
            map_location: str or torch.device() which represents where the intermediate state dict
                (from the pretrained model or checkpoint) will be loaded.

        """
        args = [
            'init_from_nemo_model',
            'init_from_pretrained_model',
            'init_from_ptl_ckpt',
        ]
        arg_matches = [(1 if arg in cfg and arg is not None else 0) for arg in args]

        if sum(arg_matches) == 0:
            # model weights do not need to be restored
            return

        if sum(arg_matches) > 1:
            raise ValueError(
                f"Cannot pass more than one model initialization arguments to config!\n"
                f"Found : {[args[idx] for idx, arg_present in enumerate(arg_matches) if arg_present]}"
            )

        if 'init_from_nemo_model' in cfg and cfg.init_from_nemo_model is not None:
            with open_dict(cfg):
                if isinstance(cfg.init_from_nemo_model, str):
                    model_path = cfg.init_from_nemo_model
                    # Restore model
                    restored_model = self.restore_from(
                        model_path, map_location=map_location, strict=cfg.get("init_strict", True)
                    )
                    # Restore checkpoint into current model
                    self.load_state_dict(restored_model.state_dict(), strict=False)
                    logging.info(f'Model checkpoint restored from nemo file with path : `{model_path}`')
                    del restored_model
                elif isinstance(cfg.init_from_nemo_model, (DictConfig, dict)):
                    model_load_dict = cfg.init_from_nemo_model
                    for model_load_cfg in model_load_dict.values():
                        model_path = model_load_cfg.path
                        # Restore model
                        restored_model = self.restore_from(
                            model_path, map_location=map_location, strict=cfg.get("init_strict", True)
                        )

                        include = model_load_cfg.pop('include', [""])
                        exclude = model_load_cfg.pop('exclude', [])

                        self.load_part_of_state_dict(
                            restored_model.state_dict(), include, exclude, f'nemo file with path `{model_path}`'
                        )

                        del restored_model
                else:
                    raise TypeError("Invalid type: init_from_nemo_model is not a string or a dict!")

        if 'init_from_pretrained_model' in cfg and cfg.init_from_pretrained_model is not None:
            with open_dict(cfg):
                # Restore model

                if isinstance(cfg.init_from_pretrained_model, str):
                    model_name = cfg.pop('init_from_pretrained_model')

                    # Check if model is being resumed or not - only works if `Trainer` is attached to model
                    if hasattr(self, 'trainer') and self.trainer is not None:
                        trainer = self.trainer
                        if (
                            hasattr(trainer, 'resume_from_checkpoint')
                            and trainer._checkpoint_connector.resume_checkpoint_path is not None
                        ):
                            logging.info(
                                "Model training is being resumed via Pytorch Lightning.\n"
                                "Initialization from pretrained model (via cloud) will be skipped."
                            )
                            return

                    restored_model = self.from_pretrained(
                        model_name, map_location=map_location, strict=cfg.get("init_strict", True)
                    )

                    # Restore checkpoint into current model
                    self.load_state_dict(restored_model.state_dict(), strict=False)
                    logging.info(f'Model checkpoint restored from pretrained checkpoint with name : `{model_name}`')

                    del restored_model
                elif isinstance(cfg.init_from_pretrained_model, (DictConfig, dict)):
                    model_load_dict = cfg.init_from_pretrained_model
                    for model_load_cfg in model_load_dict.values():
                        model_name = model_load_cfg.name
                        # Restore model
                        restored_model = self.from_pretrained(
                            model_name, map_location=map_location, strict=cfg.get("init_strict", True)
                        )

                        include = model_load_cfg.pop('include', [""])
                        exclude = model_load_cfg.pop('exclude', [])

                        self.load_part_of_state_dict(
                            restored_model.state_dict(),
                            include,
                            exclude,
                            f'pretrained checkpoint with name `{model_name}`',
                        )

                        del restored_model
                else:
                    raise TypeError("Invalid type: init_from_pretrained_model is not a string or a dict!")

        if 'init_from_ptl_ckpt' in cfg and cfg.init_from_ptl_ckpt is not None:
            with open_dict(cfg):
                if isinstance(cfg.init_from_ptl_ckpt, str):
                    # Restore checkpoint
                    ckpt_path = cfg.pop('init_from_ptl_ckpt')
                    ckpt = torch.load(ckpt_path, map_location=map_location)

                    # Restore checkpoint into current model
                    self.load_state_dict(ckpt['state_dict'], strict=False)
                    logging.info(
                        f'Model checkpoint restored from pytorch lightning checkpoint with path : `{ckpt_path}`'
                    )

                    del ckpt
                elif isinstance(cfg.init_from_ptl_ckpt, (DictConfig, dict)):
                    model_load_dict = cfg.init_from_ptl_ckpt
                    for model_load_cfg in model_load_dict.values():
                        ckpt_path = model_load_cfg.path
                        # Restore model
                        ckpt = torch.load(ckpt_path, map_location=map_location)

                        include = model_load_cfg.pop('include', [""])
                        exclude = model_load_cfg.pop('exclude', [])

                        self.load_part_of_state_dict(
                            ckpt['state_dict'], include, exclude, f'nemo file with path `{ckpt_path}`'
                        )

                        del ckpt
                else:
                    raise TypeError("Invalid type: init_from_ptl_ckpt is not a string or a dict!")

    def teardown(self, stage: str):
        """
        Called at the end of fit and test.

        Args:
            stage: either 'fit' or 'test'
        """
        if stage == 'fit':
            # Update env variable to bypass multi gpu issue after training
            # This fix affects usage of trainer.test() after trainer.train()
            # If trainer.train() was done on multiple GPUs, then trainer.test()
            # will try to do ddp, even if its a new Trainer object with just 1 GPU.
            # Temporary patch to fix that
            if 'PL_TRAINER_GPUS' in os.environ:
                os.environ.pop('PL_TRAINER_GPUS')

        super().teardown(stage)

    @classmethod
    def extract_state_dict_from(
        cls,
        restore_path: str,
        save_dir: str,
        split_by_module: bool = False,
        save_restore_connector: SaveRestoreConnector = None,
    ):
        """
        Extract the state dict(s) from a provided .nemo tarfile and save it to a directory.

        Args:
            restore_path: path to .nemo file from which state dict(s) should be extracted
            save_dir: directory in which the saved state dict(s) should be stored
            split_by_module: bool flag, which determins whether the output checkpoint should
                be for the entire Model, or the individual module's that comprise the Model
            save_restore_connector (SaveRestoreConnector): Can be overrided to add custom save and restore logic.

        Example:
            To convert the .nemo tarfile into a single Model level PyTorch checkpoint
            ::
            state_dict = nemo.collections.asr.models.EncDecCTCModel.extract_state_dict_from('asr.nemo', './asr_ckpts')


            To restore a model from a Model level checkpoint
            ::
            model = nemo.collections.asr.models.EncDecCTCModel(cfg)  # or any other method of restoration
            model.load_state_dict(torch.load("./asr_ckpts/model_weights.ckpt"))


            To convert the .nemo tarfile into multiple Module level PyTorch checkpoints
            ::
            state_dict = nemo.collections.asr.models.EncDecCTCModel.extract_state_dict_from('asr.nemo', './asr_ckpts', split_by_module=True)


            To restore a module from a Module level checkpoint
            ::
            model = nemo.collections.asr.models.EncDecCTCModel(cfg)  # or any other method of restoration

            # load the individual components
            model.preprocessor.load_state_dict(torch.load("./asr_ckpts/preprocessor.ckpt"))
            model.encoder.load_state_dict(torch.load("./asr_ckpts/encoder.ckpt"))
            model.decoder.load_state_dict(torch.load("./asr_ckpts/decoder.ckpt"))


        Returns:
            The state dict that was loaded from the original .nemo checkpoint
        """
        if save_restore_connector is None:
            save_restore_connector = SaveRestoreConnector()

        if not path.exists(restore_path):
            raise FileExistsError(f"Can't find {restore_path}")

        cls.update_save_restore_connector(save_restore_connector)
        state_dict = cls._save_restore_connector.extract_state_dict_from(restore_path, save_dir, split_by_module)
        return state_dict

    def prepare_test(self, trainer: 'Trainer') -> bool:
        """
        Helper method to check whether the model can safely be tested
        on a dataset after training (or loading a checkpoint).

        ::

            trainer = Trainer()
            if model.prepare_test(trainer):
                trainer.test(model)

        Returns:
            bool which declares the model safe to test. Provides warnings if it has to
            return False to guide the user.
        """
        if not hasattr(self._cfg, 'test_ds'):
            logging.info("No `test_ds` config found within the manifest.")
            return False

        # Replace ddp multi-gpu until PTL has a fix
        DDP_WARN = """\n\nDuring testing, it is currently advisable to construct a new Trainer "
                    "with single GPU and no DDP to obtain accurate results.
                    "Following pattern should be used: "
                    "trainer = Trainer(devices=1, accelerator='gpu')"
                    "if model.prepare_test(trainer):"
                    "  trainer.test(model)\n\n"""

        if trainer is not None:
            if trainer.num_devices > 1:
                logging.warning(DDP_WARN)
                return False

        # Assign trainer to the model
        self.set_trainer(trainer)
        return True

    def set_trainer(self, trainer: Trainer):
        """
        Set an instance of Trainer object.

        Args:
            trainer: PyTorch Lightning Trainer object.
        """
        self.trainer = trainer
        self._trainer = trainer
        self.set_world_size(trainer)

    def set_world_size(self, trainer: Trainer):
        """
        Determines the world size from the PyTorch Lightning Trainer.
        And then updates AppState.

        Args:
            trainer (Trainer): PyTorch Lightning Trainer object
        """
        # Update AppState with world information from trainer
        self.world_size = 1

        if trainer is not None:
            if isinstance(trainer, Trainer):
                if trainer.num_devices and trainer.num_nodes:
                    self.world_size = trainer.num_devices * trainer.num_nodes
            else:
                logging.warning(f'World size can only be set by PyTorch Lightning Trainer.')
        app_state = AppState()
        app_state.world_size = self.world_size

    def summarize(self, max_depth: int = 1) -> model_summary.ModelSummary:
        """Summarize this LightningModule.

        Args:
            max_depth: The maximum depth of layer nesting that the summary will include. A value of 0 turns the
                layer summary off. Default: 1.

        Return:
            The model summary object
        """
        return model_summary.summarize(self, max_depth=max_depth)

    def _update_dataset_config(self, dataset_name: str, config: Optional[Union[DictConfig, Dict]]):
        """
        Update the config (if not None) of the dataset by given name.
        Preserves said config after updating.

        Args:
            dataset_name: str name of the dataset whose config is being updated.
                Can be one of `train`, `validation` and `test`.
            config: Optional DictConfig or dict. If None is passed, this method simply returns.
                If dict is passed, it is cast into a DictConfig.
                The internal config is updated with the passed config.
        """
        if hasattr(self, '_multi_dataset_mode') and self._multi_dataset_mode is True:
            return

        if config is not None:
            if not isinstance(config, DictConfig):
                config = OmegaConf.create(config)

            if dataset_name in ['train', 'validation', 'test']:
                OmegaConf.set_struct(self.cfg, False)

                key_name = dataset_name + "_ds"
                self.cfg[key_name] = config

                OmegaConf.set_struct(self.cfg, True)

                # Update hyper parameters by calling property setter
                self.cfg = self._cfg
            else:
                raise ValueError("`dataset_name` when updating config must be one of [train, validation, test]")

    @property
    def num_weights(self):
        """
        Utility property that returns the total number of parameters of the Model.
        """
        num: int = 0
        for p in self.parameters():
            if p.requires_grad:
                num += p.numel()
        return num

    @property
    def cfg(self):
        """
        Property that holds the finalized internal config of the model.

        Note:
            Changes to this config are not reflected in the state of the model.
            Please create a new model using an updated config to properly update the model.
        """
        return self._cfg

    @LightningModule.trainer.getter
    def trainer(self):
        return self._trainer

    @cfg.setter
    def cfg(self, cfg):
        """
        Property that holds the finalized internal config of the model.

        Note:
            Changes to this config are not reflected in the state of the model.
            Please create a new model using an updated config to properly update the model.
        """
        self._cfg = cfg
        self._set_hparams(OmegaConf.create({'cfg': self._cfg}))

        # TODO: Remove in NeMo 1.7 (or when PTL fixes this on their end)
        if hasattr(self, '_hparams_initial') and 'cfg' in self._hparams_initial:
            self._hparams_initial['cfg'] = OmegaConf.to_object(self._cfg)

    @property
    def validation_step_outputs(self):
        """
        Cached outputs of validation_step. It can be a list of items (for single data loader) or a list of lists
        (for multiple data loaders).

        Returns:
            List of outputs of validation_step.
        """
        if self._validation_step_outputs is not None:
            return self._validation_step_outputs

        # Initialize new output list
        self._validation_step_outputs = []
        # Check len(self._validation_dl) > 1 as sometimes single dataloader can be in a list: [<Dataloader obj>] when ds_item in
        # config has 1 item passed in a list
        if (
            self._validation_dl is not None
            and isinstance(self._validation_dl, (list, tuple))
            and len(self._validation_dl) > 1
        ):
            for _ in range(len(self._validation_dl)):
                self._validation_step_outputs.append([])

        return self._validation_step_outputs

    @validation_step_outputs.setter
    def validation_step_outputs(self, value):
        self._validation_step_outputs = value

    @property
    def test_step_outputs(self):
        """
        Cached outputs of test_step. It can be a list of items (for single data loader) or a list of lists (for multiple data loaders).

        Returns:
            List of outputs of test_step.
        """
        if self._test_step_outputs is not None:
            return self._test_step_outputs

        # Initialize new output list
        self._test_step_outputs = []
        # Check len(self._test_dl) > 1 as sometimes single dataloader can be in a list: [<Dataloader obj>] when ds_item in
        # config has 1 item passed in a list
        if self._test_dl is not None and isinstance(self._test_dl, (list, tuple)) and len(self._test_dl) > 1:
            for _ in range(len(self._test_dl)):
                self._test_step_outputs.append([])

        return self._test_step_outputs

    @test_step_outputs.setter
    def test_step_outputs(self, value):
        self._test_step_outputs = value

    @staticmethod
    def _is_model_being_restored() -> bool:
        app_state = AppState()
        return app_state.is_model_being_restored

    @staticmethod
    def _set_model_restore_state(is_being_restored: bool, folder: str = None):
        app_state = AppState()
        app_state.is_model_being_restored = is_being_restored
        app_state.nemo_file_folder = folder

    def _set_model_guid(self):
        if not hasattr(self, 'model_guid'):
            appstate = AppState()

            # Generate a unique uuid for the instance
            # also determine if the model is being restored or not, and preserve the path
            self.model_guid = str(uuid.uuid4())
            if self._is_model_being_restored():
                restore_path = appstate.model_restore_path
            else:
                restore_path = None

            appstate.register_model_guid(self.model_guid, restoration_path=restore_path)

    @classmethod
    def update_save_restore_connector(cls, save_restore_connector):
        if hasattr(cls, '_save_restore_connector'):
            cls._save_restore_connector = save_restore_connector
        else:
            setattr(cls, '_save_restore_connector', save_restore_connector)

    def _setup_profiling(self):
        """ Enables nsys profiling
            To use, add the following optoins to the model config:
            ## Nsys profiling options
            nsys_profile: False
                start_step: 10  # Global batch to start profiling
                end_step: 10 # Global batch to end profiling
                ranks: [0] # Global rank IDs to profile
                gen_shape: False # Generate model and kernel details including input shapes
            And then wrap the model training script with:
            nsys profile -s none -o <profile filepath>  -t cuda,nvtx --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop python ./examples/...
            See more options at: https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-profiling

            Enables CUDA memory profiling
            To use, add the following optoins to the model config:
            ## CUDA memory profiling options
            memory_profile: False
                start_step: 10  # Global batch to start profiling
                end_step: 10 # Global batch to end profiling
                rank: 0 # Global rank ID to profile
                output_path: None # Path to store the profile output file
        """
        if self.cfg.get('nsys_profile', None) is not None:
            if self.cfg.nsys_profile.get('enabled', False):
                # Nsys profiling options
                self._nsys_profile_enabled = True
                self._nsys_profile_start_step = self.cfg.nsys_profile.get('start_step', 0)
                self._nsys_profile_end_step = self.cfg.nsys_profile.get('end_step', 0)
                self._nsys_profile_ranks = self.cfg.nsys_profile.get('ranks', [0])
                self._nsys_profile_gen_shape = self.cfg.nsys_profile.get('gen_shape', False)

                if type(self._nsys_profile_start_step) == int:
                    logging.info(f'Nsys profiling setup with start_step: {self._nsys_profile_start_step}')
                else:
                    raise ValueError(
                        f'Nsys start_step must be of type int. Found: {type(self._nsys_profile_start_step)}'
                    )

                if type(self._nsys_profile_end_step) == int:
                    logging.info(f'Nsys profiling setup with end_step: {self._nsys_profile_end_step}')
                else:
                    raise ValueError(f'Nsys end_step must be of type int. Found: {type(self._nsys_profile_end_step)}')

                if self._nsys_profile_end_step >= self._nsys_profile_start_step:
                    pass
                else:
                    raise ValueError(f'Nsys end_step must be greater than or equal to nsys start_step')

        if self.cfg.get('memory_profile', None) is not None:
            if self.cfg.memory_profile.get('enabled', False):
                # CUDA memory profiling options
                self._memory_profile_enabled = True
                self._memory_profile_start_step = self.cfg.memory_profile.get('start_step', 0)
                self._memory_profile_end_step = self.cfg.memory_profile.get('end_step', 0)
                self._memory_profile_rank = self.cfg.memory_profile.get('rank', 0)
                self._memory_profile_output_path = self.cfg.memory_profile.get('output_path', None)

                if type(self._memory_profile_start_step) == int:
                    logging.info(f'Nsys profiling setup with start_step: {self._memory_profile_start_step}')
                else:
                    raise ValueError(
                        f'CUDA memory start_step must be of type int. Found: {type(self._memory_profile_start_step)}'
                    )

                if type(self._memory_profile_end_step) == int:
                    logging.info(f'CUDA memory profiling setup with end_step: {self._memory_profile_end_step}')
                else:
                    raise ValueError(
                        f'CUDA memory end_step must be of type int. Found: {type(self._memory_profile_end_step)}'
                    )

                if self._memory_profile_end_step >= self._memory_profile_start_step:
                    pass
                else:
                    raise ValueError(f'CUDA memory end_step must be greater than or equal to memory start_step')

                if self._memory_profile_output_path is None or not os.path.isdir(self._memory_profile_output_path):
                    raise ValueError(
                        f'Memory profile output path ({self._memory_profile_output_path}) is not set or does not exist.'
                    )

    def on_train_start(self):
        """ PyTorch Lightning hook:
            https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-train-start
            We use it here to copy the relevant config for dynamic freezing.
        """

        # dynamic freezing
        # should fire only once, on the very first batch of training and never again
        if not hasattr(self, '_freeze_cfg'):
            if (
                hasattr(self.cfg, 'freeze_updates')
                and self.cfg.freeze_updates is not None
                and self.cfg.freeze_updates.get('enabled', False)
            ):
                setattr(self, '_freeze_cfg', OmegaConf.to_container(self.cfg.freeze_updates))
                self._freeze_cfg['is_frozen'] = {k: False for k in self._freeze_cfg['modules'].keys()}
            else:
                setattr(self, '_freeze_cfg', None)

    def on_train_batch_start(self, batch: Any, batch_idx: int, unused: int = 0) -> Optional[int]:
        """ PyTorch Lightning hook:
            https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-train-batch-start
            We use it here to enable nsys profiling and dynamic freezing.
        """

        # nsys profiling
        if self.device.type == 'cuda':
            if hasattr(self, '_nsys_profile_enabled'):
                if self._nsys_profile_enabled and not self._nsys_profile_started:
                    if batch_idx >= self._nsys_profile_start_step and get_rank() in self._nsys_profile_ranks:
                        logging.info("====== Start nsys profiling ======")
                        torch.cuda.cudart().cudaProfilerStart()
                        if self._nsys_profile_gen_shape:
                            torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()
                        self._nsys_profile_started = True

            if hasattr(self, '_memory_profile_enabled'):
                if self._memory_profile_enabled and not self._memory_profile_started:
                    if batch_idx >= self._memory_profile_start_step and get_rank() == self._memory_profile_rank:
                        logging.info("====== Start CUDA memory profiling ======")
                        torch.cuda.memory._record_memory_history(max_entries=100000)
                        self._memory_profile_started = True

        # dynamic freezing
        if hasattr(self, '_freeze_cfg') and self._freeze_cfg is not None:
            if self.training and hasattr(self, "trainer") and self.trainer is not None:
                num_updates = self.trainer.global_step + 1

                for ml, m_steps in self._freeze_cfg['modules'].items():
                    # we could do hasattr check here, but it's too expensive for each step
                    # consequently you'll throw an error if the module name doesn't exist
                    # or was spelled wrong in the config.yaml
                    if isinstance(m_steps, list):
                        assert len(m_steps) == 2, "freeze_updates modules list cannot have more than two elements"
                        should_freeze = (num_updates >= m_steps[0]) and (num_updates <= m_steps[1] or m_steps[1] == -1)
                    else:
                        should_freeze = num_updates <= m_steps or m_steps == -1
                    if should_freeze and not self._freeze_cfg['is_frozen'][ml]:
                        getattr(self, ml).freeze()
                        getattr(self, ml).train()
                        self._freeze_cfg['is_frozen'][ml] = True
                    elif not should_freeze and self._freeze_cfg['is_frozen'][ml]:
                        getattr(self, ml).unfreeze()
                        self._freeze_cfg['is_frozen'][ml] = False

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int, unused: int = 0) -> None:
        """ PyTorch Lightning hook:
            https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-train-batch-end
            We use it here to enable nsys profiling.
        """

        if self.device.type == 'cuda':
            if hasattr(self, '_nsys_profile_enabled'):
                if self._nsys_profile_enabled and not self._nsys_profile_complete:
                    if batch_idx >= self._nsys_profile_end_step and get_rank() in self._nsys_profile_ranks:
                        logging.info("====== End nsys profiling ======")
                        torch.cuda.cudart().cudaProfilerStop()
                        self._nsys_profile_complete = True

            if hasattr(self, '_memory_profile_enabled'):
                if self._memory_profile_enabled and not self._memory_profile_complete:
                    if batch_idx >= self._memory_profile_end_step and get_rank() == self._memory_profile_rank:
                        logging.info("====== End CUDA memory profiling ======")
                        torch.cuda.memory._dump_snapshot(
                            f'{self._memory_profile_output_path}/memory_profile_rank{self._memory_profile_rank}.pickle'
                        )
                        torch.cuda.memory._record_memory_history(enabled=None)
                        self._memory_profile_complete = True

    def _cleanup_on_execution_end(self):
        """
        Utility function to clean up the module state at the end of execution.
        """

        # dynamic freezing cleanup
        if hasattr(self, '_freeze_cfg'):
            delattr(self, '_freeze_cfg')

        # Clear up the val and test output caches
        self._validation_step_outputs = None
        self._test_step_outputs = None

    def on_train_end(self):
        """ PyTorch Lightning hook:
            https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-train-end
            We use it here to cleanup the dynamic freezing config.
        """

        self._cleanup_on_execution_end()

    def on_test_end(self):
        """ PyTorch Lightning hook:
            https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-test-end
        """

        self._cleanup_on_execution_end()

    def on_predict_end(self):
        """ PyTorch Lightning hook:
            https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-test-end
        """

        self._cleanup_on_execution_end()

    # TODO: Remove in PTL 1.7.2
    def cuda(self, device=None):
        """ PTL is overriding this method and changing the pytorch behavior of a module.
            The PTL LightingModule override will move the module to device 0 if device is None.
            See the PTL method here: https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/core/mixins/device_dtype_mixin.py#L113

            Here we are overriding this to maintain the default Pytorch nn.module behavior:
            https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/module.py#L728

        Moves all model parameters and buffers to the GPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on GPU while being optimized.

        .. note::
            This method modifies the module in-place.

        Args:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(device, int):
            device = torch.device("cuda", index=device)
        return super().cuda(device=device)

    def _optim_config_copy(self, optim_config: Optional[Union[DictConfig, Dict]]) -> Optional[DictConfig]:
        """
        Return a copy of `optim_config` if provided (and otherwise of the internal optim config, if available).
        """
        if optim_config is None:
            # See if internal config has `optim` namespace
            if self._cfg is not None and hasattr(self._cfg, 'optim'):
                optim_config = self._cfg.optim

        if optim_config is None:
            return None

        if isinstance(optim_config, DictConfig):
            return copy.deepcopy(optim_config)
        else:
            return OmegaConf.create(optim_config)
