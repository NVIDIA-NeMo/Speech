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

import json
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.plugins.io.wrapper import _WrappingCheckpointIO
from lightning.pytorch.trainer.states import TrainerFn
from typing_extensions import override

from nemo.lightning.ckpt_utils import ADAPTER_META_FILENAME
from nemo.lightning.io.mixin import IOMixin
from nemo.lightning.io.pl import ckpt_to_dir, ckpt_to_weights_subdir
from nemo.lightning.megatron_parallel import MegatronParallel
from nemo.lightning.pytorch.callbacks.model_transform import ModelTransform
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.lightning.pytorch.utils import is_trainer_attached
from nemo.utils import logging
from nemo.utils.callbacks.dist_ckpt_io import AsyncCompatibleCheckpointIO

if TYPE_CHECKING:
    from megatron.core.dist_checkpointing.mapping import ShardedStateDict


class PEFT(IOMixin, ABC, ModelTransform):
    """Abstract base class for Parameter-Efficient Fine-Tuning (PEFT) methods.

    This class defines the interface for PEFT methods, which are used to fine-tune
    large language models efficiently by modifying only a small subset of the model's
    parameters.

    Example:
        class MyPEFT(PEFT):
            def transform(self, module, name=None, prefix=None):
                # Implement the transform logic
                pass


        peft = MyPEFT()
        peft_model = LargeLanguageModel(model_transform=peft)
    """

    @abstractmethod
    def transform(self, module, name=None, prefix=None):
        """Transform a single module according to the PEFT method.

        This method is called for each module in the model during the PEFT application process.
        It should be implemented by subclasses to define how individual modules are transformed
        for the specific PEFT technique.

        Args:
            module (nn.Module): The individual module to be transformed.
            name (Optional[str]): The name of the module within the model structure. Defaults to None.
            prefix (Optional[str]): A prefix to be added to the module name, typically used for
                                    nested modules. Defaults to None.

        Returns:
            nn.Module: The transformed module. This can be the original module with modifications,
                       a new module replacing the original, or the original module if no
                       transformation is needed for this specific module.

        Note:
            This method is automatically called for each module in the model when the PEFT
            instance is applied to the model using the __call__ method.
        """
        raise NotImplementedError("The transform method should be implemented by subclasses.")

    def __call__(self, model: nn.Module) -> nn.Module:
        """Apply the PEFT method to the entire model.

        This method freezes the model parameters and walks through the model
        structure, applying the transform method to each module.

        Args:
            model (nn.Module): The model to be fine-tuned.

        Returns:
            nn.Module: The transformed model with PEFT applied.
        """
        self.freeze_model(model)

        # apply walk to model(s)
        if isinstance(model, MegatronParallel) and len(model) > 1:
            for model_chunk in model:
                model_chunk.walk(self.transform)
        elif isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
            model.module.walk(self.transform)
        else:
            model.walk(self.transform)

        if is_trainer_attached(model) and model.trainer.state.fn != TrainerFn.FITTING:
            self.freeze_model(model)
        return model

    def freeze_model(self, model: nn.Module) -> None:
        """Apply a default freeze method to the model.

        This method freezes all the model parameters. This method can be overridden by subclasses to
        implement custom freeze strategies (e.g. freeze only parts of the model)

        Args:
            model (nn.Module): The model to be fine-tuned.

        Returns:
            nn.Module: The transformed model with PEFT applied.
        """
        if isinstance(model, MegatronParallel) and len(model) > 1:
            for model_chunk in model:
                model_chunk.freeze()
        if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
            model.module.freeze()
        else:
            model.freeze()
        if is_trainer_attached(model) and model.trainer.state.fn == TrainerFn.FITTING:
            model.train(mode=True)

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """PTL callback setup function."""
        from nemo.lightning.pytorch.strategies.utils import create_checkpoint_io

        super().setup(trainer, pl_module, stage=stage)

        self._is_fsdp_v1 = type(trainer.strategy).__name__ == 'FSDPStrategy'
        trainer.strategy.trainer = trainer
        wrapped_io = partial(WrappedAdapterIO, peft=self)

        ckpt_io_kwarg_names = [
            "save_ckpt_format",
            "async_save",
            "torch_dist_multiproc",
            "assume_constant_structure",
            "parallel_save",
            "parallel_save_within_dp",
            "parallel_load",
            "load_directly_on_device",
        ]
        ckpt_io_kwargs = {
            arg: getattr(trainer.strategy, arg)
            for arg in filter(lambda x: hasattr(trainer.strategy, x), ckpt_io_kwarg_names)
        }
        trainer.strategy._checkpoint_io = create_checkpoint_io(wrapping_ckpt_io=wrapped_io, **ckpt_io_kwargs)
        self.wrapped_io = (
            trainer.strategy._checkpoint_io._checkpoint_io
            if getattr(trainer.strategy, 'async_save', False)
            else trainer.strategy._checkpoint_io
        )
        trainer.strategy._init_model_parallel = False
        trainer.strategy._setup_optimizers = False

    def apply_transform(self, trainer):
        """
        This function does the following:
        1. Apply PEFT model transform.
        2. Set up model parallel and optimizer, which were skipped in setup
        3. Load weights and optimizer state dict
        4. Set up `finalize_model_grads` from mcore.
        """
        super().apply_transform(trainer)
        self.trainable_params = set(
            name for name, param in trainer.lightning_module.named_parameters() if param.requires_grad
        )

        adapter_sharded_state_dict = {}
        if self.wrapped_io.adapter_ckpt_path is not None:
            logging.info(f"Loading adapters from {self.wrapped_io.adapter_ckpt_path}")
            # create sharded state dict for adapter weights only to enable PEFT resume
            adapter_sharded_state_dict['state_dict'] = {
                k: v for k, v in trainer.model.sharded_state_dict().items() if self.adapter_key_filter(k)
            }

        if hasattr(trainer.strategy, "init_model_parallel"):
            logging.info("Initializing model parallel")
            trainer.strategy.init_model_parallel()

        if trainer.state.fn == TrainerFn.FITTING:
            logging.info("Setting up optimizers")
            trainer.strategy.setup_optimizers(trainer)
            if self.wrapped_io.adapter_ckpt_path is not None and trainer.strategy.should_restore_optimizer_states():
                # PEFT resume, load optimizer state
                adapter_sharded_state_dict['optimizer'] = [
                    trainer.strategy.optimizer_sharded_state_dict(is_loading=True)
                ]

        if adapter_sharded_state_dict:
            adapter_state = self.wrapped_io.load_checkpoint(
                self.wrapped_io.adapter_ckpt_path, sharded_state_dict=adapter_sharded_state_dict
            )
            trainer.strategy.load_model_state_dict(adapter_state, strict=False)
            if trainer.state.fn == TrainerFn.FITTING:
                # Load optimizer
                trainer.strategy.load_optimizer_state_dict(adapter_state, selective_restore=False)
                # Load lr scheduler
                if (lr_schedulers := adapter_state.get('lr_schedulers', None)) is not None:
                    for config, lrs_state in zip(trainer.lr_scheduler_configs, lr_schedulers):
                        config.scheduler.load_state_dict(lrs_state)

        for cb in trainer.callbacks[::-1]:
            if isinstance(cb, MegatronOptimizerModule):
                cb.on_fit_start(trainer, trainer.lightning_module)
                break
        else:
            logging.warning(
                "MegatronOptimizerModule not found in trainer callbacks. finalize_model_grads is not "
                "properly set up for PEFT."
            )

    def adapter_key_filter(self, key: str) -> bool:
        """
        Given a key in the state dict, return whether the key is an adapter (or base model).
        This function can be subclassed in each PEFT method class.
        """
        return key in self.trainable_params or ".adapter." in key or key.endswith(".adapters")


class AdapterWrapper(nn.Module):
    """Abstract base class for wrapping modules with adapters in Parameter-Efficient Fine-Tuning (PEFT).

    This class wraps a module and its associated adapter, providing methods for
    managing the state dictionaries of both the main module and the adapter. It does not
    implement the forward method, which must be implemented by concrete subclasses.

    Attributes:
        to_wrap (nn.Module): The main module to be wrapped.
        adapter (nn.Module): The adapter module to be applied.

    Note:
        This class is abstract and cannot be instantiated directly. Subclasses must
        implement the forward method.

    Example:
        class LoRALinear(AdapterWrapper):
            def __init__(self, to_wrap, adapter):
                super().__init__(to_wrap, adapter)

            def forward(self, x):
                return self.to_wrap(x) + self.adapter(x)

        main_module = nn.Linear(100, 100)
        adapter = nn.Linear(100, 100)
        parallel_adapter = LoRALinear(main_module, adapter)
    """

    def __init__(self, to_wrap: nn.Module, adapter: nn.Module):
        super(AdapterWrapper, self).__init__()
        self.to_wrap = to_wrap
        self.adapter = adapter

    def base_linear_forward(self, x):
        """
        Run the forward method of the linear module `to_wrap`.
        Return a tuple of three elements: linear_output, bias, layernorm_output

        x -> [layernorm/identity] -> layernorm_output -> [linear] -> linear_output, bias

        layernorm_output is different from input x only when linear layer is LayerNormColumnParallelLinear.
        """
        linear_output = self.to_wrap(x)
        assert isinstance(
            linear_output, tuple
        ), f"{self.to_wrap} should return a tuple but instead returns {linear_output}"
        """ Four cases for the wrapped module's return values
        1. nothing: (out, None)
        2. return_bias: (out, bias)
        2. return_layernorm_output: ((out, ln_out), None)
        3. both: (out, bias, ln_out)
        """
        bias = None
        layernorm_output = x
        if len(linear_output) == 2:
            linear_output, bias = linear_output
            if isinstance(linear_output, tuple) and len(linear_output) == 2:
                linear_output, layernorm_output = linear_output
        elif len(linear_output) == 3:
            linear_output, bias, layernorm_output = linear_output

        return linear_output, bias, layernorm_output

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Retrieve the state dictionary of the wrapped module and adapter.

        This method overrides the default state_dict behavior to include both
        the main module's state and the adapter's state under a special 'adapters' key.

        Args:
            destination (Optional[dict]): A dictionary to store the state. If None, a new
                                          dictionary is created. Defaults to None.
            prefix (str): A prefix added to parameter and buffer names. Defaults to ''.
            keep_vars (bool): If True, returns variables instead of tensor values.
                              Defaults to False.

        Returns:
            dict: The state dictionary containing both the main module and adapter states.
        """

        if destination is None:
            destination = {}

        # Get state dict of the main module
        self.to_wrap.state_dict(destination, prefix, keep_vars)

        # Store adapter state dict under the "adapter" prefix in the destination dict
        self.adapter.state_dict(destination, f'{prefix}adapter.', keep_vars)
        return destination

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> "ShardedStateDict":
        """Retrieve the sharded state dictionary of the wrapped module and adapter.

        This method is used for distributed checkpointing, combining the sharded states
        of both the main module and the adapter.

        Args:
            prefix (str): A prefix added to parameter and buffer names. Defaults to ''.
            sharded_offsets (Tuple[Tuple[int, int, int]]): Offsets for sharded parameters.
                                                           Defaults to an empty tuple.
            metadata (Optional[dict]): Additional metadata for the sharded state.
                                       Defaults to None.

        Returns:
            ShardedStateDict: The combined sharded state dictionary.
        """
        sharded_state_dict = {}
        sharded_state_dict.update(self.to_wrap.sharded_state_dict(prefix, sharded_offsets, metadata))
        sharded_state_dict.update(self.adapter.sharded_state_dict(f"{prefix}adapter.", sharded_offsets, metadata))
        return sharded_state_dict


class WrappedAdapterIO(_WrappingCheckpointIO, AsyncCompatibleCheckpointIO):
    """
    A wrapper class for checkpoint I/O operations, specifically designed for PEFT (Parameter-Efficient Fine-Tuning).

    This class handles the complexities of saving and loading checkpoints for both initial PEFT training and resuming
    PEFT training. It ensures that only the necessary adapter weights are saved and loaded, while also preserving the
    base model weights.

    **Usage:**

    1. **Initial PEFT Training:**
       - The class handles the saving of only adapter weights.
       - Metadata about the base model checkpoint is stored for future reference.

    2. **PEFT Resume:**
       - The class loads both base model and adapter weights.
       - The previously stored metadata is used to locate the correct base model checkpoint.

    **Attributes:**

    - `peft`: The PEFT instance associated with the wrapped checkpoint I/O.
    - `model_ckpt_path`: The path to the base model checkpoint.
    - `adapter_ckpt_path`: The path to the adapter checkpoint.
    Note that the paths are set by save/load functions and users do not need to set them.

    **Methods:**

    - `save_checkpoint`: Saves the adapter weights and metadata to the specified path.
    - `load_checkpoint`: Loads the base model and adapter weights based on the specified path and metadata.
    """

    peft: Optional[PEFT] = None
    model_ckpt_path: Optional[Path] = None
    adapter_ckpt_path: Optional[Path] = None

    def __init__(self, checkpoint_io: Optional["CheckpointIO"] = None, peft: Optional[PEFT] = None) -> None:
        self.peft = peft
        super().__init__(checkpoint_io)

    @override
    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        assert self.checkpoint_io is not None

        state_key = 'sharded_state_dict'
        if not state_key in checkpoint:
            state_key = 'state_dict'
        checkpoint[state_key] = dict(
            filter(lambda item: self.peft.adapter_key_filter(item[0]), checkpoint[state_key].items())
        )
        request = self.checkpoint_io.save_checkpoint(checkpoint, path, storage_options=storage_options)

        from nemo.utils.get_rank import is_global_rank_zero

        if is_global_rank_zero():
            metadata = {"model_ckpt_path": str(self.model_ckpt_path)}
            base_dir = ckpt_to_weights_subdir(path, is_saving=True)
            base_dir.mkdir(parents=True, exist_ok=True)
            adapter_meta_path = base_dir / ADAPTER_META_FILENAME
            with open(adapter_meta_path, "w") as f:
                json.dump(metadata, f)
        return request

    @override
    def load_checkpoint(
        self,
        path: _PATH,
        sharded_state_dict=None,
        map_location: Optional[Callable] = None,
        strict: Optional['StrictHandling'] | bool = None,
    ) -> Dict[str, Any]:
        """
        =====================
        Initial PEFT Training
        =====================
        Initial PEFT training requires loading the base model weights. In this case, this function is called by
        trainer.strategy.setup() -> megatron_strategy.restore_model() -> megatron_strategy.load_checkpoint().
        `path = PosixPath(<base_path>)`, and sharded_state_dict contains only base model weights

        ===========
        PEFT Resume
        ===========
        PEFT resume requires loading two set of model weights, 1) base model weights and 2) adapter weights
        Base model weights could be imported from e.g. HF, and is frozen during PEFT training.
        Adapter weights contains the training metadata that will need to be loaded.
        As such, this function will be entered twice during PEFT training resume.

        For the FIRST TIME this function is called by trainer._checkpoint_connector._restore_modules_and_callbacks.
        `path = AdapterPath(<adapter_path>, base_model_path=<base_path>)`, and sharded_state_dict contains only base
        model weights

        For the SECOND TIME this function is called by PEFT.apply_transform (above, in the same file).
        `path = PosixPath(<adapter_path>)`, and sharded_state_dict contains only adapter weights.
        """

        assert self.checkpoint_io is not None

        adapter_meta_path = ckpt_to_dir(path) / ADAPTER_META_FILENAME
        adapter_ckpt = None
        if getattr(path, "base_model_path", None):
            ## PEFT Resume, FIRST TIME
            self.adapter_ckpt_path = Path(str(path))
            adapter_ckpt = self.checkpoint_io.load_checkpoint(path, sharded_state_dict={})  # Loads only metadata
            # path is adapter path to restore the training metadata, but switch to loading base model here.
            path = self.model_ckpt_path = path.base_model_path
        elif adapter_meta_path.exists():
            ## PEFT Resume, SECOND TIME
            with open(adapter_meta_path, "r") as f:
                metadata = json.load(f)
            self.model_ckpt_path = Path(metadata['model_ckpt_path'])
            self.adapter_ckpt_path = path
        else:
            ## Initial PEFT Training
            self.model_ckpt_path = path

        # Note: this will include the Trainer-state of the model-checkpoint
        model_ckpt = self.checkpoint_io.load_checkpoint(path, sharded_state_dict, map_location, strict)
        if adapter_ckpt is not None:
            ## PEFT Resume, FIRST TIME
            adapter_ckpt['state_dict'].update(model_ckpt['state_dict'])
            return adapter_ckpt
        return model_ckpt
