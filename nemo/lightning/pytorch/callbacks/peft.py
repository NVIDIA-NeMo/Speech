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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch.nn as nn
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.plugins.io.wrapper import _WrappingCheckpointIO
from pytorch_lightning.trainer.states import TrainerFn
from typing_extensions import override

from nemo.lightning.io.pl import ckpt_to_dir
from nemo.lightning.pytorch.callbacks.model_transform import ModelTransform
from nemo.utils import logging

if TYPE_CHECKING:
    from megatron.core.dist_checkpointing.mapping import ShardedStateDict


_ADAPTER_META_FILENAME = "adapter_metadata.json"


class PEFT(ABC, ModelTransform):
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

        model.freeze()
        model.walk(self.transform)

        return model

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage=stage)

        trainer.strategy.trainer = trainer
        self.wrapped_io = WrappedAdapterIO(trainer.strategy.checkpoint_io, self)
        trainer.strategy._checkpoint_io = self.wrapped_io
        trainer.strategy._init_model_parallel = False
        trainer.strategy._setup_optimizers = False

    def apply_transform(self, trainer):
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
                trainer.strategy.load_optimizer_state_dict(adapter_state, selective_restore=True)

    def adapter_key_filter(self, key: str) -> bool:
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
        class AdapterParallelAdd(AdapterWrapper):
            def __init__(self, to_wrap, adapter):
                super().__init__(to_wrap, adapter)

            def forward(self, x):
                return self.to_wrap(x) + self.adapter(x)

        main_module = nn.Linear(100, 100)
        adapter = nn.Linear(100, 100)
        parallel_adapter = AdapterParallelAdd(main_module, adapter)
    """

    def __init__(self, to_wrap: nn.Module, adapter: nn.Module):
        super(AdapterWrapper, self).__init__()
        self.to_wrap = to_wrap
        self.adapter = adapter

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
        main_state_dict = self.to_wrap.state_dict(destination, prefix, keep_vars)

        # Store adapter state dict under the special "adapters" key in the destination dict
        adapter_state_dict = self.adapter.state_dict(None, prefix, keep_vars)
        destination[f'{prefix}adapters'] = adapter_state_dict
        return main_state_dict

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

    def load_state_dict(self, state_dict, strict=True):
        """Load a state dictionary into the wrapped module and adapter.

        This method overrides the default load_state_dict behavior to handle
        loading states for both the main module and the adapter.

        Args:
            state_dict (dict): The state dictionary to load.
            strict (bool): Whether to strictly enforce that the keys in state_dict
                           match the keys returned by this module's state_dict()
                           function. Defaults to True.
        """
        # Check if the 'adapters' key is present in the state_dict
        if 'adapters' in state_dict:
            adapter_state_dict = state_dict.pop('adapters')
        else:
            adapter_state_dict = {}

        # Load the main module state dict
        self.to_wrap.load_state_dict(state_dict, strict)

        # Load the adapter module state dict if present
        if adapter_state_dict:
            self.adapter.load_state_dict(adapter_state_dict, strict)


class WrappedAdapterIO(_WrappingCheckpointIO):
    peft: Optional[PEFT] = None
    model_ckpt_path: Optional[Path] = None
    adapter_ckpt_path: Optional[Path] = None

    def __init__(self, checkpoint_io: Optional["CheckpointIO"] = None, peft: Optional[PEFT] = None) -> None:
        self.peft = peft
        super().__init__(checkpoint_io)

    @override
    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        assert self.checkpoint_io is not None

        checkpoint['sharded_state_dict'] = dict(
            filter(lambda item: self.peft.adapter_key_filter(item[0]), checkpoint['sharded_state_dict'].items())
        )
        self.checkpoint_io.save_checkpoint(checkpoint, path, storage_options=storage_options)

        from nemo.utils.get_rank import is_global_rank_zero

        if is_global_rank_zero():
            metadata = {"model_ckpt_path": str(self.model_ckpt_path)}
            adapter_meta_path = ckpt_to_dir(path) / _ADAPTER_META_FILENAME
            with open(adapter_meta_path, "w") as f:
                json.dump(metadata, f)

    @override
    def load_checkpoint(
        self, path: _PATH, sharded_state_dict=None, map_location: Optional[Callable] = None
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
        `path = AdapterPath(<adapter_path>, base_model_path=<base_path>)`, and sharded_state_dict contains only base model weights

        For the SECOND TIME this function is called by PEFT.apply_transform (above, in the same file).
        `path = PosixPath(<adapter_path>)`, and sharded_state_dict contains only adapter weights.
        """

        assert self.checkpoint_io is not None

        adapter_meta_path = ckpt_to_dir(path) / _ADAPTER_META_FILENAME
        adapter_ckpt = None
        if getattr(path, "base_model_path", None):
            ## PEFT Resume, FIRST TIME
            self.adapter_ckpt_path = Path(str(path))
            adapter_ckpt = self.checkpoint_io.load_checkpoint(path)  # Loads only metadata
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
        model_ckpt = self.checkpoint_io.load_checkpoint(path, sharded_state_dict, map_location)
        if adapter_ckpt is not None:
            ## PEFT Resume, FIRST TIME
            adapter_ckpt['state_dict'].update(model_ckpt['state_dict'])
            return adapter_ckpt
        return model_ckpt
