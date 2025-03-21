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
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import lightning.pytorch as pl
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities import model_summary
from typing_extensions import override

from nemo.collections.llm.fn import base as fn
from nemo.lightning.ckpt_utils import ADAPTER_META_FILENAME
from nemo.lightning.io.pl import ckpt_to_dir
from nemo.lightning.megatron_parallel import MegatronParallel
from nemo.lightning.pytorch.callbacks.peft import PEFT, WrappedAdapterIO
from nemo.utils import logging

SPEECHLM_PEFT_RESUME = 'speechlm_peft_resume'


class SpeechToTextLLMPEFT(PEFT):
    def __init__(self, peft: PEFT):
        super().__init__()
        self.peft = peft

    def get_wrappped_io(self):
        """
        This is a helper function to return a partial function that wraps the checkpoint I/O with the PEFT adapter.
        """
        return partial(SpeechLMWrappedAdapterIO, peft=self)

    def __call__(
        self, model: "nemo.collections.speechlm.model.SpeechToTextLLM"  # noqa: F821
    ) -> "nemo.collections.speechlm.model.SpeechToTextLLM":  # noqa: F821
        """Apply the PEFT method to the LLM.

        This method freezes the LLM parameters and walks through the model
        structure, applying the transform method to LLM module.

        Args:
            model (nn.Module): The model to be fine-tuned.

        Returns:
            nn.Module: The transformed model with PEFT applied.
        """
        # only apply PEFT to the language model
        model.freeze_llm()
        module = model.module
        logging.info(f"Applying PEFT to language model with: {self.peft}")

        # If using megatron virtual pipeline parallelism, model is a list of
        # model chunks so iterate over model
        if isinstance(model, MegatronParallel) and len(model) > 1:
            for model_chunk in model:
                self._transform_module(model_chunk)
        else:
            self._transform_module(module)

        if hasattr(model, "trainer") and model.trainer.state.fn != TrainerFn.FITTING:
            self.freeze_model(model)

        logging.info(f"\n{model_summary.summarize(model, max_depth=4)}")
        return model

    def _transform_module(self, module):
        while not hasattr(module, "language_model"):
            module = module.module
        fn.walk(module.language_model, self.transform, _skip_map=True)

    def transform(self, module, name=None, prefix=None):
        return self.peft.transform(module, name=name, prefix=prefix)

    def set_params_to_save(self, trainer: pl.Trainer):
        """
        Set params that should be saved for PEFT, including some params that don't require gradients,
        such as the running mean and var of batchnorm.
        """
        model = trainer.lightning_module  # type: nemo.collections.speechlm.model.SpeechToTextLLM # noqa: F821
        self.params_to_save = set([name for name, _ in model.trainable_parameters()])
        if len(self.params_to_save) == 0:
            raise RuntimeError("No trainable parameters found for PEFT!")


class SpeechLMWrappedAdapterIO(WrappedAdapterIO):

    @override
    def load_checkpoint(
        self,
        path: _PATH,
        sharded_state_dict=None,
        map_location: Optional[Callable] = None,
        strict: Optional['StrictHandling'] | bool = None,
    ) -> Dict[str, Any]:
        """
        Overwrite the load_checkpoint method to handle PEFT resume for SpeechLM.

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

        adapter_meta_path = ckpt_to_dir(path) / ADAPTER_META_FILENAME
        adapter_ckpt = None
        load_base = False

        if getattr(path, "base_model_path", None):
            ## PEFT Resume, FIRST TIME
            self.adapter_ckpt_path = Path(str(path))
            adapter_ckpt = self.checkpoint_io.load_checkpoint(path, sharded_state_dict={})  # Loads only metadata
            # path is adapter path to restore the training metadata, but switch to loading base model here.
            path = self.model_ckpt_path = path.base_model_path
            load_base = True
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
        model_ckpt = self._load_checkpoint(path, sharded_state_dict, map_location, load_base, strict=strict)

        if adapter_ckpt is not None:
            ## PEFT Resume, FIRST TIME
            adapter_ckpt['state_dict'].update(model_ckpt['state_dict'])
            if SPEECHLM_PEFT_RESUME in model_ckpt:
                adapter_ckpt[SPEECHLM_PEFT_RESUME] = True
            return adapter_ckpt
        return model_ckpt

    def _load_checkpoint(
        self,
        path: _PATH,
        sharded_state_dict,
        map_location: Optional[Callable] = None,
        load_base: bool = False,
        strict: Optional['StrictHandling'] | bool = None,
    ) -> None:
        if load_base:
            # Return empty state_dict to skip loading PTL checkpoint in first stage of PEFT
            # Must use with nemo.collections.speechlm.strategies.megatron_strategy.SpeechLMMegatronStrategy
            return {'state_dict': dict(), SPEECHLM_PEFT_RESUME: True}
        else:
            model_ckpt = self.checkpoint_io.load_checkpoint(path, sharded_state_dict, map_location, strict)

        return model_ckpt
