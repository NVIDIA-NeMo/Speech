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
import inspect
import json
from pathlib import Path
from typing import Optional, Union

import lightning.pytorch as pl
import torch
import torch.distributed
from lightning.pytorch.trainer.states import TrainerFn
from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.inference.engines.mcore_engine import MCoreEngine
from megatron.core.inference.model_inference_wrappers.abstract_model_inference_wrapper import (
    AbstractModelInferenceWrapper,
)
from megatron.core.inference.text_generation_controllers.simple_text_generation_controller import (
    SimpleTextGenerationController,
)
from megatron.core.transformer.module import MegatronModule

import nemo.lightning as nl
from nemo.lightning import io
from nemo.lightning.ckpt_utils import ADAPTER_META_FILENAME, ckpt_to_context_subdir
from nemo.lightning.io.pl import ckpt_to_weights_subdir
from nemo.lightning.pytorch.callbacks import PEFT
from nemo.lightning.pytorch.strategies.megatron_strategy import MegatronStrategy
from nemo.lightning.pytorch.strategies.utils import RestoreConfig


class MCoreTokenizerWrappper:
    """
    We need this wrapper since mcore generate uses methods/properties such as
    tokenizer.detokenize, tokenizer.tokenize, tokenizer.bos, tokenizer.pad, etc. to encode and decode prompts
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eod = tokenizer.eod
        self.vocab_size = tokenizer.vocab_size

    def detokenize(self, tokens, remove_special_tokens=False):
        """
        Detokenizes a list of tokens into a string.

        Args:
            tokens (list): The list of tokens to detokenize.
            remove_special_tokens (bool, optional): Whether to remove special tokens. Defaults to False.

        Returns:
            str: The detokenized string.
        """
        if 'remove_special_tokens' in inspect.signature(self.tokenizer.ids_to_text).parameters:
            return self.tokenizer.ids_to_text(tokens, remove_special_tokens)
        else:
            return self.tokenizer.ids_to_text(tokens)

    def tokenize(self, prompt):
        """
        Tokenizes a prompt into a list of tokens.

        Args:
            prompt (str): The prompt to tokenize.

        Returns:
            list: The list of tokens.
        """
        return self.tokenizer.text_to_ids(prompt)

    @property
    def additional_special_tokens_ids(self):
        """
        Gets the IDs of additional special tokens.

        Returns:
            list: The IDs of additional special tokens.
        """
        return self.tokenizer.additional_special_tokens_ids

    @property
    def bos(self):
        """
        Gets the ID of the beginning of sequence token.

        Returns:
            int: The ID of the beginning of sequence token.
        """
        return self.tokenizer.bos_id

    @property
    def pad(self):
        """
        Gets the ID of the padding token.

        Returns:
            int: The ID of the padding token.
        """
        return self.tokenizer.pad_id


# TODO: Move to lightning Fabric API.
def _setup_trainer_and_restore_model(path: Path, trainer: nl.Trainer, model: pl.LightningModule):
    """
    Sets up the trainer and restores the model from the given checkpoint path.

    It does the following:
    - Defines a RestoreConfig to restore only model weights
    - Disables setting up optimizers in the Trainer
    - Calls strategy.setup_environment(), model.configure_model() and strategy.setup_megatron_parallel(trainer=trainer)
    - Finally loads the model weights

    Args:
        path (Path): The path to the checkpoint file.
        trainer (nl.Trainer): The trainer object.
        model (pl.LightningModule): The model object.

    Returns:
        None
    """
    assert isinstance(trainer.strategy, MegatronStrategy), "Only MegatronStrategy is supported for trainer.strategy."
    assert trainer.strategy.context_parallel_size <= 1, "Context parallelism is not supported for inference."
    if (adapter_meta_path := ckpt_to_weights_subdir(path, is_saving=False) / ADAPTER_META_FILENAME).exists():
        with open(adapter_meta_path, "r") as f:
            metadata = json.load(f)
        restore_config = RestoreConfig(
            path=metadata["model_ckpt_path"],
            load_model_state=True,
            load_optim_state=False,
        )
    else:
        restore_config = RestoreConfig(
            path=path,
            load_model_state=True,
            load_optim_state=False,
        )

    trainer.strategy.restore_config = restore_config
    trainer.strategy._setup_optimizers = False
    trainer.ckpt_path = None
    trainer.strategy.connect(model)
    if trainer.strategy.launcher is not None:
        trainer.strategy.launcher.launch(lambda: None, trainer=trainer)
    trainer.strategy.setup_environment()

    if not model.state_dict():
        model.configure_model()

    trainer.state.fn = TrainerFn.TESTING
    trainer.strategy.setup_megatron_parallel(trainer=trainer)
    trainer.strategy.trainer = trainer
    trainer.strategy.selective_restore()

    peft: Union[io.TrainerContext, PEFT] = io.load_context(ckpt_to_context_subdir(path), "model.model_transform")
    if isinstance(peft, PEFT):
        model = peft(model)
        adapter_sharded_state_dict = {k: v for k, v in model.sharded_state_dict().items() if ".adapter." in k}
        adapter_state = trainer.strategy.checkpoint_io.load_checkpoint(
            ckpt_to_weights_subdir(path, is_saving=False), sharded_state_dict=adapter_sharded_state_dict
        )
        trainer.strategy.load_model_state_dict(adapter_state, strict=False)


def setup_model_and_tokenizer(
    path: Path,
    trainer: nl.Trainer,
    params_dtype: torch.dtype = torch.bfloat16,
    inference_batch_times_seqlen_threshold: int = 1000,
) -> tuple[MegatronModule, MCoreTokenizerWrappper]:
    """
    Sets up the model and tokenizer for inference.

    This function loads the model and tokenizer from the given checkpoint path,
    sets up the trainer, and returns the Megatron inference-wrapped model and tokenizer.

    Args:
        path (Path): The path to the checkpoint file.
        trainer (nl.Trainer): The trainer object.
        params_dtype (torch.dtype, optional): The data type of the model parameters.
            Defaults to torch.bfloat16.
        inference_batch_times_seqlen_threshold (int, optional): If batch-size times sequence-length is smaller
           than this threshold then we will not use pipelining, otherwise we will.

    Returns:
        tuple[MegatronModule, MCoreTokenizerWrappper]:
            A tuple containing the inference-wrapped model and Mcore wrapped tokenizer.
    """
    model: io.TrainerContext = io.load_context(path=ckpt_to_context_subdir(path), subpath="model")
    _setup_trainer_and_restore_model(path=path, trainer=trainer, model=model)

    inference_wrapped_model = model.get_inference_wrapper(params_dtype, inference_batch_times_seqlen_threshold)
    return inference_wrapped_model, MCoreTokenizerWrappper(model.tokenizer)


def generate(
    model: AbstractModelInferenceWrapper,
    tokenizer: MCoreTokenizerWrappper,
    prompts: list[str],
    encoder_prompts: Optional[list[str]] = None,
    add_BOS: bool = False,
    max_batch_size: int = 4,
    random_seed: Optional[int] = None,
    inference_params: Optional[CommonInferenceParams] = None,
) -> dict:
    """
    Runs generate on the model with the given prompts.

    This function uses the loaded model, loaded tokenizer, and prompts to generate text.
    It returns a dictionary containing the generated text.

    Args:
        model (AbstractModelInferenceWrapper): The inference-wrapped model.
        tokenizer (MCoreTokenizerWrappper): The tokenizer.
        prompts (list[str]): The list of prompts to generate text for.
        encoder_prompts (Optional[list[str]], optional): The list of encoder prompts. Defaults to None.
        add_BOS (bool, optional): Whether to add the beginning of sequence token. Defaults to False.
        max_batch_size (int, optional): The maximum batch size. Defaults to 4.
        random_seed (Optional[int], optional): The random seed. Defaults to None.
        inference_params (Optional[CommonInferenceParams], optional): The inference parameters defined in
            Mcore's CommonInferenceParams. Defaults to None.

    Returns:
        dict: A dictionary containing the generated results.
    """
    from megatron.core.inference.text_generation_controllers.encoder_decoder_text_generation_controller import (
        EncoderDecoderTextGenerationController,
    )

    if encoder_prompts is not None:
        text_generation_controller = EncoderDecoderTextGenerationController(
            inference_wrapped_model=model, tokenizer=tokenizer
        )
    else:
        text_generation_controller = SimpleTextGenerationController(inference_wrapped_model=model, tokenizer=tokenizer)
    mcore_engine = MCoreEngine(
        text_generation_controller=text_generation_controller, max_batch_size=max_batch_size, random_seed=random_seed
    )

    common_inference_params = inference_params or CommonInferenceParams(num_tokens_to_generate=512, top_k=1)

    results = mcore_engine.generate(
        prompts=prompts,
        add_BOS=add_BOS,
        encoder_prompts=encoder_prompts,
        common_inference_params=common_inference_params,
    )

    return results
