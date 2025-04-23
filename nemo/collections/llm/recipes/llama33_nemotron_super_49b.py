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

from typing import Callable, Optional

import lightning.pytorch as pl
import nemo_run as run
from nemo.collections.llm import Llama33NemotronSuper49BConfig, LlamaNemotronModel
from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.gpt.data.packed_sequence import PackedSequenceSpecs
from nemo.collections.llm.peft import PEFT_STR2CLS
from nemo.collections.llm.recipes.finetune_default import default_finetune_recipe

NAME = "llama33_nemotron_super_49b"


@run.cli.factory(name=NAME)
def model() -> run.Config[pl.LightningModule]:
    """
    Factory function to create a Llama-3.3-Nemotron-Super-49B model configuration.

    Returns:
        run.Config[pl.LightningModule]: Configuration for the Llama-3.3-Nemotron-Super-49B model.

    Examples:
        CLI usage:
            $ nemo llm pretrain model=llama33_nemotron_super_49b ...

        Python API usage:
            >>> model_config = model()
            >>> print(model_config)
    """
    conf = run.Config(Llama33NemotronSuper49BConfig)
    conf.seq_length = 8192
    return run.Config(LlamaNemotronModel, config=conf)


@run.cli.factory(target=pretrain, name=NAME)
def pretrain_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    fn: Callable = pretrain,
) -> run.Partial:
    """
    This function is not implemented as Llama-3.3-Nemotron-Super-49B is a distilled model.

    The Llama-3.3-Nemotron-Super-49B model is a distilled version of the Llama-3.3-70B model,
    and therefore does not support pre-training. Instead, it should be fine-tuned using the
    finetune_recipe function.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the pre-training run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        fn (Callable): The pre-training function to use.

    Raises:
        NotImplementedError: Always raises this error as pre-training is not supported for this model.

    Examples:
        CLI usage:
            $ nemo llm pretrain --factory llama33_nemotron_super_49b
            # This will raise NotImplementedError

        Python API usage:
            >>> recipe = pretrain_recipe(name="llama33_nemotron_super_49b_pretrain", num_nodes=1)
            # This will raise NotImplementedError

    Note:
        This model is a distilled version of Llama-3.3-70B and only supports fine-tuning.
        Use the finetune_recipe function instead for model adaptation.
    """
    raise NotImplementedError('Llama33 Nemotron Super model is a distilled model based on Llama33-70B.')


@run.cli.factory(target=finetune, name=NAME)
def finetune_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    peft_scheme: Optional[str] = 'lora',
    seq_length: Optional[int] = None,
    packed_sequence: Optional[bool] = None,
    performance_mode: bool = False,
) -> run.Partial:
    """
    Create a fine-tuning recipe for Llama-3.3-Nemotron-Super-49B model.

    This function sets up a complete configuration for fine-tuning, including
    model, trainer, data, logging, optimization, and resumption settings.
    The recipe uses LoRA (Low-Rank Adaptation) for efficient fine-tuning, unless peft_scheme is set to None.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the fine-tuning run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        peft_scheme (Optional[str]): Name of the peft scheme to use for fine-tuning.
            Allowed values: 'lora'/'dora'/'none'/None.
        seq_length (int): Maximum number of tokens per microbatch.
        packed_sequence (Optional[bool]): If true, fine-tuning sequences will be packed into batches up to the given
            maximum seq_length for better efficiency. By default, this value equals performance_mode.
        performance_mode (bool): If true, enables optimizations for maximum performance.

    Returns:
        run.Partial: Partial configuration for fine-tuning.

    Examples:
        CLI usage:
            $ nemo llm finetune --factory llama33_nemotron_super_49b

        Python API usage:
            >>> recipe = finetune_recipe(name="llama33_nemotron_super_49b_finetune", num_nodes=2)
            >>> print(recipe)

    Note:
        This recipe uses the SQuAD dataset for fine-tuning. For more information
        on fine-tuning LLMs with NeMo, see the fine-tuning guide in the
        `examples/llm/finetune/` directory.
    """
    # Default to unpacked data in normal mode and packed data in performance mode
    # once packing recipe is well tested, change this default to true
    if packed_sequence is None:
        packed_sequence = performance_mode

    # For unpacked sequence, most samples in SQuAD dataset are shorter than 2K
    if seq_length is None:
        seq_length = 4096 if packed_sequence else 2048

    recipe = default_finetune_recipe(
        model(), "nvidia/Llama-3_3-Nemotron-Super-49B-v1", dir, name, num_nodes, num_gpus_per_node, packed_sequence
    )
    if peft_scheme is None or peft_scheme.lower() == 'none':
        recipe.trainer.strategy.tensor_model_parallel_size = 8
        recipe.trainer.strategy.pipeline_model_parallel_size = 2
        recipe.optim.config.lr = 5e-6
    elif peft_scheme.lower() in ['lora', 'dora']:
        recipe.trainer.strategy.tensor_model_parallel_size = 4
        recipe.peft = run.Config(PEFT_STR2CLS[peft_scheme.lower()])
        recipe.peft.dim = 8
        recipe.peft.alpha = 16
        recipe.optim.config.use_distributed_optimizer = False

        # some settings currently do not function correctly with LoRA
        recipe.model.config.cross_entropy_loss_fusion = False

        recipe.optim.config.lr = 1e-4
    else:
        raise ValueError(f"Unrecognized peft scheme: {peft_scheme}")

    # Sequence length settings in the model and dataset must agree
    recipe.model.config.seq_length = seq_length
    recipe.data.seq_length = seq_length
    if packed_sequence:
        recipe.data.dataset_kwargs = {'pad_to_max_length': True}
        recipe.data.packed_sequence_specs = run.Config(PackedSequenceSpecs, packed_sequence_size=seq_length)

    return recipe
