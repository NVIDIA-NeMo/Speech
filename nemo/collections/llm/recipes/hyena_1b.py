# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024 Arc Institute. All rights reserved.
# Copyright (c) 2024 Michael Poli. All rights reserved.
# Copyright (c) 2024 Stanford University. All rights reserved
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

import lightning.pytorch as pl
import nemo_run as run

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.recipes.hyena_base import model_recipe, pretrain_recipe_creater, tokenizer_recipe


NAME = "hyena_1b"


@run.cli.factory(name=NAME)
def tokenizer() -> run.Config[TokenizerSpec]:
    """
    Defines a factory function for creating a tokenizer configuration.

    This function is registered as a CLI factory with the specified name and
    returns a tokenizer configuration based on the `tokenizer_recipe`.

    Returns:
        run.Config[TokenizerSpec]: A configuration object for the tokenizer.
    """
    return tokenizer_recipe()


@run.cli.factory(name=NAME)
def model(tp_comm_overlap: bool = False, seq_length: int = 8192) -> run.Config[pl.LightningModule]:
    """
    Factory function to create a Striped-Hyena 1B model configuration.

    Returns:
        run.Config[pl.LightningModule]: Configuration for the Striped-Hyena 1B model.
    """
    return model_recipe('1b', tp_comm_overlap, seq_length)


@run.cli.factory(target=pretrain, name=NAME)
def pretrain_recipe(
    dir=None,
    micro_batch_size=1,
    global_batch_size=8,
    num_nodes=1,
    num_gpus_per_node=8,
    tensor_parallel_size=1,
    context_parallel_size=1,
    model_size='1b',
    fn=pretrain,
    **kwargs,
) -> run.Partial:
    """
    Create a pre-training recipe for Striped-Hyena 1B model.

    This function sets up a complete configuration for pre-training, including
    model, trainer, data, logging, optimization, and resumption settings.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the pre-training run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        fn (Callable): The pre-training function to use.

    Returns:
        run.Partial: Partial configuration for pre-training.

    """
    return pretrain_recipe_creater(
        dir=dir,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        tensor_parallel_size=tensor_parallel_size,
        context_parallel_size=context_parallel_size,
        model_size=model_size,
        **kwargs,
    )


@run.cli.factory(target=finetune, name=NAME)
def finetune_recipe(
    dir=None,
    resume_path="hyena-1b-pretrain",
    micro_batch_size=1,
    global_batch_size=8,
    num_nodes=1,
    num_gpus_per_node=8,
    tensor_parallel_size=1,
    context_parallel_size=1,
    model_size='1b',
    fn=finetune,
    **kwargs,
) -> run.Partial:
    """ """

    return pretrain_recipe_creater(
        dir=dir,
        resume_path=resume_path,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        tensor_parallel_size=tensor_parallel_size,
        context_parallel_size=context_parallel_size,
        model_size=model_size,
        fn=fn,
        **kwargs,
    )
