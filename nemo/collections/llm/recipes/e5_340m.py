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

from typing import Optional

import lightning.pytorch as pl
import nemo_run as run
from nemo.collections import llm
from nemo.collections.llm.api import finetune
from nemo.collections.llm.recipes.bert_embedding import bert_embedding_model
from nemo.collections.llm.recipes.finetune_default import default_finetune_recipe

NAME = "e5_340m"


@run.cli.factory(name=NAME)
def model() -> run.Config[pl.LightningModule]:
    """
    Factory function to create a E5-Large (340 million) model configuration.

    Returns:
        run.Config[pl.LightningModule]: Configuration for the E5-Large (340 million) model.

    Examples:
        CLI usage:
            $ nemo llm pretrain model=e5_340m ...

        Python API usage:
            >>> model_config = model()
            >>> print(model_config)
    """
    return bert_embedding_model(version=NAME)


@run.cli.factory(target=finetune, name=NAME)
def finetune_recipe(
    dir: Optional[str] = None,
    resume_path: str = "intfloat/e5-large-v2",
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    peft_scheme: Optional[str] = None,
    seq_length: int = 512,
    micro_batch_size: int = 4,
    global_batch_size: int = 32,
) -> run.Partial:
    """
    Create a fine-tuning recipe for E5-large (340 million) model.

    This function sets up a complete configuration for fine-tuning, including
    model, trainer, data, logging, optimization, and resumption settings.
    Only SFT is currently supported for E5 model.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the fine-tuning run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        peft_scheme (Optional[str]): Name of the peft scheme to use for fine-tuning.
            Allowed values: 'none'/None.
        resume_path (str): Path to the NeMo checkpoint
        seq_length (int): Maximum number of tokens per microbatch.
        micro_batch_size (int): Micro batch size.
        global_batch_size (int): Global batch size.


    Returns:
        run.Partial: Partial configuration for fine-tuning.

    Examples:
        CLI usage:
            $ nemo llm finetune --factory e5_340m

        Python API usage:
            >>> recipe = finetune_recipe(name="e5_340m_finetune", num_nodes=1)
            >>> print(recipe)

    Note:
        This recipe uses the Specter dataset for fine-tuning. For more information
        on fine-tuning LLMs with NeMo, see the fine-tuning guide in the
        `examples/llm/finetune/` directory.
    """
    recipe = default_finetune_recipe(model(), resume_path, dir, name, num_nodes, num_gpus_per_node)
    datamodule = run.Config(
        llm.SpecterDataModule,
        seq_length=seq_length,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
    )
    recipe.data = datamodule

    assert peft_scheme is None or peft_scheme.lower() == 'none', 'E5 only supports SFT.'
    return recipe
