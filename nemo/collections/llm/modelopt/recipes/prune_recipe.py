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


import nemo_run as run

from nemo.collections import llm
from nemo.collections.llm.modelopt import PruningConfig

NAME = "default"


@run.cli.factory(target=llm.prune, name=NAME)
def prune_recipe(nemo_checkpoint: str, save_path: str) -> run.Partial:
    # pylint: disable=line-too-long
    """Create a generic pruning recipe.

    This function sets up a complete configuration for pruning, including trainer and data.

    Args:
        nemo_checkpoint (str): The path to the NeMo checkpoint to be pruned.
        save_path (str): The path to save the pruned NeMo checkpoint.

    Returns:
        run.Partial: Partial configuration for pruning.

    Python recipe usage:
        >>> import nemo_run as run
        >>> from nemo.collections import llm
        >>> from nemo.collections.llm.modelopt.recipes import prune_recipe
        >>> recipe = prune_recipe(
        ...     nemo_checkpoint="/path/to/llama3.1-8b/nemo-ckpt/",
        ...     save_path="/path/to/pruned/llama3.1-8b/nemo-ckpt/",
        ... )
        >>> recipe.devices = 8
        >>> recipe.pp_size = 8
        >>> recipe.data = run.Config(
        ...     llm.PreTrainingDataModule,
        ...     paths=["1.0", "path/to/tokenized/data"],
        ...     seq_length=8192,
        ...     micro_batch_size=1,
        ...     global_batch_size=1,  # should be equal to micro_batch_size
        ... )
        >>> recipe.pruning_config.target_ffn_hidden_size = 9216
        >>> recipe.pruning_config.target_hidden_size = 3072
        >>> ...
        >>> run.run(recipe)
    """
    recipe = run.Partial(
        llm.prune,
        nemo_checkpoint=nemo_checkpoint,
        save_path=save_path,
        pruning_config=run.Config(PruningConfig),
        devices=1,
        num_nodes=1,
        tp_size=1,
        pp_size=1,
        num_train_samples=1024,
        data=run.Config(llm.MockDataModule, seq_length=256, micro_batch_size=1, global_batch_size=1),
        legacy_ckpt=False,
    )

    return recipe
