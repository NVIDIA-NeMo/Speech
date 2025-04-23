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

import nemo_run as run
import pytest

from nemo.collections.llm import Llama33NemotronSuper49BConfig, LlamaNemotronModel
from nemo.collections.llm.api import finetune
from nemo.collections.llm.peft import PEFT_STR2CLS
from nemo.collections.llm.peft.lora import LoRA
from nemo.collections.llm.recipes import llama33_nemotron_super_49b
from nemo.lightning import Trainer


class TestLlama33NemotronSuper49B:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        return llama33_nemotron_super_49b

    def test_model(self, recipe_module):
        model_config = recipe_module.model()
        assert isinstance(model_config, run.Config)
        assert model_config.__fn_or_cls__ == LlamaNemotronModel
        assert isinstance(model_config.config, run.Config)
        assert model_config.config.__fn_or_cls__ == Llama33NemotronSuper49BConfig
        assert model_config.config.seq_length == 8192

    def test_pretrain_recipe(self, recipe_module):
        with pytest.raises(
            NotImplementedError, match='Llama33 Nemotron Super model is a distilled model based on Llama33-70B'
        ):
            recipe_module.pretrain_recipe()

    def test_finetune_recipe_no_peft(self, recipe_module):
        recipe = recipe_module.finetune_recipe(peft_scheme=None)
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == finetune
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == LlamaNemotronModel
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert recipe.trainer.strategy.tensor_model_parallel_size == 8
        assert recipe.trainer.strategy.pipeline_model_parallel_size == 2
        assert recipe.optim.config.lr == 5e-6
        assert recipe.peft is None

    def test_finetune_recipe_lora(self, recipe_module):
        recipe = recipe_module.finetune_recipe(peft_scheme='lora')
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == finetune
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == LlamaNemotronModel
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert recipe.trainer.strategy.tensor_model_parallel_size == 4
        assert recipe.trainer.strategy.pipeline_model_parallel_size == 1
        assert recipe.optim.config.lr == 1e-4
        assert isinstance(recipe.peft, run.Config)
        assert recipe.peft.__fn_or_cls__ == LoRA
        assert recipe.peft.dim == 8
        assert recipe.peft.alpha == 16
        assert recipe.optim.config.use_distributed_optimizer is False
        assert recipe.model.config.cross_entropy_loss_fusion is False

    def test_finetune_recipe_dora(self, recipe_module):
        recipe = recipe_module.finetune_recipe(peft_scheme='dora')
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == finetune
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == LlamaNemotronModel
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert recipe.trainer.strategy.tensor_model_parallel_size == 4
        assert recipe.trainer.strategy.pipeline_model_parallel_size == 1
        assert recipe.optim.config.lr == 1e-4
        assert isinstance(recipe.peft, run.Config)
        assert recipe.peft.__fn_or_cls__ == PEFT_STR2CLS['dora']
        assert recipe.peft.dim == 8
        assert recipe.peft.alpha == 16
        assert recipe.optim.config.use_distributed_optimizer is False
        assert recipe.model.config.cross_entropy_loss_fusion is False

    def test_finetune_recipe_invalid_peft(self, recipe_module):
        with pytest.raises(ValueError, match="Unrecognized peft scheme: invalid"):
            recipe_module.finetune_recipe(peft_scheme='invalid')

    @pytest.mark.parametrize("num_nodes,num_gpus_per_node", [(1, 8), (2, 4), (4, 2)])
    def test_finetune_recipe_with_different_configurations(self, recipe_module, num_nodes, num_gpus_per_node):
        recipe = recipe_module.finetune_recipe(num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)
        assert recipe.trainer.num_nodes == num_nodes
        assert recipe.trainer.devices == num_gpus_per_node
