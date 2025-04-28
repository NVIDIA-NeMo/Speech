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
import pytest

from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.data.squad import SquadDataModule
from nemo.collections.llm.gpt.model.deepseek import DeepSeekModel, DeepSeekV3Config
from nemo.collections.llm.peft.lora import LoRA
from nemo.collections.llm.recipes import deepseek_v3
from nemo.lightning import Trainer


class TestDeepSeekV3:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        return deepseek_v3

    def test_model(self, recipe_module):
        model_config = recipe_module.model()
        assert isinstance(model_config, run.Config)
        assert model_config.__fn_or_cls__ == DeepSeekModel
        assert isinstance(model_config.config, run.Config)
        assert model_config.config.__fn_or_cls__ == DeepSeekV3Config

    def test_pretrain_recipe(self, recipe_module):
        recipe = recipe_module.pretrain_recipe()
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == pretrain
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == DeepSeekModel
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__ == MockDataModule
        assert recipe.data.seq_length == 4096
        assert recipe.data.global_batch_size == 4096
        assert recipe.data.micro_batch_size == 1

        # Check default parallelism settings
        assert recipe.trainer.strategy.tensor_model_parallel_size == 1
        assert recipe.trainer.strategy.expert_model_parallel_size == 64

    def test_finetune_recipe(self, recipe_module):
        recipe = recipe_module.finetune_recipe()
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == finetune
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == DeepSeekModel
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__ == SquadDataModule
        assert recipe.data.seq_length == 2048
        assert isinstance(recipe.peft, run.Config)
        assert recipe.peft.__fn_or_cls__ == LoRA
        assert recipe.optim.config.lr == 1e-4

        # Check LoRA target modules
        assert recipe.peft.target_modules == [
            'linear_q_down_proj',
            'linear_q_up_proj',
            'linear_kv_down_proj',
            'linear_kv_up_proj',
            'linear_proj',
        ]

        # Check parallelism settings for LoRA
        assert recipe.trainer.strategy.sequence_parallel is True
        assert recipe.trainer.strategy.tensor_model_parallel_size == 8
        assert recipe.trainer.strategy.expert_model_parallel_size == 1
        assert recipe.trainer.strategy.pipeline_model_parallel_size == 5
        assert recipe.trainer.strategy.num_layers_in_first_pipeline_stage == 13
        assert recipe.trainer.strategy.num_layers_in_last_pipeline_stage == 12

    def test_finetune_recipe_without_peft(self, recipe_module):
        recipe = recipe_module.finetune_recipe(peft_scheme=None)
        assert recipe.trainer.strategy.sequence_parallel is False
        assert recipe.trainer.strategy.expert_model_parallel_size == 64
        assert recipe.trainer.strategy.tensor_model_parallel_size == 1
        assert recipe.trainer.strategy.pipeline_model_parallel_size == 8
        assert recipe.trainer.strategy.num_layers_in_first_pipeline_stage == 6
        assert recipe.trainer.strategy.num_layers_in_last_pipeline_stage == 7
        assert recipe.optim.config.lr == 5e-6
        assert not hasattr(recipe, 'peft') or recipe.peft is None

    def test_finetune_recipe_with_invalid_peft(self, recipe_module):
        with pytest.raises(ValueError, match="Unrecognized peft scheme: invalid_scheme"):
            recipe_module.finetune_recipe(peft_scheme="invalid_scheme")

    def test_finetune_recipe_with_packed_sequence(self, recipe_module):
        with pytest.raises(ValueError, match="Packed sequence for DeepSeek is not yet supported"):
            recipe_module.finetune_recipe(packed_sequence=True)
