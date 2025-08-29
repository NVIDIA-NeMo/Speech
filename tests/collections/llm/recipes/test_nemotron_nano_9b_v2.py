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
from nemo.collections.llm.recipes import nemotron_nano_9b_v2
from nemo.lightning import Trainer


class TestNemotronNano9Bv2:
    @pytest.fixture
    def recipe(self):
        return nemotron_nano_9b_v2

    def test_model_config(self, recipe):
        model_config = recipe.model()
        assert model_config.config.__fn_or_cls__.__name__ == "NemotronNano9Bv2"

    def test_model(self, recipe):
        model_config = recipe.model()
        assert isinstance(model_config, run.Config)
        assert model_config.__fn_or_cls__.__name__ == "MambaModel"
        assert isinstance(model_config.config, run.Config)
        assert model_config.config.__fn_or_cls__.__name__ == "NemotronNano9Bv2"

    def test_trainer(self, recipe):
        trainer_config = recipe.trainer()
        assert isinstance(trainer_config, run.Config)
        assert trainer_config.__fn_or_cls__ == Trainer
        assert trainer_config.accelerator == "gpu"
        assert trainer_config.devices == 8
        assert trainer_config.num_nodes == 8
        assert trainer_config.max_steps == 10

        # Check strategy configuration
        assert isinstance(trainer_config.strategy, run.Config)
        assert trainer_config.strategy.__fn_or_cls__.__name__ == "MegatronStrategy"
        assert trainer_config.strategy.tensor_model_parallel_size == 2
        assert trainer_config.strategy.pipeline_model_parallel_size == 1
        assert trainer_config.strategy.sequence_parallel is True

        # Check other trainer configurations
        assert trainer_config.limit_val_batches == 32
        assert trainer_config.log_every_n_steps == 10

    def test_pretrain_recipe(self, recipe):
        recipe = recipe.pretrain_recipe()
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == pretrain
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__.__name__ == "MambaModel"
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__ == MockDataModule
        assert recipe.data.seq_length == 8192
        assert recipe.data.global_batch_size == 768

    def test_finetune_recipe(self, recipe):
        recipe = recipe.finetune_recipe(resume_path="dummy_path", vocab_file="dummy_vocab")
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == finetune
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__.__name__ == "MambaModel"
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__.__name__ == "MockDataModule"
        assert recipe.data.seq_length == 8192
        assert recipe.data.global_batch_size == 768

    @pytest.mark.parametrize("num_nodes,num_gpus_per_node", [(32, 8), (16, 16)])
    def test_pretrain_recipe_with_different_configurations(self, recipe, num_nodes, num_gpus_per_node):
        recipe = recipe.pretrain_recipe(num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)
        assert recipe.trainer.num_nodes == num_nodes
        assert recipe.trainer.devices == num_gpus_per_node
