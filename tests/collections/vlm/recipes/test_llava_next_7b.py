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
from nemo.collections.vlm import LlavaNextConfig7B, LlavaNextModel, LoRA
from nemo.collections.vlm.recipes import llava_next_7b
from nemo.lightning import Trainer


class TestLlavaNext7B:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        return llava_next_7b

    def test_model(self, recipe_module):
        model_config = recipe_module.model()
        # Check that the model configuration is a run.Config instance wrapping the LlavaNextModel
        assert isinstance(model_config, run.Config)
        # Verify that the factory function is the LlavaNextModel
        assert model_config.__fn_or_cls__ == LlavaNextModel
        # Verify the inner configuration is a run.Config for LlavaNextConfig7B
        assert isinstance(model_config.config, run.Config)
        assert model_config.config.__fn_or_cls__ == LlavaNextConfig7B

    def test_finetune_recipe_default(self, recipe_module):
        recipe = recipe_module.finetune_recipe()
        # Check that the returned recipe is a run.Partial wrapping finetune
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == finetune

        # Verify the model is correctly set
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == LlavaNextModel

        # Verify trainer configuration
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert recipe.trainer.accelerator == "gpu"
        assert recipe.trainer.num_nodes == 1
        assert recipe.trainer.devices == 8

        # Verify strategy settings
        strat = recipe.trainer.strategy
        assert isinstance(strat, run.Config)
        assert strat.tensor_model_parallel_size == 2
        assert strat.pipeline_model_parallel_size == 1
        assert strat.encoder_pipeline_model_parallel_size == 0

        # Verify data configuration
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__.__name__ == "MockDataModule"
        assert recipe.data.seq_length == 4096
        assert recipe.data.global_batch_size == 8
        assert recipe.data.micro_batch_size == 2
        assert recipe.data.num_workers == 4

        # Verify logging and resume configurations are set
        assert recipe.log is not None
        assert recipe.resume is not None

    def test_finetune_recipe_peft_lora(self, recipe_module):
        recipe = recipe_module.finetune_recipe(peft_scheme="lora")
        # Verify LoRA configuration is present and correct
        assert hasattr(recipe, "peft")
        assert isinstance(recipe.peft, run.Config)
        assert recipe.peft.__fn_or_cls__ == LoRA
        assert recipe.optim.config.lr == 1e-4

    def test_pretrain_recipe_default(self, recipe_module):
        recipe = recipe_module.pretrain_recipe()
        # Check that the returned recipe is a run.Partial wrapping pretrain
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == pretrain

        # Verify model configuration
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == LlavaNextModel
        assert recipe.model.config.freeze_language_model is True
        assert recipe.model.config.freeze_vision_model is True
        assert recipe.model.config.freeze_vision_projection is False

        # Verify data configuration
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__.__name__ == "MockDataModule"
        assert recipe.data.seq_length == 4096
        assert recipe.data.global_batch_size == 8

    @pytest.mark.parametrize("num_nodes,num_gpus", [(1, 8), (2, 4)])
    def test_recipe_different_configurations(self, recipe_module, num_nodes, num_gpus):
        finetune_recipe = recipe_module.finetune_recipe(num_nodes=num_nodes, num_gpus_per_node=num_gpus)
        assert finetune_recipe.trainer.num_nodes == num_nodes
        assert finetune_recipe.trainer.devices == num_gpus

        pretrain_recipe = recipe_module.pretrain_recipe(num_nodes=num_nodes, num_gpus_per_node=num_gpus)
        assert pretrain_recipe.trainer.num_nodes == num_nodes
        assert pretrain_recipe.trainer.devices == num_gpus
