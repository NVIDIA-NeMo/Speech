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

from nemo.collections.llm.api import finetune
from nemo.collections.vlm import Llava15Config7B, LlavaModel, LoRA
from nemo.collections.vlm.recipes import llava15_7b
from nemo.lightning import Trainer


class TestLlava15_7B:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        # Assuming the new recipe is available in the module "llava15_7b"
        return llava15_7b

    def test_model(self, recipe_module):
        model_config = recipe_module.model()
        # Check that the model configuration is a run.Config instance wrapping the LlavaModel
        assert isinstance(model_config, run.Config)
        # Verify that the factory function is the LlavaModel
        assert model_config.__fn_or_cls__ == LlavaModel
        # Verify the inner configuration is a run.Config for Llava15Config7B
        assert isinstance(model_config.config, run.Config)
        assert model_config.config.__fn_or_cls__ == Llava15Config7B

    def test_finetune_recipe_default(self, recipe_module):
        recipe = recipe_module.finetune_recipe()
        # Check that the returned recipe is a run.Partial wrapping finetune
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == finetune

        # Verify the model is correctly set
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == LlavaModel

        # Verify trainer configuration
        assert isinstance(recipe.trainer, run.Config)
        # Trainer should be the one from nemo.lightning (Trainer)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert recipe.trainer.accelerator == "gpu"
        # Default values: num_nodes=1 and num_gpus_per_node=8
        assert recipe.trainer.num_nodes == 1
        assert recipe.trainer.devices == 8

        # Verify strategy settings (note: for 'none' peft, tensor_model_parallel_size is updated to 2)
        strat = recipe.trainer.strategy
        assert isinstance(strat, run.Config)
        assert strat.tensor_model_parallel_size == 2
        assert strat.pipeline_model_parallel_size == 1
        # encoder_pipeline_model_parallel_size is set to 0 and sequence_parallel should be True
        assert strat.encoder_pipeline_model_parallel_size == 0
        assert strat.sequence_parallel is True

        # Verify data configuration
        assert isinstance(recipe.data, run.Config)
        # The recipe uses the MockDataModule with the following parameters:
        assert recipe.data.__fn_or_cls__.__name__ == "MockDataModule"
        assert recipe.data.seq_length == 4096
        assert recipe.data.global_batch_size == 128
        assert recipe.data.micro_batch_size == 2
        assert recipe.data.num_workers == 4

        # Verify logging and resume configurations are set (non-null)
        assert recipe.log is not None
        assert recipe.resume is not None

    def test_finetune_recipe_peft_lora(self, recipe_module):
        # Test the fine-tuning recipe with peft_scheme set to "lora"
        recipe = recipe_module.finetune_recipe(peft_scheme="lora")
        # In this case, a peft field should be present and configured for LoRA
        assert hasattr(recipe, "peft")
        assert isinstance(recipe.peft, run.Config)
        assert recipe.peft.__fn_or_cls__ == LoRA

        # The learning rate should have been updated for LoRA usage
        assert recipe.optim.config.lr == 1e-4

    @pytest.mark.parametrize("num_nodes,num_gpus", [(1, 8), (2, 4)])
    def test_finetune_recipe_different_configurations(self, recipe_module, num_nodes, num_gpus):
        # Verify that the recipe honors different numbers of nodes and GPUs per node
        recipe = recipe_module.finetune_recipe(num_nodes=num_nodes, num_gpus_per_node=num_gpus)
        assert recipe.trainer.num_nodes == num_nodes
        assert recipe.trainer.devices == num_gpus
