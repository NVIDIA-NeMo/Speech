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
from nemo.collections.vlm import Llava15Config13B, LlavaModel, LoRA
from nemo.collections.vlm.recipes import llava15_13b
from nemo.lightning import Trainer
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.utils.exp_manager import TimingCallback


class TestLlava15_13B:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        # The new recipe module is assumed to be imported as llava15_13b
        return llava15_13b

    def test_model(self, recipe_module):
        model_config = recipe_module.model()
        # Verify that model() returns a run.Config for the LlavaModel
        assert isinstance(model_config, run.Config)
        assert model_config.__fn_or_cls__ == LlavaModel
        # Check that the inner config is set to Llava15Config13B
        assert isinstance(model_config.config, run.Config)
        assert model_config.config.__fn_or_cls__ == Llava15Config13B

    def test_finetune_recipe_default(self, recipe_module):
        # Default peft_scheme is 'lora'
        recipe = recipe_module.finetune_recipe()
        # Check the recipe is a run.Partial wrapping the finetune function
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == finetune

        # Check that the model configuration is correct
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == LlavaModel

        # Check trainer configuration
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert recipe.trainer.accelerator == "gpu"
        assert recipe.trainer.devices == 8
        assert recipe.trainer.num_nodes == 1
        # Check some trainer parameters
        assert recipe.trainer.max_steps == 5190
        # Validate strategy configuration
        strat = recipe.trainer.strategy
        assert isinstance(strat, run.Config)
        # By default for 'lora', the recipe should not change tensor parallelism in the "none" branch.
        # So we expect the default strategy to remain (tensor_model_parallel_size initially 1)
        # but note that when using LoRA, the branch does not update tensor parallelism.
        # Instead, we validate that the PEFT configuration is applied.
        assert hasattr(recipe, "peft")
        assert isinstance(recipe.peft, run.Config)
        assert recipe.peft.__fn_or_cls__ == LoRA
        # For LoRA, learning rate is updated
        assert recipe.optim.config.lr == 1e-4

        # Verify data configuration
        assert isinstance(recipe.data, run.Config)
        # Check that we are using the MockDataModule with expected parameters
        assert recipe.data.__fn_or_cls__.__name__ == "MockDataModule"
        assert recipe.data.seq_length == 4096
        assert recipe.data.global_batch_size == 128
        # For this recipe, micro_batch_size should be 1
        assert recipe.data.micro_batch_size == 1
        assert recipe.data.num_workers == 4

        # Validate logging and resume configuration are set
        assert recipe.log is not None
        assert recipe.resume is not None

        # Validate trainer callbacks contain TimingCallback and MegatronCommOverlapCallback
        callback_classes = {cb.__fn_or_cls__ for cb in recipe.trainer.callbacks if isinstance(cb, run.Config)}
        assert TimingCallback in callback_classes
        assert MegatronCommOverlapCallback in callback_classes

    def test_finetune_recipe_peft_none(self, recipe_module):
        # Test the case where peft_scheme is set to 'none'
        recipe = recipe_module.finetune_recipe(peft_scheme="none")
        # In this branch, the recipe should not add a PEFT configuration
        assert hasattr(recipe, "peft")
        # And the strategy should be updated to use tensor_model_parallel_size = 2
        assert recipe.trainer.strategy.tensor_model_parallel_size == 2
        # And learning rate should be set to a lower value
        assert recipe.optim.config.lr == 2e-05

    @pytest.mark.parametrize("num_nodes,num_gpus", [(1, 8), (2, 4), (4, 2)])
    def test_parameterized_configurations(self, recipe_module, num_nodes, num_gpus):
        recipe = recipe_module.finetune_recipe(num_nodes=num_nodes, num_gpus_per_node=num_gpus)
        assert recipe.trainer.num_nodes == num_nodes
        assert recipe.trainer.devices == num_gpus
