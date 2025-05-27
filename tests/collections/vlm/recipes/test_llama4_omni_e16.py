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
from nemo.collections.vlm import Llama4OmniModel, Llama4ScoutExperts16Config
from nemo.collections.vlm.recipes import llama4_omni_e16
from nemo.lightning import Trainer


class TestLlama4OmniE16:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        # Reference the llama4_omni_e16 recipe module.
        return llama4_omni_e16

    def test_model(self, recipe_module):
        model_config = recipe_module.model()
        # Check that model() returns a run.Config wrapping Llama4OmniModel.
        assert isinstance(model_config, run.Config)
        assert model_config.__fn_or_cls__ == Llama4OmniModel
        # Check the inner configuration is a run.Config for Llama4ScoutExperts16Config.
        assert isinstance(model_config.config, run.Config)
        assert model_config.config.__fn_or_cls__ == Llama4ScoutExperts16Config

    def test_pretrain_recipe_default(self, recipe_module):
        recipe = recipe_module.pretrain_recipe()
        # Check the overall recipe is a run.Partial wrapping pretrain.
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == pretrain

        # Verify model configuration.
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == Llama4OmniModel

        # Verify trainer configuration.
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert recipe.trainer.accelerator == "gpu"
        # Default num_nodes and devices are 32 and 8, respectively.
        assert recipe.trainer.num_nodes == 32
        assert recipe.trainer.devices == 8

        # Verify the strategy settings.
        strat = recipe.trainer.strategy
        assert strat.tensor_model_parallel_size == 4
        assert strat.pipeline_model_parallel_size == 1
        assert strat.context_parallel_size == 1
        assert strat.sequence_parallel is True
        assert strat.expert_tensor_parallel_size == 4
        assert strat.expert_model_parallel_size == 16

        # Verify data configuration.
        data = recipe.data
        assert isinstance(data, run.Config)
        assert data.__fn_or_cls__.__name__ == "MockDataModule"
        assert data.seq_length == 8192
        assert data.global_batch_size == 512
        assert data.micro_batch_size == 1

        # Verify logging and resume configurations are set.
        assert recipe.log is not None
        assert recipe.resume is not None

    def test_pretrain_recipe_performance_mode(self, recipe_module):
        recipe = recipe_module.pretrain_recipe(performance_mode=True)
        # Check that performance optimizations are applied
        assert any(
            isinstance(cb, run.Config) and cb.__fn_or_cls__.__name__ == "GarbageCollectionCallback"
            for cb in recipe.trainer.callbacks
        )
        assert any(
            isinstance(cb, run.Config) and cb.__fn_or_cls__.__name__ == "MegatronCommOverlapCallback"
            for cb in recipe.trainer.callbacks
        )
        assert any(
            isinstance(cb, run.Config) and cb.__fn_or_cls__.__name__ == "MegatronTokenDropCallback"
            for cb in recipe.trainer.callbacks
        )
        assert recipe.trainer.plugins.grad_reduce_in_fp32 is False

    def test_finetune_recipe_default(self, recipe_module):
        recipe = recipe_module.finetune_recipe()
        # Check the overall recipe is a run.Partial wrapping finetune.
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == finetune

        # Verify model configuration.
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == Llama4OmniModel

        # Verify trainer configuration.
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert recipe.trainer.accelerator == "gpu"
        # Default num_nodes and devices are 32 and 8, respectively.
        assert recipe.trainer.num_nodes == 32
        assert recipe.trainer.devices == 8

        # Verify the strategy settings.
        strat = recipe.trainer.strategy
        assert strat.tensor_model_parallel_size == 8
        assert strat.pipeline_model_parallel_size == 4
        assert strat.sequence_parallel is True
        assert strat.expert_tensor_parallel_size == 4
        assert strat.expert_model_parallel_size == 16

        # Verify data configuration.
        data = recipe.data
        assert isinstance(data, run.Config)
        assert data.__fn_or_cls__.__name__ == "MockDataModule"
        assert data.seq_length == 8192
        assert data.global_batch_size == 128
        assert data.micro_batch_size == 1

        # Verify logging and resume configurations are set.
        assert recipe.log is not None
        assert recipe.resume is not None

    def test_finetune_recipe_peft_none(self, recipe_module):
        recipe = recipe_module.finetune_recipe(peft_scheme="none")
        # Verify strategy settings for no PEFT case
        assert recipe.trainer.strategy.sequence_parallel is True
        assert recipe.optim.config.lr == 2e-05

    def test_finetune_recipe_peft_lora(self, recipe_module):
        recipe = recipe_module.finetune_recipe(peft_scheme="lora")
        # Verify strategy settings for LoRA case
        assert recipe.trainer.strategy.sequence_parallel is True
        assert hasattr(recipe, "peft")
        assert recipe.optim.config.lr == 1e-4

    @pytest.mark.parametrize("num_nodes, num_gpus", [(1, 8), (2, 4), (4, 2)])
    def test_finetune_recipe_configurations(self, recipe_module, num_nodes, num_gpus):
        # Test that custom numbers for nodes and GPUs are correctly propagated.
        recipe = recipe_module.finetune_recipe(num_nodes=num_nodes, num_gpus_per_node=num_gpus)
        assert recipe.trainer.num_nodes == num_nodes
        assert recipe.trainer.devices == num_gpus

    def test_finetune_recipe_invalid_peft(self, recipe_module):
        # Test that invalid PEFT scheme raises ValueError
        with pytest.raises(ValueError, match="Unrecognized peft scheme"):
            recipe_module.finetune_recipe(peft_scheme="invalid_scheme")
