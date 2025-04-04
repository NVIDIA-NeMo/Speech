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

from nemo.collections.llm.api import pretrain
from nemo.collections.vlm import NevaModel
from nemo.collections.vlm.recipes import neva_llama3_8b
from nemo.collections.vlm.recipes.neva_llama3_8b import NevaConfig8B
from nemo.lightning import Trainer
from nemo.lightning.pytorch.callbacks.garbage_collection import GarbageCollectionCallback
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback


class TestNevaLlama38B:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        return neva_llama3_8b

    def test_model(self, recipe_module):
        model_config = recipe_module.model()
        # Check that the model configuration is a run.Config instance wrapping the NevaModel
        assert isinstance(model_config, run.Config)
        # Verify that the factory function is the NevaModel
        assert model_config.__fn_or_cls__ == NevaModel
        # Verify the inner configuration is a run.Config for NevaConfig8B
        assert isinstance(model_config.config, run.Config)
        assert model_config.config.__fn_or_cls__ == NevaConfig8B

    def test_trainer_config(self, recipe_module):
        trainer_config = recipe_module.trainer()
        # Verify trainer configuration
        assert isinstance(trainer_config, run.Config)
        assert trainer_config.__fn_or_cls__ == Trainer
        assert trainer_config.accelerator == "gpu"
        assert trainer_config.num_nodes == 1
        assert trainer_config.devices == 8

        # Verify strategy settings
        strat = trainer_config.strategy
        assert isinstance(strat, run.Config)
        assert strat.tensor_model_parallel_size == 1
        assert strat.pipeline_model_parallel_size == 1
        assert strat.context_parallel_size == 2

    def test_finetune_recipe_default(self, recipe_module):
        recipe = recipe_module.finetune_recipe()
        # Check that the returned recipe is a run.Partial wrapping pretrain
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == pretrain

        # Verify the model is correctly set
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == NevaModel

        # Verify data configuration
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__.__name__ == "MockDataModule"
        assert recipe.data.seq_length == 8192
        assert recipe.data.global_batch_size == 512
        assert recipe.data.micro_batch_size == 1
        assert recipe.data.num_workers == 4

        # Verify logging and resume configurations are set
        assert recipe.log is not None
        assert recipe.resume is not None

    def test_finetune_recipe_performance_mode(self, recipe_module):
        recipe = recipe_module.finetune_recipe(performance_mode=True)
        # Verify performance optimizations are applied
        assert any(
            isinstance(callback, run.Config) and callback.__fn_or_cls__ == MegatronCommOverlapCallback
            for callback in recipe.trainer.callbacks
        )
        assert any(
            isinstance(callback, run.Config) and callback.__fn_or_cls__ == GarbageCollectionCallback
            for callback in recipe.trainer.callbacks
        )

    @pytest.mark.parametrize("num_nodes,num_gpus", [(1, 8), (2, 4)])
    def test_recipe_different_configurations(self, recipe_module, num_nodes, num_gpus):
        recipe = recipe_module.finetune_recipe(num_nodes=num_nodes, num_gpus_per_node=num_gpus)
        assert recipe.trainer.num_nodes == num_nodes
        assert recipe.trainer.devices == num_gpus

        trainer_config = recipe_module.trainer(num_nodes=num_nodes, num_gpus_per_node=num_gpus)
        assert trainer_config.num_nodes == num_nodes
        assert trainer_config.devices == num_gpus
