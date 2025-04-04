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
from nemo.collections.vlm import CLIPConfigB32, CLIPModel
from nemo.collections.vlm.recipes import clip_b32
from nemo.lightning import Trainer


class TestClipB32:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        return clip_b32

    def test_model(self, recipe_module):
        model_config = recipe_module.model()
        # Check that the model configuration is a run.Config instance wrapping the CLIPModel
        assert isinstance(model_config, run.Config)
        # Verify that the factory function is the CLIPModel
        assert model_config.__fn_or_cls__ == CLIPModel
        # Verify the inner configuration is a run.Config for CLIPConfigB32
        assert isinstance(model_config.config, run.Config)
        assert model_config.config.__fn_or_cls__ == CLIPConfigB32

    def test_pretrain_recipe_default(self, recipe_module):
        recipe = recipe_module.pretrain_recipe()
        # Check that the returned recipe is a run.Partial wrapping pretrain
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == pretrain

        # Verify the model is correctly set
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == CLIPModel

        # Verify trainer configuration
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert recipe.trainer.accelerator == "gpu"
        assert recipe.trainer.num_nodes == 1
        assert recipe.trainer.devices == 8

        # Verify strategy settings
        strat = recipe.trainer.strategy
        assert isinstance(strat, run.Config)
        assert strat.tensor_model_parallel_size == 1
        assert strat.pipeline_model_parallel_size == 1
        assert strat.encoder_pipeline_model_parallel_size == 0

        # Verify data configuration
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__.__name__ == "MockDataModule"
        assert recipe.data.seq_length == 80
        assert recipe.data.global_batch_size == 128
        assert recipe.data.micro_batch_size == 2
        assert recipe.data.num_workers == 4

        # Verify logging and resume configurations are set
        assert recipe.log is not None
        assert recipe.resume is not None

    @pytest.mark.parametrize("num_nodes,num_gpus", [(1, 8), (2, 4)])
    def test_pretrain_recipe_different_configurations(self, recipe_module, num_nodes, num_gpus):
        recipe = recipe_module.pretrain_recipe(num_nodes=num_nodes, num_gpus_per_node=num_gpus)
        assert recipe.trainer.num_nodes == num_nodes
        assert recipe.trainer.devices == num_gpus
