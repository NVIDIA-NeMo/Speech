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

from nemo.collections.avlm import AVLMConfig8B, AVLMModel
from nemo.collections.avlm.recipes import avlm_8b
from nemo.collections.llm.api import finetune
from nemo.collections.llm.peft import LoRA
from nemo.lightning import Trainer


class TestAVLM8B:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        return avlm_8b

    def test_model(self, recipe_module):
        model_config = recipe_module.model()
        # Check that the model configuration is a run.Config instance wrapping the AVLMModel
        assert isinstance(model_config, run.Config)
        assert model_config.__fn_or_cls__ == AVLMModel
        # Verify the inner configuration is a run.Config for AVLMConfig8B
        assert isinstance(model_config.config, run.Config)
        assert model_config.config.__fn_or_cls__ == AVLMConfig8B

    def test_finetune_recipe_default(self, recipe_module):
        # Provide freeze_modules dict as required by avlm_8b.finetune_recipe
        freeze_modules = {
            "freeze_language_model": True,
            "freeze_vision_model": True,
            "freeze_audio_model": True,
            "freeze_vision_projection": True,
            "freeze_audio_projection": True,
        }
        recipe = recipe_module.finetune_recipe(freeze_modules=freeze_modules)
        # Check that the returned recipe is a run.Partial wrapping finetune
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == finetune

        # Verify the model is correctly set
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == AVLMModel

        # Verify trainer configuration
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert recipe.trainer.accelerator == "gpu"
        assert recipe.trainer.num_nodes == 1
        assert recipe.trainer.devices == 8

        # Verify strategy settings
        strat = recipe.trainer.strategy
        assert isinstance(strat, run.Config)
        assert strat.tensor_model_parallel_size == 2 or strat.tensor_model_parallel_size == 4
        assert strat.pipeline_model_parallel_size == 1
        assert strat.encoder_pipeline_model_parallel_size == 0

        # Verify data configuration
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__.__name__ == "MockDataModule"
        assert recipe.data.seq_length == 8192
        assert recipe.data.global_batch_size == 8
        assert recipe.data.micro_batch_size == 2
        assert recipe.data.num_workers == 4

        # Verify logging and resume configurations are set (non-null)
        assert recipe.log is not None
        assert recipe.resume is not None

    def test_finetune_recipe_peft_lora(self, recipe_module):
        freeze_modules = {
            "freeze_language_model": True,
            "freeze_vision_model": True,
            "freeze_audio_model": True,
            "freeze_vision_projection": True,
            "freeze_audio_projection": True,
        }
        # Test the fine-tuning recipe with peft_scheme set to "lora"
        recipe = recipe_module.finetune_recipe(peft_scheme="lora", freeze_modules=freeze_modules)
        # In this case, a peft field should be present and configured for LoRA
        assert hasattr(recipe, "peft")
        assert isinstance(recipe.peft, run.Config)
        assert recipe.peft.__fn_or_cls__ == LoRA

        # The learning rate should have been updated for LoRA usage
        assert recipe.optim.config.lr == 1e-4

    @pytest.mark.parametrize("num_nodes,num_gpus", [(1, 8), (2, 4)])
    def test_finetune_recipe_different_configurations(self, recipe_module, num_nodes, num_gpus):
        freeze_modules = {
            "freeze_language_model": True,
            "freeze_vision_model": True,
            "freeze_audio_model": True,
            "freeze_vision_projection": True,
            "freeze_audio_projection": True,
        }
        # Verify that the recipe honors different numbers of nodes and GPUs per node
        recipe = recipe_module.finetune_recipe(
            num_nodes=num_nodes, num_gpus_per_node=num_gpus, freeze_modules=freeze_modules
        )
        assert recipe.trainer.num_nodes == num_nodes
        assert recipe.trainer.devices == num_gpus


class TestAVLM8BPretrain:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        return avlm_8b

    def test_pretrain_recipe_default(self, recipe_module):
        freeze_modules = {
            "freeze_language_model": True,
            "freeze_vision_model": True,
            "freeze_audio_model": True,
            "freeze_vision_projection": True,
            "freeze_audio_projection": True,
        }
        recipe = recipe_module.pretrain_recipe(freeze_modules=freeze_modules)
        # Check that the returned recipe is a run.Partial
        assert isinstance(recipe, run.Partial)

        # Check model config
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == AVLMModel
        assert isinstance(recipe.model.config, run.Config)
        assert recipe.model.config.__fn_or_cls__ == AVLMConfig8B

        # Check trainer config
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert recipe.trainer.accelerator == "gpu"
        assert recipe.trainer.num_nodes == 1
        assert recipe.trainer.devices == 8

        # Check strategy config
        strat = recipe.trainer.strategy
        assert isinstance(strat, run.Config)
        assert strat.tensor_model_parallel_size == 4
        assert strat.pipeline_model_parallel_size == 1
        assert strat.encoder_pipeline_model_parallel_size == 0

        # Check data config
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__.__name__ == "MockDataModule"
        assert recipe.data.seq_length == 8192
        assert recipe.data.global_batch_size == 8
        assert recipe.data.micro_batch_size == 1
        assert recipe.data.num_workers == 4

        # Check log and optim
        assert recipe.log is not None
        assert recipe.optim is not None

    @pytest.mark.parametrize("num_nodes,num_gpus", [(1, 8), (2, 4)])
    def test_pretrain_recipe_different_configurations(self, recipe_module, num_nodes, num_gpus):
        freeze_modules = {
            "freeze_language_model": True,
            "freeze_vision_model": True,
            "freeze_audio_model": True,
            "freeze_vision_projection": True,
            "freeze_audio_projection": True,
        }
        recipe = recipe_module.pretrain_recipe(
            num_nodes=num_nodes, num_gpus_per_node=num_gpus, freeze_modules=freeze_modules
        )
        assert recipe.trainer.num_nodes == num_nodes
        assert recipe.trainer.devices == num_gpus

    def test_pretrain_recipe_language_model_from_pretrained(self, recipe_module):
        freeze_modules = {
            "freeze_language_model": True,
            "freeze_vision_model": True,
            "freeze_audio_model": True,
            "freeze_vision_projection": True,
            "freeze_audio_projection": True,
        }
        lm_path = "/some/local/path"
        recipe = recipe_module.pretrain_recipe(language_model_from_pretrained=lm_path, freeze_modules=freeze_modules)
        # Check that the config is set correctly
        assert recipe.model.config.language_model_from_pretrained == lm_path
