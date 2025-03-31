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
import torch

from nemo.collections.llm import Llama31Nemotron70BConfig, LlamaNemotronModel
from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.peft.lora import LoRA
from nemo.collections.llm.recipes import llama31_nemotron_70b
from nemo.lightning import Trainer


class TestLlama31Nemotron70B:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        return llama31_nemotron_70b

    def test_model(self, recipe_module):
        model_config = recipe_module.model()
        assert isinstance(model_config, run.Config)
        assert model_config.__fn_or_cls__ == LlamaNemotronModel
        assert isinstance(model_config.config, run.Config)
        assert model_config.config.__fn_or_cls__ == Llama31Nemotron70BConfig
        assert model_config.config.seq_length == 8192

    def test_trainer(self, recipe_module):
        trainer_config = recipe_module.trainer()
        assert isinstance(trainer_config, run.Config)
        assert trainer_config.__fn_or_cls__ == Trainer
        assert trainer_config.accelerator == "gpu"
        assert trainer_config.devices == 8
        assert trainer_config.num_nodes == 4
        assert trainer_config.max_steps == 1168251

        # Check strategy configuration
        assert isinstance(trainer_config.strategy, run.Config)
        assert trainer_config.strategy.__fn_or_cls__.__name__ == "MegatronStrategy"
        assert trainer_config.strategy.tensor_model_parallel_size == 4
        assert trainer_config.strategy.pipeline_model_parallel_size == 4
        assert trainer_config.strategy.pipeline_dtype == torch.bfloat16
        assert trainer_config.strategy.virtual_pipeline_model_parallel_size == 5
        assert trainer_config.strategy.context_parallel_size == 2
        assert trainer_config.strategy.sequence_parallel is True
        assert trainer_config.strategy.gradient_as_bucket_view is True
        assert trainer_config.strategy.ckpt_async_save is True
        assert trainer_config.strategy.ckpt_parallel_load is True

        # Check other trainer configurations
        assert trainer_config.accumulate_grad_batches == 1
        assert trainer_config.limit_test_batches == 50
        assert trainer_config.limit_val_batches == 32
        assert trainer_config.log_every_n_steps == 10
        assert trainer_config.use_distributed_sampler is False
        assert trainer_config.val_check_interval == 2000

    def test_pretrain_recipe(self, recipe_module):
        recipe = recipe_module.pretrain_recipe()
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == pretrain
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == LlamaNemotronModel
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__ == MockDataModule
        assert recipe.data.seq_length == 8192
        assert recipe.data.global_batch_size == 512
        assert recipe.data.micro_batch_size == 1

    def test_finetune_recipe(self, recipe_module):
        recipe = recipe_module.finetune_recipe()
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == finetune
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == LlamaNemotronModel
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert recipe.trainer.strategy.tensor_model_parallel_size == 8
        assert isinstance(recipe.peft, run.Config)
        assert recipe.peft.__fn_or_cls__ == LoRA
        assert recipe.peft.dim == 16
        assert recipe.peft.alpha == 32

    @pytest.mark.parametrize("model_type", ["instruct", "reward"])
    def test_finetune_model_types(self, recipe_module, model_type):
        recipe = recipe_module.finetune_recipe(model_type=model_type)
        expected_path = f"nvidia/Llama-3.1-Nemotron-70B-{model_type.capitalize()}-HF"
        assert recipe.resume.restore_config.path == "nemo://" + expected_path

    def test_invalid_model_type(self, recipe_module):
        with pytest.raises(ValueError, match="Unknown model_type"):
            recipe_module.finetune_recipe(model_type="invalid")

    def test_invalid_peft_scheme(self, recipe_module):
        with pytest.raises(ValueError, match="Unrecognized peft scheme"):
            recipe_module.finetune_recipe(peft_scheme="invalid")

    def test_pretrain_performance_optimizations(self, recipe_module):
        recipe = recipe_module.pretrain_recipe(performance_mode=True)
        assert any(cb.__fn_or_cls__.__name__ == "MegatronCommOverlapCallback" for cb in recipe.trainer.callbacks)

    def test_finetune_performance_optimizations(self, recipe_module):
        recipe = recipe_module.finetune_recipe(performance_mode=True)
        assert recipe.trainer.strategy.sequence_parallel is True
