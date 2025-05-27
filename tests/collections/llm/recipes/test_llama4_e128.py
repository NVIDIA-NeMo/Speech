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
from nemo.collections.llm.gpt.model.llama import Llama4Experts128Config, LlamaModel
from nemo.collections.llm.peft import DoRA
from nemo.collections.llm.peft.lora import LoRA
from nemo.collections.llm.recipes import llama4_e128
from nemo.lightning import Trainer


class TestLlama4_E128:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        return llama4_e128

    def test_model(self, recipe_module):
        model_config = recipe_module.model()
        assert isinstance(model_config, run.Config)
        assert model_config.__fn_or_cls__ == LlamaModel
        assert isinstance(model_config.config, run.Config)
        assert model_config.config.__fn_or_cls__ == Llama4Experts128Config

    def test_trainer(self, recipe_module):
        trainer_config = recipe_module.trainer()
        assert isinstance(trainer_config, run.Config)
        assert trainer_config.__fn_or_cls__ == Trainer
        assert trainer_config.accelerator == "gpu"
        assert trainer_config.devices == 8
        assert trainer_config.num_nodes == 64
        assert trainer_config.max_steps == 1168251

        # Check strategy configuration
        assert isinstance(trainer_config.strategy, run.Config)
        assert trainer_config.strategy.__fn_or_cls__.__name__ == "MegatronStrategy"
        assert trainer_config.strategy.tensor_model_parallel_size == 4
        assert trainer_config.strategy.pipeline_model_parallel_size == 1
        assert trainer_config.strategy.pipeline_dtype is None
        assert trainer_config.strategy.virtual_pipeline_model_parallel_size is None
        assert trainer_config.strategy.context_parallel_size == 1
        assert trainer_config.strategy.sequence_parallel is True
        assert trainer_config.strategy.expert_tensor_parallel_size == 4
        assert trainer_config.strategy.expert_model_parallel_size == 128
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

        # Check plugins
        assert isinstance(trainer_config.plugins, run.Config)
        assert trainer_config.plugins.__fn_or_cls__.__name__ == "MegatronMixedPrecision"

    def test_pretrain_recipe(self, recipe_module):
        recipe = recipe_module.pretrain_recipe()
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == pretrain
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == LlamaModel
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__ == MockDataModule
        assert recipe.data.seq_length == 8192
        assert recipe.data.global_batch_size == 512

    def test_finetune_recipe(self, recipe_module):
        recipe = recipe_module.finetune_recipe()
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == finetune
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == LlamaModel
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__ == SquadDataModule
        assert recipe.data.seq_length == 2048
        assert recipe.data.global_batch_size == 128
        assert isinstance(recipe.peft, run.Config)
        assert recipe.peft.__fn_or_cls__ == LoRA

    @pytest.mark.parametrize("num_nodes,num_gpus_per_node", [(1, 8), (2, 4), (4, 2)])
    def test_pretrain_recipe_with_different_configurations(self, recipe_module, num_nodes, num_gpus_per_node):
        recipe = recipe_module.pretrain_recipe(num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)
        assert recipe.trainer.num_nodes == num_nodes
        assert recipe.trainer.devices == num_gpus_per_node

    def test_pretrain_performance_optimizations(self, recipe_module):
        recipe = recipe_module.pretrain_recipe(performance_mode=True)
        assert any(cb.__fn_or_cls__.__name__ == "MegatronCommOverlapCallback" for cb in recipe.trainer.callbacks)
        assert any(cb.__fn_or_cls__.__name__ == "GarbageCollectionCallback" for cb in recipe.trainer.callbacks)
        assert any(cb.__fn_or_cls__.__name__ == "MegatronTokenDropCallback" for cb in recipe.trainer.callbacks)
        assert recipe.trainer.plugins.grad_reduce_in_fp32 is False

    def test_trainer_parallelism_options(self, recipe_module):
        trainer_config = recipe_module.trainer(
            tensor_parallelism=2,
            pipeline_parallelism=2,
            context_parallelism=4,
            sequence_parallelism=True,
            expert_tensor_parallelism=2,
            expert_model_parallelism=64,
        )
        assert trainer_config.strategy.tensor_model_parallel_size == 2
        assert trainer_config.strategy.pipeline_model_parallel_size == 2
        assert trainer_config.strategy.context_parallel_size == 4
        assert trainer_config.strategy.sequence_parallel is True
        assert trainer_config.strategy.expert_tensor_parallel_size == 2
        assert trainer_config.strategy.expert_model_parallel_size == 64

    def test_finetune_performance_optimizations(self, recipe_module):
        recipe = recipe_module.finetune_recipe(performance_mode=True)
        assert recipe.trainer.strategy.tensor_model_parallel_size == 1
        assert any(cb.__fn_or_cls__.__name__ == "MegatronCommOverlapCallback" for cb in recipe.trainer.callbacks)
        assert any(cb.__fn_or_cls__.__name__ == "GarbageCollectionCallback" for cb in recipe.trainer.callbacks)
        assert recipe.trainer.plugins.grad_reduce_in_fp32 is False

    def test_finetune_peft_options(self, recipe_module):
        # Test LoRA configuration
        recipe = recipe_module.finetune_recipe(peft_scheme='lora')
        assert recipe.peft.__fn_or_cls__ == LoRA
        assert recipe.peft.dim == 8
        assert recipe.peft.alpha == 16
        assert recipe.optim.config.lr == 1e-4

        # Test DoRA configuration
        recipe = recipe_module.finetune_recipe(peft_scheme='dora')
        assert recipe.peft.__fn_or_cls__ == DoRA
        assert recipe.peft.dim == 8
        assert recipe.peft.alpha == 16
        assert recipe.optim.config.lr == 1e-4

        # Test no PEFT configuration
        recipe = recipe_module.finetune_recipe(peft_scheme=None)
        assert recipe.optim.config.lr == 5e-6

        # Test invalid PEFT scheme
        with pytest.raises(ValueError):
            recipe_module.finetune_recipe(peft_scheme='invalid')

    def test_packed_sequence_options(self, recipe_module):
        # Test with packed sequences
        recipe = recipe_module.finetune_recipe(packed_sequence=True)
        assert recipe.data.seq_length == 4096
        assert recipe.data.packed_sequence_specs is not None
        assert recipe.data.dataset_kwargs['pad_to_max_length'] is True

        # Test without packed sequences
        recipe = recipe_module.finetune_recipe(packed_sequence=False)
        assert recipe.data.seq_length == 2048
