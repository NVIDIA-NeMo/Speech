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
import torch
from megatron.core.distributed import DistributedDataParallelConfig

from nemo.collections.llm.api import pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.model.mixtral import MixtralConfig8x22B, MixtralModel
from nemo.collections.llm.recipes import mixtral_8x22b_64k
from nemo.lightning import Trainer


class TestMixtral8x22B_64k:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        return mixtral_8x22b_64k

    def test_model(self, recipe_module):
        model_config = recipe_module.model()
        assert isinstance(model_config, run.Config)
        assert model_config.__fn_or_cls__ == MixtralModel
        assert isinstance(model_config.config, run.Config)
        assert model_config.config.__fn_or_cls__ == MixtralConfig8x22B
        assert model_config.config.seq_length == 65536
        assert model_config.config.max_position_embeddings == 4096

    def test_trainer(self, recipe_module):
        trainer_config = recipe_module.trainer()
        assert isinstance(trainer_config, run.Config)
        assert trainer_config.__fn_or_cls__ == Trainer
        assert trainer_config.accelerator == "gpu"
        assert trainer_config.devices == 8
        assert trainer_config.num_nodes == 16

        # Check strategy configuration
        assert isinstance(trainer_config.strategy, run.Config)
        assert trainer_config.strategy.__fn_or_cls__.__name__ == "MegatronStrategy"
        assert trainer_config.strategy.tensor_model_parallel_size == 2
        assert trainer_config.strategy.pipeline_model_parallel_size == 4
        assert trainer_config.strategy.pipeline_dtype == torch.bfloat16
        assert trainer_config.strategy.virtual_pipeline_model_parallel_size == 14
        assert trainer_config.strategy.context_parallel_size == 2
        assert trainer_config.strategy.sequence_parallel is True
        assert trainer_config.strategy.expert_model_parallel_size == 8

        # Check DDP configuration
        assert isinstance(trainer_config.strategy.ddp, run.Config)
        assert trainer_config.strategy.ddp.__fn_or_cls__ == DistributedDataParallelConfig
        assert trainer_config.strategy.ddp.check_for_nan_in_grad is True
        assert trainer_config.strategy.ddp.grad_reduce_in_fp32 is True

    def test_pretrain_recipe(self, recipe_module):
        recipe = recipe_module.pretrain_recipe()
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == pretrain
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == MixtralModel
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__ == MockDataModule
        assert isinstance(recipe.model.config, run.Config)
        assert recipe.data.seq_length == 65536
        assert recipe.data.global_batch_size == 512
        assert recipe.data.micro_batch_size == 1

    @pytest.mark.parametrize("num_nodes,num_gpus_per_node", [(8, 8), (16, 4), (32, 2)])
    def test_pretrain_recipe_with_different_configurations(self, recipe_module, num_nodes, num_gpus_per_node):
        recipe = recipe_module.pretrain_recipe(num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)
        assert recipe.trainer.num_nodes == num_nodes
        assert recipe.trainer.devices == num_gpus_per_node

    def test_valid_trainer_parallelism(self, recipe_module):
        trainer_config = recipe_module.trainer()

        assert isinstance(trainer_config.strategy, run.Config)
        assert trainer_config.strategy.__fn_or_cls__.__name__ == "MegatronStrategy"

        assert (
            trainer_config.strategy.tensor_model_parallel_size
            * trainer_config.strategy.pipeline_model_parallel_size
            * trainer_config.strategy.context_parallel_size
            * trainer_config.strategy.expert_model_parallel_size
            % trainer_config.devices
            == 0
        )
        assert (
            trainer_config.strategy.tensor_model_parallel_size
            * trainer_config.strategy.pipeline_model_parallel_size
            * trainer_config.strategy.context_parallel_size
            * trainer_config.strategy.expert_model_parallel_size
            / trainer_config.devices
            % trainer_config.num_nodes
            == 0
        )

        if trainer_config.strategy.pipeline_model_parallel_size != 1:
            assert trainer_config.strategy.pipeline_dtype is not None

        if trainer_config.strategy.tensor_model_parallel_size == 1:
            assert trainer_config.strategy.sequence_parallel is False

    def test_model_config_parameters(self, recipe_module):
        model_config = recipe_module.model()
        mixtral_config = model_config.config
        assert isinstance(mixtral_config, run.Config)
        assert mixtral_config.num_layers == 56
        assert mixtral_config.hidden_size == 6144
        assert mixtral_config.num_attention_heads == 48
        assert mixtral_config.seq_length == 65536
        assert mixtral_config.max_position_embeddings == 4096
        assert mixtral_config.num_moe_experts == 8
