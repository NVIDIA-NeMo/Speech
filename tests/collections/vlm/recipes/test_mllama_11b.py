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

from nemo.collections.llm.api import finetune
from nemo.collections.vlm import LoRA, MLlamaConfig11BInstruct, MLlamaModel
from nemo.collections.vlm.recipes import mllama_11b
from nemo.lightning import Trainer


class TestMLlama11B:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        # Reference the mllama_11b recipe module.
        return mllama_11b

    def test_model(self, recipe_module):
        model_config = recipe_module.model()
        # Check that model() returns a run.Config wrapping MLlamaModel.
        assert isinstance(model_config, run.Config)
        assert model_config.__fn_or_cls__ == MLlamaModel
        # Check the inner configuration is a run.Config for MLlamaConfig11BInstruct.
        assert isinstance(model_config.config, run.Config)
        assert model_config.config.__fn_or_cls__ == MLlamaConfig11BInstruct

    def test_finetune_recipe_default(self, recipe_module):
        # Default behavior uses peft_scheme 'lora'
        recipe = recipe_module.finetune_recipe()
        # Check the overall recipe is a run.Partial wrapping finetune.
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == finetune

        # Verify model configuration.
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == MLlamaModel

        # Verify trainer configuration.
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert recipe.trainer.accelerator == "gpu"
        # Default num_nodes and devices are 1 and 8, respectively.
        assert recipe.trainer.num_nodes == 1
        assert recipe.trainer.devices == 8

        # Verify the strategy settings.
        strat = recipe.trainer.strategy
        # For the 'lora' case, tensor_model_parallel_size remains as originally set.
        assert strat.tensor_model_parallel_size == 1
        assert strat.pipeline_model_parallel_size == 1
        assert strat.encoder_pipeline_model_parallel_size == 0
        assert strat.pipeline_dtype == torch.bfloat16

        # Verify data configuration.
        data = recipe.data
        assert isinstance(data, run.Config)
        # Check that the data module is MockDataModule and parameters match.
        assert data.__fn_or_cls__.__name__ == "MockDataModule"
        assert data.seq_length == 6404  # Vision encoder sequence length.
        assert data.decoder_seq_length == 2048  # LLM decoder sequence length.
        assert data.global_batch_size == 2
        assert data.micro_batch_size == 1
        assert data.vocab_size == 128256
        assert data.crop_size == (560, 560)
        assert data.num_workers == 0

        # Verify logging and resume configurations are set.
        assert recipe.log is not None
        assert recipe.resume is not None

        # Verify PEFT configuration for LoRA.
        assert hasattr(recipe, "peft")
        assert isinstance(recipe.peft, run.Config)
        assert recipe.peft.__fn_or_cls__ == LoRA
        # Learning rate for LoRA should be set to 1e-4.
        assert recipe.optim.config.lr == 1e-4

    def test_finetune_recipe_peft_none(self, recipe_module):
        # When peft_scheme is set to 'none', PEFT should not be applied.
        recipe = recipe_module.finetune_recipe(peft_scheme="none")
        # No PEFT configuration should be present.
        assert hasattr(recipe, "peft")
        # The strategy tensor model parallel size should be updated.
        assert recipe.trainer.strategy.tensor_model_parallel_size == 2
        # Learning rate should be adjusted accordingly.
        assert recipe.optim.config.lr == 2e-05

    @pytest.mark.parametrize("num_nodes, num_gpus", [(1, 8), (2, 4), (4, 2)])
    def test_finetune_recipe_configurations(self, recipe_module, num_nodes, num_gpus):
        # Test that custom numbers for nodes and GPUs are correctly propagated.
        recipe = recipe_module.finetune_recipe(num_nodes=num_nodes, num_gpus_per_node=num_gpus)
        assert recipe.trainer.num_nodes == num_nodes
        assert recipe.trainer.devices == num_gpus
