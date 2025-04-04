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
from nemo.collections.vlm import LoRA, MLlamaConfig90BInstruct, MLlamaModel
from nemo.collections.vlm.recipes import mllama_90b
from nemo.lightning import Trainer


class TestMLLama90B:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        # Return the module containing the mllama_90b recipe
        return mllama_90b

    def test_model(self, recipe_module):
        model_config = recipe_module.model()
        # Verify that model() returns a run.Config wrapping MLlamaModel
        assert isinstance(model_config, run.Config)
        assert model_config.__fn_or_cls__ == MLlamaModel
        # Verify that the inner configuration is for MLlamaConfig90BInstruct
        assert isinstance(model_config.config, run.Config)
        assert model_config.config.__fn_or_cls__ == MLlamaConfig90BInstruct

    def test_finetune_recipe_default(self, recipe_module):
        recipe = recipe_module.finetune_recipe()
        # Check that the returned recipe is a run.Partial wrapping finetune
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == finetune

        # Verify the model configuration within the recipe
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == MLlamaModel

        # Verify trainer configuration
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert recipe.trainer.accelerator == "gpu"
        # Default values: num_nodes=1 and num_gpus_per_node=8
        assert recipe.trainer.num_nodes == 1
        assert recipe.trainer.devices == 8

        # Check strategy settings from the trainer
        strat = recipe.trainer.strategy
        assert isinstance(strat, run.Config)
        assert strat.tensor_model_parallel_size == 8
        assert strat.pipeline_model_parallel_size == 1
        assert strat.encoder_pipeline_model_parallel_size == 0
        assert strat.pipeline_dtype == torch.bfloat16

        # Verify data configuration
        data = recipe.data
        assert isinstance(data, run.Config)
        # Confirm that the data module is the expected MockDataModule
        assert data.__fn_or_cls__.__name__ == "MockDataModule"
        assert data.seq_length == 6404  # Encoder (vision) sequence length
        assert data.decoder_seq_length == 2048  # Decoder (LLM) sequence length
        assert data.global_batch_size == 16
        assert data.micro_batch_size == 2
        assert data.vocab_size == 128256
        assert data.crop_size == (560, 560)
        assert data.num_workers == 0

        # Check that logging and resume configurations are present
        assert recipe.log is not None
        assert recipe.resume is not None

    def test_finetune_recipe_peft_lora(self, recipe_module):
        # Test with peft_scheme set to "lora"
        recipe = recipe_module.finetune_recipe(peft_scheme="lora")
        # Verify that the recipe contains a peft field configured for LoRA
        assert hasattr(recipe, "peft")
        assert isinstance(recipe.peft, run.Config)
        assert recipe.peft.__fn_or_cls__ == LoRA

        # Check LoRA-specific parameters
        peft_config = recipe.peft
        assert peft_config.freeze_vision_model is True
        expected_modules = ["linear_qkv", "linear_q", "linear_kv"]
        assert peft_config.target_modules == expected_modules
        assert peft_config.dim == 8
        assert peft_config.alpha == 32
        assert peft_config.dropout == 0.05
        assert peft_config.dropout_position == "pre"

        # The learning rate in the optimizer should be updated for LoRA usage
        assert recipe.optim.config.lr == 1e-4

    def test_invalid_peft_scheme(self, recipe_module):
        # When peft_scheme is 'none', the recipe should raise a ValueError.
        with pytest.raises(ValueError):
            recipe_module.finetune_recipe(peft_scheme="none")

    @pytest.mark.parametrize("num_nodes, num_gpus", [(1, 8), (2, 4)])
    def test_finetune_recipe_different_configurations(self, recipe_module, num_nodes, num_gpus):
        # Validate that the recipe respects different numbers of nodes and GPUs per node
        recipe = recipe_module.finetune_recipe(num_nodes=num_nodes, num_gpus_per_node=num_gpus)
        assert recipe.trainer.num_nodes == num_nodes
        assert recipe.trainer.devices == num_gpus
