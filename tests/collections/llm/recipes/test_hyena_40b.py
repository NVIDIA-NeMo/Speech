# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024 Arc Institute. All rights reserved.
# Copyright (c) 2024 Michael Poli. All rights reserved.
# Copyright (c) 2024 Stanford University. All rights reserved
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

from nemo.collections.llm.recipes.hyena_40b import finetune_recipe, model, pretrain_recipe, tokenizer


class TestHyena40B:
    def test_tokenizer(self):
        """
        Test the tokenizer factory function.
        """
        tokenizer_config = tokenizer()
        assert isinstance(tokenizer_config, run.Config)
        assert tokenizer_config.__fn_or_cls__.__name__ == "get_nmt_tokenizer"

    def test_model_default(self):
        """
        Test the model factory function with default parameters.
        """
        model_config = model()
        assert isinstance(model_config, run.Config)
        assert model_config.__fn_or_cls__.__name__ == "HyenaModel"
        assert model_config.config.seq_length == 8192
        assert model_config.config.tp_comm_overlap is False

    @pytest.mark.parametrize("tp_comm_overlap,seq_length", [(True, 4096), (False, 16384)])
    def test_model_with_parameters(self, tp_comm_overlap, seq_length):
        """
        Test the model factory function with different parameter values.
        """
        model_config = model(tp_comm_overlap=tp_comm_overlap, seq_length=seq_length)
        assert model_config.config.tp_comm_overlap == tp_comm_overlap
        assert model_config.config.seq_length == seq_length

    def test_pretrain_recipe_default(self):
        """
        Test the pretrain_recipe factory function with default parameters.
        """
        recipe = pretrain_recipe()
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__.__name__ == "pretrain"
        assert recipe.trainer.num_nodes == 4
        assert recipe.trainer.devices == 8

    @pytest.mark.parametrize("num_nodes,num_gpus_per_node", [(1, 4), (2, 8)])
    def test_pretrain_recipe_with_parameters(self, num_nodes, num_gpus_per_node):
        """
        Test the pretrain_recipe factory function with different configurations.
        """
        recipe = pretrain_recipe(num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)
        assert recipe.trainer.num_nodes == num_nodes
        assert recipe.trainer.devices == num_gpus_per_node

    def test_finetune_recipe_with_parameters(self):
        """
        Test the finetune_recipe factory function with valid parameters.
        """
        recipe = finetune_recipe(resume_path="dummy_path")
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__.__name__ == "finetune"
        assert recipe.trainer.num_nodes == 4
        assert recipe.trainer.devices == 8

    def test_tokenizer_docstring(self):
        """
        Ensure the tokenizer function has a docstring.
        """
        assert tokenizer.__doc__ is not None

    def test_model_docstring(self):
        """
        Ensure the model function has a docstring.
        """
        assert model.__doc__ is not None

    def test_pretrain_recipe_docstring(self):
        """
        Ensure the pretrain_recipe function has a docstring.
        """
        assert pretrain_recipe.__doc__ is not None

    def test_finetune_recipe_docstring(self):
        """
        Ensure the finetune_recipe function has a docstring.
        """
        assert finetune_recipe.__doc__ is not None
