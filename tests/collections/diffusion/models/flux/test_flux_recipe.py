# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

import pytest
import torch

from nemo.collections.diffusion.models.flux_controlnet.layers import ControlNetConditioningEmbedding
from nemo.collections.diffusion.recipes import flux_12b, flux_535m
from nemo.collections.llm.api import pretrain


class TestFluxRecipe:
    @pytest.mark.unit
    def test_flux_12b(self):
        recipe = flux_12b.pretrain_recipe(name="flux_12b_pretrain", num_nodes=1)

        # Check trainer configuration
        assert recipe.trainer.num_nodes == 1
        assert recipe.trainer.devices == 8

        # Check optimizer settings
        assert recipe.optim.config.lr == 1e-4
        assert recipe.optim.config.bf16 is True

        assert recipe.data.global_batch_size == 2
        assert recipe.data.micro_batch_size == 1

    @pytest.mark.unit
    def test_flux_535m(self):
        recipe = flux_535m.unit_test_recipe(name="flux_535m", num_gpus_per_node=1)

        # Check trainer configuration
        assert recipe.trainer.num_nodes == 1
        assert recipe.trainer.devices == 1

        # Check optimizer settings
        assert recipe.optim.config.lr == 1e-4
        assert recipe.optim.config.bf16 is True
