# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
import importlib.util
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo.collections.speechlm2.parts.parallel import AutomodelParallelStrategy
from nemo.utils.trainer_utils import resolve_trainer_cfg

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "distributed_oomptimizer_for_test", _REPO_ROOT / "scripts/speechlm2/distributed_oomptimizer.py"
)
_OOMPTIMIZER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_OOMPTIMIZER)


def test_single_process_trainer_cfg_keeps_recipe_strategy():
    # The Automodel strategy is what casts SALMAutomodel's perception module to the training dtype on a single GPU.
    # Dropping it profiles fp32 trainable parameters and breaks the recipe's FlashAdamW optimizer.
    cfg = OmegaConf.load(_REPO_ROOT / "examples/speechlm2/conf/salm_automodel.yaml")
    trainer_cfg = _OOMPTIMIZER._single_process_trainer_cfg(resolve_trainer_cfg(cfg.trainer))

    assert isinstance(trainer_cfg["strategy"], AutomodelParallelStrategy)
    assert trainer_cfg["devices"] == 1
    assert trainer_cfg["num_nodes"] == 1


def test_single_process_trainer_cfg_without_strategy():
    trainer_cfg = _OOMPTIMIZER._single_process_trainer_cfg(
        {"devices": 8, "num_nodes": 2, "accelerator": "gpu", "max_steps": 1}
    )

    assert "strategy" not in trainer_cfg
    assert trainer_cfg == {"devices": 1, "num_nodes": 1, "accelerator": "gpu", "max_steps": 1}
