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

from typing import Dict, Union

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from nemo.core.classes import ModelPT


class PrefixedAttributesModel(ModelPT):
    """An encoder next to an attribute whose name starts with `encoder`, as in the multitask AED models."""

    def __init__(self, cfg: DictConfig, trainer=None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.encoder = torch.nn.Linear(4, 4)
        self.encoder_decoder_proj = torch.nn.Linear(4, 6)
        self.decoder = torch.nn.Linear(6, 6)

    def forward(self, x):
        return self.decoder(self.encoder_decoder_proj(self.encoder(x)))

    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        self._train_dl = None

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        self._validation_dl = None

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        self._test_dl = None

    @classmethod
    def list_available_models(cls):
        return []


def _group_names(model, group):
    names = {id(param): name for name, param in model.named_parameters()}
    return [names[id(param)] for param in group['params']]


class TestModelPTOptimParamGroups:
    @pytest.mark.unit
    def test_attribute_group_does_not_claim_prefixed_siblings(self):
        cfg = OmegaConf.create({'optim': {'name': 'sgd', 'lr': 0.1}, 'optim_param_groups': {'encoder': {'lr': 0.01}}})
        model = PrefixedAttributesModel(cfg)
        model.setup_optimization()
        groups = model._optimizer.param_groups
        assert [(group['lr'], _group_names(model, group)) for group in groups] == [
            (0.1, ['encoder_decoder_proj.weight', 'encoder_decoder_proj.bias', 'decoder.weight', 'decoder.bias']),
            (0.01, ['encoder.weight', 'encoder.bias']),
        ]

    @pytest.mark.unit
    def test_every_parameter_is_optimized_once(self):
        cfg = OmegaConf.create(
            {
                'optim': {'name': 'sgd', 'lr': 0.1},
                'optim_param_groups': {'encoder': {'lr': 0.01}, 'encoder_decoder_proj': {'lr': 0.02}},
            }
        )
        model = PrefixedAttributesModel(cfg)
        model.setup_optimization()
        optimized = [id(param) for group in model._optimizer.param_groups for param in group['params']]
        assert sorted(optimized) == sorted(id(param) for param in model.parameters())
        assert [group['lr'] for group in model._optimizer.param_groups] == [0.1, 0.01, 0.02]
