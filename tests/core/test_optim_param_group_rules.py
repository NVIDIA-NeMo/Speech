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
from nemo.core.optim.param_groups import apply_param_group_rules


class Network(torch.nn.Module):
    """Embedding tied to the output projection, a norm, a convolution and a projection with bias."""

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(12, 8)
        self.norm = torch.nn.LayerNorm(8)
        self.conv = torch.nn.Conv1d(8, 8, kernel_size=1)
        self.hidden = torch.nn.Linear(8, 8)
        self.output = torch.nn.Linear(8, 12, bias=False)
        self.output.weight = self.embedding.weight

    def forward(self, x):
        return self.output(self.hidden(self.conv(self.norm(self.embedding(x)).transpose(1, 2)).transpose(1, 2)))


NO_WEIGHT_DECAY = {'module_types': ['torch.nn.Embedding', 'torch.nn.LayerNorm'], 'param_names': ['*.bias']}


def _names(network: torch.nn.Module, group: Dict) -> list:
    by_id = {id(param): name for name, param in network.named_parameters()}
    return [by_id[id(param)] for param in group['params']]


class TestApplyParamGroupRules:
    @pytest.mark.unit
    def test_selection_by_type_and_name(self):
        network = Network()
        groups = apply_param_group_rules(
            network,
            [{'params': list(network.parameters()), 'lr': 0.5}],
            {
                'no_weight_decay': dict(NO_WEIGHT_DECAY, weight_decay=0.0),
                'conv_flag': {'module_types': [torch.nn.Conv1d], 'use_muon': True},
            },
        )
        # one sub-group per combination of matching rules, in the order of the first parameter of each
        assert [_names(network, group) for group in groups] == [
            ['embedding.weight', 'norm.weight', 'norm.bias', 'hidden.bias'],
            ['conv.weight'],
            ['conv.bias'],
            ['hidden.weight'],
        ]
        assert groups[0] == {'lr': 0.5, 'weight_decay': 0.0, 'params': groups[0]['params']}
        assert groups[1] == {'lr': 0.5, 'use_muon': True, 'params': groups[1]['params']}
        # the convolution bias matches both rules and takes both options
        assert groups[2] == {'lr': 0.5, 'weight_decay': 0.0, 'use_muon': True, 'params': groups[2]['params']}
        assert groups[3] == {'lr': 0.5, 'params': groups[3]['params']}
        assert sum(len(group['params']) for group in groups) == len(list(network.parameters()))

    @pytest.mark.unit
    def test_shared_parameter_matches_through_any_owner_or_name(self):
        network = Network()
        tied = [network.output.weight]
        by_type = apply_param_group_rules(
            network, [{'params': tied}], {'rule': {'module_types': ['torch.nn.Embedding'], 'weight_decay': 0.0}}
        )
        assert by_type == [{'weight_decay': 0.0, 'params': tied}]
        by_name = apply_param_group_rules(
            network, [{'params': tied}], {'rule': {'param_names': ['output.weight'], 'weight_decay': 0.0}}
        )
        assert by_name == [{'weight_decay': 0.0, 'params': tied}]
        unmatched = apply_param_group_rules(
            network, [{'params': tied}], {'rule': {'param_names': ['hidden.*'], 'weight_decay': 0.0}}
        )
        assert unmatched == [{'params': tied}]

    @pytest.mark.unit
    def test_second_name_of_a_parameter_within_one_module_counts(self):
        class Aliased(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(2))
                self.alias = self.weight

        module = Aliased()
        groups = apply_param_group_rules(
            module, [{'params': [module.weight]}], {'rule': {'param_names': ['alias'], 'weight_decay': 0.0}}
        )
        assert groups == [{'weight_decay': 0.0, 'params': [module.weight]}]

    @pytest.mark.unit
    def test_plain_parameters_as_the_optimizer_takes_them(self):
        network = Network()
        rules = {'rule': {'param_names': ['*.bias'], 'weight_decay': 0.0}}
        assert apply_param_group_rules(network, network.parameters(), rules) == apply_param_group_rules(
            network, [{'params': list(network.parameters())}], rules
        )

    @pytest.mark.unit
    def test_single_parameter_as_a_group(self):
        class Scaled(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)
                self.scale = torch.nn.Parameter(torch.ones(()))

        module = Scaled()
        groups = apply_param_group_rules(
            module,
            [{'params': module.linear.weight, 'lr': 0.1}, {'params': module.scale, 'lr': 0.2}],
            {'rule': {'param_names': ['linear.weight', 'scale'], 'weight_decay': 0.0}},
        )
        # a single parameter as a group's `params`, as the optimizer takes it, stays that parameter
        assert groups == [
            {'lr': 0.1, 'weight_decay': 0.0, 'params': [module.linear.weight]},
            {'lr': 0.2, 'weight_decay': 0.0, 'params': [module.scale]},
        ]
        torch.optim.SGD(groups, lr=0.1)

    @pytest.mark.unit
    def test_several_base_groups_and_rule_order(self):
        network = Network()
        groups = apply_param_group_rules(
            network,
            [
                {'params': [network.hidden.weight, network.hidden.bias], 'lr': 1.0},
                {'params': [network.norm.weight], 'lr': 2.0, 'weight_decay': 0.3},
            ],
            {
                'first': {'param_names': ['*'], 'weight_decay': 0.1, 'momentum': 0.8},
                'second': {'param_names': ['*.bias', 'norm.*'], 'weight_decay': 0.0},
            },
        )
        assert groups == [
            {'lr': 1.0, 'weight_decay': 0.1, 'momentum': 0.8, 'params': [network.hidden.weight]},
            {'lr': 1.0, 'weight_decay': 0.0, 'momentum': 0.8, 'params': [network.hidden.bias]},
            {'lr': 2.0, 'weight_decay': 0.0, 'momentum': 0.8, 'params': [network.norm.weight]},
        ]

    @pytest.mark.unit
    def test_invalid_rules(self):
        network = Network()
        groups = [{'params': list(network.parameters())}]
        with pytest.raises(ValueError, match="selects nothing"):
            apply_param_group_rules(network, groups, {'empty': {'weight_decay': 0.0}})
        with pytest.raises(ValueError, match="sets no parameter group options"):
            apply_param_group_rules(network, groups, {'noop': {'param_names': ['*']}})
        with pytest.raises(ValueError, match="must not set `params`"):
            apply_param_group_rules(network, groups, {'bad': {'param_names': ['*'], 'params': []}})
        with pytest.raises(TypeError, match="not a torch.nn.Module class"):
            apply_param_group_rules(network, groups, {'bad': {'module_types': ['torch.nn.functional.relu'], 'lr': 1}})
        assert apply_param_group_rules(network, groups, {}) == groups


class RuledModel(ModelPT):
    def __init__(self, cfg: DictConfig, trainer=None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.network = Network()

    def forward(self, x):
        return self.network(x)

    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        self._train_dl = None

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        self._validation_dl = None

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        self._test_dl = None

    @classmethod
    def list_available_models(cls):
        return []


class FrozenConvModel(RuledModel):
    """Keeps the convolution out of the optimizer, as models with frozen sub-models do."""

    def setup_optimizer_param_groups(self):
        frozen = {id(param) for param in self.network.conv.parameters()}
        self._optimizer_param_groups = [{'params': [p for p in self.parameters() if id(p) not in frozen]}]


class TestModelPTParamGroupRules:
    @pytest.mark.unit
    def test_rules_split_the_optimizer_groups(self):
        cfg = OmegaConf.create(
            {
                'optim': {'name': 'adamw', 'lr': 1e-3, 'weight_decay': 0.01},
                'optim_param_group_rules': {'no_weight_decay': dict(NO_WEIGHT_DECAY, weight_decay=0.0)},
            }
        )
        model = RuledModel(cfg)
        model.setup_optimization()
        groups = model._optimizer.param_groups
        assert [group['weight_decay'] for group in groups] == [0.0, 0.01]
        assert [_names(model.network, group) for group in groups] == [
            ['embedding.weight', 'norm.weight', 'norm.bias', 'conv.bias', 'hidden.bias'],
            ['conv.weight', 'hidden.weight'],
        ]
        assert all(group['lr'] == 1e-3 for group in groups)

    @pytest.mark.unit
    def test_rules_apply_within_attribute_groups(self):
        cfg = OmegaConf.create(
            {
                'optim': {'name': 'sgd', 'lr': 0.1},
                'optim_param_groups': {'network': {'lr': 0.2}},
                'optim_param_group_rules': {'biases': {'param_names': ['*.bias'], 'lr': 0.3}},
            }
        )
        model = RuledModel(cfg)
        model.setup_optimization()
        groups = model._optimizer.param_groups
        assert [(group['lr'], _names(model.network, group)) for group in groups] == [
            (0.2, ['embedding.weight', 'norm.weight', 'conv.weight', 'hidden.weight']),
            (0.3, ['norm.bias', 'conv.bias', 'hidden.bias']),
        ]

    @pytest.mark.unit
    def test_without_rules_nothing_changes(self):
        cfg = OmegaConf.create({'optim': {'name': 'sgd', 'lr': 0.1}})
        model = RuledModel(cfg)
        model.setup_optimization()
        assert len(model._optimizer.param_groups) == 1
        assert len(model._optimizer.param_groups[0]['params']) == len(list(model.parameters()))

    @pytest.mark.unit
    def test_rules_apply_once_in_setup_optimization(self):
        cfg = OmegaConf.create(
            {
                'optim': {'name': 'sgd', 'lr': 0.1},
                'optim_param_group_rules': {'biases': {'param_names': ['*.bias'], 'weight_decay': 0.0}},
            }
        )
        model = RuledModel(cfg)
        # the grouping method leaves the rules to `setup_optimization`, so an override keeps its result
        model.setup_optimizer_param_groups()
        assert len(model._optimizer_param_groups) == 1
        model.setup_optimization()
        assert len(model._optimizer.param_groups) == 2

    @pytest.mark.unit
    def test_rules_apply_to_an_overridden_grouping(self):
        cfg = OmegaConf.create(
            {
                'optim': {'name': 'adamw', 'lr': 1e-3, 'weight_decay': 0.01},
                'optim_param_group_rules': {'no_weight_decay': dict(NO_WEIGHT_DECAY, weight_decay=0.0)},
            }
        )
        model = FrozenConvModel(cfg)
        model.setup_optimization()
        groups = model._optimizer.param_groups
        # the frozen convolution stays out, the remaining parameters are split by the rules
        assert [group['weight_decay'] for group in groups] == [0.0, 0.01]
        assert [_names(model.network, group) for group in groups] == [
            ['embedding.weight', 'norm.weight', 'norm.bias', 'hidden.bias'],
            ['hidden.weight'],
        ]

    @pytest.mark.unit
    def test_optimizer_state_restores_with_unchanged_rules(self):
        cfg = OmegaConf.create(
            {
                'optim': {'name': 'adamw', 'lr': 1e-3, 'weight_decay': 0.01},
                'optim_param_group_rules': {'no_weight_decay': dict(NO_WEIGHT_DECAY, weight_decay=0.0)},
            }
        )
        model = RuledModel(cfg)
        model.setup_optimization()
        for param in model.parameters():
            param.grad = torch.ones_like(param)
        model._optimizer.step()
        state = model._optimizer.state_dict()

        # the same rules give the same groups, so a saved optimizer state restores
        continued = RuledModel(cfg)
        continued.setup_optimization()
        continued._optimizer.load_state_dict(state)
        assert [group['weight_decay'] for group in continued._optimizer.param_groups] == [0.0, 0.01]
        assert len(continued._optimizer.state) == len(list(model.parameters()))

        # other rules give other groups: the saved state no longer fits groups of other sizes
        changed = OmegaConf.create(
            {
                'optim': {'name': 'adamw', 'lr': 1e-3, 'weight_decay': 0.01},
                'optim_param_group_rules': {
                    'embedding_only': {'module_types': ['torch.nn.Embedding'], 'weight_decay': 0.0}
                },
            }
        )
        other = RuledModel(changed)
        other.setup_optimization()
        with pytest.raises(ValueError, match="parameter group"):
            other._optimizer.load_state_dict(state)

        # and it loads onto groups of the same sizes, with the state on other parameters
        changed = OmegaConf.create(
            {
                'optim': {'name': 'adamw', 'lr': 1e-3, 'weight_decay': 0.01},
                'optim_param_group_rules': {
                    'first_modules': {
                        'param_names': ['network.embedding.*', 'network.norm.*', 'network.conv.*'],
                        'weight_decay': 0.0,
                    }
                },
            }
        )
        shifted = RuledModel(changed)
        shifted.setup_optimization()
        assert [len(group['params']) for group in shifted._optimizer.param_groups] == [5, 2]
        shifted._optimizer.load_state_dict(state)
        hidden = shifted.network.hidden
        assert shifted._optimizer.state[hidden.bias]['exp_avg'].shape == hidden.weight.shape


class TestConformerParamGroupRules:
    """The rules on a Conformer encoder, the module structure the rules are written for."""

    @pytest.mark.unit
    def test_weight_decay_and_muon_selection_on_a_conformer(self):
        from nemo.collections.asr.modules import ConformerEncoder

        encoder = ConformerEncoder(
            feat_in=8, n_layers=1, d_model=16, n_heads=2, ff_expansion_factor=2, conv_kernel_size=3
        )
        groups = apply_param_group_rules(
            encoder,
            [{'params': list(encoder.parameters()), 'weight_decay': 0.01}],
            {
                'no_weight_decay': {'param_names': ['*.bias'], 'weight_decay': 0.0},
                # the batch-norm shift keeps its decay although it is named bias
                'batch_norm_decays': {'module_types': ['torch.nn.BatchNorm1d'], 'weight_decay': 0.01},
                'pointwise_matrices': {
                    'param_names': ['*.pointwise_conv1.weight', '*.pointwise_conv2.weight'],
                    'use_muon': True,
                },
            },
        )
        options = {}
        for group in groups:
            for name in _names(encoder, group):
                options[name] = {key: value for key, value in group.items() if key != 'params'}
        assert len(options) == len(list(encoder.parameters()))
        assert options['layers.0.conv.batch_norm.bias'] == {'weight_decay': 0.01}
        assert options['layers.0.conv.batch_norm.weight'] == {'weight_decay': 0.01}
        assert options['layers.0.conv.pointwise_conv1.weight'] == {'weight_decay': 0.01, 'use_muon': True}
        assert options['layers.0.conv.pointwise_conv1.bias'] == {'weight_decay': 0.0}
        assert options['layers.0.conv.depthwise_conv.weight'] == {'weight_decay': 0.01}
        assert options['layers.0.self_attn.linear_q.weight'] == {'weight_decay': 0.01}
        assert options['layers.0.self_attn.linear_q.bias'] == {'weight_decay': 0.0}
        assert options['layers.0.norm_out.bias'] == {'weight_decay': 0.0}
        assert options['pre_encode.out.bias'] == {'weight_decay': 0.0}
        biases = [name for name in options if name.endswith('.bias') and 'batch_norm' not in name]
        assert biases and all(options[name]['weight_decay'] == 0.0 for name in biases)
