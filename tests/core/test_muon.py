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

import copy

import pytest
import torch

from nemo.core import config, optim
from nemo.core.optim.muon import Muon, update_scale, zeropower_via_newtonschulz5
from nemo.core.optim.optimizers import AVAILABLE_OPTIMIZERS

LR = 0.02
WEIGHT_DECAY = 0.01


def _parameters(seed: int = 0) -> torch.nn.ParameterDict:
    """Parameters of every kind the optimizer distinguishes, in a fixed order."""
    generator = torch.Generator().manual_seed(seed)

    def param(*shape):
        return torch.nn.Parameter(torch.randn(*shape, generator=generator))

    # assigned one by one: ParameterDict sorts the keys of a plain dict
    parameters = torch.nn.ParameterDict()
    parameters['hidden'] = param(48, 32)  # Muon
    parameters['tall'] = param(64, 16)  # Muon, fan_in scale 2
    parameters['bias'] = param(48)  # AdamW
    parameters['conv'] = param(8, 4, 3, 3)  # AdamW by shape
    parameters['embedding'] = param(8200, 16)  # AdamW, wide
    return parameters


def _set_gradients(parameters: torch.nn.ParameterDict, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    for param in parameters.values():
        param.grad = torch.randn(param.shape, generator=generator)


def _run(optimizer: Muon, parameters: torch.nn.ParameterDict, steps: int, first_seed: int = 100) -> None:
    for step in range(steps):
        _set_gradients(parameters, first_seed + step)
        optimizer.step()


class TestMuon:
    @pytest.mark.unit
    def test_registry(self):
        assert AVAILABLE_OPTIMIZERS['muon'] is Muon
        parameters = _parameters()
        optimizer = optim.get_optimizer('muon')(parameters.parameters(), lr=LR)
        assert isinstance(optimizer, Muon)

        parsed = optim.parse_optimizer_args('muon', {'name': 'auto', 'momentum': 0.9, 'weight_decay': WEIGHT_DECAY})
        assert parsed['momentum'] == 0.9
        assert parsed['weight_decay'] == WEIGHT_DECAY
        assert parsed['adam_lr_ratio'] == config.optimizers.MuonParams().adam_lr_ratio
        assert parsed['scale_mode'] == 'fan_in'
        assert parsed['use_muon'] is None

    @pytest.mark.unit
    def test_parameter_selection(self):
        parameters = _parameters()
        selected = {
            name: Muon.selects_muon(param, use_muon=None, adam_dim_threshold=8192)
            for name, param in parameters.items()
        }
        assert selected == {'hidden': True, 'tall': True, 'bias': False, 'conv': False, 'embedding': False}
        assert not Muon.selects_muon(parameters['hidden'], use_muon=None, adam_dim_threshold=48)
        assert not Muon.selects_muon(parameters['hidden'], use_muon=False, adam_dim_threshold=8192)
        assert Muon.selects_muon(parameters['conv'], use_muon=True, adam_dim_threshold=8192)
        assert Muon.selects_muon(parameters['embedding'], use_muon=True, adam_dim_threshold=8192)

        with pytest.raises(ValueError, match="at least two dimensions"):
            Muon([{'params': [parameters['bias']], 'use_muon': True}], lr=LR)
        with pytest.raises(ValueError, match="scale_mode"):
            Muon(parameters.parameters(), lr=LR, scale_mode='rows')

    @pytest.mark.unit
    def test_newton_schulz_orthogonalizes(self):
        generator = torch.Generator().manual_seed(1)
        grad = torch.randn(16, 40, generator=generator)
        orthogonal = zeropower_via_newtonschulz5(grad)
        assert orthogonal.shape == grad.shape
        assert orthogonal.dtype == grad.dtype
        singular_values = torch.linalg.svdvals(orthogonal.double())
        assert singular_values.min() > 0.5
        assert singular_values.max() < 1.5
        # the input is far from orthogonal
        normalized = torch.linalg.svdvals(grad.double() / grad.double().norm())
        assert normalized.min() < 0.5

        # the iteration runs on the wide orientation, so a transposed input gives exactly the transposed result
        assert torch.equal(zeropower_via_newtonschulz5(grad.T), orthogonal.T)
        with pytest.raises(ValueError, match="matrix"):
            zeropower_via_newtonschulz5(torch.zeros(3, 4, 5))

    @pytest.mark.unit
    def test_muon_step_matches_definition(self):
        parameters = _parameters()
        optimizer = Muon(parameters.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        initial = {name: param.detach().clone() for name, param in parameters.items()}
        _run(optimizer, parameters, steps=2)

        expected = initial['hidden'].clone()
        buf = torch.zeros_like(expected)
        for step in range(2):
            generator = torch.Generator().manual_seed(100 + step)
            grad = torch.randn(48, 32, generator=generator)
            buf = buf * 0.95 + grad
            nesterov = grad + 0.95 * buf
            orthogonal = zeropower_via_newtonschulz5(nesterov, steps=5)
            expected = expected * (1 - LR * WEIGHT_DECAY) - LR * max(1.0, 48 / 32) ** 0.5 * orthogonal
        assert torch.allclose(parameters['hidden'], expected, atol=1e-7, rtol=1e-6)
        assert torch.equal(optimizer.state[parameters['hidden']]['momentum_buffer'], buf)

    @pytest.mark.unit
    def test_adamw_fallback_matches_torch_adamw(self):
        parameters = _parameters()
        reference = copy.deepcopy(parameters)
        betas, eps = (0.9, 0.95), 1e-8
        optimizer = Muon(
            parameters.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            adam_betas=betas,
            adam_eps=eps,
            adam_lr_ratio=1.0,
        )
        adamw_names = ['bias', 'conv', 'embedding']
        adamw = torch.optim.AdamW(
            [reference[name] for name in adamw_names], lr=LR, betas=betas, eps=eps, weight_decay=WEIGHT_DECAY
        )
        for step in range(3):
            _set_gradients(parameters, 200 + step)
            _set_gradients(reference, 200 + step)
            optimizer.step()
            adamw.step()
        for name in adamw_names:
            torch.testing.assert_close(parameters[name], reference[name], atol=1e-7, rtol=1e-6)
            assert optimizer.state[parameters[name]]['step'] == 3

    @pytest.mark.unit
    def test_weight_decay_uses_the_group_learning_rate(self):
        parameters = _parameters()
        initial = {name: param.detach().clone() for name, param in parameters.items()}
        optimizer = Muon(
            [
                {'params': [parameters['hidden'], parameters['bias']]},
                {'params': [parameters['tall'], parameters['conv']], 'weight_decay': 0.0},
            ],
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            adam_lr_ratio=0.1,
        )
        for param in parameters.values():
            param.grad = torch.zeros_like(param)
        optimizer.step()
        # with zero gradients both updates vanish and only the decay acts
        torch.testing.assert_close(parameters['hidden'], initial['hidden'] * (1 - LR * WEIGHT_DECAY))
        torch.testing.assert_close(parameters['bias'], initial['bias'] * (1 - LR * 0.1 * WEIGHT_DECAY))
        assert torch.equal(parameters['tall'], initial['tall'])
        assert torch.equal(parameters['conv'], initial['conv'])

    @pytest.mark.unit
    def test_scale_modes(self):
        assert update_scale((64, 16), 'fan_in') == 2.0
        assert update_scale((64, 16), 'fan_out') == 1.0
        assert update_scale((16, 64), 'fan_in') == 1.0
        assert update_scale((16, 64), 'fan_out') == 2.0
        assert update_scale((64, 16), 'none') == 1.0

        updates = {}
        for mode in ('fan_in', 'fan_out', 'none'):
            parameters = _parameters()
            initial = parameters['tall'].detach().clone()
            optimizer = Muon([parameters['tall']], lr=LR, scale_mode=mode)
            _run(optimizer, parameters, steps=1)
            updates[mode] = initial - parameters['tall']
        torch.testing.assert_close(updates['fan_in'], 2 * updates['fan_out'])
        assert torch.equal(updates['fan_out'], updates['none'])

        # the same matrix stored transposed receives the transposed update when the modes are paired
        parameters = _parameters()
        transposed = torch.nn.Parameter(parameters['tall'].detach().T.contiguous())
        optimizer = Muon([parameters['tall']], lr=LR, scale_mode='fan_in')
        optimizer_t = Muon([transposed], lr=LR, scale_mode='fan_out')
        for step in range(3):
            _set_gradients(parameters, 300 + step)
            transposed.grad = parameters['tall'].grad.T.contiguous()
            optimizer.step()
            optimizer_t.step()
        assert torch.equal(transposed, parameters['tall'].T)

    @pytest.mark.unit
    def test_forced_muon_flattens_filters(self):
        parameters = _parameters()
        flat = torch.nn.Parameter(parameters['conv'].detach().reshape(8, -1).clone())
        optimizer = Muon([{'params': [parameters['conv']], 'use_muon': True}], lr=LR, weight_decay=WEIGHT_DECAY)
        optimizer_flat = Muon([flat], lr=LR, weight_decay=WEIGHT_DECAY)
        for step in range(2):
            _set_gradients(parameters, 400 + step)
            flat.grad = parameters['conv'].grad.reshape(8, -1).clone()
            optimizer.step()
            optimizer_flat.step()
        assert torch.equal(parameters['conv'].reshape(8, -1), flat)
        assert 'momentum_buffer' in optimizer.state[parameters['conv']]

    @pytest.mark.unit
    def test_state_dict_round_trip(self):
        parameters = _parameters()
        optimizer = Muon(parameters.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        _run(optimizer, parameters, steps=2)
        saved_parameters = copy.deepcopy(parameters)
        saved_state = copy.deepcopy(optimizer.state_dict())
        _run(optimizer, parameters, steps=1, first_seed=102)

        restored = Muon(saved_parameters.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        restored.load_state_dict(saved_state)
        _run(restored, saved_parameters, steps=1, first_seed=102)
        for name in parameters:
            assert torch.equal(saved_parameters[name], parameters[name]), name
        assert restored.state[saved_parameters['bias']]['step'] == 3

    @pytest.mark.unit
    def test_scheduler_drives_both_learning_rates(self):
        parameters = _parameters()
        optimizer = Muon(parameters.parameters(), lr=1.0, adam_lr_ratio=0.1)
        scheduler = optim.lr_scheduler.WarmupPolicy(optimizer, warmup_steps=4, max_steps=10)
        initial = {name: param.detach().clone() for name, param in parameters.items()}
        for param in parameters.values():
            param.grad = torch.zeros_like(param)
        optimizer.param_groups[0]['weight_decay'] = 0.5
        optimizer.step()
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        assert lr == pytest.approx(2 / 5)
        # the first step ran at the warmup learning rate 1/5
        torch.testing.assert_close(parameters['hidden'], initial['hidden'] * (1 - 0.2 * 0.5))
        torch.testing.assert_close(parameters['bias'], initial['bias'] * (1 - 0.02 * 0.5))

    @pytest.mark.unit
    def test_production_shape_update_rms(self):
        # the first feed-forward matrix of a width-1024 Conformer, in torch's (out, in) layout
        matrix = torch.nn.Parameter(torch.randn(4096, 1024, generator=torch.Generator().manual_seed(5)))
        assert Muon.selects_muon(matrix, use_muon=None, adam_dim_threshold=8192)
        initial = matrix.detach().clone()
        optimizer = Muon([matrix], lr=1.0)
        matrix.grad = torch.randn(matrix.shape, generator=torch.Generator().manual_seed(6))
        optimizer.step()
        update = initial - matrix.detach()
        # fan_in: the update of an (out, in) matrix has root mean square 1 / sqrt(fan_in),
        # up to the Newton-Schulz approximation of the singular values
        assert update.pow(2).mean().sqrt().item() == pytest.approx(1 / 1024**0.5, rel=0.15)
        assert update.shape == (4096, 1024)

    @pytest.mark.unit
    def test_small_vocabulary_embedding_needs_an_explicit_exclusion(self):
        # a 1026 x 512 embedding is below the default threshold, so the shape heuristic selects Muon for it
        embedding = torch.nn.Embedding(1026, 512)
        assert Muon.selects_muon(embedding.weight, use_muon=None, adam_dim_threshold=8192)
        optimizer = Muon([{'params': [embedding.weight], 'use_muon': False, 'weight_decay': 0.0}], lr=LR)
        embedding.weight.grad = torch.ones_like(embedding.weight)
        optimizer.step()
        state = optimizer.state[embedding.weight]
        assert 'exp_avg' in state
        assert 'momentum_buffer' not in state
        # a wide embedding falls back by shape alone
        assert not Muon.selects_muon(torch.empty(8192, 16), use_muon=None, adam_dim_threshold=8192)
