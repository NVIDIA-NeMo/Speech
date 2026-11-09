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
import math
from unittest import mock

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
            zeropower_via_newtonschulz5(torch.zeros(5))

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


def _layered_parameters(seed: int = 0) -> torch.nn.ParameterDict:
    """Parameters in repeated shapes of both orientations, as the layers of one model have them."""
    generator = torch.Generator().manual_seed(seed)

    def param(*shape):
        return torch.nn.Parameter(torch.randn(*shape, generator=generator))

    parameters = torch.nn.ParameterDict()
    for layer in range(3):
        parameters[f'tall_{layer}'] = param(40, 16)  # Muon, fan_in scale sqrt(2.5)
        parameters[f'wide_{layer}'] = param(16, 40)  # Muon, iterated on the transposed orientation
        parameters[f'square_{layer}'] = param(24, 24)  # Muon
        parameters[f'bias_{layer}'] = param(40)  # AdamW
        parameters[f'conv_{layer}'] = param(8, 4, 3)  # AdamW by shape, Muon when forced
    parameters['embedding'] = param(8200, 16)  # AdamW, wide
    return parameters


def _layered_optimizer(parameters: torch.nn.ParameterDict, **options) -> Muon:
    """Two parameter groups: the default options, and the forced filters without weight decay."""
    convs = [param for name, param in parameters.items() if name.startswith('conv_')]
    others = [param for name, param in parameters.items() if not name.startswith('conv_')]
    return Muon(
        [{'params': others}, {'params': convs, 'use_muon': True, 'weight_decay': 0.0}],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        **options,
    )


def _run_layered(optimizer: Muon, parameters: torch.nn.ParameterDict, steps: int) -> None:
    """Steps with the same gradients per step; two parameters have no gradient in the second step."""
    for step in range(steps):
        _set_gradients(parameters, 500 + step)
        if step == 1:
            parameters['bias_1'].grad = None
            parameters['tall_1'].grad = None
        optimizer.step()


# The batched Newton-Schulz iteration computes the same bfloat16 arithmetic as the per-matrix one,
# but a batched matrix product may round differently on some devices and shapes
# (on CPU the two agree bitwise; on a GB300 four of 351 matrices of a 750-million-parameter model differed,
# by up to 2.4 percent of the update after five iterations amplified the rounding),
# so the Muon parameters are compared to within five percent of the update's scale
# (the learning rate times the largest shape factor), and everything else bitwise.
MUON_FOREACH_ATOL = 0.05 * LR * 2.5**0.5


def _assert_same_result(loop: Muon, foreach: Muon, loop_params, foreach_params) -> None:
    for name in loop_params:
        a, b = loop_params[name], foreach_params[name]
        loop_state, foreach_state = loop.state[a], foreach.state[b]
        assert set(loop_state) == set(foreach_state), name
        if 'momentum_buffer' in loop_state:
            assert torch.equal(loop_state['momentum_buffer'], foreach_state['momentum_buffer']), name
            torch.testing.assert_close(b, a, atol=MUON_FOREACH_ATOL, rtol=0.0, msg=name)
        else:
            assert torch.equal(a, b), name
            assert loop_state['step'] == foreach_state['step'], name
            assert torch.equal(loop_state['exp_avg'], foreach_state['exp_avg']), name
            assert torch.equal(loop_state['exp_avg_sq'], foreach_state['exp_avg_sq']), name


class TestMuonForeach:
    @pytest.mark.unit
    def test_option_default_and_parsing(self):
        parameters = _parameters()
        assert Muon(parameters.parameters(), lr=LR).param_groups[0]['foreach'] is True
        assert Muon(parameters.parameters(), lr=LR, foreach=False).param_groups[0]['foreach'] is False
        parsed = optim.parse_optimizer_args('muon', {'name': 'auto', 'foreach': False})
        assert parsed['foreach'] is False
        assert config.optimizers.MuonParams().foreach is True

    @pytest.mark.unit
    def test_foreach_matches_the_loop_on_mixed_shape_groups(self):
        loop_params = _layered_parameters()
        foreach_params = _layered_parameters()
        loop = _layered_optimizer(loop_params, foreach=False)
        foreach = _layered_optimizer(foreach_params, foreach=True)
        _run_layered(loop, loop_params, steps=3)
        _run_layered(foreach, foreach_params, steps=3)
        _assert_same_result(loop, foreach, loop_params, foreach_params)
        # the parameters without a gradient in the second step kept their step count behind the others
        assert loop.state[loop_params['bias_1']]['step'] == 2
        assert loop.state[loop_params['bias_0']]['step'] == 3
        assert 'momentum_buffer' in foreach.state[foreach_params['conv_0']]
        # the updates moved the parameters
        for name, param in foreach_params.items():
            assert not torch.equal(param, _layered_parameters()[name]), name

    @pytest.mark.unit
    def test_foreach_without_nesterov_and_without_decay(self):
        loop_params = _layered_parameters()
        foreach_params = _layered_parameters()
        loop = Muon(loop_params.parameters(), lr=LR, nesterov=False, foreach=False)
        foreach = Muon(foreach_params.parameters(), lr=LR, nesterov=False, foreach=True)
        _run_layered(loop, loop_params, steps=2)
        _run_layered(foreach, foreach_params, steps=2)
        _assert_same_result(loop, foreach, loop_params, foreach_params)

    @pytest.mark.unit
    def test_state_dict_moves_between_the_forms(self):
        for first, second in ((False, True), (True, False)):
            parameters = _layered_parameters()
            optimizer = _layered_optimizer(parameters, foreach=first)
            _run_layered(optimizer, parameters, steps=2)
            continued_params = copy.deepcopy(parameters)
            continued = _layered_optimizer(continued_params, foreach=second)
            continued.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            _set_gradients(parameters, 600)
            _set_gradients(continued_params, 600)
            optimizer.step()
            continued.step()
            if first:
                _assert_same_result(continued, optimizer, continued_params, parameters)
            else:
                _assert_same_result(optimizer, continued, parameters, continued_params)

    @pytest.mark.unit
    @pytest.mark.parametrize('capturable', [False, True])
    def test_state_dict_saved_without_the_options_loads(self, capturable):
        # a state saved by the Muon before these options has neither `foreach` nor `capturable` in its groups
        parameters = _layered_parameters()
        optimizer = _layered_optimizer(parameters, foreach=False)
        _run_layered(optimizer, parameters, steps=2)
        saved = copy.deepcopy(optimizer.state_dict())
        for group in saved['param_groups']:
            del group['foreach']
            del group['capturable']
        restored_params = copy.deepcopy(parameters)
        restored = _layered_optimizer(restored_params, capturable=capturable)
        restored.load_state_dict(saved)
        assert all(group['foreach'] is True and group['capturable'] is capturable for group in restored.param_groups)
        assert isinstance(restored.state[restored_params['bias_0']]['step'], torch.Tensor) is capturable
        _set_gradients(parameters, 650)
        _set_gradients(restored_params, 650)
        optimizer.step()
        restored.step()
        if capturable:
            _assert_close_result(optimizer, restored, parameters, restored_params)
        else:
            _assert_same_result(optimizer, restored, parameters, restored_params)

    @pytest.mark.unit
    def test_reduced_precision_parameters_differ_at_the_coefficient_rounding(self):
        # the CPU torch._foreach scalar operations round each scalar coefficient to the parameter dtype,
        # so with bfloat16 parameters the two forms agree to two bfloat16 ulps of a tensor's largest value,
        # bitwise on CUDA and not on CPU
        def two_ulps(tensor):
            return 2.0 ** (math.floor(math.log2(tensor.abs().max().item())) - 6)

        loop_params = _layered_parameters().bfloat16()
        foreach_params = _layered_parameters().bfloat16()
        loop = _layered_optimizer(loop_params, foreach=False)
        foreach = _layered_optimizer(foreach_params, foreach=True)
        for step in range(3):
            generator = torch.Generator().manual_seed(800 + step)
            for a, b in zip(loop_params.values(), foreach_params.values()):
                a.grad = torch.randn(a.shape, generator=generator).bfloat16()
                b.grad = a.grad.clone()
            loop.step()
            foreach.step()
        for name in loop_params:
            expected = loop_params[name]
            torch.testing.assert_close(foreach_params[name], expected, rtol=0.0, atol=two_ulps(expected), msg=name)
            for key, value in loop.state[expected].items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(
                        foreach.state[foreach_params[name]][key], value, rtol=0.0, atol=two_ulps(value), msg=name
                    )

    @pytest.mark.unit
    def test_foreach_uses_the_multi_tensor_operations(self):
        parameters = _layered_parameters()
        _set_gradients(parameters, 700)
        with mock.patch.object(torch, '_foreach_mul_', wraps=torch._foreach_mul_) as spy:
            _layered_optimizer(parameters, foreach=False).step()
            spy.assert_not_called()
            _layered_optimizer(parameters, foreach=True).step()
            assert spy.call_count > 0

    @pytest.mark.unit
    def test_newton_schulz_on_a_batch(self):
        generator = torch.Generator().manual_seed(2)
        matrices = torch.randn(4, 16, 40, generator=generator)
        batched = zeropower_via_newtonschulz5(matrices)
        assert batched.shape == matrices.shape
        assert batched.dtype == matrices.dtype
        for index in range(4):
            single = zeropower_via_newtonschulz5(matrices[index])
            torch.testing.assert_close(batched[index], single, atol=2**-5, rtol=0.0)
            singular_values = torch.linalg.svdvals(batched[index].double())
            assert 0.5 < singular_values.min() and singular_values.max() < 1.5
        # the batch of tall matrices is iterated on the transposed orientation
        assert torch.equal(zeropower_via_newtonschulz5(matrices.mT), batched.mT)
        with pytest.raises(ValueError, match="matrix"):
            zeropower_via_newtonschulz5(torch.zeros(2, 3, 4, 5))


def _run_with_learning_rates(optimizer: Muon, parameters: torch.nn.ParameterDict, learning_rates) -> None:
    """Steps at the given learning rates (a tensor learning rate is filled in place); two parameters skip step 1."""
    for step, learning_rate in enumerate(learning_rates):
        for group in optimizer.param_groups:
            if isinstance(group['lr'], torch.Tensor):
                group['lr'].fill_(learning_rate)
            else:
                group['lr'] = learning_rate
        _set_gradients(parameters, 900 + step)
        if step == 1:
            parameters['bias_1'].grad = None
            parameters['tall_1'].grad = None
        optimizer.step()


# A tensor learning rate and the device step count fold the step size into the AdamW denominator
# and scale the orthogonal factor as a separate product, so the updates differ at float32 rounding.
CAPTURABLE_TOLERANCE = dict(rtol=1e-5, atol=1e-6)


def _assert_close_result(reference: Muon, other: Muon, reference_params, other_params) -> None:
    for name in reference_params:
        a, b = reference_params[name], other_params[name]
        torch.testing.assert_close(b, a, **CAPTURABLE_TOLERANCE, msg=name)
        reference_state, other_state = reference.state[a], other.state[b]
        assert set(reference_state) == set(other_state), name
        for key, value in reference_state.items():
            if key == 'step':
                assert float(value) == float(other_state[key]), name
            else:
                torch.testing.assert_close(other_state[key], value, **CAPTURABLE_TOLERANCE, msg=name)


class TestMuonCapturable:
    @pytest.mark.unit
    def test_option_default_and_parsing(self):
        parameters = _parameters()
        assert Muon(parameters.parameters(), lr=LR).param_groups[0]['capturable'] is False
        assert optim.parse_optimizer_args('muon', {'name': 'auto', 'capturable': True})['capturable'] is True
        assert config.optimizers.MuonParams().capturable is False
        learning_rate = torch.tensor(LR)
        assert Muon(parameters.parameters(), lr=learning_rate).param_groups[0]['lr'] is learning_rate
        with pytest.raises(ValueError, match="Invalid learning rate"):
            Muon(parameters.parameters(), lr=torch.tensor(-LR))
        with pytest.raises(ValueError, match="one element"):
            Muon(parameters.parameters(), lr=torch.tensor([0.1, 0.2]))

    @pytest.mark.unit
    @pytest.mark.parametrize('foreach', [False, True])
    def test_one_element_learning_rate_tensor_of_any_shape(self, foreach):
        # a `(1,)` tensor, as a scheduler or a configuration may hand it over, is read as a scalar
        plain_params = _layered_parameters()
        tensor_params = _layered_parameters()
        plain = _layered_optimizer(plain_params, foreach=foreach)
        with_tensor = _layered_optimizer(tensor_params, foreach=foreach, capturable=True)
        for group in with_tensor.param_groups:
            group['lr'] = torch.tensor([LR])
        _run_with_learning_rates(plain, plain_params, [LR, LR / 2])
        _run_with_learning_rates(with_tensor, tensor_params, [LR, LR / 2])
        _assert_close_result(plain, with_tensor, plain_params, tensor_params)

    @pytest.mark.unit
    @pytest.mark.parametrize('foreach', [False, True])
    def test_capturable_step_reads_nothing_on_the_host(self, foreach):
        # a host read (`.item()`, a Python bool of a tensor) inside the step would be a `_local_scalar_dense` call
        from torch.utils._python_dispatch import TorchDispatchMode

        class CountHostReads(TorchDispatchMode):
            def __init__(self):
                super().__init__()
                self.host_reads = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func is torch.ops.aten._local_scalar_dense.default:
                    self.host_reads += 1
                return func(*args, **(kwargs or {}))

        parameters = _layered_parameters()
        optimizer = _layered_optimizer(parameters, foreach=foreach, capturable=True)
        for group in optimizer.param_groups:
            group['lr'] = torch.tensor(LR)
        _set_gradients(parameters, 700)
        optimizer.step()  # the state initialization
        _set_gradients(parameters, 701)
        parameters['bias_1'].grad = None
        with CountHostReads() as counter:
            optimizer.step()
        assert counter.host_reads == 0

    @pytest.mark.unit
    @pytest.mark.parametrize('foreach', [False, True])
    def test_capturable_matches_the_plain_step(self, foreach):
        plain_params = _layered_parameters()
        capturable_params = _layered_parameters()
        plain = _layered_optimizer(plain_params, foreach=foreach)
        capturable = _layered_optimizer(capturable_params, foreach=foreach, capturable=True)
        learning_rates = [LR, LR / 2, LR / 4]
        _run_with_learning_rates(plain, plain_params, learning_rates)
        _run_with_learning_rates(capturable, capturable_params, learning_rates)
        _assert_close_result(plain, capturable, plain_params, capturable_params)
        # the Muon update does not depend on the option, the step count lives on the device
        for name in ('tall_0', 'wide_2', 'square_1', 'conv_0'):
            assert torch.equal(plain_params[name], capturable_params[name]), name
        step = capturable.state[capturable_params['bias_0']]['step']
        assert isinstance(step, torch.Tensor) and step.device == capturable_params['bias_0'].device
        assert float(step) == 3.0 and float(capturable.state[capturable_params['bias_1']]['step']) == 2.0
        assert plain.state[plain_params['bias_1']]['step'] == 2

    @pytest.mark.unit
    @pytest.mark.parametrize('foreach', [False, True])
    def test_tensor_learning_rate_is_read_in_place(self, foreach):
        plain_params = _layered_parameters()
        tensor_params = _layered_parameters()
        plain = _layered_optimizer(plain_params, foreach=foreach)
        learning_rate = torch.tensor(LR)
        with_tensor = _layered_optimizer(tensor_params, foreach=foreach, capturable=True)
        for group in with_tensor.param_groups:
            group['lr'] = learning_rate
        learning_rates = [LR, LR / 2, LR / 4]
        _run_with_learning_rates(plain, plain_params, learning_rates)
        _run_with_learning_rates(with_tensor, tensor_params, learning_rates)
        _assert_close_result(plain, with_tensor, plain_params, tensor_params)
        # the same tensor object stays in the groups and holds the last value
        assert all(group['lr'] is learning_rate for group in with_tensor.param_groups)
        assert learning_rate.item() == pytest.approx(LR / 4)
        # a tensor learning rate works without `capturable` as well (the step count stays a Python integer)
        more_params = _layered_parameters()
        more = _layered_optimizer(more_params, foreach=foreach)
        for group in more.param_groups:
            group['lr'] = torch.tensor(LR)
        _run_with_learning_rates(more, more_params, learning_rates)
        _assert_close_result(plain, more, plain_params, more_params)
        assert more.state[more_params['bias_0']]['step'] == 3

    @pytest.mark.unit
    def test_state_dict_moves_between_capturable_settings(self):
        for first, second in ((False, True), (True, False)):
            parameters = _layered_parameters()
            optimizer = _layered_optimizer(parameters, capturable=first)
            _run_with_learning_rates(optimizer, parameters, [LR, LR])
            continued_params = copy.deepcopy(parameters)
            continued = _layered_optimizer(continued_params, capturable=second)
            continued.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            step = continued.state[continued_params['bias_0']]['step']
            assert isinstance(step, torch.Tensor) is second
            assert float(step) == 2.0
            _run_with_learning_rates(optimizer, parameters, [LR / 2])
            _run_with_learning_rates(continued, continued_params, [LR / 2])
            _assert_close_result(optimizer, continued, parameters, continued_params)

    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture needs a GPU")
    def test_captured_step_replays_with_new_gradients_and_learning_rate(self):
        device = torch.device('cuda')

        def setup():
            parameters = _layered_parameters().to(device)
            for param in parameters.values():
                param.grad = torch.zeros_like(param)
            optimizer = _layered_optimizer(parameters, capturable=True)
            for group in optimizer.param_groups:
                group['lr'] = torch.tensor(LR, device=device)
            return parameters, optimizer

        def feed(parameters, optimizer, seed, learning_rate):
            generator = torch.Generator().manual_seed(seed)
            for param in parameters.values():
                param.grad.copy_(torch.randn(param.shape, generator=generator))
            for group in optimizer.param_groups:
                group['lr'].fill_(learning_rate)

        eager_params, eager = setup()
        captured_params, captured = setup()
        schedule = [(1000, LR), (1001, LR), (1002, LR / 2), (1003, LR / 4), (1004, LR / 8), (1005, LR / 16)]
        # warm-up on a side stream, then the capture, then replays with new gradients and learning rates
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for seed, learning_rate in schedule[:3]:
                feed(captured_params, captured, seed, learning_rate)
                captured.step()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured.step()
        for seed, learning_rate in schedule[3:]:
            feed(captured_params, captured, seed, learning_rate)
            graph.replay()
        for seed, learning_rate in schedule:
            feed(eager_params, eager, seed, learning_rate)
            eager.step()
        torch.cuda.synchronize()
        _assert_close_result(eager, captured, eager_params, captured_params)
        assert float(captured.state[captured_params['bias_0']]['step']) == len(schedule)

    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture needs a GPU")
    @pytest.mark.parametrize('foreach', [False, True])
    def test_capture_needs_capturable_and_a_device_learning_rate(self, foreach):
        parameters = _layered_parameters().to('cuda')
        for param in parameters.values():
            param.grad = torch.zeros_like(param)

        def capture(optimizer):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                optimizer.step()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                optimizer.step()

        with pytest.raises(RuntimeError, match="capturable=True"):
            capture(_layered_optimizer(parameters, foreach=foreach))
        with pytest.raises(RuntimeError, match="tensor on the parameters' device"):
            capture(_layered_optimizer(parameters, foreach=foreach, capturable=True))
        on_the_host = _layered_optimizer(parameters, foreach=foreach, capturable=True)
        for group in on_the_host.param_groups:
            group['lr'] = torch.tensor(LR)
        with pytest.raises(RuntimeError, match="tensor on the parameters' device"):
            capture(on_the_host)
