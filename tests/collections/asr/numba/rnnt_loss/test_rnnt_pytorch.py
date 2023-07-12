# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import random

import numpy as np
import pytest
import torch

from nemo.collections.asr.losses.rnnt import MultiblankRNNTLossPytorch, RNNTLossPytorch, TDTLossPytorch
from nemo.collections.asr.parts.numba.rnnt_loss.rnnt_numpy import RNNTLoss as RNNTLoss_Numpy
from nemo.collections.asr.parts.numba.rnnt_loss.rnnt_pytorch import (
    MultiblankRNNTLossNumba,
    RNNTLossNumba,
    TDTLossNumba,
)
from nemo.core.utils import numba_utils
from nemo.core.utils.numba_utils import __NUMBA_MINIMUM_VERSION__

DEVICES = ['cpu']

if torch.cuda.is_available():
    DEVICES.append('cuda')


DTYPES = [np.float32]
if numba_utils.is_numba_cuda_fp16_supported():
    DTYPES.append(np.float16)


def wrap_and_call(fn, acts, labels, device):
    if not torch.is_tensor(acts):
        acts = torch.tensor(acts)

    if 'cuda' in device:
        acts = acts.cuda()

    if not acts.requires_grad:
        acts.requires_grad = True

    lengths = [acts.shape[1]] * acts.shape[0]
    label_lengths = [len(l) for l in labels]
    labels = torch.LongTensor(labels)
    lengths = torch.LongTensor(lengths)
    label_lengths = torch.LongTensor(label_lengths)
    if 'cuda' in device:
        labels = labels.cuda()
        lengths = lengths.cuda()
        label_lengths = label_lengths.cuda()

    costs = fn(acts, labels, lengths, label_lengths)
    cost = torch.sum(costs)
    cost.backward()

    if 'cuda' in device:
        torch.cuda.synchronize()

    if acts.grad is not None:
        grad = acts.grad.data.cpu().numpy()
    else:
        grad = None

    return costs.data.cpu().numpy(), grad


class TestRNNTLossPytorch:
    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    @pytest.mark.parametrize('dtype', DTYPES)
    def test_case_small(self, device, dtype):
        if device == 'cuda':
            numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        acts = np.array(
            [
                [
                    [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.2, 0.8, 0.1]],
                    [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.2, 0.1, 0.1], [0.7, 0.1, 0.2, 0.1, 0.1]],
                ]
            ]
        ).astype(dtype)
        labels = [[1, 2]]

        cost_threshold = 1e-8 if dtype == np.float32 else 5e-4
        grad_threshold = 1e-8 if dtype == np.float32 else 1e-4
        rtol = 1e-5 if dtype == np.float32 else 1e-3

        fn_pt = RNNTLossNumba(blank=0, reduction='sum')
        pt_cost, pt_grads = wrap_and_call(fn_pt, acts, labels, device)

        fn_np = RNNTLoss_Numpy()
        np_cost, np_grads = wrap_and_call(fn_np, acts, labels, device)

        fn_ag = RNNTLossPytorch(blank=0, reduction='sum')  # ag for automatic gradient computation
        ag_cost, ag_grads = wrap_and_call(fn_ag, acts, labels, device)

        expected_cost = 4.495666
        expected_grads = np.array(
            [
                [
                    [
                        [-0.13116688, -0.3999269, 0.17703125, 0.17703125, 0.17703125],
                        [-0.18572757, 0.12247056, -0.18168412, 0.12247056, 0.12247056],
                        [-0.32091254, 0.06269141, 0.06928472, 0.12624499, 0.06269141],
                    ],
                    [
                        [0.05456069, -0.21824276, 0.05456069, 0.05456069, 0.05456069],
                        [0.12073959, 0.12073959, -0.48295835, 0.12073959, 0.12073959],
                        [-0.6925882, 0.16871116, 0.18645467, 0.16871116, 0.16871116],
                    ],
                ]
            ]
        )

        assert np.allclose(pt_cost, expected_cost, atol=cost_threshold, rtol=1e-6), "small_test costs mismatch."
        assert np.allclose(pt_grads, expected_grads, atol=grad_threshold, rtol=rtol), "small_test gradient mismatch."

        assert np.allclose(pt_cost, np_cost, atol=cost_threshold, rtol=rtol), "small_test costs mismatch."
        assert np.allclose(pt_grads, np_grads, atol=grad_threshold, rtol=rtol), "small_test gradient mismatch."

        assert np.allclose(ag_cost, np_cost, atol=cost_threshold, rtol=rtol), "small_test costs mismatch."
        assert np.allclose(ag_grads, np_grads, atol=cost_threshold, rtol=rtol), "small_test gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    @pytest.mark.parametrize('dtype', DTYPES)
    def test_case_small_random(self, device, dtype):
        if device == 'cuda':
            numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        cost_threshold = 1e-8 if dtype == np.float32 else 5e-4
        grad_threshold = 1e-8 if dtype == np.float32 else 1e-4
        rtol = 1e-5 if dtype == np.float32 else 1e-3

        rng = np.random.RandomState(0)
        acts = rng.randn(1, 4, 3, 3).astype(dtype)
        labels = [[1, 2]]

        fn_pt = RNNTLossNumba(blank=0, reduction='sum')
        pt_cost, pt_grads = wrap_and_call(fn_pt, acts, labels, device)

        fn_np = RNNTLoss_Numpy()
        np_cost, np_grads = wrap_and_call(fn_np, acts, labels, device)

        fn_ag = RNNTLossPytorch(blank=0, reduction='sum')  # ag for automatic gradient computation
        ag_cost, ag_grads = wrap_and_call(fn_ag, acts, labels, device)

        assert np.allclose(pt_cost, np_cost, atol=cost_threshold, rtol=rtol), "small_random_test costs mismatch."
        assert np.allclose(pt_grads, np_grads, atol=grad_threshold, rtol=rtol), "small_random_test gradient mismatch."

        assert np.allclose(pt_cost, ag_cost, atol=cost_threshold, rtol=rtol), "small_random_test costs mismatch."
        assert np.allclose(pt_grads, ag_grads, atol=grad_threshold, rtol=rtol), "small_random_test gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('fastemit_lambda', [1.0, 0.01, 0.00001])
    def test_case_small_random_fastemit_reg(self, device, dtype, fastemit_lambda):
        if device == 'cuda':
            numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        rng = np.random.RandomState(0)
        acts = rng.randn(1, 4, 3, 3)
        labels = [[1, 2]]

        fn_pt = RNNTLossNumba(blank=0, reduction='sum', fastemit_lambda=fastemit_lambda)
        pt_cost, pt_grads = wrap_and_call(fn_pt, acts, labels, device)

        fn_np = RNNTLoss_Numpy(fastemit_lambda=fastemit_lambda)
        np_cost, np_grads = wrap_and_call(fn_np, acts, labels, device)

        assert np.allclose(pt_cost, np_cost, rtol=1e-6), "small_random_test costs mismatch."
        assert np.allclose(pt_grads, np_grads, rtol=1e-5), "small_random_test gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    @pytest.mark.parametrize('dtype', DTYPES)
    def test_case_big_tensor(self, device, dtype):
        if device == 'cuda':
            numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        # minibatch x T x U x alphabet_size
        activations = [
            [
                [
                    [0.06535690384862791, 0.7875301411923206, 0.08159176605666074],
                    [0.5297155426466327, 0.7506749639230854, 0.7541348379087998],
                    [0.6097641124736383, 0.8681404965673826, 0.6225318186056529],
                ],
                [
                    [0.6685222872103057, 0.8580392805336061, 0.16453892311765583],
                    [0.989779515236694, 0.944298460961015, 0.6031678586829663],
                    [0.9467833543605416, 0.666202507295747, 0.28688179752461884],
                ],
                [
                    [0.09418426230195986, 0.3666735970751962, 0.736168049462793],
                    [0.1666804425271342, 0.7141542198635192, 0.3993997272216727],
                    [0.5359823524146038, 0.29182076440286386, 0.6126422611507932],
                ],
                [
                    [0.3242405528768486, 0.8007644367291621, 0.5241057606558068],
                    [0.779194617063042, 0.18331417220174862, 0.113745182072432],
                    [0.24022162381327106, 0.3394695622533106, 0.1341595066017014],
                ],
            ],
            [
                [
                    [0.5055615569388828, 0.051597282072282646, 0.6402903936686337],
                    [0.43073311517251, 0.8294731834714112, 0.1774668847323424],
                    [0.3207001991262245, 0.04288308912457006, 0.30280282975568984],
                ],
                [
                    [0.6751777088333762, 0.569537369330242, 0.5584738347504452],
                    [0.08313242153985256, 0.06016544344162322, 0.10795752845152584],
                    [0.7486153608562472, 0.943918041459349, 0.4863558118797222],
                ],
                [
                    [0.4181986264486809, 0.6524078485043804, 0.024242983423721887],
                    [0.13458171554507403, 0.3663418070512402, 0.2958297395361563],
                    [0.9236695822497084, 0.6899291482654177, 0.7418981733448822],
                ],
                [
                    [0.25000547599982104, 0.6034295486281007, 0.9872887878887768],
                    [0.5926057265215715, 0.8846724004467684, 0.5434495396894328],
                    [0.6607698886038497, 0.3771277082495921, 0.3580209022231813],
                ],
            ],
        ]

        expected_costs = [4.2806528590890736, 3.9384369822503591]
        expected_grads = [
            [
                [
                    [-1.86843902e-01, -6.25548810e-02, 2.49398798e-01],
                    [-2.03376666e-01, 2.02399328e-01, 9.77333169e-04],
                    [-1.41016081e-01, 7.91234672e-02, 6.18926100e-02],
                ],
                [
                    [-1.15517676e-02, -8.12802389e-02, 9.28319991e-02],
                    [-1.54257029e-01, 2.29432687e-01, -7.51756504e-02],
                    [-2.46593088e-01, 1.46404594e-01, 1.00188486e-01],
                ],
                [
                    [-1.29182907e-02, -6.15932420e-02, 7.45115355e-02],
                    [-5.59857301e-02, 2.19830811e-01, -1.63845062e-01],
                    [-4.97626871e-01, 2.09239945e-01, 2.88386941e-01],
                ],
                [
                    [1.36048580e-02, -3.02196294e-02, 1.66147724e-02],
                    [1.13924511e-01, 6.27811998e-02, -1.76705718e-01],
                    [-6.67078257e-01, 3.67658824e-01, 2.99419403e-01],
                ],
            ],
            [
                [
                    [-3.56343776e-01, -5.53474613e-02, 4.11691219e-01],
                    [-9.69219357e-02, 2.94591039e-02, 6.74628317e-02],
                    [-6.35175705e-02, 2.76544970e-02, 3.58630717e-02],
                ],
                [
                    [-1.54499024e-01, -7.39420280e-02, 2.28441030e-01],
                    [-1.66789949e-01, -8.78955179e-05, 1.66877866e-01],
                    [-1.72369644e-01, 1.05565332e-01, 6.68043196e-02],
                ],
                [
                    [2.38748826e-02, -1.18255816e-01, 9.43809375e-02],
                    [-1.04707085e-01, -1.08934477e-01, 2.13641584e-01],
                    [-3.69844258e-01, 1.80118099e-01, 1.89726159e-01],
                ],
                [
                    [2.57137045e-02, -7.94617534e-02, 5.37480488e-02],
                    [1.22328237e-01, -2.38788679e-01, 1.16460443e-01],
                    [-5.98686993e-01, 3.02203178e-01, 2.96483815e-01],
                ],
            ],
        ]

        activations = np.array(activations).astype(dtype)
        labels = [[1, 2], [1, 1]]

        cost_threshold = 1e-8 if dtype == np.float32 else 5e-4
        grad_threshold = 1e-8 if dtype == np.float32 else 1e-4
        rtol = 1e-3 if dtype == np.float32 else 0.1

        fn_pt = RNNTLossNumba(blank=0, reduction='sum')
        pt_costs, pt_grads = wrap_and_call(fn_pt, activations, labels, device)

        fn_np = RNNTLoss_Numpy()
        np_costs, np_grads = wrap_and_call(fn_np, activations, labels, device)

        fn_ag = RNNTLossPytorch(blank=0, reduction='sum')
        ag_costs, ag_grads = wrap_and_call(fn_ag, activations, labels, device)

        assert np.allclose(pt_costs, sum(expected_costs), atol=cost_threshold), "big_test average costs mismatch."
        assert np.allclose(
            pt_grads, expected_grads, atol=grad_threshold, rtol=1e-3
        ), "big_test grads for average cost mismatch."

        assert np.allclose(pt_costs, np_costs, atol=cost_threshold, rtol=rtol), "big_test average costs mismatch."
        assert np.allclose(
            pt_grads, np_grads, atol=grad_threshold, rtol=rtol
        ), "big_test grads for average cost mismatch."

        assert np.allclose(pt_costs, ag_costs, atol=cost_threshold, rtol=rtol), "big_test average costs mismatch."
        assert np.allclose(
            pt_grads, ag_grads, atol=grad_threshold, rtol=rtol
        ), "big_test grads for average cost mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    @pytest.mark.parametrize('dtype', DTYPES)
    def test_case_large_random(self, device, dtype):
        if device == 'cuda':
            numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        rng = np.random.RandomState(0)
        acts = rng.randn(4, 8, 11, 5).astype(dtype)
        labels = [
            [1, 2, 4, 3, 2, 2, 1, 1, 1, 1],
            [3, 2, 2, 3, 4, 1, 1, 1, 1, 1],
            [4, 4, 1, 2, 1, 3, 4, 3, 1, 2],
            [1, 1, 2, 1, 2, 3, 3, 1, 1, 1],
        ]

        cost_threshold = 1e-8 if dtype == np.float32 else 5e-4
        grad_threshold = 1e-8 if dtype == np.float32 else 1e-4
        rtol = 1e-3 if dtype == np.float32 else 5e-2

        fn_pt = RNNTLossNumba(blank=0, reduction='sum')
        pt_cost, pt_grads = wrap_and_call(fn_pt, acts, labels, device)

        fn_np = RNNTLoss_Numpy()
        np_cost, np_grads = wrap_and_call(fn_np, acts, labels, device)

        fn_ag = RNNTLossPytorch(blank=0, reduction='sum')
        ag_cost, ag_grads = wrap_and_call(fn_ag, acts, labels, device)

        assert np.allclose(pt_cost, np_cost, atol=cost_threshold, rtol=rtol), "large_random_test costs mismatch."
        assert np.allclose(ag_cost, np_cost, atol=cost_threshold, rtol=rtol), "large_random_test costs mismatch."
        assert np.allclose(pt_grads, np_grads, atol=grad_threshold, rtol=rtol), "large_random_test gradient mismatch."
        assert np.allclose(ag_grads, np_grads, atol=grad_threshold, rtol=rtol), "large_random_test gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    @pytest.mark.parametrize('dtype', DTYPES)
    def test_case_small_clamp(self, device, dtype):
        if device == 'cuda':
            numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        GRAD_CLAMP = 0.1
        acts = np.array(
            [
                [
                    [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.2, 0.8, 0.1]],
                    [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.2, 0.1, 0.1], [0.7, 0.1, 0.2, 0.1, 0.1]],
                ]
            ]
        ).astype(dtype)
        labels = [[1, 2]]

        cost_threshold = 1e-8 if dtype == np.float32 else 5e-4
        grad_threshold = 1e-8 if dtype == np.float32 else 5e-5
        rtol = 1e-5 if dtype == np.float32 else 1e-3

        fn_pt = RNNTLossNumba(blank=0, reduction='sum', clamp=GRAD_CLAMP)
        pt_cost, pt_grads = wrap_and_call(fn_pt, acts, labels, device)

        fn_np = RNNTLoss_Numpy(blank=0, clamp=GRAD_CLAMP)
        np_cost, np_grads = wrap_and_call(fn_np, acts, labels, device)

        expected_cost = 4.495666
        expected_grads = np.array(
            [
                [
                    [
                        [-0.1, -0.1, 0.1, 0.1, 0.1],
                        [-0.1, 0.1, -0.1, 0.1, 0.1],
                        [-0.1, 0.06269141, 0.06928472, 0.1, 0.06269141],
                    ],
                    [
                        [0.05456069, -0.1, 0.05456069, 0.05456069, 0.05456069],
                        [0.1, 0.1, -0.1, 0.1, 0.1],
                        [-0.1, 0.1, 0.1, 0.1, 0.1],
                    ],
                ]
            ]
        )

        assert np.allclose(pt_cost, expected_cost, atol=cost_threshold, rtol=rtol), "small_test costs mismatch."
        assert np.allclose(pt_grads, expected_grads, atol=grad_threshold, rtol=rtol), "small_test gradient mismatch."

        assert np.allclose(pt_cost, np_cost, atol=cost_threshold, rtol=rtol), "small_test costs mismatch."
        assert np.allclose(pt_grads, np_grads, atol=grad_threshold, rtol=rtol), "small_test gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('fastemit_lambda', [1.0, 0.01, 0.00001])
    def test_case_small_fastemit_clamp(self, device, dtype, fastemit_lambda):
        if device == 'cuda':
            numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        GRAD_CLAMP = 0.1
        acts = np.array(
            [
                [
                    [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.2, 0.8, 0.1]],
                    [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.2, 0.1, 0.1], [0.7, 0.1, 0.2, 0.1, 0.1]],
                ]
            ]
        ).astype(dtype)
        labels = [[1, 2]]

        cost_threshold = 1e-8 if dtype == np.float32 else 1e-3
        grad_threshold = 1e-8 if dtype == np.float32 else 5e-4
        rtol = 1e-5 if dtype == np.float32 else 1e-3

        fn_pt = RNNTLossNumba(blank=0, reduction='sum', fastemit_lambda=fastemit_lambda, clamp=GRAD_CLAMP)
        pt_cost, pt_grads = wrap_and_call(fn_pt, acts, labels, device)

        fn_np = RNNTLoss_Numpy(blank=0, fastemit_lambda=fastemit_lambda, clamp=GRAD_CLAMP)
        np_cost, np_grads = wrap_and_call(fn_np, acts, labels, device)

        expected_cost = 4.495666
        expected_cost += expected_cost * fastemit_lambda

        assert np.allclose(pt_cost, expected_cost, atol=cost_threshold, rtol=rtol), "small_test costs mismatch."
        assert np.allclose(pt_cost, np_cost, atol=cost_threshold, rtol=rtol), "small_test costs mismatch."
        assert np.allclose(pt_grads, np_grads, atol=grad_threshold, rtol=rtol), "small_test gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    def test_case_small_random_accumulated(self, device):
        if device == 'cuda':
            numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

        torch.manual_seed(0)
        base_layer = torch.randn(3, 5, requires_grad=True)

        mid1 = torch.randn(1, 4, 3, 3, requires_grad=True)
        labels1 = [[1, 3]]

        mid2 = torch.randn(1, 6, 5, 3, requires_grad=True)
        labels2 = [[1, 2, 3, 4]]

        def zero_grad():
            if base_layer.grad is not None:
                base_layer.grad = None
            if mid1.grad is not None:
                mid1.grad = None
            if mid2.grad is not None:
                mid2.grad = None

        fn_pt = RNNTLossNumba(blank=0, reduction='sum')
        fn_np = RNNTLoss_Numpy()

        # run 1
        acts1 = torch.matmul(mid1, base_layer)  # [1, 4, 3, 5]
        pt_cost1, _ = wrap_and_call(fn_pt, acts1, labels1, device)
        pt_grads1 = base_layer.grad.detach().cpu().numpy()

        zero_grad()

        acts1 = torch.matmul(mid1, base_layer)  # [1, 4, 3, 5]
        np_cost1, _ = wrap_and_call(fn_np, acts1, labels1, device)
        np_grads1 = base_layer.grad.detach().cpu().numpy()

        zero_grad()

        assert np.allclose(pt_grads1, np_grads1, atol=1e-6)

        # run 2
        acts2 = torch.matmul(mid2, base_layer)  # [1, 4, 3, 5]
        pt_cost2, _ = wrap_and_call(fn_pt, acts2, labels2, device)
        pt_grads2 = base_layer.grad.clone().cpu().numpy()

        zero_grad()

        acts2 = torch.matmul(mid2, base_layer)  # [1, 4, 3, 5]
        np_cost2, _ = wrap_and_call(fn_np, acts2, labels2, device)
        np_grads2 = base_layer.grad.clone().cpu().numpy()

        zero_grad()

        assert np.allclose(pt_grads2, np_grads2, atol=1e-6)

        # run 1 + 2
        acts1 = torch.matmul(mid1, base_layer)  # [1, 4, 3, 5]
        pt_cost1, _ = wrap_and_call(fn_pt, acts1, labels1, device)

        acts2 = torch.matmul(mid2, base_layer)  # [1, 6, 5, 5]
        pt_cost2, _ = wrap_and_call(fn_pt, acts2, labels2, device)
        pt_grads1_p_2 = base_layer.grad.clone().cpu().numpy()

        assert np.allclose(pt_grads1_p_2, np_grads1 + np_grads2, atol=1e-5)


class TestMultiblankRNNTLoss:
    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    def test_case_randomized_act_label(self, device):
        if device == 'cuda':
            numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

            B, T, U, V = 4, 8, 4, 8  # here V is number of non blank labels
            big_blank_durations = [2, 4, 8]
            sigma = 0.1

            acts = torch.rand([B, T, U, V + 1 + len(big_blank_durations)])
            labels = [[random.randrange(0, V) for i in range(U - 1)] for j in range(B)]

            fn_pt = MultiblankRNNTLossNumba(
                blank=V + len(big_blank_durations),
                reduction='sum',
                big_blank_durations=big_blank_durations,
                sigma=sigma,
            )
            pt_cost, pt_grads = wrap_and_call(fn_pt, acts, labels, device)

            fn_ag = MultiblankRNNTLossPytorch(
                blank=V + len(big_blank_durations),
                reduction='sum',
                big_blank_durations=big_blank_durations,
                sigma=sigma,
            )  # ag for automatic gradient computation
            ag_cost, ag_grads = wrap_and_call(fn_ag, acts, labels, device)

            assert np.allclose(pt_cost, ag_cost, rtol=1e-6), "multi-blank costs mismatch."
            assert np.allclose(pt_grads, ag_grads, rtol=1e-2), "multi-blank gradient mismatch."


class TestTDTLoss:
    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    def test_case_randomized_act_label(self, device):
        if device == 'cuda':
            numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

            B, T, U, V = 4, 8, 4, 8  # here V is number of non blank labels
            durations = [0, 1, 2, 3, 4, 5]
            sigma = 0.05

            acts = torch.rand([B, T, U, V + 1 + len(durations)])
            labels = [[random.randrange(0, V) for i in range(U - 1)] for j in range(B)]

            fn_pt = TDTLossNumba(blank=V, reduction='sum', durations=durations, sigma=sigma)
            pt_cost, pt_grads = wrap_and_call(fn_pt, acts, labels, device)

            fn_ag = TDTLossPytorch(
                blank=V, reduction='sum', durations=durations, sigma=sigma
            )  # ag for automatic gradient computation
            ag_cost, ag_grads = wrap_and_call(fn_ag, acts, labels, device)

            assert np.allclose(pt_cost, ag_cost, rtol=1e-6), "tdt costs mismatch."
            assert np.allclose(pt_grads, ag_grads, rtol=1e-2), "td gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    def test_case_fixed_case_act_label(self, device):
        if device == 'cuda':
            numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)

            B, T, U, V = 1, 3, 2, 3  # here V is number of non blank labels
            durations = [0, 1, 2]
            sigma = 0.05

            acts = torch.zeros([B, T, U, V + 1 + len(durations)])
            labels = [[(i + j) % (V - 1) for i in range(U - 1)] for j in range(B)]

            fn_pt = TDTLossNumba(blank=V, reduction='sum', durations=durations, sigma=sigma)
            pt_cost, pt_grads = wrap_and_call(fn_pt, acts, labels, device)

            expected_cost = 4.155739
            expected_grads = [
                [
                    [
                        [-0.64962804, 0.25, 0.25, 0.14962798, 0.2672583, -0.16792619, -0.09933221],
                        [0.01651875, 0.01651875, 0.01651875, -0.04955626, 0.022025, -0.01227201, -0.009753],
                    ],
                    [
                        [-0.04892651, 0.01714851, 0.01714851, 0.01462949, -0.01143234, -0.01143234, 0.02286467],
                        [0.12531489, 0.12531489, 0.12531489, -0.37594467, 0.16708651, 0.13027048, -0.29735702],
                    ],
                    [
                        [-0.02572276, 0.00857425, 0.00857425, 0.00857425, -0.02286468, 0.01143234, 0.01143234],
                        [0.13388914, 0.13388914, 0.13388914, -0.40166742, 0.17851885, -0.35703772, 0.17851885],
                    ],
                ]
            ]

            assert np.allclose(pt_cost, expected_cost, rtol=1e-6), "tdt costs mismatch."
            assert np.allclose(pt_grads, expected_grads, rtol=1e-2), "td gradient mismatch."


if __name__ == "__main__":
    pytest.main([__file__])
