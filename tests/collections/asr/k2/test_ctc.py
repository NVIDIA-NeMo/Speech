# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import pytest
import torch
from torch.nn import CTCLoss as CTCLoss_Pytorch

DEVICES = ['cpu']

if torch.cuda.is_available():
    DEVICES.append('cuda')


def wrap_and_call(fn, acts, labels, device):
    if not torch.is_tensor(acts):
        acts = torch.FloatTensor(acts)

    if 'cuda' in device:
        acts = acts.cuda()

    if not acts.requires_grad:
        acts.requires_grad = True

    lengths = [acts.shape[1]] * acts.shape[0]
    label_lengths = [len(l) for l in labels]
    labels = torch.LongTensor(labels)
    lengths = torch.LongTensor(lengths)
    label_lengths = torch.LongTensor(label_lengths)
    log_probs = torch.nn.functional.log_softmax(acts.transpose(0, 1), -1)
    if 'cuda' in device:
        labels = labels.cuda()
        lengths = lengths.cuda()
        label_lengths = label_lengths.cuda()

    costs = fn(log_probs, labels, lengths, label_lengths)
    cost = torch.sum(costs)
    cost.backward()

    if 'cuda' in device:
        torch.cuda.synchronize()

    if acts.grad is not None:
        grad = acts.grad.data.cpu().numpy()
    else:
        grad = None

    return costs.data.cpu().numpy(), grad


def init_k2_ctc(**kwargs):
    from nemo.collections.asr.parts.k2.ml_loss import CtcLoss

    ctc = CtcLoss(**kwargs)
    return lambda log_probs, labels, lengths, label_lengths: ctc(
        log_probs.transpose(0, 1), labels, lengths, label_lengths
    )[0]


def skip_test_if_unsupported(device, k2_is_appropriate, k2_cuda_is_enabled):
    if device == 'cpu':
        supported, msg = k2_is_appropriate
    elif device == 'cuda':
        supported, msg = k2_cuda_is_enabled
    else:
        raise ValueError(f"Unknown device: {device}")
    if not supported:
        pytest.skip(f"k2 test is skipped. Reason : {msg}")


class TestCTCLossK2:
    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    def test_case_small(self, device, k2_is_appropriate, k2_cuda_is_enabled):
        skip_test_if_unsupported(device, k2_is_appropriate, k2_cuda_is_enabled)

        acts = np.array(
            [
                [
                    [0.1, 0.6, 0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.6, 0.1, 0.1],
                    [0.1, 0.1, 0.2, 0.8, 0.1],
                    [0.1, 0.6, 0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.2, 0.1, 0.1],
                    [0.7, 0.1, 0.2, 0.1, 0.1],
                ]
            ]
        )
        labels = [[1, 2, 3]]

        fn_k2 = init_k2_ctc(num_classes=acts.shape[-1], blank=0, reduction='sum')
        k2_cost, k2_grads = wrap_and_call(fn_k2, acts, labels, device)

        expected_cost = 5.0279555
        expected_grads = np.array(
            [
                [
                    [0.00157518, -0.53266853, 0.17703111, 0.17703111, 0.17703111],
                    [-0.02431531, -0.17048728, -0.15925968, 0.17703113, 0.17703113],
                    [-0.06871005, 0.03236287, -0.2943067, 0.16722652, 0.16342735],
                    [-0.09178554, 0.25313747, -0.17673965, -0.16164337, 0.17703108],
                    [-0.10229809, 0.19587973, 0.05823242, -0.34769377, 0.19587973],
                    [-0.22203964, 0.1687112, 0.18645471, -0.30183747, 0.1687112],
                ]
            ]
        )

        assert np.allclose(k2_cost, expected_cost, rtol=1e-6), "small_test costs mismatch."
        assert np.allclose(k2_grads, expected_grads, atol=1e-6), "small_test gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    def test_case_small_blank_last(self, device, k2_is_appropriate, k2_cuda_is_enabled):
        skip_test_if_unsupported(device, k2_is_appropriate, k2_cuda_is_enabled)

        acts = np.array(
            [
                [
                    [0.0, 1.0, 3.0],
                    [0.0, 2.0, 3.0],
                    [1.0, 1.0, 3.0],
                    [2.0, 3.0, 2.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [2.0, 2.0, 0.0],
                    [0.0, 2.0, 5.0],
                    [0.0, 3.0, 5.0],
                    [1.0, 2.0, 5.0],
                    [2.0, 4.0, 4.0],
                    [0.0, 3.0, 4.0],
                    [0.0, 4.0, 4.0],
                    [1.0, 3.0, 4.0],
                    [2.0, 5.0, 3.0],
                    [2.0, 2.0, 1.0],
                    [2.0, 3.0, 1.0],
                    [3.0, 2.0, 1.0],
                    [4.0, 4.0, 0.0],
                ]
            ]
        )
        labels = [[0, 1, 0, 0, 1, 0]]

        fn_k2 = init_k2_ctc(num_classes=acts.shape[-1], blank=acts.shape[-1] - 1, reduction='sum')
        k2_cost, k2_grads = wrap_and_call(fn_k2, acts, labels, device)

        expected_cost = 6.823422
        expected_grads = np.array(
            [
                [
                    [-0.09792291, 0.11419516, -0.01627225],
                    [-0.08915664, 0.22963384, -0.14047718],
                    [-0.19687234, 0.06477807, 0.13209426],
                    [-0.22838503, 0.1980845, 0.03030053],
                    [-0.07985485, -0.0589368, 0.13879165],
                    [-0.04722299, 0.01424287, 0.03298012],
                    [0.01492161, 0.02710512, -0.04202673],
                    [-0.43219852, 0.4305843, 0.00161422],
                    [-0.00332598, 0.0440818, -0.04075582],
                    [-0.01329869, 0.11521607, -0.10191737],
                    [-0.03721291, 0.04389342, -0.00668051],
                    [-0.2723349, 0.43273386, -0.16039898],
                    [-0.03499417, 0.1896997, -0.15470551],
                    [-0.02911933, 0.29706067, -0.26794133],
                    [-0.04593367, -0.04479058, 0.09072424],
                    [-0.07227867, 0.16096972, -0.08869105],
                    [0.13993078, -0.20230117, 0.06237038],
                    [-0.05889719, 0.04007925, 0.01881794],
                    [-0.09667239, 0.07077749, 0.0258949],
                    [-0.49002117, 0.4954626, -0.00544143],
                ]
            ]
        )

        assert np.allclose(k2_cost, expected_cost, rtol=1e-6), "small_test_blank_last costs mismatch."
        assert np.allclose(k2_grads, expected_grads, atol=1e-6), "small_test_blank_last gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    def test_case_small_random(self, device, k2_is_appropriate, k2_cuda_is_enabled):
        skip_test_if_unsupported(device, k2_is_appropriate, k2_cuda_is_enabled)

        rng = np.random.RandomState(0)
        acts = rng.randn(1, 4, 3)
        labels = [[1, 2]]

        fn_k2 = init_k2_ctc(num_classes=acts.shape[-1], blank=0, reduction='sum')
        k2_cost, k2_grads = wrap_and_call(fn_k2, acts, labels, device)

        fn_pt = CTCLoss_Pytorch(reduction='sum', zero_infinity=True)
        pt_cost, pt_grads = wrap_and_call(fn_pt, acts, labels, device)

        assert np.allclose(k2_cost, pt_cost, rtol=1e-6), "small_random_test costs mismatch."
        assert np.allclose(k2_grads, pt_grads, atol=1e-6), "small_random_test gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    def test_case_big_tensor(self, device, k2_is_appropriate, k2_cuda_is_enabled):
        skip_test_if_unsupported(device, k2_is_appropriate, k2_cuda_is_enabled)

        # minibatch x T x alphabet_size
        acts = [
            [
                [0.06535690384862791, 0.7875301411923206, 0.08159176605666074],
                [0.5297155426466327, 0.7506749639230854, 0.7541348379087998],
                [0.6097641124736383, 0.8681404965673826, 0.6225318186056529],
                [0.6685222872103057, 0.8580392805336061, 0.16453892311765583],
                [0.989779515236694, 0.944298460961015, 0.6031678586829663],
                [0.9467833543605416, 0.666202507295747, 0.28688179752461884],
                [0.09418426230195986, 0.3666735970751962, 0.736168049462793],
                [0.1666804425271342, 0.7141542198635192, 0.3993997272216727],
                [0.5359823524146038, 0.29182076440286386, 0.6126422611507932],
                [0.3242405528768486, 0.8007644367291621, 0.5241057606558068],
                [0.779194617063042, 0.18331417220174862, 0.113745182072432],
                [0.24022162381327106, 0.3394695622533106, 0.1341595066017014],
            ],
            [
                [0.5055615569388828, 0.051597282072282646, 0.6402903936686337],
                [0.43073311517251, 0.8294731834714112, 0.1774668847323424],
                [0.3207001991262245, 0.04288308912457006, 0.30280282975568984],
                [0.6751777088333762, 0.569537369330242, 0.5584738347504452],
                [0.08313242153985256, 0.06016544344162322, 0.10795752845152584],
                [0.7486153608562472, 0.943918041459349, 0.4863558118797222],
                [0.4181986264486809, 0.6524078485043804, 0.024242983423721887],
                [0.13458171554507403, 0.3663418070512402, 0.2958297395361563],
                [0.9236695822497084, 0.6899291482654177, 0.7418981733448822],
                [0.25000547599982104, 0.6034295486281007, 0.9872887878887768],
                [0.5926057265215715, 0.8846724004467684, 0.5434495396894328],
                [0.6607698886038497, 0.3771277082495921, 0.3580209022231813],
            ],
        ]

        expected_costs = [6.388067, 5.2999153]
        expected_grads = [
            [
                [0.06130501, -0.3107036, 0.24939862],
                [0.08428053, -0.07131141, -0.01296911],
                [-0.04510102, 0.21943177, -0.17433074],
                [-0.1970142, 0.37144178, -0.17442757],
                [-0.08807078, 0.35828218, -0.2702114],
                [-0.24209887, 0.33242193, -0.09032306],
                [-0.07871056, 0.3116736, -0.23296304],
                [-0.27552277, 0.43320477, -0.157682],
                [-0.16173504, 0.27361175, -0.1118767],
                [-0.13012655, 0.42030025, -0.2901737],
                [-0.2378576, 0.26685005, -0.02899244],
                [0.08487711, 0.36765888, -0.45253596],
            ],
            [
                [-0.14147596, -0.2702151, 0.41169107],
                [-0.05323913, -0.18442528, 0.23766442],
                [-0.24160458, -0.11692462, 0.3585292],
                [-0.1004294, -0.17919227, 0.27962166],
                [-0.01819841, -0.12625945, 0.14445786],
                [-0.00131121, 0.06060241, -0.0592912],
                [-0.09093696, 0.2536721, -0.16273515],
                [-0.08962183, 0.34198248, -0.25236064],
                [-0.19668606, 0.25176668, -0.05508063],
                [0.0232805, 0.1351273, -0.1584078],
                [0.09494846, -0.17026341, 0.07531495],
                [0.00775955, -0.30424336, 0.29648378],
            ],
        ]

        acts = np.array(acts)
        expected_costs = np.array(expected_costs)
        labels = [[1, 2, 2, 2, 2], [1, 1, 2, 2, 1]]

        fn_k2 = init_k2_ctc(num_classes=acts.shape[-1], blank=0, reduction='none')
        k2_costs, k2_grads = wrap_and_call(fn_k2, acts, labels, device)

        assert np.allclose(k2_costs, expected_costs), "big_test average costs mismatch."
        assert np.allclose(k2_grads, expected_grads, rtol=1e-3), "big_test grads for average cost mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    def test_case_large_random(self, device, k2_is_appropriate, k2_cuda_is_enabled):
        skip_test_if_unsupported(device, k2_is_appropriate, k2_cuda_is_enabled)

        rng = np.random.RandomState(0)
        acts = rng.randn(4, 80, 5)
        labels = [
            [1, 2, 4, 3, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 3, 3, 1, 1, 1],
            [3, 2, 2, 3, 4, 1, 1, 1, 1, 1, 4, 4, 1, 2, 1, 3, 4, 3, 1, 2],
            [4, 4, 1, 2, 1, 3, 4, 3, 1, 2, 3, 2, 2, 3, 4, 1, 1, 1, 1, 1],
            [1, 1, 2, 1, 2, 3, 3, 1, 1, 1, 1, 2, 4, 3, 2, 2, 1, 1, 1, 1],
        ]

        fn_k2 = init_k2_ctc(num_classes=acts.shape[-1], blank=0, reduction='sum')
        k2_costs, k2_grads = wrap_and_call(fn_k2, acts, labels, device)

        fn_pt = CTCLoss_Pytorch(reduction='sum', zero_infinity=True)
        pt_costs, pt_grads = wrap_and_call(fn_pt, acts, labels, device)

        assert np.allclose(k2_costs, pt_costs, atol=1e-5, rtol=1e-3), "large_random_test costs mismatch."
        assert np.allclose(k2_grads, pt_grads, atol=1e-5, rtol=1e-3), "large_random_test gradient mismatch."


if __name__ == "__main__":
    pytest.main([__file__])
