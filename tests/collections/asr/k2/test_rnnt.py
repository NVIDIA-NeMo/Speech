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

from nemo.collections.asr.parts.numba.rnnt_loss.rnnt_numpy import RNNTLoss as RNNTLoss_Numpy

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


def init_k2_rnnt(**kwargs):
    from nemo.collections.asr.parts.k2.ml_loss import RnntLoss

    rnnt = RnntLoss(**kwargs)
    return lambda acts, labels, lengths, label_lengths: rnnt(
        torch.nn.functional.log_softmax(acts, -1),
        labels.to(dtype=torch.long),
        lengths.to(dtype=torch.long),
        label_lengths.to(dtype=torch.long),
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


class TestRNNTLossK2:
    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    def test_case_small(self, device, k2_is_appropriate, k2_cuda_is_enabled):
        skip_test_if_unsupported(device, k2_is_appropriate, k2_cuda_is_enabled)

        acts = np.array(
            [
                [
                    [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.2, 0.8, 0.1]],
                    [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.2, 0.1, 0.1], [0.7, 0.1, 0.2, 0.1, 0.1]],
                ]
            ]
        )
        labels = [[1, 2]]

        fn_k2 = init_k2_rnnt(num_classes=acts.shape[-1], blank=0, reduction='sum')
        k2_cost, k2_grads = wrap_and_call(fn_k2, acts, labels, device)

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

        assert np.allclose(k2_cost, expected_cost, rtol=1e-6), "small_test costs mismatch."
        assert np.allclose(k2_grads, expected_grads, atol=1e-6), "small_test gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    def test_case_small_blank_last(self, device, k2_is_appropriate, k2_cuda_is_enabled):
        skip_test_if_unsupported(device, k2_is_appropriate, k2_cuda_is_enabled)

        acts = np.array(
            [
                [
                    [[0.0, 1.0, 3.0], [0.0, 2.0, 3.0], [1.0, 1.0, 3.0], [2.0, 3.0, 2.0]],
                    [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [2.0, 2.0, 0.0]],
                    [[0.0, 2.0, 5.0], [0.0, 3.0, 5.0], [1.0, 2.0, 5.0], [2.0, 4.0, 4.0]],
                    [[0.0, 3.0, 4.0], [0.0, 4.0, 4.0], [1.0, 3.0, 4.0], [2.0, 5.0, 3.0]],
                    [[2.0, 2.0, 1.0], [2.0, 3.0, 1.0], [3.0, 2.0, 1.0], [4.0, 4.0, 0.0]],
                ]
            ]
        )
        labels = [[0, 1, 0]]

        fn_k2 = init_k2_rnnt(num_classes=acts.shape[-1], blank=acts.shape[-1] - 1, reduction='sum')
        k2_cost, k2_grads = wrap_and_call(fn_k2, acts, labels, device)

        expected_cost = 6.789285182952881
        expected_grads = np.array(
            [
                [
                    [
                        [-0.03551076725125313, 0.11419519782066345, -0.07868456840515137],
                        [0.0027224558871239424, 0.00704305712133646, -0.009765520691871643],
                        [0.0013856772566214204, 0.0013924005907028913, -0.0027780719101428986],
                        [1.4249643527364242e-06, 3.873454716085689e-06, -5.298420546751004e-06],
                    ],
                    [
                        [-0.1934257447719574, 0.19551163911819458, -0.0020859241485595703],
                        [0.07043898105621338, 0.05738453567028046, -0.12782356142997742],
                        [0.061031512916088104, 0.02286236733198166, -0.08389391005039215],
                        [0.0005252412520349026, 0.0005252412520349026, -0.0010504829697310925],
                    ],
                    [
                        [-0.007841046899557114, 0.025142310187220573, -0.017301201820373535],
                        [0.0019501042552292347, 0.0005148053169250488, -0.0024650096893310547],
                        [0.0027856370434165, 0.008609085343778133, -0.01139475405216217],
                        [9.526080975774676e-05, 0.0007038871408440173, -0.000799147819634527],
                    ],
                    [
                        [-0.01533521432429552, 0.1386115401983261, -0.12327653169631958],
                        [0.002850571647286415, -0.006693005561828613, 0.003842458128929138],
                        [0.009236274287104607, 0.08995233476161957, -0.0991886705160141],
                        [0.0001865450612967834, 0.0037468576338142157, -0.003933403175324202],
                    ],
                    [
                        [-0.2888762652873993, 0.211185485124588, 0.07769080251455307],
                        [0.15952755510807037, -0.2182144820690155, 0.05868690833449364],
                        [-0.3332723379135132, 0.2436419129371643, 0.0896308496594429],
                        [0.4954628646373749, 0.4954628646373749, -0.9909257292747498],
                    ],
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
        acts = rng.randn(1, 4, 3, 3)
        labels = [[1, 2]]

        fn_k2 = init_k2_rnnt(num_classes=acts.shape[-1], blank=0, reduction='sum')
        k2_cost, k2_grads = wrap_and_call(fn_k2, acts, labels, device)

        fn_np = RNNTLoss_Numpy()
        np_cost, np_grads = wrap_and_call(fn_np, acts, labels, device)

        assert np.allclose(k2_cost, np_cost, rtol=1e-6), "small_random_test costs mismatch."
        assert np.allclose(k2_grads, np_grads, atol=1e-6), "small_random_test gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    def test_case_big_tensor(self, device, k2_is_appropriate, k2_cuda_is_enabled):
        skip_test_if_unsupported(device, k2_is_appropriate, k2_cuda_is_enabled)

        # minibatch x T x U x alphabet_size
        acts = [
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

        acts = np.array(acts)
        expected_costs = np.array(expected_costs)
        labels = [[1, 2], [1, 1]]

        fn_k2 = init_k2_rnnt(num_classes=acts.shape[-1], blank=0, reduction='none')
        k2_costs, k2_grads = wrap_and_call(fn_k2, acts, labels, device)

        assert np.allclose(k2_costs, expected_costs), "big_test average costs mismatch."
        assert np.allclose(k2_grads, expected_grads, rtol=1e-3), "big_test grads for average cost mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    def test_case_large_random(self, device, k2_is_appropriate, k2_cuda_is_enabled):
        skip_test_if_unsupported(device, k2_is_appropriate, k2_cuda_is_enabled)

        rng = np.random.RandomState(0)
        acts = rng.randn(4, 8, 11, 5)
        labels = [
            [1, 2, 4, 3, 2, 2, 1, 1, 1, 1],
            [3, 2, 2, 3, 4, 1, 1, 1, 1, 1],
            [4, 4, 1, 2, 1, 3, 4, 3, 1, 2],
            [1, 1, 2, 1, 2, 3, 3, 1, 1, 1],
        ]

        fn_k2 = init_k2_rnnt(num_classes=acts.shape[-1], blank=0, reduction='sum')
        k2_costs, k2_grads = wrap_and_call(fn_k2, acts, labels, device)

        fn_np = RNNTLoss_Numpy()
        np_costs, np_grads = wrap_and_call(fn_np, acts, labels, device)

        assert np.allclose(k2_costs, np_costs, atol=1e-5, rtol=1e-3), "large_random_test costs mismatch."
        assert np.allclose(k2_grads, np_grads, atol=1e-5, rtol=1e-3), "large_random_test gradient mismatch."


if __name__ == "__main__":
    pytest.main([__file__])
