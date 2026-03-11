# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from nemo.collections.asr.parts.rnnt_triton.rnnt_loss import TritonRnntLoss
from nemo.core.utils.optional_libs import TRITON_AVAILABLE

if not TRITON_AVAILABLE:
    pytest.skip(reason="Triton in unavailable, skipping Triton RNN-T loss tests")

DEVICES = []

if torch.cuda.is_available():
    DEVICES.append('cuda')


def wrap_and_call(fn, acts, labels, device):
    if not torch.is_tensor(acts):
        acts = torch.from_numpy(acts)

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


class TestTritonRnntLoss:
    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    def test_case_small(self, device):

        acts = np.array(
            [
                [
                    [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.2, 0.8, 0.1]],
                    [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.2, 0.1, 0.1], [0.7, 0.1, 0.2, 0.1, 0.1]],
                ]
            ]
        )
        labels = [[1, 2]]

        triton_rnnt = TritonRnntLoss(blank=0)
        triton_loss, triton_grad = wrap_and_call(triton_rnnt, acts, labels, device)

        expected_loss = 4.495666
        expected_grad = np.array(
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

        assert np.allclose(triton_loss, expected_loss, rtol=1e-6), "small_test costs mismatch."
        assert np.allclose(triton_grad, expected_grad, atol=1e-6), "small_test gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    def test_case_small_blank_last(self, device):
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

        triton_rnnt = TritonRnntLoss(blank=acts.shape[-1] - 1)
        triton_loss, triton_grad = wrap_and_call(triton_rnnt, acts, labels, device)

        expected_loss = 6.789285182952881
        expected_grad = np.array(
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

        assert np.allclose(triton_loss, expected_loss, rtol=1e-6), "small_test_blank_last costs mismatch."
        assert np.allclose(triton_grad, expected_grad, atol=1e-6), "small_test_blank_last gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    def test_case_medium_tensor(self, device):
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

        expected_loss = [4.2806528590890736, 3.9384369822503591]
        expected_grad = [
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
        expected_loss = np.array(expected_loss)
        labels = [[1, 2], [1, 1]]

        triton_rnnt = TritonRnntLoss(blank=0)
        triton_loss, triton_grad = wrap_and_call(triton_rnnt, acts, labels, device)

        assert np.allclose(triton_loss, expected_loss), "big_test average costs mismatch."
        assert np.allclose(triton_grad, expected_grad, rtol=1e-3), "big_test grads for average cost mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    def test_case_small_random(self, device):
        rng = np.random.RandomState(0)
        acts = rng.randn(1, 4, 3, 3)
        labels = [[1, 2]]

        triton_rnnt = TritonRnntLoss(blank=0)
        triton_loss, triton_grad = wrap_and_call(triton_rnnt, acts, labels, device)

        fn_np = RNNTLoss_Numpy()
        np_loss, np_grad = wrap_and_call(fn_np, acts, labels, device)

        assert np.allclose(triton_loss, np_loss, rtol=1e-6), "small_random_test costs mismatch."
        assert np.allclose(triton_grad, np_grad, atol=1e-6), "small_random_test gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize('src_length,tgt_length', [(10, 2), (1, 13)])
    @pytest.mark.parametrize('device', DEVICES)
    def test_case_small_randomized(self, device, src_length: int, tgt_length: int):
        rng = np.random.RandomState(0)
        vocab_size = 5
        acts = rng.randn(1, src_length, tgt_length + 1, vocab_size)
        labels = rng.randint(low=1, high=vocab_size, size=[1, tgt_length])

        triton_rnnt = TritonRnntLoss(blank=0)
        triton_loss, triton_grad = wrap_and_call(triton_rnnt, acts, labels, device)

        fn_np = RNNTLoss_Numpy()
        np_loss, np_grad = wrap_and_call(fn_np, acts, labels, device)

        assert np.allclose(triton_loss, np_loss, rtol=1e-6), "small_random_test costs mismatch."
        assert np.allclose(triton_grad, np_grad, atol=1e-6), "small_random_test gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    def test_case_medium_random(self, device):
        rng = np.random.RandomState(0)
        acts = rng.randn(4, 8, 11, 5)
        labels = [
            [1, 2, 4, 3, 2, 2, 1, 1, 1, 1],
            [3, 2, 2, 3, 4, 1, 1, 1, 1, 1],
            [4, 4, 1, 2, 1, 3, 4, 3, 1, 2],
            [1, 1, 2, 1, 2, 3, 3, 1, 1, 1],
        ]

        triton_rnnt = TritonRnntLoss(blank=0)
        triton_loss, triton_grad = wrap_and_call(triton_rnnt, acts, labels, device)

        fn_np = RNNTLoss_Numpy()
        np_loss, np_grad = wrap_and_call(fn_np, acts, labels, device)

        # numpy reference returns sum of all losses as a single scalar
        assert np.allclose(triton_loss.sum(), np_loss.sum(), atol=1e-5, rtol=1e-3), "large_random_test costs mismatch."
        assert np.allclose(triton_grad, np_grad, atol=1e-5, rtol=1e-3), "large_random_test gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    @pytest.mark.parametrize('vocab_size', [1024, 1025])
    def test_case_large_random(self, device, vocab_size):
        rng = np.random.RandomState(42)
        B, T, U_plus_1 = 16, 129, 65
        U = U_plus_1 - 1
        acts = rng.randn(B, T, U_plus_1, vocab_size).astype(np.float32)
        labels = rng.randint(1, vocab_size, size=(B, U)).tolist()

        acts_tensor = torch.from_numpy(acts).cuda().requires_grad_(True)
        labels_tensor = torch.LongTensor(labels).cuda()
        lengths = torch.full([B], T, dtype=torch.long).cuda()
        label_lengths = torch.full([B], U, dtype=torch.long).cuda()

        triton_rnnt = TritonRnntLoss(blank=0)
        triton_costs = triton_rnnt(acts_tensor, labels_tensor, lengths, label_lengths)
        triton_total = triton_costs.sum()
        triton_total.backward()
        torch.cuda.synchronize()
        triton_loss = triton_costs.detach().cpu().numpy()
        triton_grad = acts_tensor.grad.data.cpu().numpy()

        fn_np = RNNTLoss_Numpy()
        np_loss, np_grad = wrap_and_call(fn_np, acts, labels, device)

        # numpy reference returns sum of all losses as a single scalar
        assert np.allclose(triton_loss.sum(), np_loss.sum(), atol=1e-4, rtol=1e-3), (
            f"large_random_test (vocab={vocab_size}) costs mismatch: "
            f"triton_sum={triton_loss.sum()}, np_sum={np_loss.sum()}"
        )
        assert np.allclose(triton_grad, np_grad, atol=1e-4, rtol=1e-3), (
            f"large_random_test (vocab={vocab_size}) gradient mismatch: "
            f"max_diff={np.max(np.abs(triton_grad - np_grad))}"
        )

    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    def test_case_single_frame(self, device):
        """B=1, T=1, U+1=2 — minimal non-trivial case."""
        rng = np.random.RandomState(123)
        acts = rng.randn(1, 1, 2, 3).astype(np.float32)
        labels = [[1]]

        triton_rnnt = TritonRnntLoss(blank=0)
        triton_loss, triton_grad = wrap_and_call(triton_rnnt, acts, labels, device)

        fn_np = RNNTLoss_Numpy()
        np_loss, np_grad = wrap_and_call(fn_np, acts, labels, device)

        assert np.allclose(triton_loss, np_loss, rtol=1e-6), "single_frame_test costs mismatch."
        assert np.allclose(triton_grad, np_grad, atol=1e-6), "single_frame_test gradient mismatch."

    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    def test_case_variable_lengths(self, device):
        """B=4 with different src_lengths and tgt_lengths per element."""
        rng = np.random.RandomState(456)
        B, T_max, U_max_plus_1, V = 4, 8, 6, 5
        U_max = U_max_plus_1 - 1
        acts = rng.randn(B, T_max, U_max_plus_1, V).astype(np.float32)

        src_lengths = [3, 5, 8, 4]
        tgt_lengths = [2, 4, 5, 1]
        labels_list = []
        for b in range(B):
            u = tgt_lengths[b]
            lab = rng.randint(1, V, size=U_max).tolist()
            labels_list.append(lab[:U_max])

        acts_tensor = torch.from_numpy(acts).cuda().requires_grad_(True)
        labels_tensor = torch.LongTensor(labels_list).cuda()
        lengths_tensor = torch.LongTensor(src_lengths).cuda()
        label_lengths_tensor = torch.LongTensor(tgt_lengths).cuda()

        triton_rnnt = TritonRnntLoss(blank=0)
        triton_costs = triton_rnnt(acts_tensor, labels_tensor, lengths_tensor, label_lengths_tensor)
        triton_total = triton_costs.sum()
        triton_total.backward()
        torch.cuda.synchronize()
        triton_loss_vals = triton_costs.detach().cpu().numpy()
        triton_grad_vals = acts_tensor.grad.data.cpu().numpy()

        # Compare per-element against numpy reference
        fn_np = RNNTLoss_Numpy()
        for b in range(B):
            t = src_lengths[b]
            u = tgt_lengths[b] + 1
            single_acts = acts[b : b + 1, :t, :u, :]
            single_labels = [labels_list[b][: u - 1]]

            np_loss, np_grad = wrap_and_call(fn_np, single_acts, single_labels, device)
            assert np.allclose(
                triton_loss_vals[b], np_loss[0], atol=1e-5, rtol=1e-3
            ), f"variable_lengths_test costs mismatch for batch element {b}"
            assert np.allclose(
                triton_grad_vals[b, :t, :u, :], np_grad[0], atol=1e-5, rtol=1e-3
            ), f"variable_lengths_test gradient mismatch for batch element {b}"

    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    def test_case_bfloat16(self, device):
        """Verify bfloat16 inputs produce reasonable results (wider tolerance)."""
        rng = np.random.RandomState(789)
        acts = rng.randn(2, 4, 3, 5).astype(np.float32)
        labels = [[1, 2], [3, 1]]

        # Reference in float32
        fn_np = RNNTLoss_Numpy()
        np_loss, _ = wrap_and_call(fn_np, acts, labels, device)

        # Triton with bfloat16
        acts_tensor = torch.from_numpy(acts).to(torch.bfloat16).cuda().requires_grad_(True)
        labels_tensor = torch.LongTensor(labels).cuda()
        lengths = torch.LongTensor([4, 4]).cuda()
        label_lengths = torch.LongTensor([2, 2]).cuda()

        triton_rnnt = TritonRnntLoss(blank=0)
        triton_costs = triton_rnnt(acts_tensor, labels_tensor, lengths, label_lengths)
        triton_total = triton_costs.sum()
        triton_total.backward()
        torch.cuda.synchronize()
        triton_loss = triton_costs.detach().cpu().float().numpy()

        # numpy reference returns sum of all losses as a single scalar
        assert np.allclose(
            triton_loss.sum(), np_loss.sum(), atol=0.1, rtol=0.05
        ), f"bfloat16_test costs mismatch: triton_sum={triton_loss.sum()}, np_sum={np_loss.sum()}"
        assert acts_tensor.grad is not None, "bfloat16_test: gradients were not computed"

    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    def test_no_cuda_sync_operations(self, device):
        """Verify no CPU-GPU sync during loss computation."""
        from tests.collections.asr.decoding.utils import avoid_sync_operations

        rng = np.random.RandomState(101)
        acts = rng.randn(2, 4, 3, 5).astype(np.float32)
        labels = [[1, 2], [3, 1]]

        acts_tensor = torch.from_numpy(acts).cuda().requires_grad_(True)
        labels_tensor = torch.LongTensor(labels).cuda()
        lengths = torch.LongTensor([4, 4]).cuda()
        label_lengths = torch.LongTensor([2, 2]).cuda()

        triton_rnnt = TritonRnntLoss(blank=0)

        with avoid_sync_operations(acts_tensor.device):
            triton_costs = triton_rnnt(acts_tensor, labels_tensor, lengths, label_lengths)
            triton_total = triton_costs.sum()
            triton_total.backward()

    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    @pytest.mark.parametrize('fastemit_lambda', [0.001, 0.01, 0.1])
    def test_fastemit_small_random(self, device, fastemit_lambda):
        """Compare FastEmit loss and gradients against numpy reference."""
        rng = np.random.RandomState(0)
        acts = rng.randn(1, 4, 3, 3).astype(np.float32)
        labels = [[1, 2]]

        triton_rnnt = TritonRnntLoss(blank=0, fastemit_lambda=fastemit_lambda)
        triton_loss, triton_grad = wrap_and_call(triton_rnnt, acts, labels, device)

        fn_np = RNNTLoss_Numpy(fastemit_lambda=fastemit_lambda)
        np_loss, np_grad = wrap_and_call(fn_np, acts, labels, device)

        assert np.allclose(triton_loss, np_loss, rtol=1e-5), (
            f"fastemit_small_random (lambda={fastemit_lambda}) costs mismatch: " f"triton={triton_loss}, np={np_loss}"
        )
        assert np.allclose(triton_grad, np_grad, atol=1e-5), (
            f"fastemit_small_random (lambda={fastemit_lambda}) gradient mismatch: "
            f"max_diff={np.max(np.abs(triton_grad - np_grad))}"
        )

    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    def test_fastemit_medium_random(self, device):
        """Compare FastEmit with multi-batch against numpy reference."""
        rng = np.random.RandomState(42)
        acts = rng.randn(4, 8, 5, 6).astype(np.float32)
        labels = [[1, 2, 3, 4], [2, 3, 1, 5], [4, 1, 2, 3], [3, 5, 1, 2]]
        fastemit_lambda = 0.01

        triton_rnnt = TritonRnntLoss(blank=0, fastemit_lambda=fastemit_lambda)
        triton_loss, triton_grad = wrap_and_call(triton_rnnt, acts, labels, device)

        fn_np = RNNTLoss_Numpy(fastemit_lambda=fastemit_lambda)
        np_loss, np_grad = wrap_and_call(fn_np, acts, labels, device)

        # numpy reference returns sum of all losses as a single scalar
        assert np.allclose(triton_loss.sum(), np_loss.sum(), atol=1e-4, rtol=1e-3), (
            f"fastemit_medium_random costs mismatch: " f"triton_sum={triton_loss.sum()}, np_sum={np_loss.sum()}"
        )
        assert np.allclose(triton_grad, np_grad, atol=1e-4, rtol=1e-3), (
            f"fastemit_medium_random gradient mismatch: " f"max_diff={np.max(np.abs(triton_grad - np_grad))}"
        )

    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    @pytest.mark.parametrize('fastemit_lambda', [0.01, 0.1])
    def test_fastemit_variable_lengths(self, device, fastemit_lambda):
        """B=4 with different src_lengths and tgt_lengths per element + FastEmit."""
        rng = np.random.RandomState(456)
        B, T_max, U_max_plus_1, V = 4, 8, 6, 5
        U_max = U_max_plus_1 - 1
        acts = rng.randn(B, T_max, U_max_plus_1, V).astype(np.float32)

        src_lengths = [3, 5, 8, 4]
        tgt_lengths = [2, 4, 5, 1]
        labels_list = []
        for b in range(B):
            lab = rng.randint(1, V, size=U_max).tolist()
            labels_list.append(lab[:U_max])

        acts_tensor = torch.from_numpy(acts).cuda().requires_grad_(True)
        labels_tensor = torch.LongTensor(labels_list).cuda()
        lengths_tensor = torch.LongTensor(src_lengths).cuda()
        label_lengths_tensor = torch.LongTensor(tgt_lengths).cuda()

        triton_rnnt = TritonRnntLoss(blank=0, fastemit_lambda=fastemit_lambda)
        triton_costs = triton_rnnt(acts_tensor, labels_tensor, lengths_tensor, label_lengths_tensor)
        triton_total = triton_costs.sum()
        triton_total.backward()
        torch.cuda.synchronize()
        triton_loss_vals = triton_costs.detach().cpu().numpy()
        triton_grad_vals = acts_tensor.grad.data.cpu().numpy()

        # Compare per-element against numpy reference
        fn_np = RNNTLoss_Numpy(fastemit_lambda=fastemit_lambda)
        for b in range(B):
            t = src_lengths[b]
            u = tgt_lengths[b] + 1
            single_acts = acts[b : b + 1, :t, :u, :]
            single_labels = [labels_list[b][: u - 1]]

            np_loss, np_grad = wrap_and_call(fn_np, single_acts, single_labels, device)
            assert np.allclose(
                triton_loss_vals[b], np_loss[0], atol=1e-5, rtol=1e-3
            ), f"fastemit_variable_lengths (lambda={fastemit_lambda}) costs mismatch for batch element {b}"
            assert np.allclose(
                triton_grad_vals[b, :t, :u, :], np_grad[0], atol=1e-5, rtol=1e-3
            ), f"fastemit_variable_lengths (lambda={fastemit_lambda}) gradient mismatch for batch element {b}"

    @pytest.mark.unit
    @pytest.mark.parametrize('device', DEVICES)
    def test_fastemit_zero_lambda_unchanged(self, device):
        """Verify that fastemit_lambda=0.0 gives identical results to no FastEmit."""
        rng = np.random.RandomState(7)
        acts = rng.randn(2, 5, 4, 6).astype(np.float32)
        labels = [[1, 2, 3], [4, 1, 2]]

        triton_no_fe = TritonRnntLoss(blank=0)
        loss_no_fe, grad_no_fe = wrap_and_call(triton_no_fe, acts, labels, device)

        triton_fe_zero = TritonRnntLoss(blank=0, fastemit_lambda=0.0)
        loss_fe_zero, grad_fe_zero = wrap_and_call(triton_fe_zero, acts, labels, device)

        assert np.array_equal(loss_no_fe, loss_fe_zero), "fastemit_lambda=0.0 should give identical loss"
        assert np.array_equal(grad_no_fe, grad_fe_zero), "fastemit_lambda=0.0 should give identical gradients"


if __name__ == "__main__":
    pytest.main([__file__])
