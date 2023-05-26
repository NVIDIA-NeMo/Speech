# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass
from typing import Optional, Type

import numpy as np
import pytest
import torch


class RNNTTestHelper:
    @staticmethod
    def wrap_and_call(fn, acts, labels, device, input_lengths=None, target_lengths=None):
        if not torch.is_tensor(acts):
            acts = torch.FloatTensor(acts)

        if 'cuda' in device:
            acts = acts.cuda()

        if not acts.requires_grad:
            acts.requires_grad = True

        labels = torch.LongTensor(labels)

        if input_lengths is None:
            lengths = [acts.shape[1]] * acts.shape[0]
            lengths = torch.LongTensor(lengths)
        else:
            lengths = input_lengths

        if target_lengths is None:
            label_lengths = [len(l) for l in labels]
            label_lengths = torch.LongTensor(label_lengths)
        else:
            label_lengths = target_lengths

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


@dataclass
class RnntLossSampleData:
    vocab_size: int
    blank_id: int

    logits: torch.Tensor
    targets: torch.Tensor
    input_lengths: torch.Tensor
    target_lengths: torch.Tensor

    expected_cost: Optional[torch.Tensor] = None
    expected_grads: Optional[torch.Tensor] = None

    @classmethod
    def get_sample_small(cls) -> "RnntLossSampleData":
        activations = np.array(
            [
                [
                    [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.2, 0.8, 0.1]],
                    [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.2, 0.1, 0.1], [0.7, 0.1, 0.2, 0.1, 0.1]],
                ]
            ]
        )
        labels = np.asarray([[1, 2]])

        expected_cost = [4.495666]
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
        return RnntLossSampleData(
            vocab_size=3,
            blank_id=0,
            logits=torch.from_numpy(activations).to(torch.float32),
            targets=torch.from_numpy(labels),
            input_lengths=torch.tensor([2]),
            target_lengths=torch.tensor([2]),
            expected_cost=torch.tensor(expected_cost).to(torch.float32),
            expected_grads=torch.from_numpy(expected_grads),
        )

    @classmethod
    def get_sample_small_blank_last(cls) -> "RnntLossSampleData":
        activations = np.array(
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
        labels = np.array([[0, 1, 0]])

        expected_cost = [6.789285182952881]
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
        return RnntLossSampleData(
            vocab_size=3,
            blank_id=2,
            logits=torch.from_numpy(activations).to(torch.float32),
            targets=torch.from_numpy(labels),
            input_lengths=torch.tensor([5]),
            target_lengths=torch.tensor([3]),
            expected_cost=torch.tensor(expected_cost).to(torch.float32),
            expected_grads=torch.from_numpy(expected_grads),
        )

    @classmethod
    def get_sample_medium(cls) -> "RnntLossSampleData":
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

        expected_cost = [4.2806528590890736, 3.9384369822503591]
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
        activations = np.array(activations)
        labels = np.array([[1, 2], [1, 1]])
        expected_grads = np.array(expected_grads)

        return RnntLossSampleData(
            vocab_size=3,
            blank_id=0,
            logits=torch.from_numpy(activations).to(torch.float32),
            targets=torch.from_numpy(labels),
            input_lengths=torch.tensor([4, 4]),
            target_lengths=torch.tensor([2, 2]),
            expected_cost=torch.tensor(expected_cost).to(torch.float32),
            expected_grads=torch.from_numpy(expected_grads),
        )

    @classmethod
    def get_sample_small_random(cls, blank_first: bool, device=torch.device("cpu")) -> "RnntLossSampleData":
        vocab_size = 4
        blank_id = 0 if blank_first else vocab_size - 1
        num_frames = 4
        text_len = 2
        if blank_first:
            text = np.asarray([1, 3])
        else:
            text = np.asarray([0, 2])

        targets = torch.from_numpy(text).unsqueeze(0).to(device)
        logits = torch.rand([1, num_frames, text_len + 1, vocab_size], requires_grad=True, device=device)
        input_lengths = torch.tensor([num_frames], device=device)
        target_lengths = torch.tensor([text_len], device=device)
        return RnntLossSampleData(
            vocab_size=vocab_size,
            blank_id=blank_id,
            logits=logits,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
        )

    @classmethod
    def get_sample_medium_random_var_size(cls, blank_first: bool, device=torch.device("cpu")) -> "RnntLossSampleData":
        vocab_size = 32
        blank_id = 0 if blank_first else vocab_size - 1
        num_frames = 32
        text_len = 27
        min_symbol = 1 if blank_first else 0
        max_symbol = vocab_size if blank_first else vocab_size - 1
        batch_size = 4

        rs = np.random.RandomState(2021)
        text = rs.randint(min_symbol, max_symbol, size=(batch_size, text_len))
        targets = torch.from_numpy(text).to(device)

        logits = torch.rand([batch_size, num_frames, text_len + 1, vocab_size], requires_grad=True, device=device)
        input_lengths = torch.tensor([num_frames, num_frames // 2, text_len, text_len // 2], device=device).long()
        target_lengths = torch.tensor([text_len, text_len - 1, text_len - 3, text_len - 10], device=device)
        return RnntLossSampleData(
            vocab_size=vocab_size,
            blank_id=blank_id,
            logits=logits,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
        )


@pytest.fixture(scope="session")
def rnnt_test_helper() -> Type[RNNTTestHelper]:
    return RNNTTestHelper


@pytest.fixture(scope="session")
def rnn_loss_sample_data() -> Type[RnntLossSampleData]:
    return RnntLossSampleData
