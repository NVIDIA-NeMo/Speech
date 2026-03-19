# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import math
import random

import numpy as np
import pytest
import torch

from nemo.collections.tts.modules.ffn_modules import ConvolutionLayer, PositionwiseConvFF
from nemo.collections.tts.modules.moe_modules import MoERouter, PositionwiseConvFFMoE
from nemo.collections.tts.modules.transformer_2501 import CrossAttention, SelfAttention, Transformer, TransformerLayer
from nemo.collections.tts.parts.utils.tts_dataset_utils import beta_binomial_prior_distribution


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths


@pytest.mark.unit
class TestConvolutionLayer:
    @classmethod
    def setup_class(cls):
        cls.in_channels = 3
        cls.out_channels = 6
        cls.kernel_size = 3
        cls.stride = 1
        cls.dilation = 1
        cls.bias = True
        # fmt:off
        cls.input_tensor = torch.Tensor(
            [[[-1.0542,  0.2675,  0.6963,  0.4738,  0.3910, -0.1505,  0.9171, -0.1528,  3.7269,  0.1779],
              [-1.0317,  1.6818,  1.4257, -0.5003, -1.7254,  0.8830, -0.4541, -0.4631, -0.0986,  0.5083],
              [-0.3231, -1.0899,  0.5774,  0.1661,  0.9620, -2.3307, -0.6158, -0.3663,  1.2469, -1.0208]]]
        )
        cls.input_mask = torch.ones(1, cls.input_tensor.shape[2])
        # fmt:on

    def test_non_causal_forward(self):
        set_seed(0)
        layer = ConvolutionLayer(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            dilation=self.dilation,
            bias=self.bias,
            is_causal=False,
        )

        with torch.no_grad():
            output_tensor = layer(self.input_tensor, self.input_mask)

        # fmt:off
        expected_output_tensor = torch.Tensor(
            [[[ 0.1912, -0.0555, -0.2681, -0.2289,  1.0788, -0.3908,  0.0936, -0.7962,  1.3754, -0.0731],
              [-0.3715, -0.6326, -0.9596, -0.0933, -0.1024, -0.2082, -0.5924, 0.1097, -0.5418, -0.0854],
              [ 0.3974,  0.4537,  0.3299,  0.1471, -0.5983, -0.8645,  0.0975, 0.6063, -0.6619, -0.9711],
              [-0.3048,  0.3862, -0.2462, -0.9903, -0.6189,  0.7389,  0.0785, -1.0870, -1.0018, -1.2426],
              [-0.4357, -0.0446,  0.0879,  0.0930, -0.2242,  0.5285,  0.4006, -0.1846,  0.5668, -0.5242],
              [-0.0625,  0.4123, -0.6289, -0.4317,  0.1595,  0.0386, -1.0774, 0.2218,  0.8483, -0.4886]]]
        )
        # fmt:on

        assert torch.allclose(output_tensor, expected_output_tensor, atol=1e-4)

    def test_causal_forward(self):
        set_seed(0)
        layer = ConvolutionLayer(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            dilation=self.dilation,
            bias=self.bias,
            is_causal=True,
        )

        with torch.no_grad():
            output_tensor = layer(self.input_tensor, self.input_mask)

        # fmt:off
        expected_output_tensor = torch.Tensor(
            [[[ 0.4301,  0.1912, -0.0555, -0.2681, -0.2289,  1.0788, -0.3908, 0.0936, -0.7962,  1.3754],
              [-0.0501, -0.3715, -0.6326, -0.9596, -0.0933, -0.1024, -0.2082, -0.5924,  0.1097, -0.5418],
              [-0.4204,  0.3974,  0.4537,  0.3299,  0.1471, -0.5983, -0.8645, 0.0975,  0.6063, -0.6619],
              [ 0.1543, -0.3048,  0.3862, -0.2462, -0.9903, -0.6189,  0.7389, 0.0785, -1.0870, -1.0018],
              [-0.1337, -0.4357, -0.0446,  0.0879,  0.0930, -0.2242,  0.5285, 0.4006, -0.1846,  0.5668],
              [-0.6127, -0.0625,  0.4123, -0.6289, -0.4317,  0.1595,  0.0386, -1.0774,  0.2218,  0.8483]]]
        )
        # fmt:on

        assert torch.allclose(output_tensor, expected_output_tensor, atol=1e-4)


class TestPositionwiseConvFF:
    @classmethod
    def setup_class(cls):
        cls.d_model = 3
        cls.d_ffn = 12
        cls.p_dropout = 0.0
        cls.kernel_size = 3
        cls.bias = True
        # fmt:off
        cls.input_tensor = torch.Tensor(
            [[[-1.6682, -0.6069,  0.1321],
              [-1.5489,  0.3279, -0.9159],
              [-0.7490,  1.8984,  0.5030],
              [-0.8130,  0.0058, -1.9979],
              [-1.4994, -0.3270,  1.4961],
              [-1.6613, -1.7827,  0.8932],
              [-0.6276, -1.0770, -0.9971],
              [ 1.5424,  1.3590,  1.2287],
              [-0.1543,  0.3365,  1.7475],
              [-0.1753,  0.4115,  0.0772]]]
        )
        cls.input_mask = torch.ones(1, cls.input_tensor.shape[1])
        # fmt:on

    def test_causal_forward(self):
        set_seed(0)
        layer = PositionwiseConvFF(
            self.d_model, self.d_ffn, self.p_dropout, self.kernel_size, bias=self.bias, is_causal=True
        )

        with torch.no_grad():
            output_tensor = layer(self.input_tensor, self.input_mask)

        # fmt:off
        expected_output_tensor = torch.Tensor(
            [[[-0.1242, -0.0114,  0.0212],
              [-0.0441, -0.0555, -0.0795],
              [-0.0282,  0.0366, -0.2033],
              [-0.0421,  0.0305, -0.2573],
              [-0.1877, -0.2492, -0.1638],
              [-0.4300, -0.1160,  0.2177],
              [-0.1652, -0.3130, -0.3329],
              [-0.1737,  0.1133, -0.1802],
              [-0.2599, -0.0381,  0.1362],
              [-0.0584, -0.2936,  0.2719]]]
        )
        # fmt:on

        assert torch.allclose(output_tensor, expected_output_tensor, atol=1e-4)

    def test_non_causal_forward(self):
        set_seed(0)
        layer = PositionwiseConvFF(
            self.d_model, self.d_ffn, self.p_dropout, self.kernel_size, bias=self.bias, is_causal=False
        )

        with torch.no_grad():
            output_tensor = layer(self.input_tensor, self.input_mask)

        # fmt:off
        expected_output_tensor = torch.Tensor(
            [[[-0.0617, -0.0321, -0.1646],
              [-0.0421,  0.0305, -0.2573],
              [-0.1877, -0.2492, -0.1638],
              [-0.4300, -0.1160,  0.2177],
              [-0.1652, -0.3130, -0.3329],
              [-0.1737,  0.1133, -0.1802],
              [-0.2599, -0.0381,  0.1362],
              [-0.0584, -0.2936,  0.2719],
              [ 0.0361,  0.1110,  0.0441],
              [-0.0244,  0.0682,  0.0340]]]
        )
        # fmt:on

        assert torch.allclose(output_tensor, expected_output_tensor, atol=1e-4)


class TestSelfAttention:
    @classmethod
    def setup_class(cls):
        cls.n_heads = 2
        cls.d_model = 4
        cls.p_dropout = 0.0
        cls.max_length_causal_mask = 6
        # fmt:off
        cls.query_tensor = torch.Tensor(
            [[[ 0.7239, -0.2362, -0.6610, -1.3759],
              [ 1.7381,  0.0793, -1.1241,  0.9529],
              [-1.9809,  0.2217,  0.0795,  0.0307],
              [ 0.3208,  0.4485,  0.3046, -0.0704],
              [-1.4412,  0.8981,  0.1219,  0.0481],
              [ 1.7811, -0.1358,  0.6073,  0.8275]]]
        )
        # fmt:on

    def test_causal_forward(self):
        set_seed(0)
        layer = SelfAttention(
            self.n_heads,
            self.d_model,
            self.p_dropout,
            is_causal=True,
            max_length_causal_mask=self.max_length_causal_mask,
        )
        query_mask = torch.ones(1, self.max_length_causal_mask).bool()
        with torch.no_grad():
            output_tensor, attn_output = layer(self.query_tensor, query_mask)

        # fmt:off
        expected_output_tensor = torch.Tensor(
            [[[-0.2569,  0.2782, -0.0348,  0.2480],
              [-0.3949,  0.4054, -0.0876,  0.2574],
              [-0.1033,  0.0659, -0.0259,  0.1738],
              [-0.1485,  0.0995, -0.0415,  0.0684],
              [-0.0123,  0.0185, -0.0027,  0.0708],
              [-0.0672,  0.0566, -0.0214, -0.0021]]]
        )
        expected_attn_prob = torch.Tensor(
            [[[[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
               [0.5828, 0.4172, 0.0000, 0.0000, 0.0000, 0.0000],
               [0.3216, 0.4260, 0.2525, 0.0000, 0.0000, 0.0000],
               [0.2385, 0.2238, 0.2872, 0.2504, 0.0000, 0.0000],
               [0.1807, 0.1973, 0.2057, 0.2045, 0.2118, 0.0000],
               [0.1159, 0.1388, 0.2010, 0.1721, 0.2161, 0.1562]],
              [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
               [0.2866, 0.7134, 0.0000, 0.0000, 0.0000, 0.0000],
               [0.2799, 0.2472, 0.4729, 0.0000, 0.0000, 0.0000],
               [0.2964, 0.2535, 0.2075, 0.2427, 0.0000, 0.0000],
               [0.1864, 0.1616, 0.2394, 0.1974, 0.2152, 0.0000],
               [0.1666, 0.2030, 0.1391, 0.1649, 0.1546, 0.1719]]]]
        )
        expected_attn_score = torch.Tensor(
            [[[[ 0.5248, float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf')],
               [-0.1948, -0.5291, float('-inf'), float('-inf'), float('-inf'), float('-inf')],
               [-0.0279,  0.2533, -0.2698, float('-inf'), float('-inf'), float('-inf')],
               [-0.0508, -0.1145,  0.1350, -0.0020, float('-inf'), float('-inf')],
               [-0.0985, -0.0105,  0.0315,  0.0257,  0.0604, float('-inf')],
               [-0.3253, -0.1457,  0.2250,  0.0694,  0.2971, -0.0275]],
              [[ 0.5075, float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf')],
               [-0.5541,  0.3578, float('-inf'), float('-inf'), float('-inf'), float('-inf')],
               [-0.2499, -0.3738,  0.2746, float('-inf'), float('-inf'), float('-inf')],
               [ 0.2215,  0.0654, -0.1351,  0.0216, float('-inf'), float('-inf')],
               [-0.1011, -0.2439,  0.1488, -0.0438,  0.0425, float('-inf')],
               [ 0.0526,  0.2502, -0.1277,  0.0424, -0.0221,  0.0840]]]]
        )
        # fmt:on

        assert torch.allclose(output_tensor, expected_output_tensor, atol=1e-4)
        assert torch.allclose(attn_output[0], expected_attn_prob, atol=1e-4)
        assert torch.allclose(attn_output[1], expected_attn_score, atol=1e-4)

    def test_non_causal_forward(self):
        set_seed(0)
        layer = SelfAttention(
            self.n_heads,
            self.d_model,
            self.p_dropout,
            is_causal=False,
            max_length_causal_mask=self.max_length_causal_mask,
        )
        query_mask = torch.ones(1, self.max_length_causal_mask).bool()
        with torch.no_grad():
            output_tensor, attn_output = layer(self.query_tensor, query_mask)

        # fmt:off
        expected_output_tensor = torch.Tensor(
            [[[-0.0954,  0.1131, -0.0195,  0.0704],
              [-0.0401,  0.0364, -0.0156,  0.0088],
              [ 0.0324,  0.0368,  0.0174,  0.0501],
              [-0.0633,  0.0610, -0.0176,  0.0180],
              [-0.0017,  0.0361,  0.0030,  0.0319],
              [-0.0672,  0.0566, -0.0214, -0.0021]]]
        )
        expected_attn_prob = torch.Tensor(
            [[[[0.2835, 0.1482, 0.1723, 0.1426, 0.1422, 0.1111],
               [0.1254, 0.0898, 0.2819, 0.1493, 0.2682, 0.0854],
               [0.1549, 0.2051, 0.1216, 0.1663, 0.1294, 0.2228],
               [0.1583, 0.1485, 0.1906, 0.1662, 0.1890, 0.1475],
               [0.1498, 0.1635, 0.1705, 0.1696, 0.1755, 0.1711],
               [0.1159, 0.1388, 0.2010, 0.1721, 0.2161, 0.1562]],
              [[0.2536, 0.1945, 0.1079, 0.1628, 0.1233, 0.1579],
               [0.0866, 0.2156, 0.1709, 0.1551, 0.1903, 0.1815],
               [0.1361, 0.1202, 0.2300, 0.1626, 0.1941, 0.1569],
               [0.2038, 0.1744, 0.1427, 0.1669, 0.1488, 0.1634],
               [0.1565, 0.1357, 0.2010, 0.1657, 0.1807, 0.1604],
               [0.1666, 0.2030, 0.1391, 0.1649, 0.1546, 0.1719]]]]
        )
        expected_attn_score = torch.Tensor(
            [[[[ 5.2482e-01, -1.2346e-01,  2.7022e-02, -1.6210e-01, -1.6488e-01, -4.1190e-01],
               [-1.9484e-01, -5.2910e-01,  6.1538e-01, -2.0263e-02,  5.6540e-01, -5.7875e-01],
               [-2.7873e-02,  2.5326e-01, -2.6980e-01,  4.3127e-02, -2.0759e-01, 3.3584e-01],
               [-5.0756e-02, -1.1455e-01,  1.3498e-01, -2.0208e-03,  1.2687e-01, -1.2113e-01],
               [-9.8482e-02, -1.0463e-02,  3.1513e-02,  2.5712e-02,  6.0431e-02, 3.4494e-02],
               [-3.2530e-01, -1.4574e-01,  2.2503e-01,  6.9375e-02,  2.9711e-01, -2.7542e-02]],
              [[ 5.0748e-01,  2.4198e-01, -3.4704e-01,  6.4137e-02, -2.1347e-01, 3.3762e-02],
               [-5.5410e-01,  3.5778e-01,  1.2559e-01,  2.8689e-02,  2.3316e-01, 1.8558e-01],
               [-2.4989e-01, -3.7383e-01,  2.7462e-01, -7.2002e-02,  1.0508e-01, -1.0771e-01],
               [ 2.2154e-01,  6.5375e-02, -1.3510e-01,  2.1609e-02, -9.3194e-02, 3.4042e-04],
               [-1.0105e-01, -2.4395e-01,  1.4884e-01, -4.3842e-02,  4.2481e-02, -7.6735e-02],
               [ 5.2595e-02,  2.5018e-01, -1.2765e-01,  4.2375e-02, -2.2093e-02, 8.4005e-02]]]]
        )
        # fmt:on

        assert torch.allclose(output_tensor, expected_output_tensor, atol=1e-4)
        assert torch.allclose(attn_output[0], expected_attn_prob, atol=1e-4)
        assert torch.allclose(attn_output[1], expected_attn_score, atol=1e-5)


class TestCrossAttention:
    @classmethod
    def setup_class(cls):
        cls.n_heads = 2
        cls.d_model = 4
        cls.d_memory = 3
        cls.p_dropout = 0.0
        cls.max_length = 6
        # fmt:off
        # shape = (1, cls.max_length, cls.d_model)
        cls.query_tensor = torch.Tensor(
            [[[0.7352, -0.5871, -0.1204, -2.0200],
              [-0.4618, -0.3604, 1.2287, -0.3434],
              [0.7838, -0.7646, -1.3349, -0.1538],
              [-0.9749, -1.0789, -0.0126, -0.7225],
              [2.6929, -0.2091, 2.1242, -1.0123],
              [0.5094, -2.0566, 1.3922, -0.2156]]]
        )
        # fmt:on
        cls.query_mask = torch.ones(1, cls.query_tensor.shape[1]).bool()
        # fmt:off
        # shape = (1, 5, cls.d_memory)
        cls.memory_tensor = torch.Tensor(
            [[[ 2.0132e-01, -5.6582e-01,  1.1191e+00],
              [-6.2371e-01, -9.3398e-02, -1.3744e+00],
              [-9.8265e-01, -8.1742e-01,  4.5611e-01],
              [-5.4802e-01, -1.1218e+00,  7.6138e-01],
              [-1.9899e+00, -1.7910e-03,  9.0718e-01]]]
        )
        # fmt:on
        cls.memory_mask = torch.ones(1, cls.memory_tensor.shape[1]).bool()
        # shape = (1, cls.query_tensor.shape[1], cls.memory_tensor.shape[1])
        cls.attn_prior = torch.from_numpy(
            beta_binomial_prior_distribution(
                phoneme_count=cls.memory_tensor.shape[1], mel_count=cls.query_tensor.shape[1]
            )
        ).unsqueeze(0)

    def test_forward(self):
        set_seed(0)
        layer = CrossAttention(self.n_heads, self.d_model, self.d_memory, self.p_dropout)

        with torch.no_grad():
            output_tensor, attn_output = layer(
                self.query_tensor, self.query_mask, self.memory_tensor, self.memory_mask, self.attn_prior
            )

        # fmt:off
        expected_output_tensor = torch.Tensor(
            [[[ 0.2267, -0.2271,  0.0573, -0.0681],
              [ 0.2672, -0.1823,  0.0722, -0.0859],
              [ 0.3212, -0.2218,  0.0835, -0.0715],
              [ 0.3568, -0.2573,  0.0918, -0.0789],
              [ 0.3962, -0.4112,  0.0816, -0.1972],
              [ 0.3457, -0.4253,  0.0568, -0.2216]]]
        )
        expected_attn_prob = torch.Tensor(
            [[[[0.4220, 0.4859, 0.0709, 0.0188, 0.0025],
               [0.3944, 0.3475, 0.1642, 0.0784, 0.0155],
               [0.1335, 0.3448, 0.2794, 0.1752, 0.0671],
               [0.0914, 0.3300, 0.2343, 0.2437, 0.1006],
               [0.0256, 0.1138, 0.2145, 0.3343, 0.3119],
               [0.0117, 0.0617, 0.1112, 0.3354, 0.4800]],
              [[0.8045, 0.1024, 0.0661, 0.0242, 0.0028],
               [0.4020, 0.2953, 0.1914, 0.0907, 0.0207],
               [0.1446, 0.2798, 0.3026, 0.1965, 0.0766],
               [0.0718, 0.2151, 0.2778, 0.2719, 0.1634],
               [0.0673, 0.0341, 0.1929, 0.4534, 0.2522],
               [0.0064, 0.0264, 0.0999, 0.2872, 0.5802]]]]
        )
        expected_attn_score = torch.Tensor(
            [[[[-0.5044,  0.4476, -0.4961, -0.5728, -0.8010],
               [-0.0761, -0.2027, -0.5103, -0.4385, -0.6724],
               [-0.1525,  0.2576,  0.0471, -0.0138,  0.0075],
               [-0.3084, -0.0058, -0.7538, -0.7144, -1.0604],
               [-0.0799,  0.0260, -0.1511, -0.1493, -0.2187],
               [-0.1935, -0.3225, -0.9867, -0.8637, -1.3162]],
              [[ 0.4704, -0.7801, -0.2374,  0.0126, -0.3430],
               [ 0.0623, -0.2461, -0.2380, -0.1743, -0.2658],
               [ 0.0104,  0.1313,  0.2096,  0.1834,  0.2217],
               [-0.0859,  0.0300, -0.1194, -0.1410, -0.1111],
               [ 0.7876, -1.2782, -0.3570,  0.0556, -0.5312],
               [ 0.1046, -0.2652, -0.1856, -0.1104, -0.2180]]]]
        )
        # fmt:on

        assert torch.allclose(output_tensor, expected_output_tensor, atol=1e-4)
        assert torch.allclose(attn_output[0], expected_attn_prob, atol=1e-4)
        assert torch.allclose(attn_output[1], expected_attn_score, atol=1e-4)


class TestTransformerLayer:
    @classmethod
    def setup_class(cls):
        cls.d_model = 2
        cls.d_ffn = 8
        cls.sa_n_heads = 2
        cls.kernel_size = 3
        cls.p_dropout = 0.0
        cls.max_length_causal_mask = 5
        # fmt:off
        # shape = (1, cls.max_length_causal_mask, cls.d_model)
        cls.x = torch.Tensor(
            [[[ 0.5115,  0.0889],
              [-0.8568, -2.9632],
              [-1.3728,  0.7325],
              [-2.4593, -0.9018],
              [ 0.9621,  0.4212]]]
        )
        # fmt:on
        cls.x_mask = torch.ones(1, cls.max_length_causal_mask).bool()
        # fmt:off
        # shape = (1, 3, cls.d_model)
        cls.cond = torch.Tensor(
            [[[ 1.4441,  0.1393],
              [ 0.2828, -0.2456],
              [-0.3075,  0.6581]]]
        )
        # fmt:on
        cls.cond_mask = torch.ones(1, cls.cond.shape[1]).bool()
        # shape = (1, cls.x.shape[1], cls.cond.shape[1])
        cls.attn_prior = torch.from_numpy(
            beta_binomial_prior_distribution(phoneme_count=cls.cond.shape[1], mel_count=cls.x.shape[1])
        ).unsqueeze(0)

    def test_forward_causal_self_attn_and_has_xattn(self):
        set_seed(0)
        layer = TransformerLayer(
            self.d_model,
            self.d_ffn,
            self.sa_n_heads,
            self.kernel_size,
            self.p_dropout,
            has_xattn=True,
            xa_n_heads=2,
            xa_d_memory=2,
            is_causal=True,
            max_length_causal_mask=self.max_length_causal_mask,
        )

        with torch.no_grad():
            output_dict = layer(self.x, self.x_mask, self.cond, self.cond_mask, self.attn_prior)

        # fmt:off
        expected_output = {
            'output': torch.Tensor(
                [[[ 0.1936,  0.5387],
                  [-1.0270, -2.5452],
                  [-1.6884,  0.8765],
                  [-2.7496, -0.7887],
                  [ 1.2837,  0.1172]]]
            ),
            'attn_probabilities': {
                'self_attn_probabilities': [
                    torch.Tensor(
                        [[[[1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                           [0.5000, 0.5000, 0.0000, 0.0000, 0.0000],
                           [0.3068, 0.3068, 0.3864, 0.0000, 0.0000],
                           [0.2213, 0.2213, 0.2787, 0.2787, 0.0000],
                           [0.2180, 0.2180, 0.1730, 0.1730, 0.2180]],
                          [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                           [0.5000, 0.5000, 0.0000, 0.0000, 0.0000],
                           [0.3237, 0.3237, 0.3527, 0.0000, 0.0000],
                           [0.2393, 0.2393, 0.2607, 0.2607, 0.0000],
                           [0.2068, 0.2068, 0.1898, 0.1898, 0.2068]]]]
                    ),
                    torch.Tensor(
                        [[[[0.1154, float('-inf'), float('-inf'), float('-inf'), float('-inf')],
                           [0.1154, 0.1154, float('-inf'), float('-inf'), float('-inf')],
                           [-0.1154, -0.1154, 0.1154, float('-inf'), float('-inf')],
                           [-0.1154, -0.1154, 0.1154, 0.1154, float('-inf')],
                           [0.1154, 0.1154, -0.1154, -0.1154, 0.1154]],
                          [[0.0429, float('-inf'), float('-inf'), float('-inf'), float('-inf')],
                           [0.0429, 0.0429, float('-inf'), float('-inf'), float('-inf')],
                           [-0.0429, -0.0429, 0.0429, float('-inf'), float('-inf')],
                           [-0.0429, -0.0429, 0.0429, 0.0429, float('-inf')],
                           [0.0429, 0.0429, -0.0429, -0.0429, 0.0429]]]]
                    )
                ],
                'cross_attn_probabilities': [
                    torch.Tensor(
                        [[[[0.7181, 0.2394, 0.0426],
                           [0.4843, 0.3874, 0.1283],
                           [0.2753, 0.4129, 0.3118],
                           [0.1344, 0.3583, 0.5074],
                           [0.0520, 0.2599, 0.6882]],
                          [[0.5959, 0.1987, 0.2054],
                           [0.2837, 0.2270, 0.4893],
                           [0.3740, 0.5610, 0.0651],
                           [0.2355, 0.6280, 0.1365],
                           [0.0108, 0.0542, 0.9349]]]]
                    ),
                    torch.Tensor(
                        [[[[0.0586, 0.0586, -0.0586],
                           [0.0624, 0.0624, -0.0624],
                           [-0.0624, -0.0624, 0.0624],
                           [-0.0624, -0.0624, 0.0624],
                           [0.0624, 0.0624, -0.0624]],
                          [[-0.8214, -0.8214, 0.8214],
                           [-0.8745, -0.8744, 0.8745],
                           [0.8745, 0.8744, -0.8745],
                           [0.8745, 0.8744, -0.8745],
                           [-0.8744, -0.8744, 0.8744]]]]
                    )
                ]
            }
        }
        # fmt:on

        assert torch.allclose(output_dict["output"], expected_output["output"], atol=1e-4)
        for i in range(2):
            assert torch.allclose(
                output_dict["attn_probabilities"]["self_attn_probabilities"][i],
                expected_output["attn_probabilities"]["self_attn_probabilities"][i],
                atol=1e-4,
            )
            assert torch.allclose(
                output_dict["attn_probabilities"]["cross_attn_probabilities"][i],
                expected_output["attn_probabilities"]["cross_attn_probabilities"][i],
                atol=1e-4,
            )


@pytest.mark.unit
class TestTransformer:
    @classmethod
    def setup_class(cls):
        cls.n_layers = 1
        cls.d_model = 4
        cls.d_ffn = 16
        cls.sa_n_heads = 2
        cls.kernel_size = 3
        cls.p_dropout = 0.0
        cls.p_dropout_out = 0.0
        cls.is_causal = True
        cls.max_length_causal_mask = 6

        # fmt:off
        cls.input_tensor = torch.Tensor(
            [[[ 0.7049,  0.0305, -0.8542,  0.5388],
              [-0.5265, -1.3320,  1.5451,  0.4086],
              [-2.0546,  0.5259,  0.5995, -0.4078],
              [ 0.4530, -0.3918,  2.1403, -0.2062],
              [-0.0984,  0.4855,  0.7076,  0.0431],
              [-0.4394, -0.6761,  1.7389, -0.9423]]]
        )
        # fmt:on

    def test_forward_causal_self_attn_and_no_xattn(self):
        set_seed(0)
        model = Transformer(
            n_layers=self.n_layers,
            d_model=self.d_model,
            d_ffn=self.d_ffn,
            sa_n_heads=self.sa_n_heads,
            kernel_size=self.kernel_size,
            p_dropout=self.p_dropout,
            p_dropout_out=self.p_dropout_out,
            has_xattn=False,
            is_causal=self.is_causal,
            max_length_causal_mask=self.max_length_causal_mask,
        )

        # Check model init
        assert torch.isclose(torch.mean(model.layers[0].pos_ff.proj.conv.weight), torch.tensor(0.0), atol=1e-2)
        assert torch.isclose(torch.std(model.layers[0].pos_ff.proj.conv.weight), torch.tensor(0.02), atol=1e-2)
        assert torch.isclose(torch.mean(model.layers[0].pos_ff.o_net.conv.weight), torch.tensor(0.0), atol=1e-2)
        assert torch.isclose(
            torch.std(model.layers[0].pos_ff.o_net.conv.weight), torch.tensor(0.02 / math.sqrt(2.0)), atol=1e-3
        )

        mask_tensor = torch.ones(1, self.max_length_causal_mask).bool()
        with torch.no_grad():
            output_dict = model(x=self.input_tensor, x_mask=mask_tensor)

        # fmt:off
        expected_output_tensor = {
            'output': torch.Tensor(
                [[[0.7047, 0.0305, -0.8555, 0.5402],
                  [-0.5192, -1.3324, 1.5455, 0.4148],
                  [-2.0593, 0.5290, 0.5969, -0.4101],
                  [0.4517, -0.3968, 2.1392, -0.2041],
                  [-0.1019, 0.4854, 0.7077, 0.0431],
                  [-0.4458, -0.6789, 1.7447, -0.9447]]]
            ),
            'attn_probabilities': [
                {
                    'self_attn_probabilities': [
                        torch.Tensor(
                            [[[[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                               [0.4998, 0.5002, 0.0000, 0.0000, 0.0000, 0.0000],
                               [0.3337, 0.3333, 0.3330, 0.0000, 0.0000, 0.0000],
                               [0.2498, 0.2500, 0.2501, 0.2501, 0.0000, 0.0000],
                               [0.2002, 0.2001, 0.2000, 0.1999, 0.1999, 0.0000],
                               [0.1666, 0.1666, 0.1667, 0.1667, 0.1667, 0.1667]],
                              [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                               [0.5005, 0.4995, 0.0000, 0.0000, 0.0000, 0.0000],
                               [0.3332, 0.3331, 0.3336, 0.0000, 0.0000, 0.0000],
                               [0.2507, 0.2494, 0.2510, 0.2489, 0.0000, 0.0000],
                               [0.2002, 0.1995, 0.2006, 0.1994, 0.2003, 0.0000],
                               [0.1671, 0.1663, 0.1674, 0.1660, 0.1670, 0.1662]]]]
                        ),
                        torch.Tensor(
                            [[[[-3.4823e-04, float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf')],
                               [-7.0210e-04, -5.6984e-05, float('-inf'), float('-inf'), float('-inf'), float('-inf')],
                               [1.2551e-03, 1.1431e-04, -5.9334e-04, float('-inf'), float('-inf'), float('-inf')],
                               [-8.1514e-04, 2.6650e-05, 5.2952e-04, 5.6903e-04, float('-inf'), float('-inf')],
                               [8.0150e-04, 1.4366e-04, -2.7793e-04, -7.0636e-04, -8.2140e-04, float('-inf')],
                               [-4.6137e-04, 6.2648e-05, 3.6768e-04, 2.8095e-04, 4.9188e-04, 3.6888e-04]],
                              [[-7.5861e-04, float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf')],
                               [8.0373e-04, -1.2745e-03, float('-inf'), float('-inf'), float('-inf'), float('-inf')],
                               [-4.5038e-04, -6.7648e-04, 7.5978e-04, float('-inf'), float('-inf'), float('-inf')],
                               [1.5376e-03, -3.6984e-03, 3.0423e-03, -5.4870e-03, float('-inf'), float('-inf')],
                               [3.7014e-04, -2.7010e-03, 2.4310e-03, -3.2604e-03, 9.3840e-04, float('-inf')],
                               [1.3868e-03, -3.8372e-03, 3.2144e-03, -5.4860e-03, 5.0273e-04, -4.4343e-03]]]]
                        ),
                    ],
                    'cross_attn_probabilities': None,
                }
            ],
        }
        # fmt:on

        assert output_dict["output"].shape == expected_output_tensor["output"].shape
        assert torch.allclose(output_dict["output"], expected_output_tensor["output"], atol=1e-4)
        for i in range(2):
            assert torch.allclose(
                output_dict["attn_probabilities"][0]["self_attn_probabilities"][i],
                expected_output_tensor["attn_probabilities"][0]["self_attn_probabilities"][i],
                atol=1e-4,
            )
        assert output_dict["attn_probabilities"][0]["cross_attn_probabilities"] is None

    def test_forward_causal_self_attn_and_has_xattn(self):
        set_seed(0)
        model = Transformer(
            n_layers=2,
            d_model=self.d_model,
            d_ffn=self.d_ffn,
            sa_n_heads=self.sa_n_heads,
            kernel_size=self.kernel_size,
            p_dropout=self.p_dropout,
            p_dropout_out=self.p_dropout_out,
            has_xattn=True,
            xa_d_memory=4,
            xa_n_heads=2,
            is_causal=self.is_causal,
            max_length_causal_mask=self.max_length_causal_mask,
        )

        # fmt:off
        cond = [
            # shape (1, 3, 4)
            torch.Tensor(
                [[[-0.7475,  1.1461,  0.7300,  1.4471],
                  [ 1.8744, -0.1654,  1.2418, -1.6983],
                  [-0.3123,  0.2320,  0.7457,  1.9868]]]
            ),
            # shape (1, 5, 4)
            torch.Tensor(
                [[[-0.6683, -1.2178,  1.3696,  0.9941],
                  [ 0.0297, -0.1616,  0.1891,  0.0580],
                  [-1.0771,  0.2547, -1.4023,  0.0971],
                  [ 1.1132,  0.6311, -0.1449,  0.2351],
                  [ 0.8920,  2.3663,  0.2248, -0.7298]]]
            )
        ]
        # fmt:on

        cond_mask = [torch.ones(1, cond[0].shape[1]).bool(), torch.ones(1, cond[1].shape[1]).bool()]
        mask_tensor = torch.ones(1, self.max_length_causal_mask).bool()
        multi_encoder_mapping = [0, 1]
        with torch.no_grad():
            output_dict = model(
                x=self.input_tensor,
                x_mask=mask_tensor,
                cond=cond,
                cond_mask=cond_mask,
                multi_encoder_mapping=multi_encoder_mapping,
            )

        # fmt:off
        expected_output = {
            'output': torch.Tensor(
                [[[0.7043, 0.0288, -0.8547, 0.5384],
                  [-0.5283, -1.3311, 1.5429, 0.4083],
                  [-2.0560, 0.5259, 0.6020, -0.4099],
                  [0.4554, -0.3829, 2.1433, -0.2036],
                  [-0.0986, 0.4794, 0.7067, 0.0432],
                  [-0.4392, -0.6772, 1.7428, -0.9393]]]
            ),
            'attn_probabilities': [
                {
                    'self_attn_probabilities': [
                        torch.Tensor(
                            [[[[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                               [0.4989, 0.5011, 0.0000, 0.0000, 0.0000, 0.0000],
                               [0.3331, 0.3332, 0.3336, 0.0000, 0.0000, 0.0000],
                               [0.2495, 0.2496, 0.2504, 0.2505, 0.0000, 0.0000],
                               [0.1998, 0.1994, 0.2002, 0.2001, 0.2005, 0.0000],
                               [0.1662, 0.1662, 0.1668, 0.1668, 0.1671, 0.1669]],
                              [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                               [0.5008, 0.4992, 0.0000, 0.0000, 0.0000, 0.0000],
                               [0.3334, 0.3336, 0.3331, 0.0000, 0.0000, 0.0000],
                               [0.2504, 0.2500, 0.2496, 0.2500, 0.0000, 0.0000],
                               [0.2001, 0.2002, 0.2000, 0.1999, 0.1998, 0.0000],
                               [0.1670, 0.1667, 0.1665, 0.1667, 0.1665, 0.1666]]]]
                        ),
                    ],
                    'cross_attn_probabilities': [
                        torch.Tensor(
                            [[[[0.3331, 0.3336, 0.3334],
                               [0.3335, 0.3331, 0.3334],
                               [0.3336, 0.3331, 0.3332],
                               [0.3335, 0.3332, 0.3334],
                               [0.3336, 0.3331, 0.3333],
                               [0.3335, 0.3332, 0.3333]],
                              [[0.3333, 0.3335, 0.3332],
                               [0.3334, 0.3335, 0.3331],
                               [0.3333, 0.3330, 0.3337],
                               [0.3334, 0.3335, 0.3332],
                               [0.3333, 0.3331, 0.3336],
                               [0.3334, 0.3334, 0.3333]]]]
                        )
                    ]
                },
                {
                    'self_attn_probabilities': [
                        torch.Tensor(
                            [[[[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                               [0.5005, 0.4995, 0.0000, 0.0000, 0.0000, 0.0000],
                               [0.3336, 0.3330, 0.3334, 0.0000, 0.0000, 0.0000],
                               [0.2503, 0.2499, 0.2498, 0.2500, 0.0000, 0.0000],
                               [0.2002, 0.1999, 0.2000, 0.2000, 0.2000, 0.0000],
                               [0.1669, 0.1666, 0.1666, 0.1667, 0.1666, 0.1666]],
                              [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                               [0.5001, 0.4999, 0.0000, 0.0000, 0.0000, 0.0000],
                               [0.3329, 0.3330, 0.3340, 0.0000, 0.0000, 0.0000],
                               [0.2499, 0.2498, 0.2505, 0.2499, 0.0000, 0.0000],
                               [0.1997, 0.1997, 0.2003, 0.2000, 0.2004, 0.0000],
                               [0.1665, 0.1664, 0.1669, 0.1666, 0.1669, 0.1667]]]]
                        ),
                    ],
                    'cross_attn_probabilities': [
                        torch.Tensor(
                            [[[[0.1999, 0.1998, 0.2002, 0.1999, 0.2001],
                               [0.2000, 0.1997, 0.2004, 0.1998, 0.2002],
                               [0.2001, 0.2000, 0.2001, 0.1999, 0.2000],
                               [0.2000, 0.2001, 0.1998, 0.2001, 0.1999],
                               [0.2001, 0.2002, 0.1998, 0.2001, 0.1998],
                               [0.2000, 0.2002, 0.1998, 0.2001, 0.1999]],
                              [[0.1998, 0.1998, 0.2001, 0.2004, 0.2000],
                               [0.2003, 0.2003, 0.1998, 0.1995, 0.2001],
                               [0.2003, 0.2003, 0.1998, 0.1995, 0.2001],
                               [0.2001, 0.2001, 0.2000, 0.1998, 0.2000],
                               [0.2002, 0.2001, 0.1999, 0.1997, 0.2000],
                               [0.2002, 0.2001, 0.2000, 0.1997, 0.2000]]]]
                        ),
                    ],
                }
            ],
        }
        # fmt:on

        assert torch.allclose(output_dict["output"], expected_output["output"], atol=1e-4)
        for i in range(2):
            assert torch.allclose(
                output_dict["attn_probabilities"][i]["self_attn_probabilities"][0],
                expected_output["attn_probabilities"][i]["self_attn_probabilities"][0],
                atol=1e-4,
            )
            assert torch.allclose(
                output_dict["attn_probabilities"][i]["cross_attn_probabilities"][0],
                expected_output["attn_probabilities"][i]["cross_attn_probabilities"][0],
                atol=1e-4,
            )


@pytest.mark.unit
class TestTransformerBatchedInference:
    @classmethod
    def setup_class(cls):
        cls.n_layers = 3
        cls.d_model = 4
        cls.d_ffn = 16
        cls.sa_n_heads = 2
        cls.p_dropout = 0.0
        cls.p_dropout_out = 0.0
        cls.max_length_causal_mask = 10
        cls.short_length = 4
        cls.long_length = 10

    def test_forward(self):
        set_seed(0)
        query_tensor1 = torch.randn(1, self.long_length, self.d_model)
        query_tensor2 = torch.randn(1, self.short_length, self.d_model)

        padding_tensor = torch.randn(1, self.long_length - self.short_length, self.d_model)
        query_tensor2_padded = torch.cat([query_tensor2, padding_tensor], dim=1)
        lengths = torch.tensor([self.long_length, self.short_length])
        mask_batched = get_mask_from_lengths(lengths)

        query_batched = torch.cat([query_tensor1, query_tensor2_padded], dim=0)

        mask_bs1_1 = torch.ones(1, self.long_length)
        mask_bs1_2 = torch.ones(1, self.short_length)

        for is_causal in [True, False]:
            for kernel_size in [1, 3]:
                model = Transformer(
                    n_layers=self.n_layers,
                    d_model=self.d_model,
                    d_ffn=self.d_ffn,
                    sa_n_heads=self.sa_n_heads,
                    kernel_size=kernel_size,
                    p_dropout=self.p_dropout,
                    p_dropout_out=self.p_dropout_out,
                    is_causal=is_causal,
                    max_length_causal_mask=self.max_length_causal_mask,
                )

                output_batched = model(query_batched, mask_batched)
                output_bs1_1 = model(query_tensor1, mask_bs1_1)
                output_bs1_2 = model(query_tensor2, mask_bs1_2)

                assert torch.allclose(
                    output_batched['output'][0][: self.long_length, :], output_bs1_1['output'], atol=1e-4
                )
                assert torch.allclose(
                    output_batched['output'][1][: self.short_length, :], output_bs1_2['output'], atol=1e-4
                )


@pytest.mark.unit
class TestMoERouter:
    """Test the MoERouter class for expert selection and auxiliary losses."""

    @classmethod
    def setup_class(cls):
        cls.d_model = 8
        cls.num_experts = 4
        cls.top_k = 2
        cls.batch_size = 2
        cls.seq_len = 5

    def test_router_initialization(self):
        """Test that router initializes correctly with valid parameters."""
        router = MoERouter(
            d_model=self.d_model,
            num_experts=self.num_experts,
            top_k=self.top_k,
        )

        assert router.d_model == self.d_model
        assert router.num_experts == self.num_experts
        assert router.top_k == self.top_k
        assert router.router.in_features == self.d_model
        assert router.router.out_features == self.num_experts

    def test_router_forward_shape(self):
        """Test that router output shapes are correct."""
        set_seed(42)
        router = MoERouter(
            d_model=self.d_model,
            num_experts=self.num_experts,
            top_k=self.top_k,
        )

        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        x_mask = torch.ones(self.batch_size, self.seq_len)

        with torch.no_grad():
            expert_weights, expert_indices, router_logits, router_probs = router(x, x_mask)

        # Check shapes
        assert expert_weights.shape == (self.batch_size, self.seq_len, self.top_k)
        assert expert_indices.shape == (self.batch_size, self.seq_len, self.top_k)
        assert router_logits.shape == (self.batch_size, self.seq_len, self.num_experts)
        assert router_probs.shape == (self.batch_size, self.seq_len, self.num_experts)

        # Check that weights sum to 1
        weight_sums = expert_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

        # Check that expert indices are valid (all valid tokens)
        assert expert_indices.min() >= 0
        assert expert_indices.max() < self.num_experts

    def test_router_returns_logits_and_probs(self):
        """Test that router returns logits and probs for loss computation."""
        set_seed(42)
        router = MoERouter(
            d_model=self.d_model,
            num_experts=self.num_experts,
            top_k=self.top_k,
        )
        router.train()

        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        x_mask = torch.ones(self.batch_size, self.seq_len)

        expert_weights, expert_indices, router_logits, router_probs = router(x, x_mask)

        # Check that logits and probs are returned for loss computation
        assert router_logits.shape == (self.batch_size, self.seq_len, self.num_experts)
        assert router_probs.shape == (self.batch_size, self.seq_len, self.num_experts)

        # Probs should sum to 1 for valid tokens
        assert torch.allclose(router_probs.sum(dim=-1), x_mask, atol=1e-5)

    def test_router_top_k_selection(self):
        """Test that exactly top_k experts are selected."""
        set_seed(42)
        for top_k in [1, 2, 3]:
            router = MoERouter(
                d_model=self.d_model,
                num_experts=self.num_experts,
                top_k=top_k,
            )

            x = torch.randn(self.batch_size, self.seq_len, self.d_model)
            x_mask = torch.ones(self.batch_size, self.seq_len)

            with torch.no_grad():
                expert_weights, expert_indices, _, _ = router(x, x_mask)

            assert expert_weights.shape[-1] == top_k
            assert expert_indices.shape[-1] == top_k

    def test_router_sinkhorn_strategy(self):
        """Test that Sinkhorn routing works."""
        set_seed(42)
        router = MoERouter(
            d_model=self.d_model,
            num_experts=self.num_experts,
            top_k=self.top_k,
            routing_strategy="sinkhorn",
        )

        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        x_mask = torch.ones(self.batch_size, self.seq_len)

        with torch.no_grad():
            expert_weights, expert_indices, _, _ = router(x, x_mask)

        # Just check it runs and produces valid outputs
        assert expert_weights.shape == (self.batch_size, self.seq_len, self.top_k)
        assert torch.allclose(expert_weights.sum(dim=-1), torch.ones(self.batch_size, self.seq_len), atol=1e-5)

    def test_router_jitter_noise(self):
        """Test that jitter noise affects routing during training."""
        set_seed(42)
        router_with_noise = MoERouter(
            d_model=self.d_model,
            num_experts=self.num_experts,
            top_k=self.top_k,
            router_jitter_noise=0.1,
        )
        router_with_noise.train()

        router_no_noise = MoERouter(
            d_model=self.d_model,
            num_experts=self.num_experts,
            top_k=self.top_k,
            router_jitter_noise=0.0,
        )
        router_no_noise.train()

        # Copy weights to make them identical
        router_no_noise.router.weight.data = router_with_noise.router.weight.data.clone()

        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        x_mask = torch.ones(self.batch_size, self.seq_len)

        # Get outputs with different random seeds to ensure noise causes differences
        set_seed(100)
        _, indices_with_noise, logits_with, probs_with = router_with_noise(x, x_mask)
        set_seed(101)  # Different seed
        _, indices_no_noise, logits_no, probs_no = router_no_noise(x, x_mask)

        # Shapes should match
        assert indices_with_noise.shape == indices_no_noise.shape

        # With noise, routing decisions or probabilities should differ
        # (Noise is added to logits, which affects probs and potentially indices)
        # At least one should be different due to noise
        indices_differ = not torch.all(indices_with_noise == indices_no_noise)
        probs_differ = not torch.allclose(probs_with, probs_no, atol=1e-5)

        assert indices_differ or probs_differ, "Jitter noise should cause different routing decisions or probabilities"

    def test_router_padding_masking(self):
        """Test that padded positions are properly masked with expert_indices=-1."""
        set_seed(42)
        router = MoERouter(
            d_model=self.d_model,
            num_experts=self.num_experts,
            top_k=self.top_k,
        )

        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        x_mask = torch.ones(self.batch_size, self.seq_len)

        # Mask out last 2 positions of first batch
        x_mask[0, 3:] = 0

        with torch.no_grad():
            expert_weights, expert_indices, router_logits, router_probs = router(x, x_mask)

        # Check that padded positions have expert_indices = -1
        assert torch.all(expert_indices[0, 3:, :] == -1), "Padded positions should have expert_indices=-1"

        # Check that padded positions have zero weights
        assert torch.allclose(expert_weights[0, 3:, :], torch.zeros_like(expert_weights[0, 3:, :]))

        # Check that valid positions have valid expert indices (0 to num_experts-1)
        assert torch.all(expert_indices[0, :3, :] >= 0), "Valid positions should have expert_indices >= 0"
        assert torch.all(
            expert_indices[0, :3, :] < self.num_experts
        ), "Valid positions should have valid expert indices"

        # Check that router_logits are zero at padded positions
        assert torch.allclose(router_logits[0, 3:, :], torch.zeros_like(router_logits[0, 3:, :]))

        # Check that router_probs are zero at padded positions
        assert torch.allclose(router_probs[0, 3:, :], torch.zeros_like(router_probs[0, 3:, :]))


@pytest.mark.unit
class TestPositionwiseConvFFMoE:
    """Test the PositionwiseConvFFMoE class."""

    @classmethod
    def setup_class(cls):
        cls.d_model = 8
        cls.d_ffn = 32
        cls.p_dropout = 0.0
        cls.num_experts = 4
        cls.top_k_experts = 2
        cls.kernel_size = 1  # MoE requires kernel_size=1
        cls.batch_size = 2
        cls.seq_len = 10

    def test_moe_ffn_initialization(self):
        """Test that MoE FFN initializes correctly."""
        set_seed(42)
        moe_ffn = PositionwiseConvFFMoE(
            d_model=self.d_model,
            d_ffn=self.d_ffn,
            p_dropout=self.p_dropout,
            num_experts=self.num_experts,
            top_k_experts=self.top_k_experts,
            kernel_size=self.kernel_size,
        )

        assert moe_ffn.d_model == self.d_model
        assert moe_ffn.d_ffn == self.d_ffn
        assert moe_ffn.num_experts == self.num_experts
        assert moe_ffn.top_k_experts == self.top_k_experts
        assert len(moe_ffn.experts) == self.num_experts

    def test_moe_ffn_forward_shape(self):
        """Test that MoE FFN produces correct output shape."""
        set_seed(42)
        moe_ffn = PositionwiseConvFFMoE(
            d_model=self.d_model,
            d_ffn=self.d_ffn,
            p_dropout=self.p_dropout,
            num_experts=self.num_experts,
            top_k_experts=self.top_k_experts,
            kernel_size=self.kernel_size,
            is_causal=True,
        )

        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        x_mask = torch.ones(self.batch_size, self.seq_len)

        with torch.no_grad():
            output, router_logits, router_probs, expert_indices = moe_ffn(x, x_mask)

        # Check output shape
        assert output.shape == x.shape

        # Check routing info is returned
        assert router_logits.shape == (self.batch_size, self.seq_len, self.num_experts)
        assert router_probs.shape == (self.batch_size, self.seq_len, self.num_experts)
        assert expert_indices.shape == (self.batch_size, self.seq_len, self.top_k_experts)

    def test_moe_ffn_masking(self):
        """Test that masking works correctly and padded outputs are zero."""
        set_seed(42)
        moe_ffn = PositionwiseConvFFMoE(
            d_model=self.d_model,
            d_ffn=self.d_ffn,
            p_dropout=self.p_dropout,
            num_experts=self.num_experts,
            top_k_experts=self.top_k_experts,
            kernel_size=1,
            is_causal=False,
        )

        x = torch.randn(self.batch_size, self.seq_len, self.d_model)

        # Create a mask that zeros out some positions
        x_mask = torch.ones(self.batch_size, self.seq_len)
        x_mask[0, 5:] = 0  # Mask out last 5 positions of first sample

        with torch.no_grad():
            output, router_logits, router_probs, expert_indices = moe_ffn(x, x_mask)

        # Check that output shape is correct
        assert output.shape == x.shape

        # Check that padded positions in output are zero (not processed by any expert)
        assert torch.allclose(output[0, 5:, :], torch.zeros_like(output[0, 5:, :]))

        # Check that router_logits and router_probs are zero at padded positions
        assert torch.allclose(router_logits[0, 5:, :], torch.zeros_like(router_logits[0, 5:, :]))
        assert torch.allclose(router_probs[0, 5:, :], torch.zeros_like(router_probs[0, 5:, :]))

        # Check that expert_indices are -1 at padded positions
        assert torch.all(expert_indices[0, 5:, :] == -1)

    def test_moe_ffn_different_expert_counts(self):
        """Test that different numbers of experts work."""
        set_seed(42)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        x_mask = torch.ones(self.batch_size, self.seq_len)

        for num_experts in [2, 4, 8]:
            for top_k in [1, 2]:
                if top_k <= num_experts:
                    moe_ffn = PositionwiseConvFFMoE(
                        d_model=self.d_model,
                        d_ffn=self.d_ffn,
                        p_dropout=self.p_dropout,
                        num_experts=num_experts,
                        top_k_experts=top_k,
                        kernel_size=1,
                    )

                    with torch.no_grad():
                        output, router_logits, router_probs, expert_indices = moe_ffn(x, x_mask)

                    assert output.shape == x.shape


@pytest.mark.unit
class TestTransformerLayerWithMoE:
    """Test TransformerLayer with MoE enabled."""

    @classmethod
    def setup_class(cls):
        cls.d_model = 8
        cls.d_ffn = 32
        cls.sa_n_heads = 2
        cls.kernel_size = 1  # MoE requires kernel_size=1
        cls.p_dropout = 0.0
        cls.max_length_causal_mask = 20
        cls.batch_size = 2
        cls.seq_len = 10

    def test_transformer_layer_moe_initialization(self):
        """Test that TransformerLayer initializes with MoE correctly."""
        set_seed(42)
        layer = TransformerLayer(
            d_model=self.d_model,
            d_ffn=self.d_ffn,
            sa_n_heads=self.sa_n_heads,
            kernel_size=self.kernel_size,
            p_dropout=self.p_dropout,
            has_xattn=False,
            is_causal=True,
            max_length_causal_mask=self.max_length_causal_mask,
            use_moe=True,
            num_experts=4,
            top_k_experts=2,
        )

        assert layer.use_moe is True
        assert isinstance(layer.pos_ff, PositionwiseConvFFMoE)

    def test_transformer_layer_without_moe(self):
        """Test that TransformerLayer without MoE still works."""
        set_seed(42)
        layer = TransformerLayer(
            d_model=self.d_model,
            d_ffn=self.d_ffn,
            sa_n_heads=self.sa_n_heads,
            kernel_size=self.kernel_size,
            p_dropout=self.p_dropout,
            has_xattn=False,
            is_causal=True,
            max_length_causal_mask=self.max_length_causal_mask,
            use_moe=False,
        )

        assert layer.use_moe is False
        assert isinstance(layer.pos_ff, PositionwiseConvFF)

    def test_transformer_layer_moe_forward(self):
        """Test forward pass of TransformerLayer with MoE."""
        set_seed(42)
        layer = TransformerLayer(
            d_model=self.d_model,
            d_ffn=self.d_ffn,
            sa_n_heads=self.sa_n_heads,
            kernel_size=self.kernel_size,
            p_dropout=self.p_dropout,
            has_xattn=False,
            is_causal=True,
            max_length_causal_mask=self.max_length_causal_mask,
            use_moe=True,
            num_experts=4,
            top_k_experts=2,
        )

        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        x_mask = torch.ones(self.batch_size, self.seq_len).bool()

        with torch.no_grad():
            output_dict = layer(x, x_mask)

        # Check output
        assert 'output' in output_dict
        assert output_dict['output'].shape == x.shape

        # Check MoE routing info is present
        assert 'moe_routing_info' in output_dict
        if output_dict['moe_routing_info'] is not None:
            assert 'router_logits' in output_dict['moe_routing_info']
            assert 'router_probs' in output_dict['moe_routing_info']
            assert 'expert_indices' in output_dict['moe_routing_info']

    def test_transformer_layer_moe_with_xattn(self):
        """Test TransformerLayer with both MoE and cross-attention."""
        set_seed(42)
        layer = TransformerLayer(
            d_model=self.d_model,
            d_ffn=self.d_ffn,
            sa_n_heads=self.sa_n_heads,
            kernel_size=self.kernel_size,
            p_dropout=self.p_dropout,
            has_xattn=True,
            xa_d_memory=self.d_model,
            xa_n_heads=2,
            is_causal=True,
            max_length_causal_mask=self.max_length_causal_mask,
            use_moe=True,
            num_experts=4,
            top_k_experts=2,
        )

        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        x_mask = torch.ones(self.batch_size, self.seq_len).bool()
        cond = torch.randn(self.batch_size, 5, self.d_model)
        cond_mask = torch.ones(self.batch_size, 5).bool()

        with torch.no_grad():
            output_dict = layer(x, x_mask, cond, cond_mask)

        assert output_dict['output'].shape == x.shape
        assert 'moe_routing_info' in output_dict


@pytest.mark.unit
class TestTransformerWithMoE:
    """Test full Transformer with MoE."""

    @classmethod
    def setup_class(cls):
        cls.n_layers = 2
        cls.d_model = 8
        cls.d_ffn = 32
        cls.sa_n_heads = 2
        cls.kernel_size = 1  # MoE requires kernel_size=1
        cls.p_dropout = 0.0
        cls.batch_size = 2
        cls.seq_len = 10
        cls.max_length_causal_mask = 20

    def test_transformer_moe_initialization(self):
        """Test that Transformer initializes with MoE correctly."""
        set_seed(42)
        model = Transformer(
            n_layers=self.n_layers,
            d_model=self.d_model,
            d_ffn=self.d_ffn,
            sa_n_heads=self.sa_n_heads,
            kernel_size=self.kernel_size,
            p_dropout=self.p_dropout,
            has_xattn=False,
            is_causal=True,
            max_length_causal_mask=self.max_length_causal_mask,
            use_moe=True,
            num_experts=4,
            top_k_experts=2,
        )

        assert model.use_moe is True
        assert len(model.layers) == self.n_layers
        for layer in model.layers:
            assert isinstance(layer.pos_ff, PositionwiseConvFFMoE)

    def test_transformer_moe_forward(self):
        """Test forward pass of Transformer with MoE."""
        set_seed(42)
        model = Transformer(
            n_layers=self.n_layers,
            d_model=self.d_model,
            d_ffn=self.d_ffn,
            sa_n_heads=self.sa_n_heads,
            kernel_size=self.kernel_size,
            p_dropout=self.p_dropout,
            has_xattn=False,
            is_causal=True,
            max_length_causal_mask=self.max_length_causal_mask,
            use_moe=True,
            num_experts=4,
            top_k_experts=2,
        )

        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        x_mask = torch.ones(self.batch_size, self.seq_len).bool()

        with torch.no_grad():
            output_dict = model(x, x_mask)

        # Check output
        assert 'output' in output_dict
        assert output_dict['output'].shape == x.shape

        # Check MoE routing info is collected from all layers
        assert 'moe_routing_info' in output_dict
        assert output_dict['moe_routing_info'] is not None
        assert len(output_dict['moe_routing_info']) == self.n_layers

        # Each layer should have logits, probs, and indices
        for layer_routing in output_dict['moe_routing_info']:
            assert 'router_logits' in layer_routing
            assert 'router_probs' in layer_routing
            assert 'expert_indices' in layer_routing

    def test_transformer_moe_with_cross_attention(self):
        """Test Transformer with both MoE and cross-attention."""
        set_seed(42)
        model = Transformer(
            n_layers=self.n_layers,
            d_model=self.d_model,
            d_ffn=self.d_ffn,
            sa_n_heads=self.sa_n_heads,
            kernel_size=self.kernel_size,
            p_dropout=self.p_dropout,
            has_xattn=True,
            xa_d_memory=self.d_model,
            xa_n_heads=2,
            is_causal=True,
            max_length_causal_mask=self.max_length_causal_mask,
            use_moe=True,
            num_experts=4,
            top_k_experts=2,
        )

        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        x_mask = torch.ones(self.batch_size, self.seq_len).bool()
        cond = torch.randn(self.batch_size, 5, self.d_model)
        cond_mask = torch.ones(self.batch_size, 5).bool()

        with torch.no_grad():
            output_dict = model(x, x_mask, cond, cond_mask)

        assert output_dict['output'].shape == x.shape
        assert 'moe_routing_info' in output_dict

    def test_transformer_moe_different_configurations(self):
        """Test various MoE configurations."""
        set_seed(42)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        x_mask = torch.ones(self.batch_size, self.seq_len).bool()

        configs = [
            {'num_experts': 2, 'top_k_experts': 1},
            {'num_experts': 4, 'top_k_experts': 2},
            {'num_experts': 8, 'top_k_experts': 2},
            {'num_experts': 4, 'top_k_experts': 1, 'routing_strategy': 'sinkhorn'},
        ]

        for config in configs:
            model = Transformer(
                n_layers=self.n_layers,
                d_model=self.d_model,
                d_ffn=self.d_ffn,
                sa_n_heads=self.sa_n_heads,
                kernel_size=self.kernel_size,
                p_dropout=self.p_dropout,
                has_xattn=False,
                is_causal=True,
                max_length_causal_mask=self.max_length_causal_mask,
                use_moe=True,
                **config,
            )

            with torch.no_grad():
                output_dict = model(x, x_mask)

            assert output_dict['output'].shape == x.shape
