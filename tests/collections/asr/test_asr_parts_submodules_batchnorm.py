# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from nemo.collections.asr.parts.submodules.batchnorm import (
    FusedBatchNorm1d,
    replace_bn_with_fused_bn,
    replace_bn_with_fused_bn_all,
)


class TestFusedBatchNorm1d:
    @pytest.mark.unit
    def test_constructor(self):
        num_features = 10
        fused_bn = FusedBatchNorm1d(num_features=num_features)
        assert fused_bn.weight.shape[0] == num_features
        assert fused_bn.bias.shape[0] == num_features
        # check initialization: weight is ones, bias is zeros (identity)
        assert torch.allclose(fused_bn.weight, torch.ones(num_features))
        assert torch.allclose(fused_bn.bias, torch.zeros(num_features))

    @pytest.mark.unit
    def test_from_batchnorm(self):
        num_features = 10

        # construct batchnorm
        bn = nn.BatchNorm1d(num_features=num_features)

        # update bn stats
        bn.train()
        batch_size = 4
        for _ in range(10):
            _ = bn(torch.rand(batch_size, num_features))

        # test eval mode is equivalent
        fused_bn = FusedBatchNorm1d.from_batchnorm(bn)
        bn.eval()

        sample_2d = torch.rand(batch_size, num_features)
        assert torch.allclose(bn(sample_2d), fused_bn(sample_2d))

        sample_3d = torch.rand(batch_size, num_features, 5)
        assert torch.allclose(bn(sample_3d), fused_bn(sample_3d))


class TestReplaceBNWithFusedBN:
    @pytest.mark.unit
    def test_replace_bn_with_fused_bn(self):
        model = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(1, 10)),
                    ("bn1", nn.BatchNorm1d(10)),
                    ("relu1", nn.ReLU()),
                    ("linear2", nn.Linear(10, 11)),
                    ("bn2", nn.BatchNorm1d(11)),
                    (
                        "submodule1",
                        nn.Sequential(OrderedDict([("linear3", nn.Linear(11, 12)), ("bn3", nn.BatchNorm1d(12))])),
                    ),
                ]
            )
        )
        replace_bn_with_fused_bn(model, "submodule1.bn3")
        assert isinstance(model.bn1, nn.BatchNorm1d)
        assert isinstance(model.bn2, nn.BatchNorm1d)
        assert isinstance(model.submodule1.bn3, FusedBatchNorm1d)

    @pytest.mark.unit
    def test_replace_bn_with_fused_bn_all(self):
        model = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(1, 10)),
                    ("bn1", nn.BatchNorm1d(10)),
                    ("relu1", nn.ReLU()),
                    ("linear2", nn.Linear(10, 11)),
                    ("bn2", nn.BatchNorm1d(11)),
                    (
                        "submodule1",
                        nn.Sequential(OrderedDict([("linear3", nn.Linear(11, 12)), ("bn3", nn.BatchNorm1d(12))])),
                    ),
                ]
            )
        )
        replace_bn_with_fused_bn_all(model)
        assert isinstance(model.bn1, FusedBatchNorm1d)
        assert isinstance(model.bn2, FusedBatchNorm1d)
        assert isinstance(model.submodule1.bn3, FusedBatchNorm1d)
