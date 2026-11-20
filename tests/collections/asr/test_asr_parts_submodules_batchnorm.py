# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from nemo.collections.asr.parts.submodules.batchnorm import (
    FusedBatchNorm1d,
    MaskedBatchNorm1d,
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


class TestMaskedBatchNorm1d:
    @pytest.mark.unit
    def test_matches_batchnorm1d_without_lengths(self):
        torch.manual_seed(0)
        reference = nn.BatchNorm1d(4, eps=1e-3, momentum=0.2)
        masked = MaskedBatchNorm1d(4, eps=1e-3, momentum=0.2)
        with torch.no_grad():
            reference.weight.uniform_(0.5, 1.5)
            reference.bias.uniform_(-1, 1)
            masked.load_state_dict(reference.state_dict())
        for _ in range(3):
            x = torch.randn(2, 4, 6) * 3 + 1
            torch.testing.assert_close(masked(x), reference(x), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(masked.running_mean, reference.running_mean)
        torch.testing.assert_close(masked.running_var, reference.running_var)
        assert masked.num_batches_tracked == reference.num_batches_tracked == 3
        masked.eval()
        reference.eval()
        x = torch.randn(2, 4, 6)
        torch.testing.assert_close(masked(x), reference(x), atol=1e-5, rtol=1e-5)
        rows = torch.randn(5, 4)
        torch.testing.assert_close(masked(rows), reference(rows), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(masked.forward_packed(rows), reference(rows), atol=1e-5, rtol=1e-5)

    @pytest.mark.unit
    def test_statistics_over_valid_frames(self):
        torch.manual_seed(1)
        module = MaskedBatchNorm1d(4, eps=1e-3, momentum=0.5, affine=False, unbiased_running_var=False).train()
        x = torch.randn(2, 4, 6) * 3 + 1
        lengths = torch.tensor([6, 3])
        x[1, :, 3:] = 1e6  # padding must not enter the statistics
        out = module(x, lengths)
        valid = torch.cat([x[0], x[1, :, :3]], dim=1)
        mean, var = valid.mean(1), valid.var(1, unbiased=False)
        torch.testing.assert_close(out[0], (x[0] - mean[:, None]) / torch.sqrt(var[:, None] + 1e-3))
        torch.testing.assert_close(module.running_mean, 0.5 * mean)
        torch.testing.assert_close(module.running_var, 0.5 + 0.5 * var)
        assert module.num_batches_tracked == 1
        # evaluation uses the running statistics, and the packed entry point agrees with the padded one
        module.eval()
        evaluated = module(x, lengths)
        expected = (x - module.running_mean[None, :, None]) * torch.rsqrt(module.running_var[None, :, None] + 1e-3)
        torch.testing.assert_close(evaluated, expected)
        torch.testing.assert_close(module.forward_packed(valid.T), torch.cat([evaluated[0], evaluated[1, :, :3]], 1).T)
        # a batch without valid frames leaves the running statistics unchanged
        module.train()
        module(x, torch.tensor([0, 0]))
        torch.testing.assert_close(module.running_mean, 0.5 * mean)
        torch.testing.assert_close(module.running_var, 0.5 + 0.5 * var)
        assert module.num_batches_tracked == 1
        assert set(module.state_dict()) == {"running_mean", "running_var", "num_batches_tracked"}
        with pytest.raises(ValueError, match="needs"):
            module(valid.T, lengths)

    @pytest.mark.unit
    def test_unbiased_running_var_option(self):
        torch.manual_seed(2)
        x = torch.randn(2, 4, 6)
        lengths = torch.tensor([6, 3])
        valid = torch.cat([x[0], x[1, :, :3]], dim=1)
        biased = MaskedBatchNorm1d(4, momentum=1.0, unbiased_running_var=False).train()
        unbiased = MaskedBatchNorm1d(4, momentum=1.0).train()
        biased(x, lengths)
        unbiased(x, lengths)
        torch.testing.assert_close(biased.running_var, valid.var(1, unbiased=False))
        torch.testing.assert_close(unbiased.running_var, valid.var(1, unbiased=True))
        torch.testing.assert_close(biased.running_mean, unbiased.running_mean)

    @pytest.mark.unit
    def test_input_precision(self):
        # two float64 values that are equal in float32: the statistics must keep the input's precision
        rows = torch.tensor([[1e8], [1e8 + 1]], dtype=torch.float64)
        reference = nn.BatchNorm1d(1).double()
        masked = MaskedBatchNorm1d(1).double()
        out = masked(rows)
        torch.testing.assert_close(out, reference(rows))
        torch.testing.assert_close(out, torch.tensor([[-1.0], [1.0]], dtype=torch.float64), atol=1e-4, rtol=0)
        torch.testing.assert_close(masked.running_mean, reference.running_mean)
        torch.testing.assert_close(masked.running_var, reference.running_var)
        assert masked.running_var.item() == pytest.approx(0.95)
        # the same two values selected by lengths from (B, D, T) input, and as packed rows
        padded = torch.tensor([[[1e8, 7e8]], [[1e8 + 1, 9e8]]], dtype=torch.float64)
        from_padded = MaskedBatchNorm1d(1).double()
        torch.testing.assert_close(from_padded(padded, torch.tensor([1, 1]))[:, :, 0], out)
        torch.testing.assert_close(from_padded.running_var, reference.running_var)
        from_packed = MaskedBatchNorm1d(1).double()
        torch.testing.assert_close(from_packed.forward_packed(rows), out)
        torch.testing.assert_close(from_packed.running_var, reference.running_var)
        # half and bfloat16 inputs accumulate in float32 and come back in their dtype
        torch.manual_seed(4)
        x = torch.randn(2, 4, 6) * 3 + 1
        for dtype in (torch.float16, torch.bfloat16):
            reference = nn.BatchNorm1d(4)
            masked = MaskedBatchNorm1d(4)
            out = masked(x.to(dtype))
            assert out.dtype == dtype
            torch.testing.assert_close(
                out.float(), reference(x.to(dtype).float()).to(dtype).float(), atol=1e-2, rtol=1e-2
            )
            torch.testing.assert_close(masked.running_mean, reference.running_mean, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(masked.running_var, reference.running_var, atol=1e-2, rtol=1e-2)

    @pytest.mark.unit
    def test_momentum_none_and_no_running_stats(self):
        torch.manual_seed(3)
        cumulative = MaskedBatchNorm1d(4, momentum=None).train()
        first, second = torch.randn(2, 4, 5), torch.randn(2, 4, 5) + 2
        lengths = torch.tensor([5, 2])
        cumulative(first, lengths)
        cumulative(second, lengths)
        means = [torch.cat([b[0], b[1, :, :2]], dim=1).mean(1) for b in (first, second)]
        torch.testing.assert_close(cumulative.running_mean, (means[0] + means[1]) / 2)
        assert cumulative.num_batches_tracked == 2
        stateless = MaskedBatchNorm1d(4, track_running_stats=False, affine=False).eval()
        assert set(stateless.state_dict()) == set()
        out = stateless(first, lengths)
        valid = torch.cat([first[0], first[1, :, :2]], dim=1)
        expected = (first - valid.mean(1)[None, :, None]) * torch.rsqrt(
            valid.var(1, unbiased=False)[None, :, None] + 1e-5
        )
        torch.testing.assert_close(out, expected)


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
