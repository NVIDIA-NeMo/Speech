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

import math

import pytest
import torch

from nemo.collections.asr.losses import BCELoss, BCEWithLogitsLoss


def _valid_frames(tensor, lengths):
    return torch.cat([tensor[index, : lengths[index]] for index in range(tensor.shape[0])], dim=0)


@pytest.mark.unit
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_bce_loss_honors_reduction_after_masking(reduction):
    probs = torch.tensor(
        [
            [[0.2, 0.7], [0.8, 0.4], [0.9, 0.9]],
            [[0.3, 0.6], [0.1, 0.2], [0.8, 0.5]],
        ]
    )
    labels = torch.tensor(
        [
            [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        ]
    )
    lengths = torch.tensor([2, 1])
    expected = torch.nn.BCELoss(reduction=reduction)(
        _valid_frames(probs, lengths).float(),
        _valid_frames(labels, lengths).float(),
    )

    actual = BCELoss(reduction=reduction)(probs=probs, labels=labels, target_lens=lengths)

    torch.testing.assert_close(actual, expected)


@pytest.mark.unit
@pytest.mark.parametrize("shape, valid_lengths", [((3, 5, 4), (5, 3, 1))])
def test_bce_loss_default_matches_unweighted_fp32_loss(shape, valid_lengths):
    probs = torch.rand(shape)
    labels = torch.randint(0, 2, probs.shape).float()
    lengths = torch.tensor(valid_lengths)
    expected = torch.nn.BCELoss()(
        _valid_frames(probs, lengths).float(),
        _valid_frames(labels, lengths).float(),
    )

    actual = BCELoss()(probs=probs, labels=labels, target_lens=lengths)

    torch.testing.assert_close(actual, expected)


@pytest.mark.unit
@pytest.mark.parametrize("weight, reduction", [(None, "mean")])
def test_bce_loss_accepts_explicit_none_weight(weight, reduction):
    loss = BCELoss(weight=weight, reduction=reduction)

    assert loss.loss_f.weight is None
    assert loss.loss_f.reduction == reduction


@pytest.mark.unit
@pytest.mark.parametrize("shape, valid_lengths", [((2, 4, 3), (4, 2))])
def test_bce_with_logits_loss_excludes_padded_frames(shape, valid_lengths):
    logits = torch.randn(shape)
    labels = torch.randint(0, 2, logits.shape).float()
    lengths = torch.tensor(valid_lengths)
    expected = torch.nn.BCEWithLogitsLoss()(
        _valid_frames(logits, lengths).float(),
        _valid_frames(labels, lengths).float(),
    )

    loss = BCEWithLogitsLoss()
    actual = loss(logits=logits, labels=labels, target_lens=lengths)

    torch.testing.assert_close(actual, expected)
    assert loss.loss_f.pos_weight is None
    assert "loss_f.pos_weight" not in loss.state_dict()


@pytest.mark.unit
@pytest.mark.parametrize(
    "minimum, maximum, shape, valid_lengths",
    [(-3.0, 3.0, (2, 5, 3), (5, 3))],
)
def test_bce_with_logits_matches_probability_bce_for_ordinary_logits(minimum, maximum, shape, valid_lengths):
    logits = torch.linspace(minimum, maximum, steps=math.prod(shape)).reshape(shape)
    labels = torch.randint(0, 2, logits.shape).float()
    lengths = torch.tensor(valid_lengths)

    logits_loss = BCEWithLogitsLoss()(logits=logits, labels=labels, target_lens=lengths)
    probability_loss = BCELoss()(probs=torch.sigmoid(logits), labels=labels, target_lens=lengths)

    torch.testing.assert_close(logits_loss, probability_loss)


@pytest.mark.unit
@pytest.mark.parametrize("logit_value, expected_loss, absolute_tolerance", [(7.0, 7.0, 0.02)])
def test_bce_with_logits_preserves_bfloat16_saturated_gradient(logit_value, expected_loss, absolute_tolerance):
    logits = torch.tensor([[[logit_value]]], dtype=torch.bfloat16, requires_grad=True)
    labels = torch.zeros_like(logits)

    loss = BCEWithLogitsLoss()(logits=logits, labels=labels, target_lens=torch.tensor([1]))
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(expected_loss, abs=absolute_tolerance)
    assert logits.grad.item() > 0
