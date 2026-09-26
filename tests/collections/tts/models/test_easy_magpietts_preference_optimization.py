# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.tts.models.easy_magpietts_preference_optimization import EasyMagpieTTSModelOnlinePO


pytestmark = pytest.mark.unit


def _make_loss_only_model():
    model = EasyMagpieTTSModelOnlinePO.__new__(EasyMagpieTTSModelOnlinePO)
    torch.nn.Module.__init__(model)
    model._cfg = OmegaConf.create({"grpo_beta": 0.01})
    model.reference_free = False
    model.loss_type = "grpo"
    model.max_decoder_steps = 10
    return model


def test_action_po_uses_exact_forward_kl():
    model = _make_loss_only_model()
    policy_probs = torch.tensor([0.75, 0.25])
    reference_probs = torch.tensor([0.5, 0.5])

    po_loss, kl_loss, _ = model._compute_action_po_components(
        logits=policy_probs.log().view(1, 1, 2),
        reference_logits=reference_probs.log().view(1, 1, 2),
        targets=torch.tensor([[[0]]]),
        target_lens=torch.tensor([1]),
        vocab_size=2,
        advantages=torch.tensor([0.0]),
        group_validities=torch.tensor([1.0]),
        sampling_temperature=1.0,
    )

    expected_kl = (policy_probs * (policy_probs.log() - reference_probs.log())).sum()
    assert po_loss.item() == pytest.approx(0.0)
    assert kl_loss.item() == pytest.approx(expected_kl.item())


def test_action_po_masks_kl_for_invalid_groups():
    model = _make_loss_only_model()

    po_loss, kl_loss, _ = model._compute_action_po_components(
        logits=torch.tensor([[[4.0, -4.0]]]),
        reference_logits=torch.tensor([[[-4.0, 4.0]]]),
        targets=torch.tensor([[[0]]]),
        target_lens=torch.tensor([1]),
        vocab_size=2,
        advantages=torch.tensor([3.0]),
        group_validities=torch.tensor([0.0]),
        sampling_temperature=1.0,
    )

    assert po_loss.item() == pytest.approx(0.0)
    assert kl_loss.item() == pytest.approx(0.0)
