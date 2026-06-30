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

import pytest
import torch

from nemo.collections.common.losses.latent_speaker_supervision_loss import LatentSpeakerSupervisionLoss

# Vocabulary layout used across tests: tokens 5/6/7 are the three speaker tokens.
SPK_IDS = [5, 6, 7]
VOCAB = 10


def _make_log_probs(labels: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Build a valid (B, T, V) log-prob tensor for the given labels."""
    torch.manual_seed(seed)
    logits = torch.randn(labels.shape[0], labels.shape[1], VOCAB)
    return torch.log_softmax(logits, dim=-1)


@pytest.mark.unit
@pytest.mark.parametrize(
    "labels_row, expected_active, expected_speaker_mask, expected_word_mask",
    [
        # token 0 precedes the first speaker tag -> active -1, not a word
        (
            [0, 5, 1, 2, 6, 3],
            [-1, 0, 0, 0, 1, 1],
            [False, True, False, False, True, False],
            [False, False, True, True, False, True],
        ),
        # no speaker tokens at all -> everything inactive, no words
        (
            [1, 2, 3, 4],
            [-1, -1, -1, -1],
            [False, False, False, False],
            [False, False, False, False],
        ),
        # back-to-back speaker tags then words for speaker 2
        (
            [5, 6, 7, 1, 2],
            [0, 1, 2, 2, 2],
            [True, True, True, False, False],
            [False, False, False, True, True],
        ),
    ],
)
def test_get_active_speaker_per_position(labels_row, expected_active, expected_speaker_mask, expected_word_mask):
    loss = LatentSpeakerSupervisionLoss(speaker_token_ids=SPK_IDS)
    labels = torch.tensor([labels_row], dtype=torch.long)

    active, speaker_mask, word_mask = loss._get_active_speaker_per_position(labels)

    assert active.tolist() == [expected_active]
    assert speaker_mask.tolist() == [expected_speaker_mask]
    assert word_mask.tolist() == [expected_word_mask]


@pytest.mark.unit
def test_forward_requires_mask_or_pad_id():
    loss = LatentSpeakerSupervisionLoss(speaker_token_ids=SPK_IDS)  # pad_id=None
    labels = torch.tensor([[5, 1, 2]], dtype=torch.long)
    log_probs = _make_log_probs(labels)

    with pytest.raises(ValueError):
        loss(log_probs=log_probs, labels=labels)  # no output_mask, no pad_id


@pytest.mark.unit
@pytest.mark.parametrize("per_speaker_normalization", [True, False])
def test_speaker_loss_weight_scales_linearly(per_speaker_normalization):
    labels = torch.tensor([[5, 1, 2, 6, 3, 4]], dtype=torch.long)
    log_probs = _make_log_probs(labels, seed=1)
    mask = torch.ones_like(labels, dtype=torch.float32)

    base = LatentSpeakerSupervisionLoss(
        speaker_token_ids=SPK_IDS,
        speaker_loss_weight=1.0,
        include_ce_loss=False,
        per_speaker_normalization=per_speaker_normalization,
    )
    scaled = LatentSpeakerSupervisionLoss(
        speaker_token_ids=SPK_IDS,
        speaker_loss_weight=3.0,
        include_ce_loss=False,
        per_speaker_normalization=per_speaker_normalization,
    )

    out_base = base(log_probs=log_probs, labels=labels, output_mask=mask)
    out_scaled = scaled(log_probs=log_probs, labels=labels, output_mask=mask)

    assert out_base.ndim == 0
    assert out_base.item() > 0.0
    assert torch.isclose(out_scaled, 3.0 * out_base, rtol=1e-5, atol=1e-6)


@pytest.mark.unit
def test_zero_weight_no_ce_is_zero():
    labels = torch.tensor([[5, 1, 2, 6, 3]], dtype=torch.long)
    log_probs = _make_log_probs(labels)
    mask = torch.ones_like(labels, dtype=torch.float32)

    loss = LatentSpeakerSupervisionLoss(speaker_token_ids=SPK_IDS, speaker_loss_weight=0.0, include_ce_loss=False)
    out = loss(log_probs=log_probs, labels=labels, output_mask=mask)
    assert torch.isclose(out, torch.zeros(()))


@pytest.mark.unit
def test_no_words_returns_zero():
    # Only non-speaker tokens -> no word positions -> speaker loss is exactly 0.
    labels = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    log_probs = _make_log_probs(labels)
    mask = torch.ones_like(labels, dtype=torch.float32)

    loss = LatentSpeakerSupervisionLoss(speaker_token_ids=SPK_IDS, speaker_loss_weight=2.0, include_ce_loss=False)
    out = loss(log_probs=log_probs, labels=labels, output_mask=mask)
    assert torch.isclose(out, torch.zeros(()))


@pytest.mark.unit
def test_include_ce_loss_adds_ce_term():
    labels = torch.tensor([[5, 1, 2, 6, 3]], dtype=torch.long)
    log_probs = _make_log_probs(labels, seed=2)
    mask = torch.ones_like(labels, dtype=torch.float32)

    # weight=0 so only CE remains; compare against the standalone CE helper.
    loss = LatentSpeakerSupervisionLoss(speaker_token_ids=SPK_IDS, speaker_loss_weight=0.0, include_ce_loss=True)
    total = loss(log_probs=log_probs, labels=labels, output_mask=mask)
    ce_only = loss._compute_standard_ce(log_probs, labels, mask)

    assert torch.isclose(total, ce_only, rtol=1e-5, atol=1e-6)


@pytest.mark.unit
def test_pad_id_builds_output_mask():
    # pad_id=0 should exclude trailing pad tokens; an all-padded extension must
    # not change the loss value vs an explicit mask.
    labels = torch.tensor([[5, 1, 2, 6, 3]], dtype=torch.long)
    labels_padded = torch.tensor([[5, 1, 2, 6, 3, 0, 0]], dtype=torch.long)
    log_probs = _make_log_probs(labels, seed=3)
    # extend log_probs for the padded positions (values irrelevant; masked out)
    pad_lp = _make_log_probs(labels_padded, seed=3)
    pad_lp[:, : labels.shape[1], :] = log_probs

    explicit = LatentSpeakerSupervisionLoss(speaker_token_ids=SPK_IDS, include_ce_loss=False)
    via_pad = LatentSpeakerSupervisionLoss(speaker_token_ids=SPK_IDS, include_ce_loss=False, pad_id=0)

    out_explicit = explicit(
        log_probs=log_probs, labels=labels, output_mask=torch.ones_like(labels, dtype=torch.float32)
    )
    out_pad = via_pad(log_probs=pad_lp, labels=labels_padded)

    assert torch.isclose(out_pad, out_explicit, rtol=1e-5, atol=1e-6)


@pytest.mark.unit
def test_backward_flows_to_log_probs():
    labels = torch.tensor([[5, 1, 2, 6, 3]], dtype=torch.long)
    log_probs = _make_log_probs(labels, seed=4).requires_grad_(True)
    mask = torch.ones_like(labels, dtype=torch.float32)

    loss = LatentSpeakerSupervisionLoss(speaker_token_ids=SPK_IDS, include_ce_loss=False)
    out = loss(log_probs=log_probs, labels=labels, output_mask=mask)
    out.backward()

    assert log_probs.grad is not None
    assert torch.isfinite(log_probs.grad).all()
