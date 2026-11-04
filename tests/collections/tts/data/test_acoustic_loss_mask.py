# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-example acoustic supervision with real speech conditioning."""

from types import SimpleNamespace

import pytest
import torch
from lhotse import CutSet
from torch import nn

from nemo.collections.tts.data.text_to_speech_dataset_lhotse_multiturn import MagpieTTSLhotseMultiturnDataset
from nemo.collections.tts.models.easy_magpietts import EasyMagpieTTSModel

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('all_disabled', [False, True])
def test_acoustic_mask_excludes_rows_from_loss_and_gradient(all_disabled):
    model = SimpleNamespace(num_all_tokens_per_codebook=4, cross_entropy_loss=nn.CrossEntropyLoss(reduction='none'))
    logits = torch.randn(2, 3, 8, requires_grad=True)
    targets = torch.randint(4, (2, 2, 3))
    lengths = torch.tensor([3, 2])
    agent_mask = torch.tensor([[True, False, True], [True, True, False]])
    enabled = torch.tensor([not all_disabled, False])
    loss, _ = EasyMagpieTTSModel.compute_loss(
        model, logits, targets, lengths, agent_mask_target=agent_mask, acoustic_loss_mask=enabled
    )
    if all_disabled:
        assert loss.item() == 0
    else:
        expected, _ = EasyMagpieTTSModel.compute_loss(
            model, logits[:1], targets[:1], lengths[:1], agent_mask_target=agent_mask[:1]
        )
        torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.count_nonzero(logits.grad[1]) == 0
    assert bool(torch.count_nonzero(logits.grad[0])) == (not all_disabled)


def test_acoustic_mask_rejects_wrong_shape():
    model = SimpleNamespace(num_all_tokens_per_codebook=4, cross_entropy_loss=nn.CrossEntropyLoss(reduction='none'))
    with pytest.raises(ValueError, match='shape'):
        EasyMagpieTTSModel.compute_loss(
            model,
            torch.randn(2, 3, 8),
            torch.zeros(2, 2, 3),
            torch.tensor([3, 3]),
            acoustic_loss_mask=torch.ones(2, 3, dtype=torch.bool),
        )


def test_audio_dataset_propagates_per_example_acoustic_loss_tags():
    from tests.collections.tts.data.test_magpietts_dataset_lhotse import (
        _FakeIPATokenizer,
        _multiturn_cutset,
        _multiturn_dataset_kwargs,
    )

    kwargs = _multiturn_dataset_kwargs()
    kwargs.update(input_roles=['user'], output_roles=['assistant'])
    dataset = MagpieTTSLhotseMultiturnDataset(**kwargs)
    dataset.phoneme_tokenizer = _FakeIPATokenizer()
    normal, muted = next(iter(_multiturn_cutset())), next(iter(_multiturn_cutset()))
    muted.id += '-muted'
    muted.acoustic_loss = False
    for supervision in muted.supervisions:
        if supervision.speaker == 'assistant':
            supervision.ipa = 'test ipa'
    batch = dataset[CutSet.from_cuts([normal, muted])]
    assert batch['acoustic_loss_mask'].tolist() == [True, False]
    assert batch['audio'].size(0) == 2
