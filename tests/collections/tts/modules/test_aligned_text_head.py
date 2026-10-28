# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from nemo.collections.tts.modules.aligned_text_head import AlignedTextHead, align_text_turns

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('chunk_size', [1, 3, 128])
def test_checkpointed_ce_matches_full_logits_and_gradients(chunk_size):
    torch.manual_seed(31)
    head = AlignedTextHead(7, 13, projection_dim=5)
    reference = copy.deepcopy(head)
    hidden = torch.randn(2, 6, 7, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    targets = torch.randint(0, 13, (2, 6))
    mask = torch.tensor([[True, False, True, True, False, True], [False, True, True, False, False, True]])
    actual_loss = head.loss(hidden, targets, mask, chunk_size)
    expected_loss = F.cross_entropy(reference(reference_hidden)[mask], targets[mask])
    torch.testing.assert_close(actual_loss, expected_loss)
    actual_loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad)
    for actual, expected in zip(head.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad)


def test_empty_ce_keeps_zero_decoder_and_head_gradients():
    head = AlignedTextHead(7, 13, projection_dim=5)
    hidden = torch.randn(2, 6, 7, requires_grad=True)
    loss = head.loss(hidden, torch.full((2, 6), -100), torch.zeros(2, 6, dtype=torch.bool))
    assert loss == 0
    loss.backward()
    assert hidden.grad is not None and (hidden.grad == 0).all()
    assert all(parameter.grad is not None and (parameter.grad == 0).all() for parameter in head.parameters())


class _DeterministicAligner(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.calls = []

    def forward(self, audio_features, text_features, audio_lens, text_lens):
        self.calls.append((audio_features.detach().clone(), text_features.detach().clone(), audio_lens, text_lens))
        batch, frames = audio_features.shape[:2]
        indices = torch.full((batch, frames), -1, dtype=torch.long, device=audio_features.device)
        durations = torch.zeros(batch, text_features.size(1), dtype=torch.long, device=audio_features.device)
        for row in range(batch):
            audio_length, text_length = int(audio_lens[row]), int(text_lens[row])
            path = torch.arange(audio_length, device=audio_features.device) * text_length // audio_length
            indices[row, :audio_length] = path
            durations[row].scatter_add_(0, path, torch.ones_like(path))
        mask = torch.arange(frames, device=audio_features.device)[None, :] < audio_lens[:, None]
        # A per-frame scalar lets the test verify weighting across uneven packs.
        loss = self.scale * (audio_features[:, :, 0] * mask).sum() / mask.sum()
        return SimpleNamespace(
            loss=loss,
            token_indices=indices,
            durations=durations,
            valid_turns=torch.ones(batch, dtype=torch.bool, device=audio_features.device),
        )


def _embed_text(tokens, lengths):
    del lengths
    return tokens.float().unsqueeze(2)


def _turn_fixture():
    audio = torch.arange(24, dtype=torch.float32).reshape(2, 12, 1).requires_grad_(True)
    text = torch.tensor([[11, 12, 0], [21, 22, 23], [31, 32, 0], [41, 42, 43], [51, 0, 0]])
    text_lens = torch.tensor([2, 3, 2, 3, 1])
    # Two independent turns in sample0, one in sample1, then invalid T<N/bounds.
    turns = torch.tensor([[0, 1, 5], [0, 7, 12], [1, 3, 7], [1, 8, 10], [2, 0, 2]])
    warmup = torch.zeros(2, 12, dtype=torch.long)
    warmup[0, 1:3] = text[0, :2]
    warmup[0, 7:10] = text[1, :3]
    warmup[1, 3:5] = text[2, :2]
    return audio, text, text_lens, turns, warmup


@pytest.mark.parametrize('use_alignment', [False, True])
def test_turns_restore_to_original_grid_without_spilling_or_invalid_supervision(use_alignment):
    audio, text, text_lens, turns, warmup = _turn_fixture()
    aligner = _DeterministicAligner()
    output = align_text_turns(
        aligner,
        audio,
        text,
        text_lens,
        turns,
        warmup,
        _embed_text,
        pad_id=0,
        use_alignment=use_alignment,
        max_turns_per_batch=2,
        max_frames_per_batch=8,
    )
    expected_mask = torch.zeros_like(warmup, dtype=torch.bool)
    expected_mask[0, 1:5] = True
    expected_mask[0, 7:12] = True
    expected_mask[1, 3:7] = True
    torch.testing.assert_close(output.loss_mask, expected_mask)
    if use_alignment:
        torch.testing.assert_close(output.targets[0, 1:5], torch.tensor([11, 11, 12, 12]))
        torch.testing.assert_close(output.targets[0, 7:12], torch.tensor([21, 21, 22, 22, 23]))
        torch.testing.assert_close(output.targets[1, 3:7], torch.tensor([31, 31, 32, 32]))
    else:
        torch.testing.assert_close(output.targets, warmup)
    assert (output.targets[~expected_mask] == 0).all()
    assert output.valid_turn_count == 3 and output.turn_count == 5
    assert output.token_count == 7 and output.one_frame_tokens == 1
    expected_mean = audio[:, :, 0][expected_mask].mean()
    torch.testing.assert_close(output.loss, 2.0 * expected_mean)
    output.loss.backward()
    torch.testing.assert_close(aligner.scale.grad, expected_mean)
    assert audio.grad is None
    assert len(aligner.calls) == 2
    assert all(batch.size(0) * batch.size(1) <= 8 for batch, _, _, _ in aligner.calls)


def test_text_embedding_callback_has_no_gradient_path_to_generator():
    audio, text, text_lens, turns, warmup = _turn_fixture()
    embedding = nn.Embedding(64, 4)
    aligner = _DeterministicAligner()
    output = align_text_turns(
        aligner,
        audio,
        text,
        text_lens,
        turns,
        warmup,
        lambda tokens, lengths: embedding(tokens),
        pad_id=0,
        use_alignment=True,
    )
    output.loss.backward()
    assert embedding.weight.grad is None
    assert all(not text_features.requires_grad for _, text_features, _, _ in aligner.calls)


def test_empty_and_invalid_turns_keep_all_aligner_gradients_without_calling_it():
    audio, text, text_lens, turns, warmup = _turn_fixture()
    aligner = _DeterministicAligner()
    output = align_text_turns(
        aligner, audio, text[3:], text_lens[3:], turns[3:], warmup, _embed_text, pad_id=0, use_alignment=True
    )
    assert output.loss == 0 and not output.loss_mask.any()
    assert output.valid_turn_count == 0 and output.turn_count == 2
    assert output.token_count == 0 and output.one_frame_tokens == 0
    assert not aligner.calls
    output.loss.backward()
    assert aligner.scale.grad is not None and aligner.scale.grad == 0


def test_overlapping_turns_and_oversized_turns_fail_explicitly():
    audio, text, text_lens, turns, warmup = _turn_fixture()
    turns = turns[:2].clone()
    turns[1] = torch.tensor([0, 3, 8])
    with pytest.raises(ValueError, match='overlap'):
        align_text_turns(
            _DeterministicAligner(),
            audio,
            text[:2],
            text_lens[:2],
            turns,
            warmup,
            _embed_text,
            pad_id=0,
            use_alignment=True,
        )
    with pytest.raises(ValueError, match='exceeding'):
        align_text_turns(
            _DeterministicAligner(),
            audio,
            text[:1],
            text_lens[:1],
            turns[:1],
            warmup,
            _embed_text,
            pad_id=0,
            use_alignment=True,
            max_frames_per_batch=3,
        )
