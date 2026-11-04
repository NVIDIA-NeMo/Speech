# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools

import pytest
import torch

from nemo.collections.tts.parts.utils.monotonic_alignment import TRITON_AVAILABLE, maximum_path_indices

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('num_audio,num_text', [(1, 1), (7, 1), (4, 4), (5, 3), (8, 4)])
def test_mas_matches_exhaustive_search(num_audio, num_text):
    scores = torch.randn(1, num_audio, num_text, generator=torch.Generator().manual_seed(19), device='cpu')
    path = maximum_path_indices(scores, torch.tensor([num_text]), torch.tensor([num_audio]), backend='cpu')[0]
    possible_scores = []
    for boundaries in itertools.combinations(range(1, num_audio), num_text - 1):
        indices = torch.zeros(num_audio, dtype=torch.long)
        for boundary in boundaries:
            indices[boundary:] += 1
        possible_scores.append(scores[0, torch.arange(num_audio), indices].sum())
    selected_score = scores[0, torch.arange(num_audio), path].sum()
    torch.testing.assert_close(selected_score, torch.stack(possible_scores).max())
    assert path[0] == 0 and path[-1] == num_text - 1
    assert ((path.diff() == 0) | (path.diff() == 1)).all()
    assert torch.bincount(path, minlength=num_text).min() >= 1


def test_mas_handles_padding_single_tokens_and_invalid_turns():
    scores = torch.zeros(6, 8, 5, device='cpu')
    text_lens = torch.tensor([3, 1, 0, 5, 2, 6], device='cpu')
    audio_lens = torch.tensor([5, 7, 8, 3, 0, 8], device='cpu')
    path = maximum_path_indices(scores, text_lens, audio_lens, backend='cpu')
    torch.testing.assert_close(path[0], torch.tensor([0, 0, 0, 1, 2, -1, -1, -1]))
    torch.testing.assert_close(path[1], torch.tensor([0, 0, 0, 0, 0, 0, 0, -1]))
    assert (path[2:] == -1).all()


def test_mas_rejects_nonfinite_complete_paths():
    scores = torch.zeros(2, 5, 3, device='cpu')
    scores[0, :, 2] = -torch.inf
    scores[1, :, 0] = -torch.inf
    path = maximum_path_indices(scores, torch.tensor([3, 1]), torch.tensor([5, 5]), backend='cpu')
    assert (path == -1).all()


@pytest.mark.parametrize('shape', [(0, 8, 3), (3, 0, 2), (3, 8, 0)])
def test_mas_empty_dimensions(shape):
    scores = torch.empty(shape, device='cpu')
    lengths = torch.zeros(shape[0], dtype=torch.long, device='cpu')
    path = maximum_path_indices(scores, lengths, lengths, backend='cpu')
    assert path.shape == shape[:2]
    assert (path == -1).all()


def test_mas_is_discrete_and_accepts_noncontiguous_scores():
    original = torch.randn(2, 4, 7, device='cpu', requires_grad=True)
    scores = original.transpose(1, 2)
    path = maximum_path_indices(scores, torch.tensor([4, 2]), torch.tensor([7, 5]), backend='cpu')
    assert not path.requires_grad
    safe_path = path.clamp_min(0)
    selected = scores.gather(-1, safe_path.unsqueeze(-1)).squeeze(-1)
    selected.masked_fill(path < 0, 0).sum().backward()
    assert original.grad is not None
    assert original.grad.sum() == 12
    assert (original.grad[1, :, 5:] == 0).all()


def test_mas_validates_inputs():
    with pytest.raises(ValueError, match='floating-point'):
        maximum_path_indices(torch.zeros(2, 4, dtype=torch.long), torch.tensor([2]), torch.tensor([4]))
    with pytest.raises(ValueError, match='shape'):
        maximum_path_indices(torch.zeros(2, 4, 2), torch.tensor([2]), torch.tensor([4]))
    with pytest.raises(ValueError, match='integers'):
        maximum_path_indices(torch.zeros(1, 4, 2), torch.tensor([2.0]), torch.tensor([4]))
    with pytest.raises(ValueError, match='Unsupported'):
        maximum_path_indices(torch.zeros(1, 4, 2), torch.tensor([2]), torch.tensor([4]), backend='unknown')


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason='CUDA and Triton are required')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('tied', [False, True])
def test_triton_mas_matches_cpu(dtype, tied):
    generator = torch.Generator(device='cpu').manual_seed(123)
    scores = torch.randn(6, 129, 47, generator=generator, device='cpu').to(dtype=dtype)
    if tied:
        scores.zero_()
    text_lens = torch.tensor([47, 11, 1, 0, 9, 6], device='cpu')
    audio_lens = torch.tensor([129, 80, 18, 129, 4, 0], device='cpu')
    expected = maximum_path_indices(scores, text_lens, audio_lens, backend='cpu')
    actual = maximum_path_indices(scores.cuda(), text_lens.cuda(), audio_lens.cuda(), backend='triton')
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason='CUDA and Triton are required')
def test_triton_mas_noncontiguous_and_unreachable_paths():
    scores = torch.randn(3, 5, 17, device='cuda').transpose(1, 2)
    scores[1, :, 3] = -torch.inf
    scores[2, :, 0] = -torch.inf
    text_lens = torch.tensor([5, 4, 1], device='cuda')
    audio_lens = torch.tensor([17, 12, 6], device='cuda')
    expected = maximum_path_indices(scores, text_lens, audio_lens, backend='cpu')
    actual = maximum_path_indices(scores, text_lens, audio_lens, backend='triton')
    torch.testing.assert_close(actual.cpu(), expected.cpu())


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason='CUDA and Triton are required')
def test_triton_mas_varying_shapes_share_a_text_width_bucket():
    # Exact padded lengths and strides vary within a single BLOCK_TEXT=32
    # bucket. Both contiguous and transposed layouts must remain correct without
    # specializing the kernel on each conversation's dimensions or strides.
    for batch_size, audio_steps, text_steps in [(1, 53, 17), (3, 79, 23), (2, 103, 31), (4, 71, 19)]:
        for transposed in (False, True):
            shape = (batch_size, text_steps, audio_steps) if transposed else (batch_size, audio_steps, text_steps)
            scores = torch.randn(shape, device='cuda')
            if transposed:
                scores = scores.transpose(1, 2)
            text_lens = torch.arange(text_steps, text_steps - batch_size, -1, device='cuda')
            audio_lens = torch.arange(audio_steps, audio_steps - batch_size, -1, device='cuda')
            expected = maximum_path_indices(scores, text_lens, audio_lens, backend='cpu')
            actual = maximum_path_indices(scores, text_lens, audio_lens, backend='triton')
            torch.testing.assert_close(actual.cpu(), expected.cpu())
