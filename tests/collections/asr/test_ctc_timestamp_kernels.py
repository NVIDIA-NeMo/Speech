# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU regression cases against NeMo's existing NumPy CTC recurrence."""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('triton')

from nemo.collections.speechlm2.parts.ctc_timestamp_kernels import (
    MAX_SUPPORTED_STATES,
    _ctc_viterbi_kernel,
    batched_ctc_viterbi,
    gpu_prepared_paths,
)
from nemo.collections.speechlm2.parts.ctc_timestamp_utils import (
    MultiSpeakerSOTWordTimestampAligner,
    _banded_ctc_search,
    _ctc_banded_backtrace,
    _ctc_banded_forward,
)


pytestmark = pytest.mark.unit


def _expanded(tokens):
    return np.asarray([0, *[piece for token in tokens for piece in (token, 0)]], dtype=np.int64)


def _reference(log_probs, labels, frame_lengths, state_lengths):
    batch, max_frames, _ = log_probs.shape
    paths = np.full((batch, max_frames), -1, np.int32)
    scores = np.full(batch, -np.inf, np.float32)
    for index, (frames, states) in enumerate(zip(frame_lengths, state_lengths)):
        if frames <= 0 or frames > max_frames or states <= 0 or states > labels.shape[1]:
            continue
        starts = np.zeros(frames, dtype=np.int64)
        previous, pointers = _ctc_banded_forward(
            log_probs[index, :frames],
            labels[index, :states],
            np.arange(frames, dtype=np.int64),
            None,
            starts,
            states,
        )
        path, score = _ctc_banded_backtrace(previous, pointers, starts, states)
        paths[index, :frames] = path
        scores[index] = score
    return paths, scores


def _batch(token_rows, frame_lengths, columns=7, seed=1234, *, zeros=False):
    rng = np.random.default_rng(seed)
    labels = [_expanded(tokens) for tokens in token_rows]
    max_states = max(map(len, labels))
    max_frames = max(frame_lengths)
    emissions = rng.normal(-3.0, 1.0, (len(labels), max_frames, columns)).astype(np.float32)
    if zeros:
        emissions[:] = 0
    packed_labels = np.full((len(labels), max_states), -97, np.int64)
    for row, target in enumerate(labels):
        packed_labels[row, : len(target)] = target
    return emissions, packed_labels, np.asarray(frame_lengths, np.int64), np.asarray(list(map(len, labels)), np.int64)


def _assert_matches(arrays, **kwargs):
    expected_paths, expected_scores = _reference(*arrays)
    actual_paths, actual_scores = batched_ctc_viterbi(
        *(torch.as_tensor(array, device='cuda') for array in arrays), **kwargs
    )
    torch.cuda.synchronize()
    np.testing.assert_array_equal(actual_paths.cpu().numpy(), expected_paths)
    np.testing.assert_array_equal(actual_scores.cpu().numpy(), expected_scores)
    return actual_paths.cpu(), actual_scores.cpu()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required for the CTC kernel')
@pytest.mark.parametrize('seed', [0, 4, 127])
def test_random_variable_lengths_repeats_and_padding_match_native(seed):
    _assert_matches(
        _batch(
            [[], [1], [1, 1], [1, 2, 1], [2, 3, 3, 4], [6, 2, 5, 1, 6, 3]],
            [1, 7, 9, 13, 17, 29],
            seed=seed,
        )
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required for the CTC kernel')
@pytest.mark.parametrize('banded', [False, True])
def test_changed_batch_dimensions_reuse_the_compiled_kernel_and_preserve_exact_paths(monkeypatch, banded):
    launched = []
    run = _ctc_viterbi_kernel.run

    def record_kernel(*args, **kwargs):
        kernel = run(*args, **kwargs)
        launched.append(kernel)
        return kernel

    monkeypatch.setattr(_ctc_viterbi_kernel, 'run', record_kernel)
    for tokens, frames, columns in (
        ([[1, 2], [2]], [13, 11], 4),
        ([[1, 2, 1], [2, 1], [3, 2]], [16, 10, 12], 8),
        ([[2, 1, 3]], [19], 6),
    ):
        arrays = _batch(tokens, frames, columns=columns)
        bands = {}
        if banded:
            bands = dict(
                band_starts=torch.zeros(arrays[0].shape[:2], device='cuda', dtype=torch.long),
                band_widths=torch.as_tensor(arrays[3], device='cuda'),
                max_band_width=arrays[1].shape[1],
            )
        _assert_matches(arrays, **bands)
    assert len(launched) == 3
    assert launched[0] is not None
    assert all(kernel is launched[0] for kernel in launched)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required for the CTC kernel')
@pytest.mark.parametrize('tokens,frames', [([], 7), ([1], 1), ([1], 7), ([1, 1], 2), ([1, 1], 3), ([1, 2, 1], 3)])
def test_all_ties_follow_stay_step_skip_and_terminal_blank_preference(tokens, frames):
    paths, scores = _assert_matches(_batch([tokens], [frames], zeros=True))
    if tokens == [1] and frames == 7:
        assert paths[0].tolist() == [1, 2, 2, 2, 2, 2, 2]
    if tokens == [1, 1] and frames == 2:
        assert paths.eq(-1).all() and torch.isneginf(scores).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required for the CTC kernel')
@pytest.mark.parametrize('token_count', [8, 17, 63, 128])
def test_non_power_of_two_target_widths_match_native(token_count):
    tokens = [(index % 6) + 1 for index in range(token_count)]
    _assert_matches(_batch([tokens], [token_count + 41], seed=token_count))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required for the CTC kernel')
def test_negative_infinity_emissions_and_unreachable_record():
    arrays = _batch([[1, 2], [1, 2]], [11, 11])
    arrays[0][0, 3, :] = -np.inf
    arrays[0][1, :, 2] = -np.inf
    paths, scores = _assert_matches(arrays)
    assert paths.eq(-1).all() and torch.isneginf(scores).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required for the CTC kernel')
def test_invalid_or_zero_record_lengths_are_safely_unreachable():
    arrays = _batch([[1], [1], [1], [1], [1]], [7] * 5)
    arrays[2][:] = [0, -1, 8, 7, 7]
    arrays[3][-2:] = [0, 4]
    paths, scores = _assert_matches(arrays)
    assert paths.eq(-1).all() and torch.isneginf(scores).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required for the CTC kernel')
@pytest.mark.parametrize('default_dtype', [torch.float16, torch.bfloat16])
def test_default_dtype_does_not_change_results_or_modify_inputs(default_dtype):
    inputs = [torch.as_tensor(array, device='cuda') for array in _batch([[1, 2, 2], [3, 1]], [19, 11])]
    originals = [tensor.clone() for tensor in inputs]
    expected_paths, expected_scores = batched_ctc_viterbi(*inputs)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        paths, scores = batched_ctc_viterbi(*inputs)
        assert torch.get_default_dtype() == default_dtype
    finally:
        torch.set_default_dtype(previous_dtype)
    torch.testing.assert_close(paths, expected_paths, rtol=0, atol=0)
    torch.testing.assert_close(scores, expected_scores, rtol=0, atol=0)
    for original, observed in zip(originals, inputs):
        torch.testing.assert_close(observed, original, rtol=0, atol=0)


@pytest.mark.parametrize('shape,state_width', [((0, 8, 5), 3), ((2, 0, 5), 3), ((2, 8, 5), 0)])
def test_empty_dimensions_do_not_need_cuda(shape, state_width):
    batch, frames, _ = shape
    paths, scores = batched_ctc_viterbi(
        torch.empty(shape, dtype=torch.float32),
        torch.empty((batch, state_width), dtype=torch.long),
        torch.zeros(batch, dtype=torch.long),
        torch.zeros(batch, dtype=torch.long),
    )
    assert paths.shape == (batch, frames) and scores.shape == (batch,)
    assert paths.eq(-1).all() and torch.isneginf(scores).all()


def test_nonempty_cpu_execution_is_explicitly_unsupported():
    with pytest.raises(NotImplementedError, match='CUDA'):
        batched_ctc_viterbi(torch.zeros(1, 7, 3), torch.tensor([[0, 1, 0]]), torch.tensor([7]), torch.tensor([3]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required for the CTC kernel')
def test_excessive_target_width_is_explicitly_unsupported():
    with pytest.raises(NotImplementedError, match=str(MAX_SUPPORTED_STATES)):
        batched_ctc_viterbi(
            torch.zeros(1, 2, 3, device='cuda'),
            torch.zeros(1, MAX_SUPPORTED_STATES + 1, device='cuda', dtype=torch.long),
            torch.tensor([2], device='cuda'),
            torch.tensor([1], device='cuda'),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required for the CTC kernel')
@pytest.mark.parametrize('width,starts', [(3, [0, 0, 2, 4, 4, 4]), (1, [0, 1, 2, 3, 4, 5, 6])])
@pytest.mark.parametrize('ties', [False, True])
def test_moving_band_preserves_shifted_predecessors_and_terminal_preference(width, starts, ties):
    arrays = _batch([[1, 2, 3]], [len(starts)], zeros=ties)
    ctc, labels, _, _ = arrays
    starts = np.asarray(starts, dtype=np.int64)
    previous, pointers = _ctc_banded_forward(ctc[0], labels[0], np.arange(len(starts)), None, starts, width)
    expected_path, expected_score = _ctc_banded_backtrace(previous, pointers, starts, len(labels[0]))
    path, score = batched_ctc_viterbi(
        *(torch.as_tensor(array, device='cuda') for array in arrays),
        band_starts=torch.as_tensor(starts[None, :], device='cuda'),
        band_widths=torch.tensor([width], device='cuda'),
        max_band_width=width,
    )
    np.testing.assert_array_equal(path[0].cpu().numpy(), expected_path)
    np.testing.assert_array_equal(score.cpu().numpy(), np.asarray([expected_score], dtype=np.float32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required for the CTC kernel')
def test_long_turn_anchor_bands_and_variable_lengths_match_native_search():
    rng = np.random.default_rng(29)
    token_rows = [rng.integers(1, 19, size=count).tolist() for count in (160, 185, 145)]
    arrays = _batch(token_rows, [1100, 1150, 1000], columns=19)
    ctc, labels, frame_lengths, state_lengths = arrays
    reference = [
        _banded_ctc_search(ctc[index, :frames], labels[index, :states], np.arange(frames), None)
        for index, (frames, states) in enumerate(zip(frame_lengths, state_lengths))
    ]
    starts = np.zeros(ctc.shape[:2], np.int64)
    widths = []
    for index, (_, _, band) in enumerate(reference):
        starts[index, : len(band)] = band
        states = state_lengths[index]
        widths.append(min(states, 2 * min(states, int(np.ceil(0.10 * states + 100))) + 1))
        assert widths[-1] < states
        assert np.isin(np.diff(band), [0, 1, 2]).all()
    paths, scores = batched_ctc_viterbi(
        *(torch.as_tensor(array, device='cuda') for array in arrays),
        band_starts=torch.as_tensor(starts, device='cuda'),
        band_widths=torch.tensor(widths, device='cuda'),
        max_band_width=max(widths),
    )
    for index, (expected_path, expected_score, _) in enumerate(reference):
        count = frame_lengths[index]
        np.testing.assert_array_equal(paths[index, :count].cpu().numpy(), expected_path)
        np.testing.assert_array_equal(scores[index].cpu().numpy(), np.float32(expected_score))
        assert paths[index, count:].eq(-1).all()


def _prepared_example(*, refine):
    tokenizer = SimpleNamespace(text_to_ids=lambda text: [{'first': 0, 'second': 1}[word] for word in text.split()])
    logits = torch.full((2, 12, 3), -12.0)
    logits[:, :, 2] = 0.0
    logits[:, 2:4, 0] = 12.0
    logits[:, 9:11, 1] = 12.0
    aligner = MultiSpeakerSOTWordTimestampAligner(tokenizer=tokenizer, blank_id=2, refine_word_timestamps=refine)
    prepared = aligner.prepare_alignment(
        logits.log_softmax(-1), None, ['first second', 'second'], audio_durations=[1.2, 1.2]
    )
    return aligner, prepared


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required for the CTC kernel')
@pytest.mark.parametrize('refine', [False, True])
def test_prepared_cuda_search_reuses_all_native_word_metadata_and_refinement(refine):
    aligner, prepared = _prepared_example(refine=refine)
    original = copy.deepcopy(prepared)
    reference = aligner.align_prepared(prepared)
    observed = aligner.align_prepared(prepared, device='cuda')
    assert observed == reference
    for old, new in zip(original['records'], prepared['records']):
        torch.testing.assert_close(old['ctc_log_probs'], new['ctc_log_probs'], rtol=0, atol=0)
    words = observed[0]['speaker_word_timestamps'][0]
    assert (words[0]['end'] == words[1]['start']) == refine


def test_default_and_explicit_cpu_paths_never_dispatch_cuda():
    aligner, prepared = _prepared_example(refine=False)
    with patch(
        'nemo.collections.speechlm2.parts.ctc_timestamp_kernels.gpu_prepared_paths',
        side_effect=AssertionError('CPU alignment dispatched CUDA'),
    ):
        assert aligner.align_prepared(prepared) == aligner.align_prepared(prepared, device='cpu')
    assert gpu_prepared_paths(prepared, 'cpu') == {}


def test_missing_optional_triton_retains_native_alignment():
    aligner, prepared = _prepared_example(refine=False)
    reference = aligner.align_prepared(prepared)
    with patch('nemo.collections.speechlm2.parts.ctc_timestamp_kernels.triton', None):
        assert aligner.align_prepared(prepared, device='cuda') == reference


def test_nonzero_absent_speaker_prior_retains_native_alignment():
    aligner, prepared = _prepared_example(refine=False)
    aligner.epsilon = 2.0
    with patch(
        'nemo.collections.speechlm2.parts.ctc_timestamp_kernels.gpu_prepared_paths',
        side_effect=AssertionError('Nonzero speaker prior dispatched CUDA'),
    ):
        assert aligner.align_prepared(prepared, device='cuda') == aligner.align_prepared(prepared)


@pytest.mark.parametrize('weight', [float('inf'), float('nan')])
def test_nonfinite_speaker_prior_weight_retains_native_alignment(weight):
    aligner, prepared = _prepared_example(refine=False)
    prepared['config']['speaker_logprob_weight'] = weight
    with (
        np.errstate(invalid='ignore'),
        patch(
            'nemo.collections.speechlm2.parts.ctc_timestamp_kernels.batched_ctc_viterbi',
            side_effect=AssertionError('Nonfinite speaker prior dispatched CUDA'),
        ),
    ):
        assert gpu_prepared_paths(prepared, 'cuda') == {}
        assert aligner.align_prepared(prepared, device='cuda') == aligner.align_prepared(prepared)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required for the CTC kernel')
def test_unsupported_prepared_record_falls_back_to_native_search():
    aligner, prepared = _prepared_example(refine=False)
    prepared['records'][0]['streams'].append(copy.deepcopy(prepared['records'][0]['streams'][0]))
    paths = gpu_prepared_paths(prepared, 'cuda')
    assert (0, 0) not in paths and (1, 0) in paths
    assert aligner.align_prepared(prepared, device='cuda') == aligner.align_prepared(prepared)
