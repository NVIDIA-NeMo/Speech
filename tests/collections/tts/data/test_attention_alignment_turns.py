# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from nemo.collections.tts.data.text_to_speech_dataset_lhotse_multiturn import (
    MagpieTTSLhotseMultiturnDataset,
    build_attention_alignment_turns,
)
from tests.collections.tts.data.test_aligned_text_turns import _single_turn_dataset, _Tokenizer

pytestmark = pytest.mark.unit


def test_attention_targets_gather_exact_conditioning_ids_without_retokenizing_normalized_text():
    first = _sup(0.0, 1.0, '$10!', normalized_text='ten dollars')
    user = _sup(1.0, 1.8, 'do not align', speaker='user')
    second = _sup(2.0, 3.0, '1000', normalized_text='one thousand')
    tokens = torch.zeros(1, 30, dtype=torch.long)
    tokens[0, :4] = torch.tensor([100, 7, 7, 101])
    tokens[0, 20:24] = torch.tensor([100, 9, 10, 101])

    result = _build([_cut(3.0, first, user, second)], tokens)

    assert result['attention_alignment_text_tokens'].tolist() == [[7, 7], [9, 10]]
    assert result['attention_alignment_text_lens'].tolist() == [2, 2]
    assert result['attention_alignment_key_positions'].tolist() == [[1, 2], [21, 22]]
    assert result['attention_alignment_turn_indices'].tolist() == [[0, 0, 10], [0, 20, 30]]
    assert first.text == '$10!' and second.text == '1000'
    assert tokens[0, 1:3].tolist() == [7, 7]


def test_attention_targets_preserve_ragged_sample_indices_and_pad_key_positions_with_minus_one():
    tokens = torch.zeros(2, 10, dtype=torch.long)
    tokens[0, 2:6] = torch.tensor([100, 4, 5, 101])
    tokens[1, :2] = torch.tensor([8, 101])
    result = _build([_cut(1.0, _sup(0.2, 0.8, 'first')), _cut(0.5, _sup(0.0, 0.5, 'second'))], tokens)

    assert result['attention_alignment_text_tokens'].tolist() == [[4, 5], [8, 0]]
    assert result['attention_alignment_key_positions'].tolist() == [[3, 4], [0, -1]]
    assert result['attention_alignment_text_lens'].tolist() == [2, 1]
    assert result['attention_alignment_turn_indices'].tolist() == [[0, 2, 8], [1, 0, 5]]


@pytest.mark.parametrize(
    'start,end,kwargs,content',
    [
        (-0.2, 0.5, {}, [7, 101]),
        (0.5, 1.3, {}, [7, 101]),
        (0.0, 1.0, {'challenging_text_replaced': True}, [7, 101]),
        (0.0, 1.0, {}, [7, 8]),
        (0.0, 1.0, {}, [102, 7, 101]),
        (0.0, 1.0, {}, [101]),
    ],
)
def test_unusable_turns_never_supply_attention_alignment_targets(start, end, kwargs, content):
    supervision = _sup(start, end, 'transcript', **kwargs)
    tokens = torch.zeros(1, 10, dtype=torch.long)
    position = max(0, round(supervision.start * 10))
    tokens[0, position : position + len(content)] = torch.tensor(content)
    result = _build([_cut(1.0, supervision)], tokens)

    spans = result['attention_alignment_turn_indices']
    assert not spans.numel() or spans[:, 1].eq(spans[:, 2]).all()


def test_overlap_does_not_supervise_the_overlapping_turn():
    tokens = torch.tensor([[7, 101, 0, 0, 0, 8, 101, 0, 0, 0]])
    result = _build([_cut(1.0, _sup(0.0, 0.8, 'first'), _sup(0.5, 1.0, 'overlap'))], tokens)

    spans = result['attention_alignment_turn_indices']
    assert spans[0].tolist() == [0, 0, 8]
    assert spans.size(0) == 1 or spans[1, 1] == spans[1, 2]


def test_no_output_turns_return_well_defined_empty_tensors():
    result = _build([_cut(1.0, _sup(0.0, 1.0, 'user', speaker='user'))], torch.zeros(1, 10, dtype=torch.long))

    assert result['attention_alignment_text_tokens'].shape == (0, 0)
    assert result['attention_alignment_key_positions'].shape == (0, 0)
    assert result['attention_alignment_text_lens'].shape == (0,)
    assert result['attention_alignment_turn_indices'].shape == (0, 3)


@pytest.mark.parametrize('enabled', [False, True])
def test_speech_getitem_and_batch_collation_propagate_attention_metadata_only_when_enabled(enabled):
    dataset = _speech_dataset(enabled)
    supervision = _sup(0.0, 1.0, '$10!', normalized_text='ten dollars')

    # Exercise the actual speech __getitem__, text collation, raw-ID gathering
    # and final batch assembly; only waveform I/O and cut preparation are stubbed.
    batch = dataset[[_cut(1.0, supervision)]]

    assert batch['task'] == ['tts']
    assert batch['text'][0, :3].tolist() == [100, 1, 101]
    assert dataset.text_tokenizer.calls == [('$10!', 'bpe')]
    assert ('attention_alignment_text_tokens' in batch) is enabled
    if enabled:
        assert batch['attention_alignment_text_tokens'].tolist() == [[1]]
        assert batch['attention_alignment_text_lens'].tolist() == [1]
        assert batch['attention_alignment_key_positions'].tolist() == [[1]]
        assert batch['attention_alignment_turn_indices'].tolist() == [[0, 0, 10]]
    assert 'alignment_text_tokens' not in batch


@pytest.mark.parametrize('bos_id', [None, 100])
def test_legacy_getitem_and_collate_handle_optional_bos_without_retokenizing(monkeypatch, bos_id):
    dataset = _single_turn_dataset(monkeypatch, [{'text': '$10!', 'normalized_text': 'ten dollars'}], enabled=False)
    dataset.use_attention_alignment = True
    dataset.bos_id = bos_id

    example = dataset[0]
    batch = dataset.collate_fn([example])

    assert dataset.text_tokenizer.calls == [('$10!', 'bpe')]
    assert example['tokens'].tolist() == [1, 101]
    assert batch['attention_alignment_text_tokens'].tolist() == [[1]]
    assert batch['attention_alignment_text_lens'].tolist() == [1]
    assert batch['attention_alignment_key_positions'].tolist() == [[0]]
    assert batch['attention_alignment_reference_valid'].tolist() == [True]
    assert 'alignment_text_tokens' not in batch


class _Supervision(SimpleNamespace):
    def has_custom(self, name):
        return hasattr(self, name)


def _sup(start, end, text, speaker='agent', **kwargs):
    return _Supervision(start=start, end=end, text=text, speaker=speaker, **kwargs)


def _cut(duration, *supervisions):
    return _Supervision(
        duration=duration,
        sampling_rate=16000,
        supervisions=list(supervisions),
        task='tts',
        id='test',
        lang='en',
    )


def _build(cuts, conditioning_tokens):
    return build_attention_alignment_turns(
        cuts,
        conditioning_tokens=conditioning_tokens,
        frame_length=0.1,
        output_roles={'agent'},
        bos_id=100,
        eos_id=101,
        pad_id=0,
    )


def _speech_dataset(enabled):
    dataset = MagpieTTSLhotseMultiturnDataset.__new__(MagpieTTSLhotseMultiturnDataset)
    values = {
        'use_aligned_text_head': False,
        'use_attention_alignment': enabled,
        'dynamic_phoneme_delay': False,
        'text_tokenizer': _Tokenizer(),
        'frame_length': 0.1,
        'output_roles': {'agent'},
        'input_roles': {'user'},
        'pad_id': 0,
        'bos_id': 100,
        'eos_id': 101,
        'interruption_token_id': 102,
        'add_text_bos': True,
        'phoneme_tokenizer': None,
        'enable_phoneme_text_input': False,
        'text_phoneme_token_offset': None,
        'partial_phoneme_text_prob': 0.0,
        'partial_phoneme_portion_min': 0.25,
        'partial_phoneme_portion_max': 0.75,
        'phoneme_text_bop_marker': '<bop>',
        'phoneme_text_eop_marker': '<eop>',
        'ignore_phoneme_languages': [],
        'dataset_type': 'val',
        'load_normalized_text_percent': 0.0,
        'use_text_conditioning_tokenizer': False,
        'include_align_prior': False,
    }
    for name, value in values.items():
        setattr(dataset, name, value)
    dataset._initialize_tokenizers = lambda: None
    dataset._prepare_cuts = lambda cuts: (cuts, ['bpe'] * len(cuts))
    dataset._collate_audio_channels = lambda cuts: {
        'source_audio': torch.zeros(1, 16000),
        'source_audio_lens': torch.tensor([16000]),
        'target_audio': torch.zeros(1, 16000),
        'target_audio_lens': torch.tensor([16000]),
        'user_audio_turn_splitted': torch.empty(0, 0),
        'user_audio_turn_splitted_lens': torch.empty(0, dtype=torch.long),
        'user_audio_turn_splitted_indices': torch.empty(0, 3, dtype=torch.long),
    }
    empty_feature_names = (
        'target_codes',
        'source_codes',
        'audio_16khz',
        'context_audio',
        'context_audio_codes',
        'align_priors',
        'rewards',
    )
    dataset._collect_cut_features = lambda *args: {
        'dataset_names': ['test'],
        'languages': ['en'],
        **{name: [] for name in empty_feature_names},
    }
    dataset._maybe_shuffle_context_audio = lambda features: None
    return dataset
