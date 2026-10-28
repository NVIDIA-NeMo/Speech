# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nemo.collections.tts.data import text_to_speech_dataset
from nemo.collections.tts.data.text_to_speech_dataset import MagpieTTSDataset
from nemo.collections.tts.data.text_to_speech_dataset_lhotse_multiturn import (
    MagpieTTSLhotseMultiturnDataset,
    alignment_spoken_text,
    build_alignment_text_turns,
)

pytestmark = pytest.mark.unit


class _Supervision(SimpleNamespace):
    def has_custom(self, name):
        return hasattr(self, name)


class _Tokenizer:
    pad = 0

    def __init__(self):
        self.calls = []
        self.words = {}

    def encode(self, text, tokenizer_name):
        self.calls.append((text, tokenizer_name))
        ids = []
        for word in text.split():
            if word not in self.words:
                self.words[word] = len(self.words) + 1
            ids.append(self.words[word])
        return ids


def _sup(start, end, text, speaker='agent', **kwargs):
    return _Supervision(start=start, end=end, text=text, speaker=speaker, **kwargs)


def _cut(duration, *supervisions, task='tts'):
    return SimpleNamespace(
        duration=duration, sampling_rate=16000, supervisions=list(supervisions), task=task, id='test'
    )


def _build(cuts, tokenizer=None, **kwargs):
    return build_alignment_text_turns(
        cuts,
        tokenizer=tokenizer or _Tokenizer(),
        tokenizer_names=['bpe'] * len(cuts),
        frame_length=0.1,
        output_roles={'agent'},
        pad_id=0,
        bos_id=100,
        eos_id=101,
        **kwargs,
    )


def test_normalized_targets_are_separate_from_raw_conditioning_and_user_turns():
    tokenizer = _Tokenizer()
    first = _sup(0.0, 1.0, '$10!', normalized_text='Ten dollars!')
    user = _sup(1.0, 1.8, 'Do not align me', speaker='user')
    second = _sup(2.0, 3.0, '1000', normalized_text='one thousand.')
    cuts = [_cut(3.0, second, user, first)]
    result = _build(cuts, tokenizer=tokenizer)

    assert tokenizer.calls == [('Ten dollars', 'bpe'), ('one thousand', 'bpe')]
    assert first.text == '$10!'
    assert second.text == '1000'
    assert result['alignment_turn_indices'].tolist() == [[0, 0, 10], [0, 20, 30]]
    assert result['alignment_text_lens'].tolist() == [2, 2]
    assert result['alignment_text_tokens'].tolist() == [[1, 2], [3, 4]]
    assert result['aligned_text'][0, :4].tolist() == [1, 2, 101, 0]
    assert result['aligned_text'][0, 10:20].eq(0).all()
    assert result['aligned_text'][0, 20:24].tolist() == [3, 4, 101, 0]
    assert result['aligned_text_lens'].tolist() == [30]


def test_ragged_turn_batch_keeps_sample_indices_and_has_no_compact_special_tokens():
    result = _build(
        [
            _cut(1.0, _sup(0.2, 0.8, 'first turn', normalized_text='first turn')),
            _cut(0.5, _sup(0.0, 0.5, 'second', normalized_text='second')),
        ],
        add_text_bos=True,
    )
    assert result['alignment_turn_indices'].tolist() == [[0, 2, 8], [1, 0, 5]]
    assert result['alignment_text_lens'].tolist() == [2, 1]
    assert result['alignment_text_tokens'].tolist() == [[1, 2], [3, 0]]
    assert result['aligned_text'].shape == (2, 10)
    assert result['aligned_text'][0, 2:6].tolist() == [100, 1, 2, 101]
    assert result['aligned_text'][1, :3].tolist() == [100, 3, 101]
    assert result['aligned_text'][1, 5:].eq(0).all()


@pytest.mark.parametrize('strip_punctuation', [True, False])
def test_cleaning_retokenizes_text_and_preserves_lexical_and_numeric_punctuation(strip_punctuation):
    tokenizer = _Tokenizer()
    text = '“Don’t,” well-known, 3.5 and $10!'
    _build(
        [_cut(2.0, _sup(0.0, 2.0, text, normalized_text=text))],
        tokenizer=tokenizer,
        strip_punctuation=strip_punctuation,
    )
    expected = 'Don’t well-known 3.5 and $10' if strip_punctuation else text
    assert tokenizer.calls == [(expected, 'bpe')]


def test_invalid_turns_do_not_create_false_acoustic_matches_or_warmup_labels():
    tokenizer = _Tokenizer()
    result = _build(
        [
            _cut(1.0, _sup(-0.2, 0.5, 'left crop', normalized_text='left crop')),
            _cut(1.0, _sup(0.5, 1.3, 'right crop', normalized_text='right crop')),
            _cut(
                1.0,
                _sup(0.0, 1.0, 'wrong text', normalized_text='wrong text', challenging_text_replaced=True),
            ),
            _cut(1.0, _sup(0.0, 1.0, '...', normalized_text='...')),
        ],
        tokenizer=tokenizer,
    )
    assert result['alignment_turn_indices'].tolist() == [[0, 0, 0], [1, 5, 5], [2, 0, 0], [3, 0, 0]]
    assert result['alignment_text_lens'].tolist() == [2, 2, 2, 0]
    assert result['aligned_text'].eq(0).all()


def test_overlapping_output_turn_cannot_overwrite_previous_turn():
    result = _build(
        [
            _cut(
                1.0,
                _sup(0.0, 0.8, 'first', normalized_text='first'),
                _sup(0.5, 1.0, 'overlap', normalized_text='overlap'),
            )
        ]
    )
    assert result['alignment_turn_indices'].tolist() == [[0, 0, 8], [0, 5, 5]]
    assert result['aligned_text'][0, :3].tolist() == [1, 101, 0]
    assert result['aligned_text'][0, 5:].eq(0).all()


def test_short_turn_transcript_is_not_truncated_to_make_mas_feasible():
    result = _build([_cut(0.2, _sup(0.0, 0.2, 'one two three', normalized_text='one two three'))])
    assert result['alignment_turn_indices'].tolist() == [[0, 0, 2]]
    assert result['alignment_text_lens'].tolist() == [3]
    assert result['alignment_text_tokens'].tolist() == [[1, 2, 3]]
    assert result['aligned_text'].eq(0).all()


def test_exact_fit_never_discards_a_spoken_token_for_bos_or_eos():
    result = _build([_cut(0.2, _sup(0.0, 0.2, 'one two', normalized_text='one two'))], add_text_bos=True)
    assert result['aligned_text'].tolist() == [[1, 2]]
    assert result['alignment_text_lens'].tolist() == [2]


@pytest.mark.parametrize(
    'cuts,expected_shape', [([], (0, 0)), ([_cut(1.0, _sup(0.0, 1.0, 'user', speaker='user'))], (1, 10))]
)
def test_no_output_turns_have_well_defined_empty_shapes(cuts, expected_shape):
    result = _build(cuts)
    assert result['alignment_text_tokens'].shape == (0, 0)
    assert result['alignment_text_lens'].shape == (0,)
    assert result['alignment_turn_indices'].shape == (0, 3)
    assert result['aligned_text'].shape == expected_shape


@pytest.mark.parametrize('enabled', [True, False])
def test_dataset_adds_alignment_metadata_only_when_enabled(enabled):
    dataset = MagpieTTSLhotseMultiturnDataset.__new__(MagpieTTSLhotseMultiturnDataset)
    dataset.use_aligned_text_head = enabled
    dataset.aligned_text_strip_punctuation = True
    dataset.text_tokenizer = _Tokenizer()
    dataset.frame_length = 0.1
    dataset.output_roles = {'agent'}
    dataset.pad_id, dataset.bos_id, dataset.eos_id = 0, 100, 101
    dataset.add_text_bos = False
    dataset._initialize_tokenizers = lambda: None
    dataset._prepare_cuts = lambda cuts: (cuts, ['bpe'] * len(cuts))
    dataset._collate_audio_channels = lambda cuts: {'target_audio_lens': torch.tensor([16000])}
    original_text = torch.tensor([[7, 101] + [0] * 8])
    dataset._collate_text_channels = lambda cuts, names: {'target_text_tokens': original_text}
    dataset._collate_phoneme_tokens = lambda cuts: {}
    dataset._collect_cut_features = lambda *args: {}
    dataset._maybe_shuffle_context_audio = lambda features: None
    dataset._build_batch_dict = lambda cuts, audio, text, phoneme, features: text

    result = dataset[[_cut(1.0, _sup(0.0, 1.0, '$10', normalized_text='ten dollars'))]]
    assert result['target_text_tokens'] is original_text
    assert ('alignment_text_tokens' in result) is enabled
    assert ('aligned_text' in result) is enabled


def test_text_only_batch_never_constructs_acoustic_alignment_metadata():
    dataset = MagpieTTSLhotseMultiturnDataset.__new__(MagpieTTSLhotseMultiturnDataset)
    dataset.use_aligned_text_head = True
    dataset._initialize_tokenizers = lambda: None
    dataset._prepare_cuts = lambda cuts: (cuts, ['bpe'])
    expected = {'task': ['text_only']}
    dataset._collate_text_only_batch = lambda cuts, names: expected
    assert dataset[[_cut(1.0, _sup(0.0, 1.0, 'text'), task='text_only')]] is expected


@pytest.mark.parametrize('normalized', [None, '', '  ', '...'])
def test_missing_or_empty_normalized_text_never_falls_back_to_raw_or_ipa(normalized):
    supervision = _sup(0.0, 1.0, '$10', ipa='tɛn dɑlɚz')
    if normalized is not None:
        supervision.normalized_text = normalized
    tokenizer = _Tokenizer()
    result = _build([_cut(1.0, supervision)], tokenizer=tokenizer)
    assert tokenizer.calls == []
    assert result['alignment_text_lens'].tolist() == [0]
    assert result['alignment_turn_indices'].tolist() == [[0, 0, 0]]
    assert result['aligned_text'].eq(0).all()
    assert result['alignment_missing_normalized_text'].item() == 1


def _single_turn_dataset(monkeypatch, manifest_entries, enabled=True):
    dataset = MagpieTTSDataset.__new__(MagpieTTSDataset)
    dataset.use_aligned_text_head = enabled
    dataset.aligned_text_strip_punctuation = True
    dataset.text_tokenizer = _Tokenizer()
    dataset.phoneme_tokenizer = None
    dataset.default_tokenizer_name = 'bpe'
    dataset.enable_phoneme_text_input = False
    dataset.text_phoneme_token_offset = None
    dataset.phoneme_text_bop_marker = '<bop>'
    dataset.phoneme_text_eop_marker = '<eop>'
    dataset.eos_id = 101
    dataset.num_audio_codebooks = 8
    dataset.codec_model_samples_per_frame = 640
    dataset.load_cached_codes_if_available = False
    dataset.load_16khz_audio = False
    dataset.use_text_conditioning_tokenizer = False
    dataset.include_align_prior = False
    dataset.sample_rate = 16000
    dataset.volume_norm = False
    dataset.data_samples = [
        SimpleNamespace(
            text=entry['text'],
            manifest_entry={'audio_filepath': 'unused.wav', **entry},
            tokenizer_names=['bpe'],
            language='en',
            dataset_name='test',
            audio_dir=None,
        )
        for entry in manifest_entries
    ]
    monkeypatch.setattr(
        text_to_speech_dataset,
        'load_audio',
        lambda **kwargs: (np.zeros(3200, dtype=np.float32), None, 'unused.wav'),
    )
    return dataset


@pytest.mark.parametrize('enabled', [True, False])
def test_single_turn_alignment_reference_is_normalized_pure_bpe_and_gated(monkeypatch, enabled):
    dataset = _single_turn_dataset(
        monkeypatch, [{'text': '$10!', 'normalized_text': 'ten dollars!', 'ipa': 'tɛn dɑlɚz'}], enabled=enabled
    )
    example = dataset[0]
    assert example['tokens'].tolist() == [1, 101]
    assert example['raw_text'] == '$10!'
    if enabled:
        assert dataset.text_tokenizer.calls == [('$10!', 'bpe'), ('ten dollars', 'bpe')]
        assert example['alignment_reference_tokens'].tolist() == [2, 3]
        assert example['alignment_reference_len'] == 2
        assert example['alignment_missing_normalized_text'] == 0
    else:
        assert dataset.text_tokenizer.calls == [('$10!', 'bpe')]
        assert 'alignment_reference_tokens' not in example
    batch = dataset.collate_fn([example])
    assert ('alignment_text_tokens' in batch) is enabled
    assert ('alignment_text_lens' in batch) is enabled
    assert 'alignment_turn_indices' not in batch


@pytest.mark.parametrize('normalized', [None, '', '  ', '...'])
def test_single_turn_missing_normalization_is_invalid_without_changing_source(monkeypatch, normalized):
    entry = {'text': '$10!', 'ipa': 'tɛn dɑlɚz'}
    if normalized is not None:
        entry['normalized_text'] = normalized
    dataset = _single_turn_dataset(monkeypatch, [entry])
    example = dataset[0]
    assert example['tokens'].tolist() == [1, 101]
    assert example['alignment_reference_tokens'].numel() == 0
    assert example['alignment_reference_len'] == 0
    assert dataset.text_tokenizer.calls == [('$10!', 'bpe')]
    batch = dataset.collate_fn([example])
    assert batch['alignment_text_tokens'].shape == (1, 0)
    assert batch['alignment_text_lens'].tolist() == [0]
    assert batch['alignment_missing_normalized_text'].item() == 1


def test_single_turn_collation_keeps_normalized_reference_lengths_and_padding(monkeypatch):
    dataset = _single_turn_dataset(
        monkeypatch,
        [
            {'text': '$10', 'normalized_text': 'ten dollars'},
            {'text': '1000', 'normalized_text': 'thousand'},
            {'text': 'raw has no normalized reference'},
        ],
    )
    batch = dataset.collate_fn([dataset[index] for index in range(3)])
    assert batch['alignment_text_lens'].tolist() == [2, 1, 0]
    assert batch['alignment_text_tokens'].shape == (3, 2)
    assert batch['alignment_text_tokens'][1, 1] == dataset.text_tokenizer.pad
    assert batch['alignment_text_tokens'][2].eq(dataset.text_tokenizer.pad).all()
    assert batch['alignment_missing_normalized_text'].item() == 1


def test_public_cleaner_rejects_non_string_references_and_strips_timestamps():
    assert alignment_spoken_text(None) == ''
    assert alignment_spoken_text(10) == ''
    assert alignment_spoken_text('<|12|>Ten dollars!') == 'Ten dollars'
