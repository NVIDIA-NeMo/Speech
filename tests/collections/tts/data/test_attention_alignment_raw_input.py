# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest
import torch

from nemo.collections.tts.data import text_to_speech_dataset
from nemo.collections.tts.data.text_to_speech_dataset import MagpieTTSDataset
from nemo.collections.tts.data.text_to_speech_dataset_lhotse_multiturn import MagpieTTSLhotseMultiturnDataset
from tests.collections.tts.data.test_aligned_text_turns import _Tokenizer
from tests.collections.tts.data.test_attention_alignment_turns import _cut, _speech_dataset, _sup

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('raw', ['$10!', 'Pay <bop>tɛn<eop> $10!'])
def test_forced_raw_multiturn_input_and_teacher_share_exact_bpe_ids(raw):
    dataset = _multiturn_dataset(force_raw=True)
    supervision = _sup(0.0, 1.0, raw, normalized_text='pay ten dollars')

    batch = dataset[[_cut(1.0, supervision)]]

    assert dataset.force_raw_text_input is True
    assert dataset.load_normalized_text_percent == 0.0
    assert dataset.enable_phoneme_text_input is False
    assert dataset.partial_phoneme_text_prob == 0.0
    assert dataset.text_phoneme_token_offset == 102, 'Keep the model vocabulary offset available.'
    assert dataset.text_tokenizer.calls == [(raw, 'bpe')]
    assert dataset.phoneme_tokenizer.calls == []
    count = int(batch['attention_alignment_text_lens'][0])
    positions = batch['attention_alignment_key_positions'][0, :count]
    actual = batch['text'][0].gather(0, positions)
    torch.testing.assert_close(batch['attention_alignment_text_tokens'][0, :count], actual)
    assert count == len(raw.split())
    assert (actual < dataset.bos_id).all()
    assert batch['attention_alignment_turn_indices'].tolist() == [[0, 0, 10]]


def test_multiturn_default_preserves_normalized_input_and_phoneme_configuration():
    dataset = _multiturn_dataset(force_raw=False)

    dataset[[_cut(1.0, _sup(0.0, 1.0, '$10!', normalized_text='ten dollars'))]]

    assert dataset.force_raw_text_input is False
    assert dataset.load_normalized_text_percent == 1.0
    assert dataset.enable_phoneme_text_input is True
    assert dataset.partial_phoneme_text_prob == 0.5
    assert dataset.text_tokenizer.calls == [('ten dollars', 'bpe')]


def test_forced_raw_mode_disables_literal_phoneme_span_parsing_without_dropping_text():
    raw = 'Pay <bop>tɛn<eop> $10!'
    forced = _multiturn_dataset(force_raw=True)
    ordinary = _multiturn_dataset(force_raw=False)
    ordinary.load_normalized_text_percent = 0.0

    forced[[_cut(1.0, _sup(0.0, 1.0, raw))]]
    ordinary[[_cut(1.0, _sup(0.0, 1.0, raw))]]

    assert forced.text_tokenizer.calls == [(raw, 'bpe')]
    assert forced.phoneme_tokenizer.calls == []
    assert ordinary.phoneme_tokenizer.calls == ['tɛn']
    assert ordinary.text_tokenizer.calls == [('Pay ', 'bpe'), (' $10!', 'bpe')]


@pytest.mark.parametrize('force_raw', [False, True])
def test_legacy_manifest_selection_and_attention_targets_honor_raw_override(tmp_path, monkeypatch, force_raw):
    raw = 'Pay <bop>tɛn<eop> $10!'
    dataset = _legacy_dataset(tmp_path, monkeypatch, raw, force_raw=force_raw)

    example = dataset[0]
    batch = dataset.collate_fn([example])

    expected = raw if force_raw else 'pay ten dollars'
    assert dataset.data_samples[0].text == expected
    assert dataset.text_tokenizer.calls == [(expected, 'bpe')]
    # The separate output phoneme channel remains active; only input span parsing is disabled.
    assert dataset.phoneme_tokenizer.calls == [expected]
    assert dataset.enable_phoneme_text_input is (not force_raw)
    assert dataset.load_normalized_text_percent == (0.0 if force_raw else 1.0)
    assert batch['attention_alignment_reference_valid'].tolist() == [True]
    count = int(batch['attention_alignment_text_lens'][0])
    torch.testing.assert_close(batch['attention_alignment_text_tokens'][0, :count], example['tokens'][:count].long())


@pytest.mark.parametrize('value', [0, 1, 'true', None])
@pytest.mark.parametrize('dataset_type', ['legacy', 'multiturn'])
def test_raw_text_override_requires_an_explicit_boolean(tmp_path, monkeypatch, value, dataset_type):
    with pytest.raises(ValueError, match='force_raw_text_input must be a boolean'):
        if dataset_type == 'legacy':
            _legacy_dataset(tmp_path, monkeypatch, '$10!', force_raw=value)
        else:
            _multiturn_dataset(force_raw=value)


class _PhonemeTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad = 0

    def __init__(self):
        self.calls = []

    def encode(self, text):
        self.calls.append(text)
        return [4]


def _multiturn_dataset(force_raw):
    dataset = _speech_dataset(True)
    MagpieTTSLhotseMultiturnDataset.__init__(
        dataset,
        sample_rate=16000,
        codec_model_samples_per_frame=1600,
        codec_model_input_sample_rate=16000,
        frame_stacking_factor=1,
        num_audio_codebooks=8,
        dataset_type='val',
        use_attention_alignment=True,
        enable_phoneme_text_input=True,
        text_phoneme_token_offset=102,
        partial_phoneme_text_prob=0.5,
        load_normalized_text_percent=1.0,
        add_text_bos=True,
        force_raw_text_input=force_raw,
    )
    dataset.text_tokenizer = _Tokenizer()
    dataset.phoneme_tokenizer = _PhonemeTokenizer()
    dataset._collate_phoneme_tokens = _phoneme_batch
    return dataset


def _phoneme_batch(cuts):
    # Keep the independent output channel valid without reusing its encoder for input-span assertions.
    return {
        'target_phoneme_tokens': torch.tensor([[1, 4, 2]], dtype=torch.long).repeat(len(cuts), 1),
        'target_phoneme_lens': torch.full((len(cuts),), 3, dtype=torch.long),
        'phoneme_turn_dropout': torch.zeros(len(cuts), dtype=torch.bool),
    }


def _legacy_dataset(tmp_path, monkeypatch, raw, *, force_raw):
    manifest = tmp_path / 'raw_text.json'
    manifest.write_text(
        json.dumps(
            {'audio_filepath': 'unused.wav', 'duration': 0.2, 'text': raw, 'normalized_text': 'pay ten dollars'}
        )
        + '\n'
    )
    dataset = MagpieTTSDataset(
        dataset_meta={'test': {'manifest_path': manifest, 'audio_dir': tmp_path, 'tokenizer_names': ['bpe']}},
        sample_rate=16000,
        volume_norm=False,
        codec_model_samples_per_frame=640,
        eos_id=101,
        num_audio_codebooks=8,
        load_cached_codes_if_available=False,
        load_16khz_audio=False,
        enable_phoneme_text_input=True,
        text_phoneme_token_offset=102,
        load_normalized_text_percent=1.0,
        force_raw_text_input=force_raw,
    )
    dataset.text_tokenizer = _Tokenizer()
    dataset.phoneme_tokenizer = _PhonemeTokenizer()
    dataset.use_attention_alignment = True
    dataset.attention_alignment_bos_id = 100
    dataset.attention_alignment_pad_id = 0
    monkeypatch.setattr(
        text_to_speech_dataset,
        'load_audio',
        lambda **kwargs: (np.zeros(3200, dtype=np.float32), None, 'unused.wav'),
    )
    return dataset
