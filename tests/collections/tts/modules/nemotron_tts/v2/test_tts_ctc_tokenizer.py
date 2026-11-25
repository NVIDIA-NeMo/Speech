# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from nemo.collections.tts.modules.nemotron_tts.v2.timestamp_head import TTSCtcTokenizer, decode_ctc_tokens

pytestmark = pytest.mark.unit


def test_tts_ctc_reuses_full_native_input_encoding_and_aggregate_offsets():
    aggregate = _aggregate()
    adapter = TTSCtcTokenizer(aggregate, 'second')
    assert adapter.text_to_ids('one two') == [4, 5, 6]
    aggregate.encode.assert_called_once_with('one two', tokenizer_name='second')
    assert adapter.vocab_size == 8
    assert adapter.ids_to_text([4, 5, 6]) == '<s>one two'
    assert adapter.ids_to_tokens([4, 5, 6]) == ['<s>', 'one', ' two']


def test_tts_ctc_preserves_native_special_ids_without_appending_model_controls():
    aggregate = _aggregate(single=True)
    adapter = TTSCtcTokenizer(aggregate)
    assert adapter.text_to_ids('one two') == [0, 1, 2]
    assert adapter.text_to_ids('') == [0]
    assert adapter.ids_to_text([3]) == '<unk>'
    assert adapter.ids_to_tokens([3]) == ['<unk>']


def test_ctc_predictions_decode_foreign_tokenizer_ids_without_assuming_the_selected_offset():
    aggregate = _aggregate()
    adapter = TTSCtcTokenizer(aggregate, 'second')
    assert adapter.ids_to_text(torch.tensor([1, 2, 5, 6, 3])) == 'one twoone two<unk>'
    assert adapter.ids_to_tokens([1, 2, 5, 6, 3]) == ['one', ' two', 'one', ' two', '<unk>']
    assert aggregate.decode.call_args_list[0].kwargs == dict(tokenizer_name='first')
    assert aggregate.decode.call_args_list[1].kwargs == dict(tokenizer_name='second')
    assert aggregate.decode.call_args_list[2].kwargs == dict(tokenizer_name='first')


def test_tts_ctc_handles_native_base_tokenizer_piece_lookup_and_empty_predictions():
    aggregate = _aggregate(single=True)
    aggregate.tokenizers['first'] = SimpleNamespace(tokens=['<s>', 'one', ' two', '<unk>'])
    adapter = TTSCtcTokenizer(aggregate)
    assert adapter.ids_to_tokens([1, 2]) == ['one', ' two']
    assert adapter.ids_to_text([]) == ''
    assert adapter.ids_to_tokens([]) == []


def test_ctc_blank_and_repeat_collapse_precede_native_tts_decoding():
    adapter = TTSCtcTokenizer(_aggregate(single=True))
    assert decode_ctc_tokens([1, 1, 4, 1, 2], 4, adapter) == 'oneone two'


@pytest.mark.parametrize('token', [-100, -1, 8, 9])
def test_tts_ctc_rejects_external_blank_or_ignored_ids_without_silently_dropping_them(token):
    adapter = TTSCtcTokenizer(_aggregate(), 'second')
    with pytest.raises(ValueError, match='blank and ignored'):
        adapter.ids_to_text([token])
    with pytest.raises(ValueError, match='blank and ignored'):
        adapter.ids_to_tokens([token])


def test_tts_ctc_requires_unambiguous_native_tokenizer_selection():
    with pytest.raises(ValueError, match='multiple input tokenizers'):
        TTSCtcTokenizer(_aggregate())
    with pytest.raises(ValueError, match='Unknown TTS CTC tokenizer'):
        TTSCtcTokenizer(_aggregate(), 'missing')


def test_native_huggingface_tokenizer_ids_and_decoding_match_the_existing_tts_aggregate():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import AggregatedTTSTokenizer

    backend = Tokenizer(models.WordLevel(vocab={'<unk>': 0, 'hello': 3, 'world': 1, '<pad>': 2}, unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    native = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='<unk>', pad_token='<pad>')
    aggregate = AggregatedTTSTokenizer([native, native], ['first', 'second'])
    adapter = TTSCtcTokenizer(aggregate, 'second')
    assert adapter.text_to_ids('hello world') == aggregate.encode('hello world', tokenizer_name='second') == [7, 5]
    assert adapter.ids_to_tokens([7, 5]) == ['hello', 'world']
    assert adapter.ids_to_text([7, 5]) == aggregate.decode([7, 5], tokenizer_name='second') == 'hello world'


@pytest.mark.parametrize('add_bos', [False, True])
def test_multiturn_collector_uses_final_visible_native_ids_without_later_turn_contamination(add_bos):
    from nemo.collections.tts.data.text_to_speech_dataset_lhotse_multiturn import build_token_channel

    cut = _channel_cut(second_start=0.64)
    tokenizer = _ChannelTokenizer()
    turns = []
    values = build_token_channel(
        cut,
        tokenizer,
        frame_length=0.08,
        roles={'agent'},
        tokenizer_name='native',
        pad_id=0,
        bos_id=900,
        eos_id=901,
        interruption_token_id=902,
        add_text_bos=add_bos,
        timestamp_turns=turns,
    )
    expected = build_token_channel(
        cut,
        _ChannelTokenizer(),
        frame_length=0.08,
        roles={'agent'},
        tokenizer_name='native',
        pad_id=0,
        bos_id=900,
        eos_id=901,
        interruption_token_id=902,
        add_text_bos=add_bos,
    )
    torch.testing.assert_close(values, expected, rtol=0, atol=0)
    assert tokenizer.calls == ['first', 'second']
    first, second = turns
    assert first['text_token_start'] == 2 + int(add_bos)
    assert first['text_token_end'] == 10 + int(add_bos)
    assert first['input_token_ids'] == ([1, 2, 3, 4] if add_bos else [1, 2, 3, 4, 5])
    assert second['input_token_ids'] == [101, 102, 103]
    assert 101 in values[first['text_token_start'] : first['text_token_end']]
    assert 101 not in first['input_token_ids']
    assert all(special not in turn['input_token_ids'] for turn in turns for special in (0, 900, 901, 902))


@pytest.mark.parametrize('add_bos', [False, True])
def test_multiturn_collector_trims_native_targets_to_the_actual_text_channel_crop(add_bos):
    from nemo.collections.tts.data.text_to_speech_dataset_lhotse_multiturn import build_token_channel

    turns = []
    values = build_token_channel(
        _channel_cut(second_start=0.96),
        _ChannelTokenizer(),
        frame_length=0.08,
        roles={'agent'},
        tokenizer_name='native',
        pad_id=0,
        bos_id=900,
        eos_id=901,
        interruption_token_id=902,
        add_text_bos=add_bos,
        timestamp_turns=turns,
    )
    second = turns[1]
    assert second['text_token_start'] == 12 + int(add_bos)
    assert second['text_token_end'] == len(values) == 15
    assert second['input_token_ids'] == ([101, 102] if add_bos else [101, 102, 103])
    assert second['input_token_ids'] == values[second['text_token_start'] : second['text_token_end']].tolist()
    assert not second['has_end_marker']


def test_multiturn_collector_retains_old_tokens_that_remain_visible_after_a_short_overwrite():
    from nemo.collections.tts.data.text_to_speech_dataset_lhotse_multiturn import build_token_channel

    tokenizer = _ChannelTokenizer(first_ids=list(range(1, 21)), second_ids=[101])
    turns = []
    values = build_token_channel(
        _channel_cut(second_start=0.64),
        tokenizer,
        frame_length=0.08,
        roles={'agent'},
        tokenizer_name='native',
        pad_id=0,
        bos_id=900,
        eos_id=901,
        interruption_token_id=902,
        add_text_bos=True,
        timestamp_turns=turns,
    )
    assert turns[0]['input_token_ids'] == [1, 2, 3, 4, 9, 10, 11, 12]
    assert turns[1]['input_token_ids'] == [101]
    assert values[11:].tolist() == [9, 10, 11, 12]


@pytest.mark.parametrize('partial_substitution', [False, True])
def test_multiturn_collector_preserves_visible_phoneme_ids_for_native_ctc_turn_rejection(
    monkeypatch, partial_substitution
):
    from nemo.collections.tts.data.text_to_speech_dataset_lhotse_multiturn import build_token_channel

    cut = _channel_cut(second_start=0.96)
    first = cut.supervisions[0]
    first.text = 'first second' if partial_substitution else '<bop>fɜst<eop> second'
    first.language = 'en'
    first.custom = dict(ipa='fɜst sɛkənd', ipa_alignment=[(0, 5, 'first', 'fɜst'), (6, 12, 'second', 'sɛkənd')])
    monkeypatch.setattr('random.sample', lambda population, count: [0])
    tokenizer = _ChannelTokenizer()
    tokenizer.ids[' second'] = [101, 102, 103]
    turns = []
    kwargs = dict(
        frame_length=0.08,
        roles={'agent'},
        tokenizer_name='native',
        pad_id=0,
        bos_id=900,
        eos_id=901,
        interruption_token_id=902,
        add_text_bos=True,
        enable_phoneme_text_input=True,
        phoneme_tokenizer=SimpleNamespace(encode=Mock(return_value=[5, 6])),
        text_phoneme_token_offset=1000,
        apply_partial_phoneme_text=partial_substitution,
        partial_phoneme_text_prob=1.0,
        partial_phoneme_portion_min=0.5,
        partial_phoneme_portion_max=0.5,
    )
    values = build_token_channel(cut, tokenizer, timestamp_turns=turns, **kwargs)
    expected = build_token_channel(cut, tokenizer, **kwargs)
    torch.testing.assert_close(values, expected, rtol=0, atol=0)
    assert turns[0]['input_token_ids'] == [1005, 1006, 101, 102]
    assert turns[0]['input_token_ids'] == values[3:7].tolist()
    assert turns[1]['input_token_ids'] == [101, 102]


class _ChannelTokenizer:
    def __init__(self, first_ids=None, second_ids=None):
        self.calls = []
        self.ids = dict(
            first=list(range(1, 9)) if first_ids is None else first_ids,
            second=[101, 102, 103] if second_ids is None else second_ids,
        )

    def encode(self, text, tokenizer_name):
        assert tokenizer_name == 'native'
        self.calls.append(text)
        return self.ids[text]


def _channel_cut(second_start):
    from lhotse import AudioSource, MonoCut, Recording, SupervisionSegment

    recording = Recording(
        id='recording',
        sources=[AudioSource(type='file', channels=[0], source='unused.wav')],
        sampling_rate=100,
        num_samples=120,
        duration=1.2,
    )
    return MonoCut(
        id='turns',
        start=0.0,
        duration=1.2,
        channel=0,
        recording=recording,
        supervisions=[
            SupervisionSegment(
                id='first', recording_id='recording', start=0.16, duration=0.4, text='first', speaker='agent'
            ),
            SupervisionSegment(
                id='second',
                recording_id='recording',
                start=second_start,
                duration=1.2 - second_start,
                text='second',
                speaker='agent',
            ),
        ],
    )


def _aggregate(single=False):
    pieces = ['<s>', 'one', ' two', '<unk>']
    names = ['first'] if single else ['first', 'second']
    offsets = {name: index * len(pieces) for index, name in enumerate(names)}
    native = {
        name: SimpleNamespace(convert_ids_to_tokens=lambda ids: [pieces[index] for index in ids]) for name in names
    }

    def encode(text, tokenizer_name):
        assert text in ('one two', '')
        ids = [0, 1, 2] if text else [0]
        return [offsets[tokenizer_name] + token for token in ids]

    def decode(ids, tokenizer_name):
        return ''.join(pieces[token - offsets[tokenizer_name]] for token in ids)

    return SimpleNamespace(
        tokenizers=native,
        tokenizer_offsets=offsets,
        tokens=pieces * len(names),
        encode=Mock(side_effect=encode),
        decode=Mock(side_effect=decode),
    )
