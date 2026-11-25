# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from nemo.collections.tts.modules.nemotron_tts.v2.timestamp_head import (
    TTSCtcTokenizer,
    decode_ctc_tokens,
    map_tts_ctc_word_intervals,
)

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
    intervals = map_tts_ctc_word_intervals(adapter, 'hello world', [7, 5], _words('hello', 'world'), 1.0)
    assert intervals['error'] is None
    assert intervals['segments'] == [
        dict(token_start=0, token_end=1, start=0.0, end=0.5),
        dict(token_start=1, token_end=2, start=0.5, end=1.0),
    ]


def test_word_intervals_preserve_full_input_ids_and_affiliate_punctuation_without_retokenizing_words():
    adapter, ids, native = _interval_tokenizer([(0, 2), (2, 3), (3, 9), (9, 10)])
    mapped = map_tts_ctc_word_intervals(adapter, 'Hi, world!', ids, _words('Hi,', 'world!'), 1.0)
    assert mapped == dict(
        segments=[
            dict(token_start=0, token_end=2, start=0.0, end=0.5),
            dict(token_start=2, token_end=4, start=0.5, end=1.0),
        ],
        error=None,
        token_coverage=1.0,
        merged_word_groups=0,
    )
    native.assert_called_once_with('Hi, world!', return_offsets_mapping=True, add_special_tokens=False)
    adapter.tokenizer.encode.assert_not_called()


def test_word_intervals_merge_connected_cross_word_bpes_without_duplicate_target_occurrences():
    adapter, ids, _ = _interval_tokenizer([(0, 3), (2, 5)])
    mapped = map_tts_ctc_word_intervals(adapter, 'a b c', ids, _words('a', 'b', 'c'), 1.5)
    assert mapped['error'] is None
    assert mapped['segments'] == [dict(token_start=0, token_end=2, start=0.0, end=1.5)]
    assert mapped['merged_word_groups'] == 2


def test_word_intervals_map_unicode_normalization_and_character_units_to_original_offsets():
    adapter, ids, _ = _interval_tokenizer([(0, 4), (4, 6), (6, 7)])
    mapped = map_tts_ctc_word_intervals(adapter, 'CAFÉ 中文', ids, _words('cafe\u0301', '中', '文'), 1.5)
    assert mapped['error'] is None
    assert [(segment['token_start'], segment['token_end']) for segment in mapped['segments']] == [
        (0, 1),
        (1, 2),
        (2, 3),
    ]


def test_word_intervals_keep_raw_number_tokens_and_do_not_expand_or_retokenize_numbers():
    adapter, ids, _ = _interval_tokenizer([(0, 3), (3, 9), (9, 15), (15, 16)])
    mapped = map_tts_ctc_word_intervals(adapter, '$10 costs 1,000.', ids, _words('$10', 'costs', '1,000.'), 1.5)
    assert mapped['error'] is None
    assert [(segment['token_start'], segment['token_end']) for segment in mapped['segments']] == [
        (0, 1),
        (1, 2),
        (2, 4),
    ]
    mismatched = map_tts_ctc_word_intervals(
        adapter, '$10 costs 1,000.', ids, _words('ten', 'dollars', 'costs', 'one', 'thousand'), 2.5
    )
    assert mismatched['error'] == 'transcript_mismatch'
    assert mismatched['segments'] == []
    assert mismatched['token_coverage'] == 0.0


def test_word_intervals_keep_repeated_bpe_occurrences_as_separate_targets():
    adapter, ids, _ = _interval_tokenizer([(0, 2), (3, 5)], local_ids=[1, 1])
    mapped = map_tts_ctc_word_intervals(adapter, 'ha ha', torch.tensor(ids), _words('ha', 'ha'), 1.0)
    assert mapped['error'] is None
    assert [(segment['token_start'], segment['token_end']) for segment in mapped['segments']] == [(0, 1), (1, 2)]


def test_word_intervals_reject_actual_input_crops_instead_of_guessing_token_correspondence():
    adapter, ids, _ = _interval_tokenizer([(0, 2), (3, 5)])
    mapped = map_tts_ctc_word_intervals(adapter, 'ha ha', ids[:1], _words('ha', 'ha'), 1.0)
    assert mapped['error'] == 'input_token_mismatch'
    assert mapped['segments'] == []


def test_word_intervals_skip_native_tokenizers_without_character_offsets():
    adapter = TTSCtcTokenizer(_aggregate(single=True))
    mapped = map_tts_ctc_word_intervals(adapter, 'one two', [0, 1, 2], _words('one', 'two'), 1.0)
    assert mapped['error'] == 'offsets_unavailable'


@pytest.mark.parametrize('offsets', [[(0, 0)], [(0, 5)], [(1, 2), (0, 1)]])
def test_word_intervals_reject_invalid_native_character_offsets(offsets):
    adapter, ids, _ = _interval_tokenizer(offsets)
    mapped = map_tts_ctc_word_intervals(adapter, 'a b', ids, _words('a', 'b'), 1.0)
    assert mapped['error'] == 'invalid_token_offsets'


@pytest.mark.parametrize('offsets', [None, 1, [None], [1], ['01'], [(0.0, 1.0)], [(False, True)]])
def test_word_intervals_skip_malformed_native_offsets_without_raising(offsets):
    adapter, ids, native = _interval_tokenizer([(0, 1)])
    native.return_value['offset_mapping'] = offsets
    mapped = map_tts_ctc_word_intervals(adapter, 'a', ids, _words('a'), 0.5)
    assert mapped['error'] == 'invalid_token_offsets'
    assert mapped['segments'] == []


def test_word_intervals_propagate_runtime_failures_instead_of_treating_them_as_bad_alignment():
    adapter, ids, native = _interval_tokenizer([(0, 1)])
    native.side_effect = RuntimeError('CUDA out of memory')
    with pytest.raises(RuntimeError, match='out of memory'):
        map_tts_ctc_word_intervals(adapter, 'a', ids, _words('a'), 0.5)


def test_word_intervals_reject_words_not_covered_by_any_native_token_offset():
    adapter, ids, _ = _interval_tokenizer([(0, 1)])
    mapped = map_tts_ctc_word_intervals(adapter, 'a b', ids, _words('a', 'b'), 1.0)
    assert mapped['error'] == 'uncovered_word'


@pytest.mark.parametrize('times', [[(0.0, 0.0)], [(0.0, float('nan'))], [(0.0, 0.7), (0.5, 1.0)]])
def test_word_intervals_reject_invalid_or_overlapping_teacher_times(times):
    adapter, ids, _ = _interval_tokenizer([(0, 1)])
    words = [dict(word='a', start=start, end=end) for start, end in times]
    mapped = map_tts_ctc_word_intervals(adapter, 'a', ids, words, 1.0)
    assert mapped['error'] == 'invalid_word_times'


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


def _interval_tokenizer(offsets, local_ids=None):
    local_ids = list(range(1, len(offsets) + 1)) if local_ids is None else local_ids
    native = Mock(return_value=dict(input_ids=local_ids, offset_mapping=offsets))
    aggregate = SimpleNamespace(
        tokenizers={'native': native},
        tokenizer_offsets={'native': 17},
        tokens=[''] * (18 + max(local_ids)),
        encode=Mock(),
        decode=Mock(),
    )
    return TTSCtcTokenizer(aggregate, 'native'), [token + 17 for token in local_ids], native


def _words(*words):
    return [dict(word=word, start=index * 0.5, end=(index + 1) * 0.5) for index, word in enumerate(words)]
