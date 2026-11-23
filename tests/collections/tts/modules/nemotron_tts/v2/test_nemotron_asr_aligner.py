# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nemo.collections.tts.modules.nemotron_tts.v2 import nemotron_asr_aligner as module

pytestmark = pytest.mark.unit


def test_teacher_stays_frozen_eval_and_preserves_precision(monkeypatch):
    teacher, components, _ = _teacher(monkeypatch)
    parent = nn.Module()
    parent.teacher = teacher
    originals = {name: tensor.clone() for name, tensor in teacher.state_dict().items()}

    parent.train().half()

    assert not teacher.training
    assert all(not child.training for child in teacher.modules())
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert components.encoder.weight.dtype == torch.bfloat16
    assert components.encoder.critical.dtype == torch.float32
    assert components.decoder.weight.dtype == torch.float32
    for name, tensor in teacher.state_dict().items():
        assert tensor.dtype == originals[name].dtype
        torch.testing.assert_close(tensor, originals[name], rtol=0, atol=0)


def test_forward_isolates_default_dtype_and_preserves_nested_bf16_encoder(monkeypatch):
    teacher, components, calls = _teacher(monkeypatch)
    audio, lengths = torch.ones(1, 8, dtype=torch.float32), torch.tensor([8], dtype=torch.long)
    previous = torch.get_default_dtype()
    resample_defaults = []
    original_resample = module._resample_turns

    def resample(*args, **kwargs):
        resample_defaults.append(torch.get_default_dtype())
        return original_resample(*args, **kwargs)

    monkeypatch.setattr(module, '_resample_turns', resample)
    try:
        for dtype in (torch.float32, torch.bfloat16):
            torch.set_default_dtype(dtype)
            teacher(audio, lengths, 16000, ['same waveform'])
            assert torch.get_default_dtype() == dtype
            assert components.preprocessor.default_dtype == torch.float32
            assert components.encoder.default_dtype == torch.float32
            assert components.encoder.autocast_enabled
            assert components.encoder.weight.dtype == torch.bfloat16
        with pytest.raises(ValueError):
            teacher(audio.unsqueeze(0), lengths, 16000, ['invalid shape'])
        assert torch.get_default_dtype() == torch.bfloat16
    finally:
        torch.set_default_dtype(previous)
    assert resample_defaults == [torch.float32, torch.float32]
    assert all(call['default_dtype'] == torch.float32 for call in calls)


def test_ctc_sdpa_matches_native_positions_and_padding_without_changing_weights():
    from nemo.collections.speechlm2.parts.ctc_timestamp_utils import TransformerCTCDecoder

    torch.manual_seed(1234)
    decoder = (
        TransformerCTCDecoder(
            feat_in=64,
            num_classes=4,
            n_heads=2,
            n_layers=1,
            drop_rate=0.0,
            qk_norm=True,
            rotary_fraction=0.5,
            upsampling_factor=2,
            residual=True,
            residual_scale=0.1,
        )
        .float()
        .eval()
    )
    states, lengths = torch.randn(2, 64, 5), torch.tensor([5, 3])
    parameters = {id(parameter): parameter.detach().clone() for parameter in decoder.parameters()}
    with torch.inference_mode():
        expected = decoder(states, encoded_lengths=lengths)
        for layer in decoder.transformer.layers:
            layer.attn = module._CTCSDPAAttention(layer.attn)
        actual = decoder(states, encoded_lengths=lengths)
        states[1, :, 3:] = 1000.0
        padded = decoder(states, encoded_lengths=lengths)

    assert set(parameters) == {id(parameter) for parameter in decoder.parameters()}
    for parameter in decoder.parameters():
        assert parameter.dtype == torch.float32
        torch.testing.assert_close(parameter, parameters[id(parameter)], rtol=0, atol=0)
    for index, count in enumerate(lengths.tolist()):
        torch.testing.assert_close(actual[index, : 2 * count], expected[index, : 2 * count], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(padded[index, : 2 * count], actual[index, : 2 * count], rtol=1e-5, atol=1e-6)


def test_teacher_uses_ground_truth_lengths_and_separate_language_policies(monkeypatch):
    teacher, components, calls = _teacher(monkeypatch, measure_timings=True)
    audio = torch.arange(24, dtype=torch.bfloat16).reshape(3, 8).requires_grad_()
    # The invalid record is removed before the acoustic encoder runs.
    audio.data[1, 4:] = 1000
    records = teacher(audio, torch.tensor([8, 4, 0]), 16000, ['hello', '世界', 'empty'], ['en', 'zh-CN', 'en'])

    assert len(records) == 3
    assert [record['error'] for record in records] == [None, None, 'invalid_audio_length']
    assert records[0]['words'][0]['word'] == 'hello'
    assert records[1]['words'][0]['word'] == '世界'
    assert [call['language'] for call in calls] == ['whitespace', 'zh']
    assert [call['durations'] for call in calls] == [[8 / 16000], [4 / 16000]]
    assert all(call['inference'] and not call['grad_enabled'] for call in calls)
    assert all(call['refine_word_timestamps'] is False for call in calls)
    assert components.preprocessor.seen.dtype == torch.float32
    torch.testing.assert_close(components.preprocessor.seen[1], torch.tensor([8, 9, 10, 11, 0, 0, 0, 0]).float())
    assert components.preprocessor.lengths.tolist() == [8, 4]
    assert records[0]['timings']['total'] >= records[0]['timings']['encoder'] >= 0
    assert records[0]['timings'] == records[1]['timings']
    assert audio.grad is None


def test_bad_transcript_does_not_discard_other_turns(monkeypatch):
    teacher, _, calls = _teacher(monkeypatch)
    records = teacher(torch.ones(3, 8), torch.tensor([8, 8, 8]), 16000, ['good', 'bad', '   '])

    assert records[0]['words'][0]['word'] == 'good'
    assert records[0]['error'] is None
    assert records[1] == {'words': [], 'error': 'ValueError: unrepresentable transcript'}
    assert records[2] == {'words': [], 'error': 'empty_transcript'}
    assert len(calls) == 3  # One batch preparation, followed by one per valid record.


def test_encoder_microbatches_preserve_record_order_and_aggregate_timings(monkeypatch):
    teacher, _, calls = _teacher(monkeypatch, max_batch_size=2, measure_timings=True)
    records = teacher(torch.ones(3, 8), torch.tensor([8, 8, 8]), 16000, ['first', 'second', 'third'])

    assert [record['words'][0]['word'] for record in records] == ['first', 'second', 'third']
    assert [len(call['durations']) for call in calls] == [2, 1]
    assert all(record['timings'] == teacher.last_timings for record in records)
    assert teacher.last_timings['total'] >= teacher.last_timings['encoder']
    assert module._turn_batches([16000, 16000, 32000, 16000], 16000, 10, 3.0) == [(0, 2), (2, 3), (3, 4)]


def test_shared_whitespace_policy_batches_languages_and_keeps_special_policies(monkeypatch):
    teacher, _, calls = _teacher(monkeypatch)
    teacher(
        torch.ones(7, 8),
        torch.full((7,), 8),
        16000,
        ['English', 'Portuguese', 'Spanish', '中文', '日本語', 'ไทย', 'unspecified'],
        ['en', 'pt-BR', 'es', 'zh', 'ja', 'th', None],
    )
    assert [call['language'] for call in calls] == ['whitespace', 'zh', 'ja', 'th', None]
    assert [len(call['durations']) for call in calls] == [3, 1, 1, 1, 1]
    assert all(aligner._CTC_WINDOW_MAX_ELEMENTS == 128 * 2**20 for aligner in teacher._aligners.values())


def test_reference_grouping_and_ctc_budget_can_be_restored(monkeypatch):
    teacher, _, calls = _teacher(
        monkeypatch, group_languages_by_policy=False, use_cuda_alignment=False, ctc_max_batch_elements=16 * 2**20
    )
    teacher(torch.ones(3, 8), torch.full((3,), 8), 16000, ['one', 'dois', 'tres'], ['en', 'pt', 'es'])
    assert [call['language'] for call in calls] == ['en', 'pt', 'es']
    assert all(aligner._CTC_WINDOW_MAX_ELEMENTS == 16 * 2**20 for aligner in teacher._aligners.values())


def test_alignment_helper_forwards_cuda_device_without_changing_public_spans():
    seen = []
    native = {'speaker_word_timestamps': {0: [{'word': 'hello', 'start': 0.2, 'end': 0.6}]}}

    def align(prepared, *, device):
        seen.append(device)
        return [native]

    prepared = {'records': [{'streams': [{'words': ['hello']}]}]}
    result = module._checked_alignment_results(SimpleNamespace(align_prepared=align), prepared, [1.0], device='cuda:0')
    assert seen == ['cuda:0']
    assert result[0]['words'][0]['start'] == 0.0
    assert result[0]['words'][0]['end'] == 1.0


def test_resampling_trims_each_turn_and_keeps_fp32(monkeypatch):
    calls = []

    def resample(waveform, orig_freq, new_freq):
        calls.append((waveform.clone(), orig_freq, new_freq))
        assert waveform.dtype == torch.float32
        return waveform[::2]

    monkeypatch.setitem(sys.modules, 'torchaudio.functional', SimpleNamespace(resample=resample))
    audio = torch.tensor([[1, 2, 3, 4, 90, 90], [5, 6, 7, 8, 9, 10]], dtype=torch.bfloat16)

    waveforms, lengths = module._resample_turns(audio, [4, 6], [0, 1], 32000)

    assert [call[0].tolist() for call in calls] == [[1, 2, 3, 4], [5, 6, 7, 8, 9, 10]]
    assert all(call[1:] == (32000, 16000) for call in calls)
    assert waveforms.dtype == torch.float32
    assert waveforms.tolist() == [[1, 3, 0], [5, 7, 9]]
    assert lengths.tolist() == [2, 3]


@pytest.mark.parametrize('policy', ['checkpoint', 'bfloat16', 'float32'])
def test_selective_checkpoint_loading_never_reads_llm_and_preserves_fp32(monkeypatch, policy):
    encoder = _Encoder()
    checkpoint = {
        'perception.encoder.asr_encoder.weight': torch.tensor([2.0], dtype=torch.bfloat16),
        'perception.encoder.asr_encoder.critical': torch.tensor([1.0001], dtype=torch.float32),
        'llm.very_large_weight': torch.zeros(1),
        'perception.encoder.diarization_model.weight': torch.zeros(1),
    }
    reads = []

    class Handle:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def keys(self):
            return checkpoint.keys()

        def get_tensor(self, key):
            assert key.startswith('perception.encoder.asr_encoder.')
            reads.append(key)
            return checkpoint[key]

    monkeypatch.setitem(sys.modules, 'safetensors', SimpleNamespace(safe_open=lambda *args, **kwargs: Handle()))

    module._load_selected_weights(encoder, 'model.safetensors', 'perception.encoder.asr_encoder.', policy)

    assert set(reads) == {
        'perception.encoder.asr_encoder.weight',
        'perception.encoder.asr_encoder.critical',
    }
    assert encoder.weight.dtype == (torch.float32 if policy == 'float32' else torch.bfloat16)
    assert encoder.critical.dtype == torch.float32
    torch.testing.assert_close(encoder.critical, checkpoint['perception.encoder.asr_encoder.critical'], rtol=0, atol=0)


def test_failed_ctc_path_is_reported_and_native_metadata_is_retained():
    native = {
        'speaker_word_timestamps': {},
        'alignment_diagnostics': {
            'failed_streams': [{'speaker': 0, 'reason': 'no_legal_path_in_anchor_band', 'words': ['missing', 'words']}]
        },
    }

    result = module._public_result(native, 1.0)

    assert result == {'words': [], 'error': 'no_legal_ctc_path', 'alignment': native}


def test_alignment_cardinality_and_missing_units_are_never_silently_accepted():
    prepared = {'records': [{'streams': [{'words': ['first', 'second']}]}]}
    with pytest.raises(RuntimeError, match='0 records for 1 input turns'):
        module._checked_alignment_results(SimpleNamespace(align_prepared=lambda prepared: []), prepared, [1.0])
    native = {
        'speaker_word_timestamps': {0: [{'word': 'first', 'start': 0.0, 'end': 0.1}]},
        'alignment_diagnostics': {'failed_streams': []},
    }
    result = module._checked_alignment_results(
        SimpleNamespace(align_prepared=lambda prepared: [native]), prepared, [1.0]
    )
    assert result[0]['error'] == 'incomplete_alignment'
    assert result[0]['words'] == native['speaker_word_timestamps'][0]


def test_word_intervals_include_leading_silence_and_end_at_next_onset_or_exact_audio_duration():
    native = {
        'speaker_word_timestamps': {
            0: [
                {'word': 'first', 'start': 0.08, 'end': 0.12},
                {'word': 'second', 'start': 0.64, 'end': 0.72},
                {'word': 'last', 'start': 0.94, 'end': 1.04},
            ]
        },
        'alignment_diagnostics': {'failed_streams': []},
    }
    result = module._public_result(native, 1.35, expected_words=3)

    assert result['error'] is None
    assert [word['start'] for word in result['words']] == [0.0, 0.64, 0.94]
    assert [word['end'] for word in result['words']] == [0.64, 0.94, 1.35]
    assert [word['ctc_start'] for word in result['words']] == [0.08, 0.64, 0.94]
    assert [word['ctc_end'] for word in result['words']] == [0.12, 0.72, 1.04]
    assert result['alignment'] is native
    assert [word['end'] for word in native['speaker_word_timestamps'][0]] == [0.12, 0.72, 1.04]
    assert not any('ctc_start' in word or 'ctc_end' in word for word in native['speaker_word_timestamps'][0])


def test_single_word_includes_all_leading_and_trailing_audio():
    native = {'speaker_word_timestamps': {0: [{'word': 'only', 'start': 0.5, 'end': 0.6}]}}

    result = module._public_result(native, 2.0)

    assert result['error'] is None
    assert result['words'] == [{'word': 'only', 'start': 0.0, 'end': 2.0, 'ctc_start': 0.5, 'ctc_end': 0.6}]


@pytest.mark.parametrize(
    'intervals,duration',
    [
        ([(0.1, 0.2), (0.1, 0.3)], 1.0),
        ([(0.6, 0.7), (0.2, 0.3)], 1.0),
        ([(float('nan'), 0.2)], 1.0),
        ([(float('inf'), 0.2)], 1.0),
        ([(-0.1, 0.2)], 1.0),
        ([(1.0, 1.1)], 1.0),
        ([(0.1, 0.1)], 1.0),
        ([(0.1, float('inf'))], 1.0),
        ([(0.1, 1.2)], 1.0),
        ([(0.1, 0.2)], 0.0),
        ([(0.1, 0.2)], float('nan')),
    ],
)
def test_invalid_or_duplicate_word_onsets_are_masked_without_fabricated_boundaries(intervals, duration):
    native = {
        'speaker_word_timestamps': {
            0: [{'word': str(index), 'start': start, 'end': end} for index, (start, end) in enumerate(intervals)]
        }
    }

    result = module._public_result(native, duration)

    assert result['error'] == 'invalid_word_timestamps'
    assert result['words'] == []
    assert result['alignment'] is native


def test_native_end_roundoff_does_not_move_exact_final_boundary():
    native = {'speaker_word_timestamps': {0: [{'word': 'last', 'start': 0.4, 'end': 1.0 + 1e-15}]}}

    result = module._public_result(native, 1.0)

    assert result['error'] is None
    assert result['words'][0]['end'] == 1.0
    assert result['words'][0]['ctc_end'] == 1.0 + 1e-15


@pytest.mark.parametrize('language,expected', [('und', None), ('auto', None), ('zh-Hant', 'zh'), ('Japanese', 'ja')])
def test_language_policy_aliases(language, expected):
    assert module._timestamp_language(language) == expected


@pytest.mark.parametrize(
    'audio,lengths,rate,texts,languages',
    [
        (torch.zeros(1, 1, 4), torch.tensor([4]), 16000, ['hi'], None),
        (torch.zeros(1, 4), torch.tensor([4.0]), 16000, ['hi'], None),
        (torch.zeros(1, 4), torch.tensor([4]), 0, ['hi'], None),
        (torch.zeros(1, 4), torch.tensor([4]), 16000, [], None),
        (torch.zeros(1, 4), torch.tensor([4]), 16000, ['hi'], []),
    ],
)
def test_batch_validation(monkeypatch, audio, lengths, rate, texts, languages):
    teacher, _, _ = _teacher(monkeypatch)
    with pytest.raises(ValueError):
        teacher(audio, lengths, rate, texts, languages)


class _Preprocessor(nn.Module):
    def forward(self, input_signal, length):
        assert torch.is_inference_mode_enabled() and not self.training
        self.seen = input_signal
        self.lengths = length
        self.default_dtype = torch.get_default_dtype()
        return input_signal.unsqueeze(1), length


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
        self.critical = nn.Parameter(torch.tensor([1.0001], dtype=torch.float32))

    def forward(self, audio_signal, length):
        assert torch.is_inference_mode_enabled() and not self.training
        self.default_dtype = torch.get_default_dtype()
        self.autocast_enabled = torch.is_autocast_enabled(audio_signal.device.type)
        return audio_signal * self.weight, length


def _teacher(monkeypatch, **config):
    decoder = nn.Linear(1, 1).float()
    with torch.no_grad():
        decoder.weight.fill_(1.0001)
    components = module._TeacherComponents(
        preprocessor=_Preprocessor(),
        encoder=_Encoder(),
        decoder=decoder,
        tokenizer=object(),
        normalize_type=None,
        alignment_options={},
        info={},
    )
    calls = []

    class Aligner:
        def __init__(self, options):
            self.options = options

        def prepare_from_inputs(self, inputs, texts, durations):
            calls.append(
                {
                    'language': self.options.get('timestamp_language'),
                    'durations': durations,
                    'inference': torch.is_inference_mode_enabled(),
                    'grad_enabled': torch.is_grad_enabled(),
                    'refine_word_timestamps': self.options.get('refine_word_timestamps'),
                    'default_dtype': torch.get_default_dtype(),
                }
            )
            if 'bad' in texts:
                raise ValueError('unrepresentable transcript')
            return {
                'texts': texts,
                'durations': durations,
                'records': [{'streams': [{'words': [text]}]} for text in texts],
            }

        def align_prepared(self, prepared):
            texts, durations = prepared['texts'], prepared['durations']
            return [
                {
                    'speaker_word_timestamps': {0: [{'word': text, 'start': 0.0, 'end': duration}]},
                    'alignment_diagnostics': {'failed_streams': []},
                }
                for text, duration in zip(texts, durations)
            ]

    monkeypatch.setattr(module, '_load_teacher_components', lambda config: components)
    monkeypatch.setattr(module, '_make_aligner', lambda decoder, tokenizer, options: Aligner(options))
    teacher = module.NemotronASRTimestampAligner(config)
    return teacher, components, calls
