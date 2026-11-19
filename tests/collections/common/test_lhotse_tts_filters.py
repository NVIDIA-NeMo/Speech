# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import replace

import pytest
from lhotse import SupervisionSegment
from lhotse.array import Array, TemporalArray
from lhotse.audio import AudioSource, Recording
from lhotse.cut import MonoCut

from nemo.collections.common.data.lhotse.sampling import (
    CERFilter,
    ContextSpeakerSimilarityFilter,
    SpeakerFilter,
    ValidationStatusFilter,
)


@pytest.fixture
def cut_example():
    cut = MonoCut(
        id='cut-rec-Zdud2gXLTXY-238.16-6.88_repeat0',
        start=238.16,
        duration=6.88,
        channel=0,
        supervisions=[
            SupervisionSegment(
                id='sup-rec-Zdud2gXLTXY',
                recording_id='rec-Zdud2gXLTXY',
                start=238.16,
                duration=6.88,
                channel=0,
                text='and in like manner, as do other parts in which there appears to exist an adaptation to an end.',
                language='en',
                speaker='| Language:en Dataset:nvyt2505 Speaker:Zdud2gXLTXY_SPEAKER_02 |',
                gender=None,
                custom={
                    'cer': 0.03,
                    'bandwidth': 10875,
                    'stoi_squim': 0.921,
                    'sisdr_squim': 15.17,
                    'pesq_squim': 1.845,
                    'dataset_id': '5a6446c5-6114-4380-b875-9de17fda2b8d',
                    'dataset_version': '2024_11_07_131440',
                    'dataset_name': 'yt_mixed',
                    'context_speaker_similarity': 0.9172529578208923,
                    'context_audio_offset': 7001.95659375,
                    'context_audio_duration': 14.64,
                    'context_audio_text': 'Uat gives an excellent illustration of the effects of a course of selection, which may be considered as unconscious, insofar that the breeders could never have expected, or even wished, to produce the result which ensued,',
                    'context_recording_id': 'rec-Zdud2gXLTXY',
                },
                alignment=None,
            )
        ],
        features=None,
        recording=Recording(
            id='rec-Zdud2gXLTXY',
            sources=[AudioSource(type='file', channels=[0], source='/audio/Zdud2gXLTXY.wav')],
            sampling_rate=22050,
            num_samples=952064173,
            duration=43177.51351473923,
            channel_ids=[0],
            transforms=None,
        ),
        custom={
            'validation_status': 'pass',
            'target_audio': Recording(
                id='cut-rec-Zdud2gXLTXY-238.16-6.88',
                sources=[AudioSource(type='memory', channels=[0], source='<binary-data>')],
                sampling_rate=22050,
                num_samples=151704,
                duration=6.88,
                channel_ids=[0],
                transforms=None,
            ),
            'context_audio': Recording(
                id='context_cut-rec-Zdud2gXLTXY-7001.96-14.64',
                sources=[AudioSource(type='memory', channels=[0], source='<binary-data>')],
                sampling_rate=22050,
                num_samples=322812,
                duration=14.64,
                channel_ids=[0],
                transforms=None,
            ),
            'target_codes': TemporalArray(
                array=Array(storage_type='memory_npy', storage_path='', storage_key='<binary-data>', shape=[8, 149]),
                temporal_dim=-1,
                frame_shift=0.046511627906976744,
                start=0,
            ),
            'context_codes': TemporalArray(
                array=Array(storage_type='memory_npy', storage_path='', storage_key='<binary-data>', shape=[8, 316]),
                temporal_dim=-1,
                frame_shift=0.046511627906976744,
                start=0,
            ),
            'shard_origin': '/cuts/cuts.000001.jsonl.gz',
            'shar_epoch': 0,
            'tokenizer_names': ['english_phoneme'],
        },
    )
    return cut


def test_cut_cer_filter(cut_example):
    f = CERFilter(0.4)
    assert f(cut_example) == True

    f = CERFilter(0.01)
    assert f(cut_example) == False

    f = CERFilter(float("inf"))
    assert f(cut_example) == True


@pytest.mark.parametrize(
    'scores,threshold,expected',
    [
        ([{'asr_cer': 0.03}], 0.04, True),
        ([{'asr_cer': 0.03}], 0.02, False),
        ([{'cer': 0.6}, {'cer': 0.0}], 0.4, True),
        ([{'cer': 0.0}, {'cer': 0.9}], 0.4, False),
        ([{'cer': 0.2}, {'asr_cer': 0.6}], 0.4, True),
        ([{'cer': 0.1, 'asr_cer': 0.99}, {'asr_cer': 0.6}], 0.4, True),
        ([{'cer': 0.9, 'asr_cer': 0.0}], 0.4, False),
        ([{'cer': None, 'asr_cer': 0.3}], 0.4, True),
        ([{'cer': 0.0, 'asr_cer': 0.9}], 0.0, True),
        ([{}, {'asr_cer': 0.6}, {'cer': None}], 0.4, False),
        ([{}, {'cer': None, 'asr_cer': None}], 0.0, True),
        ([], 0.0, True),
        ([{'asr_cer': 100.0}], None, True),
    ],
)
def test_cut_cer_filter_averages_available_supervision_scores(cut_example, scores, threshold, expected):
    cut_example.supervisions = [
        SupervisionSegment(
            id=f'sup-{index}',
            recording_id=cut_example.recording_id,
            start=0,
            duration=cut_example.duration,
            channel=0,
            text='test',
            custom=custom,
        )
        for index, custom in enumerate(scores)
    ]
    assert CERFilter(threshold)(cut_example) is expected


def test_cer_filter_applies_to_mixed_cuts(cut_example):
    mixed = cut_example.mix(cut_example.with_id('second-cut'))
    assert CERFilter(0.04)(mixed)
    assert not CERFilter(0.02)(mixed)
    assert CERFilter(0.04, {'en': 0.01})(mixed)


@pytest.mark.parametrize('example', ['text example', None, object()])
def test_cer_filter_passes_non_cut_examples(example):
    assert CERFilter(0.0)(example)
    assert CERFilter(0.0, {'en': 0.0})(example)


@pytest.mark.parametrize(
    'language,default,overrides,expected',
    [
        (None, 0.01, {'en': 0.05}, False),
        ('en', 0.01, {'en': 0.05}, True),
        ('de', 0.01, {'en': 0.05}, False),
        ('en', 0.4, {'en': 0.01}, False),
        ('en', 0.01, {'en': None}, True),
        ('en', None, {'en': 0.01}, False),
        ('en', 0.0, {'en': 0.03}, True),
    ],
)
def test_cut_cer_filter_by_lang(cut_example, language, default, overrides, expected):
    if language is not None:
        cut_example.lang = language
    assert CERFilter(default, overrides)(cut_example) is expected


def test_cut_cer_filter_by_lang_uses_mean_of_all_available_scores(cut_example):
    cut_example.lang = 'en'
    first = cut_example.supervisions[0]
    first.custom['cer'] = 0.0
    second = replace(first, id='second-supervision', custom={'asr_cer': 0.08})
    cut_example.supervisions = [first, second]
    assert CERFilter(0.01, {'en': 0.05})(cut_example)
    assert not CERFilter(0.2, {'en': 0.03})(cut_example)


def test_cut_context_speaker_similarity_filter(cut_example):
    f = ContextSpeakerSimilarityFilter(0.6)
    assert f(cut_example) == True

    f = ContextSpeakerSimilarityFilter(0.95)
    assert f(cut_example) == False

    f = ContextSpeakerSimilarityFilter(-1)
    assert f(cut_example) == True


@pytest.mark.parametrize(
    'language,default,overrides,expected',
    [
        (None, 0.95, {'en': 0.6}, False),
        ('en', 0.95, {'en': 0.6}, True),
        ('de', 0.95, {'en': 0.6}, False),
        ('en', 0.3, {'en': 0.95}, False),
        ('en', 0.95, {'en': None}, True),
        ('en', None, {'en': 0.95}, False),
    ],
)
def test_cut_context_speaker_similarity_filter_by_lang(cut_example, language, default, overrides, expected):
    if language is not None:
        cut_example.lang = language
    assert ContextSpeakerSimilarityFilter(default, overrides)(cut_example) is expected


@pytest.mark.parametrize('score', [None, 'missing'])
def test_cut_context_speaker_similarity_filter_by_lang_passes_missing_scores(cut_example, score):
    cut_example.lang = 'en'
    if score == 'missing':
        del cut_example.supervisions[0].custom['context_speaker_similarity']
    else:
        cut_example.supervisions[0].custom['context_speaker_similarity'] = score
    assert ContextSpeakerSimilarityFilter(1.0, {'en': 1.0})(cut_example)


def test_cut_validation_status_filter(cut_example):
    f = ValidationStatusFilter("pass")
    assert f(cut_example) == True

    f = ValidationStatusFilter("wrong_text")
    assert f(cut_example) == False

    f = ValidationStatusFilter("any_other_status")
    assert f(cut_example) == False


@pytest.mark.parametrize(
    'keep,status,expected',
    [
        ('pass', 'pass', True),
        (['pass', 'fail_ssim'], 'fail_ssim', True),
        (['wrong_text', 'fail_ssim'], 'pass', False),
        ([], 'pass', False),
        (('pass', 'fail_ssim'), 'pass', True),
    ],
)
def test_cut_validation_status_filter_multiple_statuses(cut_example, keep, status, expected):
    cut_example.validation_status = status
    assert ValidationStatusFilter(keep)(cut_example) is expected


def test_cut_validation_status_filter_multiple_statuses_passes_missing_status(cut_example):
    del cut_example.custom['validation_status']
    assert ValidationStatusFilter([])(cut_example)


@pytest.mark.parametrize('example', ['text example', None, object()])
def test_quality_filters_pass_non_cut_examples(example):
    assert ContextSpeakerSimilarityFilter(1.0, {'en': 1.0})(example)
    assert ValidationStatusFilter(['pass', 'fail_ssim'])(example)


def test_cut_speaker_filter_by_speaker(cut_example):
    f = SpeakerFilter(
        excluded_speaker_ids=["Zdud2gXLTXY_SPEAKER_02"],
        speaker_fields=["speaker"],
    )
    assert f(cut_example) == False

    f = SpeakerFilter(
        excluded_speaker_ids=["some_other_speaker"],
        speaker_fields=["speaker"],
    )
    assert f(cut_example) == True


def test_cut_speaker_filter_by_custom_speaker_id(cut_example):
    cut_example.supervisions[0].custom["speaker_id"] = "test_speaker_001"

    f = SpeakerFilter(
        excluded_speaker_ids=["test_speaker_001"],
        speaker_fields=["speaker_id"],
    )
    assert f(cut_example) == False

    f = SpeakerFilter(
        excluded_speaker_ids=["some_other_speaker"],
        speaker_fields=["speaker_id"],
    )
    assert f(cut_example) == True


def test_cut_speaker_filter_multiple_fields(cut_example):
    cut_example.supervisions[0].custom["speaker_id"] = "test_speaker_001"

    f = SpeakerFilter(
        excluded_speaker_ids=["test_speaker_001"],
        speaker_fields=["speaker", "speaker_id"],
    )
    assert f(cut_example) == False

    f = SpeakerFilter(
        excluded_speaker_ids=["some_other_speaker"],
        speaker_fields=["speaker", "speaker_id"],
    )
    assert f(cut_example) == True


def test_cut_speaker_filter_disabled(cut_example):
    f = SpeakerFilter(
        excluded_speaker_ids=None,
        speaker_fields=["speaker_id"],
    )
    assert f(cut_example) == True

    f = SpeakerFilter(
        excluded_speaker_ids=[],
        speaker_fields=["speaker_id"],
    )
    assert f(cut_example) == True


def test_cut_speaker_filter_requires_fields_when_enabled():
    with pytest.raises(ValueError):
        SpeakerFilter(
            excluded_speaker_ids=["test_speaker_001"],
            speaker_fields=None,
        )
