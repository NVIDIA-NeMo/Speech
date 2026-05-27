# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import pytest
from lhotse import SupervisionSegment
from lhotse.array import Array, TemporalArray
from lhotse.audio import AudioSource, Recording
from lhotse.cut import MonoCut

from nemo.collections.common.data.lhotse.sampling import (
    CERFilter,
    ContextSpeakerSimilarityFilter,
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


def test_cut_context_speaker_similarity_filter(cut_example):
    f = ContextSpeakerSimilarityFilter(0.6)
    assert f(cut_example) == True

    f = ContextSpeakerSimilarityFilter(0.95)
    assert f(cut_example) == False

    f = ContextSpeakerSimilarityFilter(-1)
    assert f(cut_example) == True


def test_cut_validation_status_filter(cut_example):
    f = ValidationStatusFilter("pass")
    assert f(cut_example) == True

    f = ValidationStatusFilter("wrong_text")
    assert f(cut_example) == False

    f = ValidationStatusFilter("any_other_status")
    assert f(cut_example) == False
