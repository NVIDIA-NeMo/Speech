# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from lhotse import CutSet, MonoCut, SupervisionSegment
from lhotse.testing.dummies import dummy_recording

from nemo.collections.common.data.lhotse.sampling import EmptyTextFilter


def make_cut(text='hello', normalized_text=None):
    return MonoCut(
        id='example',
        start=0,
        duration=1,
        channel=0,
        supervisions=[
            SupervisionSegment(
                id='sup',
                recording_id='rec',
                start=0,
                duration=1,
                text=text,
                custom={'normalized_text': normalized_text},
            )
        ],
    )


@pytest.mark.unit
@pytest.mark.parametrize('text', [None, 123, [], {}])
def test_invalid_raw_text(text):
    assert not EmptyTextFilter()(make_cut(text=text))


@pytest.mark.unit
@pytest.mark.parametrize('text', [123, [], {}])
def test_invalid_normalized_text(text):
    assert not EmptyTextFilter()(make_cut(normalized_text=text))


@pytest.mark.unit
@pytest.mark.parametrize('text', [None, '', ' \n', 'Hello!', '  hello  ', '你好'])
def test_valid_normalized_text_or_raw_fallback(text):
    assert EmptyTextFilter()(make_cut(normalized_text=text))


@pytest.mark.unit
def test_any_invalid_turn_rejects_entire_cut():
    cut = make_cut()
    cut.supervisions.append(make_cut(text=None).supervisions[0])
    assert not EmptyTextFilter()(cut)


@pytest.mark.unit
def test_no_supervisions_and_non_audio():
    cut = make_cut()
    cut.supervisions = []
    assert not EmptyTextFilter()(cut)
    assert EmptyTextFilter()('text-only example')


@pytest.mark.unit
def test_filter_before_batching():
    cuts = CutSet.from_cuts([make_cut(), make_cut(text=None), make_cut(text=' ')])
    assert len(list(cuts.filter(EmptyTextFilter()))) == 2


@pytest.mark.unit
@pytest.mark.parametrize('enabled, expected', [(False, 2), (True, 1)])
def test_sampler_config(monkeypatch, enabled, expected):
    from omegaconf import OmegaConf
    from nemo.collections.common.data.lhotse import dataloader

    good, bad = make_cut(), make_cut(text=None)
    bad.id = 'bad'
    good.recording = dummy_recording(0, duration=1)
    bad.recording = dummy_recording(1, duration=1)
    cuts = CutSet.from_cuts([good, bad])
    monkeypatch.setattr(dataloader, 'read_cutset_from_config', lambda config: (cuts, False))
    config = OmegaConf.structured(dataloader.LhotseDataLoadingConfig)
    config.filter_empty_text = enabled
    config.use_bucketing = False
    config.batch_size = 2
    sampler, _ = dataloader.get_lhotse_sampler_from_config(config, global_rank=0, world_size=1)
    assert sum(len(batch) for batch in sampler) == expected


@pytest.mark.unit
@pytest.mark.parametrize('text', ['', ' \t\n'])
def test_annotated_non_speech_turn_is_retained(text):
    cut = make_cut()
    event = make_cut(text=text).supervisions[0]
    event.speaker = 'user'
    event.custom['text_with_annotations'] = '[tap]'
    cut.supervisions.append(event)
    assert EmptyTextFilter()(cut)
