# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import io
import json
import tarfile
from collections import Counter
from copy import deepcopy
from itertools import islice
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from lhotse import CutSet
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.data.lhotse.cutset import read_cutset_from_config


class IdentityDataset:
    def __getitem__(self, cuts):
        return cuts


@pytest.fixture
def hq_manifest(tmp_path):
    """A real tarred source with conflicting text fields and non-boolean HQ flags."""
    wav = io.BytesIO()
    sf.write(wav, np.zeros(1600, dtype=np.float32), 16000, format='WAV')
    flags = [True, False, None, 'true', 1, True, True]
    rows = []
    tar_path = tmp_path / 'audio_0.tar'
    with tarfile.open(tar_path, 'w') as archive:
        for idx, flag in enumerate(flags):
            name = f'{idx}.wav'
            data = wav.getvalue()
            (tmp_path / name).write_bytes(data)
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
            row = dict(audio_filepath=name, duration=0.1, shard_id=0, text='Wrong target.', tn_raw='Right target.')
            if flag is not None:
                row['high_quality'] = flag
            if idx == 5:
                row['tn_raw'] = '  '
            if idx == 6:
                row['_skipme'] = True
            rows.append(row)
    path = tmp_path / 'manifest_0.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return dict(manifest_filepath=str(path), tarred_audio_filepaths=str(tar_path), text_field='tn_raw')


@pytest.mark.parametrize('nested', [False, True])
def test_strict_hq_original_tarred_manifest_zero_workers(hq_manifest, nested):
    config = dict(
        num_workers=0,
        batch_size=2,
        force_finite=True,
        keep_high_quality_only=True,
        skip_empty_text=True,
        shuffle=False,
        seed=42,
        shard_seed=42,
    )
    if nested:
        config['input_cfg'] = [dict(type='group', input_cfg=[dict(type='nemo_tarred', **hq_manifest)])]
    else:
        config.update(hq_manifest)
    loader = get_lhotse_dataloader_from_config(config, global_rank=0, world_size=1, dataset=IdentityDataset())
    assert loader.num_workers == 0
    cuts = [cut for batch in loader for cut in batch]
    assert len(cuts) == 1
    assert cuts[0].custom['high_quality'] is True
    assert cuts[0].supervisions[0].text == 'Right target.'
    assert cuts[0].load_audio().shape == (1, 1600)


def test_hq_filter_is_lazy_and_disabled_by_default(hq_manifest):
    cuts, _ = read_cutset_from_config(OmegaConf.create(dict(**hq_manifest, force_finite=True)))
    assert isinstance(cuts, CutSet)
    assert cuts.is_lazy
    assert len(list(cuts)) == 6  # only the explicit _skipme row is removed


def test_hq_filter_runs_before_tokenization_and_noise(hq_manifest, monkeypatch):
    from nemo.collections.common.data.lhotse import dataloader as module

    calls = []

    def tokenize(cut, tokenizer, **kwargs):
        assert cut.custom['high_quality'] is True
        assert cut.supervisions[0].text.strip()
        calls.append(cut.id)
        return cut

    monkeypatch.setattr(module, 'tokenize', tokenize)
    loader = get_lhotse_dataloader_from_config(
        dict(
            **hq_manifest,
            num_workers=0,
            batch_size=2,
            force_finite=True,
            keep_high_quality_only=True,
            skip_empty_text=True,
            noise_path=dict(manifest_filepath=hq_manifest['manifest_filepath']),
            noise_mix_prob=1.0,
            pretokenize=True,
            seed=42,
            shard_seed=42,
        ),
        global_rank=0,
        world_size=1,
        dataset=IdentityDataset(),
        tokenizer=lambda text: [1],
    )
    cuts = [cut for batch in loader for cut in batch]
    assert len(calls) == 1
    assert len(cuts) == 1


@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('partition', [1, 5])
def test_indexed_rejected_partition_terminates(hq_manifest, monkeypatch, nested, partition):
    """A rejected-only worker slice must not loop inside an infinite source."""
    from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoTarredIterator

    monkeypatch.setenv('USE_AIS_GET_BATCH', 'true')
    monkeypatch.setattr('lhotse.dataset.dataloading.get_worker_partition', lambda: (partition, 8))
    decode = LazyNeMoTarredIterator._decode_cut_at
    reads = []

    def bounded_decode(self, index):
        reads.append(index)
        assert len(reads) <= 8, 'The rejected indexed partition is being repeated indefinitely'
        return decode(self, index)

    monkeypatch.setattr(LazyNeMoTarredIterator, '_decode_cut_at', bounded_decode)
    config = dict(
        indexed=True,
        input_sampling_rate=16000,
        keep_high_quality_only=True,
        skip_empty_text=True,
        shuffle=False,
        shard_seed=42,
    )
    if nested:
        config['input_cfg'] = [dict(type='group', input_cfg=[dict(type='nemo_tarred', **hq_manifest)])]
    else:
        config.update(hq_manifest)
    cuts, _ = read_cutset_from_config(OmegaConf.create(config))
    with pytest.raises(StopIteration):
        next(iter(cuts))
    assert reads == [partition]


@pytest.mark.parametrize('layout', ['group', 'weighted_list'])
@pytest.mark.parametrize('has_eligible_source', [False, True])
def test_indexed_mixture_handles_rejected_sources(hq_manifest, monkeypatch, tmp_path, layout, has_eligible_source):
    """An unavailable source must not block an eligible source in the mix."""
    from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoTarredIterator

    monkeypatch.setenv('USE_AIS_GET_BATCH', 'true')
    monkeypatch.setattr('lhotse.dataset.dataloading.get_worker_partition', lambda: (1, 8))
    rows = [json.loads(line) for line in Path(hq_manifest['manifest_filepath']).read_text().splitlines()]
    rows[1]['high_quality'] = has_eligible_source
    live_path = tmp_path / 'live.jsonl'
    live_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    config = dict(
        indexed=True,
        input_sampling_rate=16000,
        keep_high_quality_only=True,
        skip_empty_text=True,
        shuffle=False,
        shard_seed=42,
        text_field='tn_raw',
    )
    if layout == 'group':
        config['input_cfg'] = [
            dict(type='nemo_tarred', **hq_manifest, weight=9),
            dict(type='nemo_tarred', **{**hq_manifest, 'manifest_filepath': str(live_path)}, weight=1),
        ]
    else:
        config.update(
            manifest_filepath=[[hq_manifest['manifest_filepath'], 9], [str(live_path), 1]],
            tarred_audio_filepaths=[[hq_manifest['tarred_audio_filepaths']]] * 2,
        )
    decode = LazyNeMoTarredIterator._decode_cut_at
    reads = 0

    def bounded_decode(self, index):
        nonlocal reads
        reads += 1
        assert reads <= 120, 'A rejected source prevented the mixture from advancing'
        return decode(self, index)

    monkeypatch.setattr(LazyNeMoTarredIterator, '_decode_cut_at', bounded_decode)
    cuts, _ = read_cutset_from_config(OmegaConf.create(config))
    if not has_eligible_source:
        with pytest.raises(StopIteration):
            next(iter(cuts))
        return
    sampled = list(islice(cuts, 100))
    assert len(sampled) == 100
    assert all(cut.manifest_origin == str(live_path) for cut in sampled)
    assert all(cut.custom['high_quality'] is True for cut in sampled)


def test_indexed_filtered_mixture_preserves_weights_and_resume(hq_manifest, monkeypatch, tmp_path):
    """Filtering before repetition retains configured weights and graph restore."""
    monkeypatch.setenv('USE_AIS_GET_BATCH', 'true')
    monkeypatch.setattr('lhotse.dataset.dataloading.get_worker_partition', lambda: (0, 8))
    other = tmp_path / 'other.jsonl'
    other.write_bytes(Path(hq_manifest['manifest_filepath']).read_bytes())
    config = OmegaConf.create(
        dict(
            indexed=True,
            input_sampling_rate=16000,
            keep_high_quality_only=True,
            skip_empty_text=True,
            shuffle=False,
            shard_seed=42,
            text_field='tn_raw',
            input_cfg=[
                dict(type='nemo_tarred', **hq_manifest, weight=3),
                dict(type='nemo_tarred', **{**hq_manifest, 'manifest_filepath': str(other)}, weight=1),
            ],
        )
    )
    cuts, _ = read_cutset_from_config(config)
    iterator = iter(cuts)
    counts = Counter(c.manifest_origin for c in islice(iterator, 4000))
    assert abs(counts[hq_manifest['manifest_filepath']] / 4000 - 0.75) < 0.03
    state = deepcopy(cuts.data.state_dict())
    expected = [(c.id, c.manifest_origin, c.supervisions[0].text) for c in islice(iterator, 32)]
    restored, _ = read_cutset_from_config(config)
    restored.data.load_state_dict(state)
    actual = [(c.id, c.manifest_origin, c.supervisions[0].text) for c in islice(restored, 32)]
    assert actual == expected


@pytest.mark.parametrize('layout', ['directory', 'weighted', 'mapping'])
def test_shar_rejected_source_terminates(monkeypatch, tmp_path, layout):
    from lhotse.testing.dummies import dummy_cut
    from lhotse.utils import fastcopy

    reads = []

    def bounded_read(cut):
        reads.append(cut.id)
        assert len(reads) <= 16, 'A rejected SHAR source is being repeated indefinitely'
        return cut

    path = tmp_path / 'cuts.jsonl'
    CutSet.from_cuts(fastcopy(dummy_cut(i), custom={'high_quality': False}) for i in range(4)).to_file(path)
    cuts = CutSet.from_file(path).map(bounded_read)
    monkeypatch.setattr(CutSet, 'from_shar', lambda **kwargs: cuts)
    monkeypatch.setattr('nemo.collections.common.data.lhotse.cutset._resolve_shar_inputs', lambda *args: {})
    path = {
        'directory': 'unused-shar',
        'weighted': [['unused-shar-a', 1], ['unused-shar-b', 2]],
        'mapping': {'cuts': ['unused-cuts.jsonl']},
    }[layout]
    config = OmegaConf.create(dict(shar_path=path, keep_high_quality_only=True, shard_seed=42))
    selected, _ = read_cutset_from_config(config)
    with pytest.raises(StopIteration):
        next(iter(selected))
    assert len(reads) == (8 if layout == 'weighted' else 4)
