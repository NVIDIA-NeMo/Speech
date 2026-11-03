# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import copy
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo.collections.tts.modules.magpietts_inference import utils
from nemo.collections.tts.modules.magpietts_inference.inference import EasyMagpieInferenceConfig
from nemo.collections.tts.modules.magpietts_inference.singleturn_distributed import (
    SINGLE_TURN_EVAL_MANIFEST,
    merge_singleturn_rank_outputs,
    write_singleturn_rank_manifest,
)
from nemo.collections.tts.modules.magpietts_inference.token_timestamps import save_token_timestamps

pytestmark = pytest.mark.unit


def test_timestamp_flag_does_not_enable_guidance_or_alignment_debug_capture():
    parser = argparse.ArgumentParser()
    utils._add_easy_magpie_args(parser)
    assert utils._alignment_inference_overrides(parser.parse_args([])) is None
    assert utils._alignment_inference_overrides(parser.parse_args(['--save_timestamps'])) == {
        'record_timestamps': True,
    }
    args = parser.parse_args(
        [
            '--alignment_guidance',
            'none',
            '--save_timestamps',
            '--alignment_layer',
            'middle',
            '--alignment_head',
            '0',
            '--timestamp_method',
            'monotonic',
            '--timestamp_lookahead_frames',
            '4',
            '--timestamp_skip_penalty',
            '6.0',
            '--timestamp_confirmation_frames',
            '1',
            '--timestamp_min_confidence',
            '0.2',
            '--timestamp_min_text_mass',
            '0.3',
            '--timestamp_min_margin',
            '0.1',
        ]
    )
    assert utils._alignment_inference_overrides(args) == dict(
        enabled=False,
        record_timestamps=True,
        layer='middle',
        head=0,
        timestamp_method='monotonic',
        timestamp_lookahead_frames=4,
        timestamp_skip_penalty=6.0,
        timestamp_confirmation_frames=1,
        timestamp_min_confidence=0.2,
        timestamp_min_text_mass=0.3,
        timestamp_min_margin=0.1,
    )
    assert utils._alignment_inference_overrides(parser.parse_args(['--no-save_timestamps'])) == {
        'record_timestamps': False,
    }


@pytest.mark.parametrize('mode', ['checkpoint', 'nemo', 'pretrained'])
def test_timestamp_only_overrides_are_applied_before_each_loading_path(monkeypatch, mode):
    base = OmegaConf.create(
        {
            'decoder_type': 'nemotron_h',
            'nemotron_h_config': {'rswa_window': None},
            'attention_alignment_inference': {'enabled': False},
        }
    )
    seen = []
    resolver = utils.EasyMagpieTTSInferenceModel._resolve_attention_alignment_inference_config

    class FakeModel:
        _resolve_attention_alignment_inference_config = staticmethod(resolver)

        def __init__(self, cfg):
            seen.append(copy.deepcopy(cfg))

        @classmethod
        def restore_from(cls, path, return_config=False, override_config_path=None, **kwargs):
            return copy.deepcopy(base) if return_config else cls(override_config_path)

        from_pretrained = restore_from

        def load_state_dict(self, values):
            assert values == {'sentinel': 1}

        def to(self, device):
            return self

        def eval(self):
            return self

        def float(self):
            return self

    monkeypatch.setattr(utils, 'EasyMagpieTTSInferenceModel', FakeModel)
    monkeypatch.setattr(utils.OmegaConf, 'load', lambda path: copy.deepcopy(base))
    monkeypatch.setattr(utils.torch, 'load', lambda *args, **kwargs: {'state_dict': {'sentinel': 1}})
    options = dict(alignment_inference_overrides={'record_timestamps': True})
    if mode == 'checkpoint':
        options.update(hparams_file='hparams.yaml', checkpoint_file='model.ckpt')
    else:
        options['nemo_file'] = 'nvidia/model' if mode == 'pretrained' else 'model.nemo'
    utils.load_easy_magpie_model(utils.ModelLoadConfig(**options), device='cpu')
    assert len(seen) == 1
    assert seen[0].attention_alignment_inference.record_timestamps
    assert not seen[0].attention_alignment_inference.enabled
    assert not seen[0].attention_alignment_inference.record_alignments


def test_disabled_timestamp_defaults_do_not_change_existing_guidance_run_identifiers():
    config = EasyMagpieInferenceConfig()
    config.alignment_inference_config = dict(enabled=True, source='attention', record_alignments=True)
    before = config.build_identifier()
    config.alignment_inference_config.update(
        record_timestamps=False,
        timestamp_confirmation_frames=2,
        timestamp_min_confidence=0.1,
        timestamp_min_text_mass=0.2,
        timestamp_min_margin=0.0,
        timestamp_method='monotonic',
        timestamp_lookahead_frames=4,
        timestamp_skip_penalty=6.0,
    )
    assert config.build_identifier() == before
    config.alignment_inference_config.update(enabled=False, record_alignments=False, record_timestamps=True)
    assert '_Timestamps_' in config.build_identifier()
    assert '_Align_' not in config.build_identifier()


def test_timestamp_method_selects_the_estimator_without_enabling_any_guidance():
    parser = argparse.ArgumentParser()
    utils._add_easy_magpie_args(parser)
    args = parser.parse_args(['--save_timestamps', '--timestamp_method', 'argmax'])
    assert utils._alignment_inference_overrides(args) == dict(record_timestamps=True, timestamp_method='argmax')
    with pytest.raises(SystemExit):
        parser.parse_args(['--timestamp_method', 'unknown'])


@pytest.mark.parametrize('mode', ['single_turn', 'multiturn'])
def test_distributed_relative_output_merge_preserves_timestamp_sidecars(tmp_path, monkeypatch, mode):
    monkeypatch.chdir(tmp_path)
    root = Path('outputs')
    for rank in range(2):
        directory = root / f'rank_{rank:04d}'
        directory.mkdir(parents=True)
        index = rank if mode == 'single_turn' else 0
        for stem in ('predicted_audio', 'target_audio', 'context_audio'):
            (directory / f'{stem}_{index}.wav').write_bytes(b'wav')
        sidecar = save_token_timestamps(directory / f'predicted_audio_{index}.wav', _trace())
        record = dict(
            item_idx=rank,
            text='test',
            audio_filepath=f'target_audio_{index}.wav',
            context_audio_filepath=f'context_audio_{index}.wav',
            timestamps_filepath=sidecar,
        )
        if mode == 'single_turn':
            (directory / SINGLE_TURN_EVAL_MANIFEST).write_text(json.dumps(record) + '\n')
            write_singleturn_rank_manifest(directory, rank, 2, 2, [rank], False)
        else:
            (directory / f'multiturn_user_audio_turn_manifest_rank{rank:04d}.jsonl').write_text(
                json.dumps(record) + '\n'
            )
    if mode == 'single_turn':
        merge_singleturn_rank_outputs(root, 2, 2, save_predicted_codes=False, timeout_sec=0)
        manifest = root / SINGLE_TURN_EVAL_MANIFEST
    else:
        manifest = Path(utils._merge_multiturn_rank_outputs(str(root), 2, save_predicted_codes=False))
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(records) == 2
    for index, record in enumerate(records):
        sidecar = (root / f'predicted_audio_{index}.timestamps.json').resolve()
        assert record['timestamps_filepath'] == str(sidecar)
        content = json.loads(sidecar.read_text())
        assert content['predicted_audio_filepath'] == str((root / f'predicted_audio_{index}.wav').resolve())
        assert content['timestamps'][0]['token_index'] == 0
        assert content['timestamps'][0]['start'] == 0.0


def _trace():
    return dict(
        source='attention',
        audio_duration_seconds=0.08,
        frame_step_seconds=0.08,
        timestamps=[dict(token_index=0, token_id=1, start=0.0, end=0.08, confidence=0.8, status='finalized')],
    )
