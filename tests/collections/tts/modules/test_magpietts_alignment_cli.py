# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from nemo.collections.tts.modules.magpietts_inference import utils
from nemo.collections.tts.modules.magpietts_inference.inference import EasyMagpieInferenceConfig
from nemo.collections.tts.modules.magpietts_inference.singleturn_distributed import (
    SINGLE_TURN_EVAL_MANIFEST,
    merge_singleturn_rank_outputs,
    write_singleturn_rank_manifest,
)
from nemo.collections.tts.modules.magpietts_inference.token_alignment import save_token_alignment_debug

pytestmark = pytest.mark.unit


def test_omitted_alignment_arguments_do_not_override_checkpoint_settings():
    parser = argparse.ArgumentParser()
    utils._add_easy_magpie_args(parser)
    assert utils._alignment_inference_overrides(parser.parse_args([])) is None


def test_cli_routes_guidance_thresholds_and_capture_independently():
    parser = argparse.ArgumentParser()
    utils._add_easy_magpie_args(parser)
    args = parser.parse_args(
        [
            '--alignment_guidance',
            'text_head',
            '--alignment_layer',
            '1',
            '--alignment_head',
            '2',
            '--alignment_look_back_tokens',
            '2',
            '--alignment_look_ahead_tokens',
            '4',
            '--alignment_prior_epsilon',
            '0.2',
            '--alignment_min_confidence',
            '0.3',
            '--alignment_min_margin',
            '0.1',
            '--alignment_min_token_frames',
            '3',
            '--save_token_alignments',
        ]
    )
    assert utils._alignment_inference_overrides(args) == dict(
        enabled=True,
        source='text_head',
        layer=1,
        head=2,
        look_back_tokens=2,
        look_ahead_tokens=4,
        prior_epsilon=0.2,
        min_confidence=0.3,
        min_margin=0.1,
        min_token_frames=3,
        record_alignments=True,
    )
    args = parser.parse_args(['--alignment_guidance', 'none', '--save_token_alignments'])
    assert utils._alignment_inference_overrides(args) == {'enabled': False, 'record_alignments': True}


def test_alignment_cli_rejects_wrong_model_and_invalid_layer():
    parser = argparse.ArgumentParser()
    utils._add_easy_magpie_args(parser)
    args = parser.parse_args(['--alignment_guidance', 'attention'])
    args.model_type = 'magpie'
    with pytest.raises(ValueError, match='require --model_type easy_magpie'):
        utils._select_runner_cls(args)
    with pytest.raises(SystemExit):
        parser.parse_args(['--alignment_layer', '-1'])


@pytest.mark.parametrize('mode', ['checkpoint', 'nemo', 'pretrained'])
def test_loader_applies_explicit_overrides_before_model_construction(monkeypatch, mode):
    base = OmegaConf.create(
        {
            'decoder_type': 'nemotron_h',
            'nemotron_h_config': {'rswa_window': None},
            'attention_alignment_inference': {'enabled': False, 'look_ahead_tokens': 7},
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
    options = dict(alignment_inference_overrides={'enabled': True, 'source': 'text_head', 'record_alignments': True})
    if mode == 'checkpoint':
        options.update(hparams_file='hparams.yaml', checkpoint_file='model.ckpt')
    else:
        options['nemo_file'] = 'nvidia/model' if mode == 'pretrained' else 'model.nemo'
    utils.load_easy_magpie_model(utils.ModelLoadConfig(**options), device='cpu')
    assert len(seen) == 1
    alignment = seen[0].attention_alignment_inference
    assert alignment.enabled and alignment.record_alignments
    assert alignment.source == 'text_head'
    assert alignment.look_ahead_tokens == 7, 'Omitted controls must preserve checkpoint values'
    assert not base.attention_alignment_inference.enabled


def test_enabled_identifiers_include_all_controls_but_old_disabled_identifiers_stay_unchanged():
    config = EasyMagpieInferenceConfig()
    original = config.build_identifier()
    config.alignment_inference_config = dict(enabled=False, record_alignments=False, min_confidence=0.9)
    assert config.build_identifier() == original
    config.alignment_inference_config.update(enabled=True, source='attention')
    attention = config.build_identifier()
    config.alignment_inference_config['min_margin'] = 0.1
    assert config.build_identifier() != attention
    config.alignment_inference_config['source'] = 'text_head'
    assert '_Align_text_head_' in config.build_identifier()


def test_singleturn_rank_merge_preserves_timing_sidecars_and_updates_paths(tmp_path):
    for rank in range(2):
        directory = tmp_path / f'rank_{rank:04d}'
        directory.mkdir()
        audio = directory / f'predicted_audio_{rank}.wav'
        audio.write_bytes(b'wav')
        sidecar = audio.with_suffix('.alignment.json')
        sidecar.write_text(json.dumps({'predicted_audio_filepath': str(audio), 'frames': []}))
        record = dict(
            item_idx=rank, text='test', predicted_audio_filepath=str(audio), token_alignment_filepath=str(sidecar)
        )
        (directory / SINGLE_TURN_EVAL_MANIFEST).write_text(json.dumps(record) + '\n')
        write_singleturn_rank_manifest(directory, rank, 2, 2, [rank], False)
    merge_singleturn_rank_outputs(tmp_path, 2, 2, save_predicted_codes=False, timeout_sec=0)
    records = [json.loads(line) for line in (tmp_path / SINGLE_TURN_EVAL_MANIFEST).read_text().splitlines()]
    for index, record in enumerate(records):
        path = tmp_path / f'predicted_audio_{index}.alignment.json'
        assert record['token_alignment_filepath'] == str(path)
        assert json.loads(path.read_text())['predicted_audio_filepath'] == str(
            tmp_path / f'predicted_audio_{index}.wav'
        )


def test_multiturn_rank_merge_keeps_token_timings_with_remapped_output_indices(tmp_path):
    for rank in range(2):
        directory = tmp_path / f'rank_{rank:04d}'
        directory.mkdir()
        for name in ('predicted_audio_0.wav', 'target_audio_0.wav', 'context_audio_0.wav'):
            (directory / name).write_bytes(b'wav')
        sidecar = directory / 'predicted_audio_0.alignment.json'
        sidecar.write_text(
            json.dumps({'predicted_audio_filepath': str(directory / 'predicted_audio_0.wav'), 'frames': []})
        )
        record = dict(
            audio_filepath='target_audio_0.wav',
            context_audio_filepath='context_audio_0.wav',
            text='test',
            token_alignment_filepath=str(sidecar),
        )
        (directory / f'multiturn_user_audio_turn_manifest_rank{rank:04d}.jsonl').write_text(json.dumps(record) + '\n')
    manifest = utils._merge_multiturn_rank_outputs(str(tmp_path), 2, save_predicted_codes=False)
    records = [json.loads(line) for line in open(manifest).read().splitlines()]
    assert len(records) == 2
    for index, record in enumerate(records):
        path = tmp_path / f'predicted_audio_{index}.alignment.json'
        assert record['token_alignment_filepath'] == str(path)
        assert json.loads(path.read_text())['predicted_audio_filepath'] == str(
            tmp_path / f'predicted_audio_{index}.wav'
        )


@pytest.mark.parametrize('mode', ['single_turn', 'multiturn_user_audio'])
def test_relative_output_directory_records_absolute_sidecars_and_merges_once(tmp_path, monkeypatch, mode):
    monkeypatch.chdir(tmp_path)
    root = Path('outputs')
    directory = root / 'rank_0000'
    directory.mkdir(parents=True)
    for name in ('predicted_audio_0.wav', 'target_audio_0.wav', 'context_audio_0.wav'):
        (directory / name).write_bytes(b'wav')
    debug = {'audio_duration_seconds': 0.16, 'frame_step_seconds': 0.08, 'frames': [], 'turns': []}
    model = SimpleNamespace(tokenizer=SimpleNamespace(tokens=[]))
    sidecar = save_token_alignment_debug(directory / 'predicted_audio_0.wav', debug, model)
    assert Path(sidecar).is_absolute()
    assert json.loads(Path(sidecar).read_text())['predicted_audio_filepath'] == str(
        (directory / 'predicted_audio_0.wav').resolve()
    )
    record = dict(item_idx=0, text='test', token_alignment_filepath=sidecar)
    if mode == 'single_turn':
        (directory / SINGLE_TURN_EVAL_MANIFEST).write_text(json.dumps(record) + '\n')
        write_singleturn_rank_manifest(directory, 0, 1, 1, [0], False)
        merge_singleturn_rank_outputs(root, 1, 1, save_predicted_codes=False, timeout_sec=0)
        manifest = root / SINGLE_TURN_EVAL_MANIFEST
    else:
        record.update(audio_filepath='target_audio_0.wav', context_audio_filepath='context_audio_0.wav')
        (directory / 'multiturn_user_audio_turn_manifest_rank0000.jsonl').write_text(json.dumps(record) + '\n')
        manifest = Path(utils._merge_multiturn_rank_outputs(str(root), 1, save_predicted_codes=False))
    merged = json.loads(manifest.read_text().splitlines()[0])
    expected = (root / 'predicted_audio_0.alignment.json').resolve()
    assert merged['token_alignment_filepath'] == str(expected)
    assert json.loads(expected.read_text())['predicted_audio_filepath'] == str(
        (root / 'predicted_audio_0.wav').resolve()
    )
