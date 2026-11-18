# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from pathlib import Path

import pytest
import torch

from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
from nemo.collections.speechlm2.parts import ctc_timestamp_utils as ts

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('symlink', [False, True])
@pytest.mark.parametrize('embedded', [False, True])
def test_json_checkpoint_parameters_override_metadata(tmp_path, checkpoint_parts, save_checkpoint, symlink, embedded):
    _, _, config, parameters = checkpoint_parts
    path = save_checkpoint(calibrated=embedded)
    model_dir = tmp_path / 'snapshot'
    model_dir.mkdir()
    if symlink:
        link = model_dir / 'model.safetensors'
        link.symlink_to(path)
        path = link
    else:
        path = path.rename(model_dir / 'model.safetensors')
    expected = copy.deepcopy(parameters)
    expected['speaker_logprob_weight'] = 0.81
    expected['boundary_interpolation']['offset'] = [0.02, 0.03]
    _write_config(model_dir, decoder_config=config, alignment_config=_alignment_envelope(expected))
    artifact = ts.load_ctc_timestamp_artifact(path)
    assert artifact.alignment_parameters == expected
    aligner = ts.get_ctc_timestamp_aligner(torch.nn.Linear(8, 8), str(path), torch.device('cpu'))
    assert aligner.speaker_logprob_weight == 0.81
    assert aligner.boundary_interpolation == ts._expand_boundary_calibration(expected['boundary_interpolation'], 33)


@pytest.mark.parametrize('override', [None, 0.0])
def test_json_config_only_loading_and_explicit_override(tmp_path, checkpoint_parts, save_checkpoint, override):
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    parameters = checkpoint_parts[-1]
    path = save_checkpoint()
    with safe_open(path, framework='pt') as handle:
        metadata = handle.metadata()
    block = {
        'decoder_config': json.loads(metadata.pop('decoder_config')),
        'tokenizer_config': json.loads(metadata.pop('tokenizer_config')),
        'alignment_config': json.loads(metadata.pop('ctc_timestamp_alignment_config')),
    }
    save_file(load_file(path), path, metadata=metadata)
    _write_config(tmp_path, **block)
    owner = torch.nn.Linear(8, 8)
    owner.ctc_timestamp_alignment = {} if override is None else {'speaker_logprob_weight': override}
    aligner = ts.get_ctc_timestamp_aligner(owner, str(path), torch.device('cpu'))
    assert aligner.speaker_logprob_weight == (0.37 if override is None else override)
    assert aligner.boundary_interpolation == ts._expand_boundary_calibration(parameters['boundary_interpolation'], 33)
    expected = copy.deepcopy(block)
    expected['alignment_config']['parameters']['vad_threshold'] = 0.72
    rebuilt = ts.get_ctc_timestamp_aligner(owner, str(path), torch.device('cpu'), checkpoint_config=expected)
    assert rebuilt.vad_threshold == 0.72
    assert rebuilt.speaker_logprob_weight == aligner.speaker_logprob_weight
    assert ts.get_ctc_timestamp_aligner(owner, str(path), torch.device('cpu')) is rebuilt
    assert ts.get_ctc_timestamp_aligner(owner, str(path), torch.device('cpu'), checkpoint_config=expected) is rebuilt


@pytest.mark.parametrize('contents', ['{broken', '{"ctc_timestamps": []}'], ids=['json', 'block'])
def test_invalid_json_config_never_falls_back_to_metadata(tmp_path, save_checkpoint, contents):
    path = save_checkpoint()
    (tmp_path / 'config.json').write_text(contents)
    with pytest.raises((ValueError, TypeError)):
        ts.load_ctc_timestamp_artifact(path)


@pytest.mark.parametrize('source', ['metadata', 'json'])
@pytest.mark.parametrize(
    'field,value',
    [
        ('schema_version', 999),
        ('basis_version', 999),
        ('upsampling_factor', 1),
        ('parameters.not_an_alignment_option', 1),
        ('parameters.vad_threshold', None),
        ('parameters.boundary_interpolation.offset', [float('nan'), -0.02]),
    ],
    ids=['schema', 'basis', 'upsampling', 'unknown', 'missing', 'nonfinite'],
)
def test_checkpoint_rejects_incompatible_alignment_config(
    tmp_path, checkpoint_parts, save_checkpoint, source, field, value
):
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    path = save_checkpoint()
    embedded = _alignment_envelope(copy.deepcopy(checkpoint_parts[-1]))
    target = embedded
    *parents, key = field.split('.')
    for parent in parents:
        target = target[parent]
    if value is None:
        del target[key]
    else:
        target[key] = value
    if source == 'json':
        _write_config(tmp_path, alignment_config=embedded)
    else:
        with safe_open(path, framework='pt') as handle:
            metadata = handle.metadata()
        metadata['ctc_timestamp_alignment_config'] = json.dumps(embedded)
        save_file(load_file(path), path, metadata=metadata)
    with pytest.raises(ValueError, match='alignment'):
        ts.load_ctc_timestamp_artifact(path)


@pytest.mark.parametrize('suffix,bundled', [('.pt', False), ('.safetensors', False), ('.safetensors', True)])
@pytest.mark.parametrize('override', [None, 0.0])
def test_checkpoint_parameters_roundtrip_and_override(
    tmp_path, checkpoint_parts, save_checkpoint, suffix, bundled, override
):
    _, tokenizer, _, parameters = checkpoint_parts
    path = save_checkpoint('head' + suffix)
    if bundled:
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file

        tensors = {'ctc_timestamp.' + k: v for k, v in load_file(path).items()}
        tensors['llm.weight'] = torch.ones(1)
        with safe_open(path, framework='pt') as handle:
            metadata = handle.metadata()
        metadata = {
            ('ctc_timestamp_' + k if k != 'ctc_timestamp_alignment_config' else k): v for k, v in metadata.items()
        }
        path = tmp_path / 'model.safetensors'
        save_file(tensors, path, metadata=metadata)

    assert ts.load_ctc_timestamp_artifact(path).alignment_parameters == parameters
    owner = torch.nn.Linear(8, 8)
    owner.ctc_timestamp_alignment = {} if override is None else {'speaker_logprob_weight': override}
    aligner = ts.get_ctc_timestamp_aligner(owner, str(path), torch.device('cpu'))
    assert aligner.speaker_logprob_weight == (0.37 if override is None else override)
    assert aligner.vad_threshold == 0.61
    assert aligner.boundary_interpolation == ts._expand_boundary_calibration(parameters['boundary_interpolation'], 33)
    prepared = aligner.prepare_alignment(
        torch.zeros(1, 32, tokenizer.vocab_size + 1), None, ['hello'], audio_durations=[1.28]
    )
    assert ts.align_prepared_batch(prepared) == aligner.align_prepared(prepared)
    assert ts.get_ctc_timestamp_aligner(owner, str(path), torch.device('cpu')) is aligner
    owner.ctc_timestamp_alignment = {'max_word_frames': 7}
    rebuilt = ts.get_ctc_timestamp_aligner(owner, str(path), torch.device('cpu'))
    assert rebuilt is not aligner and rebuilt.ctc_decoder is aligner.ctc_decoder
    assert rebuilt.max_word_frames == 7 and rebuilt.speaker_logprob_weight == 0.37
    assert rebuilt.boundary_interpolation == aligner.boundary_interpolation


@pytest.mark.parametrize('suffix', ['.pt', '.safetensors'])
def test_legacy_checkpoint_does_not_inherit_another_heads_interpolation(save_checkpoint, suffix):
    path = save_checkpoint('legacy' + suffix, calibrated=False)
    aligner = ts.get_ctc_timestamp_aligner(torch.nn.Linear(8, 8), str(path), torch.device('cpu'))
    assert aligner.boundary_interpolation is None


@pytest.mark.parametrize('source', ['metadata', 'json'])
@pytest.mark.parametrize('weight', [None, 0.0])
def test_automatic_checkpoint_calibration_changes_word_edges(
    tmp_path, monkeypatch, checkpoint_parts, save_checkpoint, source, weight
):
    decoder, tokenizer, _, parameters = checkpoint_parts
    parameters.update(
        speaker_logprob_weight=0.25,
        ctc_frame_seconds=0.04,
        boundary_outside_word_duration_seconds=0.24,
        boundary_interpolation={'coefficients': [[0.0] * 24, [0.0] * 24], 'offset': [-0.02, 0.03]},
    )
    path = save_checkpoint(calibrated=source == 'metadata')
    if source == 'json':
        _write_config(tmp_path, alignment_config=_alignment_envelope(parameters))
    owner = torch.nn.Linear(8, 8)
    owner.ctc_timestamp_alignment = {} if weight is None else {'speaker_logprob_weight': weight}
    automatic = ts.get_ctc_timestamp_aligner(owner, str(path), torch.device('cpu'))
    monkeypatch.setattr(automatic.tokenizer, 'text_to_ids', lambda _: [1])
    explicit = ts.MultiSpeakerSOTWordTimestampAligner(
        tokenizer=automatic.tokenizer, ctc_decoder=decoder, **{**parameters, **owner.ctc_timestamp_alignment}
    )
    logits = torch.full((1, 20, tokenizer.vocab_size + 1), -10.0)
    logits[:, :, tokenizer.vocab_size] = 0.0
    logits[:, 8, 1] = 10.0
    prepared = automatic.prepare_alignment(logits.log_softmax(-1), None, ['hello'], audio_durations=[0.8])
    actual = automatic.align_prepared(prepared)
    expected = explicit.align_prepared(
        explicit.prepare_alignment(logits.log_softmax(-1), None, ['hello'], audio_durations=[0.8])
    )
    assert actual == expected
    assert automatic.speaker_logprob_weight == (0.25 if weight is None else weight)
    word = actual[0]['speaker_word_timestamps'][0][0]
    # The outside-word duration cap also applies, hence .58 rather than uncapped .59.
    assert (word['start'], word['end']) == pytest.approx((0.30, 0.58))


@pytest.mark.parametrize('options,threshold', [({}, 0.61), ({'vad_threshold': 0.5}, 0.5)])
def test_encoder_capture_uses_checkpoint_vad_threshold(save_checkpoint, options, threshold):
    from nemo.collections.asr.modules.parallel_expert_encoder import ParallelExpertEncoder

    owner = torch.nn.Linear(8, 8)
    owner.max_speaker_count, owner.frame_shift_seconds = 1, 0.01
    owner.ctc_timestamp_model_path = str(save_checkpoint())
    owner.ctc_timestamp_alignment = copy.deepcopy(options)
    probs = torch.tensor([[[0.55], [0.65]]])
    inputs = ParallelExpertEncoder._build_ctc_timestamp_inputs(
        owner, torch.zeros(1, 8, 2), torch.tensor([2]), probs, torch.tensor([2]), probs, torch.tensor([2])
    )
    assert torch.equal(inputs.diarization_labels, (probs > threshold).transpose(1, 2))


@pytest.fixture
def checkpoint_parts(monkeypatch):
    """Build a tiny calibrated head without any external parameter file."""
    monkeypatch.setattr(ts, '_read_word_alignment_parameters', lambda _: pytest.fail('unexpected YAML read'))
    tokenizer = SentencePieceTokenizer(
        str(Path(__file__).resolve().parents[2] / '.data/asr/tokenizers/an4_spe_128/tokenizer.model')
    )
    config = dict(feat_in=8, num_classes=tokenizer.vocab_size, use_transformer=False, upsampling_factor=2)
    parameters = ts.load_word_alignment_parameters()
    parameters.update(
        speaker_logprob_weight=0.37,
        vad_threshold=0.61,
        boundary_interpolation={'coefficients': [[0.001] * 24, [-0.002] * 24], 'offset': [0.01, -0.02]},
    )
    return ts.TransformerCTCDecoder(**config), tokenizer, config, parameters


@pytest.fixture
def save_checkpoint(tmp_path, checkpoint_parts):
    decoder, tokenizer, config, parameters = checkpoint_parts

    def save(name='head.safetensors', *, calibrated=True):
        return ts.save_ctc_timestamp_artifact(
            tmp_path / name, decoder, tokenizer, config, alignment_parameters=parameters if calibrated else None
        )

    return save


def _alignment_envelope(parameters):
    return dict(schema_version=1, basis_version=1, upsampling_factor=2, parameters=parameters)


def _write_config(directory, **block):
    (directory / 'config.json').write_text(json.dumps({'ctc_timestamps': block}))
