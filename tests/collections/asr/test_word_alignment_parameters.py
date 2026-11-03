# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from nemo.collections.speechlm2.parts import ctc_timestamp_utils as ts


@pytest.fixture
def alignment_parameters():
    """Use synthetic calibration, independent of any model checkpoint or packaged file."""
    parameters = ts.load_word_alignment_parameters()
    parameters['boundary_interpolation'] = {
        'coefficients': (np.arange(48).reshape(2, 24) * 0.0001).tolist(),
        'offset': [0.01, -0.02],
    }
    return parameters


@pytest.fixture
def alignment_yaml(tmp_path, alignment_parameters):
    """Exercise explicitly supplied YAML using a temporary, test-owned file."""
    path = tmp_path / 'alignment.yaml'
    path.write_text(yaml.safe_dump(alignment_parameters))
    return str(path)


@pytest.mark.unit
@pytest.mark.parametrize('name', ts._WORD_ALIGNMENT_PARAMETER_NAMES)
def test_loaded_defaults_match_constructor_signature(name):
    default = inspect.signature(ts.MultiSpeakerSOTWordTimestampAligner.__init__).parameters[name].default
    assert ts.load_word_alignment_parameters()[name] == default
    if name != 'interpolation_enabled':
        assert getattr(ts.MultiSpeakerSOTWordTimestampAligner(), name) == default


@pytest.mark.unit
@pytest.mark.parametrize(
    'name,expected',
    [
        ('speaker_logprob_weight', 0.6057),
        ('max_word_frames', 19),
        ('vad_threshold', 0.4641),
        ('band_min_states', 50),
        ('band_full_below_frames', 0),
        ('boundary_max_correction_seconds', 0.04),
    ],
)
def test_word_alignment_defaults(name, expected):
    aligner = ts.MultiSpeakerSOTWordTimestampAligner()
    assert getattr(aligner, name) == expected


@pytest.mark.unit
def test_defaults_skip_calibration_without_yaml(monkeypatch):
    monkeypatch.setattr(ts, '_read_word_alignment_parameters', lambda _: pytest.fail('unexpected YAML read'))
    assert ts.MultiSpeakerSOTWordTimestampAligner().boundary_interpolation is None
    assert ts.load_word_alignment_parameters()['boundary_interpolation'] is None


@pytest.mark.unit
@pytest.mark.parametrize('source', ['yaml', 'mapping'])
def test_explicit_parameters_apply_boundary_calibration(alignment_yaml, alignment_parameters, source):
    settings = alignment_yaml if source == 'yaml' else alignment_parameters
    aligner = ts.MultiSpeakerSOTWordTimestampAligner(word_alignment_parameters=settings)
    assert aligner.boundary_interpolation is not None
    assert 'calibration_id' not in aligner.boundary_interpolation
    assert set(aligner.boundary_interpolation) == {'weights', 'offset'}
    assert sum(np.asarray(v).size for v in aligner.boundary_interpolation.values()) == 134
    stored = ts.load_word_alignment_parameters(settings)['boundary_interpolation']
    assert set(stored) == {'coefficients', 'offset'}
    assert np.asarray(stored['coefficients']).shape == (2, 24)
    assert sum(np.asarray(v).size for v in stored.values()) == 50


@pytest.mark.unit
@pytest.mark.parametrize('field', ['coefficients', 'offset'])
def test_yaml_roundtrip_preserves_calibration_values(alignment_yaml, alignment_parameters, field):
    loaded = ts.load_word_alignment_parameters(alignment_yaml)
    assert loaded['boundary_interpolation'][field] == alignment_parameters['boundary_interpolation'][field]


@pytest.mark.unit
def test_calibration_is_applied_once_and_instance_local(alignment_yaml):
    settings = ts.load_word_alignment_parameters(alignment_yaml)
    first = ts.MultiSpeakerSOTWordTimestampAligner(word_alignment_parameters=alignment_yaml)
    original = copy.deepcopy(settings['boundary_interpolation'])
    assert first.boundary_interpolation == ts._expand_boundary_calibration(original, 33)
    assert original == settings['boundary_interpolation']
    assert not {'start_gain', 'end_gain', 'start_offset_seconds', 'end_offset_seconds'} & settings.keys()
    second = ts.MultiSpeakerSOTWordTimestampAligner(boundary_interpolation=first.boundary_interpolation)
    assert second.boundary_interpolation == first.boundary_interpolation
    first.boundary_interpolation['weights'][0][0] = 123
    assert second.boundary_interpolation['weights'][0][0] != 123
    assert ts.load_word_alignment_parameters(alignment_yaml)['boundary_interpolation'] == original


@pytest.mark.unit
@pytest.mark.parametrize('points,count', [(33, 1), (33, 24), (33, 66), (9, 7)])
def test_compressed_calibration_matches_fixed_basis(points, count):
    from scipy.fft import dct

    rng = np.random.default_rng(123)
    coefficients = rng.normal(size=(2, count))
    temporal = dct(np.eye(points), norm='ortho', axis=0).T
    basis = np.empty((2 * points, 2 * points))
    basis[:points, 0::2] = temporal / np.sqrt(2)
    basis[points:, 0::2] = temporal / np.sqrt(2)
    basis[:points, 1::2] = temporal / np.sqrt(2)
    basis[points:, 1::2] = -temporal / np.sqrt(2)
    calibration = dict(coefficients=coefficients.tolist(), offset=[0.01, -0.02])
    expanded = ts.MultiSpeakerSOTWordTimestampAligner(
        boundary_interpolation=calibration, boundary_profile_points=points
    ).boundary_interpolation
    np.testing.assert_allclose(expanded['weights'], coefficients @ basis[:, :count].T, atol=1e-13, rtol=0)
    assert expanded['offset'] == calibration['offset']
    assert 'weights' not in calibration


@pytest.mark.unit
@pytest.mark.parametrize('values', [[], [[1]], [[1], [float('nan')]], [[], []], [[0] * 67] * 2])
def test_invalid_compressed_calibration_is_rejected(values):
    with pytest.raises(ValueError, match='coefficients'):
        ts.MultiSpeakerSOTWordTimestampAligner(boundary_interpolation=dict(coefficients=values, offset=[0, 0]))


@pytest.mark.unit
@pytest.mark.parametrize('extra', ['weights', 'mean'])
def test_mixed_calibration_formats_are_rejected(extra):
    calibration = dict(coefficients=[[1], [1]], offset=[0, 0], **{extra: [[0] * 66] * 2})
    with pytest.raises(ValueError, match='cannot'):
        ts.MultiSpeakerSOTWordTimestampAligner(boundary_interpolation=calibration)


@pytest.mark.unit
def test_folded_calibration_preserves_legacy_boundary_corrections():
    rng = np.random.default_rng(17)
    weights = rng.normal(0, 0.0001, (2, 66))
    mean = rng.uniform(-20, 0, (2, 66))
    offset = np.array([0.01, -0.02])
    legacy = dict(weights=weights.tolist(), mean=mean.tolist(), offset=offset.tolist())
    folded = dict(weights=weights.tolist(), offset=(offset - np.sum(mean * weights, axis=1)).tolist())
    profiles = rng.uniform(-20, 0, (2, 100, 66))
    original = ((profiles - mean[:, None, :]) @ weights[..., None])[..., 0] + offset[:, None]
    compact = (profiles @ weights[..., None])[..., 0] + np.asarray(folded['offset'])[:, None]
    np.testing.assert_allclose(compact, original, rtol=0, atol=1e-14)
    logp = np.log(rng.dirichlet([2, 3], size=100))
    rows = [
        dict(start=0.4, end=0.56, speaker_tag=0, _boundary_labels=[1, 1]),
        dict(start=1.6, end=1.8, speaker_tag=0, _boundary_labels=[1, 1]),
    ]
    old_rows, new_rows = copy.deepcopy(rows), copy.deepcopy(rows)
    segments = [dict(speaker=0, start=0.2, end=1.0)]
    ts._interpolate_word_boundaries(old_rows, logp, 0.04, 0, 4.0, segments, {0: 0}, legacy)
    ts._interpolate_word_boundaries(new_rows, logp, 0.04, 0, 4.0, segments, {0: 0}, folded)
    assert new_rows == old_rows


@pytest.mark.unit
@pytest.mark.parametrize('weight,word_frames', [(0.0, 7), (0.25, 12)])
@pytest.mark.parametrize('interpolation_enabled', [False, True])
def test_explicit_options_override_yaml_and_can_disable_interpolation(
    alignment_yaml, weight, word_frames, interpolation_enabled
):
    aligner = ts.MultiSpeakerSOTWordTimestampAligner(
        speaker_logprob_weight=weight,
        max_word_frames=word_frames,
        interpolation_enabled=interpolation_enabled,
        word_alignment_parameters=alignment_yaml,
    )
    assert aligner.speaker_logprob_weight == weight
    assert aligner.max_word_frames == word_frames
    assert (aligner.boundary_interpolation is not None) == interpolation_enabled


@pytest.mark.unit
@pytest.mark.parametrize('name,yaml_value', [('speaker_logprob_weight', 0.25), ('max_word_frames', 7)])
def test_explicit_default_value_overrides_yaml(tmp_path, name, yaml_value):
    config = tmp_path / 'override.yaml'
    config.write_text(f'{name}: {yaml_value}\n')
    default = ts.load_word_alignment_parameters()[name]
    aligner = ts.MultiSpeakerSOTWordTimestampAligner(word_alignment_parameters=str(config), **{name: default})
    assert getattr(aligner, name) == default
    assert getattr(ts.MultiSpeakerSOTWordTimestampAligner(word_alignment_parameters=str(config)), name) == yaml_value


@pytest.mark.unit
@pytest.mark.parametrize('weight,threshold', [(0.25, 0.25), (0.0, 0.5)])
def test_custom_yaml_is_portable(tmp_path, weight, threshold):
    config = tmp_path / 'alignment_params_40ms-to-10ms.yaml'
    config.write_text(f'speaker_logprob_weight: {weight}\nvad_threshold: {threshold}\ninterpolation_enabled: false\n')
    aligner = ts.MultiSpeakerSOTWordTimestampAligner(word_alignment_parameters=str(config))
    assert aligner.speaker_logprob_weight == weight
    assert aligner.vad_threshold == threshold
    assert aligner.boundary_interpolation is None


@pytest.mark.unit
@pytest.mark.parametrize('key', ['speker_logprob_weight', 'unknown_setting'])
def test_custom_yaml_rejects_unknown_keys(tmp_path, key):
    bad = tmp_path / 'bad.yaml'
    bad.write_text(f'{key}: 1\n')
    with pytest.raises(ValueError, match='Unknown'):
        ts.MultiSpeakerSOTWordTimestampAligner(word_alignment_parameters=str(bad))


class _Tokenizer:
    def text_to_ids(self, text):
        return [0]


@pytest.mark.unit
def test_prepared_roundtrip_retains_effective_settings_and_frame_units(alignment_yaml):
    aligner = ts.MultiSpeakerSOTWordTimestampAligner(
        tokenizer=_Tokenizer(),
        blank_id=1,
        ctc_decoder=SimpleNamespace(upsampling_factor=2),
        word_alignment_parameters=alignment_yaml,
    )
    prepared = aligner.prepare_alignment(
        torch.log_softmax(torch.tensor([[[4.0, 0.0], [0.0, 4.0]]] * 1).repeat(1, 20, 1), -1),
        None,
        ['hello'],
        audio_durations=[1.6],
    )
    assert prepared['config']['max_word_frames'] == 38
    assert prepared['config']['band_min_states'] == 50
    assert prepared['config']['vad_threshold'] == aligner.vad_threshold
    before = copy.deepcopy(prepared)
    results = ts.align_prepared_batch(prepared)
    assert results == aligner.align_prepared(prepared)
    assert 'calibration_id' not in results[0]['boundary_interpolation']
    assert prepared['config'] == before['config']


@pytest.mark.unit
@pytest.mark.parametrize(
    'options,expected',
    [
        pytest.param({}, [False, True, True], id='constructor-default'),
        pytest.param({'vad_threshold': 0.5}, [False, False, False], id='explicit-threshold'),
    ],
)
def test_native_vad_threshold_uses_alignment_options(options, expected):
    from nemo.collections.asr.modules.parallel_expert_encoder import ParallelExpertEncoder

    owner = SimpleNamespace(max_speaker_count=2, frame_shift_seconds=0.01, ctc_timestamp_alignment=options)
    native = torch.tensor([[[0.46], [0.47], [0.5]]])
    result = ParallelExpertEncoder._build_ctc_timestamp_inputs(
        owner, torch.zeros(1, 2, 3), torch.tensor([3]), native, torch.tensor([3]), native, torch.tensor([3])
    )
    assert result.diarization_labels.tolist() == [[expected, [False, False, False]]]
