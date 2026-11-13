# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import io
import pickle
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from nemo.collections.tts.models import easy_magpietts_preference_optimization as preference
from nemo.collections.tts.parts.utils import helpers

pytestmark = pytest.mark.unit


class _Scorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 2)
        self.register_buffer('offset', torch.tensor([0.5, 1.0]))
        with torch.no_grad():
            self.projection.weight.fill_(0.25)
            self.projection.bias.fill_(0.75)

    def forward(self, inputs: torch.Tensor, *, gain: float = 1.0):
        """Return nonzero scores for a batch of feature vectors."""
        if inputs.ndim != 2:
            raise ValueError('A batch matrix is required.')
        return self.projection(inputs) * gain + self.offset


def _cuda_eligible_scorer():
    """Keep real CPU parameters while mocking only the helper's device eligibility."""
    model = _Scorer().eval()
    location = SimpleNamespace(device=torch.device('cuda'))
    model.parameters = lambda: iter([location])
    return model, location


@pytest.mark.parametrize('case', ['disabled', 'cpu', 'parameterless'])
def test_reward_compilation_skips_disabled_cpu_and_parameterless_models(monkeypatch, case):
    if case == 'disabled':
        model, _ = _cuda_eligible_scorer()
        mode = None
    else:
        model = _Scorer().eval() if case == 'cpu' else nn.Identity().eval()
        mode = 'default'
    original = model.forward
    monkeypatch.setattr(torch, 'compile', lambda *args, **kwargs: pytest.fail('Compilation must be skipped.'))

    helpers.compile_inference_model(model, mode)

    assert model.forward == original
    assert not getattr(model, '_reward_compile_configured', False)


def test_reward_compilation_preserves_parameters_checkpoint_dtype_and_forward_metadata(monkeypatch):
    model, _ = _cuda_eligible_scorer()
    model.to(dtype=torch.bfloat16)
    original = model.forward
    parameter_ids = {name: id(value) for name, value in model.named_parameters()}
    state = {name: value.clone() for name, value in model.state_dict().items()}
    inputs = torch.ones(2, 3, dtype=torch.bfloat16)
    expected = original(inputs, gain=2.0)
    compiled_calls = []

    def compile_forward(forward, **kwargs):
        assert forward == original
        compiled_calls.append(kwargs)
        return lambda *args, **kwargs: forward(*args, **kwargs)

    monkeypatch.setattr(torch, 'compile', compile_forward)
    helpers.compile_inference_model(model, 'default', dynamic=True)
    helpers.compile_inference_model(model, 'default', dynamic=True)

    assert compiled_calls == [{'backend': 'inductor', 'mode': 'default', 'dynamic': True}]
    assert inspect.signature(model.forward) == inspect.signature(original)
    assert model.forward.__name__ == original.__name__ and model.forward.__doc__ == original.__doc__
    assert model.forward.__wrapped__ == original
    torch.testing.assert_close(model(inputs, gain=2.0), expected)
    assert {name: id(value) for name, value in model.named_parameters()} == parameter_ids
    assert model.state_dict().keys() == state.keys()
    for name, value in model.state_dict().items():
        assert value.dtype == state[name].dtype == torch.bfloat16
        torch.testing.assert_close(value, state[name])

    archive = io.BytesIO()
    torch.save(model.state_dict(), archive)
    archive.seek(0)
    restored = _Scorer().to(dtype=torch.bfloat16).eval()
    restored.load_state_dict(torch.load(archive, weights_only=True))
    torch.testing.assert_close(restored(inputs, gain=2.0), expected)


def test_reward_compilation_runtime_failure_restores_eager_scores_without_recompiling(monkeypatch):
    model, _ = _cuda_eligible_scorer()
    original = model.forward
    inputs = torch.ones(1, 3)
    expected = original(inputs, gain=3.0)
    calls, warnings = [], []

    def compile_forward(*args, **kwargs):
        calls.append('compile')

        def fail(*args, **kwargs):
            calls.append('compiled_forward')
            raise RuntimeError('compiler execution failed')

        return fail

    monkeypatch.setattr(torch, 'compile', compile_forward)
    monkeypatch.setattr(helpers.logging, 'warning', warnings.append)
    helpers.compile_inference_model(model, 'reduce-overhead')

    torch.testing.assert_close(model(inputs, gain=3.0), expected)
    assert model.forward == original
    helpers.compile_inference_model(model, 'reduce-overhead')
    torch.testing.assert_close(model(inputs, gain=3.0), expected)
    assert calls == ['compile', 'compiled_forward']
    assert len(warnings) == 1 and 'compiler execution failed' in warnings[0] and 'eager' in warnings[0]
    assert expected.abs().sum() > 0


def test_reward_compilation_setup_failure_keeps_eager_forward_and_is_attempted_once(monkeypatch):
    model, _ = _cuda_eligible_scorer()
    original = model.forward
    calls, warnings = [], []

    def fail(*args, **kwargs):
        calls.append('compile')
        raise RuntimeError('backend unavailable')

    monkeypatch.setattr(torch, 'compile', fail)
    monkeypatch.setattr(helpers.logging, 'warning', warnings.append)
    helpers.compile_inference_model(model, 'default')
    helpers.compile_inference_model(model, 'default')

    assert model.forward == original and calls == ['compile']
    assert len(warnings) == 1 and 'backend unavailable' in warnings[0]
    torch.testing.assert_close(model(torch.ones(1, 3)), original(torch.ones(1, 3)))


def test_reward_compilation_fallback_preserves_real_eager_input_errors(monkeypatch):
    model, _ = _cuda_eligible_scorer()

    def fail(*args, **kwargs):
        raise RuntimeError('compiled failure')

    monkeypatch.setattr(torch, 'compile', lambda *args, **kwargs: fail)
    helpers.compile_inference_model(model, 'default')

    with pytest.raises(ValueError, match='batch matrix is required'):
        model(torch.ones(3))


def test_reward_compilation_bypasses_compiled_forward_after_moving_to_cpu(monkeypatch):
    model, location = _cuda_eligible_scorer()
    original = model.forward
    monkeypatch.setattr(
        torch, 'compile', lambda *args, **kwargs: lambda *a, **kw: pytest.fail('The scorer has moved to CPU.')
    )
    helpers.compile_inference_model(model, 'default')
    location.device = torch.device('cpu')

    torch.testing.assert_close(model(torch.ones(1, 3)), original(torch.ones(1, 3)))


def test_reward_compilation_requires_an_eval_scorer(monkeypatch):
    model, _ = _cuda_eligible_scorer()
    model.train()
    monkeypatch.setattr(torch, 'compile', lambda *args, **kwargs: pytest.fail('Training scorers must be rejected.'))

    with pytest.raises(ValueError, match='eval model'):
        helpers.compile_inference_model(model, 'default')


@pytest.mark.parametrize('mode', [None, 'default'])
def test_speaker_reward_compiles_only_encoder_and_decoder_with_dynamic_shapes(monkeypatch, mode):
    encoder, decoder = nn.Identity(), nn.Identity()
    speaker = SimpleNamespace(encoder=encoder, decoder=decoder)
    references = ['first_reference.wav', 'second_reference.wav']
    predictions = ['first_0.wav', 'first_1.wav', 'second_0.wav', 'second_1.wav']
    reference_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    prediction_embeddings = reference_embeddings.repeat_interleave(2, dim=0)
    compiled, extracted = [], []
    policy = SimpleNamespace(
        cfg=OmegaConf.create({'speaker_reward_compile_mode': mode}),
        _eval_speaker_verification_model=speaker,
        device=torch.device('cpu'),
        _get_reference_audio_paths=lambda batch: references,
    )

    def extract(paths, model, device):
        assert model is speaker and device == policy.device
        extracted.append(paths)
        return prediction_embeddings if paths == predictions else reference_embeddings

    monkeypatch.setattr(preference, 'compile_inference_model', lambda *args, **kwargs: compiled.append((args, kwargs)))
    monkeypatch.setattr(preference, 'get_speaker_embeddings_from_filepaths', extract)
    predicted, reference = preference.OnlinePreferenceOptimizationMixin._compute_speaker_embeddings_parallel(
        policy, predictions, {}, 2
    )

    assert compiled == (
        [] if mode is None else [((encoder, mode), {'dynamic': True}), ((decoder, mode), {'dynamic': True})]
    )
    assert extracted == [predictions, references]
    torch.testing.assert_close(predicted, prediction_embeddings)
    torch.testing.assert_close(reference, prediction_embeddings)


@pytest.mark.parametrize('wrapped,num_threads', [(False, 1), (True, 1), (True, 4)])
def test_utmos_reward_compiles_torch_submodules_after_device_move_and_preserves_partial_batch(
    monkeypatch, wrapped, num_threads
):
    scorer = nn.Module()
    scorer.ssl = nn.Module()
    scorer.ssl.encoder = nn.Module()
    scorer.ssl.encoder.model = nn.Linear(3, 2)
    scorer.spec_long = nn.Linear(2, 1)
    scorer.ssl.encoder.processor = SimpleNamespace(name='CPU feature processor')
    wrapper = SimpleNamespace(_model=scorer) if wrapped else scorer
    events = []
    paths = ['/audio/first.wav', '/audio/second.wav', '/audio/last_partial.wav']

    class Calculator:
        model = wrapper

        def to(self, device):
            events.append(('to', device))

        def process_directory(self, directory, **kwargs):
            assert directory == '/audio'
            assert kwargs == {
                'batch_size': 8,
                'num_workers': 0,
                'num_threads': num_threads,
                'val_list': ['first.wav', 'second.wav', 'last_partial.wav'],
            }
            events.append(('score',))
            return [{'predicted_mos': value} for value in [4.25, 3.5, 2.75]]

    policy = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                'utmos_compile_mode': 'reduce-overhead',
                'utmos_batch_size': 8,
                **({'utmos_cpu_threads': num_threads} if num_threads != 1 else {}),
            }
        ),
        use_utmos=True,
        utmos_device='cuda',
        device=torch.device('cuda:1'),
        _utmos_calculator=Calculator(),
        _get_audio_dir=lambda: '/audio',
    )
    monkeypatch.setattr(
        preference, 'compile_inference_model', lambda *args, **kwargs: events.append(('compile', args, kwargs))
    )

    scores = preference.OnlinePreferenceOptimizationMixin._compute_utmos_scores_batched(policy, paths)

    assert scores == [4.25, 3.5, 2.75]
    assert events == [
        ('to', policy.device),
        ('compile', (scorer.ssl.encoder.model, 'reduce-overhead'), {}),
        ('compile', (scorer.spec_long, 'reduce-overhead'), {}),
        ('score',),
    ]


@pytest.mark.parametrize('enabled,paths,expected', [(False, ['one.wav', 'two.wav'], [0.0, 0.0]), (True, [], [])])
def test_utmos_off_or_empty_batch_does_not_compile_or_access_the_scorer(monkeypatch, enabled, paths, expected):
    policy = SimpleNamespace(use_utmos=enabled)
    monkeypatch.setattr(preference, 'compile_inference_model', lambda *a, **kw: pytest.fail('No scoring is needed.'))

    assert preference.OnlinePreferenceOptimizationMixin._compute_utmos_scores_batched(policy, paths) == expected


@pytest.mark.parametrize('key', ['utmos_compile_mode', 'speaker_reward_compile_mode'])
@pytest.mark.parametrize('value', ['defualt', False])
def test_invalid_reward_compile_mode_is_rejected_before_model_loading(key, value):
    class Parent:
        def __init__(self, *args, **kwargs):
            pytest.fail('Validate the compilation mode before loading model weights.')

    class Policy(preference.OnlinePreferenceOptimizationMixin, Parent):
        pass

    with pytest.raises(ValueError, match=key):
        Policy(OmegaConf.create({key: value}))


class _PickleDataset:
    pass


class _PickleTransform:
    def __call__(self, value):
        return value


@pytest.mark.parametrize('workers', [0, 1, 4])
def test_utmos_spawn_config_removes_imported_modules_only_and_preserves_score_order(workers):
    pytest.importorskip('utmosv2')
    from nemo.collections.tts.modules.utmosv2 import UTMOSv2Calculator

    transform = _PickleTransform()
    config = SimpleNamespace(
        transforms=ModuleType('imported_transforms'),
        dataset_class=_PickleDataset,
        test_transform=transform,
        sample_rate=16000,
    )
    with pytest.raises(TypeError, match='module'):
        pickle.dumps(config)

    class WrappedModel:
        def __init__(self):
            self._cfg_value = config
            self._model = SimpleNamespace(cfg=config)
            self.seen = None

        @property
        def _cfg(self):
            return self._cfg_value

        def predict(self, **kwargs):
            self.seen = kwargs
            assert self._cfg.dataset_class is _PickleDataset and self._cfg.test_transform is transform
            assert self._cfg.sample_rate == 16000
            assert self._model.cfg is config
            if workers:
                restored = pickle.loads(pickle.dumps(self._cfg))
                assert restored.dataset_class is _PickleDataset
                assert isinstance(restored.test_transform, _PickleTransform)
                assert not hasattr(self._cfg, 'transforms')
            else:
                assert self._cfg is config and isinstance(self._cfg.transforms, ModuleType)
            return [
                {'file_path': '/audio/third.wav', 'predicted_mos': 2.75},
                {'file_path': '/audio/first.wav', 'predicted_mos': 4.25},
                {'file_path': '/audio/second.wav', 'predicted_mos': 3.5},
            ]

    calculator = UTMOSv2Calculator.__new__(UTMOSv2Calculator)
    calculator.device = torch.device('cpu')
    calculator.verbose = False
    calculator.model = WrappedModel()
    names = ['first.wav', 'second.wav', 'third.wav']

    scores = calculator.process_directory('/audio', batch_size=8, num_workers=workers, val_list=names)

    assert [row['predicted_mos'] for row in scores] == [4.25, 3.5, 2.75]
    assert calculator.model.seen == {
        'input_dir': '/audio',
        'device': calculator.device,
        'num_repetitions': 1,
        'num_workers': workers,
        'batch_size': 8,
        'val_list': names,
        'verbose': False,
    }


def test_utmos_cpu_thread_limit_is_scoped_to_scoring(monkeypatch):
    pytest.importorskip('utmosv2')
    from nemo.collections.tts.modules import utmosv2

    events = []

    @contextmanager
    def thread_limit(*, limits):
        events.append(('enter', limits))
        try:
            yield
        finally:
            events.append(('exit', limits))

    def predict(**kwargs):
        assert events == [('enter', 4)]
        assert 'num_threads' not in kwargs
        events.append(('score',))
        return [{'file_path': '/audio/one.wav', 'predicted_mos': 3.75}]

    calculator = utmosv2.UTMOSv2Calculator.__new__(utmosv2.UTMOSv2Calculator)
    calculator.model = SimpleNamespace(predict=predict)
    calculator.device = torch.device('cpu')
    calculator.verbose = False
    monkeypatch.setattr(utmosv2, 'threadpool_limits', thread_limit)

    results = calculator.process_directory('/audio', num_workers=0, num_threads=4, val_list=['one.wav'])

    assert events == [('enter', 4), ('score',), ('exit', 4)]
    assert results[0]['predicted_mos'] == 3.75


def test_utmos_rejects_nonpositive_cpu_thread_limit_before_scoring(monkeypatch):
    pytest.importorskip('utmosv2')
    from nemo.collections.tts.modules import utmosv2

    def forbidden(*args, **kwargs):
        pytest.fail('Invalid thread counts must be rejected before scoring or entering a thread-pool context.')

    calculator = utmosv2.UTMOSv2Calculator.__new__(utmosv2.UTMOSv2Calculator)
    calculator.model = SimpleNamespace(predict=forbidden)
    calculator.device = torch.device('cpu')
    calculator.verbose = False
    monkeypatch.setattr(utmosv2, 'threadpool_limits', forbidden)

    with pytest.raises(ValueError, match='num_threads'):
        calculator.process_directory('/audio', num_workers=0, num_threads=0)
