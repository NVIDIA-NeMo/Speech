# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from types import SimpleNamespace

import pytest
import torch

from nemo.collections.tts.models.easy_magpietts import EasyMagpieTTSModel
from nemo.collections.tts.models.easy_magpietts_preference_optimization import EasyMagpieTTSModelOnlinePO


class TrackingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.moves = []

    def to(self, device, *args, **kwargs):
        self.moves.append(torch.device(device))
        return super().to(device, *args, **kwargs)


def make_model() -> EasyMagpieTTSModel:
    model = object.__new__(EasyMagpieTTSModel)
    torch.nn.Module.__init__(model)
    model.offload_validation_models = True
    model._codec_model = torch.nn.Module()
    model._codec_model.audio_encoder = TrackingModule()
    model._codec_model.vector_quantizer = TrackingModule()
    model._codec_model.audio_decoder = TrackingModule()
    model._codec_converter = TrackingModule()
    model._eval_asr_model = TrackingModule()
    model._eval_speaker_verification_model = TrackingModule()
    model.whisper_model = TrackingModule()
    model.register_parameter("training_parameter", torch.nn.Parameter(torch.ones(1)))
    return model


@pytest.mark.unit
def test_validation_model_offload_keeps_training_quantizers_resident():
    model = make_model()

    model._move_validation_models(torch.device('cpu'))

    assert model._codec_model.audio_encoder.moves == [torch.device('cpu')]
    assert model._codec_model.audio_decoder.moves == [torch.device('cpu')]
    assert model._eval_asr_model.moves == [torch.device('cpu')]
    assert model._eval_speaker_verification_model.moves == [torch.device('cpu')]
    assert model.whisper_model.moves == [torch.device('cpu')]
    assert model._codec_model.vector_quantizer.moves == []
    assert model._codec_converter.moves == []


@pytest.mark.unit
def test_validation_epoch_start_restores_offloaded_modules(monkeypatch):
    model = make_model()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(EasyMagpieTTSModel, "device", property(lambda _: torch.device("cpu")), raising=False)

    model.on_validation_epoch_start()

    assert model._codec_model.audio_encoder.moves == [torch.device("cpu")]
    assert model._eval_asr_model.moves == [torch.device("cpu")]


@pytest.mark.unit
def test_online_po_keeps_reward_models_available_during_training():
    assert EasyMagpieTTSModel._uses_validation_models_during_training(None) is False
    assert EasyMagpieTTSModelOnlinePO._uses_validation_models_during_training(None) is True


@pytest.mark.unit
def test_offloaded_modules_are_ignored_by_ddp():
    model = make_model()
    model._codec_model.audio_decoder = torch.nn.Sequential(torch.nn.Linear(2, 2))
    aliased_scorer = torch.nn.Module()
    aliased_scorer.register_buffer('running_state', torch.ones(2))
    model.scorer_alias = aliased_scorer
    model._eval_asr_model = aliased_scorer

    model._mark_validation_models_ddp_ignored()

    ignored = set(model._ddp_params_and_buffers_to_ignore)
    assert '_codec_model.audio_decoder.0.weight' in ignored
    assert 'scorer_alias.running_state' in ignored
    assert '_eval_asr_model.running_state' in ignored
    assert '_codec_model.vector_quantizer' not in ignored
    assert 'training_parameter' not in ignored


@pytest.mark.unit
def test_validation_offload_refreshes_ddp_ignores_for_lazy_buffers():
    model = make_model()
    model._eval_asr_model = torch.nn.Module()
    model._eval_asr_model.register_buffer('initial_state', torch.ones(2))
    model._mark_validation_models_ddp_ignored()

    ddp_model = object.__new__(torch.nn.parallel.DistributedDataParallel)
    torch.nn.Module.__init__(ddp_model)
    ddp_model.broadcast_buffers = True
    ddp_model.module = model
    ddp_model.parameters_to_ignore = set(model._ddp_params_and_buffers_to_ignore)
    ddp_model._assign_modules_buffers()
    model._trainer = SimpleNamespace(strategy=SimpleNamespace(model=ddp_model))

    model._eval_asr_model.register_buffer('lazy_state', torch.ones(2))
    model._offload_validation_models()
    ddp_model._assign_modules_buffers()

    assert '_eval_asr_model.lazy_state' not in ddp_model.named_module_buffers
    assert 'training_parameter' in dict(model.named_parameters())
