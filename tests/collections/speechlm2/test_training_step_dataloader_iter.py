# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
"""speechlm2 models must use ``training_step(self, dataloader_iter)`` so that Lightning does not prefetch a batch.

With the default ``training_step(self, batch, batch_idx)``, Lightning uses ``_PrefetchDataFetcher``, which reads one
batch ahead when the train dataloader length is unknown. The saved StatefulDataLoader state is then one batch ahead
and the resumed run skips a batch.
"""
import importlib
import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from lightning.pytorch.loops.fetchers import _DataLoaderIterDataFetcher
from lightning.pytorch.loops.utilities import _select_data_fetcher
from lightning.pytorch.trainer.states import RunningStage

MODELS = [
    ("nemo.collections.speechlm2.models.salm", "SALM"),
    ("nemo.collections.speechlm2.models.salm_asr_decoder", "SALMWithAsrDecoder"),
    ("nemo.collections.speechlm2.models.salm_automodel", "SALMAutomodel"),
    ("nemo.collections.speechlm2.models.duplex_ear_tts", "DuplexEARTTS"),
    ("nemo.collections.speechlm2.models.duplex_stt_model", "DuplexSTTModel"),
    ("nemo.collections.speechlm2.models.duplex_s2s_model", "DuplexS2SModel"),
    ("nemo.collections.speechlm2.models.duplex_s2s_speech_decoder_model", "DuplexS2SSpeechDecoderModel"),
]


def _load(module_name, cls_name):
    module = importlib.import_module(module_name)
    return module, getattr(module, cls_name)


@pytest.mark.unit
@pytest.mark.parametrize("module_name,cls_name", MODELS)
def test_training_step_selects_non_prefetching_data_fetcher(module_name, cls_name):
    _, cls = _load(module_name, cls_name)
    assert list(inspect.signature(cls.training_step).parameters) == ["self", "dataloader_iter"]
    fake_module = SimpleNamespace(training_step=cls.training_step.__get__(object()))
    trainer = SimpleNamespace(lightning_module=fake_module)
    assert isinstance(_select_data_fetcher(trainer, RunningStage.TRAINING), _DataLoaderIterDataFetcher)


@pytest.mark.unit
@pytest.mark.parametrize("module_name,cls_name", MODELS)
def test_training_step_reads_one_batch_and_forwards_it(monkeypatch, module_name, cls_name):
    module, cls = _load(module_name, cls_name)
    read_calls = []

    def fake_read_batch(dataloader_iter, model):
        batch, batch_idx, _ = next(dataloader_iter)
        read_calls.append((batch, batch_idx, model))
        return batch, batch_idx

    monkeypatch.setattr(module, "read_batch", fake_read_batch)
    model = SimpleNamespace(_training_step_batch=Mock(return_value={"loss": 0.0}))
    dataloader_iter = iter([("batch0", 7, 0), ("batch1", 8, 0)])

    assert cls.training_step(model, dataloader_iter) == {"loss": 0.0}
    model._training_step_batch.assert_called_once_with("batch0", 7)
    assert read_calls == [("batch0", 7, model)]
    # The second batch must still be available: the step consumes exactly one batch.
    assert next(dataloader_iter) == ("batch1", 8, 0)
