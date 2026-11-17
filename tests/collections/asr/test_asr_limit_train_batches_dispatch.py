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
"""Which training configs convert a float ``limit_train_batches`` into a batch count."""

from types import MethodType, SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel
from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models_prompt import EncDecHybridRNNTCTCBPEModelWithPrompt
from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt
from nemo.collections.asr.models.transformer_bpe_models import EncDecTransfModelBPE
from nemo.core.classes.modelPT import ModelPT


class _SizedIterableDataset(torch.utils.data.IterableDataset):
    def __iter__(self):
        return iter(range(16))

    def __len__(self):
        return 16


class _UnsizedIterableDataset(torch.utils.data.IterableDataset):
    def __iter__(self):
        return iter(range(16))


def _setup(model_cls, dataset, is_tarred, limit_train_batches, world_size=2):
    model = SimpleNamespace(
        _trainer=SimpleNamespace(limit_train_batches=limit_train_batches),
        world_size=world_size,
        _update_dataset_config=lambda dataset_name, config: None,
        _setup_dataloader_from_config=lambda config: torch.utils.data.DataLoader(dataset, batch_size=2),
    )
    model._set_limit_batches_for_iterable_dataset = MethodType(ModelPT._set_limit_batches_for_iterable_dataset, model)
    model_cls.setup_training_data(model, DictConfig({'is_tarred': is_tarred, 'batch_size': 2, 'shuffle': True}))
    return model._trainer.limit_train_batches


_TARRED_GATED_MODELS = [
    EncDecTransfModelBPE,
    EncDecMultiTaskModel,
    EncDecRNNTBPEModelWithPrompt,
    EncDecHybridRNNTCTCBPEModelWithPrompt,
]


@pytest.mark.unit
@pytest.mark.parametrize("model_cls", _TARRED_GATED_MODELS)
@pytest.mark.parametrize("limit", [1.0, 0.5])
def test_non_tarred_iterable_keeps_float_limit(model_cls, limit):
    # A non-tarred ConcatDataset is iterable and reports a rank-local length; it must keep Lightning's fraction.
    result = _setup(model_cls, _SizedIterableDataset(), is_tarred=False, limit_train_batches=limit)
    assert result == limit
    assert type(result) is float


@pytest.mark.unit
@pytest.mark.parametrize("model_cls", _TARRED_GATED_MODELS)
def test_tarred_sized_dataset_converts_global_length(model_cls):
    # NeMo tarred datasets report the global manifest length: ceil(16 / world_size 2 / batch_size 2) = 4.
    assert _setup(model_cls, _SizedIterableDataset(), is_tarred=True, limit_train_batches=1.0) == 4
    assert _setup(model_cls, _SizedIterableDataset(), is_tarred=True, limit_train_batches=3) == 3


@pytest.mark.unit
@pytest.mark.parametrize("model_cls", _TARRED_GATED_MODELS)
def test_tarred_unsized_lhotse_dataset_keeps_float_limit(model_cls):
    assert _setup(model_cls, _UnsizedIterableDataset(), is_tarred=True, limit_train_batches=1.0) == 1.0
