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
import dataclasses
from pathlib import Path

from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse.dataloader import LhotseDataLoadingConfig


REPO_ROOT = Path(__file__).parents[3]


def test_salm_automodel_uses_portable_moe_dispatcher():
    cfg = OmegaConf.load(REPO_ROOT / "examples/speechlm2/conf/salm_automodel.yaml")

    assert cfg.model.automodel_backend.dispatcher == "torch"


def test_salm_nemotron_keeps_silent_failure_guards():
    cfg = OmegaConf.load(REPO_ROOT / "examples/speechlm2/conf/salm_nemotron.yaml")
    train_ds = cfg.data.train_ds

    # bf16-true casts the LLM's FP32 tensors to BF16 without raising.
    assert cfg.trainer.precision == "bf16-automodel"
    # max_tokens is a filter: anything longer than a batch would be dropped silently.
    assert train_ds.max_tokens == train_ds.batch_tokens
    assert train_ds.use_packed_sequence_sampling is True
    assert train_ds.packing_buffer_size >= 1024


def test_salm_nemotron_train_ds_keys_are_known_to_the_dataloader():
    cfg = OmegaConf.load(REPO_ROOT / "examples/speechlm2/conf/salm_nemotron.yaml")
    known = {field.name for field in dataclasses.fields(LhotseDataLoadingConfig)}

    assert set(cfg.data.train_ds.keys()) <= known
