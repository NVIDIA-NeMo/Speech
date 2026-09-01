# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from types import SimpleNamespace

import pytest
import torch
from lhotse import CutSet
from lhotse.testing.dummies import DummyManifest
from lightning.pytorch.utilities import CombinedLoader
from omegaconf import DictConfig

import nemo.collections.speechlm2.data.datamodule as datamodule_module
from nemo.collections.common.data.fallback import FallbackDataset
from nemo.collections.common.data.lhotse.broadcasting import BroadcastingDataLoader
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer, create_spt_model
from nemo.collections.speechlm2.data import DataModule


@pytest.fixture
def data_config(tmp_path):
    ap, cp = tmp_path / "audio", str(tmp_path) + "/{tag}_cuts.jsonl.gz"

    def _assign(k, v):
        def _inner(obj):
            setattr(obj, k, v)
            return obj

        return _inner

    for tag in ("train", "val_set_0", "val_set_1"):
        (
            DummyManifest(CutSet, begin_id=0, end_id=2, with_data=True)
            .map(_assign("tag", tag))
            .save_audios(ap)
            .drop_in_memory_data()
            .to_file(cp.format(tag=tag))
        )

    return DictConfig(
        {
            "train_ds": {
                "input_cfg": [
                    {
                        "type": "lhotse",
                        "cuts_path": cp.format(tag="train"),
                    }
                ],
                "batch_size": 2,
            },
            "validation_ds": {
                "datasets": {
                    "val_set_0": {"cuts_path": cp.format(tag="val_set_0")},
                    "val_set_1": {"cuts_path": cp.format(tag="val_set_1")},
                },
                "batch_size": 2,
            },
        }
    )


@pytest.fixture
def tokenizer(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("tok")
    text_path = tmpdir / "text.txt"
    text_path.write_text("\n".join(chr(i) for i in range(256)))
    create_spt_model(
        text_path,
        vocab_size=512,
        sample_size=-1,
        do_lower_case=False,
        output_dir=str(tmpdir),
        bos=True,
        eos=True,
        remove_extra_whitespaces=True,
    )
    return SentencePieceTokenizer(str(tmpdir / "tokenizer.model"))


class Identity(torch.utils.data.Dataset):
    def __getitem__(self, item):
        return item


class PolicyAwareDataset(torch.utils.data.Dataset):
    def __init__(self, calls, fault_tolerant_audio_loading=None):
        self.calls = calls
        self.fault_tolerant_audio_loading = fault_tolerant_audio_loading

    def __getitem__(self, item):
        return item

    def with_fault_tolerant_audio_loading(self, enabled):
        self.calls.append(enabled)
        return PolicyAwareDataset(self.calls, fault_tolerant_audio_loading=enabled)


def test_datamodule_train_dataloader(data_config, tokenizer):
    data = DataModule(data_config, tokenizer=tokenizer, dataset=Identity())
    dl = data.train_dataloader()
    assert isinstance(dl, (BroadcastingDataLoader, torch.utils.data.DataLoader))
    dli = iter(dl)

    batch = next(dli)
    assert isinstance(batch, CutSet)
    assert len(batch) == 2
    assert all(c.tag == "train" for c in batch)


@pytest.mark.parametrize("skip_missing_manifest_entries", [False, True])
@pytest.mark.parametrize("fault_tolerant_audio_loading", [False, True])
def test_datamodule_audio_loading_policy_is_independent_of_missing_manifest_policy(
    skip_missing_manifest_entries, fault_tolerant_audio_loading
):
    calls = []
    dataset = PolicyAwareDataset(calls)
    cfg = DictConfig(
        {
            "train_ds": {
                "skip_missing_manifest_entries": skip_missing_manifest_entries,
                "fault_tolerant_audio_loading": fault_tolerant_audio_loading,
            }
        }
    )
    datamodule = DataModule(cfg, tokenizer=None, dataset=dataset)
    training = datamodule._dataset_for_config(cfg.train_ds, training=True)
    configured = training.dataset if isinstance(training, FallbackDataset) else training
    assert isinstance(training, FallbackDataset) is fault_tolerant_audio_loading
    assert configured.fault_tolerant_audio_loading is fault_tolerant_audio_loading

    validation = datamodule._dataset_for_config(cfg.train_ds, training=False)
    assert isinstance(validation, PolicyAwareDataset)
    assert validation.fault_tolerant_audio_loading is fault_tolerant_audio_loading
    assert calls == [fault_tolerant_audio_loading, fault_tolerant_audio_loading]


def test_datamodule_fault_tolerant_audio_loading_defaults_true():
    dataset = PolicyAwareDataset([])
    cfg = DictConfig({"train_ds": {}})
    configured = DataModule(cfg, tokenizer=None, dataset=dataset)._dataset_for_config(cfg.train_ds, training=True)
    assert isinstance(configured, FallbackDataset)
    assert configured.dataset.fault_tolerant_audio_loading is True


def test_datamodule_train_dataloader_caches_broadcast_wrapper_and_passes_dp_group(data_config, tokenizer, monkeypatch):
    data = DataModule(data_config, tokenizer=tokenizer, dataset=Identity())
    mesh = SimpleNamespace(mesh_dim_names=())
    dp_group = object()
    source = object()
    calls = []

    monkeypatch.setattr(data, "_get_device_mesh", lambda: mesh)
    monkeypatch.setattr(data, "_get_dp_rank", lambda: 3)
    monkeypatch.setattr(data, "_get_world_size", lambda: 8)
    monkeypatch.setattr(data, "_get_dp_group", lambda: dp_group)
    monkeypatch.setattr(datamodule_module, "is_dp_source_rank", lambda candidate: candidate is mesh)

    def fake_get_lhotse_dataloader_from_config(**kwargs):
        calls.append(kwargs)
        return source

    monkeypatch.setattr(
        datamodule_module,
        "get_lhotse_dataloader_from_config",
        fake_get_lhotse_dataloader_from_config,
    )

    dl1 = data.train_dataloader()
    dl2 = data.train_dataloader()

    assert dl1 is dl2
    assert isinstance(dl1, BroadcastingDataLoader)
    assert dl1._source is source
    assert len(calls) == 1
    assert calls[0]["global_rank"] == 3
    assert calls[0]["world_size"] == 8
    assert calls[0]["dp_group"] is dp_group


def test_datamodule_train_dataloader_non_source_rank_does_not_build_source(data_config, tokenizer, monkeypatch):
    data = DataModule(data_config, tokenizer=tokenizer, dataset=Identity())
    mesh = SimpleNamespace(mesh_dim_names=())

    monkeypatch.setattr(data, "_get_device_mesh", lambda: mesh)
    monkeypatch.setattr(datamodule_module, "is_dp_source_rank", lambda candidate: False)

    def fail_if_called(**kwargs):
        raise AssertionError("non-source CP/TP ranks must not build a Lhotse source loader")

    monkeypatch.setattr(datamodule_module, "get_lhotse_dataloader_from_config", fail_if_called)

    dl = data.train_dataloader()

    assert isinstance(dl, BroadcastingDataLoader)
    assert dl._source is None


def test_datamodule_validation_dataloader(data_config, tokenizer):
    val_sets = {"val_set_0", "val_set_1"}
    data = DataModule(data_config, tokenizer=tokenizer, dataset=Identity())
    dl = data.val_dataloader()
    assert isinstance(dl, CombinedLoader)
    dli = iter(dl)

    batch, batch_idx, dataloader_idx = next(dli)
    assert isinstance(batch, dict)
    assert batch.keys() == val_sets
    for vs in val_sets:
        assert len(batch[vs]) == 2
        assert all(c.tag == vs for c in batch[vs])
