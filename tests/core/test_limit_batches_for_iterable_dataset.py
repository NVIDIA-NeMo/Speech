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
from types import SimpleNamespace

import pytest
import torch

from nemo.core.classes.modelPT import ModelPT


class _SizedIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(range(self.num_samples))

    def __len__(self):
        return self.num_samples


class _UnsizedIterableDataset(torch.utils.data.IterableDataset):
    def __iter__(self):
        return iter(range(10))


class _NotImplementedLenIterableDataset(_UnsizedIterableDataset):
    # Mirrors the Hugging Face streaming datasets' missing-length contract.
    def __len__(self):
        raise NotImplementedError


class _BrokenLenIterableDataset(_UnsizedIterableDataset):
    def __len__(self):
        raise ValueError("broken length")


class _MapDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        return idx

    def __len__(self):
        return 10


def _set_limit(dataset, batch_size, world_size=1, stage="train", limit_train_batches=1.0, limit_val_batches=1.0):
    trainer = SimpleNamespace(limit_train_batches=limit_train_batches, limit_val_batches=limit_val_batches)
    model = SimpleNamespace(_trainer=trainer, world_size=world_size)
    dataloader = None if dataset is None else torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    ModelPT._set_limit_batches_for_iterable_dataset(model, dataloader, batch_size, stage=stage)
    return trainer


@pytest.mark.unit
@pytest.mark.parametrize(
    "limit,world_size,expected",
    [
        (1.0, 1, 25),  # ceil(100 / 4)
        (0.5, 1, 12),  # int(0.5 * 25)
        (1.0, 3, 9),  # ceil(100 / 3 / 4)
    ],
)
def test_sized_iterable_float_limit_is_converted_to_num_batches(limit, world_size, expected):
    trainer = _set_limit(_SizedIterableDataset(100), batch_size=4, world_size=world_size, limit_train_batches=limit)
    assert trainer.limit_train_batches == expected
    assert isinstance(trainer.limit_train_batches, int)


@pytest.mark.unit
def test_sized_iterable_int_limit_is_kept():
    trainer = _set_limit(_SizedIterableDataset(100), batch_size=4, limit_train_batches=7)
    assert trainer.limit_train_batches == 7


@pytest.mark.unit
@pytest.mark.parametrize("dataset_cls", [_UnsizedIterableDataset, _NotImplementedLenIterableDataset])
@pytest.mark.parametrize("limit", [1.0, 0.5, 3])
def test_unsized_iterable_limit_is_kept(dataset_cls, limit):
    trainer = _set_limit(dataset_cls(), batch_size=None, limit_train_batches=limit)
    assert trainer.limit_train_batches == limit
    assert type(trainer.limit_train_batches) is type(limit)


@pytest.mark.unit
@pytest.mark.parametrize("stage", ["train", "val"])
def test_sized_iterable_without_batch_size_raises_actionable_error(stage):
    with pytest.raises(ValueError, match=f"limit_{stage}_batches.*no 'batch_size'"):
        _set_limit(_SizedIterableDataset(100), batch_size=None, stage=stage)
    # An int limit doesn't need batch_size.
    trainer = _set_limit(_SizedIterableDataset(100), batch_size=None, stage=stage, **{f"limit_{stage}_batches": 5})
    assert getattr(trainer, f"limit_{stage}_batches") == 5


@pytest.mark.unit
def test_other_len_errors_propagate():
    with pytest.raises(ValueError, match="broken length"):
        _set_limit(_BrokenLenIterableDataset(), batch_size=4, limit_train_batches=1.0)
    # An int limit returns before len() is called.
    assert _set_limit(_BrokenLenIterableDataset(), batch_size=4, limit_train_batches=3).limit_train_batches == 3


@pytest.mark.unit
@pytest.mark.parametrize("dataset", [None, _MapDataset()])
def test_no_dataloader_or_map_dataset_is_noop(dataset):
    trainer = _set_limit(dataset, batch_size=4, limit_train_batches=0.5)
    assert trainer.limit_train_batches == 0.5


@pytest.mark.unit
def test_val_stage_updates_only_limit_val_batches():
    trainer = _set_limit(_SizedIterableDataset(100), batch_size=4, stage="val", limit_val_batches=1.0)
    assert trainer.limit_val_batches == 25
    assert trainer.limit_train_batches == 1.0


@pytest.mark.unit
def test_no_trainer_is_noop():
    model = SimpleNamespace(_trainer=None, world_size=1)
    dataloader = torch.utils.data.DataLoader(_SizedIterableDataset(100), batch_size=4)
    ModelPT._set_limit_batches_for_iterable_dataset(model, dataloader, 4)
