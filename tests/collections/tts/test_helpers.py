# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from omegaconf import OmegaConf

from nemo.collections.tts.parts.utils.helpers import get_max_steps, regulate_len, sort_tensor, unsort_tensor


def sample_duration_input(max_length=64, group_size=2, batch_size=3):
    generator = torch.Generator()
    generator.manual_seed(0)
    lengths = torch.randint(max_length // 4, max_length - 7, (batch_size,), generator=generator)
    durs = torch.ones(batch_size, max_length) * group_size
    durs[0, lengths[0]] += 1
    durs[2, lengths[2]] -= 1
    enc = torch.randint(16, 64, (batch_size, max_length, 17))
    return durs, enc, lengths


@pytest.mark.unit
def test_sort_unsort():
    durs_in, enc_in, dur_lens = sample_duration_input(batch_size=13)
    print("In: ", enc_in)
    sorted_enc, sorted_len, sorted_ids = sort_tensor(enc_in, dur_lens)
    unsorted_enc = unsort_tensor(sorted_enc, sorted_ids)
    print("Out: ", unsorted_enc)

    assert torch.all(unsorted_enc == enc_in)


@pytest.mark.unit
def test_regulate_len():
    group_size = 2
    durs_in, enc_in, dur_lens = sample_duration_input(group_size=group_size)
    enc_out, lens_out = regulate_len(durs_in, enc_in, group_size=group_size, dur_lens=dur_lens)
    # make sure lens_out are rounded
    sum_diff = lens_out - torch.mul(lens_out // group_size, group_size)
    assert sum_diff.sum(dim=0) == 0
    # make sure all round-ups are <= group_size
    diff = lens_out - durs_in.sum(dim=1)
    assert torch.max(diff) < group_size


class _UnsizedIterableDataset(torch.utils.data.IterableDataset):
    def __iter__(self):
        return iter(range(10))


class _SizedIterableDataset(_UnsizedIterableDataset):
    def __len__(self):
        return 10


def _trainer(limit_train_batches=1.0, accumulate_grad_batches=1):
    return SimpleNamespace(
        accumulate_grad_batches=accumulate_grad_batches,
        limit_train_batches=limit_train_batches,
        num_devices=1,
        num_nodes=1,
    )


@pytest.mark.unit
@pytest.mark.parametrize("dataset", [_SizedIterableDataset(), _UnsizedIterableDataset()])
def test_get_max_steps_explicit_keys_ignore_dataset(dataset):
    dl = torch.utils.data.DataLoader(dataset, batch_size=4)
    assert get_max_steps(OmegaConf.create({"max_steps": 7, "max_epochs": 2}), _trainer(), dl) == 7
    assert get_max_steps(OmegaConf.create({"max_epochs": 2, "steps_per_epoch": 5}), _trainer(), dl) == 10


@pytest.mark.unit
def test_get_max_steps_requires_max_steps_or_max_epochs():
    dl = torch.utils.data.DataLoader(_SizedIterableDataset(), batch_size=4)
    with pytest.raises(ValueError, match="Must specify 'max_steps' or 'max_epochs'"):
        get_max_steps(OmegaConf.create({}), _trainer(), dl)


@pytest.mark.unit
@pytest.mark.parametrize(
    "trainer,drop_last,expected",
    [
        (_trainer(), False, 6),  # 2 epochs * ceil(10 / 4)
        (_trainer(), True, 4),  # 2 epochs * floor(10 / 4)
        (_trainer(limit_train_batches=2), False, 4),  # int limit caps steps per epoch
        (_trainer(accumulate_grad_batches=2), False, 4),  # 2 epochs * ceil(3 / 2)
    ],
)
def test_get_max_steps_from_sized_dataset(trainer, drop_last, expected):
    dl = torch.utils.data.DataLoader(_SizedIterableDataset(), batch_size=4, drop_last=drop_last)
    assert get_max_steps(OmegaConf.create({"max_epochs": 2}), trainer, dl) == expected


@pytest.mark.unit
def test_get_max_steps_from_unsized_dataset_raises():
    dl = torch.utils.data.DataLoader(_UnsizedIterableDataset(), batch_size=4)
    with pytest.raises(ValueError, match="no length.*'max_steps'.*'steps_per_epoch'"):
        get_max_steps(OmegaConf.create({"max_epochs": 2}), _trainer(), dl)
