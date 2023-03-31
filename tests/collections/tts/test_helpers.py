# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
import torch

from nemo.collections.tts.parts.utils.helpers import regulate_len, sort_tensor, unsort_tensor


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
