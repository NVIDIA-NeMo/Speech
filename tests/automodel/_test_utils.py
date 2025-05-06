# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.llm.gpt.model.hf_auto_model_for_causal_lm import count_tail_padding


@pytest.mark.parametrize(
    "labels,expected,ignore_label",
    [
        # 1. Example given in the docstring (default ignore_label = -100)
        (
            torch.tensor(
                [
                    [-100, 1, 1, -100, -100],  # 2 tail -100s
                    [-100, -100, 2, 3, 4],  # 0 tail -100s
                    [5, 6, -100, -100, -100],  # 3 tail -100s
                ],
                dtype=torch.long,
            ),
            5,
            -100,
        ),
        # 2. No ignore labels at all → expect 0
        (
            torch.tensor(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ],
                dtype=torch.long,
            ),
            0,
            -100,
        ),
        # 3. Entire rows are ignore_label (here ignore_label = 0)
        (
            torch.zeros((3, 4), dtype=torch.long),  # every token is 0
            12,  # 3 * 4
            0,
        ),
        # 4. Mixed case with different sequence lengths, custom ignore_label
        (
            torch.tensor(
                [
                    [9, 8, -1, -1],
                    [-1, -1, -1, -1],
                    [7, 6, 5, 4],
                ],
                dtype=torch.long,
            ),
            6,  # 2 + 4 + 0
            -1,
        ),
    ],
)
def test_count_tail_padding(labels, expected, ignore_label):
    out = count_tail_padding(labels, ignore_label=ignore_label)
    # Function returns a 0‑D tensor; convert to Python int for comparison
    assert int(out) == expected
