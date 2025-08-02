# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import torch

from nemo.collections.speechlm2.models.salm import replace_placeholders_and_build_targets


def test_replace_placeholders():
    # fmt: off
    PAD = 0
    AUDIO = 100
    input_ids = torch.tensor([
        [7  , AUDIO, 1, 2    , AUDIO, 1],
        [PAD,   PAD, 3, AUDIO, 4    , 5]  # note: left padding required
    ])
    loss_mask = torch.tensor([
        [False, False, False, False, False, True],  # predict last token
        [False, False, False, False, True , True]  # predict last two tokens
    ])
    embeds = torch.ones(2, 6, 2)
    embeds[1, :2] = 0  # note: indicate left padding
    # 3 embedding sequences with varying shapes, corresponding to 3 AUDIO tokens
    replacements = [
        torch.full((4, 2), fill_value=2.0),
        torch.full((3, 2), fill_value=3.0),
        torch.full((2, 2), fill_value=4.0),
    ]

    embeds_r, targets_r, attention_mask_r = replace_placeholders_and_build_targets(
        input_ids=input_ids,
        embeds=embeds,
        padding_id=PAD,
        placeholder_id=AUDIO,
        replacements=replacements,
        target_ids=input_ids.where(loss_mask, -100)
    )

    assert embeds_r.shape == (2, 11, 2)
    # batch item 0
    assert (embeds_r[0, 0]    == 1.0).all()  # 1=orig
    assert (embeds_r[0, 1:5]  == 2.0).all()  # 2=repl
    assert (embeds_r[0, 5:7]  == 1.0).all()  # 1=orig
    assert (embeds_r[0, 7:10] == 3.0).all()  # 3=repl
    assert (embeds_r[0, 10]   == 1.0).all()  # 1=orig
    # batch item 1
    assert (embeds_r[1, :6]   == 0.0).all()  # 0=pad
    assert (embeds_r[1, 6:7]  == 1.0).all()  # 1=orig
    assert (embeds_r[1, 7:9]  == 4.0).all()  # 4=repl
    assert (embeds_r[1, 9:]   == 1.0).all()  # 1=orig

    assert targets_r.shape == (2, 11)
    torch.testing.assert_close(
        targets_r,
        torch.tensor([
            [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 1],
            [-100, -100, -100, -100, -100, -100, -100, -100, -100, 4   , 5],
        ])
    )

    assert attention_mask_r.shape == (2, 11)
    torch.testing.assert_close(
        attention_mask_r,
        torch.tensor([
            [True, True, True, True, True,  True,  True,  True,  True,  True,  True],
            [False, False, False, False, False, False, True, True, True, True, True],
        ])
    )
    # fmt: on


def test_replace_placeholders_removes_excessive_left_padding():
    # fmt: off
    PAD = 0
    AUDIO = 100
    input_ids = torch.tensor([
        [7  , AUDIO, 1    , 2],
        [PAD,   PAD, AUDIO, 3]  # note: left padding required
    ])
    loss_mask = torch.tensor([
        [False, False, True , True],  # predict last two tokens
        [False, False, False, True]   # predict last token
    ])
    embeds = torch.ones(2, 4, 2)
    embeds[1, :2] = 0  # note: indicate left padding
    # 3 embedding sequences with varying shapes, corresponding to 3 AUDIO tokens
    replacements = [
        torch.full((3, 2), fill_value=2.0),
        torch.full((5, 2), fill_value=4.0),
    ]

    embeds_r, targets_r, attention_mask_r = replace_placeholders_and_build_targets(
        input_ids=input_ids,
        embeds=embeds,
        padding_id=PAD,
        placeholder_id=AUDIO,
        replacements=replacements,
        target_ids=input_ids.where(loss_mask, -100)
    )

    assert embeds_r.shape == (2, 6, 2)
    # batch item 0
    assert (embeds_r[0, 0  ]  == 1.0).all()  # 1=orig
    assert (embeds_r[0, 1:4]  == 2.0).all()  # 2=repl
    assert (embeds_r[0, 4: ]  == 1.0).all()  # 1=orig
    # batch item 1
    assert (embeds_r[1, :5]  == 4.0).all()  # 4=repl
    assert (embeds_r[1, 5 ]  == 1.0).all()  # 1=orig

    assert targets_r.shape == (2, 6)
    torch.testing.assert_close(
        targets_r,
        torch.tensor([
            [-100, -100, -100, -100,    1, 2],
            [-100, -100, -100, -100, -100, 3],
        ])
    )

    assert attention_mask_r.shape == (2, 6)
    torch.testing.assert_close(
        attention_mask_r,
        torch.tensor([
            [True, True, True, True, True, True],
            [True, True, True, True, True, True],
        ])
    )
    # fmt: on
