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

import pytest
import torch

from nemo.collections.speechlm2.parts.text_utils import tokens_to_str


class _Tokenizer:
    bos = 1
    eos = 2

    def ids_to_text(self, ids):
        return " ".join(str(int(i)) for i in ids)


PAD_ID = 0


@pytest.mark.parametrize("eval_text_turn_taking", [False, True])
@pytest.mark.parametrize("show_eot_timestamps", [False, True])
def test_tokens_to_str_ignores_tokens_past_the_row_length(eval_text_turn_taking, show_eot_timestamps):
    """A row padded to the batch width decodes like the same row alone."""
    tokenizer = _Tokenizer()
    tokens = torch.tensor([[1, 5, 6, 0, 2, 7, 1, 8], [1, 5, 0, 2, 9, 1, 8, 0]])
    lengths = torch.tensor([8, 4])
    kwargs = dict(
        tokenizer=tokenizer,
        pad_id=PAD_ID,
        eval_text_turn_taking=eval_text_turn_taking,
        show_eot_timestamps=show_eot_timestamps,
    )

    batched = tokens_to_str(tokens, lengths, **kwargs)
    full_row = tokens_to_str(tokens[:1], lengths[:1], **kwargs)
    short_row = tokens_to_str(tokens[1:, :4], lengths[1:], **kwargs)

    assert batched[0] == full_row[0]
    assert batched[1] == short_row[0]
    assert "9" not in batched[1] and "8" not in batched[1]
