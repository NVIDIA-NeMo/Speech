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
import io

import pytest
import sentencepiece as spm

from nemo.collections.asr.parts.utils.timestamp_utils import get_words_offsets


@pytest.fixture(scope="module")
def sentencepiece_tokenizer():
    model = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(["hello world"]),
        model_writer=model,
        model_type="bpe",
        vocab_size=32,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    return spm.SentencePieceProcessor(model_proto=model.getvalue())


@pytest.mark.unit
@pytest.mark.parametrize(
    "tokens, expected",
    [
        (["▁hello", "▁", "<unk>", "▁world"], [("hello", 0, 1), (" ⁇ ", 2, 3), ("world", 3, 4)]),
        (["▁", "<unk>", "▁world"], [(" ⁇ ", 1, 2), ("world", 2, 3)]),
        (["▁", "<unk>"], [(" ⁇ ", 1, 2)]),
        (["▁hello", "▁"], [("hello", 0, 1)]),
        (["▁"], []),
        (["▁", "▁"], []),
        (["▁hello", "▁world"], [("hello", 0, 1), ("world", 1, 2)]),
    ],
)
def test_sentencepiece_empty_word_offsets(sentencepiece_tokenizer, tokens, expected):
    # Parakeet can emit a standalone ▁ before <unk>; ▁ decodes to empty text.
    encoded_offsets = [
        {"char": [token], "start_offset": i, "end_offset": i + 1, "start": i * 0.08, "end": (i + 1) * 0.08}
        for i, token in enumerate(tokens)
    ]
    char_offsets = [{**offset, "char": [sentencepiece_tokenizer.decode(offset["char"])]} for offset in encoded_offsets]

    offsets = get_words_offsets(
        char_offsets=char_offsets,
        encoded_char_offsets=encoded_offsets,
        decode_tokens_to_str=sentencepiece_tokenizer.decode,
        supported_punctuation={".", "!", "?"},
    )

    assert offsets == [
        {"word": word, "start_offset": start, "end_offset": end, "start": start * 0.08, "end": end * 0.08}
        for word, start, end in expected
    ]
