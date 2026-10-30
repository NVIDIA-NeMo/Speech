# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock

import pytest

from nemo.collections.common.tokenizers.canary_tokenizer import CANARY2_BOCTX, CANARY_BOS, CanaryTokenizer


@pytest.fixture
def tokenizer():
    tokenizer = object.__new__(CanaryTokenizer)
    tokenizer.special_tokens = {
        token: idx
        for idx, token in enumerate(
            [CANARY2_BOCTX, CANARY_BOS, '<|emo:undefined|>', '<|en|>', '<|de|>', '<|pnc|>', '<|endoftext|>']
        )
    }
    tokenizer.tokenizers_dict = {'en': None, 'de': None}
    tokenizer.text_to_ids = Mock(return_value=[99])
    return tokenizer


@pytest.mark.unit
@pytest.mark.parametrize(
    'text',
    [
        CANARY2_BOCTX + 'context',
        CANARY2_BOCTX + 'context' + CANARY_BOS,
        CANARY2_BOCTX + 'context' + CANARY_BOS + '<|emo:undefined|><|en|>',
        CANARY2_BOCTX + 'context' + CANARY_BOS + '<|emo:undefined|><|en|><|xx|>',
        CANARY2_BOCTX + 'context' + CANARY_BOS + '<|emo:undefined|><|en|><|de',
    ],
)
def test_canary_context_rejects_malformed_prompt(tokenizer, text):
    with pytest.raises(ValueError, match='Canary'):
        tokenizer._tokenize_special_prompt(text)
    tokenizer.text_to_ids.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize('context', ['', 'hello', 'hello <|en|> > world'])
def test_canary_context_uses_target_language_from_header(tokenizer, context):
    header = CANARY_BOS + '<|emo:undefined|><|en|><|de|><|pnc|>'
    actual = tokenizer._tokenize_special_prompt(CANARY2_BOCTX + context + header)
    assert actual == [0] + ([99] if context else []) + [1, 2, 3, 4, 5]
    if context:
        tokenizer.text_to_ids.assert_called_once_with(context, 'de')
    else:
        tokenizer.text_to_ids.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize('text', ['<|en|>trailing', '<|unknown|>', '<|en', 'plain text'])
def test_canary_special_prompt_rejects_invalid_tokens(tokenizer, text):
    with pytest.raises(ValueError, match='Canary'):
        tokenizer._tokenize_special_prompt(text)


@pytest.mark.unit
def test_canary_legacy_special_prompt(tokenizer):
    assert tokenizer._tokenize_special_prompt(CANARY_BOS + '<|en|><|pnc|>') == [1, 3, 5]
