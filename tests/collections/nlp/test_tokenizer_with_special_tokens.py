# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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


from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

TOKENIZER_SPM_FILE = '/home/TestData/nlp/tokenizer_with_special_tokens/tokenizer.model'
SPECIAL_TOKENS = [
    '<s>',
    '</s>',
    '[INST]',
    '[/INST]',
    '[TOOL_CALLS]',
    '[AVAILABLE_TOOLS]',
    '[/AVAILABLE_TOOLS]',
    '[TOOL_RESULTS]',
    '[/TOOL_RESULTS]',
]


def _build_tokenizer(spm_file, special_tokens):
    tokenizer_cfg = {
        "library": "sentencepiece",
        "type": None,
        "vocab_file": None,
        "merge_file": None,
        "delimiter": None,
        "sentencepiece_legacy": True,
        "special_tokens": special_tokens,
    }
    return get_nmt_tokenizer(
        library=tokenizer_cfg['library'],
        model_name=tokenizer_cfg.get("type", None),
        use_fast=tokenizer_cfg.get("use_fast", False),
        delimiter=tokenizer_cfg.get("delimiter", None),
        special_tokens=tokenizer_cfg.get("special_tokens", None),
        trust_remote_code=tokenizer_cfg.get("trust_remote_code", False),
        tokenizer_model=spm_file,
        legacy=True,
    )


def test_spm_with_special_tokens() -> None:
    tokenizer = _build_tokenizer(TOKENIZER_SPM_FILE, SPECIAL_TOKENS)
    assert tokenizer.text_to_ids('[INST]') == [3]
    for i, special_token in enumerate(SPECIAL_TOKENS):
        assert special_token in tokenizer.special_token_to_id, f'Expected {special_token} to be a special token'
        assert tokenizer.special_token_to_id[special_token] == i + 1


def test_trim_spm_separator_after_special_token():
    tokenizer = _build_tokenizer(TOKENIZER_SPM_FILE, SPECIAL_TOKENS)
    tokenizer.text_to_ids('<s>[INST] Who') == [1, 3, 7294]
    tokenizer.trim_spm_separator_after_special_token = False
    tokenizer.text_to_ids('<s>[INST] Who') == [1, 3, 29473, 7294]


def test_text_to_tokens_with_trim_spm_separator_after_special_token():
    tokenizer = _build_tokenizer(TOKENIZER_SPM_FILE, SPECIAL_TOKENS)
    text = "<s>[INST] Who are you?[/INST] This is a response</s>[INST] I'll ask again who are you?[/INST] I'm not a who</s>"
    tokenized = tokenizer.text_to_tokens(text)
    assert tokenized == [
        '<s>',
        '[INST]',
        '▁Who',
        '▁are',
        '▁you',
        '?',
        '[/INST]',
        '▁This',
        '▁is',
        '▁a',
        '▁response',
        '</s>',
        '[INST]',
        '▁I',
        "'",
        'll',
        '▁ask',
        '▁again',
        '▁who',
        '▁are',
        '▁you',
        '?',
        '[/INST]',
        '▁I',
        "'",
        'm',
        '▁not',
        '▁a',
        '▁who',
        '</s>',
    ]
