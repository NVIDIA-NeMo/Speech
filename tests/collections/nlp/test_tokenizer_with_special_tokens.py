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


def test_spm_with_special_tokens() -> None:
    special_tokens = [
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
    tokenizer_cfg = {
        "library": "sentencepiece",
        "type": None,
        "vocab_file": None,
        "merge_file": None,
        "delimiter": None,
        "sentencepiece_legacy": True,
        "special_tokens": special_tokens,
    }
    tokenizer = get_nmt_tokenizer(
        library=tokenizer_cfg['library'],
        model_name=tokenizer_cfg.get("type", None),
        use_fast=tokenizer_cfg.get("use_fast", False),
        delimiter=tokenizer_cfg.get("delimiter", None),
        special_tokens=tokenizer_cfg.get("special_tokens", None),
        trust_remote_code=tokenizer_cfg.get("trust_remote_code", False),
        tokenizer_model=TOKENIZER_SPM_FILE,
        legacy=True,
    )

    assert tokenizer.text_to_ids('[INST]') == [3]
    for i, special_token in enumerate(special_tokens):
        assert special_token in tokenizer.special_token_to_id, f'Expected {special_token} to be a special token'
        assert tokenizer.special_token_to_id[special_token] == i + 1
