# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
from nemo.collections.common.tokenizers.tokenizer_utils import get_tokenizer


def test_chat_template():
    transformers = pytest.importorskip("transformers")
    path = "/home/TestData/akoumparouli/tokenizer_with_chat_template/"
    tokenizers = [get_tokenizer(path), transformers.AutoTokenizer.from_pretrained(path)]
    prompt = "Give me a short introduction to pytest."
    messages = [{"role": "system", "content": "You are a helpful CI assistant."}, {"role": "user", "content": prompt}]
    texts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for tokenizer in tokenizers
    ]
    assert texts[0] == texts[1]


def test_throws_chat_template():
    path = "/home/TestData/akoumparouli/tokenizer_without_chat_template/"
    tokenizer = get_tokenizer(path)
    prompt = "Give me a short introduction to pytest."
    messages = [{"role": "system", "content": "You are a helpful CI assistant."}, {"role": "user", "content": prompt}]
    try:
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except ValueError as e:
        assert 'Cannot use chat template functions because tokenizer.chat_template is not set' in str(e)
