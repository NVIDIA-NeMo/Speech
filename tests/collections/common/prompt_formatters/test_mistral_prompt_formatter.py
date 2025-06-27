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

from nemo.collections.common.prompts.mistral import MistralPromptFormatter


def test_mistral_prompt_formatter_training(bpe_tokenizer):
    formatter = MistralPromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids", "answer_ids", "mask"}
    # fmt: off
    assert bpe_tokenizer.ids_to_text(ans["input_ids"].tolist()) == '<s> [INST] TEST [/INST] TEST</s>'
    assert bpe_tokenizer.ids_to_text(ans["context_ids"].tolist()) == '<s> [INST] TEST [/INST]'
    assert bpe_tokenizer.ids_to_text(ans["answer_ids"].tolist()) == 'TEST</s>'
    assert ans["mask"].shape[0] == ans["input_ids"].shape[0]
    # fmt: on


def test_mistral_prompt_formatter_inference(bpe_tokenizer):
    formatter = MistralPromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids"}
    # fmt: off
    assert ans["input_ids"].tolist() == ans["context_ids"].tolist()
    assert bpe_tokenizer.ids_to_text(ans["input_ids"].tolist()) == '<s> [INST] TEST [/INST]'
    # fmt: on
