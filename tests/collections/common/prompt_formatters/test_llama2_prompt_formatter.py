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

from nemo.collections.common.prompts.llama import Llama2PromptFormatter


def test_llama2_prompt_formatter_training(bpe_tokenizer):
    formatter = Llama2PromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids", "answer_ids", "mask"}
    # fmt: off
    assert bpe_tokenizer.ids_to_text(ans["input_ids"].tolist()[1:-1]) == '[INST] TEST [/INST] TEST'
    assert bpe_tokenizer.ids_to_text(ans["context_ids"].tolist()[1:]) == '[INST] TEST [/INST]'
    assert bpe_tokenizer.ids_to_text(ans["answer_ids"].tolist()[:-1]) == 'TEST'
    assert ans["mask"].shape[0] == ans["input_ids"].shape[0]
    # fmt: on


def test_llama2_prompt_formatter_inference(bpe_tokenizer):
    formatter = Llama2PromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids"}
    # fmt: off
    assert ans["input_ids"].tolist() == ans["context_ids"].tolist()
    assert bpe_tokenizer.ids_to_text(ans["input_ids"].tolist()[1:]) == '[INST] TEST [/INST]'
    # fmt: on


def test_llama2_prompt_formatter_training_with_system(bpe_tokenizer):
    formatter = Llama2PromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "system_and_user", "slots": {"system": "TEST", "message": "TEST"}},
            {"role": "assistant", "slots": {"message": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids", "answer_ids", "mask"}
    # fmt: off
    assert bpe_tokenizer.ids_to_text(ans["input_ids"].tolist()[1:-1]) == '[INST] <<SYS>>\nTEST\n<</SYS>>\n\nTEST [/INST] TEST'
    assert bpe_tokenizer.ids_to_text(ans["context_ids"].tolist()[1:]) == '[INST] <<SYS>>\nTEST\n<</SYS>>\n\nTEST [/INST]'
    assert bpe_tokenizer.ids_to_text(ans["answer_ids"].tolist()[:-1]) == 'TEST'
    assert ans["mask"].shape[0] == ans["input_ids"].shape[0]
    # fmt: on


def test_llama2_prompt_formatter_inference_with_system(bpe_tokenizer):
    formatter = Llama2PromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "system_and_user", "slots": {"system": "TEST", "message": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids"}
    # fmt: off
    assert ans["input_ids"].tolist() == ans["context_ids"].tolist()
    assert bpe_tokenizer.ids_to_text(ans["input_ids"].tolist()[1:]) == '[INST] <<SYS>>\nTEST\n<</SYS>>\n\nTEST [/INST]'
    # fmt: on
