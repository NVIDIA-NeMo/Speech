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

from nemo.collections.common.prompts.gemma import GemmaPromptFormatter


def test_gemma_prompt_formatter_training(bpe_tokenizer):
    formatter = GemmaPromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids", "answer_ids", "mask"}
    # fmt: off
    assert bpe_tokenizer.ids_to_text(ans["input_ids"].tolist()) == '<start_of_turn>user\nTEST<end_of_turn>\n<start_of_turn>model\n TEST<end_of_turn>\n'
    assert bpe_tokenizer.ids_to_text(ans["context_ids"].tolist()) == '<start_of_turn>user\nTEST<end_of_turn>\n<start_of_turn>model\n'
    assert bpe_tokenizer.ids_to_text(ans["answer_ids"].tolist()) == 'TEST<end_of_turn>\n'
    assert ans["mask"].shape[0] == ans["input_ids"].shape[0]
    # fmt: on


def test_gemma_prompt_formatter_inference(bpe_tokenizer):
    formatter = GemmaPromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids"}
    # fmt: off
    assert ans["input_ids"].tolist() == ans["context_ids"].tolist()
    assert bpe_tokenizer.ids_to_text(ans["input_ids"].tolist()) == '<start_of_turn>user\nTEST<end_of_turn>\n<start_of_turn>model\n'
    # fmt: on


def test_gemma_prompt_formatter_training_bos_eos_inserted_only_once_in_multiturn(bpe_tokenizer):
    formatter = GemmaPromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "TEST"}},
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "TEST"}},
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "TEST"}},
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "TEST"}},
        ]
    )

    assert (ans["input_ids"] == bpe_tokenizer.bos).sum() == 1
    assert (ans["input_ids"] == bpe_tokenizer.eos).sum() == 1
    assert ans["input_ids"][0] == bpe_tokenizer.bos
    assert ans["input_ids"][-1] == bpe_tokenizer.eos

    assert (ans["context_ids"] == bpe_tokenizer.bos).sum() == 1
    assert (ans["context_ids"] == bpe_tokenizer.eos).sum() == 0
    assert ans["context_ids"][0] == bpe_tokenizer.bos

    assert (ans["answer_ids"] == bpe_tokenizer.bos).sum() == 0
    assert (ans["answer_ids"] == bpe_tokenizer.eos).sum() == 1
    assert ans["answer_ids"][-1] == bpe_tokenizer.eos
