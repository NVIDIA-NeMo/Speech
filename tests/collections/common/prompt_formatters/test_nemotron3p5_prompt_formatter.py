# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from nemo.collections.common.prompts import PromptFormatter
from nemo.collections.common.prompts.nemotron3p5 import Nemotron3p5PromptFormatter


def test_nemotron3p5_is_registered():
    assert PromptFormatter.resolve("nemotron3p5") is Nemotron3p5PromptFormatter


def test_nemotron3p5_training_basic(bpe_tokenizer_with_think):
    formatter = Nemotron3p5PromptFormatter(bpe_tokenizer_with_think)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "TEST"}},
        ]
    )

    assert set(ans) == {"input_ids", "context_ids", "answer_ids", "mask"}
    assert (
        bpe_tokenizer_with_think.ids_to_text(ans["input_ids"].tolist())
        == "<|im_start|>system\n<|im_end|>\n <|im_start|>user\nTEST<|im_end|>\n "
        "<|im_start|>assistant\n<think></think>TEST<|im_end|>\n"
    )
    assert ans["mask"].tolist() == [False] * len(ans["context_ids"]) + [True] * len(ans["answer_ids"])


def test_nemotron3p5_inference_generation_prompt(bpe_tokenizer_with_think):
    formatter = Nemotron3p5PromptFormatter(bpe_tokenizer_with_think)
    thinking = formatter.encode_dialog([{"role": "user", "slots": {"message": "TEST"}}], enable_thinking=True)
    no_thinking = formatter.encode_dialog([{"role": "user", "slots": {"message": "TEST"}}], enable_thinking=False)

    assert (
        bpe_tokenizer_with_think.ids_to_text(thinking["input_ids"].tolist())
        == "<|im_start|>system\n<|im_end|>\n <|im_start|>user\nTEST<|im_end|>\n "
        "<|im_start|>assistant\n<think>\n"
    )
    assert (
        bpe_tokenizer_with_think.ids_to_text(no_thinking["input_ids"].tolist())
        == "<|im_start|>system\n<|im_end|>\n <|im_start|>user\nTEST<|im_end|>\n "
        "<|im_start|>assistant\n<think></think>"
    )
