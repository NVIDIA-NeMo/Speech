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
import torch

from nemo.collections.common.prompts.qwen import QwenPromptFormatter


def test_qwen_prompt_formatter_training(bpe_tokenizer):
    formatter = QwenPromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids", "answer_ids", "mask"}
    # fmt: off
    # The test tokenizer inserts an extra space, but it was verified that AutoTokenizer("Qwen/Qwen3-1.7B") doesn't.
    assert bpe_tokenizer.ids_to_text(ans["input_ids"].tolist()) == '<|im_start|>user\nTEST<|im_end|>\n <|im_start|>assistant\nTEST<|im_end|>\n'
    assert bpe_tokenizer.ids_to_text(ans["context_ids"].tolist()) == '<|im_start|>user\nTEST<|im_end|>\n'
    assert bpe_tokenizer.ids_to_text(ans["answer_ids"].tolist()) == '<|im_start|>assistant\nTEST<|im_end|>\n'
    assert torch.is_tensor(ans["mask"])
    # fmt: on


def test_qwen_prompt_formatter_inference(bpe_tokenizer):
    formatter = QwenPromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids"}
    # fmt: off
    # The test tokenizer inserts an extra space, but it was verified that AutoTokenizer("Qwen/Qwen3-1.7B") doesn't.
    assert ans["input_ids"].tolist() == ans["context_ids"].tolist()
    assert bpe_tokenizer.ids_to_text(ans["input_ids"].tolist()) == '<|im_start|>user\nTEST<|im_end|>\n <|im_start|>assistant\n'
    # fmt: on
