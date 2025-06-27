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

from nemo.collections.common.prompts.canary import CanaryPromptFormatter


def test_canary_prompt_formatter_training(canary_tokenizer):
    formatter = CanaryPromptFormatter(canary_tokenizer)
    ans = formatter.encode_dialog(
        [
            {
                "role": "user",
                "slots": {
                    "source_lang": "<|en|>",
                    "target_lang": "<|en|>",
                    "task": "<|transcribe|>",
                    "pnc": "<|pnc|>",
                    "prompt_language": "spl_tokens",
                },
            },
            {"role": "assistant", "slots": {"text": "TEST", "prompt_language": "en"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids", "answer_ids", "mask"}
    # fmt: off
    assert canary_tokenizer.ids_to_text(ans["input_ids"].tolist()) == '<|startoftranscript|><|en|><|transcribe|><|en|><|pnc|> TEST<|endoftext|>'
    assert canary_tokenizer.ids_to_text(ans["context_ids"].tolist()) == '<|startoftranscript|><|en|><|transcribe|><|en|><|pnc|>'
    assert canary_tokenizer.ids_to_text(ans["answer_ids"].tolist()) == ' TEST<|endoftext|>'
    assert ans["mask"].shape[0] == ans["input_ids"].shape[0]
    # fmt: on


def test_canary_prompt_formatter_inference(canary_tokenizer):
    formatter = CanaryPromptFormatter(canary_tokenizer)
    ans = formatter.encode_dialog(
        [
            {
                "role": "user",
                "slots": {
                    "source_lang": "<|en|>",
                    "target_lang": "<|en|>",
                    "task": "<|transcribe|>",
                    "pnc": "<|pnc|>",
                    "prompt_language": "spl_tokens",
                },
            },
        ]
    )
    assert set(ans) == {"input_ids", "context_ids"}
    # fmt: off
    assert ans["input_ids"].tolist() == ans["context_ids"].tolist()
    assert canary_tokenizer.ids_to_text(ans["input_ids"].tolist()) == '<|startoftranscript|><|en|><|transcribe|><|en|><|pnc|>'
    # fmt: on
