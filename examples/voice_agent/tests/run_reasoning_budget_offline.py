#!/usr/bin/env python3
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

"""Offline test for ReasoningBudgetLogitsProcessor using vllm.LLM.

Usage:
    python examples/voice_agent/tests/run_reasoning_budget_offline.py
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# Add the local NeMo directory to Python path.
nemo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(nemo_root))

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from nemo.agents.voice_agent.vllm.v1.sample.logits_processor.reasoning_budget_logits_processor import (
    ReasoningBudgetLogitsProcessor,
)

MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant. /think"},
    {"role": "user", "content": "Write a haiku about a cat"},
]

MESSAGES_MATH = [
    {"role": "system", "content": "You are a helpful assistant. /think"},
    {"role": "user", "content": "What is 25 * 37?"},
]


def main():
    print(f"Loading tokenizer: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print(f"Loading model with ReasoningBudgetLogitsProcessor...")
    llm = LLM(
        model=MODEL_NAME,
        logits_processors=[ReasoningBudgetLogitsProcessor],
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        max_model_len=10000,
    )

    # ------------------------------------------------------------------
    # Test 1: Single prompt with thinking_budget=64
    # ------------------------------------------------------------------
    prompt = tok.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)

    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=256,
        extra_args={
            "thinking_budget": 64,
            "thinking_budget_grace_period": 10,
        },
    )

    print(f"\n{'='*70}")
    print(f"Test 1: thinking_budget=64, max_tokens=256")
    print(f"Messages: {MESSAGES}")
    print(f"{'='*70}")

    outputs = llm.generate([prompt], [sampling_params])
    for output in outputs:
        generated_text = output.outputs[0].text
        num_tokens = len(output.outputs[0].token_ids)
        print(f"Prompt:  {output.prompt!r}")
        print(f"Output ({num_tokens} tokens):")
        print(f"  {generated_text}")
    print(f"{'='*70}")

    # ------------------------------------------------------------------
    # Test 2: Two prompts with different budgets and custom end tokens
    # ------------------------------------------------------------------
    prompt_1 = tok.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
    prompt_2 = tok.apply_chat_template(MESSAGES_MATH, tokenize=False, add_generation_prompt=True)

    sampling_params_list = [
        SamplingParams(
            temperature=0.6,
            max_tokens=512,
            extra_args={
                "thinking_budget": 150,
                "thinking_budget_grace_period": 30,
                "think_end_tokens": "\nFinalize response\n</think>",
            },
        ),
        SamplingParams(
            temperature=0.6,
            max_tokens=512,
            extra_args={
                "thinking_budget": 20,
                "thinking_budget_grace_period": 5,
                "think_end_tokens": "\n</think>",
            },
        ),
    ]

    print(f"\n{'='*70}")
    print(f"Test 2: Multiple prompts with different budgets")
    print(f"{'='*70}")

    outputs = llm.generate([prompt_1, prompt_2], sampling_params_list)

    labels = [
        "budget=150, grace=30, think_end_tokens='Reached thinking limit.\\n</think>'",
        "budget=20,  grace=5,  think_end_tokens='</think>'",
    ]
    all_messages = [MESSAGES, MESSAGES_MATH]

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        num_tokens = len(output.outputs[0].token_ids)
        print(f"\n[Request {i}] {labels[i]}")
        print(f"  Messages: {all_messages[i]}")
        print(f"  Output ({num_tokens} tokens):")
        print(f"  {generated_text}")
        print(f"  {'-'*60}")

    print(f"\n{'='*70}")
    print("Done.")


if __name__ == "__main__":
    main()
