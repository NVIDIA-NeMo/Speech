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

"""Tests for ReasoningBudgetLogitsProcessor.

Uses the tokenizer from nvidia/Nemotron-3-Nano-30B-A3B to obtain real token IDs
and simulates BatchUpdate / apply() cycles on CUDA tensors with realistic
tokenized prompts to verify:
    1. Tokens are unconstrained before the budget is reached.
    2. Grace-period boosting kicks in before the budget.
    3. End tokens are forced at the hard budget limit.
    4. Generation is unconstrained after </think> is emitted.
    5. Requests without thinking_budget are not tracked.
    6. Step-by-step simulation of a full thinking sequence.
    7. End-to-end generation via vllm.LLM with real prompts.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add the local NeMo directory to Python path to use development version
nemo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(nemo_root))

import pytest
import torch
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.v1.sample.logits_processor import BatchUpdate

from nemo.agents.voice_agent.vllm.v1.sample.logits_processor.reasoning_budget_logits_processor import (
    ReasoningBudgetLogitsProcessor,
)

MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DEVICE = torch.device("cuda:0")

# Real prompts used for testing.
MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant. /think"},
    {"role": "user", "content": "Write a haiku about a cat"},
]

# A realistic thinking block that the model might produce.
THINKING_TEXT = (
    "<think>\nOkay, the user wants a haiku about a cat. "
    "A haiku has a 5-7-5 syllable structure. "
    "Let me think of imagery: whiskers, sunbeam, soft paws, purring.\n</think>"
)

# The answer portion after thinking.
ANSWER_TEXT = "\nSoft paws on the sill\nWhiskers catch the morning light\nPurring fills the room"


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _make_vllm_config():
    """Build a minimal mock VllmConfig that carries tokenizer info."""
    model_cfg = MagicMock()
    model_cfg.tokenizer = MODEL_NAME
    model_cfg.trust_remote_code = True
    model_cfg.tokenizer_revision = None

    vllm_config = MagicMock()
    vllm_config.model_config = model_cfg
    return vllm_config


@pytest.fixture(scope="module")
def tokenizer():
    """Module-scoped tokenizer."""
    return AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


@pytest.fixture(scope="module")
def processor():
    """Module-scoped processor on cuda:0 (tokenizer loaded once)."""
    return ReasoningBudgetLogitsProcessor(_make_vllm_config(), device=DEVICE, is_pin_memory=False)


def _sampling_params(thinking_budget=None, grace_period=None, think_start_tokens=None, think_end_tokens=None):
    """Create a SamplingParams with optional thinking budget params."""
    extra = {}
    if thinking_budget is not None:
        extra["thinking_budget"] = thinking_budget
    if grace_period is not None:
        extra["thinking_budget_grace_period"] = grace_period
    if think_start_tokens is not None:
        extra["think_start_tokens"] = think_start_tokens
    if think_end_tokens is not None:
        extra["think_end_tokens"] = think_end_tokens
    return SamplingParams.from_optional(extra_args=extra or None)


def _batch_update_add(index, params, output_tok_ids, batch_size=1):
    """Create a BatchUpdate that adds a single request."""
    return BatchUpdate(
        batch_size=batch_size,
        removed=[],
        added=[(index, params, None, output_tok_ids)],
        moved=[],
    )


def _uniform_logits(batch_size=1, vocab_size=131072):
    """Return a (batch_size, vocab_size) tensor of zeros on DEVICE."""
    return torch.zeros(batch_size, vocab_size, dtype=torch.float32, device=DEVICE)


# ------------------------------------------------------------------
# Tests — tokenizer sanity
# ------------------------------------------------------------------


class TestTokenizerSetup:
    """Verify the processor resolved token IDs correctly from the real tokenizer."""

    def test_think_delimiters(self, processor, tokenizer):
        assert processor.think_start_ids == tokenizer.encode("<think>", add_special_tokens=False)
        assert processor.think_end_detect_ids == tokenizer.encode("</think>", add_special_tokens=False)
        assert processor.think_end_force_ids == tokenizer.encode("\n</think>", add_special_tokens=False)

    def test_newline_is_single_token(self, processor, tokenizer):
        assert processor.newline_ids == tokenizer.encode("\n", add_special_tokens=False)


# ------------------------------------------------------------------
# Tests — validate_params
# ------------------------------------------------------------------


class TestValidateParams:
    def test_no_extra_args(self):
        params = SamplingParams.from_optional()
        ReasoningBudgetLogitsProcessor.validate_params(params)

    def test_valid_budget(self):
        params = _sampling_params(thinking_budget=100)
        ReasoningBudgetLogitsProcessor.validate_params(params)

    def test_valid_with_all_options(self):
        params = _sampling_params(
            thinking_budget=100,
            grace_period=20,
            think_start_tokens="<think>",
            think_end_tokens="</think>",
        )
        ReasoningBudgetLogitsProcessor.validate_params(params)

    def test_invalid_budget_negative(self):
        params = _sampling_params(thinking_budget=-1)
        with pytest.raises(ValueError, match="thinking_budget"):
            ReasoningBudgetLogitsProcessor.validate_params(params)

    def test_invalid_budget_string(self):
        params = SamplingParams.from_optional(extra_args={"thinking_budget": "abc"})
        with pytest.raises(ValueError, match="thinking_budget"):
            ReasoningBudgetLogitsProcessor.validate_params(params)


# ------------------------------------------------------------------
# Tests — no-budget requests are ignored
# ------------------------------------------------------------------


class TestRequestWithoutBudget:
    def test_no_tracking(self, processor, tokenizer):
        processor.req_states.clear()
        output_tok_ids = tokenizer.encode("<think>hello world</think>", add_special_tokens=False)
        params = SamplingParams.from_optional()
        update = _batch_update_add(0, params, output_tok_ids)
        processor.update_state(update)
        assert len(processor.req_states) == 0

    def test_logits_unchanged(self, processor):
        processor.req_states.clear()
        logits = _uniform_logits()
        result = processor.apply(logits)
        assert torch.equal(result, logits)


# ------------------------------------------------------------------
# Tests — unconstrained before budget
# ------------------------------------------------------------------


class TestUnconstrainedBeforeBudget:
    def test_logits_not_modified_with_real_tokens(self, processor, tokenizer):
        processor.req_states.clear()
        budget = 50
        short_thinking = "<think>\nOkay, the user wants"
        output_tok_ids = tokenizer.encode(short_thinking, add_special_tokens=False)

        params = _sampling_params(thinking_budget=budget)
        update = _batch_update_add(0, params, output_tok_ids)
        processor.update_state(update)

        state = processor.req_states[0]
        assert state.inside_thinking is True
        grace_start = budget - state.grace_period
        assert state.thinking_token_count < grace_start

        logits = _uniform_logits()
        logits_before = logits.clone()
        processor.apply(logits)
        assert torch.equal(logits, logits_before)


# ------------------------------------------------------------------
# Tests — grace period
# ------------------------------------------------------------------


class TestGracePeriod:
    def test_boost_applied(self, processor):
        processor.req_states.clear()
        think_start_id = processor.think_start_ids[0]

        budget = 10
        grace_period = 1  # explicit: grace starts at token 9
        grace_start = budget - grace_period

        output_tok_ids = [think_start_id] + [100] * grace_start
        params = _sampling_params(thinking_budget=budget, grace_period=grace_period)

        update = _batch_update_add(0, params, output_tok_ids)
        processor.update_state(update)

        state = processor.req_states[0]
        assert state.thinking_token_count >= grace_start
        assert state.thinking_token_count < budget

        logits = _uniform_logits()
        processor.apply(logits)

        # Grace boost applies to force_end_ids tokens.
        for tok_id in state.force_end_ids:
            assert logits[0, tok_id].item() > 0, f"Expected boost for token {tok_id}"
        assert logits[0, 500].item() == pytest.approx(0.0)


# ------------------------------------------------------------------
# Tests — hard cutoff
# ------------------------------------------------------------------


class TestHardCutoff:
    def test_force_end_token(self, processor):
        processor.req_states.clear()
        think_start_id = processor.think_start_ids[0]
        budget = 5

        output_tok_ids = [think_start_id] + [100] * budget
        params = _sampling_params(thinking_budget=budget)

        update = _batch_update_add(0, params, output_tok_ids)
        processor.update_state(update)

        state = processor.req_states[0]
        first_forced = state.force_end_ids[0]

        logits = _uniform_logits()
        vocab_size = logits.shape[1]
        processor.apply(logits)

        assert logits[0, first_forced].item() != float("-inf")
        mask = torch.ones(vocab_size, dtype=torch.bool, device=DEVICE)
        mask[first_forced] = False
        assert (logits[0, mask] == float("-inf")).all()

    def test_force_custom_think_end_tokens(self, processor, tokenizer):
        """When think_end_tokens is provided, those tokens are forced in sequence."""
        processor.req_states.clear()
        think_start_id = processor.think_start_ids[0]
        budget = 3

        # ".\n</think>" will be tokenized into multiple token IDs.
        custom_end_text = ".\n</think>"
        custom_end_ids = tokenizer.encode(custom_end_text, add_special_tokens=False)
        assert len(custom_end_ids) >= 2, f"Expected multi-token end sequence, got {custom_end_ids}"

        output_tok_ids = [think_start_id] + [100] * budget
        params = _sampling_params(thinking_budget=budget, think_end_tokens=custom_end_text)

        update = _batch_update_add(0, params, output_tok_ids)
        processor.update_state(update)

        # First apply: should force the first token of the end sequence.
        logits = _uniform_logits()
        processor.apply(logits)
        first_forced = custom_end_ids[0]
        mask = torch.ones(logits.shape[1], dtype=torch.bool, device=DEVICE)
        mask[first_forced] = False
        assert (logits[0, mask] == float("-inf")).all()
        assert logits[0, first_forced].item() != float("-inf")

        # Simulate that first token was generated.
        output_tok_ids.append(first_forced)
        processor.update_state(None)

        # Second apply: should force the second token of the end sequence.
        logits2 = _uniform_logits()
        processor.apply(logits2)
        second_forced = custom_end_ids[1]
        mask2 = torch.ones(logits2.shape[1], dtype=torch.bool, device=DEVICE)
        mask2[second_forced] = False
        assert (logits2[0, mask2] == float("-inf")).all()
        assert logits2[0, second_forced].item() != float("-inf")

    def test_unconstrained_after_end(self, processor):
        processor.req_states.clear()
        think_start_id = processor.think_start_ids[0]
        budget = 5

        # Append the full end sequence (e.g. [\n, </think>]) after the thinking tokens.
        output_tok_ids = [think_start_id] + [100] * budget + processor.think_end_detect_ids
        params = _sampling_params(thinking_budget=budget)

        update = _batch_update_add(0, params, output_tok_ids)
        processor.update_state(update)

        assert 0 not in processor.req_states

        logits = _uniform_logits()
        logits_before = logits.clone()
        processor.apply(logits)
        assert torch.equal(logits, logits_before)


# ------------------------------------------------------------------
# Tests — natural </think> before budget
# ------------------------------------------------------------------


class TestNaturalEndThinking:
    def test_natural_stop_with_real_text(self, processor, tokenizer):
        processor.req_states.clear()
        budget = 1000
        output_tok_ids = tokenizer.encode(THINKING_TEXT, add_special_tokens=False)
        params = _sampling_params(thinking_budget=budget)

        update = _batch_update_add(0, params, output_tok_ids)
        processor.update_state(update)
        assert 0 not in processor.req_states


# ------------------------------------------------------------------
# Tests — step-by-step simulation
# ------------------------------------------------------------------


class TestStepByStepSimulation:
    """Simulate token-by-token generation with a real tokenized thinking block."""

    def test_full_lifecycle(self, processor, tokenizer):
        processor.req_states.clear()

        full_thinking_ids = tokenizer.encode(THINKING_TEXT, add_special_tokens=False)
        think_start_id = processor.think_start_ids[0]

        budget = 15
        grace_period = 2
        grace_start = budget - grace_period

        assert full_thinking_ids[0] == think_start_id

        print(f"\n{'='*70}")
        print(f"Step-by-step simulation — budget={budget}, grace_period={grace_period}, grace_start={grace_start}")
        print(f"Full thinking text ({len(full_thinking_ids)} tokens):")
        print(f"  {THINKING_TEXT!r}")
        print(f"{'='*70}")

        output_tok_ids: list[int] = []
        params = _sampling_params(thinking_budget=budget, grace_period=grace_period)
        update = _batch_update_add(0, params, output_tok_ids)
        processor.update_state(update)

        saw_unconstrained = False
        saw_grace = False
        saw_forced = False

        for step, tok in enumerate(full_thinking_ids):
            output_tok_ids.append(tok)
            processor.update_state(None)

            decoded = tokenizer.decode([tok])
            if 0 not in processor.req_states:
                print(f"  step {step:3d}: tok={tok:6d}  {decoded!r:20s}  → request removed (thinking ended)")
                break

            state = processor.req_states[0]
            first_forced = state.force_end_ids[0]
            logits = _uniform_logits()
            processor.apply(logits)

            if state.forcing_end_idx >= 0 or state.thinking_token_count >= budget:
                phase = "FORCED"
                saw_forced = True
                assert logits[0, first_forced].item() != float("-inf")
                check_mask = torch.ones(logits.shape[1], dtype=torch.bool, device=DEVICE)
                check_mask[first_forced] = False
                assert (logits[0, check_mask] == float("-inf")).all()
                print(
                    f"  step {step:3d}: tok={tok:6d}  {decoded!r:20s}  count={state.thinking_token_count:3d}  [{phase}] → only end tokens allowed"
                )
                break
            elif state.thinking_token_count >= grace_start:
                phase = "GRACE"
                saw_grace = True
                for force_tok in state.force_end_ids:
                    assert logits[0, force_tok].item() > 0
            else:
                phase = "FREE " if state.inside_thinking else "-----"
                if state.inside_thinking and state.thinking_token_count > 0:
                    saw_unconstrained = True

            print(f"  step {step:3d}: tok={tok:6d}  {decoded!r:20s}  count={state.thinking_token_count:3d}  [{phase}]")

        print(f"{'='*70}")

        assert saw_unconstrained, "Should have seen unconstrained phase"
        assert saw_grace, "Should have seen grace period"
        assert saw_forced, "Should have seen forced cutoff"


# ------------------------------------------------------------------
# Tests — multi-request batch
# ------------------------------------------------------------------


class TestMultiRequestBatch:
    def test_selective_tracking(self, processor, tokenizer):
        processor.req_states.clear()

        output_0 = tokenizer.encode("<think>\nLet me think", add_special_tokens=False)
        params_0 = _sampling_params(thinking_budget=10)

        output_1 = tokenizer.encode("<think>\nSome thinking", add_special_tokens=False)
        params_1 = SamplingParams.from_optional()

        update = BatchUpdate(
            batch_size=2,
            removed=[],
            added=[
                (0, params_0, None, output_0),
                (1, params_1, None, output_1),
            ],
            moved=[],
        )
        processor.update_state(update)

        assert 0 in processor.req_states
        assert 1 not in processor.req_states

    def test_independent_budgets(self, processor):
        processor.req_states.clear()
        think_start_id = processor.think_start_ids[0]

        output_0 = [think_start_id] + [100] * 5
        params_0 = _sampling_params(thinking_budget=5)

        output_1 = [think_start_id, 200]
        params_1 = _sampling_params(thinking_budget=50)

        update = BatchUpdate(
            batch_size=2,
            removed=[],
            added=[
                (0, params_0, None, output_0),
                (1, params_1, None, output_1),
            ],
            moved=[],
        )
        processor.update_state(update)

        logits = _uniform_logits(batch_size=2)
        processor.apply(logits)

        # Request 0: forced — only the first force token should be non-(-inf).
        state_0 = processor.req_states[0]
        first_forced = state_0.force_end_ids[0]
        mask = torch.ones(logits.shape[1], dtype=torch.bool, device=DEVICE)
        mask[first_forced] = False
        assert (logits[0, mask] == float("-inf")).all()

        # Request 1: untouched.
        assert torch.equal(logits[1], torch.zeros(logits.shape[1], device=DEVICE))


# ------------------------------------------------------------------
# Tests — incremental scanning
# ------------------------------------------------------------------


class TestIncrementalScanning:
    def test_incremental_count(self, processor):
        processor.req_states.clear()
        think_start_id = processor.think_start_ids[0]
        budget = 50
        output_tok_ids = [think_start_id]
        params = _sampling_params(thinking_budget=budget)

        update = _batch_update_add(0, params, output_tok_ids)
        processor.update_state(update)

        for i in range(5):
            output_tok_ids.append(100 + i)
            processor.update_state(None)
            assert processor.req_states[0].thinking_token_count == i + 1


# ------------------------------------------------------------------
# Tests — request removal
# ------------------------------------------------------------------


class TestRequestRemoval:
    def test_removed_request(self, processor):
        processor.req_states.clear()
        think_start_id = processor.think_start_ids[0]
        output_tok_ids = [think_start_id, 100]
        params = _sampling_params(thinking_budget=10)

        add_update = _batch_update_add(0, params, output_tok_ids)
        processor.update_state(add_update)
        assert 0 in processor.req_states

        remove_update = BatchUpdate(batch_size=0, removed=[0], added=[], moved=[])
        processor.update_state(remove_update)
        assert 0 not in processor.req_states


# ------------------------------------------------------------------
# Tests — real prompt tokenization roundtrip
# ------------------------------------------------------------------


class TestRealPromptTokenization:
    def test_thinking_plus_answer(self, processor, tokenizer):
        processor.req_states.clear()

        full_output = THINKING_TEXT + ANSWER_TEXT
        all_ids = tokenizer.encode(full_output, add_special_tokens=False)
        thinking_ids = tokenizer.encode(THINKING_TEXT, add_special_tokens=False)

        budget = len(thinking_ids) + 100
        output_tok_ids: list[int] = []
        params = _sampling_params(thinking_budget=budget)

        update = _batch_update_add(0, params, output_tok_ids)
        processor.update_state(update)

        print(f"\n{'='*70}")
        print(f"Real prompt: thinking + answer (budget={budget})")
        print(f"Messages: {MESSAGES}")
        print(f"Thinking ({len(thinking_ids)} tokens): {THINKING_TEXT!r}")
        print(f"Answer: {ANSWER_TEXT!r}")
        print(f"{'='*70}")

        for step, tok in enumerate(all_ids):
            output_tok_ids.append(tok)
            processor.update_state(None)

            decoded = tokenizer.decode([tok])
            if 0 not in processor.req_states:
                print(f"  step {step:3d}: tok={tok:6d}  {decoded!r:20s}  → request removed (thinking done)")
                break
            else:
                state = processor.req_states[0]
                phase = "THINK" if state.inside_thinking else "-----"
                print(
                    f"  step {step:3d}: tok={tok:6d}  {decoded!r:20s}  count={state.thinking_token_count:3d}  [{phase}]"
                )

        assert 0 not in processor.req_states

        remaining = len(all_ids) - len(output_tok_ids)
        if remaining > 0:
            for tok in all_ids[len(output_tok_ids) :]:
                output_tok_ids.append(tok)
            print(f"  ... {remaining} more answer tokens fed — all unconstrained")

        logits = _uniform_logits()
        logits_before = logits.clone()
        processor.apply(logits)
        assert torch.equal(logits, logits_before)
        print(f"  Logits are unconstrained after </think>")
        print(f"{'='*70}")

    def test_chat_template_tokens(self, processor, tokenizer):
        prompt = tokenizer.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

        print(f"\n{'='*70}")
        print(f"Chat template for: {MESSAGES}")
        print(f"Rendered prompt ({len(prompt_ids)} tokens):")
        print(f"  {prompt}")
        print(f"{'='*70}")

        think_end_id = processor.think_end_detect_ids[0]
        assert len(prompt_ids) > 0
        assert think_end_id not in prompt_ids


# ------------------------------------------------------------------
# Integration test — offline vllm.LLM generation
# ------------------------------------------------------------------


class TestVLLMOfflineGeneration:
    """Generate with vllm.LLM and the logits processor on real prompts."""

    @pytest.fixture(scope="class")
    def llm(self):
        from vllm import LLM

        return LLM(
            model=MODEL_NAME,
            logits_processors=[ReasoningBudgetLogitsProcessor],
            trust_remote_code=True,
            gpu_memory_utilization=0.8,
            max_model_len=4096,
        )

    @pytest.fixture(scope="class")
    def tok(self):
        return AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    def test_generate_with_thinking_budget(self, llm, tok):
        """Send MESSAGES with thinking_budget=64, max_tokens=256, print full response."""
        prompt = tok.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)

        sampling_params_list = [
            SamplingParams(
                temperature=0.6,
                max_tokens=256,
                extra_args={
                    "thinking_budget": 64,
                    "thinking_budget_grace_period": 10,
                },
            ),
        ]

        outputs = llm.generate([prompt], sampling_params_list)

        print(f"\n{'='*70}")
        print(f"vllm.LLM generation — thinking_budget=64, max_tokens=256")
        print(f"Messages: {MESSAGES}")
        print(f"{'='*70}")
        for output in outputs:
            generated_text = output.outputs[0].text
            num_tokens = len(output.outputs[0].token_ids)
            print(f"Prompt:    {output.prompt!r}")
            print(f"Output ({num_tokens} tokens):")
            print(f"  {generated_text}")
            print(f"-" * 60)

            assert generated_text is not None and len(generated_text) > 0

    def test_generate_multiple_budgets(self, llm, tok):
        """Compare outputs with different thinking budgets side by side."""
        prompt = tok.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
        # Also test a different prompt.
        messages_2 = [
            {"role": "system", "content": "You are a helpful assistant. /think"},
            {"role": "user", "content": "What is 25 * 37?"},
        ]
        prompt_2 = tok.apply_chat_template(messages_2, tokenize=False, add_generation_prompt=True)

        sampling_params_list = [
            SamplingParams(
                temperature=0.6,
                max_tokens=512,
                extra_args={
                    "thinking_budget": 150,
                    "thinking_budget_grace_period": 30,
                    "think_end_tokens": "Reached thinking limit.\n</think>",
                },
            ),
            SamplingParams(
                temperature=0.6,
                max_tokens=512,
                extra_args={
                    "thinking_budget": 20,
                    "thinking_budget_grace_period": 5,
                    "think_end_tokens": "</think>",
                },
            ),
        ]

        outputs = llm.generate([prompt, prompt_2], sampling_params_list)

        print(f"\n{'='*70}")
        print(f"vllm.LLM generation — multiple budgets")
        print(f"{'='*70}")
        labels = [
            f"budget=150, grace=30, end='Reached thinking limit.\\n</think>'",
            f"budget=20,  grace=5,  end='</think>'",
        ]
        prompts_used = [MESSAGES, messages_2]
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            num_tokens = len(output.outputs[0].token_ids)
            print(f"\n[Request {i}] {labels[i]}")
            print(f"  Messages: {prompts_used[i]}")
            print(f"  Output ({num_tokens} tokens):")
            print(f"  {generated_text}")
            print(f"  {'-'*60}")

            assert generated_text is not None and len(generated_text) > 0
