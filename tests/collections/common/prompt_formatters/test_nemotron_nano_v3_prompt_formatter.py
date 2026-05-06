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
import os

import pytest

from nemo.collections.common.prompts.nemotron_nano_v3 import NemotronNanoV3PromptFormatter

# ──────────────────────────────────────────────────────────────────────
# Unit tests (BPE tokenizer, always run)
# ──────────────────────────────────────────────────────────────────────


def test_nemotron_nano_v3_training_basic(bpe_tokenizer_with_think):
    """User + assistant, no explicit system → empty system auto-inserted, <think></think> prepended."""
    formatter = NemotronNanoV3PromptFormatter(bpe_tokenizer_with_think)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids", "answer_ids", "mask"}
    # fmt: off
    # The test BPE tokenizer inserts an extra space at turn boundaries, but real tokenizers don't.
    assert bpe_tokenizer_with_think.ids_to_text(ans["input_ids"].tolist()) == '<|im_start|>system\n<|im_end|>\n <|im_start|>user\nTEST<|im_end|>\n <|im_start|>assistant\n<think></think>TEST<|im_end|>\n'
    assert bpe_tokenizer_with_think.ids_to_text(ans["context_ids"].tolist()) == '<|im_start|>system\n<|im_end|>\n <|im_start|>user\nTEST<|im_end|>\n'
    assert bpe_tokenizer_with_think.ids_to_text(ans["answer_ids"].tolist()) == '<|im_start|>assistant\n<think></think>TEST<|im_end|>\n'
    assert ans["mask"].shape[0] == ans["input_ids"].shape[0]
    # fmt: on


def test_nemotron_nano_v3_training_with_system(bpe_tokenizer_with_think):
    """Explicit system message preserved."""
    formatter = NemotronNanoV3PromptFormatter(bpe_tokenizer_with_think)
    ans = formatter.encode_dialog(
        [
            {"role": "system", "slots": {"message": "You are helpful."}},
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids", "answer_ids", "mask"}
    # fmt: off
    assert bpe_tokenizer_with_think.ids_to_text(ans["input_ids"].tolist()) == '<|im_start|>system\nYou are helpful.<|im_end|>\n <|im_start|>user\nTEST<|im_end|>\n <|im_start|>assistant\n<think></think>TEST<|im_end|>\n'
    assert bpe_tokenizer_with_think.ids_to_text(ans["context_ids"].tolist()) == '<|im_start|>system\nYou are helpful.<|im_end|>\n <|im_start|>user\nTEST<|im_end|>\n'
    assert bpe_tokenizer_with_think.ids_to_text(ans["answer_ids"].tolist()) == '<|im_start|>assistant\n<think></think>TEST<|im_end|>\n'
    assert ans["mask"].shape[0] == ans["input_ids"].shape[0]
    # fmt: on


def test_nemotron_nano_v3_empty_system_auto_inserted(bpe_tokenizer_with_think):
    """No system provided → empty system turn emitted (uses TEST to stay in BPE vocab)."""
    formatter = NemotronNanoV3PromptFormatter(bpe_tokenizer_with_think)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "TEST"}},
        ]
    )
    # fmt: off
    # Same as training_basic — the empty system turn is auto-inserted identically.
    assert bpe_tokenizer_with_think.ids_to_text(ans["input_ids"].tolist()) == '<|im_start|>system\n<|im_end|>\n <|im_start|>user\nTEST<|im_end|>\n <|im_start|>assistant\n<think></think>TEST<|im_end|>\n'
    # fmt: on


def test_nemotron_nano_v3_training_with_thinking(bpe_tokenizer_with_think):
    """Assistant has <think>...</think> content — kept as-is."""
    formatter = NemotronNanoV3PromptFormatter(bpe_tokenizer_with_think)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "<think>TEST</think>TEST"}},
        ]
    )
    # fmt: off
    assert bpe_tokenizer_with_think.ids_to_text(ans["input_ids"].tolist()) == '<|im_start|>system\n<|im_end|>\n <|im_start|>user\nTEST<|im_end|>\n <|im_start|>assistant\n<think>TEST</think>TEST<|im_end|>\n'
    assert bpe_tokenizer_with_think.ids_to_text(ans["context_ids"].tolist()) == '<|im_start|>system\n<|im_end|>\n <|im_start|>user\nTEST<|im_end|>\n'
    assert bpe_tokenizer_with_think.ids_to_text(ans["answer_ids"].tolist()) == '<|im_start|>assistant\n<think>TEST</think>TEST<|im_end|>\n'
    assert ans["mask"].shape[0] == ans["input_ids"].shape[0]
    # fmt: on


def test_nemotron_nano_v3_training_no_think_prepended(bpe_tokenizer_with_think):
    """Assistant without think tags gets <think></think> prepended (same output as training_basic)."""
    formatter = NemotronNanoV3PromptFormatter(bpe_tokenizer_with_think)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "TEST"}},
        ]
    )
    # fmt: off
    assert bpe_tokenizer_with_think.ids_to_text(ans["input_ids"].tolist()) == '<|im_start|>system\n<|im_end|>\n <|im_start|>user\nTEST<|im_end|>\n <|im_start|>assistant\n<think></think>TEST<|im_end|>\n'
    # fmt: on


def test_nemotron_nano_v3_training_multiturn_past_asst_no_think(bpe_tokenizer_with_think):
    """Multi-turn training, past assistant WITHOUT think tags: <think></think> prepended to match HF jinja."""
    formatter = NemotronNanoV3PromptFormatter(bpe_tokenizer_with_think)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "TEST"}},
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "TEST"}},
        ]
    )
    # fmt: off
    assert bpe_tokenizer_with_think.ids_to_text(ans["input_ids"].tolist()) == '<|im_start|>system\n<|im_end|>\n <|im_start|>user\nTEST<|im_end|>\n <|im_start|>assistant\n<think></think>TEST<|im_end|>\n <|im_start|>user\nTEST<|im_end|>\n <|im_start|>assistant\n<think></think>TEST<|im_end|>\n'
    assert ans["mask"].shape[0] == ans["input_ids"].shape[0]
    # fmt: on


def test_nemotron_nano_v3_history_thinking_truncation(bpe_tokenizer_with_think):
    """Multi-turn: earlier assistant thinking replaced with <think></think>."""
    formatter = NemotronNanoV3PromptFormatter(bpe_tokenizer_with_think)
    ans = formatter.encode_dialog(
        [
            {"role": "system", "slots": {"message": ""}},
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "<think>SYSTEM</think>TEST"}},
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "TEST"}},
        ]
    )
    # fmt: off
    assert bpe_tokenizer_with_think.ids_to_text(ans["input_ids"].tolist()) == '<|im_start|>system\n<|im_end|>\n <|im_start|>user\nTEST<|im_end|>\n <|im_start|>assistant\n<think></think>TEST<|im_end|>\n <|im_start|>user\nTEST<|im_end|>\n <|im_start|>assistant\n<think></think>TEST<|im_end|>\n'
    assert bpe_tokenizer_with_think.ids_to_text(ans["context_ids"].tolist()) == '<|im_start|>system\n<|im_end|>\n <|im_start|>user\nTEST<|im_end|>\n <|im_start|>assistant\n<think></think>TEST<|im_end|>\n <|im_start|>user\nTEST<|im_end|>\n'
    assert bpe_tokenizer_with_think.ids_to_text(ans["answer_ids"].tolist()) == '<|im_start|>assistant\n<think></think>TEST<|im_end|>\n'
    assert ans["mask"].shape[0] == ans["input_ids"].shape[0]
    # fmt: on


def test_nemotron_nano_v3_inference_thinking_enabled(bpe_tokenizer_with_think):
    """Inference with thinking enabled: generation prompt ends with <think>\\n."""
    formatter = NemotronNanoV3PromptFormatter(bpe_tokenizer_with_think)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
        ],
        enable_thinking=True,
    )
    # fmt: off
    assert set(ans) == {"input_ids", "context_ids"}
    assert bpe_tokenizer_with_think.ids_to_text(ans["input_ids"].tolist()) == '<|im_start|>system\n<|im_end|>\n <|im_start|>user\nTEST<|im_end|>\n <|im_start|>assistant\n<think>\n'
    assert ans["input_ids"].tolist() == ans["context_ids"].tolist()
    # fmt: on


def test_nemotron_nano_v3_inference_thinking_disabled(bpe_tokenizer_with_think):
    """Inference with thinking disabled: generation prompt ends with <think></think>."""
    formatter = NemotronNanoV3PromptFormatter(bpe_tokenizer_with_think)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
        ],
        enable_thinking=False,
    )
    # fmt: off
    assert set(ans) == {"input_ids", "context_ids"}
    assert bpe_tokenizer_with_think.ids_to_text(ans["input_ids"].tolist()) == '<|im_start|>system\n<|im_end|>\n <|im_start|>user\nTEST<|im_end|>\n <|im_start|>assistant\n<think></think>'
    assert ans["input_ids"].tolist() == ans["context_ids"].tolist()
    # fmt: on


def test_nemotron_nano_v3_inference_keys(bpe_tokenizer_with_think):
    """Inference with explicit system: output has only input_ids and context_ids."""
    formatter = NemotronNanoV3PromptFormatter(bpe_tokenizer_with_think)
    ans = formatter.encode_dialog(
        [
            {"role": "system", "slots": {"message": "SYSTEM"}},
            {"role": "user", "slots": {"message": "TEST"}},
        ]
    )
    # fmt: off
    assert set(ans) == {"input_ids", "context_ids"}
    assert bpe_tokenizer_with_think.ids_to_text(ans["input_ids"].tolist()) == '<|im_start|>system\nSYSTEM<|im_end|>\n <|im_start|>user\nTEST<|im_end|>\n <|im_start|>assistant\n<think>\n'
    assert ans["input_ids"].tolist() == ans["context_ids"].tolist()
    # fmt: on


def test_nemotron_nano_v3_inference_multiturn_thinking_disabled(bpe_tokenizer_with_think):
    """Multi-turn inference, thinking disabled: past <think>...</think> truncated, generation prompt ends with <think></think>."""
    formatter = NemotronNanoV3PromptFormatter(bpe_tokenizer_with_think)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "<think>PAST</think>TEST"}},
            {"role": "user", "slots": {"message": "TEST"}},
        ],
        enable_thinking=False,
    )
    # fmt: off
    assert set(ans) == {"input_ids", "context_ids"}
    assert bpe_tokenizer_with_think.ids_to_text(ans["input_ids"].tolist()) == '<|im_start|>system\n<|im_end|>\n <|im_start|>user\nTEST<|im_end|>\n <|im_start|>assistant\n<think></think>TEST<|im_end|>\n <|im_start|>user\nTEST<|im_end|>\n <|im_start|>assistant\n<think></think>'
    assert ans["input_ids"].tolist() == ans["context_ids"].tolist()
    # fmt: on


def test_nemotron_nano_v3_inference_multiturn_thinking_enabled(bpe_tokenizer_with_think):
    """Multi-turn inference, thinking enabled: past <think>...</think> truncated, generation prompt ends with <think>\\n."""
    formatter = NemotronNanoV3PromptFormatter(bpe_tokenizer_with_think)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "<think>PAST</think>TEST"}},
            {"role": "user", "slots": {"message": "TEST"}},
        ],
        enable_thinking=True,
    )
    # fmt: off
    assert set(ans) == {"input_ids", "context_ids"}
    assert bpe_tokenizer_with_think.ids_to_text(ans["input_ids"].tolist()) == '<|im_start|>system\n<|im_end|>\n <|im_start|>user\nTEST<|im_end|>\n <|im_start|>assistant\n<think></think>TEST<|im_end|>\n <|im_start|>user\nTEST<|im_end|>\n <|im_start|>assistant\n<think>\n'
    assert ans["input_ids"].tolist() == ans["context_ids"].tolist()
    # fmt: on


# ──────────────────────────────────────────────────────────────────────
# HuggingFace comparison tests (guarded)
# ──────────────────────────────────────────────────────────────────────

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
CI_CACHED_MODEL_PATH = "/home/TestData_/HF_HOME/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


@pytest.fixture(scope="module")
def nemo_auto_tokenizer():
    pytest.importorskip("transformers")
    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

    if os.path.exists(CI_CACHED_MODEL_PATH):
        return AutoTokenizer(CI_CACHED_MODEL_PATH, trust_remote_code=True)
    return AutoTokenizer(MODEL_ID, trust_remote_code=True)


def _hf_apply(nemo_tok, messages, *, tokenize, add_generation_prompt=False, enable_thinking=True):
    """Call apply_chat_template via the underlying HF tokenizer."""
    kwargs = dict(
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    if tokenize:
        kwargs["return_dict"] = False
    else:
        kwargs["add_special_tokens"] = False
    return nemo_tok.tokenizer.apply_chat_template(messages, **kwargs)


class TestNemotronNanoV3HFComparison:
    """Compare NeMo formatter output against HuggingFace apply_chat_template."""

    @pytest.mark.xfail(reason="Nemotron Nano V3 checkpoint is temporarily corrupted on the CI runner.")
    def test_hf_simple_inference(self, nemo_auto_tokenizer):
        """System + user, add_generation_prompt=True, enable_thinking=True."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        turns = [{"role": m["role"], "slots": {"message": m["content"]}} for m in messages]
        formatter = NemotronNanoV3PromptFormatter(nemo_auto_tokenizer)
        result = formatter.encode_dialog(turns, enable_thinking=True)

        nemo_str = nemo_auto_tokenizer.ids_to_text(result["input_ids"].tolist(), remove_special_tokens=False)
        hf_str = _hf_apply(
            nemo_auto_tokenizer, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        assert nemo_str == hf_str, f"String mismatch:\nNeMo: {nemo_str!r}\nHF:   {hf_str!r}"

        nemo_ids = result["input_ids"].tolist()
        hf_ids = _hf_apply(
            nemo_auto_tokenizer, messages, tokenize=True, add_generation_prompt=True, enable_thinking=True
        )
        assert nemo_ids == hf_ids, f"Token mismatch:\nNeMo: {nemo_ids}\nHF:   {hf_ids}"

    @pytest.mark.xfail(reason="Nemotron Nano V3 checkpoint is temporarily corrupted on the CI runner.")
    def test_hf_simple_training(self, nemo_auto_tokenizer):
        """System + user + assistant, no thinking tags."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        turns = [{"role": m["role"], "slots": {"message": m["content"]}} for m in messages]
        formatter = NemotronNanoV3PromptFormatter(nemo_auto_tokenizer)
        result = formatter.encode_dialog(turns)

        nemo_str = nemo_auto_tokenizer.ids_to_text(result["input_ids"].tolist(), remove_special_tokens=False)
        hf_str = _hf_apply(nemo_auto_tokenizer, messages, tokenize=False)
        # HF template for training doesn't add generation prompt and includes full assistant turn.
        # The NeMo formatter prepends <think></think> to assistant content without think tags.
        # For training comparison, we need to match the format that the model would produce.
        # The HF template also prepends <think></think> for assistant content without think tags.
        assert nemo_str == hf_str, f"String mismatch:\nNeMo: {nemo_str!r}\nHF:   {hf_str!r}"

        nemo_ids = result["input_ids"].tolist()
        hf_ids = _hf_apply(nemo_auto_tokenizer, messages, tokenize=True)
        assert nemo_ids == hf_ids, f"Token mismatch:\nNeMo: {nemo_ids}\nHF:   {hf_ids}"

    @pytest.mark.xfail(reason="Nemotron Nano V3 checkpoint is temporarily corrupted on the CI runner.")
    def test_hf_training_with_system(self, nemo_auto_tokenizer):
        """Explicit system message in training mode."""
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Summarize AI."},
            {"role": "assistant", "content": "AI is machine intelligence."},
        ]
        turns = [{"role": m["role"], "slots": {"message": m["content"]}} for m in messages]
        formatter = NemotronNanoV3PromptFormatter(nemo_auto_tokenizer)
        result = formatter.encode_dialog(turns)

        nemo_str = nemo_auto_tokenizer.ids_to_text(result["input_ids"].tolist(), remove_special_tokens=False)
        hf_str = _hf_apply(nemo_auto_tokenizer, messages, tokenize=False)
        assert nemo_str == hf_str, f"String mismatch:\nNeMo: {nemo_str!r}\nHF:   {hf_str!r}"

        nemo_ids = result["input_ids"].tolist()
        hf_ids = _hf_apply(nemo_auto_tokenizer, messages, tokenize=True)
        assert nemo_ids == hf_ids, f"Token mismatch:\nNeMo: {nemo_ids}\nHF:   {hf_ids}"

    @pytest.mark.xfail(reason="Nemotron Nano V3 checkpoint is temporarily corrupted on the CI runner.")
    def test_hf_multiturn_with_thinking_in_history(self, nemo_auto_tokenizer):
        """Multi-turn with thinking in history — truncation compared against HF."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "<think>Let me calculate 2+2.</think>4"},
            {"role": "user", "content": "And 3+3?"},
            {"role": "assistant", "content": "6"},
        ]
        turns = [{"role": m["role"], "slots": {"message": m["content"]}} for m in messages]
        formatter = NemotronNanoV3PromptFormatter(nemo_auto_tokenizer)
        result = formatter.encode_dialog(turns)

        nemo_str = nemo_auto_tokenizer.ids_to_text(result["input_ids"].tolist(), remove_special_tokens=False)
        hf_str = _hf_apply(nemo_auto_tokenizer, messages, tokenize=False)
        assert nemo_str == hf_str, f"String mismatch:\nNeMo: {nemo_str!r}\nHF:   {hf_str!r}"

        nemo_ids = result["input_ids"].tolist()
        hf_ids = _hf_apply(nemo_auto_tokenizer, messages, tokenize=True)
        assert nemo_ids == hf_ids, f"Token mismatch:\nNeMo: {nemo_ids}\nHF:   {hf_ids}"

    @pytest.mark.xfail(reason="Nemotron Nano V3 checkpoint is temporarily corrupted on the CI runner.")
    def test_hf_inference_thinking_disabled(self, nemo_auto_tokenizer):
        """Inference with enable_thinking=False."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        turns = [{"role": m["role"], "slots": {"message": m["content"]}} for m in messages]
        formatter = NemotronNanoV3PromptFormatter(nemo_auto_tokenizer)
        result = formatter.encode_dialog(turns, enable_thinking=False)

        nemo_str = nemo_auto_tokenizer.ids_to_text(result["input_ids"].tolist(), remove_special_tokens=False)
        hf_str = _hf_apply(
            nemo_auto_tokenizer, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        assert nemo_str == hf_str, f"String mismatch:\nNeMo: {nemo_str!r}\nHF:   {hf_str!r}"

        nemo_ids = result["input_ids"].tolist()
        hf_ids = _hf_apply(
            nemo_auto_tokenizer, messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        assert nemo_ids == hf_ids, f"Token mismatch:\nNeMo: {nemo_ids}\nHF:   {hf_ids}"
