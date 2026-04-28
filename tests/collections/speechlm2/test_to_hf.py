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
"""Unit tests for ``examples/speechlm2/to_hf.py::prepare_for_vllm``.

The script lives under ``examples/`` (not an importable package), so we load
it via ``importlib`` and patch ``AutoTokenizer`` / ``_detect_vllm_architecture``
to avoid any network or real-model dependencies.
"""
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

_TO_HF_PATH = Path(__file__).parents[3] / "examples" / "speechlm2" / "to_hf.py"
_spec = importlib.util.spec_from_file_location("to_hf_for_test", _TO_HF_PATH)
to_hf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(to_hf)

AUDIO_TOKEN = "<|audio|>"
CHAT_TEMPLATE_INLINE = "{% for msg in messages %}{{msg.content}}{% endfor %}"
CHAT_TEMPLATE_LARGE = "{% for msg in messages %}" + "X" * 4096 + "{{msg.content}}{% endfor %}"


class _FakeTokenizer:
    """Minimal stand-in for an HF ``AutoTokenizer`` instance.

    Mimics only the surface used by ``prepare_for_vllm``:
      * ``get_vocab`` / ``add_special_tokens``
      * ``save_pretrained`` (writes tokenizer_config.json, optionally
        splitting a large chat_template into a separate .jinja file)
      * ``eos_token_id``
    """

    def __init__(
        self,
        *,
        vocab_tokens=(),
        chat_template=CHAT_TEMPLATE_INLINE,
        split_chat_template=False,
        tokenizer_class="Qwen2Tokenizer",
        eos_token_id=42,
    ):
        self._vocab = {tok: i for i, tok in enumerate(vocab_tokens)}
        self._chat_template = chat_template
        self._split_chat_template = split_chat_template
        self._tokenizer_class = tokenizer_class
        self.eos_token_id = eos_token_id
        self.add_special_tokens_calls = []

    def get_vocab(self):
        return dict(self._vocab)

    def add_special_tokens(self, special_tokens_dict):
        added = special_tokens_dict.get("additional_special_tokens", [])
        for tok in added:
            if tok not in self._vocab:
                self._vocab[tok] = len(self._vocab)
        self.add_special_tokens_calls.append(added)
        return len(added)

    def save_pretrained(self, output_dir):
        output_dir = Path(output_dir)
        # Transformers writes extra_special_tokens as a LIST (not dict); mimic
        # that here so the dict-form coercion in to_hf.py is exercised.
        tok_cfg = {
            "tokenizer_class": self._tokenizer_class,
            "extra_special_tokens": [AUDIO_TOKEN],
        }
        if self._split_chat_template:
            (output_dir / "chat_template.jinja").write_text(self._chat_template)
        else:
            tok_cfg["chat_template"] = self._chat_template
        (output_dir / "tokenizer_config.json").write_text(json.dumps(tok_cfg))
        (output_dir / "tokenizer.json").write_text('{"fake": true}')


def _seed_output_dir(tmp_path, llm_arch="Qwen2ForCausalLM"):
    """Pre-populate ``output_dir`` with what ``save_hf_checkpoint`` would write."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": [llm_arch],
                "hidden_size": 2048,
                "num_hidden_layers": 24,
            }
        )
    )
    return tmp_path


# ──────────────────────────────────────────────────────────────────────
# Error paths (no mocking required — checks run before any HF calls)
# ──────────────────────────────────────────────────────────────────────


def test_prepare_for_vllm_missing_pretrained_llm(tmp_path):
    with pytest.raises(ValueError, match="pretrained_llm"):
        to_hf.prepare_for_vllm(str(tmp_path), {"audio_locator_tag": AUDIO_TOKEN})


def test_prepare_for_vllm_missing_audio_locator_tag(tmp_path):
    with pytest.raises(ValueError, match="audio_locator_tag"):
        to_hf.prepare_for_vllm(str(tmp_path), {"pretrained_llm": "fake-model"})


# ──────────────────────────────────────────────────────────────────────
# Happy paths (mock AutoTokenizer + _detect_vllm_architecture)
# ──────────────────────────────────────────────────────────────────────


def _run_prepare(tmp_path, fake_tok, arch="NeMoSpeechLMForConditionalGeneration", llm_arch="Qwen2ForCausalLM"):
    output_dir = _seed_output_dir(tmp_path, llm_arch=llm_arch)
    with (
        patch.object(to_hf, "_detect_vllm_architecture", return_value=arch),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tok),
    ):
        to_hf.prepare_for_vllm(
            str(output_dir),
            {"pretrained_llm": "fake-model", "audio_locator_tag": AUDIO_TOKEN},
        )
    return output_dir


def test_prepare_for_vllm_patches_config_json(tmp_path):
    """config.json gets model_type, architectures, and audio_locator_tag."""
    output_dir = _run_prepare(
        tmp_path,
        _FakeTokenizer(),
        arch="NeMoSpeechLMHybridForConditionalGeneration",
    )
    cfg = json.loads((output_dir / "config.json").read_text())
    assert cfg["model_type"] == "nemo_speechlm"
    assert cfg["architectures"] == ["NeMoSpeechLMHybridForConditionalGeneration"]
    assert cfg["audio_locator_tag"] == AUDIO_TOKEN
    # Original LLM fields are preserved.
    assert cfg["hidden_size"] == 2048


def test_prepare_for_vllm_adds_audio_token_to_vocab(tmp_path):
    """Audio token is registered via add_special_tokens when not already in vocab."""
    fake_tok = _FakeTokenizer(vocab_tokens=["<|im_start|>", "<|im_end|>"])
    _run_prepare(tmp_path, fake_tok)
    assert fake_tok.add_special_tokens_calls == [[AUDIO_TOKEN]]
    assert AUDIO_TOKEN in fake_tok.get_vocab()


def test_prepare_for_vllm_skips_add_if_audio_token_already_in_vocab(tmp_path):
    """Avoid re-adding a token that was already present in the backbone vocab."""
    fake_tok = _FakeTokenizer(vocab_tokens=["<|im_start|>", "<|im_end|>", AUDIO_TOKEN])
    _run_prepare(tmp_path, fake_tok)
    assert fake_tok.add_special_tokens_calls == []


def test_prepare_for_vllm_tokenizer_config_normalized(tmp_path):
    """tokenizer_config.json has dict-form extra_special_tokens + forced tokenizer_class."""
    output_dir = _run_prepare(tmp_path, _FakeTokenizer(tokenizer_class="TokenizersBackend"))
    tok_cfg = json.loads((output_dir / "tokenizer_config.json").read_text())
    assert tok_cfg["tokenizer_class"] == "PreTrainedTokenizerFast"
    assert tok_cfg["extra_special_tokens"] == {"audio_token": AUDIO_TOKEN}


def test_prepare_for_vllm_preserves_inline_chat_template_verbatim(tmp_path):
    """No enable_thinking patching: chat_template is byte-identical after prep."""
    output_dir = _run_prepare(tmp_path, _FakeTokenizer(chat_template=CHAT_TEMPLATE_INLINE))
    tok_cfg = json.loads((output_dir / "tokenizer_config.json").read_text())
    assert tok_cfg["chat_template"] == CHAT_TEMPLATE_INLINE


def test_prepare_for_vllm_rescues_chat_template_jinja_file(tmp_path):
    """Qwen3-style: chat_template split to .jinja file → inlined + file removed."""
    fake_tok = _FakeTokenizer(chat_template=CHAT_TEMPLATE_LARGE, split_chat_template=True)
    output_dir = _run_prepare(tmp_path, fake_tok)
    tok_cfg = json.loads((output_dir / "tokenizer_config.json").read_text())
    assert tok_cfg["chat_template"] == CHAT_TEMPLATE_LARGE
    assert not (output_dir / "chat_template.jinja").exists()


def test_prepare_for_vllm_generation_config(tmp_path):
    """generation_config.json gets written with the tokenizer's eos_token_id."""
    output_dir = _run_prepare(tmp_path, _FakeTokenizer(eos_token_id=99))
    gen_cfg = json.loads((output_dir / "generation_config.json").read_text())
    assert gen_cfg == {"eos_token_id": [99]}
