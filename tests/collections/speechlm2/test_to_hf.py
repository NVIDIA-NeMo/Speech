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
"""Unit tests for ``examples/speechlm2/to_hf.py::prepare_for_vllm``.

The script lives under ``examples/`` (not an importable package), so we load
it via ``importlib`` and patch ``AutoTokenizer`` / ``_detect_vllm_architecture``
to avoid any network or real-model dependencies.
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file

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
        pad_token=None,
    ):
        self._vocab = {tok: i for i, tok in enumerate(vocab_tokens)}
        self._chat_template = chat_template
        self._split_chat_template = split_chat_template
        self._tokenizer_class = tokenizer_class
        self.eos_token_id = eos_token_id
        self.pad_token = pad_token
        self.add_special_tokens_calls = []

    @property
    def pad_token_id(self):
        return self._vocab.get(self.pad_token)

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
        if self.pad_token is not None:
            tok_cfg["pad_token"] = self.pad_token
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
                "audio_token_index": 17,
            }
        )
    )
    return tmp_path


class _FakeLLMConfig:
    def save_pretrained(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen2",
                    "architectures": ["Qwen2ForCausalLM"],
                    "hidden_size": 2048,
                }
            )
        )


class _FakeExportModel:
    cfg = {
        "pretrained_llm": "fake-model",
        "pretrained_asr": "fake-asr",
        "pretrained_weights": False,
        "dtype": "bf16",
        "torch_dtype": "bf16",
        "audio_locator_tag": AUDIO_TOKEN,
    }
    llm = type("_FakeLLM", (), {"config": _FakeLLMConfig()})()


def test_hf_export_config_does_not_persist_remote_code_trust():
    model = SimpleNamespace(cfg={"trust_remote_code": True})

    config = to_hf._hf_export_config(model, "bfloat16")

    assert "trust_remote_code" not in config
    assert model.cfg["trust_remote_code"] is True


def test_save_hf_checkpoint_writes_llm_backbone_config(tmp_path):
    cfg = to_hf.HfExportConfig(
        class_path="fake.Class",
        ckpt_path="fake.ckpt",
        ckpt_config="fake.yaml",
        output_dir=str(tmp_path),
        dtype="bfloat16",
    )
    to_hf.save_hf_checkpoint(_FakeExportModel(), {"weight": torch.zeros(1)}, cfg)

    root_cfg = json.loads((tmp_path / "config.json").read_text())
    llm_cfg = json.loads((tmp_path / "llm_backbone" / "config.json").read_text())

    assert "llm_config" not in root_cfg
    assert root_cfg["pretrained_llm"] == "fake-model"
    assert root_cfg["dtype"] == "bfloat16"
    assert root_cfg["torch_dtype"] == "bfloat16"
    assert llm_cfg["model_type"] == "qwen2"
    assert llm_cfg["architectures"] == ["Qwen2ForCausalLM"]
    assert _FakeExportModel.cfg["dtype"] == "bf16"


def test_save_hf_checkpoint_accepts_bf16_export_dtype(tmp_path):
    cfg = to_hf.HfExportConfig(
        class_path="fake.Class",
        ckpt_path="fake.ckpt",
        ckpt_config="fake.yaml",
        output_dir=str(tmp_path),
        dtype="bf16",
    )
    to_hf.save_hf_checkpoint(_FakeExportModel(), {"weight": torch.zeros(1)}, cfg)

    root_cfg = json.loads((tmp_path / "config.json").read_text())
    state_dict = load_file(tmp_path / "model.safetensors")

    assert root_cfg["dtype"] == "bfloat16"
    assert root_cfg["torch_dtype"] == "bfloat16"
    assert state_dict["weight"].dtype == torch.bfloat16


def test_save_hf_checkpoint_preserves_adapter_forced_fp32_tensors(tmp_path):
    class _Adapter:
        @staticmethod
        def forced_hf_dtype_mapping(state_dict):
            return {"mamba_state": "F32"}

    model = _FakeExportModel()
    model.llm = SimpleNamespace(config=_FakeLLMConfig(), state_dict_adapter=_Adapter())
    cfg = to_hf.HfExportConfig(
        class_path="fake.Class",
        ckpt_path="fake.ckpt",
        ckpt_config="fake.yaml",
        output_dir=str(tmp_path),
        dtype="bfloat16",
    )
    to_hf.save_hf_checkpoint(
        model,
        {"mamba_state": torch.ones(1), "ordinary_weight": torch.ones(1)},
        cfg,
    )
    exported = load_file(tmp_path / "model.safetensors")

    assert exported["mamba_state"].dtype == torch.float32
    assert exported["ordinary_weight"].dtype == torch.bfloat16


def test_hf_export_config_embeds_portable_phpee_architecture():
    original_cfg = {
        "pe_encoder_path": "/models/placeholderParallelExpertEncoder.nemo",
        "pe_encoder_overrides": {
            "speaker_feature_config_version": 1,
            "speaker_feature_mode": "thresholded",
            "speaker_activity_threshold": 0.5,
        },
    }
    pe_encoder = SimpleNamespace(
        _bundle_config=OmegaConf.create({"target": "ParallelExpertEncoderPT", "asr_chunk_size_seconds": None}),
        asr_normalize_type="per_feature",
        diar_normalize_type="per_feature",
        asr_chunk_size_seconds=30.0,
        diar_chunk_size_seconds=30.0,
        frame_shift_seconds=0.01,
        missing_rttm_target=-1.0,
        speaker_feature_mode="thresholded",
        speaker_activity_threshold=0.5,
        spk_kernel_scale=1.0,
    )
    model = SimpleNamespace(cfg=original_cfg, perception=SimpleNamespace(encoder=pe_encoder))

    exported = to_hf._hf_export_config(model, "bfloat16")

    assert exported["pe_encoder_path"] is None
    assert exported["pe_encoder_config"]["target"] == "ParallelExpertEncoderPT"
    assert exported["pe_encoder_config"]["speaker_feature_config_version"] == 1
    assert exported["pe_encoder_config"]["speaker_feature_mode"] == "thresholded"
    assert exported["pe_encoder_config"]["speaker_activity_threshold"] == 0.5
    assert exported["pe_encoder_config"]["diar_normalize_type"] == "per_feature"
    assert exported["pe_encoder_config"]["asr_chunk_size_seconds"] == 30.0
    assert exported["pe_encoder_config"]["diar_chunk_size_seconds"] == 30.0
    assert "pe_encoder_overrides" not in exported
    assert original_cfg["pe_encoder_overrides"]["speaker_feature_mode"] == "thresholded"

    pe_encoder.asr_chunk_size_seconds = 60.0
    model.cfg = exported
    reexported = to_hf._hf_export_config(model, "bfloat16")
    assert reexported["pe_encoder_path"] is None
    assert reexported["pe_encoder_config"]["asr_chunk_size_seconds"] == 60.0
    assert exported["pe_encoder_config"]["asr_chunk_size_seconds"] == 30.0


def test_hf_export_config_embeds_portable_independent_dual_architecture():
    original_cfg = {
        "speaker_encoder": {"path": "/models/speaker-transformer", "frozen": True},
    }
    dual = SimpleNamespace(
        auxiliary_encoder_config={"_target_": "example.SpeakerEncoder", "d_model": 512},
        freeze_auxiliary=True,
        auxiliary_chunk_size_seconds=120.0,
        asr_chunk_size_seconds=None,
    )
    model = SimpleNamespace(cfg=original_cfg, perception=SimpleNamespace(encoder=dual))

    exported = to_hf._hf_export_config(model, "bfloat16")

    assert exported["speaker_encoder"] == {
        "encoder_config": {"_target_": "example.SpeakerEncoder", "d_model": 512},
        "frozen": True,
        "chunk_size_seconds": 120.0,
        "asr_chunk_size_seconds": None,
    }
    assert "path" not in exported["speaker_encoder"]
    assert original_cfg == {"speaker_encoder": {"path": "/models/speaker-transformer", "frozen": True}}


def test_prepare_for_vllm_missing_pretrained_llm(tmp_path):
    with pytest.raises(ValueError, match="pretrained_llm"):
        to_hf.prepare_for_vllm(str(tmp_path), {"audio_locator_tag": AUDIO_TOKEN})


def test_prepare_for_vllm_missing_audio_locator_tag(tmp_path):
    with pytest.raises(ValueError, match="audio_locator_tag"):
        to_hf.prepare_for_vllm(str(tmp_path), {"pretrained_llm": "fake-model"})


def test_detect_vllm_architecture_returns_model_embedding_vocab():
    """Exporter bounds must use the model config, not tokenizer.vocab_size."""
    backbone = SimpleNamespace(architectures=["Qwen2ForCausalLM"], vocab_size=151936)
    with patch("transformers.AutoConfig.from_pretrained", return_value=backbone):
        architecture, vocab_size = to_hf._detect_vllm_architecture({"pretrained_llm": "fake-model"})

    assert architecture == "NeMoSpeechLMForConditionalGeneration"
    assert vocab_size == 151936


@pytest.mark.parametrize("vocab_size", [None, True, 0, -1])
def test_detect_vllm_architecture_rejects_invalid_model_vocab(vocab_size):
    backbone = SimpleNamespace(architectures=["Qwen2ForCausalLM"], vocab_size=vocab_size)
    with (
        patch("transformers.AutoConfig.from_pretrained", return_value=backbone),
        pytest.raises(ValueError, match="vocab_size"),
    ):
        to_hf._detect_vllm_architecture({"pretrained_llm": "fake-model"})


# ──────────────────────────────────────────────────────────────────────
# Happy paths (mock AutoTokenizer + _detect_vllm_architecture)
# ──────────────────────────────────────────────────────────────────────


def _run_prepare(
    tmp_path,
    fake_tok,
    arch="NeMoSpeechLMForConditionalGeneration",
    llm_arch="Qwen2ForCausalLM",
    backbone_vocab_size=100,
    model_cfg=None,
):
    output_dir = _seed_output_dir(tmp_path, llm_arch=llm_arch)
    with (
        patch.object(
            to_hf,
            "_detect_vllm_architecture",
            return_value=(arch, backbone_vocab_size),
        ),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tok),
    ):
        cfg = {"pretrained_llm": "fake-model", "audio_locator_tag": AUDIO_TOKEN}
        cfg.update(model_cfg or {})
        to_hf.prepare_for_vllm(str(output_dir), cfg)
    return output_dir


def test_prepare_for_vllm_patches_config_json(tmp_path):
    """config.json gets model metadata without persisting token-index fields."""
    output_dir = _run_prepare(tmp_path, _FakeTokenizer())
    cfg = json.loads((output_dir / "config.json").read_text())
    assert cfg["model_type"] == "nemo_speechlm"
    assert cfg["architectures"] == ["NeMoSpeechLMForConditionalGeneration"]
    assert cfg["audio_locator_tag"] == AUDIO_TOKEN
    assert "audio_token_index" not in cfg
    # Original LLM fields are preserved.
    assert cfg["hidden_size"] == 2048


def test_prepare_for_vllm_embeds_bundled_backbone_config(tmp_path):
    """The vLLM root config must not reload the stale training backbone reference."""
    output_dir = _seed_output_dir(tmp_path)
    backbone_dir = output_dir / "llm_backbone"
    backbone_dir.mkdir()
    backbone_config = {
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 2048,
        "vocab_size": 100,
    }
    (backbone_dir / "config.json").write_text(json.dumps(backbone_config))

    with (
        patch.object(
            to_hf,
            "_detect_vllm_architecture",
            return_value=("NeMoSpeechLMForConditionalGeneration", 100),
        ) as detect_architecture,
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=_FakeTokenizer(),
        ),
    ):
        to_hf.prepare_for_vllm(
            str(output_dir),
            {
                "pretrained_llm": "stale-training-model",
                "audio_locator_tag": AUDIO_TOKEN,
            },
        )

    cfg = json.loads((output_dir / "config.json").read_text())
    assert cfg["pretrained_llm"] == "llm_backbone"
    assert cfg["llm_config"] == backbone_config
    detect_architecture.assert_called_once_with(
        {
            "pretrained_llm": str(backbone_dir),
            "audio_locator_tag": AUDIO_TOKEN,
        }
    )


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


def test_prepare_for_vllm_rejects_invalid_audio_token_id(tmp_path):
    fake_tok = _FakeTokenizer(vocab_tokens=[AUDIO_TOKEN])
    fake_tok._vocab[AUDIO_TOKEN] = -1

    with pytest.raises(ValueError, match="valid ID"):
        _run_prepare(tmp_path, fake_tok)


def test_prepare_for_vllm_rejects_audio_token_outside_padded_embeddings(tmp_path):
    tokens = [f"<extra_{i}>" for i in range(11)] + [AUDIO_TOKEN]
    fake_tok = _FakeTokenizer(vocab_tokens=tokens)

    with pytest.raises(ValueError, match="outside the SpeechLM embedding table"):
        _run_prepare(tmp_path, fake_tok, backbone_vocab_size=1)


def test_prepare_for_vllm_accepts_audio_token_inside_non_boundary_embedding_row(
    tmp_path,
):
    """Qwen-style reserved rows may put the audio token below the model-vocab boundary."""
    fake_tok = _FakeTokenizer(vocab_tokens=["<pad>", AUDIO_TOKEN, "<extra>"])

    output_dir = _run_prepare(tmp_path, fake_tok, backbone_vocab_size=100)

    cfg = json.loads((output_dir / "config.json").read_text())
    assert "audio_token_index" not in cfg


def test_prepare_for_vllm_uses_training_tokenizer_path(tmp_path):
    """Conversion must persist the tokenizer that supplied training token IDs."""
    output_dir = _seed_output_dir(tmp_path)
    fake_tok = _FakeTokenizer()
    with (
        patch.object(
            to_hf,
            "_detect_vllm_architecture",
            return_value=("NeMoSpeechLMForConditionalGeneration", 100),
        ),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tok) as load_tokenizer,
    ):
        to_hf.prepare_for_vllm(
            str(output_dir),
            {
                "pretrained_llm": "fake-model",
                "tokenizer_path": "custom-tokenizer",
                "audio_locator_tag": AUDIO_TOKEN,
            },
        )

    load_tokenizer.assert_called_once_with("custom-tokenizer", trust_remote_code=True)


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


def test_prepare_for_vllm_preserves_model_pad_and_eos_contract(tmp_path):
    fake_tok = _FakeTokenizer(vocab_tokens=["<unk>", "<|im_end|>"], eos_token_id=1, pad_token="<|im_end|>")
    output_dir = _run_prepare(tmp_path, fake_tok, model_cfg={"pad_token": "<unk>"})

    cfg = json.loads((output_dir / "config.json").read_text())
    tok_cfg = json.loads((output_dir / "tokenizer_config.json").read_text())
    gen_cfg = json.loads((output_dir / "generation_config.json").read_text())
    assert fake_tok.pad_token == "<unk>"
    assert tok_cfg["pad_token"] == "<unk>"
    assert cfg["pad_token_id"] == 0
    assert cfg["eos_token_id"] == [1]
    assert gen_cfg == {"eos_token_id": [1], "pad_token_id": 0}


def test_prepare_for_vllm_missing_pad_token_leaves_checkpoint_unchanged(tmp_path):
    output_dir = _seed_output_dir(tmp_path)
    original_config = (output_dir / "config.json").read_bytes()
    fake_tok = _FakeTokenizer(vocab_tokens=[AUDIO_TOKEN, "<unk>"], pad_token="<unk>")

    with (
        patch.object(
            to_hf,
            "_detect_vllm_architecture",
            return_value=("NeMoSpeechLMForConditionalGeneration", 100),
        ),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tok),
        pytest.raises(ValueError, match="pad_token"),
    ):
        to_hf.prepare_for_vllm(
            str(output_dir),
            {
                "pretrained_llm": "fake-model",
                "audio_locator_tag": AUDIO_TOKEN,
                "pad_token": "<missing>",
            },
        )

    assert (output_dir / "config.json").read_bytes() == original_config
    assert not (output_dir / "tokenizer_config.json").exists()
    assert not (output_dir / "tokenizer.json").exists()
    assert not (output_dir / "generation_config.json").exists()


def test_try_prepare_for_vllm_serialization_failure_leaves_checkpoint_unchanged(tmp_path):
    class _FailingTokenizer(_FakeTokenizer):
        def save_pretrained(self, output_dir):
            (Path(output_dir) / "tokenizer_config.json").write_text('{"partial": true}')
            raise ValueError("simulated tokenizer serialization failure")

    output_dir = _seed_output_dir(tmp_path)
    original_artifacts = {
        "config.json": (output_dir / "config.json").read_bytes(),
        "tokenizer_config.json": b'{"original": "tokenizer config"}',
        "tokenizer.json": b'{"original": "tokenizer"}',
        "generation_config.json": b'{"original": "generation"}',
    }
    for name, content in original_artifacts.items():
        (output_dir / name).write_bytes(content)

    fake_tok = _FailingTokenizer(vocab_tokens=[AUDIO_TOKEN])
    with (
        patch.object(
            to_hf,
            "_detect_vllm_architecture",
            return_value=("NeMoSpeechLMForConditionalGeneration", 100),
        ),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tok),
    ):
        to_hf._try_prepare_for_vllm(
            str(output_dir),
            {"pretrained_llm": "fake-model", "audio_locator_tag": AUDIO_TOKEN},
        )

    for name, content in original_artifacts.items():
        assert (output_dir / name).read_bytes() == content
    assert list(output_dir.parent.glob(f".{output_dir.name}-vllm-*")) == []


def test_prepare_for_vllm_publication_failure_rolls_back_all_artifacts(tmp_path):
    output_dir = _seed_output_dir(tmp_path)
    original_artifacts = {
        "config.json": (output_dir / "config.json").read_bytes(),
        "tokenizer_config.json": b'{"original": "tokenizer config"}',
        "tokenizer.json": b'{"original": "tokenizer"}',
        "generation_config.json": b'{"original": "generation"}',
        "chat_template.jinja": b"original external template",
    }
    for name, content in original_artifacts.items():
        (output_dir / name).write_bytes(content)

    real_replace = Path.replace
    publication_calls = 0

    def fail_second_artifact_replace(source, target):
        nonlocal publication_calls
        if source.parent.name == "artifacts":
            publication_calls += 1
            if publication_calls == 2:
                raise OSError("simulated publication failure")
        return real_replace(source, target)

    fake_tok = _FakeTokenizer(vocab_tokens=[AUDIO_TOKEN])
    with (
        patch.object(
            to_hf,
            "_detect_vllm_architecture",
            return_value=("NeMoSpeechLMForConditionalGeneration", 100),
        ),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tok),
        patch.object(Path, "replace", new=fail_second_artifact_replace),
        pytest.raises(OSError, match="simulated publication failure"),
    ):
        to_hf.prepare_for_vllm(
            str(output_dir),
            {"pretrained_llm": "fake-model", "audio_locator_tag": AUDIO_TOKEN},
        )

    assert publication_calls == 2
    for name, content in original_artifacts.items():
        assert (output_dir / name).read_bytes() == content
    assert list(output_dir.parent.glob(f".{output_dir.name}-vllm-*")) == []


def test_adapt_strategy_collapses_incompatible_hsdp_replicate_axis() -> None:
    original = {
        "dp_size": None,
        "dp_replicate_size": 16,
        "tp_size": 1,
        "pp_size": 1,
        "cp_size": 1,
        "ep_size": 8,
    }
    adapted = to_hf._adapt_strategy_for_conversion_world(original, world_size=8)
    assert adapted["dp_replicate_size"] == 1
    assert adapted["ep_size"] == 8
    assert original["dp_replicate_size"] == 16


def test_adapt_strategy_preserves_compatible_hsdp_replicate_axis() -> None:
    original = {
        "dp_size": None,
        "dp_replicate_size": 2,
        "tp_size": 1,
        "pp_size": 1,
        "cp_size": 1,
    }
    adapted = to_hf._adapt_strategy_for_conversion_world(original, world_size=8)
    assert adapted["dp_replicate_size"] == 2
