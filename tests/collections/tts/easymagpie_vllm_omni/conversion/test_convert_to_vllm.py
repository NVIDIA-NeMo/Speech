# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from __future__ import annotations

import sys
import types
from pathlib import Path

import convert_to_vllm as converter  # noqa: E402
import pytest
import torch
from omegaconf import OmegaConf


class _FakeEmbeddingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.text_embedding = None
        self.cfg_unk_token_id = 12
        self.interruption_token_id = 13
        self.cfg = types.SimpleNamespace(embedding_dim=2)

    def embed_text_tokens(self, ids, text_lens, disable_cas_embedding):
        del text_lens, disable_cas_embedding
        return ids.unsqueeze(-1).expand(-1, -1, self.cfg.embedding_dim).float()


def test_precompute_text_embeddings_includes_multiturn_interruption_token():
    table = converter.precompute_text_embeddings(_FakeEmbeddingModel(), batch_size=8)

    assert table.shape == (14, 2)
    torch.testing.assert_close(table[-1], torch.tensor([13.0, 13.0]))


def test_precompute_text_embeddings_uses_explicit_cas_only_vocabulary_size():
    model = _FakeEmbeddingModel()
    model.text_vocab_size = 20

    table = converter.precompute_text_embeddings(model, batch_size=8)

    assert table.shape == (20, 2)
    torch.testing.assert_close(table[-1], torch.tensor([19.0, 19.0]))


def test_build_config_exports_multiturn_text_metadata(monkeypatch):
    class _FakeNemotronHConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "nemo.collections.tts.modules.nemotron_h_decoder",
        types.SimpleNamespace(NemotronHConfig=_FakeNemotronHConfig),
    )
    mode = types.SimpleNamespace(
        text_input_mode="streaming",
        streaming_phonemes_delay=3,
        streaming_speech_delay=5,
    )
    model = types.SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "decoder_type": "nemotron_h",
                "hidden_dim": 4,
                "embedding_dim": 4,
                "nemotron_h_config": {"hidden_size": 4},
                "local_transformer_type": "ar",
                "use_multiturn_dataset": True,
            }
        ),
        eos_id=101,
        interruption_token_id=103,
        num_audio_codebooks=2,
        codebook_size=32,
        frame_stacking_factor=1,
        phoneme_tokenizer=None,
        mode_name_to_mode={"default": mode},
        default_inference_mode="default",
        training_modes=[],
        task_embedding=None,
        audio_bos_id=32,
        audio_eos_id=33,
        mask_token_id=36,
    )

    config = converter.build_config(model, vocab_size=104, torch_dtype="float32")

    assert config["text_eos_id"] == 101
    assert config["text_interruption_id"] == 103
    assert config["use_multiturn_dataset"] is True
    assert config["enable_phoneme_text_input"] is False


def test_build_config_exports_pronunciation_control_metadata(monkeypatch):
    class _FakeNemotronHConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "nemo.collections.tts.modules.nemotron_h_decoder",
        types.SimpleNamespace(NemotronHConfig=_FakeNemotronHConfig),
    )
    mode = types.SimpleNamespace(
        text_input_mode="streaming",
        streaming_phonemes_delay=3,
        streaming_speech_delay=5,
    )
    phoneme_tokenizer = types.SimpleNamespace(bos_token_id=5, eos_token_id=6, unk_token_id=7)
    model = types.SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "decoder_type": "nemotron_h",
                "hidden_dim": 4,
                "embedding_dim": 4,
                "nemotron_h_config": {"hidden_size": 4},
                "local_transformer_type": "ar",
            }
        ),
        eos_id=101,
        num_audio_codebooks=2,
        codebook_size=32,
        frame_stacking_factor=1,
        phoneme_tokenizer=phoneme_tokenizer,
        phoneme_stacking_factor=1,
        phoneme_vocab_size=8,
        phoneme_confidence_unk_threshold=0.0,
        mode_name_to_mode={"default": mode},
        default_inference_mode="default",
        training_modes=[],
        task_embedding=None,
        audio_bos_id=32,
        audio_eos_id=33,
        mask_token_id=36,
        enable_phoneme_text_input=True,
        text_phoneme_token_offset=104,
        text_phoneme_vocab_size=8,
        phoneme_text_bop_marker="<bop>",
        phoneme_text_eop_marker="<eop>",
    )

    config = converter.build_config(model, vocab_size=112, torch_dtype="float32")

    assert config["enable_phoneme_text_input"] is True
    assert config["text_phoneme_token_offset"] == 104
    assert config["text_phoneme_vocab_size"] == 8
    assert config["phoneme_text_bop_marker"] == "<bop>"
    assert config["phoneme_text_eop_marker"] == "<eop>"
    assert config["text_phoneme_tokenizer_file"] == "phoneme_text_tokenizer/tokenizer.json"


def test_save_phoneme_text_tokenizer_exports_raw_tokenizer(tmp_path):
    class _FakeRawTokenizer:
        def get_vocab(self):
            return {f"p{i}": i for i in range(5)}

        def save(self, path):
            Path(path).write_text("{}")

    model = types.SimpleNamespace(
        enable_phoneme_text_input=True,
        text_phoneme_vocab_size=8,
        phoneme_tokenizer=types.SimpleNamespace(_tokenizer=_FakeRawTokenizer()),
    )

    converter.save_phoneme_text_tokenizer(model, str(tmp_path))

    assert (tmp_path / "phoneme_text_tokenizer" / "tokenizer.json").is_file()


def _validation_model():
    mode = types.SimpleNamespace(
        text_input_mode="streaming",
        streaming_phonemes_delay=3,
        streaming_speech_delay=5,
    )
    return types.SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "decoder_type": "nemotron_h",
                "hidden_dim": 4,
                "embedding_dim": 4,
                "nemotron_h_config": {"hidden_size": 4},
                "local_transformer_type": "ar",
            }
        ),
        mode_name_to_mode={"default": mode},
        default_inference_mode="default",
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("decoder_type", "huggingface", "Nemotron-H"),
        ("local_transformer_type", "none", "local_transformer_type='autoregressive'"),
        ("hidden_dim", 8, "hidden_dim.*embedding_dim"),
    ],
)
def test_validate_model_config_rejects_unsupported_model(field, value, message):
    model = _validation_model()
    model.cfg[field] = value

    with pytest.raises(ValueError, match=message):
        converter.validate_model_config(model)


def test_validate_model_config_accepts_autoregressive_local_transformer():
    model = _validation_model()
    model.cfg.local_transformer_type = "autoregressive"

    converter.validate_model_config(model)


def test_validate_model_config_rejects_non_streaming_default_mode():
    model = _validation_model()
    model.mode_name_to_mode["default"].text_input_mode = "full"

    with pytest.raises(ValueError, match="text_input_mode='streaming'"):
        converter.validate_model_config(model)
