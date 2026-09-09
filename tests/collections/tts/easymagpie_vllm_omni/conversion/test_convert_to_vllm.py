# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from __future__ import annotations

import json
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


def _fake_reference_speaker_encoder():
    layer = types.SimpleNamespace(
        pos_ff=types.SimpleNamespace(
            proj=types.SimpleNamespace(conv=types.SimpleNamespace(out_channels=8, kernel_size=(1,)))
        ),
        self_attention=types.SimpleNamespace(n_heads=2),
    )
    return types.SimpleNamespace(
        layers=[layer],
        position_embeddings=types.SimpleNamespace(num_embeddings=128),
    )


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


def test_audio_encoder_bundling_is_explicit_opt_in(monkeypatch):
    required = [
        "convert_to_vllm.py",
        "--nemo_file",
        "model.nemo",
        "--codec_model_path",
        "codec.nemo",
        "--outdir",
        "converted",
    ]

    monkeypatch.setattr(sys, "argv", required)
    assert converter.parse_args().bundle_audio_encoders is False
    assert converter.parse_args().max_audio_seconds == 30.0

    monkeypatch.setattr(sys, "argv", [*required, "--bundle-audio-encoders", "--max-audio-seconds", "12.5"])
    args = converter.parse_args()
    assert args.bundle_audio_encoders is True
    assert args.max_audio_seconds == 12.5

    monkeypatch.setattr(sys, "argv", [*required, "--bundle_audio_encoders"])
    with pytest.raises(SystemExit):
        converter.parse_args()


def test_extract_speaker_embedding_without_reference_transformer(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "nemo.collections.tts.modules.magpietts_modules",
        types.SimpleNamespace(add_special_tokens=lambda codes, codes_len, **kwargs: (codes, codes_len)),
    )

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.sample_rate = 16000
            self.codec_model_samples_per_frame = 640
            self._codec_helper = types.SimpleNamespace(
                audio_to_codes=lambda audio, audio_lens: (
                    torch.ones(1, 2, 2, dtype=torch.long),
                    torch.tensor([2]),
                )
            )
            self._codec_converter = None
            self.context_audio_bos_id = 34
            self.context_audio_eos_id = 35
            self.frame_stacking_factor = 1
            self.num_audio_codebooks = 2
            self.use_speaker_encoder = False

        def _load_audio_for_inference(self, path, sample_rate):
            return torch.ones(1, 4)

        def _adjust_audio_to_duration_for_inference(self, audio, *args):
            return audio

        def stack_codes(self, codes, codes_len, *args):
            return codes, codes_len

        def embed_audio_tokens(self, codes):
            return torch.arange(6, dtype=torch.float32).view(1, 2, 3)

        def encode_context_audio_embeddings(self, **kwargs):
            raise AssertionError("reference-speaker Transformer must remain optional")

    result = converter.extract_speaker_embedding(_Model(), "voice.wav", 5.0)

    torch.testing.assert_close(result, torch.arange(6, dtype=torch.float32).view(2, 3))


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
    speaker_encoder = _fake_reference_speaker_encoder()
    model = types.SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "decoder_type": "nemotron_h",
                "hidden_dim": 4,
                "embedding_dim": 4,
                "nemotron_h_config": {"hidden_size": 4},
                "local_transformer_type": "ar",
                "use_multiturn_dataset": True,
                "condition_on_user_speech": True,
                "use_user_speaking_token": True,
                "use_user_speaking_end_token": True,
            }
        ),
        eos_id=101,
        interruption_token_id=103,
        num_audio_codebooks=2,
        codebook_size=32,
        frame_stacking_factor=2,
        phoneme_tokenizer=None,
        mode_name_to_mode={"default": mode},
        default_inference_mode="default",
        training_modes=[],
        task_embedding=None,
        audio_bos_id=32,
        audio_eos_id=33,
        audio_user_speaking_id=37,
        audio_user_speaking_end_id=38,
        mask_token_id=36,
    )

    config = converter.build_config(model, vocab_size=104, torch_dtype="float32")

    assert config["text_eos_id"] == 101
    assert config["text_interruption_id"] == 103
    assert config["use_multiturn_dataset"] is True
    assert config["enable_phoneme_text_input"] is False
    assert config["codec_encoder_bundled"] is False
    assert config["condition_on_user_speech"] is True
    assert config["use_user_speaking_token"] is True
    assert config["use_user_speaking_end_token"] is True
    assert config["forced_audio_user_speaking_id"] == 37
    assert config["forced_audio_user_speaking_end_id"] == 38
    assert "audio_input_token_id" not in config
    assert "max_audio_seconds" not in config
    assert "codec_input_sample_rate" not in config
    assert "reference_speaker_encoder_n_layers" not in config

    with pytest.raises(ValueError, match="use_speaker_encoder=True"):
        converter.build_config(model, vocab_size=104, torch_dtype="float32", bundle_audio_encoders=True)

    model.speaker_encoder = speaker_encoder
    model.use_speaker_encoder = True
    model.sample_rate = 16000
    model.codec_model_samples_per_frame = 640
    bundled_config = converter.build_config(
        model,
        vocab_size=104,
        torch_dtype="float32",
        bundle_audio_encoders=True,
    )

    assert bundled_config["codec_encoder_bundled"] is True
    assert bundled_config["audio_input_token_id"] == 1
    assert bundled_config["max_audio_seconds"] == 30.0
    assert bundled_config["codec_input_sample_rate"] == 16000
    assert bundled_config["codec_samples_per_frame"] == 640
    assert bundled_config["reference_speaker_encoder_n_layers"] == 1
    assert bundled_config["reference_speaker_encoder_d_ffn"] == 8
    assert bundled_config["reference_speaker_encoder_n_heads"] == 2
    assert bundled_config["reference_speaker_encoder_kernel_size"] == 1
    assert bundled_config["reference_speaker_encoder_max_length"] == 128


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
        speaker_encoder=_fake_reference_speaker_encoder(),
        sample_rate=16000,
        codec_model_samples_per_frame=640,
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


def _group_fsq(num_codebooks=6, levels=(5, 5, 5, 5)):
    return types.SimpleNamespace(
        num_codebooks=num_codebooks,
        fsqs=[
            types.SimpleNamespace(num_levels=torch.tensor(levels, dtype=torch.int32).view(1, -1, 1))
            for _ in range(num_codebooks)
        ],
    )


@pytest.mark.parametrize("uses_codec_converter", [False, True])
def test_resolve_codec_layout_uses_the_effective_checkpoint_quantizer(uses_codec_converter):
    quantizer = _group_fsq()
    model = types.SimpleNamespace(
        _codec_converter=types.SimpleNamespace(vector_quantizer_new=quantizer) if uses_codec_converter else None,
        _codec_model=types.SimpleNamespace(vector_quantizer=quantizer),
        num_audio_codebooks=6,
        codebook_size=625,
        frame_stacking_factor=3,
    )

    assert converter.resolve_codec_layout(model) == (6, [5, 5, 5, 5], 3)


def test_resolve_codec_layout_rejects_unsupported_quantizer():
    model = types.SimpleNamespace(
        _codec_converter=None,
        _codec_model=types.SimpleNamespace(vector_quantizer=types.SimpleNamespace()),
        num_audio_codebooks=6,
        codebook_size=625,
        frame_stacking_factor=3,
    )

    with pytest.raises(ValueError, match="only GroupFiniteScalarQuantizer"):
        converter.resolve_codec_layout(model)


def test_resolve_codec_layout_rejects_different_fsq_groups():
    quantizer = _group_fsq(num_codebooks=2, levels=(4, 4))
    quantizer.fsqs[1].num_levels = torch.tensor([2, 8], dtype=torch.int32).view(1, -1, 1)
    model = types.SimpleNamespace(
        _codec_converter=types.SimpleNamespace(vector_quantizer_new=quantizer),
        _codec_model=None,
        num_audio_codebooks=2,
        codebook_size=16,
        frame_stacking_factor=1,
    )

    with pytest.raises(ValueError, match="one shared num_levels_per_group"):
        converter.resolve_codec_layout(model)


def test_convert_codec_artifacts_always_exports_decoder_and_guards_audio_encoders(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(converter.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))

    layout = {
        "num_codebooks": 6,
        "frame_stacking_factor": 3,
        "num_levels_per_group": [5, 5, 5, 5],
    }
    encoder_path = converter.convert_codec_artifacts("codec.nemo", str(tmp_path), **layout)

    assert encoder_path is None
    assert "--encoder-output" not in calls[0][0]
    assert calls[0][0][4:] == [
        "--num-codebooks",
        "6",
        "--frame-stacking-factor",
        "3",
        "--num-levels-per-group",
        "5",
        "5",
        "5",
        "5",
    ]
    assert calls[0][1]["check"] is True

    encoder_path = converter.convert_codec_artifacts(
        "codec.nemo",
        str(tmp_path),
        **layout,
        bundle_audio_encoders=True,
    )

    assert encoder_path == str(tmp_path / "codec_encoder.safetensors")
    assert calls[1][0][-2:] == ["--encoder-output", encoder_path]
    assert calls[1][1]["check"] is True


def test_configure_codec_reference_speaker_encoder_updates_separate_tower_config(tmp_path):
    (tmp_path / "codec_encoder.json").write_text('{"sample_rate": 16000}')
    config = {
        "codebook_size": 32,
        "embedding_dim": 4,
        "reference_speaker_encoder_n_layers": 1,
        "reference_speaker_encoder_d_ffn": 8,
        "reference_speaker_encoder_n_heads": 2,
        "reference_speaker_encoder_kernel_size": 1,
        "reference_speaker_encoder_max_length": 16,
    }

    converter.configure_codec_reference_speaker_encoder(str(tmp_path), config)

    updated = json.loads((tmp_path / "codec_encoder.json").read_text())
    assert "use_speaker_encoder" not in updated
    assert updated["reference_speaker_encoder_d_ffn"] == 8
    assert updated["context_audio_bos_id"] == 34
    assert updated["context_audio_eos_id"] == 35


def test_append_reference_speaker_encoder_weights_keeps_them_in_codec_tower_shard(tmp_path):
    encoder_path = tmp_path / "codec_encoder.safetensors"
    converter.save_file({"audio_encoder.anchor": torch.ones(1)}, str(encoder_path))
    state = {
        "speaker_encoder.layers.0.norm_self.weight": torch.ones(4, dtype=torch.float16),
        "speaker_encoder.layers.0.self_attention.causal_mask": torch.ones(1),
        "decoder.weight": torch.zeros(1),
    }

    converter.append_reference_speaker_encoder_weights(str(encoder_path), state)

    loaded = converter.load_file(str(encoder_path), device="cpu")
    assert set(loaded) == {
        "audio_encoder.anchor",
        "reference_speaker_encoder.layers.0.norm_self.weight",
    }
    assert loaded["reference_speaker_encoder.layers.0.norm_self.weight"].dtype == torch.float32


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
        ("local_transformer_type", "none", "local_transformer_type='ar'/'autoregressive'"),
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
