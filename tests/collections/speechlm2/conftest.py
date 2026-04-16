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

"""Shared fixtures for speechlm2 tests."""

from __future__ import annotations

import json
import os

# nemotron_voicechat_pipeline_{parity,nocrash} tests set
# torch.use_deterministic_algorithms(True), which requires CuBLAS to have a
# deterministic workspace.  CuBLAS reads this env var only once — at
# initialization (first CUDA matmul in the process) — so it must be set here,
# before any fixture or test triggers CUDA work.  The setting is harmless for
# non-deterministic tests: it only reserves 32 KB of extra GPU workspace and
# has no effect unless deterministic mode is active.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pytest
import soundfile as sf
import torch

from nemo.collections.speechlm2.models import NemotronVoiceChat

_pretrained_llm = "TinyLlama/TinyLlama_v1.1"
if os.path.exists("/home/TestData/speechlm/pretrained_models"):
    _pretrained_llm = "/home/TestData/speechlm/pretrained_models/TinyLlama--TinyLlama_v1.1"


def _tiny_voicechat_config(*, predict_user_text: bool = True, streaming_encoder: bool = False) -> dict:
    """Return a minimal NemotronVoiceChat config with random weights.

    Args:
        predict_user_text: Enable ASR head for user text prediction.
        streaming_encoder: When True, configure the conformer encoder for
            cache-aware streaming (causal convolutions, chunked_limited
            attention) matching the real checkpoint.  When False, use
            default (non-causal) settings suitable for offline tests.
    """
    encoder_cfg: dict = {
        "_target_": "nemo.collections.asr.modules.ConformerEncoder",
        "feat_in": 80,
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 1,
        "subsampling_factor": 8,
    }
    if streaming_encoder:
        encoder_cfg.update({
            "subsampling": "dw_striding",
            "causal_downsampling": True,
            "att_context_size": [70, 0],
            "att_context_style": "chunked_limited",
            "conv_kernel_size": 9,
            "conv_context_size": "causal",
        })

    return {
        "model": {
            "scoring_asr": "stt_en_fastconformer_transducer_large",
            "stt": {
                "model": {
                    "pretrained_llm": _pretrained_llm,
                    "pretrained_weights": False,
                    "predict_user_text": predict_user_text,
                    "audio_loss_weight": 1,
                    "text_loss_weight": 3,
                    "source_sample_rate": 16000,
                    "validation_save_path": "/tmp/test_duplex_stt_logs",
                    "perception": {
                        "_target_": "nemo.collections.speechlm2.modules.perception.AudioPerceptionModule",
                        "preprocessor": {
                            "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
                            "features": 80,
                        },
                        "encoder": encoder_cfg,
                        "modality_adapter": {
                            "_target_": "nemo.collections.speechlm2.modules.perception.IdentityConnector",
                            "d_model": 512,
                        },
                        "output_dim": 2048,
                    },
                    "optimizer": {"_target_": "torch.optim.AdamW"},
                },
                "data": {"source_sample_rate": 16000},
                "exp_manager": {"explicit_log_dir": "/tmp/test_duplex_stt_logs"},
            },
            "speech_generation": {
                "model": {
                    "pretrained_lm_name": _pretrained_llm,
                    "pretrained_ae_dir": None,
                    "pretrained_tts_model": None,
                    "scoring_asr": "stt_en_fastconformer_transducer_large",
                    "freeze_params": [r"^audio_codec\..+$", r"^embed_tokens\..+$"],
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                    "pad_token": "<SPECIAL_12>",
                    "audio_codec_run_dtype": "float32",
                    "prevent_freeze_params": [],
                    "audio_save_path": "",
                    "inference_guidance_scale": 0.5,
                    "inference_noise_scale": 0.8,
                    "inference_top_p_or_k": 0.8,
                    "inference_guidance_enabled": False,
                    "subword_mask_exactly_as_eartts": False,
                    "context_hidden_mask_exactly_as_eartts": False,
                    "optimizer": {
                        "_target_": "torch.optim.AdamW",
                        "lr": 4e-5,
                        "betas": [0.9, 0.98],
                        "weight_decay": 0,
                        "foreach": True,
                    },
                    "lr_scheduler": {
                        "_target_": "nemo.core.optim.lr_scheduler.InverseSquareRootAnnealing",
                        "warmup_steps": 2500,
                        "min_lr": 1e-6,
                        "max_steps": 100_000_000,
                    },
                    "codec_config": {
                        "latent_size": 512,
                        "n_fft": 16,
                        "hop_length": 4,
                        "base_hidden_size": 384,
                        "channel_mult": [1, 2, 4],
                        "rates": [7, 7, 9],
                        "num_blocks": 3,
                        "kernel_size": 7,
                        "groups": 1,
                        "codebook_size": 1024,
                        "num_quantizers": 31,
                        "wav_to_token_ratio": 1764,
                    },
                    "tts_config": {
                        "use_gated_fusion_for_text_audio": True,
                        "disable_eos_prediction": True,
                        "use_bos_eos_emb": True,
                        "use_subword_flag_emb": True,
                        "num_delay_speech_tokens": 2,
                        "backbone_type": "gemma3_text",
                        "backbone_model_class": None,
                        "backbone_config_class": None,
                        "backbone_config": {
                            "hidden_size": 1152,
                            "intermediate_size": 4608,
                            "num_hidden_layers": 1,
                            "num_attention_heads": 16,
                            "num_key_value_heads": 16,
                            "head_dim": 72,
                            "attention_dropout": 0.1,
                            "use_cache": False,
                        },
                        "latent_size": 512,
                        "codebook_size": 1024,
                        "num_quantizers": 31,
                        "context_hidden_size": None,
                        "cas_config": {
                            "backbone_type": "t5gemma",
                            "backbone_model_class": None,
                            "backbone_config_class": None,
                            "backbone_config": {
                                "is_encoder_decoder": False,
                                "encoder": {
                                    "hidden_size": 1152,
                                    "intermediate_size": 4608,
                                    "num_hidden_layers": 1,
                                    "num_attention_heads": 16,
                                    "num_key_value_heads": 16,
                                    "head_dim": 72,
                                    "use_cache": False,
                                    "attention_dropout": 0.1,
                                },
                            },
                        },
                        "mog_head_config": {
                            "intermediate_size": 4608,
                            "num_layers": 3,
                            "low_rank": 64,
                            "num_predictions": 1024,
                            "min_log_std": -4.0,
                            "eps": 1e-6,
                        },
                        "p_uncond": 0.1,
                        "label_smoothing": 0.01,
                        "max_training_rate": 0.8,
                        "quantizer_dropout": 0.5,
                        "random_target_masking": False,
                        "exponent": 3.0,
                    },
                },
                "data": {
                    "add_text_bos_and_eos_in_each_turn": True,
                    "add_audio_prompt": True,
                    "audio_prompt_duration": 3.0,
                    "frame_length": 0.08,
                    "source_sample_rate": 16000,
                    "target_sample_rate": 22050,
                },
                "exp_manager": {"explicit_log_dir": "/tmp/test_duplex_stt_logs"},
            },
        },
        "data": {
            "frame_length": 0.08,
            "source_sample_rate": 16000,
            "target_sample_rate": 22050,
            "input_roles": ["user", "User"],
            "output_roles": ["agent", "Assistant", "assistant", "Agent"],
        },
        "exp_manager": {"explicit_log_dir": "/tmp/test_parity_logs"},
    }


@pytest.fixture(scope="session")
def tiny_model_artifacts(tmp_path_factory):
    """Build a tiny NemotronVoiceChat with random weights, write test audio files.

    Session-scoped so the model is built only once across all test modules.
    The fixture returns only file paths (immutable), so sharing is safe.

    Returns ``(model_dir, audio_path, speaker_ref_path)``.
    """
    base = tmp_path_factory.mktemp("tiny_model")

    audio_path = str(base / "test_audio.wav")
    sf.write(audio_path, np.random.RandomState(42).randn(3 * 16000).astype(np.float32), 16000)

    speaker_ref_path = str(base / "speaker_ref.wav")
    sf.write(speaker_ref_path, np.random.RandomState(99).randn(22050).astype(np.float32), 22050)

    cfg = _tiny_voicechat_config(streaming_encoder=True)
    model = NemotronVoiceChat(cfg)
    model.to("cuda")
    model.eval()

    model_dir = str(base / "model")
    model.save_pretrained(model_dir)

    # save_pretrained writes the tokenizer to llm_artifacts/, but config.json
    # still references the HF hub name (e.g. "TinyLlama/TinyLlama_v1.1").
    # Save the LLM model config alongside the tokenizer so llm_artifacts/
    # is a complete local model reference, then rewrite config.json to point
    # at it.  This avoids HuggingFace network requests on every from_pretrained.
    llm_artifacts = os.path.join(model_dir, "llm_artifacts")
    model.stt_model.llm.config.save_pretrained(llm_artifacts)
    cfg["model"]["stt"]["model"]["pretrained_llm"] = llm_artifacts
    cfg["model"]["speech_generation"]["model"]["pretrained_lm_name"] = llm_artifacts
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(cfg, f)

    del model
    torch.cuda.empty_cache()

    return model_dir, audio_path, speaker_ref_path


@pytest.fixture(scope="session")
def tiny_voicechat_model():
    """Build a tiny NemotronVoiceChat model (predict_user_text=False).

    Used by ``test_nemotron_voicechat.py`` for validation and offline
    generation tests.
    """
    cfg = _tiny_voicechat_config(predict_user_text=False)
    model = NemotronVoiceChat(cfg)
    if torch.cuda.is_available():
        model.to("cuda")
    return model
