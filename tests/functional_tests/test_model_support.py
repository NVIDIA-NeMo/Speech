# Copyright (c) 2025, NVIDIA CORPORATION.
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

"""
Functional tests covering initialization, training step, and inference
for every non-deprecated model listed in model-support-table.csv.

Usage:
    # Run tests for one model (each bash script does this):
    pytest tests/functional_tests/test_model_support.py \
        -k "nvidia__parakeet_ctc_0_6b" -v

    # Run ALL tests:
    pytest tests/functional_tests/test_model_support.py -v
"""

import csv
import os
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Model classes (lazy-imported to avoid slow top-level import)
# ---------------------------------------------------------------------------
_LOADER_CACHE = {}


def _get_loader(name: str):
    """Lazy-import and cache a model loader class by short name."""
    if name in _LOADER_CACHE:
        return _LOADER_CACHE[name]

    if name == "ASRModel":
        from nemo.collections.asr.models import ASRModel as cls
    elif name == "EncDecSpeakerLabelModel":
        from nemo.collections.asr.models import EncDecSpeakerLabelModel as cls
    elif name == "EncDecClassificationModel":
        from nemo.collections.asr.models import EncDecClassificationModel as cls
    elif name == "EncDecFrameClassificationModel":
        from nemo.collections.asr.models.classification_models import EncDecFrameClassificationModel as cls
    elif name == "SortformerEncLabelModel":
        from nemo.collections.asr.models import SortformerEncLabelModel as cls
    elif name == "AudioToAudioModel":
        from nemo.collections.audio.models import AudioToAudioModel as cls
    elif name == "AudioCodecModel":
        from nemo.collections.tts.models import AudioCodecModel as cls
    elif name == "FastPitchModel":
        from nemo.collections.tts.models import FastPitchModel as cls
    elif name == "HifiGanModel":
        from nemo.collections.tts.models import HifiGanModel as cls
    elif name == "MagpieTTSModel":
        from nemo.collections.tts.models import MagpieTTSModel as cls
    elif name == "SALM":
        from nemo.collections.speechlm2.models import SALM as cls
    elif name == "EncDecDenoiseMaskedTokenPredModel":
        from nemo.collections.asr.models import EncDecDenoiseMaskedTokenPredModel as cls
    else:
        raise ValueError(f"Unknown loader class: {name}")

    _LOADER_CACHE[name] = cls
    return cls


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = os.environ.get(
    "NEMO_MODEL_SUPPORT_DIR",
    os.environ.get(
        "NEMO_MODEL_SUPPORT_DIR_CI",
        "/home/TestData/nemo-speech-ci-models",
    ),
)

# ---------------------------------------------------------------------------
# Registry: model_name → {category, loader (class name string), file}
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    # ── ASR: FastConformer hybrid ──────────────────────────────────────
    "nvidia/stt_de_fastconformer_hybrid_large_pc": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_de_fastconformer_hybrid_large_pc.nemo",
    },
    "nvidia/stt_en_fastconformer_hybrid_large_pc": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_en_fastconformer_hybrid_large_pc.nemo",
    },
    "nvidia/stt_es_fastconformer_hybrid_large_pc": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_es_fastconformer_hybrid_large_pc.nemo",
    },
    "nvidia/stt_it_fastconformer_hybrid_large_pc": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_it_fastconformer_hybrid_large_pc.nemo",
    },
    "nvidia/stt_ua_fastconformer_hybrid_large_pc": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_ua_fastconformer_hybrid_large_pc.nemo",
    },
    "nvidia/stt_pl_fastconformer_hybrid_large_pc": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_pl_fastconformer_hybrid_large_pc.nemo",
    },
    "nvidia/stt_hr_fastconformer_hybrid_large_pc": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_hr_fastconformer_hybrid_large_pc.nemo",
    },
    "nvidia/stt_be_fastconformer_hybrid_large_pc": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_be_fastconformer_hybrid_large_pc.nemo",
    },
    "nvidia/stt_fr_fastconformer_hybrid_large_pc": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_fr_fastconformer_hybrid_large_pc.nemo",
    },
    "nvidia/stt_ru_fastconformer_hybrid_large_pc": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_ru_fastconformer_hybrid_large_pc.nemo",
    },
    "nvidia/stt_nl_fastconformer_hybrid_large_pc": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_nl_fastconformer_hybrid_large_pc.nemo",
    },
    "nvidia/stt_fa_fastconformer_hybrid_large": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_fa_fastconformer_hybrid_large.nemo",
    },
    "nvidia/stt_ka_fastconformer_hybrid_large_pc": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_ka_fastconformer_hybrid_large_pc.nemo",
    },
    "nvidia/stt_kk_ru_fastconformer_hybrid_large": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_kk_ru_fastconformer_hybrid_large.nemo",
    },
    "nvidia/stt_ka_fastconformer_hybrid_transducer_ctc_large_streaming_80ms_pc": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_ka_fastconformer_hybrid_transducer_ctc_large_streaming_80ms_pc.nemo",
    },
    "nvidia/stt_uz_fastconformer_hybrid_large_pc": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_uz_fastconformer_hybrid_large_pc.nemo",
    },
    "nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_ar_fastconformer_hybrid_large_pc_v1.0.nemo",
    },
    "nvidia/stt_hy_fastconformer_hybrid_large_pc": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_hy_fastconformer_hybrid_large_pc.nemo",
    },
    "nvidia/stt_en_fastconformer_hybrid_medium_streaming_80ms_pc": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_en_fastconformer_hybrid_medium_streaming_80ms_pc.nemo",
    },
    "nvidia/stt_en_fastconformer_hybrid_medium_streaming_80ms": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_en_fastconformer_hybrid_medium_streaming_80ms.nemo",
    },
    "nvidia/stt_pt_fastconformer_hybrid_large_pc": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_pt_fastconformer_hybrid_large_pc.nemo",
    },
    "nvidia/stt_es_fastconformer_hybrid_large_pc_nc": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_es_fastconformer_hybrid_large_pc_nc.nemo",
    },
    "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_ar_fastconformer_hybrid_large_pcd_v1.0.nemo",
    },
    "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_en_fastconformer_hybrid_large_streaming_multi.nemo",
    },
    # ── ASR: FastConformer CTC / Transducer / TDT ─────────────────────
    "nvidia/stt_en_fastconformer_ctc_large": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_en_fastconformer_ctc_large.nemo",
    },
    "nvidia/stt_en_fastconformer_transducer_large": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_en_fastconformer_transducer_large.nemo",
    },
    "nvidia/stt_en_fastconformer_ctc_xlarge": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_en_fastconformer_ctc_xlarge.nemo",
    },
    "nvidia/stt_en_fastconformer_transducer_xlarge": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_en_fastconformer_transducer_xlarge.nemo",
    },
    "nvidia/stt_en_fastconformer_transducer_xxlarge": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_en_fastconformer_transducer_xxlarge.nemo",
    },
    "nvidia/stt_en_fastconformer_ctc_xxlarge": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_en_fastconformer_ctc_xxlarge.nemo",
    },
    "nvidia/stt_en_fastconformer_tdt_large": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__stt_en_fastconformer_tdt_large.nemo",
    },
    # ── ASR: NGC models ───────────────────────────────────────────────
    "stt_en_fastconformer_hybrid_large_streaming_1040ms": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "stt_en_fastconformer_hybrid_large_streaming_1040ms.nemo",
    },
    "stt_multilingual_fastconformer_hybrid_large_pc_blend_eu": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "stt_multilingual_fastconformer_hybrid_large_pc_blend_eu.nemo",
    },
    # ── ASR: Parakeet ─────────────────────────────────────────────────
    "nvidia/parakeet-rnnt-1.1b": {"category": "asr", "loader": "ASRModel", "file": "nvidia__parakeet-rnnt-1.1b.nemo"},
    "nvidia/parakeet-ctc-1.1b": {"category": "asr", "loader": "ASRModel", "file": "nvidia__parakeet-ctc-1.1b.nemo"},
    "nvidia/parakeet-rnnt-0.6b": {"category": "asr", "loader": "ASRModel", "file": "nvidia__parakeet-rnnt-0.6b.nemo"},
    "nvidia/parakeet-ctc-0.6b": {"category": "asr", "loader": "ASRModel", "file": "nvidia__parakeet-ctc-0.6b.nemo"},
    "nvidia/parakeet-tdt-1.1b": {"category": "asr", "loader": "ASRModel", "file": "nvidia__parakeet-tdt-1.1b.nemo"},
    "nvidia/parakeet-tdt_ctc-1.1b": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__parakeet-tdt_ctc-1.1b.nemo",
    },
    "nvidia/parakeet-tdt_ctc-0.6b-ja": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__parakeet-tdt_ctc-0.6b-ja.nemo",
    },
    "nvidia/parakeet-tdt_ctc-110m": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__parakeet-tdt_ctc-110m.nemo",
    },
    "nvidia/parakeet-tdt-0.6b-v2": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__parakeet-tdt-0.6b-v2.nemo",
    },
    "nvidia/parakeet-rnnt-110m-da-dk": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__parakeet-rnnt-110m-da-dk.nemo",
    },
    "nvidia/parakeet-tdt-0.6b-v3": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__parakeet-tdt-0.6b-v3.nemo",
    },
    "nvidia/parakeet-ctc-0.6b-Vietnamese": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__parakeet-ctc-0.6b-Vietnamese.nemo",
    },
    # ── ASR: Canary (EncDecMultiTaskModel, loaded via ASRModel) ───────
    "nvidia/canary-1b": {"category": "asr", "loader": "ASRModel", "file": "nvidia__canary-1b.nemo"},
    "nvidia/canary-1b-flash": {"category": "asr", "loader": "ASRModel", "file": "nvidia__canary-1b-flash.nemo"},
    "nvidia/canary-180m-flash": {"category": "asr", "loader": "ASRModel", "file": "nvidia__canary-180m-flash.nemo"},
    "nvidia/canary-1b-v2": {"category": "asr", "loader": "ASRModel", "file": "nvidia__canary-1b-v2.nemo"},
    # ── ASR: Specialized ──────────────────────────────────────────────
    "nvidia/parakeet_realtime_eou_120m-v1": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__parakeet_realtime_eou_120m-v1.nemo",
    },
    "nvidia/multitalker-parakeet-streaming-0.6b-v1": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__multitalker-parakeet-streaming-0.6b-v1.nemo",
    },
    "nvidia/nemotron-speech-streaming-en-0.6b": {
        "category": "asr",
        "loader": "ASRModel",
        "file": "nvidia__nemotron-speech-streaming-en-0.6b.nemo",
    },
    # ── SALM (canary-qwen, HFHubMixin) ───────────────────────────────
    "nvidia/canary-qwen-2.5b": {"category": "salm", "loader": "SALM", "file": "nvidia__canary-qwen-2.5b"},
    # ── Diarization ───────────────────────────────────────────────────
    "nvidia/diar_sortformer_4spk-v1": {
        "category": "diarization",
        "loader": "SortformerEncLabelModel",
        "file": "nvidia__diar_sortformer_4spk-v1.nemo",
    },
    "nvidia/diar_streaming_sortformer_4spk-v2": {
        "category": "diarization",
        "loader": "SortformerEncLabelModel",
        "file": "nvidia__diar_streaming_sortformer_4spk-v2.nemo",
    },
    "nvidia/diar_streaming_sortformer_4spk-v2.1": {
        "category": "diarization",
        "loader": "SortformerEncLabelModel",
        "file": "nvidia__diar_streaming_sortformer_4spk-v2.1.nemo",
    },
    # ── Speaker ID ────────────────────────────────────────────────────
    "titanet_large": {"category": "speaker", "loader": "EncDecSpeakerLabelModel", "file": "titanet_large.nemo"},
    "nvidia/speakerverification_en_titanet_large": {
        "category": "speaker",
        "loader": "EncDecSpeakerLabelModel",
        "file": "nvidia__speakerverification_en_titanet_large.nemo",
    },
    # ── SSL ────────────────────────────────────────────────────────────
    "nvidia/ssl_en_nest_large_v1.0": {
        "category": "ssl",
        "loader": "EncDecDenoiseMaskedTokenPredModel",
        "file": "nvidia__ssl_en_nest_large_v1.0.nemo",
    },
    "nvidia/ssl_en_nest_xlarge_v1.0": {
        "category": "ssl",
        "loader": "EncDecDenoiseMaskedTokenPredModel",
        "file": "nvidia__ssl_en_nest_xlarge_v1.0.nemo",
    },
    # ── VAD ────────────────────────────────────────────────────────────
    "vad_multilingual_marblenet": {
        "category": "vad",
        "loader": "EncDecClassificationModel",
        "file": "vad_multilingual_marblenet.nemo",
    },
    # strict=False: checkpoint has legacy "loss.weight" key absent in current model class
    # EncDecFrameClassificationModel: old model was EncDecMultiClassificationModel; frame-level needs transpose
    "vad_multilingual_frame_marblenet": {
        "category": "vad",
        "loader": "EncDecFrameClassificationModel",
        "file": "vad_multilingual_frame_marblenet.nemo",
        "strict": False,
    },
    # strict=False: model class has "loss.weight" key absent in checkpoint
    "nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0": {
        "category": "vad",
        "loader": "EncDecClassificationModel",
        "file": "nvidia__Frame_VAD_Multilingual_MarbleNet_v2.0.nemo",
        "strict": False,
    },
    # ── Audio Enhancement ──────────────────────────────────────────────
    "nvidia/se_den_sb_16k_small": {
        "category": "audio_enhancement",
        "loader": "AudioToAudioModel",
        "file": "nvidia__se_den_sb_16k_small.nemo",
    },
    "nvidia/se_der_sb_16k_small": {
        "category": "audio_enhancement",
        "loader": "AudioToAudioModel",
        "file": "nvidia__se_der_sb_16k_small.nemo",
    },
    "nvidia/sr_ssl_flowmatching_16k_430m": {
        "category": "audio_enhancement",
        "loader": "AudioToAudioModel",
        "file": "nvidia__sr_ssl_flowmatching_16k_430m.nemo",
    },
    # ── Audio Codec ────────────────────────────────────────────────────
    "mel_codec_44khz_medium": {
        "category": "audio_codec",
        "loader": "AudioCodecModel",
        "file": "mel_codec_44khz_medium.nemo",
    },
    "mel_codec_22khz_fullband_medium": {
        "category": "audio_codec",
        "loader": "AudioCodecModel",
        "file": "mel_codec_22khz_fullband_medium.nemo",
    },
    "nvidia/low-frame-rate-speech-codec-22khz": {
        "category": "audio_codec",
        "loader": "AudioCodecModel",
        "file": "nvidia__low-frame-rate-speech-codec-22khz.nemo",
    },
    "nvidia/audio-codec-22khz": {
        "category": "audio_codec",
        "loader": "AudioCodecModel",
        "file": "nvidia__audio-codec-22khz.nemo",
    },
    "nvidia/audio-codec-44khz": {
        "category": "audio_codec",
        "loader": "AudioCodecModel",
        "file": "nvidia__audio-codec-44khz.nemo",
    },
    "nvidia/mel-codec-22khz": {
        "category": "audio_codec",
        "loader": "AudioCodecModel",
        "file": "nvidia__mel-codec-22khz.nemo",
    },
    "nvidia/mel-codec-44khz": {
        "category": "audio_codec",
        "loader": "AudioCodecModel",
        "file": "nvidia__mel-codec-44khz.nemo",
    },
    "nvidia/nemo-nano-codec-22khz-1.78kbps-12.5fps": {
        "category": "audio_codec",
        "loader": "AudioCodecModel",
        "file": "nvidia__nemo-nano-codec-22khz-1.78kbps-12.5fps.nemo",
    },
    "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps": {
        "category": "audio_codec",
        "loader": "AudioCodecModel",
        "file": "nvidia__nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo",
    },
    "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps": {
        "category": "audio_codec",
        "loader": "AudioCodecModel",
        "file": "nvidia__nemo-nano-codec-22khz-0.6kbps-12.5fps.nemo",
    },
    # ── TTS: FastPitch ─────────────────────────────────────────────────
    "nvidia/tts_en_fastpitch": {
        "category": "tts_fastpitch",
        "loader": "FastPitchModel",
        "file": "nvidia__tts_en_fastpitch.nemo",
    },
    # ── TTS: HifiGan ──────────────────────────────────────────────────
    "nvidia/tts_hifigan": {"category": "tts_hifigan", "loader": "HifiGanModel", "file": "nvidia__tts_hifigan.nemo"},
    # ── TTS: Magpie ───────────────────────────────────────────────────
    "nvidia/magpie_tts_multilingual_357m": {
        "category": "tts_magpie",
        "loader": "MagpieTTSModel",
        "file": "nvidia__magpie_tts_multilingual_357m.nemo",
    },
    # ── TTS: E2E (class removed from codebase) ────────────────────────
    "tts_en_e2e_fastspeech2hifigan": {
        "category": "tts_e2e",
        "loader": None,
        "file": "tts_en_e2e_fastspeech2hifigan.nemo",
    },
}

# Models whose training step requires complex/specialized batches
SKIP_TRAINING_CATEGORIES = frozenset(
    {
        "diarization",
        "audio_enhancement",
        "audio_codec",
        "tts_fastpitch",
        "tts_hifigan",
        "tts_magpie",
        "salm",
        "tts_e2e",
    }
)

# Individual ASR models that need specialized batches for training
SKIP_TRAINING_MODELS = frozenset(
    {
        "nvidia/canary-1b",
        "nvidia/canary-1b-flash",
        "nvidia/canary-180m-flash",
        "nvidia/canary-1b-v2",
        "nvidia/multitalker-parakeet-streaming-0.6b-v1",
        "nvidia/nemotron-speech-streaming-en-0.6b",
        "nvidia/parakeet_realtime_eou_120m-v1",
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_name(model_name: str) -> str:
    """Convert model name to a safe identifier for file/test naming."""
    return model_name.replace("/", "__").replace(".", "_").replace("-", "_")


_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache holds at most one model to avoid GPU OOM when running the full suite.
# When each bash script runs 3 tests for one model, the cache avoids reloading.
_model_cache: dict = {}


def load_model(model_name: str):
    """Load a model from the pre-downloaded .nemo file (or HF directory for SALM)."""
    import gc

    if model_name in _model_cache:
        return _model_cache[model_name]

    # Evict previous model(s) to free GPU memory
    if _model_cache:
        _model_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    info = MODEL_REGISTRY[model_name]
    loader_cls = _get_loader(info["loader"])
    filepath = os.path.join(MODEL_DIR, info["file"])

    strict = info.get("strict", True)
    if info["category"] == "salm":
        model = loader_cls.from_pretrained(filepath, map_location="cpu")
    else:
        model = loader_cls.restore_from(filepath, map_location="cpu", strict=strict)

    model = model.to(_DEVICE)
    _model_cache[model_name] = model
    return model


# ---------------------------------------------------------------------------
# Training-step helpers by category
# ---------------------------------------------------------------------------
def _run_training_step(model, model_name: str):
    """Execute one training step with synthetic data appropriate for the model category."""
    category = MODEL_REGISTRY[model_name]["category"]

    if category == "asr":
        _training_step_asr(model)
    elif category == "speaker":
        _training_step_speaker(model)
    elif category == "vad":
        _training_step_vad(model)
    elif category == "ssl":
        _training_step_ssl(model)
    else:
        pytest.skip(f"Training step not supported for category={category}")


def _training_step_asr(model):
    """Forward pass in train mode — verifies model graph is functional."""
    model.train()
    d = _DEVICE
    out = model.forward(
        input_signal=torch.randn(2, 16000, device=d),
        input_signal_length=torch.tensor([16000, 12000], device=d),
    )
    if isinstance(out, tuple):
        assert out[0] is not None
    else:
        assert out is not None


def _training_step_speaker(model):
    model.train()
    d = _DEVICE
    out = model.forward(
        input_signal=torch.randn(2, 16000, device=d),
        input_signal_length=torch.tensor([16000, 12000], device=d),
    )
    assert out is not None


def _training_step_vad(model):
    model.train()
    d = _DEVICE
    out = model.forward(
        input_signal=torch.randn(2, 16000, device=d),
        input_signal_length=torch.tensor([16000, 12000], device=d),
    )
    assert out is not None


def _training_step_ssl(model):
    model.train()
    d = _DEVICE
    out = model.forward(
        input_signal=torch.randn(2, 16000, device=d),
        input_signal_length=torch.tensor([16000, 12000], device=d),
        noisy_input_signal=torch.randn(2, 16000, device=d),
        noisy_input_signal_length=torch.tensor([16000, 12000], device=d),
    )
    assert out is not None


# ---------------------------------------------------------------------------
# Inference helpers by category
# ---------------------------------------------------------------------------
def _run_inference(model, model_name: str):
    """Execute one inference call with synthetic data appropriate for the model category."""
    category = MODEL_REGISTRY[model_name]["category"]

    if category == "asr":
        _inference_asr(model)
    elif category == "speaker":
        _inference_speaker(model)
    elif category == "vad":
        _inference_vad(model)
    elif category == "diarization":
        _inference_diarization(model)
    elif category == "audio_enhancement":
        _inference_audio_enhancement(model)
    elif category == "audio_codec":
        _inference_audio_codec(model)
    elif category == "tts_fastpitch":
        _inference_tts_fastpitch(model)
    elif category == "tts_hifigan":
        _inference_tts_hifigan(model)
    elif category == "tts_magpie":
        _inference_tts_magpie(model)
    elif category == "salm":
        _inference_salm(model)
    elif category == "ssl":
        _inference_ssl(model)
    else:
        pytest.skip(f"Inference not supported for category={category}")


@torch.no_grad()
def _inference_asr(model):
    model.eval()
    d = _DEVICE
    model.forward(
        input_signal=torch.randn(1, 16000, device=d),
        input_signal_length=torch.tensor([16000], device=d),
    )


@torch.no_grad()
def _inference_speaker(model):
    model.eval()
    d = _DEVICE
    model.forward(
        input_signal=torch.randn(1, 16000, device=d),
        input_signal_length=torch.tensor([16000], device=d),
    )


@torch.no_grad()
def _inference_vad(model):
    model.eval()
    d = _DEVICE
    model.forward(
        input_signal=torch.randn(1, 16000, device=d),
        input_signal_length=torch.tensor([16000], device=d),
    )


@torch.no_grad()
def _inference_diarization(model):
    model.eval()
    d = _DEVICE
    model.forward(
        audio_signal=torch.randn(1, 16000, device=d),
        audio_signal_length=torch.tensor([16000], device=d),
    )


@torch.no_grad()
def _inference_audio_enhancement(model):
    model.eval()
    d = _DEVICE
    model(
        input_signal=torch.randn(1, 1, 16000, device=d),
        input_length=torch.tensor([16000], device=d),
    )


@torch.no_grad()
def _inference_audio_codec(model):
    model.eval()
    d = _DEVICE
    tokens, tokens_len = model.encode(
        audio=torch.randn(1, 16000, device=d),
        audio_len=torch.tensor([16000], device=d),
    )
    assert tokens is not None


@torch.no_grad()
def _inference_tts_fastpitch(model):
    model.eval()
    tokens = model.parse("hello world")
    if _DEVICE.type == "cuda":
        tokens = tokens.to(_DEVICE)
    model.generate_spectrogram(tokens=tokens)


@torch.no_grad()
def _inference_tts_hifigan(model):
    model.eval()
    d = _DEVICE
    model.convert_spectrogram_to_audio(spec=torch.randn(1, 80, 100, device=d))


@torch.no_grad()
def _inference_tts_magpie(model):
    model.eval()
    model.do_tts(transcript="hello world", language="en")


@torch.no_grad()
def _inference_salm(model):
    model.eval()
    d = _DEVICE
    hidden_size = model.llm.config.hidden_size
    result = model.forward(input_embeds=torch.randn(1, 10, hidden_size, device=d))
    assert "logits" in result


@torch.no_grad()
def _inference_ssl(model):
    model.eval()
    d = _DEVICE
    kwargs = dict(
        input_signal=torch.randn(1, 16000, device=d),
        input_signal_length=torch.tensor([16000], device=d),
    )
    # EncDecDenoiseMaskedTokenPredModel requires noisy_input_signal
    from nemo.collections.asr.models.ssl_models import EncDecDenoiseMaskedTokenPredModel

    if isinstance(model, EncDecDenoiseMaskedTokenPredModel):
        kwargs["noisy_input_signal"] = torch.randn(1, 16000, device=d)
        kwargs["noisy_input_signal_length"] = torch.tensor([16000], device=d)
    model.forward(**kwargs)


# ---------------------------------------------------------------------------
# Parametrized test data
# ---------------------------------------------------------------------------
SUPPORTED_MODELS = list(MODEL_REGISTRY.keys())
_SAFE_IDS = [_safe_name(m) for m in SUPPORTED_MODELS]

# Build parametrize list: xfail for models with loader=None
_MODEL_PARAMS = [
    (
        pytest.param(
            name,
            id=_safe_name(name),
            marks=pytest.mark.xfail(reason="model class removed from codebase", strict=True),
        )
        if MODEL_REGISTRY[name]["loader"] is None
        else pytest.param(name, id=_safe_name(name))
    )
    for name in SUPPORTED_MODELS
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_name", _MODEL_PARAMS)
def test_model_init(model_name):
    """Test that a model can be loaded and its config round-trips."""
    info = MODEL_REGISTRY[model_name]
    if info["loader"] is None:
        pytest.fail("model class removed from codebase")
    model = load_model(model_name)
    assert model is not None
    if hasattr(model, "to_config_dict"):
        cfg = model.to_config_dict()
        assert cfg is not None


@pytest.mark.parametrize("model_name", _MODEL_PARAMS)
def test_model_training_step(model_name):
    """Test that a model can execute one training step with synthetic data."""
    info = MODEL_REGISTRY[model_name]
    if info["loader"] is None:
        pytest.fail("model class removed from codebase")
    if info["category"] in SKIP_TRAINING_CATEGORIES:
        pytest.skip(f"training step skipped for category={info['category']}")
    if model_name in SKIP_TRAINING_MODELS:
        pytest.skip(f"training step skipped for {model_name} (specialized batch required)")
    model = load_model(model_name)
    _run_training_step(model, model_name)


@pytest.mark.parametrize("model_name", _MODEL_PARAMS)
def test_model_inference(model_name):
    """Test that a model can run inference with synthetic data."""
    info = MODEL_REGISTRY[model_name]
    if info["loader"] is None:
        pytest.fail("model class removed from codebase")
    model = load_model(model_name)
    _run_inference(model, model_name)


# ---------------------------------------------------------------------------
# CSV coverage verification
# ---------------------------------------------------------------------------
def test_all_csv_models_have_registry_entry():
    """Verify that every supported model in model-support-table.csv has a registry entry."""
    csv_path = Path(__file__).resolve().parents[2] / "model-support-table.csv"
    if not csv_path.exists():
        pytest.skip("model-support-table.csv not found")

    missing = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 7:
                continue
            support = row[6].strip()
            if support not in ("✅", "?"):
                continue
            model_name = row[4].strip()
            if model_name not in MODEL_REGISTRY:
                missing.append(model_name)

    assert not missing, f"Models in CSV but not in MODEL_REGISTRY: {missing}"
