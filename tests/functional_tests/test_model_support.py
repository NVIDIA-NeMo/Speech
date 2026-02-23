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
CSV coverage verification: ensures every supported model in
model-support-table.csv has a corresponding per-model test file.

Individual model tests live in test_model_support_<safe_name>.py files.
"""

import csv
import os
from pathlib import Path

import pytest

# All model names that have per-model test files.
# Keep this in sync with MODEL_REGISTRY in scripts/ci/download_model_support_models.py.
SUPPORTED_MODELS = [
    # ASR: FastConformer hybrid
    "nvidia/stt_de_fastconformer_hybrid_large_pc",
    "nvidia/stt_en_fastconformer_hybrid_large_pc",
    "nvidia/stt_es_fastconformer_hybrid_large_pc",
    "nvidia/stt_it_fastconformer_hybrid_large_pc",
    "nvidia/stt_ua_fastconformer_hybrid_large_pc",
    "nvidia/stt_pl_fastconformer_hybrid_large_pc",
    "nvidia/stt_hr_fastconformer_hybrid_large_pc",
    "nvidia/stt_be_fastconformer_hybrid_large_pc",
    "nvidia/stt_fr_fastconformer_hybrid_large_pc",
    "nvidia/stt_ru_fastconformer_hybrid_large_pc",
    "nvidia/stt_nl_fastconformer_hybrid_large_pc",
    "nvidia/stt_fa_fastconformer_hybrid_large",
    "nvidia/stt_ka_fastconformer_hybrid_large_pc",
    "nvidia/stt_kk_ru_fastconformer_hybrid_large",
    "nvidia/stt_ka_fastconformer_hybrid_transducer_ctc_large_streaming_80ms_pc",
    "nvidia/stt_uz_fastconformer_hybrid_large_pc",
    "nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0",
    "nvidia/stt_hy_fastconformer_hybrid_large_pc",
    "nvidia/stt_en_fastconformer_hybrid_medium_streaming_80ms_pc",
    "nvidia/stt_en_fastconformer_hybrid_medium_streaming_80ms",
    "nvidia/stt_pt_fastconformer_hybrid_large_pc",
    "nvidia/stt_es_fastconformer_hybrid_large_pc_nc",
    "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0",
    "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi",
    # ASR: FastConformer CTC / Transducer / TDT
    "nvidia/stt_en_fastconformer_ctc_large",
    "nvidia/stt_en_fastconformer_transducer_large",
    "nvidia/stt_en_fastconformer_ctc_xlarge",
    "nvidia/stt_en_fastconformer_transducer_xlarge",
    "nvidia/stt_en_fastconformer_transducer_xxlarge",
    "nvidia/stt_en_fastconformer_ctc_xxlarge",
    "nvidia/stt_en_fastconformer_tdt_large",
    # ASR: NGC models
    "stt_en_fastconformer_hybrid_large_streaming_1040ms",
    "stt_multilingual_fastconformer_hybrid_large_pc_blend_eu",
    # ASR: Parakeet
    "nvidia/parakeet-rnnt-1.1b",
    "nvidia/parakeet-ctc-1.1b",
    "nvidia/parakeet-rnnt-0.6b",
    "nvidia/parakeet-ctc-0.6b",
    "nvidia/parakeet-tdt-1.1b",
    "nvidia/parakeet-tdt_ctc-1.1b",
    "nvidia/parakeet-tdt_ctc-0.6b-ja",
    "nvidia/parakeet-tdt_ctc-110m",
    "nvidia/parakeet-tdt-0.6b-v2",
    "nvidia/parakeet-rnnt-110m-da-dk",
    "nvidia/parakeet-tdt-0.6b-v3",
    "nvidia/parakeet-ctc-0.6b-Vietnamese",
    # ASR: Canary
    "nvidia/canary-1b",
    "nvidia/canary-1b-flash",
    "nvidia/canary-180m-flash",
    "nvidia/canary-1b-v2",
    # ASR: Specialized
    "nvidia/parakeet_realtime_eou_120m-v1",
    "nvidia/multitalker-parakeet-streaming-0.6b-v1",
    "nvidia/nemotron-speech-streaming-en-0.6b",
    # SALM
    "nvidia/canary-qwen-2.5b",
    # Diarization
    "nvidia/diar_sortformer_4spk-v1",
    "nvidia/diar_streaming_sortformer_4spk-v2",
    "nvidia/diar_streaming_sortformer_4spk-v2.1",
    # Speaker ID
    "titanet_large",
    "nvidia/speakerverification_en_titanet_large",
    # SSL
    "nvidia/ssl_en_nest_large_v1.0",
    "nvidia/ssl_en_nest_xlarge_v1.0",
    # VAD
    "vad_multilingual_marblenet",
    "vad_multilingual_frame_marblenet",
    "nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0",
    # Audio Enhancement
    "nvidia/se_den_sb_16k_small",
    "nvidia/se_der_sb_16k_small",
    "nvidia/sr_ssl_flowmatching_16k_430m",
    # Audio Codec
    "mel_codec_44khz_medium",
    "mel_codec_22khz_fullband_medium",
    "nvidia/low-frame-rate-speech-codec-22khz",
    "nvidia/audio-codec-22khz",
    "nvidia/audio-codec-44khz",
    "nvidia/mel-codec-22khz",
    "nvidia/mel-codec-44khz",
    "nvidia/nemo-nano-codec-22khz-1.78kbps-12.5fps",
    "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps",
    "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps",
    # TTS
    "nvidia/tts_en_fastpitch",
    "nvidia/tts_hifigan",
    "nvidia/magpie_tts_multilingual_357m",
    # TTS: E2E (class removed)
    "tts_en_e2e_fastspeech2hifigan",
]


def test_all_csv_models_have_test_files():
    """Verify that every supported model in model-support-table.csv has a per-model test file."""
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
            if support not in ("\u2705", "?"):
                continue
            model_name = row[4].strip()
            if model_name not in SUPPORTED_MODELS:
                missing.append(model_name)

    assert not missing, f"Models in CSV but not in SUPPORTED_MODELS: {missing}"
