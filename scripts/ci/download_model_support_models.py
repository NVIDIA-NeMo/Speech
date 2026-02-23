#!/usr/bin/env python3
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
Download all supported NeMo models to a local cache directory.

Usage:
    python scripts/ci/download_model_support_models.py \
        --output-dir nemo-speech-ci-models

    # Download a single model:
    python scripts/ci/download_model_support_models.py \
        --output-dir nemo-speech-ci-models \
        --model "nvidia/parakeet-ctc-0.6b"
"""

import argparse
import gc
import os
import shutil
import sys

import torch

# ---------------------------------------------------------------------------
# Registry: model_name -> {category, loader (class name string), file}
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    # -- ASR: FastConformer hybrid -------------------------------------------
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
    # -- ASR: FastConformer CTC / Transducer / TDT --------------------------
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
    # -- ASR: NGC models ----------------------------------------------------
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
    # -- ASR: Parakeet ------------------------------------------------------
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
    # -- ASR: Canary --------------------------------------------------------
    "nvidia/canary-1b": {"category": "asr", "loader": "ASRModel", "file": "nvidia__canary-1b.nemo"},
    "nvidia/canary-1b-flash": {"category": "asr", "loader": "ASRModel", "file": "nvidia__canary-1b-flash.nemo"},
    "nvidia/canary-180m-flash": {"category": "asr", "loader": "ASRModel", "file": "nvidia__canary-180m-flash.nemo"},
    "nvidia/canary-1b-v2": {"category": "asr", "loader": "ASRModel", "file": "nvidia__canary-1b-v2.nemo"},
    # -- ASR: Specialized ---------------------------------------------------
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
    # -- SALM ---------------------------------------------------------------
    "nvidia/canary-qwen-2.5b": {"category": "salm", "loader": "SALM", "file": "nvidia__canary-qwen-2.5b"},
    # -- Diarization --------------------------------------------------------
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
    # -- Speaker ID ---------------------------------------------------------
    "titanet_large": {"category": "speaker", "loader": "EncDecSpeakerLabelModel", "file": "titanet_large.nemo"},
    "nvidia/speakerverification_en_titanet_large": {
        "category": "speaker",
        "loader": "EncDecSpeakerLabelModel",
        "file": "nvidia__speakerverification_en_titanet_large.nemo",
    },
    # -- SSL ----------------------------------------------------------------
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
    # -- VAD ----------------------------------------------------------------
    "vad_multilingual_marblenet": {
        "category": "vad",
        "loader": "EncDecClassificationModel",
        "file": "vad_multilingual_marblenet.nemo",
    },
    "vad_multilingual_frame_marblenet": {
        "category": "vad",
        "loader": "EncDecFrameClassificationModel",
        "file": "vad_multilingual_frame_marblenet.nemo",
    },
    "nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0": {
        "category": "vad",
        "loader": "EncDecClassificationModel",
        "file": "nvidia__Frame_VAD_Multilingual_MarbleNet_v2.0.nemo",
    },
    # -- Audio Enhancement --------------------------------------------------
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
    # -- Audio Codec --------------------------------------------------------
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
    # -- TTS ----------------------------------------------------------------
    "nvidia/tts_en_fastpitch": {
        "category": "tts_fastpitch",
        "loader": "FastPitchModel",
        "file": "nvidia__tts_en_fastpitch.nemo",
    },
    "nvidia/tts_hifigan": {"category": "tts_hifigan", "loader": "HifiGanModel", "file": "nvidia__tts_hifigan.nemo"},
    "nvidia/magpie_tts_multilingual_357m": {
        "category": "tts_magpie",
        "loader": "MagpieTTSModel",
        "file": "nvidia__magpie_tts_multilingual_357m.nemo",
    },
    # -- TTS: E2E (class removed from codebase) -----------------------------
    "tts_en_e2e_fastspeech2hifigan": {
        "category": "tts_e2e",
        "loader": None,
        "file": "tts_en_e2e_fastspeech2hifigan.nemo",
    },
}

# Some HF models use a filename that differs from "{model_basename}.nemo".
# Map model_name -> actual filename on HuggingFace Hub.
HF_FILENAME_OVERRIDES = {
    "nvidia/parakeet-ctc-0.6b-Vietnamese": "parakeet-ctc-0.6b-vi.nemo",
    "nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0": "frame_vad_multilingual_marblenet_v2.0.nemo",
}

# NGC models absent from list_available_models(): map name -> direct download URL.
NGC_DIRECT_URLS = {
    "vad_multilingual_frame_marblenet": (
        "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/vad_multilingual_frame_marblenet"
        "/versions/1.20.0/files/vad_multilingual_frame_marblenet.nemo"
    ),
}


def _get_loader_class(loader_name: str):
    """Import and return the loader class by name."""
    if loader_name == "ASRModel":
        from nemo.collections.asr.models import ASRModel

        return ASRModel
    elif loader_name == "EncDecSpeakerLabelModel":
        from nemo.collections.asr.models import EncDecSpeakerLabelModel

        return EncDecSpeakerLabelModel
    elif loader_name == "EncDecClassificationModel":
        from nemo.collections.asr.models import EncDecClassificationModel

        return EncDecClassificationModel
    elif loader_name == "EncDecFrameClassificationModel":
        from nemo.collections.asr.models.classification_models import EncDecFrameClassificationModel

        return EncDecFrameClassificationModel
    elif loader_name == "SortformerEncLabelModel":
        from nemo.collections.asr.models import SortformerEncLabelModel

        return SortformerEncLabelModel
    elif loader_name == "AudioToAudioModel":
        from nemo.collections.audio.models import AudioToAudioModel

        return AudioToAudioModel
    elif loader_name == "AudioCodecModel":
        from nemo.collections.tts.models import AudioCodecModel

        return AudioCodecModel
    elif loader_name == "FastPitchModel":
        from nemo.collections.tts.models import FastPitchModel

        return FastPitchModel
    elif loader_name == "HifiGanModel":
        from nemo.collections.tts.models import HifiGanModel

        return HifiGanModel
    elif loader_name == "MagpieTTSModel":
        from nemo.collections.tts.models import MagpieTTSModel

        return MagpieTTSModel
    elif loader_name == "SALM":
        from nemo.collections.speechlm2.models import SALM

        return SALM
    elif loader_name == "EncDecDenoiseMaskedTokenPredModel":
        from nemo.collections.asr.models import EncDecDenoiseMaskedTokenPredModel

        return EncDecDenoiseMaskedTokenPredModel
    else:
        raise ValueError(f"Unknown loader class: {loader_name}")


def download_model(model_name: str, output_dir: str) -> None:
    """Download a single model and save it to output_dir."""
    info = MODEL_REGISTRY[model_name]
    loader_name = info["loader"]

    if loader_name is None:
        print(f"  SKIP {model_name} (model class removed from codebase)")
        return

    output_path = os.path.join(output_dir, info["file"])

    # Skip if already downloaded
    if os.path.exists(output_path):
        print(f"  EXISTS {output_path}")
        return

    print(f"  Downloading {model_name} ...")
    loader_cls = _get_loader_class(loader_name)

    if info["category"] == "salm":
        # SALM uses HFHubMixin: from_pretrained / save_pretrained
        model = loader_cls.from_pretrained(model_name, map_location="cpu")
        model.save_pretrained(output_path)
    elif model_name in HF_FILENAME_OVERRIDES:
        # HF model whose .nemo filename differs from "{model_basename}.nemo"
        from huggingface_hub import hf_hub_download

        hf_path = hf_hub_download(
            repo_id=model_name,
            filename=HF_FILENAME_OVERRIDES[model_name],
            library_name="nemo",
        )
        shutil.copy2(hf_path, output_path)
        print(f"  (copied from HF cache: {hf_path})")
        return
    elif model_name in NGC_DIRECT_URLS:
        # NGC model not registered in list_available_models()
        import requests

        url = NGC_DIRECT_URLS[model_name]
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
        print(f"  (downloaded directly from NGC)")
        return
    else:
        # Standard ModelPT: from_pretrained / save_to
        model = loader_cls.from_pretrained(model_name, map_location="cpu")
        model.save_to(output_path)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  SAVED {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download all supported NeMo models")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save downloaded models",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Download only a specific model (by name from MODEL_REGISTRY)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.model:
        if args.model not in MODEL_REGISTRY:
            print(f"ERROR: Unknown model '{args.model}'")
            print(f"Available models: {sorted(MODEL_REGISTRY.keys())}")
            sys.exit(1)
        download_model(args.model, args.output_dir)
    else:
        total = len(MODEL_REGISTRY)
        for i, model_name in enumerate(MODEL_REGISTRY, 1):
            print(f"[{i}/{total}] {model_name}")
            try:
                download_model(model_name, args.output_dir)
            except Exception as e:
                print(f"  FAILED: {e}")
                continue


if __name__ == "__main__":
    main()
