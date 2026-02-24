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
import json
import os
import shutil
import sys

import torch

# ---------------------------------------------------------------------------
# Registry: loaded from model_registry.json next to this script.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REGISTRY_PATH = os.path.join(_SCRIPT_DIR, "model_registry.json")

with open(_REGISTRY_PATH) as _f:
    _registry_list = json.load(_f)

MODEL_REGISTRY = {entry["name"]: entry for entry in _registry_list}


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
    elif info.get("source") == "hf_hub":
        # HF model requiring explicit hf_hub_download (non-standard filename)
        from huggingface_hub import hf_hub_download

        hf_path = hf_hub_download(
            repo_id=model_name,
            filename=info["file"],
            library_name="nemo",
        )
        shutil.copy2(hf_path, output_path)
        print(f"  (copied from HF cache: {hf_path})")
        return
    elif info.get("ngc_url"):
        # NGC model not registered in list_available_models()
        import requests

        url = info["ngc_url"]
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
