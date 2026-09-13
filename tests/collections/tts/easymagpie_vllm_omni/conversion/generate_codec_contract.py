#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Generate a tiny Speech-to-vLLM codec parity contract for the serving CI job."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from hydra.utils import instantiate
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[5]
EASYMAGPIE_ROOT = ROOT / "tools" / "easymagpie_vllm_omni"
sys.path.insert(0, str(EASYMAGPIE_ROOT))
sys.path.insert(0, str(EASYMAGPIE_ROOT / "scripts"))

import convert_codec as converter  # noqa: E402
from easymagpie_vllm_omni.codec.config import EasyMagpieCodecConfig  # noqa: E402
from easymagpie_vllm_omni.codec.weight_conversion import convert_decoder_state_dict  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def speech_decoder_config() -> dict:
    return {
        "_target_": "nemo.collections.tts.modules.audio_codec_modules.ResNetDecoder",
        "input_dim": 4,
        "input_filters": 8,
        "pre_up_sample_rates": [2],
        "pre_up_sample_filters": [8],
        "n_hidden_layers": 2,
        "hidden_filters": 16,
        "resblock_up_sample_rates": [2],
        "resblock_up_sample_filters": [4],
        "resblock_up_sample_kernel_size": 7,
        "kernel_size": 3,
        "activation": "half_snake",
        "is_causal": True,
        "pad_mode": "replicate",
    }


def serving_config() -> EasyMagpieCodecConfig:
    return EasyMagpieCodecConfig(
        input_dim=4,
        input_filters=8,
        hidden_filters=16,
        num_hidden_layers=2,
        pre_upsample_rates=[2],
        pre_upsample_filters=[8],
        resblock_upsample_rates=[2],
        resblock_upsample_filters=[4],
        kernel_size=3,
        resblock_kernel_size=7,
        activation="half_snake",
        num_codebooks=2,
        codebook_size=4,
        num_levels_per_group=[2, 2],
        frame_stacking_factor=2,
        output_sample_rate=22050,
    )


def generate_contract(output: Path) -> None:
    torch.manual_seed(20260730)
    decoder_config = speech_decoder_config()
    source_decoder = instantiate(decoder_config)
    source_state = {
        f"audio_decoder.{name}": tensor.detach().cpu() for name, tensor in source_decoder.state_dict().items()
    }
    decoder = converter.restore_speech_decoder(decoder_config, source_state)

    num_frames = 7
    latent = torch.linspace(-1.0, 1.0, steps=num_frames * decoder_config["input_dim"], dtype=torch.float32)
    latent = latent.view(num_frames, decoder_config["input_dim"])
    input_len = torch.tensor([num_frames], dtype=torch.long)
    with torch.no_grad():
        audio, audio_len = decoder(inputs=latent.T.unsqueeze(0).contiguous(), input_len=input_len)

    config = serving_config()
    expected_audio_len = num_frames * config.samples_per_codec_frame
    if int(audio_len.item()) != expected_audio_len:
        raise AssertionError(f"Speech decoder returned {int(audio_len.item())} samples, expected {expected_audio_len}")

    output.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(output)
    save_file(convert_decoder_state_dict(source_state), output / "model.safetensors")
    save_file(
        {
            "latent": latent.contiguous(),
            "expected_audio": audio.squeeze(0).contiguous(),
            "expected_audio_len": audio_len.cpu().contiguous(),
        },
        output / "contract.safetensors",
    )


if __name__ == "__main__":
    generate_contract(parse_args().output)
