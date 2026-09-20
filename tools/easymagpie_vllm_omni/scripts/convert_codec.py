#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Convert an EasyMagpie spectral codec ``.nemo`` to a native vLLM bundle."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path

import torch
import yaml
from easymagpie_vllm_omni.codec.config import EasyMagpieCodecConfig
from easymagpie_vllm_omni.codec.encoder_config import EasyMagpieCodecEncoderConfig
from easymagpie_vllm_omni.codec.weight_conversion import convert_decoder_state_dict, convert_encoder_state_dict
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("codec", type=Path, help="Input 25-fps spectral codec .nemo")
    parser.add_argument("output", type=Path, help="Output Hugging Face/vLLM model directory")
    parser.add_argument(
        "--encoder-output",
        type=Path,
        default=None,
        help="Optional separate .safetensors checkpoint for the codec encoder; its config is saved beside it",
    )
    parser.add_argument("--num-codebooks", type=int, default=8, help="EasyMagpie-side FSQ group count")
    parser.add_argument("--frame-stacking-factor", type=int, default=2)
    parser.add_argument(
        "--num-levels-per-group",
        type=int,
        nargs="+",
        default=[4, 4, 4, 4, 4],
        help="EasyMagpie-side FSQ levels; their product is the codebook size",
    )
    return parser.parse_args()


def _read_nemo(codec_path: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    with tarfile.open(codec_path) as archive:
        config_member = archive.getmember("./model_config.yaml")
        weights_member = archive.getmember("./model_weights.ckpt")
        config_file = archive.extractfile(config_member)
        weights_file = archive.extractfile(weights_member)
        if config_file is None or weights_file is None:
            raise RuntimeError(f"invalid NeMo archive: {codec_path}")
        config = yaml.safe_load(config_file)
        with tempfile.NamedTemporaryFile(suffix=".ckpt") as temporary:
            shutil.copyfileobj(weights_file, temporary)
            temporary.flush()
            state = torch.load(temporary.name, map_location="cpu", weights_only=True)
    return config, state


def validate_decoder_config(decoder_config: dict) -> None:
    """Validate NeMo decoder features implemented by the native vLLM codec."""
    expected_target = "nemo.collections.tts.modules.audio_codec_modules.ResNetDecoder"
    if decoder_config.get("_target_") != expected_target:
        raise ValueError(
            f"expected {expected_target}, got {decoder_config.get('_target_')}; "
            "add a matching native decoder before converting this codec"
        )
    if decoder_config.get("is_causal") is not True:
        raise ValueError(
            "only the causal spectral codec decoder is supported; set is_causal=true or add state handling"
        )
    activation = str(decoder_config.get("activation", "half_snake"))
    if activation != "half_snake":
        raise ValueError(
            "the native codec currently requires activation='half_snake'; implement the matching activation in "
            f"packed.py to support '{activation}'"
        )


def validate_encoder_config(encoder_config: dict, quantizer_config: dict) -> None:
    """Validate encoder and FSQ features implemented by the standalone Torch module."""
    expected_encoder = "nemo.collections.tts.modules.audio_codec_modules.MultiResolutionSTFTEncoder"
    if encoder_config.get("_target_") != expected_encoder:
        raise ValueError(
            f"expected {expected_encoder}, got {encoder_config.get('_target_')}; "
            "add a matching standalone encoder before converting this codec"
        )
    if encoder_config.get("activation", "lrelu") != "lrelu":
        raise ValueError("the standalone codec encoder currently requires activation='lrelu'")
    if encoder_config.get("pad_mode", "replicate") != "replicate":
        raise ValueError("the standalone codec encoder currently requires pad_mode='replicate'")

    expected_quantizer = "nemo.collections.tts.modules.audio_codec_modules.GroupFiniteScalarQuantizer"
    if quantizer_config.get("_target_") != expected_quantizer:
        raise ValueError(
            f"expected {expected_quantizer}, got {quantizer_config.get('_target_')}; "
            "only grouped finite scalar quantization can be repacked"
        )


def build_encoder_config(nemo_config: dict, args: argparse.Namespace) -> EasyMagpieCodecEncoderConfig:
    encoder_config = nemo_config["audio_encoder"]
    quantizer_config = nemo_config["vector_quantizer"]
    validate_encoder_config(encoder_config, quantizer_config)
    downsample_filters = list(encoder_config.get("down_sample_filter_list", []))
    downsample_rates = list(encoder_config.get("down_sample_rate_list", [2] * len(downsample_filters)))
    return EasyMagpieCodecEncoderConfig(
        sample_rate=int(nemo_config["sample_rate"]),
        samples_per_frame=int(nemo_config["samples_per_frame"]),
        output_dim=int(encoder_config["out_dim"]),
        resolutions=[list(value) for value in encoder_config["resolutions"]],
        resolution_filters=list(encoder_config["resolution_filter_list"]),
        downsample_filters=downsample_filters,
        downsample_rates=downsample_rates,
        kernel_size=int(encoder_config.get("kernel_size", 3)),
        activation=str(encoder_config.get("activation", "lrelu")),
        pad_mode=str(encoder_config.get("pad_mode", "replicate")),
        original_num_codebooks=int(quantizer_config["num_groups"]),
        original_num_levels_per_group=list(quantizer_config["num_levels_per_group"]),
        num_codebooks=args.num_codebooks,
        num_levels_per_group=list(args.num_levels_per_group),
        frame_stacking_factor=args.frame_stacking_factor,
    )


def restore_speech_encoder(encoder_config: dict, state: dict[str, torch.Tensor]) -> torch.nn.Module:
    """Instantiate and strictly restore the canonical Speech encoder."""
    from hydra.utils import instantiate

    prefix = "audio_encoder."
    encoder_state = {name.removeprefix(prefix): tensor for name, tensor in state.items() if name.startswith(prefix)}
    encoder = instantiate(encoder_config)
    encoder.load_state_dict(encoder_state, strict=True)
    return encoder.eval()


def convert_encoder_checkpoint(
    nemo_config: dict,
    state: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[EasyMagpieCodecEncoderConfig, int]:
    """Validate and save the optional standalone encoder checkpoint."""
    config = build_encoder_config(nemo_config, args)
    encoder = restore_speech_encoder(nemo_config["audio_encoder"], state)
    converted = convert_encoder_state_dict(state)
    parameter_count = sum(parameter.numel() for parameter in converted.values())
    del encoder

    if args.encoder_output.suffix != ".safetensors":
        raise ValueError("--encoder-output must end in .safetensors")
    args.encoder_output.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, args.encoder_output)
    config.to_json_file(args.encoder_output.with_suffix(".json"), use_diff=False)
    return config, parameter_count


def restore_speech_decoder(decoder_config: dict, state: dict[str, torch.Tensor]) -> torch.nn.Module:
    """Instantiate and strictly restore the canonical Speech decoder."""
    from hydra.utils import instantiate

    prefix = "audio_decoder."
    decoder_state = {name.removeprefix(prefix): tensor for name, tensor in state.items() if name.startswith(prefix)}
    decoder = instantiate(decoder_config)
    decoder.load_state_dict(decoder_state, strict=True)
    return decoder.eval()


def main() -> None:
    args = parse_args()
    nemo_config, state = _read_nemo(args.codec)
    decoder_config = nemo_config["audio_decoder"]
    encoder_result = convert_encoder_checkpoint(nemo_config, state, args) if args.encoder_output is not None else None
    validate_decoder_config(decoder_config)

    levels = list(args.num_levels_per_group)
    codebook_size = 1
    for level in levels:
        codebook_size *= level
    config = EasyMagpieCodecConfig(
        input_dim=int(decoder_config["input_dim"]),
        input_filters=int(decoder_config["input_filters"]),
        hidden_filters=int(decoder_config["hidden_filters"]),
        num_hidden_layers=int(decoder_config["n_hidden_layers"]),
        pre_upsample_rates=list(decoder_config["pre_up_sample_rates"]),
        pre_upsample_filters=list(decoder_config["pre_up_sample_filters"]),
        resblock_upsample_rates=list(decoder_config["resblock_up_sample_rates"]),
        resblock_upsample_filters=list(decoder_config["resblock_up_sample_filters"]),
        kernel_size=int(decoder_config.get("kernel_size", 3)),
        resblock_kernel_size=int(decoder_config.get("resblock_up_sample_kernel_size", 7)),
        activation=str(decoder_config.get("activation", "half_snake")),
        num_codebooks=args.num_codebooks,
        codebook_size=codebook_size,
        num_levels_per_group=levels,
        frame_stacking_factor=args.frame_stacking_factor,
        output_sample_rate=int(nemo_config.get("output_sample_rate", nemo_config["sample_rate"])),
    )

    decoder = restore_speech_decoder(decoder_config, state)
    converted = convert_decoder_state_dict(state)
    parameter_count = sum(parameter.numel() for parameter in converted.values())
    del decoder

    args.output.mkdir(parents=True, exist_ok=True)
    save_file(converted, args.output / "model.safetensors")
    config.save_pretrained(args.output)
    print(
        f"Converted {len(converted)} tensors ({parameter_count:,} parameters) to {args.output}; "
        f"{config.num_stacked_codebooks} stacked codebooks -> {config.samples_per_frame} samples/frame"
    )
    if encoder_result is not None:
        encoder_config, encoder_parameter_count = encoder_result
        print(
            f"Converted codec encoder ({encoder_parameter_count:,} parameters) to {args.encoder_output}; "
            f"16 kHz audio -> {encoder_config.num_stacked_codebooks} stacked codebooks at "
            f"{encoder_config.sample_rate / encoder_config.samples_per_frame / encoder_config.frame_stacking_factor:g} fps"
        )


if __name__ == "__main__":
    main()
