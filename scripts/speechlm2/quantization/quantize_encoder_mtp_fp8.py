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

"""Write the audio encoder and the MTP draft head of an assembled checkpoint in FP8.

Inputs are a checkpoint from ``assemble_checkpoint.py`` (BF16 encoder and MTP head) and the activation
ranges from ``calibrate_activations.py``. The output is a new directory; unchanged files are hard-linked,
or copied where the output is on another filesystem.

Every quantized Linear gets an FP8 E4M3 ``weight`` with ``weight_scale = max|W| / 448`` and a static
``input_scale = range * margin / 448``, ModelOpt's per-tensor FP8 layout, which vLLM and the SpeechLM
plugin load like the decoder's FP8 layers:

- Audio encoder: ``attn.w_qkv``, ``attn.out_proj``, ``ffn.net.0`` and ``ffn.net.3`` of every
  ``perception.encoder.asr_encoder`` layer (margin 1.5). Other perception Linears stay BF16.
- MTP draft head: ``eh_proj``, the attention projections (``q/k/v_proj`` share the fused projection's
  input range), the shared experts and every routed expert, written one tensor per expert (margin 1.0).
  Norms and the router stay BF16.

An exclusion-list config (FP8) gets the BF16 perception Linears and the MTP router added to its
exclusions, replacing ``mtp*``; with ``--skip-encoder`` it excludes ``perception*``. A mixed-precision
config gets an FP8 row for every quantized module.

Example:
    python quantize_encoder_mtp_fp8.py --checkpoint /models/ckpt-nvfp4-bf16heads \\
        --activation-ranges /work/ranges.json --output /models/ckpt-nvfp4
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
from fnmatch import fnmatch
from pathlib import Path

import torch
from checkpoint_utils import INDEX_FILE, check_output_path, link_or_copy, read_header, read_json, write_json
from safetensors import safe_open
from safetensors.torch import save_file

FP8_MAX = 448.0
PERCEPTION_FILE = "perception.safetensors"
MTP_BF16_FILE = "model-mtp-bf16.safetensors"
MTP_FP8_FILE = "model-mtp-fp8.safetensors"
ENCODER_PREFIX = "perception.encoder.asr_encoder."
ENCODER_TARGETS = ("attn.w_qkv", "attn.out_proj", "ffn.net.0", "ffn.net.3")
PERCEPTION_BF16 = [
    "perception.encoder.diarization_model*",
    "perception.encoder.asr_encoder.pre_encode*",
    "perception.proj",
]
MTP_EXCLUDE = "mtp*"
MTP_LINEAR = re.compile(
    r"^llm\.mtp\.layers\.(\d+)\.(eh_proj|mixer\.[qkvo]_proj|mixer\.shared_experts\.(?:up|down)_proj)\.weight$"
)
MTP_EXPERTS = re.compile(r"^llm\.mtp\.layers\.(\d+)\.mixer\.experts\.(gate_and_up_projs|down_projs)$")
MTP_GATE = re.compile(r"^llm\.mtp\.layers\.(\d+)\.mixer\.gate\.weight$")


def to_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight to FP8 E4M3 with one scale for the whole tensor."""
    weight = weight.float()
    scale = (weight.abs().max() / FP8_MAX).clamp_min(1e-12)
    return (weight / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).contiguous(), scale.reshape(())


def input_scale(ranges: dict[str, float], key: str, margin: float) -> torch.Tensor:
    """Return the static input scale for the module whose calibrated activation range is ``ranges[key]``."""
    if key not in ranges:
        raise SystemExit(f"no calibrated activation range for {key}")
    return torch.tensor(max(ranges[key] * margin / FP8_MAX, 1e-12), dtype=torch.float32)


def quantize_encoder(path: Path, ranges: dict[str, float], margin: float) -> tuple[dict, dict, list[str]]:
    """Return the new perception tensors, the file metadata and the quantized encoder modules."""
    with safe_open(path, "pt") as f:
        metadata = f.metadata()
        targets = sorted(
            key[: -len(".weight")]
            for key in f.keys()
            if key.startswith(ENCODER_PREFIX + "layers.")
            and key.endswith(".weight")
            and key[: -len(".weight")].endswith(tuple("." + target for target in ENCODER_TARGETS))
        )
        calibrated = {ENCODER_PREFIX + name for name in ranges}
        if set(targets) != calibrated:
            raise SystemExit(f"{len(set(targets) ^ calibrated)} encoder Linears differ from the calibrated ones")
        tensors = {}
        for key in f.keys():
            module, _, leaf = key.rpartition(".")
            if module in calibrated and leaf in ("weight_scale", "input_scale"):
                continue
            tensor = f.get_tensor(key)
            if module in calibrated and leaf == "weight":
                if tensor.dtype == torch.float8_e4m3fn:
                    raise SystemExit(f"{key} is already FP8")
                tensors[key], tensors[module + ".weight_scale"] = to_fp8(tensor)
                tensors[module + ".input_scale"] = input_scale(ranges, module[len(ENCODER_PREFIX) :], margin)
            else:
                tensors[key] = tensor
    return tensors, metadata, targets


def quantize_mtp(path: Path, ranges: dict[str, float], margin: float) -> tuple[dict, list[str], list[str]]:
    """Return the FP8 MTP head tensors, its quantized modules (vLLM names) and its router gates."""
    tensors, modules, gates = {}, set(), []
    with safe_open(path, "pt") as f:
        for name in f.keys():
            if name.endswith("._extra_state"):  # zero-length Transformer Engine bookkeeping
                continue
            tensor = f.get_tensor(name)
            if match := MTP_LINEAR.match(name):
                layer, module = match.groups()
                base = name[: -len(".weight")]
                tensors[f"{base}.weight"], tensors[f"{base}.weight_scale"] = to_fp8(tensor)
                key = f"model.layers.{layer}." + re.sub(r"[qkv]_proj$", "qkv_proj", module)
                tensors[f"{base}.input_scale"] = input_scale(ranges, key, margin)
                modules.add(f"mtp.layers.{layer}.{module}")
            elif match := MTP_EXPERTS.match(name):
                layer, kind = match.groups()
                proj, key = ("up_proj", "w13_input") if kind == "gate_and_up_projs" else ("down_proj", "w2_input")
                scale = input_scale(ranges, f"model.layers.{layer}.mixer.experts.{key}", margin)
                for expert, weight in enumerate(tensor):
                    base = f"llm.mtp.layers.{layer}.mixer.experts.{expert}.{proj}"
                    tensors[f"{base}.weight"], tensors[f"{base}.weight_scale"] = to_fp8(weight.t())
                    tensors[f"{base}.input_scale"] = scale.clone()
                modules.add(f"mtp.layers.{layer}.mixer.experts")
            else:
                if match := MTP_GATE.match(name):
                    gates.append(f"mtp.layers.{match.group(1)}.mixer.gate")
                tensors[name] = tensor
    return tensors, sorted(modules), gates


def update_config(
    section: dict, exclusion_key: str, fp8_modules: list[str], add: list[str], remove: tuple[str, ...]
) -> None:
    """Describe the new FP8 modules in one quantization config section.

    A mixed-precision section lists them with ``quant_algo: FP8``. An exclusion-list section quantizes
    every module it does not exclude, so it drops the entries in ``remove`` and gains those in ``add``.
    """
    if section.get("quant_algo") == "MIXED_PRECISION":
        rows = section.setdefault("quantized_layers", {})
        clash = sorted(set(fp8_modules) & set(rows))
        if clash:
            raise SystemExit(f"quantized_layers already lists {clash[:3]}")
        rows.update({module: {"quant_algo": "FP8"} for module in fp8_modules})
    else:
        excluded = [entry for entry in section.get(exclusion_key) or [] if entry not in remove]
        section[exclusion_key] = excluded + [entry for entry in add if entry not in excluded]


def link_tree(source: Path, destination: Path, skip: set[str]) -> None:
    """Hard-link (or copy) the files of ``source`` into ``destination``, except the top-level names in ``skip``."""
    destination.mkdir(parents=True)
    for path in sorted(source.iterdir()):
        if path.name in skip:
            continue
        if path.is_dir():
            shutil.copytree(path, destination / path.name, copy_function=link_or_copy)
        else:
            link_or_copy(path, destination / path.name)


def check_output(weights: dict[str, dict], config: dict) -> None:
    """Fail unless every FP8 weight has scalar float32 scales and the config marks each perception Linear right.

    ``weights`` maps every tensor of the output to its safetensors header entry. A perception Linear is
    marked FP8 as the SpeechLM plugin decides: an exclusion wins; an exclusion-list config without any
    perception entry keeps the whole encoder BF16; a mixed-precision config quantizes the modules it lists.
    """
    for name, info in weights.items():
        if name.endswith(".weight") and info["dtype"] == "F8_E4M3":
            for leaf in ("weight_scale", "input_scale"):
                scale = weights.get(f"{name[: -len('.weight')]}.{leaf}", {})
                if scale.get("dtype") != "F32" or math.prod(scale.get("shape", [0])) != 1:
                    raise SystemExit(f"{name[: -len('.weight')]}.{leaf} is missing or not a float32 scalar")

    quant = config["quantization_config"]
    mixed = quant.get("quant_algo") == "MIXED_PRECISION"
    ignore = quant.get("ignore") or []
    encoder_unquantized = not mixed and not any(str(entry).startswith("perception") for entry in ignore)
    for name, info in weights.items():
        if not name.startswith("perception.") or not name.endswith(".weight") or len(info["shape"]) != 2:
            continue
        module = name[: -len(".weight")]
        if encoder_unquantized or any(module == e or e in module or fnmatch(module, e) for e in ignore):
            marked_fp8 = False
        elif mixed:
            marked_fp8 = quant.get("quantized_layers", {}).get(module, {}).get("quant_algo") == "FP8"
        else:
            marked_fp8 = True
        if (info["dtype"] == "F8_E4M3") != marked_fp8:
            raise SystemExit(f"{module} is {info['dtype']} but the config marks it {'FP8' if marked_fp8 else 'BF16'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="checkpoint from assemble_checkpoint.py")
    parser.add_argument("--activation-ranges", type=Path, required=True, help="JSON from calibrate_activations.py")
    parser.add_argument("--output", type=Path, required=True, help="new checkpoint directory")
    parser.add_argument("--encoder-margin", type=float, default=1.5)
    parser.add_argument("--mtp-margin", type=float, default=1.0)
    parser.add_argument("--skip-encoder", action="store_true", help="keep the audio encoder BF16")
    parser.add_argument("--skip-mtp", action="store_true", help="keep the MTP draft head BF16")
    args = parser.parse_args()

    checkpoint, output = args.checkpoint.resolve(), args.output.resolve()
    check_output_path(output, checkpoint)
    if output.exists():
        raise FileExistsError(f"{output} exists")
    ranges = read_json(args.activation_ranges)
    config = read_json(checkpoint / "config.json")
    hf_quant = read_json(checkpoint / "hf_quant_config.json")
    do_encoder, do_mtp = not args.skip_encoder, not args.skip_mtp and (checkpoint / MTP_BF16_FILE).exists()

    new_files: dict[str, tuple[dict, dict]] = {}
    fp8_modules, add, remove = [], [], ()
    if do_encoder:
        tensors, metadata, modules = quantize_encoder(
            checkpoint / PERCEPTION_FILE, ranges["encoder"], args.encoder_margin
        )
        new_files[PERCEPTION_FILE] = (tensors, metadata)
        fp8_modules += modules
        add += PERCEPTION_BF16
        print(f"encoder: {len(modules)} Linears to FP8")
    else:
        add += ["perception*"]
    if do_mtp:
        tensors, modules, gates = quantize_mtp(checkpoint / MTP_BF16_FILE, ranges["mtp"], args.mtp_margin)
        new_files[MTP_FP8_FILE] = (tensors, {"format": "pt"})
        fp8_modules += modules
        add += gates
        remove = (MTP_EXCLUDE,)
        print(f"MTP head: {len(modules)} modules to FP8, router kept BF16 ({gates})")
    if not new_files:
        raise SystemExit("nothing to quantize")

    replaced = {INDEX_FILE, "config.json", "hf_quant_config.json", *new_files}
    link_tree(checkpoint, output, replaced | ({MTP_BF16_FILE} if do_mtp else set()))
    for name, (tensors, metadata) in new_files.items():
        save_file(tensors, output / name, metadata=metadata)

    update_config(config["quantization_config"], "ignore", fp8_modules, add, remove)
    update_config(hf_quant["quantization"], "exclude_modules", fp8_modules, add, remove)
    write_json(output / "config.json", config)
    write_json(output / "hf_quant_config.json", hf_quant)

    weights, weight_map = {}, {}
    for path in sorted(output.glob("*.safetensors")):
        header = read_header(path)[0]
        header.pop("__metadata__", None)
        for name, info in header.items():
            if name in weights:
                raise SystemExit(f"{name} is stored in both {weight_map[name]} and {path.name}")
            weights[name], weight_map[name] = info, path.name
    total_size = sum(info["data_offsets"][1] - info["data_offsets"][0] for info in weights.values())
    write_json(output / INDEX_FILE, {"metadata": {"total_size": total_size}, "weight_map": weight_map})
    check_output(weights, config)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
