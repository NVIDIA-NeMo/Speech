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

"""Assemble a SpeechLM checkpoint that vLLM serves from a ModelOpt-quantized decoder.

Inputs are the BF16 SpeechLM checkpoint (``--source``) and the LLM-only checkpoint from
``quantize_decoder.py`` (``--quantized-llm``). The output holds:

    model-0000N-of-0000M.safetensors   the quantized decoder, as ModelOpt wrote it
    perception.safetensors             the audio encoder and projection, copied from the source
    model-mtp-bf16.safetensors         the MTP draft head (``llm.mtp.*``), copied from the source
    model-ctc-bf16.safetensors         the CTC timestamp head (``ctc_timestamp.*``) and its file metadata
    config.json                        the source config with ModelOpt's ``quantization_config``
    hf_quant_config.json               ModelOpt's quantization config

Copied tensors are bit-identical to the source, checked by SHA-256. The heads stay BF16: an exclusion
list (FP8) gets ``mtp*`` and ``ctc_timestamp*`` entries, and a mixed-precision config does not list them.
Serving the CTC head needs the plugin's CTC timestamp support; ``--skip-ctc-head`` leaves it out.
Tokenizer files, docs and examples come from the source; its README.md and SHA256SUMS are left out
because they describe the BF16 release.

Example:
    python assemble_checkpoint.py --source /models/speechlm-bf16 --quantized-llm /work/llm_nvfp4 \\
        --output /work/speechlm-nvfp4-bf16-heads
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import replace
from pathlib import Path

from checkpoint_utils import (
    INDEX_FILE,
    check_output_path,
    copy_tensors,
    file_metadata,
    find_tensors,
    find_tensors_in_file,
    link_or_copy,
    read_header,
    read_json,
    read_weight_map,
    write_json,
)

# (tensor prefixes, file, exclusion entry, metadata prefix) for each module copied from the source
COPIED_MODULES = (
    (("perception.", "projection."), "perception.safetensors", None, None),
    (("llm.mtp.",), "model-mtp-bf16.safetensors", "mtp*", None),
    (("ctc_timestamp.",), "model-ctc-bf16.safetensors", "ctc_timestamp*", "ctc_timestamp_"),
)
SKIPPED_SOURCE_FILES = {"config.json", "README.md", "SHA256SUMS", INDEX_FILE}
# Mamba parameters that the source keeps in FP32 and ModelOpt's BF16 export rounds.
FP32_PARAMETER = re.compile(r"^backbone\.layers\.\d+\.mixer\.(A_log|D|dt_bias)$")
# Placeholder that ModelOpt builds without NVIDIA/Model-Optimizer#2508 leave in module names they failed
# to resolve.
NAME_SENTINEL = "\x00"


def quantization_names(config: dict) -> list[str]:
    """Return every module name in a ModelOpt quantization config: exclusions, per-layer keys, group targets."""
    names = list(config.get("ignore") or []) + list(config.get("exclude_modules") or [])
    names += list(config.get("quantized_layers") or {})
    for group in (config.get("config_groups") or {}).values():
        names += list(group.get("targets") or [])
    return names


def check_names(config: dict, file_name: str) -> None:
    """Refuse a ModelOpt export whose module names carry the unresolved-name sentinel."""
    bad = [name for name in quantization_names(config) if NAME_SENTINEL in name]
    if bad:
        raise SystemExit(
            f"{file_name} has {len(bad)} module names that ModelOpt failed to resolve, e.g. {bad[0]!r}. "
            "Re-export with a ModelOpt build that includes NVIDIA/Model-Optimizer#2508."
        )


def speechlm_quantization_config(llm_config: dict, exclusions: list[str]) -> dict:
    """Return the LLM export's ``quantization_config`` in the namespace vLLM uses for the SpeechLM model.

    The plugin serves the decoder as ``language_model``, and vLLM's NemotronH renames ``backbone`` to
    ``model``, so per-layer keys must read ``language_model.model.*`` for vLLM to match them.
    """
    config = dict(llm_config["quantization_config"])
    if isinstance(config.get("quantized_layers"), dict):
        config["quantized_layers"] = {
            (key.replace("backbone.", "language_model.model.", 1) if key.startswith("backbone.") else key): value
            for key, value in config["quantized_layers"].items()
        }
    else:
        ignore = list(config.get("ignore") or [])
        config["ignore"] = ignore + [entry for entry in exclusions if entry not in ignore]
    return config


def restore_fp32_parameters(source: Path, output: Path, decoder_files: list[str]) -> int:
    """Put the source's FP32 Mamba parameters back into the decoder shards, and return how many there were.

    ModelOpt quantizes a decoder loaded in BF16, so it exports ``A_log``, ``D`` and ``dt_bias`` rounded to
    BF16 although they are never quantized. A shard holding any of them is rewritten as a new file, so a
    hard-linked ModelOpt export stays unchanged.
    """
    originals = {}
    for entry in find_tensors(source, ("llm.model.",)):
        name = entry.name.replace("llm.model.", "backbone.", 1).replace("._fp32_params.", ".")
        if FP32_PARAMETER.match(name) and entry.dtype == "F32":
            originals[name] = replace(entry, name=name)
    restored = 0
    for file_name in decoder_files:
        path = output / file_name
        entries = find_tensors_in_file(path)
        updated = [
            originals[e.name] if e.name in originals and originals[e.name].shape == e.shape else e for e in entries
        ]
        changed = sum(new is not old for new, old in zip(updated, entries))
        if not changed:
            continue
        metadata = read_header(path)[0].get("__metadata__") or {}
        copy_tensors(updated, path.with_name(path.name + ".tmp"), metadata)
        os.replace(path.with_name(path.name + ".tmp"), path)
        restored += changed
    return restored


def copy_source_files(source: Path, output: Path) -> None:
    """Copy the source's non-weight files and folders, except the ones this script rewrites or drops."""
    for path in sorted(source.iterdir()):
        if path.name in SKIPPED_SOURCE_FILES or path.suffix == ".safetensors":
            continue
        if path.is_dir():
            shutil.copytree(path, output / path.name)
        else:
            shutil.copy2(path, output / path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, required=True, help="BF16 SpeechLM checkpoint directory")
    parser.add_argument("--quantized-llm", type=Path, required=True, help="output of quantize_decoder.py")
    parser.add_argument("--output", type=Path, required=True, help="SpeechLM checkpoint to write")
    parser.add_argument(
        "--hardlink",
        action="store_true",
        help="hard-link the decoder shards instead of copying them (they are copied across filesystems)",
    )
    parser.add_argument(
        "--skip-ctc-head",
        action="store_true",
        help="leave out the CTC timestamp head, for SpeechLM plugin builds that cannot load it",
    )
    args = parser.parse_args()

    source, llm, output = args.source.resolve(), args.quantized_llm.resolve(), args.output.resolve()
    check_output_path(output, source, llm)
    if output.exists():
        raise FileExistsError(f"{output} exists")
    llm_config = read_json(llm / "config.json")
    hf_quant = read_json(llm / "hf_quant_config.json")
    if "quantization_config" not in llm_config:
        raise ValueError(f"{llm}/config.json has no quantization_config; is it a ModelOpt export?")
    check_names(llm_config["quantization_config"], "config.json")
    check_names(hf_quant.get("quantization", {}), "hf_quant_config.json")

    output.mkdir(parents=True)
    copy_source_files(source, output)

    weight_map = read_weight_map(llm)
    decoder_files = sorted(set(weight_map.values()))
    for file_name in decoder_files:
        (link_or_copy if args.hardlink else shutil.copy2)(llm / file_name, output / file_name)
    print(f"restored {restore_fp32_parameters(source, output, decoder_files)} FP32 Mamba parameters")

    exclusions = []
    for prefixes, file_name, exclusion, metadata_prefix in COPIED_MODULES:
        if args.skip_ctc_head and file_name == "model-ctc-bf16.safetensors":
            continue
        entries = find_tensors(source, prefixes)
        if not entries:
            if file_name == "perception.safetensors":
                raise ValueError(f"{source} has no perception weights")
            continue
        metadata = file_metadata(entries, metadata_prefix) if metadata_prefix else None
        copy_tensors(entries, output / file_name, metadata)
        weight_map.update(dict.fromkeys((entry.name for entry in entries), file_name))
        if exclusion:
            exclusions.append(exclusion)
        print(f"copied {len(entries)} {'/'.join(prefixes)}* tensors to {file_name}")
    total_size = sum(e.end - e.start for name in set(weight_map.values()) for e in find_tensors_in_file(output / name))
    write_json(output / INDEX_FILE, {"metadata": {"total_size": total_size}, "weight_map": weight_map})

    config = read_json(source / "config.json")
    config["quantization_config"] = speechlm_quantization_config(llm_config, exclusions)
    config["quantization_config"].setdefault("producer", hf_quant.get("producer", {}))
    write_json(output / "config.json", config)
    quantization = hf_quant.setdefault("quantization", {})
    if quantization.get("quant_algo") != "MIXED_PRECISION":
        exclude = list(quantization.get("exclude_modules") or [])
        quantization["exclude_modules"] = exclude + [entry for entry in exclusions if entry not in exclude]
    write_json(output / "hf_quant_config.json", hf_quant)
    print(f"assembled {config['quantization_config'].get('quant_algo')} checkpoint at {output}")


if __name__ == "__main__":
    main()
