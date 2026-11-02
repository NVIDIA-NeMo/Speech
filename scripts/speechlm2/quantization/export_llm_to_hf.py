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

"""Export the language model of a SpeechLM checkpoint as a standalone Hugging Face NemotronH checkpoint.

ModelOpt quantizes Hugging Face models, so this is the input of ``quantize_decoder.py``. The NemotronH
config comes from ``<checkpoint>/llm_backbone/config.json`` and the tokenizer files from the checkpoint
root. NeMo weight names become Hugging Face ones, and NeMo's stacked MoE experts are split into one
tensor per expert.

The MTP draft head (``llm.mtp.*``) is left out: Hugging Face's NemotronH has no MTP layers and discards
those weights at load. ``assemble_checkpoint.py`` copies it from the source checkpoint instead.

Example:
    python export_llm_to_hf.py --checkpoint /models/speechlm-bf16 --output /work/llm_bf16
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterator

import torch
from checkpoint_utils import INDEX_FILE, check_output_path, read_json, weight_files, write_json
from safetensors import safe_open
from safetensors.torch import save_file

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
)


def hf_llm_weights(checkpoint: Path) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the language model's tensors under Hugging Face NemotronH names."""
    for file_name in weight_files(checkpoint):
        with safe_open(checkpoint / file_name, framework="pt", device="cpu") as f:
            for name in f.keys():
                # Transformer Engine bookkeeping (zero-length tensors) and the MTP draft head have no
                # place in Hugging Face's NemotronH.
                if not name.startswith("llm.") or name.endswith("_extra_state") or name.startswith("llm.mtp."):
                    continue
                yield from hf_names(name, f.get_tensor(name))


def hf_names(name: str, tensor: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
    """Map one NeMo language-model tensor to its Hugging Face NemotronH tensor(s)."""
    hf_name = name.replace("llm.model.", "backbone.").replace("llm.lm_head", "lm_head")
    # Automodel keeps Mamba's A_log, D and dt_bias in an FP32 holder; NemotronH has them on the mixer.
    hf_name = hf_name.replace("._fp32_params.", ".")
    if hf_name == "backbone.embed_tokens.weight":
        hf_name = "backbone.embeddings.weight"
    elif hf_name == "backbone.norm.weight":
        hf_name = "backbone.norm_f.weight"

    if hf_name.endswith(".experts.down_projs"):
        prefix = hf_name.removesuffix(".experts.down_projs")
        for i in range(tensor.shape[0]):
            yield f"{prefix}.experts.{i}.down_proj.weight", tensor[i].t().contiguous()
    elif hf_name.endswith(".experts.gate_and_up_projs"):
        prefix = hf_name.removesuffix(".experts.gate_and_up_projs")
        for i in range(tensor.shape[0]):
            yield f"{prefix}.experts.{i}.up_proj.weight", tensor[i].t().contiguous()
    else:
        yield hf_name, tensor


def save_sharded(weights: Iterator[tuple[str, torch.Tensor]], output: Path, max_shard_bytes: int) -> dict:
    """Save ``weights`` in shards of at most ``max_shard_bytes`` and return the index."""
    shards: list[list[str]] = []
    shard: dict[str, torch.Tensor] = {}
    shard_bytes = total_bytes = 0

    def flush() -> None:
        nonlocal shard, shard_bytes
        if shard:
            save_file(shard, output / f"shard-{len(shards):05d}.tmp")
            shards.append(list(shard))
            print(f"wrote shard {len(shards)}: {len(shard)} tensors, {shard_bytes / 1e9:.2f} GB")
            shard, shard_bytes = {}, 0

    for name, tensor in weights:
        nbytes = tensor.numel() * tensor.element_size()
        if shard and shard_bytes + nbytes > max_shard_bytes:
            flush()
        shard[name] = tensor
        shard_bytes += nbytes
        total_bytes += nbytes
    flush()

    weight_map = {}
    for i, names in enumerate(shards):
        final = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        (output / f"shard-{i:05d}.tmp").replace(output / final)
        weight_map.update(dict.fromkeys(names, final))
    return {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="BF16 SpeechLM checkpoint directory")
    parser.add_argument("--output", type=Path, required=True, help="Hugging Face LLM checkpoint to write")
    parser.add_argument("--max-shard-size-gb", type=float, default=8.0)
    parser.add_argument("--overwrite", action="store_true", help="replace --output if it exists")
    args = parser.parse_args()

    checkpoint, output = args.checkpoint.resolve(), args.output.resolve()
    check_output_path(output, checkpoint)
    llm_config_path = checkpoint / "llm_backbone" / "config.json"
    if not llm_config_path.exists():
        raise FileNotFoundError(f"{llm_config_path} is missing; it holds the NemotronH config")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    llm_config = read_json(llm_config_path)
    llm_config.pop("quantization_config", None)
    write_json(output / "config.json", llm_config)
    for name in TOKENIZER_FILES:
        if (checkpoint / name).exists():
            shutil.copy2(checkpoint / name, output / name)

    index = save_sharded(hf_llm_weights(checkpoint), output, int(args.max_shard_size_gb * 1e9))
    write_json(output / INDEX_FILE, index)
    print(f"exported {len(index['weight_map'])} tensors ({index['metadata']['total_size'] / 1e9:.1f} GB) to {output}")


if __name__ == "__main__":
    main()
