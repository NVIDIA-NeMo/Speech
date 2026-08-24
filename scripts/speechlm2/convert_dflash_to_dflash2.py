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

"""Bootstrap a DFlash2 checkpoint from a compatible trained DFlash checkpoint.

The public DFlash2 release adds two-tap grouped convolutions to every draft
layer and a low-rank candidate selector.  This converter preserves the trained
DFlash backbone, initializes both convolutions as exact identities, and
initializes the selector as a no-op.  The resulting checkpoint exercises the
DFlash2 runtime without pretending that the new parameters have been trained.
The source checkpoint must store its weights in one safetensors file.

Example::

    python scripts/speechlm2/convert_dflash_to_dflash2.py \
      nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash \
      /path/to/lightning-dflash2-bootstrap
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_CONV_GROUP_SIZE = 16
DEFAULT_CONV_KERNEL_SIZE = 2
DEFAULT_SELECTOR_RANK = 256
DEFAULT_SELECTOR_TOP_K = 16
_UNQUANTIZED_MODULE_PATTERNS = ("*attention_conv*", "*mlp_conv*", "*candidate_selector*")
_SOURCE_ARCHITECTURE = "DFlashDraftModel"
_TARGET_ARCHITECTURE = "DFlash2DraftModel"


def build_dflash2_config(
    source_config: dict[str, Any],
    *,
    conv_group_size: int = DEFAULT_CONV_GROUP_SIZE,
    conv_kernel_size: int = DEFAULT_CONV_KERNEL_SIZE,
    selector_rank: int = DEFAULT_SELECTOR_RANK,
    selector_top_k: int = DEFAULT_SELECTOR_TOP_K,
) -> dict[str, Any]:
    """Return a validated DFlash2 config derived from ``source_config``."""
    config = copy.deepcopy(source_config)
    architectures = config.get("architectures") or []
    if _SOURCE_ARCHITECTURE not in architectures:
        raise ValueError(f"The source checkpoint must declare DFlashDraftModel; got architectures={architectures!r}.")

    hidden_size = _positive_int(config, "hidden_size")
    _positive_int(config, "num_hidden_layers")
    _positive_int(config, "vocab_size")
    for name, value in (
        ("conv_group_size", conv_group_size),
        ("conv_kernel_size", conv_kernel_size),
        ("selector_rank", selector_rank),
        ("selector_top_k", selector_top_k),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    if conv_group_size > hidden_size or hidden_size % conv_group_size:
        raise ValueError(f"conv_group_size={conv_group_size} must divide hidden_size={hidden_size}.")
    if selector_top_k > config["vocab_size"]:
        raise ValueError(f"selector_top_k={selector_top_k} cannot exceed vocab_size={config['vocab_size']}.")

    dflash_config = dict(config.get("dflash_config") or {})
    target_layer_ids = dflash_config.get("target_layer_ids") or config.get("target_layer_ids")
    if not isinstance(target_layer_ids, list) or not target_layer_ids:
        raise ValueError("The source checkpoint must define non-empty dflash_config.target_layer_ids.")
    if any(not isinstance(layer, int) or isinstance(layer, bool) or layer < 0 for layer in target_layer_ids):
        raise ValueError(f"target_layer_ids must contain non-negative integers; got {target_layer_ids!r}.")
    if "causal" in dflash_config and not isinstance(dflash_config["causal"], bool):
        raise ValueError(f"dflash_config.causal must be a boolean; got {dflash_config['causal']!r}.")

    dflash_config.update(
        {
            "target_layer_ids": target_layer_ids,
            "conv_group_size": conv_group_size,
            "conv_kernel_size": conv_kernel_size,
            "selector_rank": selector_rank,
            "selector_top_k": selector_top_k,
        }
    )
    config["architectures"] = [_TARGET_ARCHITECTURE]
    config["dflash_config"] = dflash_config
    if "is_causal" not in config and "causal" in dflash_config:
        config["is_causal"] = bool(dflash_config["causal"])
    config.pop("num_target_layers", None)
    quantization_config = config.get("quantization_config")
    if isinstance(quantization_config, dict):
        for key in ("ignore", "exclude_modules"):
            _extend_patterns(quantization_config, key)
    return config


def dflash2_tensor_shapes(config: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Describe the additional tensors required by vLLM's DFlash2 model."""
    hidden_size = _positive_int(config, "hidden_size")
    num_layers = _positive_int(config, "num_hidden_layers")
    vocab_size = _positive_int(config, "vocab_size")
    dflash_config = config.get("dflash_config") or {}
    group_size = int(dflash_config["conv_group_size"])
    kernel_size = int(dflash_config["conv_kernel_size"])
    selector_rank = int(dflash_config["selector_rank"])
    if hidden_size % group_size:
        raise ValueError(f"conv_group_size={group_size} must divide hidden_size={hidden_size}.")

    num_groups = hidden_size // group_size
    shapes: dict[str, tuple[int, ...]] = {}
    for layer in range(num_layers):
        for name in ("attention_conv", "mlp_conv"):
            prefix = f"layers.{layer}.{name}"
            shapes[f"{prefix}.base_kernel"] = (2, kernel_size, hidden_size)
            # ReplicatedLinear stores [output, input]. The output rows flatten
            # vLLM's (side, tap, group) coefficient layout in that order.
            shapes[f"{prefix}.kernel_projection.weight"] = (
                2 * kernel_size * num_groups,
                hidden_size,
            )
    shapes["candidate_selector.predecessor_codebook"] = (vocab_size, selector_rank)
    shapes["candidate_selector.successor_codebook"] = (vocab_size, selector_rank)
    shapes["candidate_selector.hidden_projection.weight"] = (selector_rank, hidden_size)
    return shapes


def initialize_dflash2_tensors(config: dict[str, Any]):
    """Create identity convolutions and a neutral selector in BF16.

    DFlash2's unquantized runtime modules load these tensors in the model dtype.
    BF16 deliberately matches the Lightning target and keeps the bootstrap
    memory-representative of a trained Lightning DFlash2 checkpoint.
    """
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Checkpoint conversion requires PyTorch.") from error

    tensors = {}
    for name, shape in dflash2_tensor_shapes(config).items():
        tensor = torch.zeros(shape, dtype=torch.bfloat16)
        if name.endswith(".base_kernel"):
            # Both the pre-attention/MLP and post-attention/MLP convolutions
            # pass the current token through unchanged. All older taps and all
            # dynamic coefficients remain zero.
            tensor[:, 0, :].fill_(1)
        tensors[name] = tensor
    return tensors


def convert_checkpoint(
    source: str,
    output: str | Path,
    *,
    conv_group_size: int = DEFAULT_CONV_GROUP_SIZE,
    conv_kernel_size: int = DEFAULT_CONV_KERNEL_SIZE,
    selector_rank: int = DEFAULT_SELECTOR_RANK,
    selector_top_k: int = DEFAULT_SELECTOR_TOP_K,
) -> Path:
    """Convert ``source`` into a new local DFlash2 checkpoint directory."""
    source_dir, source_label = _resolve_source(source)
    config_path = source_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing source config: {config_path}")
    source_config = json.loads(config_path.read_text())
    config = build_dflash2_config(
        source_config,
        conv_group_size=conv_group_size,
        conv_kernel_size=conv_kernel_size,
        selector_rank=selector_rank,
        selector_top_k=selector_top_k,
    )

    output_dir = Path(output).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as staging:
        staging_dir = Path(staging)
        _write_checkpoint(source_dir, staging_dir, config, source_label)
        staging_dir.rename(output_dir)
    return output_dir


def _positive_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"config.{key} must be a positive integer; got {value!r}.")
    return value


def _extend_patterns(config: dict[str, Any], key: str) -> None:
    patterns = config.get(key)
    if patterns is None:
        patterns = []
    if not isinstance(patterns, list) or any(not isinstance(pattern, str) for pattern in patterns):
        raise ValueError(f"quantization_config.{key} must be a list of strings; got {patterns!r}.")
    config[key] = [*patterns, *(pattern for pattern in _UNQUANTIZED_MODULE_PATTERNS if pattern not in patterns)]


def _resolve_source(source: str) -> tuple[Path, str]:
    source_path = Path(source).expanduser()
    if source_path.is_dir():
        return source_path.resolve(), str(source_path.resolve())

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "Resolving a Hugging Face model ID requires huggingface_hub. "
            "Install NeMo's standard dependencies or pass a local checkpoint directory."
        ) from error

    snapshot = snapshot_download(
        source,
        allow_patterns=["*.json", "*.safetensors", "mask_embedding.pt", "LICENSE*", "README*"],
    )
    return Path(snapshot), source


def _copy_metadata(source_dir: Path, output_dir: Path) -> None:
    for path in source_dir.glob("LICENSE*"):
        shutil.copy2(path, output_dir / path.name)
    mask_embedding = source_dir / "mask_embedding.pt"
    if mask_embedding.is_file():
        shutil.copy2(mask_embedding, output_dir / mask_embedding.name)

    source_quant_config = source_dir / "hf_quant_config.json"
    if source_quant_config.is_file():
        quant_config = json.loads(source_quant_config.read_text())
        quantization = quant_config.get("quantization")
        if isinstance(quantization, dict):
            _extend_patterns(quantization, "exclude_modules")
        (output_dir / source_quant_config.name).write_text(json.dumps(quant_config, indent=2) + "\n")


def _write_checkpoint(source_dir: Path, output_dir: Path, config: dict[str, Any], source_label: str) -> None:
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as error:
        raise RuntimeError("Checkpoint conversion requires safetensors.") from error

    weight_files = sorted(source_dir.glob("*.safetensors"))
    if len(weight_files) != 1:
        raise ValueError(
            "This converter expects one safetensors file in the source checkpoint; "
            f"found {[path.name for path in weight_files]!r}."
        )

    source_weights = weight_files[0]
    with safe_open(source_weights, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}

    additions = initialize_dflash2_tensors(config)
    overlap = sorted(tensors.keys() & additions.keys())
    if overlap:
        raise ValueError(f"Source checkpoint already contains DFlash2 tensors: {overlap[:5]!r}.")
    tensors.update(additions)

    save_file(tensors, output_dir / "model.safetensors", metadata=metadata)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {
        "source": source_label,
        "initialization": {
            "convolutions": "identity",
            "candidate_selector": "neutral",
        },
        "trained_dflash2_parameters": False,
    }
    (output_dir / "dflash2_bootstrap.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "README.md").write_text(
        "# Lightning DFlash2 bootstrap\n\n"
        f"This checkpoint was derived from `{source_label}`. The trained DFlash "
        "backbone is preserved, while the DFlash2 convolutions are initialized as "
        "identities and its candidate selector is neutral. It is a functional "
        "DFlash2 runtime artifact, not a DFlash2-trained performance release.\n"
    )
    _copy_metadata(source_dir, output_dir)

    # TemporaryDirectory and safetensors default to 0700/0600. The converted
    # public model is commonly bind-mounted into a root-squashed inference
    # container, so normalize it to read-only model-artifact permissions.
    output_dir.chmod(0o755)
    for path in output_dir.iterdir():
        if path.is_file():
            path.chmod(0o644)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Local DFlash checkpoint directory or Hugging Face model ID")
    parser.add_argument("output", help="New local output directory")
    parser.add_argument("--conv-group-size", type=int, default=DEFAULT_CONV_GROUP_SIZE)
    parser.add_argument("--conv-kernel-size", type=int, default=DEFAULT_CONV_KERNEL_SIZE)
    parser.add_argument("--selector-rank", type=int, default=DEFAULT_SELECTOR_RANK)
    parser.add_argument("--selector-top-k", type=int, default=DEFAULT_SELECTOR_TOP_K)
    args = parser.parse_args()
    output = convert_checkpoint(
        args.source,
        args.output,
        conv_group_size=args.conv_group_size,
        conv_kernel_size=args.conv_kernel_size,
        selector_rank=args.selector_rank,
        selector_top_k=args.selector_top_k,
    )
    print(f"Created DFlash2 checkpoint: {output}")


if __name__ == "__main__":
    main()
