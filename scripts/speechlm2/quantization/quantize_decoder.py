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

"""Quantize the NemotronH decoder that ``export_llm_to_hf.py`` wrote, with ModelOpt.

Recipes:
  fp8    FP8 weights and activations with per-tensor static scales, plus an FP8 KV cache: ModelOpt's
         ``fp8`` preset with its ``fp8`` KV cache preset, the configuration ``hf_ptq.py --qformat fp8
         --kv_cache_qformat fp8`` builds. lm_head, the Mamba conv1d and the MoE routers stay BF16.
  nvfp4  NVFP4 for the routed and shared MoE experts and the attention projections, FP8 for the Mamba
         in_proj and out_proj, BF16 for lm_head, conv1d and the routers. ModelOpt exports this as
         ``MIXED_PRECISION``. It declares no KV cache scheme, so serve it with ``--kv-cache-dtype fp8``.

Calibration uses 128 samples of up to 512 tokens from one of two sets (``--calib-dataset``):
  hf_ptq         ModelOpt's default mix of ``cnn_dailymail`` and the gated
                 ``nvidia/Nemotron-Post-Training-Dataset-v2``, loaded as ``hf_ptq.py`` loads it. The
                 default for fp8. Needs Hugging Face access to the gated dataset.
  cnn_dailymail  The first ``cnn_dailymail`` training articles, one per forward pass. The default for nvfp4.

The output is an LLM-only checkpoint for ``assemble_checkpoint.py``.

Needs a ModelOpt build that includes NVIDIA/Model-Optimizer#2508; older builds write corrupted module
names that vLLM cannot resolve.

Example:
    python quantize_decoder.py --llm /work/llm_bf16 --recipe nvfp4 --output /work/llm_nvfp4
"""

from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path

import modelopt.torch.quantization as mtq
import torch
from checkpoint_utils import check_output_path
from modelopt.torch.export import export_hf_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

FP8_QUANTIZER = {"num_bits": (4, 3), "axis": None}


def fp8_config() -> dict:
    """ModelOpt's FP8 preset with its FP8 KV cache preset."""
    from modelopt.recipe.presets import KV_QUANT_CFG_CHOICES, QUANT_CFG_CHOICES

    return mtq.update_quant_cfg_with_kv_cache_quant(
        copy.deepcopy(QUANT_CFG_CHOICES["fp8"]), KV_QUANT_CFG_CHOICES["fp8"]["quant_cfg"]
    )


def nvfp4_config() -> dict:
    """NVFP4 everywhere it holds accuracy; FP8 for the Mamba projections, which lose accuracy in NVFP4."""
    from modelopt.torch.quantization.config import _nvfp4_cfg

    rules: list[dict] = [{"quantizer_name": "*", "enable": False}]
    for pattern in ("*mixer.experts.*", "*mixer.shared_experts*", "*q_proj*", "*k_proj*", "*v_proj*", "*o_proj*"):
        rules += [
            {"quantizer_name": f"{pattern}weight_quantizer", "cfg": _nvfp4_cfg},
            {"quantizer_name": f"{pattern}input_quantizer", "cfg": _nvfp4_cfg},
        ]
    for pattern in ("*mixer.in_proj*", "*mixer.out_proj*"):
        rules += [
            {"quantizer_name": f"{pattern}weight_quantizer", "cfg": FP8_QUANTIZER},
            {"quantizer_name": f"{pattern}input_quantizer", "cfg": FP8_QUANTIZER},
        ]
    for pattern in ("*lm_head*", "*output_layer*", "*mixer.conv1d*", "*mixer.gate*", "*router*"):
        rules.append({"quantizer_name": pattern, "enable": False})
    return {"quant_cfg": rules, "algorithm": "max"}


RECIPES = {"fp8": fp8_config, "nvfp4": nvfp4_config}
DEFAULT_CALIBRATION = {"fp8": "hf_ptq", "nvfp4": "cnn_dailymail"}


def cnn_dailymail_texts(num_samples: int) -> list[str]:
    """Return ``num_samples`` non-empty ``cnn_dailymail`` training articles."""
    from datasets import load_dataset

    articles = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train")[: num_samples * 2]["article"]
    texts = [text.strip() for text in articles if text and text.strip()]
    if len(texts) < num_samples:
        raise RuntimeError(f"only {len(texts)} calibration texts, need {num_samples}")
    return texts[:num_samples]


def calibration_loop(dataset: str, model: torch.nn.Module, tokenizer, num_samples: int, max_length: int):
    """Return the forward loop that runs ``num_samples`` calibration samples of ``dataset`` through a model."""
    if dataset == "hf_ptq":
        from modelopt.torch.utils.dataset_utils import create_forward_loop, get_dataset_dataloader

        # As hf_ptq.py prepares its tokenizer: short samples are padded to the longest one, on the left.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        dataloader = get_dataset_dataloader(
            dataset_name=["cnn_nemotron_v2_mix"],
            tokenizer=tokenizer,
            batch_size=1,
            num_samples=[num_samples],
            max_sample_length=max_length,
            device=model.device,
        )
        return create_forward_loop(dataloader=dataloader)

    texts = cnn_dailymail_texts(num_samples)

    def forward_loop(module: torch.nn.Module) -> None:
        for i, text in enumerate(texts, start=1):
            batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            with torch.no_grad():
                module(**{key: value.to(model.device) for key, value in batch.items()})
            if i % 32 == 0 or i == len(texts):
                print(f"calibration {i}/{len(texts)}")

    return forward_loop


def fill_unset_amax(model: torch.nn.Module) -> int:
    """Give quantizers that calibration never reached a positive range, and return how many there were.

    An MoE expert that no calibration token routes to keeps an all-zero activation range, which fails the
    NVFP4 export. Those experts get the largest range any quantizer saw. Their weight scales come from the
    weights themselves, so only the rarely used activation scale is a fallback.
    """
    quantizers = [m for m in model.modules() if type(m).__name__ == "TensorQuantizer"]
    amaxes = [q.amax for q in quantizers if torch.is_tensor(getattr(q, "amax", None))]
    largest = max((float(a[a > 0].max()) for a in amaxes if bool((a > 0).any())), default=1.0)
    filled = 0
    for quantizer in quantizers:
        amax = getattr(quantizer, "amax", None)
        if torch.is_tensor(amax) and bool((amax == 0).any()):
            quantizer.amax = torch.where(amax == 0, torch.full_like(amax, largest), amax)
            filled += 1
    return filled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--llm", type=Path, required=True, help="BF16 checkpoint from export_llm_to_hf.py")
    parser.add_argument("--recipe", choices=sorted(RECIPES), required=True)
    parser.add_argument("--output", type=Path, required=True, help="quantized LLM-only checkpoint to write")
    parser.add_argument(
        "--calib-dataset",
        choices=["hf_ptq", "cnn_dailymail"],
        default=None,
        help="calibration set; default: hf_ptq for fp8, cnn_dailymail for nvfp4",
    )
    parser.add_argument("--calib-size", type=int, default=128)
    parser.add_argument("--calib-seq", type=int, default=512, help="maximum tokens per calibration sample")
    parser.add_argument("--overwrite", action="store_true", help="replace --output if it exists")
    args = parser.parse_args()

    check_output_path(args.output, args.llm)
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")
        shutil.rmtree(args.output)

    dataset = args.calib_dataset or DEFAULT_CALIBRATION[args.recipe]
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    model = AutoModelForCausalLM.from_pretrained(args.llm, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    print(f"recipe {args.recipe}, calibration {dataset}: {args.calib_size} samples of up to {args.calib_seq} tokens")
    forward_loop = calibration_loop(dataset, model, tokenizer, args.calib_size, args.calib_seq)

    mtq.quantize(model, RECIPES[args.recipe](), forward_loop=forward_loop)
    print(f"{fill_unset_amax(model)} quantizers had no calibrated range")
    export_hf_checkpoint(model, dtype=torch.bfloat16, export_dir=str(args.output))
    print(f"wrote {args.recipe} checkpoint to {args.output}")


if __name__ == "__main__":
    main()
