# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import os

from nemo.collections import llm
from nemo.collections.llm.modelopt import ExportConfig, QuantizationConfig
from nemo.collections.llm.modelopt.quantization.quant_cfg_choices import get_quant_cfg_choices
from nemo.collections.llm.modelopt.quantization.quantizer import KV_QUANT_CFG_CHOICES
from nemo.utils import logging


def get_args():
    """Parses PTQ arguments."""
    QUANT_CFG_CHOICES_LIST = ["no_quant", *get_quant_cfg_choices()]
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="NeMo PTQ argument parser"
    )
    parser.add_argument("-nc", "--nemo_checkpoint", type=str, help="Source NeMo 2.0 checkpoint")
    parser.add_argument(
        "--tokenizer", type=str, help="Tokenizer to use. If not provided, model tokenizer will be used"
    )
    parser.add_argument("--decoder_type", type=str, help="Decoder type for Model-Optimizer")
    parser.add_argument("-ctp", "--calibration_tp", "--calib_tp", type=int, default=1)
    parser.add_argument("-cep", "--calibration_ep", "--calib_ep", type=int, default=1)
    parser.add_argument("-cpp", "--calibration_pp", "--calib_pp", type=int, default=1)
    parser.add_argument(
        "--num_layers_in_first_pipeline_stage",
        type=int,
        default=None,
        help="Number of layers in the first pipeline stage. If None, pipeline parallelism will default to evenly split layers.",
    )
    parser.add_argument(
        "--num_layers_in_last_pipeline_stage",
        type=int,
        default=None,
        help="Number of layers in the last pipeline stage. If None, pipeline parallelism will default to evenly split layers.",
    )
    parser.add_argument(
        "-itp",
        "--inference_tp",
        "--tensor_parallelism_size",
        type=int,
        default=1,
        help="TRT-LLM engine TP size. (Only used when `--export_format` is 'trtllm')",
    )
    parser.add_argument(
        "-ipp",
        "--inference_pp",
        "--pipeline_parallelism_size",
        type=int,
        default=1,
        help="TRT-LLM engine PP size. (Only used when `--export_format` is 'trtllm')",
    )
    parser.add_argument("--devices", type=int, help="Number of GPUs to use per node")
    parser.add_argument("-nodes", "--num_nodes", type=int, help="Number of nodes used")
    parser.add_argument("-out", "--export_path", "--output_path", type=str, help="Path for the exported engine")
    parser.add_argument(
        "--export_format", default="trtllm", choices=["trtllm", "nemo", "hf"], help="Model format to export as"
    )
    parser.add_argument(
        "-algo",
        "--algorithm",
        type=str,
        default="fp8",
        help="Model-Optimizer quantization algorithm",
    )
    parser.add_argument(
        "-awq_bs", "--awq_block_size", type=int, default=128, help="Block size for AWQ quantization algorithms"
    )
    parser.add_argument("--sq_alpha", type=float, default=0.5, help="Smooth-Quant alpha parameter")
    parser.add_argument("--enable_kv_cache", help="Enables KV-cache quantization", action="store_true")
    parser.add_argument("--disable_kv_cache", dest="enable_kv_cache", action="store_false")
    parser.set_defaults(enable_kv_cache=None)
    parser.add_argument(
        "--kv_cache_qformat",
        type=str,
        default="fp8",
        choices=KV_QUANT_CFG_CHOICES,
        help="KV-cache quantization format",
    )
    parser.add_argument(
        "-dt", "--dtype", default="bf16", choices=["16", "bf16"], help="Default precision for non-quantized layers"
    )
    parser.add_argument("-bs", "--batch_size", default=64, type=int, help="Calibration batch size")
    parser.add_argument("-sl", "--seq_len", default=128, type=int, help="Length of the tokenized text")
    parser.add_argument(
        "-calib_size", "--calibration_dataset_size", default=512, type=int, help="Size of calibration dataset"
    )
    parser.add_argument(
        "-calib_ds",
        "--calibration_dataset",
        default="cnn_dailymail",
        type=str,
        help='Calibration dataset to be used. Should be "wikitext", "cnn_dailymail" or path to a local .json file',
    )
    parser.add_argument(
        "--generate_sample", help="Generate sample model output after performing PTQ", action="store_true"
    )
    parser.add_argument(
        "--trust_remote_code", help="Trust remote code when loading HuggingFace models", action="store_true"
    )
    parser.add_argument("--legacy_ckpt", help="Load ckpt saved with TE < 1.14", action="store_true")
    args = parser.parse_args()

    if args.algorithm not in QUANT_CFG_CHOICES_LIST and not os.path.isfile(args.algorithm):
        raise ValueError(
            f"Quantization algorithm {args.algorithm} is not supported: choose one of {QUANT_CFG_CHOICES_LIST} "
            "or provide a path to a JSON file with a quantization configuration."
        )

    if args.export_path is None:
        if args.export_format == "trtllm":
            args.export_path = f"./qnemo_{args.algorithm}_tp{args.inference_tp}_pp{args.inference_pp}"
        else:
            args.export_path = f"./{args.export_format}_{args.algorithm}"

    if args.devices is None:
        args.devices = args.calibration_tp
    if args.num_nodes is None:
        args.num_nodes = args.calibration_pp

    return args


def main():
    """Example NeMo 2.0 Post Training Quantization workflow"""
    args = get_args()
    if os.path.exists(args.export_path):
        logging.info(f"Export path: {args.export_path} already exists. Will skip PTQ")
        return

    quantization_config = QuantizationConfig(
        algorithm=None if args.algorithm == "no_quant" else args.algorithm,
        awq_block_size=args.awq_block_size,
        sq_alpha=args.sq_alpha,
        enable_kv_cache=args.enable_kv_cache,
        kv_cache_qformat=args.kv_cache_qformat,
        calibration_dataset=args.calibration_dataset,
        calibration_dataset_size=args.calibration_dataset_size,
        calibration_batch_size=args.batch_size,
        calibration_seq_len=args.seq_len,
    )
    export_config = ExportConfig(
        export_format=args.export_format,
        path=args.export_path,
        decoder_type=args.decoder_type,
        inference_tp=args.inference_tp,
        inference_pp=args.inference_pp,
        dtype=args.dtype,
        generate_sample=args.generate_sample,
    )

    llm.ptq(
        model_path=args.nemo_checkpoint,
        export_config=export_config,
        calibration_tp=args.calibration_tp,
        calibration_pp=args.calibration_pp,
        calibration_ep=args.calibration_ep,
        num_layers_in_first_pipeline_stage=args.num_layers_in_first_pipeline_stage,
        num_layers_in_last_pipeline_stage=args.num_layers_in_last_pipeline_stage,
        devices=args.devices,
        num_nodes=args.num_nodes,
        quantization_config=quantization_config,
        tokenizer_path=args.tokenizer,
        legacy_ckpt=args.legacy_ckpt,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
