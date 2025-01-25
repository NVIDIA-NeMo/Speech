# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.llm import quantization


def get_args():
    """Parses PTQ arguments."""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="NeMo PTQ argument parser",
    )
    parser.add_argument("-nc", "--nemo_checkpoint", type=str, help="Source NeMo 2.0 checkpoint")
    parser.add_argument("--decoder_type", type=str, help="Decoder type for TensorRT-Model-Optimizer")
    parser.add_argument("-ctp", "--calibration_tp", "--calib_tp", type=int, default=1)
    parser.add_argument("-cpp", "--calibration_pp", "--calib_pp", type=int, default=1)
    parser.add_argument("-itp", "--inference_tp", "--tensor_parallelism_size", type=int, default=1)
    parser.add_argument("-ipp", "--inference_pp", "--pipeline_parallelism_size", type=int, default=1)
    parser.add_argument('-out', '--export_path', '--output_path', type=str, help='Path for the exported engine')
    parser.add_argument(
        '-algo',
        '--algorithm',
        type=str,
        default="fp8",
        choices=["no_quant", "int8", "int8_sq", "fp8", "int4_awq", "w4a8_awq", "int4"],
        help='TensorRT-Model-Optimizer quantization algorithm',
    )
    parser.add_argument(
        '-awq_bs', '--awq_block_size', type=int, default=128, help='Block size for AWQ quantization algorithms'
    )
    parser.add_argument('--sq_alpha', type=float, default=0.5, help='Smooth-Quant alpha parameter')
    parser.add_argument('--enable_kv_cache', help='Enables KV-cache quantization', action='store_true')
    parser.add_argument('--disable_kv_cache', dest='enable_kv_cache', action='store_false')
    parser.set_defaults(enable_kv_cache=None)
    parser.add_argument(
        '-dt', '--dtype', default="bf16", choices=["16", "bf16"], help='Default precision for non-quantized layers'
    )
    parser.add_argument('-bs', '--batch_size', default=64, type=int, help='Calibration batch size')
    parser.add_argument('-sl', '--seq_len', default=128, type=int, help='Length of the tokenized text')
    parser.add_argument(
        '-calib_size', '--calibration_dataset_size', default=512, type=int, help='Size of calibration dataset'
    )
    parser.add_argument(
        '-calib_ds',
        '--calibration_dataset',
        default="cnn_dailymail",
        type=str,
        help='Calibration dataset to be used. Should be \"wikitext\", \"cnn_dailymail\" or path to a local .json file',
    )
    parser.add_argument(
        '--generate_sample', help='Generate sample model output after performing PTQ', action='store_true'
    )
    parser.add_argument('--ckpt_load_strictness', type=str, help='Defines handling of checkpoint load mismatch')
    parser.set_defaults(generate_sample=False)

    args = parser.parse_args()
    if args.export_path is None:
        args.export_path = f"./qnemo_{args.algorithm}_tp{args.inference_tp}_pp{args.inference_pp}"
    return args


def main():
    """Example NeMo 2.0 Post Training Quantization workflow"""

    args = get_args()

    quantization_config = quantization.QuantizationConfig(
        algorithm=None if args.algorithm == "no_quant" else args.algorithm,
        awq_block_size=args.awq_block_size,
        sq_alpha=args.sq_alpha,
        enable_kv_cache=args.enable_kv_cache,
        calibration_dataset=args.calibration_dataset,
        calibration_dataset_size=args.calibration_dataset_size,
        calibration_batch_size=args.batch_size,
        calibration_seq_len=args.seq_len,
    )

    export_config = quantization.ExportConfig(
        path=args.export_path,
        decoder_type=args.decoder_type,
        inference_tp=args.inference_tp,
        inference_pp=args.inference_pp,
        dtype=args.dtype,
        generate_sample=args.generate_sample,
    )

    quantizer = quantization.Quantizer(quantization_config, export_config)
    model = quantization.load_with_modelopt_layer_spec(
        args.nemo_checkpoint,
        args.calibration_tp,
        args.calibration_pp,
        ckpt_load_strictness=args.ckpt_load_strictness,
    )
    model = quantizer.quantize(model)
    quantizer.export(model, args.nemo_checkpoint)


if __name__ == '__main__':
    main()
