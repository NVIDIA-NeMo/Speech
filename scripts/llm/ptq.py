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
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="NeMo PTQ argument parser",
    )
    parser.add_argument("-nc", "--nemo_checkpoint", type=str, help="Source NeMo 2.0 checkpoint")
    parser.add_argument("--decoder_type", type=str, help="Decoder type for TensorRT-Model-Optimizer")
    parser.add_argument("-ctp", "--calib_tp", type=int, default=1)
    parser.add_argument("-cpp", "--calib_pp", type=int, default=1)
    parser.add_argument("-tps", "--tensor_parallelism_size", type=int, default=1)
    parser.add_argument("-pps", "--pipeline_parallelism_size", type=int, default=1)
    parser.add_argument('-out', '--output_path', type=str, help='Path for the exported engine')
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

    args = parser.parse_args()
    if args.output_path is None:
        args.output_path = (
            f"./qnemo_{args.algorithm}_tp{args.tensor_parallelism_size}_pp{args.pipeline_parallelism_size}"
        )
    return args


def main():
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
        path=args.output_path,
        decoder_type=args.decoder_type,
        inference_tensor_parallel=args.tensor_parallelism_size,
        inference_pipeline_parallel=args.pipeline_parallelism_size,
        dtype=args.dtype,
    )

    quantizer = quantization.Quantizer(quantization_config, export_config)
    model = quantization.load_with_modelopt_layer_spec(args.nemo_checkpoint, args.calib_tp, args.calib_pp)
    model = quantizer.quantize(model)
    quantizer.export(model)


if __name__ == '__main__':
    main()
