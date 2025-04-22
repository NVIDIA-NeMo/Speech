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

import tensorrt as trt

from nemo.collections.llm.gpt.model.hf_llama_embedding import get_llama_bidirectional_hf_model
from nemo.export.onnx_llm_exporter import OnnxLLMExporter
from nemo.utils import logging


def get_args():
    parser = argparse.ArgumentParser(description='Test ONNX and TensorRT export for LLM embedding models.')
    parser.add_argument('--hf_model_path', type=str, required=True, help="Hugging Face model id or path.")
    parser.add_argument('--pooling_strategy', type=str, default="avg", help="Pooling strategy for the model.")
    parser.add_argument("--normalize", default=False, action="store_true", help="Normalize the embeddings or not.")
    parser.add_argument('--onnx_export_path', type=str, default="/tmp/onnx_model/", help="Path to store ONNX model.")
    parser.add_argument('--onnx_opset', type=int, default=17, help="ONNX version to use for export.")
    parser.add_argument('--trt_model_path', type=str, default="/tmp/trt_model/", help="Path to store TensorRT model.")
    parser.add_argument(
        "--trt_version_compatible",
        default=False,
        action="store_true",
        help="Whether to generate version compatible TensorRT models.",
    )

    return parser.parse_args()


def export_onnx_trt(args):
    # Base Llama model needs to be adapted to turn it into an embedding model.
    model, tokenizer = get_llama_bidirectional_hf_model(
        model_name_or_path=args.hf_model_path,
        normalize=args.normalize,
        pooling_mode=args.pooling_strategy,
        trust_remote_code=True,
    )

    input_names = ["input_ids", "attention_mask", "dimensions"]  # ONNX specific arguments, input names in this case.
    dynamic_axes_input = {
        "input_ids": {0: "batch_size", 1: "seq_length"},
        "attention_mask": {0: "batch_size", 1: "seq_length"},
        "dimensions": {0: "batch_size"},
    }

    output_names = ["embeddings"]  # ONNX specific arguments, output names in this case.
    dynamic_axes_output = {"embeddings": {0: "batch_size", 1: "embedding_dim"}}

    # Initialize ONNX exporter.
    onnx_exporter = OnnxLLMExporter(
        onnx_model_dir=args.onnx_export_path,
        model=model,
        tokenizer=tokenizer,
    )

    # Export ONNX model.
    onnx_exporter.export(
        input_names=input_names,
        output_names=output_names,
        opset=args.onnx_opset,
        dynamic_axes_input=dynamic_axes_input,
        dynamic_axes_output=dynamic_axes_output,
        export_dtype="fp32",
    )

    # Input profiles for TensorRT.
    input_profiles = [
        {
            "input_ids": [[1, 3], [16, 128], [64, 256]],
            "attention_mask": [[1, 3], [16, 128], [64, 256]],
            "dimensions": [[1], [16], [64]],
        }
    ]

    # TensorRT builder flags.
    trt_builder_flags = None
    if args.trt_version_compatible:
        trt_builder_flags = [trt.BuilderFlag.VERSION_COMPATIBLE]

    # Model specific layers to override the precision to fp32.
    override_layers_to_fp32 = [
        "/model/norm/",
        "/pooling_module",
        "/ReduceL2",
        "/Div",
    ]
    # Model specific operation wheter to override layernorm precision or not.
    override_layernorm_precision_to_fp32 = True
    profiling_verbosity = "layer_names_only"

    # Export ONNX to TensorRT.
    onnx_exporter.export_onnx_to_trt(
        trt_model_dir=args.trt_model_path,
        profiles=input_profiles,
        override_layernorm_precision_to_fp32=override_layernorm_precision_to_fp32,
        override_layers_to_fp32=override_layers_to_fp32,
        profiling_verbosity=profiling_verbosity,
        trt_builder_flags=trt_builder_flags,
    )

    assert os.path.exists(args.trt_model_path)
    assert os.path.exists(args.onnx_export_path)

    prompt = ["hello", "world"]

    prompt = onnx_exporter.get_tokenizer(prompt)
    prompt["dimensions"] = [[2]]

    output = onnx_exporter.forward(prompt)
    if output is None:
        logging.warning(f"Output is None because ONNX runtime is not installed.")


if __name__ == '__main__':
    export_onnx_trt(get_args())
