# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script exports multimodal model to TensorRT and do a local inference test.
For multimodal model, it supports the following models:
- NEVA
- Video-NEVA
- LITA
- VILA
- VITA
- SALM
"""

import argparse
import os

from nemo.export.tensorrt_mm_exporter import TensorRTMMExporter


def parse_args():
    parser = argparse.ArgumentParser(description='Export multimodal model to TensorRT')
    parser.add_argument('--output_dir', required=True, help='Directory to save the exported model')
    parser.add_argument(
        '--visual_checkpoint_path',
        required=True,
        help='Path to the visual model checkpoint or perception model checkpoint',
    )
    parser.add_argument('--llm_checkpoint_path', required=True, help='Source .nemo file for llm')
    parser.add_argument(
        '--modality',
        default="vision",
        choices=["vision", "audio"],
        help="Modality of the model",
    )
    parser.add_argument(
        '--model_type',
        type=str,
        required=True,
        choices=["neva", "video-neva", "lita", "vila", "vita", "salm"],
        help="Type of the model that is supported.",
    )

    parser.add_argument(
        '--llm_model_type',
        type=str,
        required=True,
        choices=["gptnext", "gpt", "llama", "falcon", "starcoder", "mixtral", "gemma"],
        help="Type of LLM. gptnext, gpt, llama, falcon, and starcoder are only supported."
        " gptnext and gpt are the same and keeping it for backward compatibility",
    )

    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='tensor parallelism size')
    parser.add_argument('--max_input_len', type=int, default=4096, help='Maximum input length')
    parser.add_argument('--max_output_len', type=int, default=256, help='Maximum output length')
    parser.add_argument('--max_batch_size', type=int, default=1, help='Maximum batch size')
    parser.add_argument(
        '--vision_max_batch_size',
        type=int,
        default=1,
        help='Max batch size of the visual inputs, for lita/vita model with video inference, this should be set to 256',
    )
    parser.add_argument('--max_multimodal_len', type=int, default=3072, help='Maximum multimodal length')
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
        type=str,
        help="dtype of the model on TensorRT",
    )
    parser.add_argument(
        '--delete_existing_files', action='store_true', help='Delete existing files in the output directory'
    )
    parser.add_argument(
        '--test_export_only', action='store_true', help='Only test the export without saving the model'
    )
    parser.add_argument('--input_text', help='Input text for inference')
    parser.add_argument('--input_media', default=None, help='Input media file for inference')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--max_output', type=int, default=128, help='Maximum output length for inference')
    parser.add_argument('--top_k', type=int, default=1, help='Top k for sampling')
    parser.add_argument('--top_p', type=float, default=0.0, help='Top p for sampling')
    parser.add_argument("--temperature", default=1.0, type=float, help="temperature")
    parser.add_argument("--repetition_penalty", default=1.0, type=float, help="repetition_penalty")
    parser.add_argument("--num_beams", default=1, type=int, help="num_beams")

    args = parser.parse_args()
    return args


def main(args):
    exporter = TensorRTMMExporter(model_dir=args.output_dir, load_model=False, modality=args.modality)
    exporter.export(
        visual_checkpoint_path=args.visual_checkpoint_path,
        llm_checkpoint_path=args.llm_checkpoint_path,
        model_type=args.model_type,
        llm_model_type=args.llm_model_type,
        tensor_parallel_size=args.tensor_parallel_size,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        max_batch_size=args.max_batch_size,
        vision_max_batch_size=args.vision_max_batch_size,
        max_multimodal_len=args.max_multimodal_len,
        dtype=args.dtype,
        delete_existing_files=args.delete_existing_files,
        load_model=not args.test_export_only,
    )
    test_inference = not args.test_export_only
    if test_inference:
        assert args.input_media is not None, "Input media file is required for inference"
        assert os.path.exists(args.input_media), f"Input media file {args.input_media} does not exist"
        output = exporter.forward(
            input_text=args.input_text,
            input_media=args.input_media,
            batch_size=args.batch_size,
            max_output_len=args.max_output,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            num_beams=args.num_beams,
        )
        print(output)


if __name__ == '__main__':
    args = parse_args()
    main(args)
