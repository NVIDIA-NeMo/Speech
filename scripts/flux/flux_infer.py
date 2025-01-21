# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import torch

from nemo.collections.diffusion.models.flux.pipeline import FluxInferencePipeline
from nemo.collections.diffusion.utils.flux_pipeline_utils import configs
from nemo.collections.diffusion.utils.mcore_parallel_utils import Utils


def parse_args():
    # pylint: disable=C0116
    parser = argparse.ArgumentParser(
        description="The flux inference pipeline is utilizing megatron core transformer.\n"
        "Please prepare the necessary checkpoints for flux model on local disk in order to use this script"
    )

    parser.add_argument("--flux_ckpt", type=str, default="", help="Path to Flux transformer checkpoint(s)")
    parser.add_argument("--vae_ckpt", type=str, default="/ckpts/ae.safetensors", help="Path to \'ae.safetensors\'")
    parser.add_argument(
        "--clip_version",
        type=str,
        default='/ckpts/text_encoder',
        help="Clip version, provide either ckpt dir or clip version like openai/clip-vit-large-patch14",
    )
    parser.add_argument(
        "--t5_version",
        type=str,
        default='/ckpts/text_encoder_2',
        help="Clip version, provide either ckpt dir or clip version like google/t5-v1_1-xxl",
    )
    parser.add_argument(
        "--do_convert_from_hf",
        action='store_true',
        default=False,
        help="Must be true if provided checkpoint is not already converted to NeMo version",
    )
    parser.add_argument(
        "--save_converted_model_to",
        type=str,
        default="/ckpts",
        help="Whether to save the converted NeMo transformer checkpoint for Flux",
    )
    parser.add_argument(
        "--version",
        type=str,
        default='dev',
        choices=['dev', 'schnell'],
        help="Must align with the checkpoint provided.",
    )
    parser.add_argument("--height", type=int, default=1024, help="Image height.")
    parser.add_argument("--width", type=int, default=1024, help="Image width.")
    parser.add_argument("--inference_steps", type=int, default=10, help="Number of inference steps to run.")
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=1, help="Number of images to generate for each prompt."
    )
    parser.add_argument("--guidance", type=float, default=0.0, help="Guidance scale.")
    parser.add_argument(
        "--offload", action='store_true', default=False, help="Offload modules to cpu after being called."
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="A cat holding a sign that says hello world",
        help="Inference prompts, use \',\' to separate if multiple prompts are provided.",
    )
    parser.add_argument("--bf16", action='store_true', default=False, help="Use bf16 in inference.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print('Initializing model parallel config')
    Utils.initialize_distributed(1, 1, 1)

    print('Initializing flux inference pipeline')
    params = configs[args.version]
    params.vae_config.ckpt = args.vae_ckpt
    params.clip_params.version = args.clip_version
    params.t5_params.version = args.t5_version
    pipe = FluxInferencePipeline(params)

    print('Loading transformer weights')
    pipe.load_from_pretrained(
        args.flux_ckpt,
        do_convert_from_hf=args.do_convert_from_hf,
        save_converted_model_to=args.save_converted_model_to,
    )
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    text = args.prompts.split(',')
    pipe(
        text,
        max_sequence_length=512,
        height=args.height,
        width=args.width,
        num_inference_steps=args.inference_steps,
        num_images_per_prompt=args.num_images_per_prompt,
        offload=args.offload,
        guidance_scale=args.guidance,
        dtype=dtype,
    )
