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

"""
Example:
  python scripts/vlm/mllama_generate.py --load_from_hf
"""

import argparse

import requests
import torch
from megatron.core.inference.common_inference_params import CommonInferenceParams
from PIL import Image
from transformers import AutoProcessor

from nemo import lightning as nl
from nemo.collections import vlm
from nemo.collections.vlm.inference import generate as vlm_generate
from nemo.collections.vlm.inference import setup_inference_wrapper

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"


def load_image(image_url: str) -> Image.Image:
    # pylint: disable=C0115,C0116
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw)
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error loading image from {image_url}: {e}")
        return None


def generate(model, processor, images, text, params):
    # pylint: disable=C0115,C0116
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": text}],
        }
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

    model = setup_inference_wrapper(model, processor.tokenizer)

    prompts = [input_text]
    images = [images]
    result = vlm_generate(
        model,
        processor.tokenizer,
        processor.image_processor,
        prompts,
        images,
        inference_params=params,
    )

    generated_texts = list(result)[0].generated_text

    if torch.distributed.get_rank() == 0:
        print("======== GENERATED TEXT OUTPUT ========")
        print(f"{generated_texts}")
        print("=======================================")
    return generated_texts


def main(args) -> None:
    # pylint: disable=C0115,C0116
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        ckpt_load_optimizer=False,
        ckpt_save_optimizer=False,
    )
    trainer = nl.Trainer(
        devices=args.tp_size,
        max_steps=1000,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        val_check_interval=1000,
        limit_val_batches=50,
    )

    processor = AutoProcessor.from_pretrained(args.processor_name)
    tokenizer = processor.tokenizer

    fabric = trainer.to_fabric()

    if args.load_from_hf:
        model = fabric.import_model(f"hf://{model_id}", vlm.MLlamaModel)
    else:
        model = vlm.MLlamaModel(vlm.MLlamaConfig11BInstruct(), tokenizer=tokenizer)
        model = fabric.load_model(args.local_model_path, model)

    # Load the image
    raw_images = [load_image(url) for url in args.image_url]
    if not raw_images:
        return  # Exit if the image can't be loaded

    params = CommonInferenceParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_tokens_to_generate=args.num_tokens_to_generate,
    )
    generate(model, processor, images=raw_images, text=args.prompt, params=params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--load_from_hf",
        action="store_true",
        help="Flag to indicate whether to load the model from Hugging Face hub.",
    )
    parser.add_argument(
        "--local_model_path",
        type=str,
        default=None,
        help="Local path to the model if not loading from Hugging Face.",
    )
    parser.add_argument(
        "--processor_name",
        type=str,
        default="meta-llama/Llama-3.2-11B-Vision-Instruct",
        help="Name or path of processor",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="<|image|>\nDescribe the image.",
        help="Input prompt",
    )
    parser.add_argument(
        "--image_url",
        nargs='+',
        type=str,
        # pylint: disable=line-too-long
        default=[
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
        ],
        help="List of the image urls to use for inference.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="""Temperature to be used in megatron.core.inference.common_inference_params.CommonInferenceParams""",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.0,
        help="""top_p to be used in megatron.core.inference.common_inference_params.CommonInferenceParams""",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="""top_k to be used in megatron.core.inference.common_inference_params.CommonInferenceParams""",
    )
    parser.add_argument(
        "--num_tokens_to_generate",
        type=int,
        default=50,
        help="""Number of tokens to generate per prompt""",
    )
    parser.add_argument("--devices", type=int, required=False, default=1)
    parser.add_argument("--tp_size", type=int, required=False, default=1)
    parser.add_argument("--pp_size", type=int, required=False, default=1)
    parser.add_argument("--encoder_pp_size", type=int, required=False, default=0)

    args = parser.parse_args()
    main(args)
