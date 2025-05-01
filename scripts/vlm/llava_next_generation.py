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

import requests
import torch
from PIL import Image
from transformers import AutoProcessor

from nemo import lightning as nl
from nemo.collections import vlm
from nemo.utils import logging


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


def generate(model, processor, raw_image, text):
    # pylint: disable=C0115,C0116
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What are these?"},
                {"type": "image"},
            ],
        }
    ]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(input_text, raw_image, return_tensors='pt').to(0, torch.float32)

    input_ids = inputs['input_ids'].cuda()
    input_ids[input_ids == 32000] = -200
    media = inputs['pixel_values'].cuda()
    media = media.reshape(media.size(1), 3, 336, 336)
    position_ids = (
        torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    )

    generated_ids = input_ids.clone()
    width, height = raw_image.size
    image_sizes = torch.tensor([[height, width]], dtype=torch.long).cuda()

    for _ in range(20):
        with torch.no_grad():
            attention_mask = (input_ids != 0).long().cuda()
            output = model(
                media=media,
                input_ids=input_ids,
                position_ids=position_ids,
                image_sizes=image_sizes,
                num_media_tiles=[media.size(0)],
                attention_mask=attention_mask,
            )
            next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

            input_ids = generated_ids
            position_ids = (
                torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand_as(input_ids)
            )
            print(f"next_token_ids {next_token_ids}")

            # If the generated token is the end of sequence token, stop generating
            if next_token_ids.item() == processor.tokenizer.eos_token_id:
                print(f"breaking")
                break
    generated_ids[generated_ids == -200] = 0
    generated_texts = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
    logging.info("======== GENERATED TEXT OUTPUT ========")
    logging.info(f"{generated_texts}")
    logging.info("=======================================")


def main(args) -> None:
    # pylint: disable=C0115,C0116
    model_id = 'llava-hf/llava-v1.6-vicuna-7b-hf'
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

    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = processor.tokenizer

    fabric = trainer.to_fabric()

    if args.load_from_hf:
        model = fabric.import_model("hf://llava-hf/llava-v1.6-vicuna-7b-hf", vlm.LlavaNextModel)
    else:
        model = vlm.LlavaNextModel(vlm.LlavaNextConfig7B(), tokenizer=tokenizer)
        model = fabric.load_model(args.local_model_path, model)

    model = model.module.cuda()
    model.eval()
    model = model.to(torch.bfloat16)

    # Load the image
    raw_image = load_image(args.image_url)
    if raw_image is None:
        return  # Exit if the image can't be loaded

    generate(model, processor, raw_image=raw_image, text="What are these?")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llava Next Generation example")
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
        "--image_url",
        type=str,
        # pylint: disable=line-too-long
        default="http://images.cocodataset.org/val2017/000000039769.jpg",
        help="URL of the image to use for inference.",
    )
    parser.add_argument("--devices", type=int, required=False, default=1)
    parser.add_argument("--tp_size", type=int, required=False, default=1)

    args = parser.parse_args()
    main(args)
