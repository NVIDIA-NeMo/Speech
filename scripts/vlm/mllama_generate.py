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

"""
Example:
  python scripts/vlm/neva_generate.py --load_from_hf
"""

import argparse

import requests
import torch
from PIL import Image
from transformers import AutoProcessor

from nemo import lightning as nl
from nemo.collections import vlm
from nemo.collections.vlm.mllama.model.utils import create_vision_mask_tensor

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


def generate(model, processor, image, text):
    # pylint: disable=C0115,C0116
    tokenizer = processor.tokenizer

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": text}],
        }
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    batch = processor(image, input_text, add_special_tokens=False, return_tensors="pt")

    input_ids = batch["input_ids"].cuda(non_blocking=True)
    position_ids = (
        torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    )
    num_tiles = processor.image_processor.preprocess(image, return_tensors='pt')["num_tiles"]

    min_prompt_len = position_ids.shape[-1]

    input_ids = input_ids[:, :min_prompt_len]
    generated_ids = input_ids.clone()

    from tqdm import tqdm

    for cur_pos in tqdm(range(min_prompt_len, min_prompt_len + 100)):
        with torch.no_grad():
            position_ids = torch.arange(0, cur_pos, dtype=torch.long, device="cuda").reshape(1, -1)
            batch_masks = create_vision_mask_tensor(generated_ids[0])

            output = model(
                batch_images=batch["pixel_values"].cuda(non_blocking=True),
                batch_masks=[batch_masks],
                num_chunks=torch.tensor(num_tiles),
                aspect_ratio_ids=batch["aspect_ratio_ids"].cuda(non_blocking=True),
                tokens=generated_ids,
                position_ids=position_ids,
            )

            next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)
            # Broadcast the tensor from rank 0 to all other ranks
            torch.distributed.broadcast(next_token_ids, src=0)
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
            if (next_token_ids == tokenizer.eos_token_id).all():
                break

    generated_ids = generated_ids.tolist()
    generated_texts = tokenizer.decode(generated_ids[0][min_prompt_len:])

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

    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = processor.tokenizer

    fabric = trainer.to_fabric()

    if args.load_from_hf:
        model = fabric.import_model(f"hf://{model_id}", vlm.MLlamaModel)
    else:
        model = vlm.MLlamaModel(vlm.MLlamaConfig11BInstruct(), tokenizer=tokenizer)
        model = fabric.load_model(args.local_model_path, model)

    model = model.module.cuda()
    model.eval()
    model = model.to(torch.bfloat16)

    # Load the image
    raw_image = load_image(args.image_url)
    if raw_image is None:
        return  # Exit if the image can't be loaded

    generate(model, processor, image=raw_image, text="<|image|>\nDescribe the image.")


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
        "--image_url",
        type=str,
        # pylint: disable=line-too-long
        default="https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
        help="URL of the image to use for inference.",
    )
    parser.add_argument("--devices", type=int, required=False, default=1)
    parser.add_argument("--tp_size", type=int, required=False, default=1)
    parser.add_argument("--pp_size", type=int, required=False, default=1)
    parser.add_argument("--encoder_pp_size", type=int, required=False, default=0)

    args = parser.parse_args()
    main(args)
