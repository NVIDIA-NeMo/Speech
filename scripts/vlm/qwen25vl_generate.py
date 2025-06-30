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
python qwen25vl_generate.py --load_from_hf --osl 50
"""

import argparse

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

import nemo.lightning as nl
from nemo.collections.vlm import Qwen2VLModel, Qwen25VLConfig3B, Qwen25VLConfig7B, Qwen25VLConfig32B, Qwen25VLConfig72B
from nemo.utils import logging


def main(args) -> None:
    # pylint: disable=C0115,C0116,C0301

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        ckpt_include_optimizer=False,
    )
    trainer = nl.Trainer(
        devices=args.tp_size * args.pp_size,
        max_steps=1000,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        val_check_interval=1000,
        limit_val_batches=50,
    )

    # Tokenize the input texts
    min_pixels = 16 * 28 * 28
    max_pixels = 64 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        f"Qwen/Qwen2.5-VL-{args.model_size}-Instruct", min_pixels=min_pixels, max_pixels=max_pixels
    )

    hf_tokenizer = processor.tokenizer

    fabric = trainer.to_fabric()
    # Decide whether to import or load the model based on the input arguments
    if args.load_from_hf:
        model = fabric.import_model(f"hf://Qwen/Qwen2.5-VL-{args.model_size}-Instruct", Qwen2VLModel)
    else:
        model_config = {
            "3B": Qwen25VLConfig3B,
            "7B": Qwen25VLConfig7B,
            "32B": Qwen25VLConfig32B,
            "72B": Qwen25VLConfig72B,
        }[args.model_size]()
        model = Qwen2VLModel(model_config, model_version="qwen25-vl", tokenizer=hf_tokenizer)
        model = fabric.load_model(args.local_model_path, model)
    model = model.module.cuda()
    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": args.image_url,
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(messages, tokenizer=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        input_ids = inputs['input_ids'].clone().to("cuda")
        # convert special tokens to nemo image ID
        input_ids[input_ids == 151655] = -200
        image_grid_thw = inputs['image_grid_thw'].clone().to("cuda")
        pixel_values = inputs['pixel_values'].clone().to("cuda")

        # Greedy generation loop
        generated_ids = input_ids

        for _ in range(args.osl):
            output = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                position_ids=None,
                attention_mask=None,
                image_grid_thw=image_grid_thw,
            )

            next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

            input_ids = generated_ids
            # If the generated token is the end of sequence token, stop generating
            if next_token_ids.item() == hf_tokenizer.eos_token_id:
                break

        generated_ids[generated_ids < 0] = 0
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

        generated_texts = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        logging.info("======== GENERATED TEXT OUTPUT ========")
        logging.info(f"{args.image_url}, \t\t{generated_texts}")
        logging.info("=======================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2.5VL Multimodal Inference")
    parser.add_argument(
        "--load_from_hf", action="store_true", help="Flag to indicate whether to load the model from Hugging Face hub."
    )
    parser.add_argument(
        "--local_model_path",
        type=str,
        default=None,
        help="Path to the local model if not loading from Hugging Face.",
    )
    parser.add_argument(
        "--image_url",
        type=str,
        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        help="URL of the image to use for inference.",
    )
    parser.add_argument("--model_size", type=str, default="3B", choices=["3B", "7B", "32B", "72B"])
    parser.add_argument('--osl', type=int, default=30, help='output seq length')
    parser.add_argument('--tp_size', type=int, default=1, help='tensor parallel size')
    parser.add_argument('--pp_size', type=int, default=1, help='pipeline parallel size')
    args = parser.parse_args()

    main(args)
