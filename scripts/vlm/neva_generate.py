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
from megatron.core.inference.common_inference_params import CommonInferenceParams
from PIL import Image
from transformers import AutoProcessor

import nemo.lightning as nl
from nemo.collections.vlm import Llava15Config7B, LlavaModel
from nemo.collections.vlm.inference import generate as vlm_generate
from nemo.collections.vlm.inference import setup_inference_wrapper
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


def generate(model, processor, images, text, params):
    # pylint: disable=C0115,C0116
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image"},
            ],
        },
    ]
    input_text = processor.apply_chat_template(conversation, add_generation_prompt=True)

    class NevaTokenizer:
        # pylint: disable=C0115,C0116
        def __init__(self, tokenizer):
            self._tokenizer = tokenizer
            self.vocab_size = tokenizer.vocab_size
            self.eos_token_id = tokenizer.eos_token_id

        def decode(self, tokens, **kwargs):
            modified_tokens = []
            for x in tokens:
                if x == -200:
                    modified_tokens.append(0)
                elif x != 1:
                    modified_tokens.append(x)
            return self._tokenizer.decode(modified_tokens, skip_special_tokens=False)

        def encode(self, prompt, **kwargs):
            prompts_tokens = self._tokenizer.encode(prompt, add_special_tokens=True)
            return [-200 if x == 32000 else x for x in prompts_tokens]

    model = setup_inference_wrapper(model, processor.tokenizer)

    prompts = [input_text]
    images = [images]
    result = vlm_generate(
        model,
        NevaTokenizer(processor.tokenizer),
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


def legacy_generate(model, processor, raw_image, text, num_tokens_to_generate):
    # pylint: disable=C0115,C0116
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    hf_tokenizer = processor.tokenizer

    inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)
    input_ids = hf_tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
    input_ids[input_ids == 32000] = -200
    images = inputs['pixel_values'].cuda()
    images = images.reshape(images.size(0), 3, 336, 336)

    position_ids = (
        torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    )

    model = model.module.cuda()
    model.eval()
    generated_ids = input_ids.clone()

    # Greedy generation loop
    for _ in range(num_tokens_to_generate):
        with torch.no_grad():
            output = model(
                images=images,
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
            )

            next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

            input_ids = generated_ids
            position_ids = (
                torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand_as(input_ids)
            )

            # If the generated token is the end of sequence token, stop generating
            if next_token_ids.item() == hf_tokenizer.eos_token_id:
                break

    generated_ids[generated_ids == -200] = 0
    generated_texts = hf_tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
    logging.info("======== GENERATED TEXT OUTPUT ========")
    logging.info(f"{generated_texts}")
    logging.info("=======================================")


def main(args) -> None:
    # pylint: disable=C0115,C0116
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        ckpt_include_optimizer=False,
    )
    trainer = nl.Trainer(
        devices=1,
        max_steps=1000,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        val_check_interval=1000,
        limit_val_batches=50,
    )

    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    hf_tokenizer = processor.tokenizer

    # Load the image
    raw_image = load_image(args.image_url)
    if raw_image is None:
        return  # Exit if the image can't be loaded

    fabric = trainer.to_fabric()

    # Decide whether to import or load the model based on the input arguments
    if args.load_from_hf:
        model = fabric.import_model("hf://llava-hf/llava-1.5-7b-hf", LlavaModel)
    else:
        model = LlavaModel(Llava15Config7B(), tokenizer=hf_tokenizer)
        model = fabric.load_model(args.local_model_path, model)

    params = CommonInferenceParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_tokens_to_generate=args.num_tokens_to_generate,
    )
    if args.legacy_generate:
        legacy_generate(model, processor, raw_image, args.prompt, args.num_tokens_to_generate)
    else:
        generate(model, processor, images=raw_image, text=args.prompt, params=params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA Multimodal Inference")
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
        default="http://images.cocodataset.org/val2017/000000039769.jpg",
        help="URL of the image to use for inference.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What are these?",
        help="Input prompt",
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
        default=0,
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
        default=20,
        help="""Number of tokens to generate per prompt""",
    )
    parser.add_argument(
        "--legacy_generate",
        action="store_true",
        help="Flag to indicate whether to use legacy generation function.",
    )
    args = parser.parse_args()

    main(args)
