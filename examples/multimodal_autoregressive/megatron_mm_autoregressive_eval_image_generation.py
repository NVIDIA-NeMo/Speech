# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import datetime
import math
import os
import re

import torch
import torchvision
from examples.nlp.language_modeling.megatron_gpt_eval import (
    load_model_from_config,
    remove_padded_prompts,
    round_to_mult,
)
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.common.parts.nlp_overrides import CustomProgressBar, NLPDDPStrategy

# pylint: disable=line-too-long
from nemo.collections.common.video_tokenizers.cosmos_tokenizer import CausalVideoTokenizer
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.core.config import hydra_runner

"""
This is the script to run multimodal autoregresssive text generation.

Make sure you  install tiktoken==0.6.0

Usage:
    Assume the model has TP=1, PP=1 in the following use cases.
    a. run greedy inference from a nemo file:
        python megatron_mm_autoregresssive_eval.py \
            gpt_model_file=PATH_TO_MODEL \
            inference.greedy=True \
            inference.add_BOS=True \
            trainer.devices=1 \
            trainer.num_nodes=1 \
            tensor_model_parallel_size=-1 \
            pipeline_model_parallel_size=-1 \
            captions=[caption1,caption2]

    b. run greedy inference from a PTL checkpoint file:
        python megatron_mm_autoregresssive_eval.py \
            checkpoint_dir=PATH_TO_CHECKPOINT_FILE \
            checkpoint_name=CHECKPOINT_FILE_NAME \
            hparams_file=HPARAMS_FILE \
            inference.greedy=True \
            inference.add_BOS=True \
            trainer.devices=1 \
            trainer.num_nodes=1 \
            tensor_model_parallel_size=-1 \
            pipeline_model_parallel_size=-1 \
            captions=[caption1,caption2]

    c. run top_p inference from a nemo file:
        python megatron_mm_autoregresssive_eval.py \
            gpt_model_file=PATH_TO_MODEL \
            inference.greedy=False \
            inference.top_k=0 \
            inference.top_p=0.9 \
            inference.repetition_penalty=1.2 \
            inference.add_BOS=True \
            trainer.devices=1 \
            trainer.num_nodes=1 \
            tensor_model_parallel_size=-1 \
            pipeline_model_parallel_size=-1 \
            captions=[caption1,caption2]

    d. If you don't need to generate tokens and need model to compute logprobs:
         python megatron_mm_autoregresssive_eval.py \
            gpt_model_file=PATH_TO_MODEL \
            inference.compute_logprob=True \
            trainer.devices=1 \
            trainer.num_nodes=1 \
            tensor_model_parallel_size=-1 \
            pipeline_model_parallel_size=-1 \
            captions=[caption1,caption2]
"""


def to_img(tokens_string, image_tokenizer):
    """Converts visual tokens to images

    Given input visual tokens, we extract the indices, pass it to the decoder to get the image
    """
    visual_token_pattern = r"<\|visual token (\d+)\|>"
    visual_tokens = [int(match) for match in re.findall(visual_token_pattern, tokens_string)]
    # We assume image is square. So if 64 tokensa are present, we reshape it to 8x8 and then pass it to decoder
    dim = int(math.sqrt(len(visual_tokens)))
    visual_tokens_tensor = torch.tensor(visual_tokens[: dim * dim])
    # Decoder accepts input of the following format [bs, channel_dim, h, w]
    visual_tokens_tensor_reshaped = visual_tokens_tensor.reshape((dim, dim)).unsqueeze(0).unsqueeze(0)
    visual_tokens_final = visual_tokens_tensor_reshaped.to(image_tokenizer._device)
    img = image_tokenizer.decode(visual_tokens_final)

    # Convert from bf16 to 16 and to format [channel_dim, h, w]
    image = torchvision.transforms.functional.to_pil_image(img.float().squeeze())
    return image


def load_prompts(cfg):
    """Function to return the prompts passed into the model"""
    prompts = []
    for caption in cfg.captions:
        prompt = f'You are a helpful assistant. Draw a picture for the caption given by the user. USER: {caption}. ASSISTANT: '
        prompts.append(prompt)
    return prompts


if not torch.cuda.is_available():
    raise EnvironmentError("GPU is needed for the inference")


@hydra_runner(config_path="conf", config_name="megatron_mm_ar_inference_image_generation")
def main(cfg) -> None:
    """Main function"""

    callbacks = []
    # enable_progress_bar is True by default. If cfg.trainer.enable_progress_bar=False, CustomProgressBar is not appended to callbacks
    if 'enable_progress_bar' not in cfg.trainer or cfg.trainer.enable_progress_bar:
        callbacks.append(CustomProgressBar())
    # trainer required for restoring model parallel models
    trainer = Trainer(
        strategy=NLPDDPStrategy(timeout=datetime.timedelta(seconds=18000)),
        **cfg.trainer,
        callbacks=callbacks,
    )

    image_tokenizer = CausalVideoTokenizer.from_pretrained(
        tokenizer_type=cfg.image_encoder, load_encoder=False, load_decoder=True, load_full_model=False
    )

    model = load_model_from_config(trainer, cfg)
    model.freeze()

    # Have to turn off activations_checkpoint_method for inference
    try:
        model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass

    length_params: LengthParam = {
        "max_length": cfg.inference.tokens_to_generate,
        "min_length": cfg.inference.min_tokens_to_generate,
    }

    sampling_params: SamplingParam = {
        "use_greedy": cfg.inference.greedy,
        "temperature": cfg.inference.temperature,
        "top_k": cfg.inference.top_k,
        "top_p": cfg.inference.top_p,
        "repetition_penalty": cfg.inference.repetition_penalty,
        "add_BOS": cfg.inference.add_BOS,
        "all_probs": cfg.inference.all_probs,
        "compute_logprob": cfg.inference.compute_logprob,
        "end_strings": cfg.inference.end_strings,
    }

    prompts = []
    with torch.no_grad():
        prompts = load_prompts(cfg)

    fp8_enabled = hasattr(model.cfg, "fp8") and (model.cfg.fp8 == True)
    if fp8_enabled and len(prompts) > 0:
        padded_len = round_to_mult(len(prompts), 8)
        nb_paddings = padded_len - len(prompts)
        if nb_paddings > 0:
            nb_paddings += [''] * nb_paddings

    # First method of running text generation, call model.generate method
    response = model.generate(inputs=prompts, length_params=length_params, sampling_params=sampling_params)

    if fp8_enabled:
        response = remove_padded_prompts(response, nb_paddings)

    output_tokens_strings = response['sentences']
    for idx, output_token_string in enumerate(output_tokens_strings):
        image = to_img(output_token_string, image_tokenizer)
        image.save(os.path.join(cfg.images_output_path, f'{idx}.jpg'))

    print(f'Images saved to {cfg.images_output_path}')


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
