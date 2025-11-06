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

import torch
import torchvision
from examples.nlp.language_modeling.megatron_gpt_eval import (
    RequestDataSet,
    load_model_from_config,
    remove_padded_prompts,
    round_to_mult,
)
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from nemo.collections.common.parts.nlp_overrides import CustomProgressBar, NLPDDPStrategy

# pylint: disable=line-too-long
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
            images_path=[image_path1,image_path2]

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
            images_path=[image_path1,image_path2]

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
            images_path=[image_path1,image_path2]

    d. If you don't need to generate tokens and need model to compute logprobs:
         python megatron_mm_autoregresssive_eval.py \
            gpt_model_file=PATH_TO_MODEL \
            inference.compute_logprob=True \
            trainer.devices=1 \
            trainer.num_nodes=1 \
            tensor_model_parallel_size=-1 \
            pipeline_model_parallel_size=-1 \
            images_path=[image_path1,image_path2]
"""

EMU_HUB = "BAAI/Emu3-Gen"
VQ_HUB = "BAAI/Emu3-VisionTokenizer"


def to_imgstr(image_tokens, tokenizer):
    """Convert integer image tokens to visual tokens string"""
    image_tokens = image_tokens.cpu().numpy().tolist()
    image_token_str = [
        ['<|visual token {token_id:0>6d}|>'.format(token_id=token_id) for token_id in token_row]
        for token_row in image_tokens
    ]
    image_row_str = ["".join(token_row) for token_row in image_token_str]
    imgstr = tokenizer.eol_token.join(image_row_str)
    return imgstr


def load_prompts(cfg, image_tokenizer, tokenizer):
    """Function to generate prompts

    The prompts generated here are fed to the model.
    """
    prompts = []
    text = "Please describe the image"
    for image_path in cfg.images_path:
        image = Image.open(image_path)
        image_tensor = torchvision.transforms.functional.pil_to_tensor(image).unsqueeze(0)
        image_tokens = image_tokenizer.encode(image_tensor.to(image_tokenizer.device, image_tokenizer.dtype))
        bs, h, w = image_tokens.shape
        imgstr = to_imgstr(image_tokens[0], tokenizer=tokenizer)
        image_prompt = (
            tokenizer.boi_token
            + f'{h}*{w}'
            + tokenizer.img_token
            + imgstr
            + tokenizer.eol_token
            + tokenizer.eof_token
            + tokenizer.eoi_token
        )
        prompt = f'{tokenizer.bos_token}You are a helpful assistant. USER: {image_prompt}{text}. ASSISTANT:'
        prompts.append(prompt)
    return prompts


if not torch.cuda.is_available():
    raise EnvironmentError("GPU is needed for the inference")


@hydra_runner(config_path="conf", config_name="megatron_mm_ar_inference_vision_understanding")
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

    tokenizer = AutoTokenizer.from_pretrained(EMU_HUB, trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(VQ_HUB, device_map="cuda", trust_remote_code=True).eval()

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
        prompts = load_prompts(cfg, image_tokenizer, tokenizer)

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
    print("***************************")
    print(response)
    print("***************************")

    # Second method of running text generation, call trainer.predict [recommended]
    bs = 8 if fp8_enabled else 2
    ds = RequestDataSet(prompts)
    request_dl = DataLoader(dataset=ds, batch_size=bs)
    config = OmegaConf.to_container(cfg.inference)
    model.set_inference_config(config)
    response = trainer.predict(model, request_dl)

    if fp8_enabled:
        response[-1] = remove_padded_prompts(response[-1], nb_paddings)
    print("***************************")
    print(response)
    print("***************************")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
