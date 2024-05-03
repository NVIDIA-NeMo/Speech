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

import json
import os
import torch
from torch.utils.data import Dataset

from nemo.collections.multimodal.parts.utils import create_neva_model_and_processor
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.core.config import hydra_runner
from nemo.utils.get_rank import is_global_rank_zero


try:
    import ammo.torch.quantization as atq

    HAVE_AMMO = True

except (ImportError, ModuleNotFoundError):

    HAVE_AMMO = False

if not torch.cuda.is_available():
    raise EnvironmentError("GPU is needed for the inference")


class RequestDataSet(Dataset):
    def __init__(self, sentences):
        super().__init__()
        self.sentences = sentences

    def __len__(self,):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


@hydra_runner(config_path="conf", config_name="neva_inference")
def main(cfg) -> None:
    model, image_processor, video_processor = create_neva_model_and_processor(cfg)

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

    with open(cfg.prompt_file, 'r') as f:
        lines = f.readlines()

    media_type_token = cfg.inference.get("media_type", "image")
    media_token = f"<{media_type_token}>"

    insert_media_token = cfg.inference.get("insert_media_token", None)
    final_prompts = []
    for line in lines:
        prompt_dict = json.loads(line)
        assert 'prompt' in prompt_dict or 'text' in prompt_dict
        if 'prompt' not in prompt_dict:
            prompt_dict['prompt'] = prompt_dict['text']
        if insert_media_token == 'left':
            prompt_dict['prompt'] = media_token + prompt_dict['prompt']
        elif insert_media_token == 'right':
            prompt_dict['prompt'] = prompt_dict['prompt'] + media_token
        if 'image' in prompt_dict:
            prompt_dict['image_path'] = prompt_dict['image']
            prompt_dict['image'] = image_processor(os.path.join(cfg.inference.media_base_path, prompt_dict['image']))
        if 'video' in prompt_dict:
            prompt_dict['video_path'] = prompt_dict['video']
            prompt_dict['video'] = video_processor(os.path.join(cfg.inference.media_base_path, prompt_dict['video']))
        final_prompts.append(prompt_dict)

    responses = model.generate(
        input_prompts=final_prompts, length_params=length_params, sampling_params=sampling_params, inference_config=cfg
    )

    # =================== Start Quantization ====================
    if HAVE_AMMO and cfg.quantization.enable == True:
        print(f"Using quantization algorithm: {cfg.quantization.algorithm}")
        if cfg.quantization.algorithm == "int8_sq":
            atq_config = atq.INT8_SMOOTHQUANT_CFG
        elif cfg.quantization.algorithm == "fp8":
            atq_config = atq.FP8_DEFAULT_CFG
        elif cfg.quantization.algorithm == "awq":
            atq_config = atq.INT4_AWQ_CFG
        else:
            raise ValueError(f"Unsupported quantization algorithm: {cfg.quantization.algorithm}")

        def forward_loop():
            model.generate(
                input_prompts=final_prompts,
                length_params=length_params,
                sampling_params=sampling_params,
                inference_config=cfg,
            )

        atq.quantize(model, atq_config, forward_loop)

        responses = model.generate(
            input_prompts=final_prompts,
            length_params=length_params,
            sampling_params=sampling_params,
            inference_config=cfg,
        )
    # ============== Quantization End =========================

    # PP middle stages do not yield any responses
    if responses is None:
        return

    if is_global_rank_zero():
        results = []
        for response, prompt in zip(responses, final_prompts):
            prompt['full_text'] = response["clean_text"]
            prompt['text'] = response["clean_response"]
            prompt['model_id'] = cfg.neva_model_file
            if 'image_path' in prompt:
                prompt['image'] = prompt.pop('image_path')
            if 'video_path' in prompt:
                prompt['video'] = prompt.pop('video_path')
            if 'answer_id' not in prompt:
                prompt['answer_id'] = 0
            if 'metadata' not in prompt:
                prompt['metadata'] = {}
            results.append(prompt)

        with open(cfg.output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
