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
from torch.utils.data import DataLoader, Dataset

from nemo.collections.multimodal.parts.utils import create_neva_model_and_processor
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.core.config import hydra_runner
from nemo.utils.get_rank import is_global_rank_zero


try:
    import modelopt.torch.quantization as mtq

    HAVE_MODELOPT = True

except (ImportError, ModuleNotFoundError):

    HAVE_MODELOPT = False

if not torch.cuda.is_available():
    raise EnvironmentError("GPU is needed for the inference")


class TemporalNevaDataset(Dataset):
    def __init__(
        self,
        prompt_dicts,
        media_base_path,
        media_token,
        insert_media_token=None,
        image_processor=None,
        video_processor=None,
        add_media_sep=False,
    ):
        self.prompt_dicts = prompt_dicts
        self.media_token = media_token
        self.insert_media_token = insert_media_token
        self.media_base_path = media_base_path
        self.image_processor = image_processor
        self.video_processor = video_processor
        self.add_media_sep = add_media_sep
        # [(media_name, [prompt_dict, prompt_dict, ...]), ...}
        self.media_prompt_list = []
        self.group_by_media(media_token)

    def group_by_media(self, media_token):
        """
        This function groups the prompt dicts by the media/video/image file name
        """
        media_dict = {}
        media = media_token.lstrip('<').rstrip('>')
        for prompt_dict in self.prompt_dicts:
            media_name = prompt_dict[media]  # video or image file name
            if media_name not in media_dict:
                media_dict[media_name] = []
            media_dict[media_name].append(prompt_dict)
        self.media_prompt_list = list(media_dict.items())

    def __len__(self) -> int:
        return len(self.media_prompt_list)

    def __getitem__(self, idx) -> dict:
        """
        Return a list of prompt dicts for the idx-th media
        For a single media file, only one media feature is returned
        This would help improve performance as well as save GPU memory
        """
        prompt_dict_list = self.media_prompt_list[idx][1]
        cur_item = []
        cur_media_feature = None
        for prompt_dict in prompt_dict_list:
            if 'prompt' not in prompt_dict:
                prompt_dict['prompt'] = prompt_dict['text'] if 'text' in prompt_dict else prompt_dict['question']
            if self.insert_media_token == 'left':
                if self.add_media_sep:
                    prompt_dict['prompt'] = self.media_token + " \n" + prompt_dict['prompt']
                else:
                    prompt_dict['prompt'] = self.media_token + prompt_dict['prompt']
            elif self.insert_media_token == 'right':
                if self.add_media_sep:
                    prompt_dict['prompt'] = prompt_dict['prompt'] + self.media_token + " \n"
                else:
                    prompt_dict['prompt'] = prompt_dict['prompt'] + self.media_token
            if 'image' in prompt_dict:
                prompt_dict['image_path'] = prompt_dict['image']
                image_path = os.path.join(self.media_base_path, prompt_dict['image'])
                if cur_media_feature is None:
                    cur_media_feature = ("image", self.image_processor(image_path))
            if 'video' in prompt_dict:
                prompt_dict['video_path'] = prompt_dict['video']
                video_path = os.path.join(self.media_base_path, prompt_dict['video'])
                if cur_media_feature is None:
                    cur_media_feature = ("video", self.video_processor(video_path))
            cur_item.append(prompt_dict)
        return cur_media_feature, cur_item


def collate_function(batch):
    # do nothing
    return batch


def do_inference(dataloader, model, length_params, sampling_params, cfg):
    responses = []
    all_prompts = []
    for idx, batch_media_prompts in enumerate(dataloader):
        if idx % 10 == 0:
            print(f"Processed {idx} batch media")
        for media_media_feature, prompts in batch_media_prompts:
            media, media_feature = media_media_feature
            all_prompts.extend(prompts.copy())
            for prompt in prompts:
                prompt[media] = media_feature
            cur_batch_responses = model.generate(
                input_prompts=prompts,
                length_params=length_params,
                sampling_params=sampling_params,
                inference_config=cfg,
            )
            responses.extend(cur_batch_responses)
    return responses, all_prompts


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

    prompt_dicts = []
    if cfg.prompt_file.endswith('.json'):
        with open(cfg.prompt_file, 'r') as f:
            prompt_dicts = json.load(f)
    elif cfg.prompt_file.endswith('.jsonl'):
        with open(cfg.prompt_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            prompt_dicts.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported prompt file format: {cfg.prompt_file}")

    media_type_token = cfg.inference.get("media_type", "image")
    media_token = f"<{media_type_token}>"

    insert_media_token = cfg.inference.get("insert_media_token", None)
    dataset = TemporalNevaDataset(
        prompt_dicts,
        cfg.inference.media_base_path,
        media_token,
        insert_media_token,
        image_processor,
        video_processor,
        cfg.get("add_media_sep", False),
    )

    num_workers = 2
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.inference.get("batch_size", 1),
        shuffle=False,
        collate_fn=collate_function,
        num_workers=num_workers,
        persistent_workers=True,
    )
    responses, final_prompts = do_inference(dataloader, model, length_params, sampling_params, cfg)

    # =================== Start Quantization ====================
    if HAVE_MODELOPT and cfg.quantization.enable == True:
        print(f"Using quantization algorithm: {cfg.quantization.algorithm}")
        if cfg.quantization.algorithm == "int8_sq":
            mtq_config = mtq.INT8_SMOOTHQUANT_CFG
        elif cfg.quantization.algorithm == "fp8":
            mtq_config = mtq.FP8_DEFAULT_CFG
        elif cfg.quantization.algorithm == "awq":
            mtq_config = mtq.INT4_AWQ_CFG
        else:
            raise ValueError(f"Unsupported quantization algorithm: {cfg.quantization.algorithm}")

        def forward_loop():
            num_samples = cfg.quantization.get("num_samples", 100)
            if num_samples == -1:
                cur_prompt_dicts = prompt_dicts
            else:
                cur_prompt_dicts = prompt_dicts[:num_samples]
            cur_dataset = TemporalNevaDataset(
                cur_prompt_dicts,
                cfg.inference.media_base_path,
                media_token,
                insert_media_token,
                image_processor,
                video_processor,
                cfg.get("add_media_sep", False),
            )
            cur_dataloader = DataLoader(
                cur_dataset,
                batch_size=cfg.inference.get("batch_size", 1),
                shuffle=False,
                collate_fn=collate_function,
                num_workers=num_workers,
            )
            _, _ = do_inference(cur_dataloader, model, length_params, sampling_params, cfg)

        mtq.quantize(model, mtq_config, forward_loop)

        responses, final_prompts = do_inference(dataloader, model, length_params, sampling_params, cfg)

    # ============== Quantization End =========================

    # PP middle stages do not yield any responses
    if responses is None:
        return

    if is_global_rank_zero():
        results = []
        for response, prompt in zip(responses, final_prompts):
            prompt['full_text'] = response["clean_text"]
            prompt['pred_answer'] = response["clean_response"]
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
            if cfg.output_file.endswith('.json'):
                json.dump(results, f, indent=2)
            else:
                for result in results:
                    f.write(json.dumps(result) + '\n')


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
