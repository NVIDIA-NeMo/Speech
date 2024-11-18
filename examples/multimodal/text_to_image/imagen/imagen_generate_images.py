# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import os
import pickle

import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from nemo.collections.multimodal.models.text_to_image.imagen.imagen_pipeline import (
    ImagenPipeline,
    ImagenPipelineConfig,
)
from nemo.core.config import hydra_runner


@hydra_runner(config_path='conf', config_name='fid_inference.yaml')
def main(inference_config):
    inference_config: ImagenPipelineConfig = OmegaConf.merge(ImagenPipelineConfig(), inference_config)
    captions = pickle.load(open('coco_captions.pkl', 'rb'))
    ntasks = 8
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        # Multi-GPU
        task_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
    else:
        # Single GPU
        task_id = 0
    chuncksize = int(len(captions) // ntasks)
    if task_id != ntasks - 1:
        input = captions[task_id * chuncksize : (task_id + 1) * chuncksize]
    else:
        input = captions[task_id * chuncksize :]
    captions = input

    trainer = Trainer()
    pipeline = ImagenPipeline.from_pretrained(cfg=inference_config, trainer=trainer)
    batch_size = 16
    batch_idx = 0

    possible_res = [64, 256]  # [64, 256]
    outpaths = []
    for res in possible_res:
        outpath = f'{inference_config.output_path}_RES{res}'
        os.makedirs(outpath, exist_ok=True)
        outpaths.append(outpath)
    while True:
        if batch_idx * batch_size >= len(captions):
            break
        batch_captions = captions[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        # Different seed for every image
        seeds = [task_id * chuncksize + batch_idx * batch_size + idx for idx in range(len(batch_captions))]
        seed = batch_idx + chuncksize

        with torch.no_grad():
            images, all_res_images, throughput = pipeline(
                prompts=batch_captions,
                seed=seeds,
                single_batch_mode=True,
            )

        for outpath, one_res in zip(outpaths, all_res_images):
            for idx, (caption, image) in enumerate(zip(batch_captions, one_res[0])):
                image.save(os.path.join(outpath, f'image_{task_id*chuncksize+batch_idx*batch_size+idx}.png'))
                with open(os.path.join(outpath, f'image_{task_id*chuncksize+batch_idx*batch_size+idx}.txt'), 'w') as f:
                    f.writelines(caption)
        batch_idx += 1


if __name__ == '__main__':
    main()
