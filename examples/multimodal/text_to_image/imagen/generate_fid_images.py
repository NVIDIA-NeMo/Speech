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

import torch
from lightning.pytorch import Trainer

from nemo.collections.multimodal.models.text_to_image.imagen.imagen_pipeline import ImagenPipeline
from nemo.core.config import hydra_runner


@hydra_runner(config_path='conf', config_name='imagen_fid_images')
def main(cfg):
    # Read configuration parameters
    nnodes_per_cfg = cfg.fid.nnodes_per_cfg
    ntasks_per_node = cfg.fid.ntasks_per_node
    local_task_id = cfg.fid.local_task_id
    num_images_to_eval = cfg.fid.num_images_to_eval
    path = cfg.fid.coco_captions_path
    save_text = cfg.fid.save_text

    node_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    node_id_per_cfg = node_id % nnodes_per_cfg

    current_node_cfg = cfg.fid.classifier_free_guidance[node_id // nnodes_per_cfg]
    save_path = os.path.join(cfg.fid.save_path, str(current_node_cfg))

    # Read and store captions
    captions = []
    caption_files = sorted(os.listdir(path))
    assert len(caption_files) >= num_images_to_eval
    for file in caption_files[:num_images_to_eval]:
        with open(os.path.join(path, file), 'r') as f:
            captions += f.readlines()
    print(f"The total number of captions to generate is: {len(captions)}")

    # Calculate partition sizes and select the partition for the current node
    partition_size_per_node = num_images_to_eval // nnodes_per_cfg
    start_idx = node_id_per_cfg * partition_size_per_node
    end_idx = (node_id_per_cfg + 1) * partition_size_per_node if node_id_per_cfg != nnodes_per_cfg - 1 else None
    captions = captions[start_idx:end_idx]
    print(f"Current node {node_id} will generate images from {start_idx} to {end_idx}")

    local_task_id = int(local_task_id) if local_task_id is not None else int(os.environ.get("SLURM_LOCALID", 0))
    partition_size_per_task = int(len(captions) // ntasks_per_node)

    # Select the partition for the current task
    start_idx = local_task_id * partition_size_per_task
    end_idx = (local_task_id + 1) * partition_size_per_task if local_task_id != ntasks_per_node - 1 else None
    input = captions[start_idx:end_idx]
    chunk_size = len(input)

    print(f"Current worker {node_id}:{local_task_id} will generate {len(input)} images")
    os.makedirs(save_path, exist_ok=True)

    trainer = Trainer()
    pipeline = ImagenPipeline.from_pretrained(cfg=cfg.infer, trainer=trainer, megatron_loading=True, megatron_cfg=cfg)

    # Generate images using the model and save them
    batch_idx = 0
    batch_size = cfg.fid.ncaptions_per_batch
    while True:
        if batch_idx * batch_size >= len(input):
            break
        batch_captions = input[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        # Different seed for every image
        seeds = [local_task_id * chunk_size + batch_idx * batch_size + idx for idx in range(len(batch_captions))]
        with torch.no_grad():
            images, all_res_images, *_ = pipeline(
                prompts=batch_captions,
                seed=seeds,
                single_batch_mode=True,
                classifier_free_guidance=current_node_cfg,
            )

        if cfg.fid.save_all_res:
            all_res = [f'_RES{model.image_size}' for model in pipeline.models]
            outpaths = []
            # for the highest resolution we save as its original name so that
            # we can automate the CLIP & FID calculation process from Megatron-Launcher
            all_res[-1] = ''
            for res in all_res:
                outpath = f"{save_path}{res}"
                os.makedirs(outpath, exist_ok=True)
                outpaths.append(outpath)
            for outpath, one_res in zip(outpaths, all_res_images):
                for idx, (caption, image) in enumerate(zip(batch_captions, one_res[0])):
                    image_idx = local_task_id * chunk_size + batch_idx * batch_size + idx
                    image.save(os.path.join(outpath, f'image{image_idx:06d}.png'))
                    if save_text:
                        with open(os.path.join(outpath, f'image{image_idx:06d}.txt'), 'w') as f:
                            f.writelines(caption)
        else:
            for idx, (caption, image) in enumerate(zip(batch_captions, images[0])):
                image_idx = local_task_id * chunk_size + batch_idx * batch_size + idx
                image.save(os.path.join(save_path, f'image{image_idx:06d}.png'))
                if save_text:
                    with open(os.path.join(save_path, f'image{image_idx:06d}.txt'), 'w') as f:
                        f.writelines(caption)
        print(
            f'Save {len(images[0])} images to {save_path} with name from image{(local_task_id*chunk_size+batch_idx*batch_size):06d}.png to image{image_idx:06d}.png'
        )
        batch_idx += 1


if __name__ == "__main__":
    main()
