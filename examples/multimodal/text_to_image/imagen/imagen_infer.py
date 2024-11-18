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

from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from nemo.collections.multimodal.models.text_to_image.imagen.imagen_pipeline import (
    ImagenPipeline,
    ImagenPipelineConfig,
)
from nemo.core.config import hydra_runner


@hydra_runner(config_path='conf', config_name='inference_pipeline.yaml')
def main(inference_config):
    if inference_config.get('infer'):
        # invoking from launcher
        trainer = Trainer(**inference_config.trainer)
        inference_config = inference_config.infer
    else:
        trainer = Trainer()
    inference_config: ImagenPipelineConfig = OmegaConf.merge(ImagenPipelineConfig(), inference_config)
    pipeline = ImagenPipeline.from_pretrained(cfg=inference_config, trainer=trainer)

    # Texts are passed in the config files
    images, all_res, throughput = pipeline()

    # Save images
    outpath = inference_config.output_path
    os.makedirs(outpath, exist_ok=True)
    for text, pils in zip(inference_config.texts, images):
        for idx, image in enumerate(pils):
            image.save(os.path.join(outpath, f'{text}_{idx}.png'))


if __name__ == '__main__':
    main()
